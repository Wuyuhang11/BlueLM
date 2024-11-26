import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from conv import Conv1D
import math
import GPTConfig

n_position = 10


class Attention(nn.Module):

    def __init__(self, config: GPTConfig, scale: bool = False) -> None:
        super().__init__()
        self.n_embd = config.n_embd  # 输入序列维度

        assert config.n_embd % config.n_head == 0

        self.scale = scale  # 注意力缩放
        self.n_head = config.n_head  # 注意力头数

        self.c_attn = Conv1D(self.n_embd, self.n_embd * 3)  # 特征变换: 目的是将输入的嵌入维度n_embd扩展到n_embd*3，为每个头生成q、k、v三种向量
        self.c_proj = Conv1D(self.n_embd, self.n_embd)  # 特征压缩

        # 如果没有 flash attention 就创造一个下三角掩码矩阵作为缓冲区，存储掩码矩阵
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                    1, 1, config.n_positions, config.n_positions
                ),
                persistent=False,  # will not be saved alongside parameters
            )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

    # 分Q、K、V
    def split_heads(self, x: Tensor, is_key: bool = False) -> Tensor:
        """从最后一个维度上进行拆分得到 Q、K、V 矩阵

        Args:
            x (Tensor): 输入特征张量
            is_key (bool, optional): _description_. Defaults to False.

        Returns:
            Tensor: _description_
        """
        # 对最后一个维度（即，输入序列的特征张量）进行拆分->(batch_size, seq_len, num_heads, n_embd / num_heads) 元组
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)

        # x->(batch_size, seq_len, num_heads, n_embd / num_heads)
        x = x.view(*new_shape)

        # 是否是键向量
        if is_key:
            # k->(batch_size, num_heads, n_embd / num_heads, seq_len)
            return x.permute(0, 2, 3, 1)
        # q、v->(batch_size, num_heads, seq_len, n_embd / num_heads)
        return x.permute(0, 2, 1, 3)

    # 合并每个注意力头学习到的信息到统一的特征空间
    def merge_heads(self, x: Tensor) -> Tensor:
        """合并注意力头学习到信息到统一的特征空间中

        Args:
            x (Tensor): 每个头的特征张量

        Returns:
            Tensor: 合并后的特征张量
        """
        # x->(batch_size, seq_len, n_heads, n_embd / n_heads)
        x = x.permute(0, 2, 1, 3).contiguous
        # x->(batch_size, seq_len, n_embd)
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    # 注意力机制
    def _attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,
    ) -> list[Tensor]:
        """GPT Decoder中的注意力机制实现

        Args:
            q (Tensor): 查询张量
            k (Tensor): 键值张量
            v (Tensor): 值张量
            attention_mask (Tensor, optional): 掩码矩阵. Defaults to None.
            output_attentions (bool, optional): 是否输出注意力权重. Defaults to False.

        Returns:
        1.利用查询张量Q和键值张量计算注意力分数
        2.对注意力分数进行掩码（对上三角矩阵设为0，点积的时候注意力分数即为0了，然后再断言为0的元素设置为负无穷，负无穷的元素在softmax时就为0），实现输出的自回归
        3.正规化分数
        4.利用值向量加权求和
        5.输出注意力权重（含义：模型对输入序列中各个位置中的关注程度）
            list[Tensor]: _description_
        """

        # 是否使用掩码
        if self.flash:
            # 1使用 flash attention:  利用PyTorch内置的函数来计算注意力输出和权重
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=True,  # 传入is_causal=True表示使用掩码来实现自回归特性，因此attn_mask为None
            )
        else:
            # 2.利用q、k计算注意力分数->(batchsize, n_heads, seq_len, seq_len)
            scores = torch.matmul(q, k)
            # 3.缩放注意力分数，使输出在正态分布内
            if self.scale:
                scores = scores / math.sqrt(v.size(-1))
            # 4.应用掩码，形成自回归（Masking）
            bias = self.bias[:, :, : scores.size(-2), : scores.size(-1)]
            # 通过bias和scores相乘，将下三角置为0，并减去一个很大的数，使得注意力分数上的掩码位置是个很小的数，softmax后未来的元素注意力权重就会为0
            scores = scores * bias + -1e9 * (1 - bias)
            # 5.掩码后应用softmax和dropout
            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            # 6.应用注意力掩码(Attention Masking)，进一步屏蔽特定位置的注意力（注：相比于Masking，更加通用，并且可以用于非自回归，屏蔽掉不相关或不应该被模型看到的输入部分）
            if attention_mask is not None:
                weights = weights + attention_mask

            del scores
            # 7.将注意力权重weights和头的值向量加权求和，得到最后的注意力输出
            attn_output = torch.matmul(weights, v)
            outputs = [attn_output]
            # 8.返回 outputs 列表，它包含了注意力输出和（如果需要的话）注意力权重。
            if output_attentions:
                outputs.append(weights)

            return outputs

    # 前向传播
    def forward(self, x: Tensor, output_attentions: bool = False) -> list[Tensor]:
        """注意力机制

        Args:
            x (Tensor): 输入特征张量
            output_attentions (bool, optional): 是否输出注意力权重. Defaults to False.

        Returns:
            list[Tensor]: _description_
        """
        # 1.特征变换，为了得到q，k，v
        x = self.c_attn(x)
        # 2.分出q、k、v，通过 split 方法将张量 x 沿着第三个维度 n_embd*3 分割成三个相等的部分，每个部分大小为 n_embd 维度
        # q、k、v->(batch_size, seq_len, n_embd)
        query, key, value = x.split(self.n_embd, dim=2)
        # query,value->(batch_size, n_head, n_embd/n_head, seq_len); key->(batch_size, n_head, seqlen, n_embd/n_head)
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        # 3.得到注意力输出
        attn_outputs = self._attn(query, key, value, output_attentions)
        attn_output = attn_outputs[0]
        # 4.合并多注意力头
        output = self.merge_heads(attn_output)
        # 5.进行特征压缩
        output = self.c_proj(output)
        # 6.dropout
        output = self.proj_dropout(output)
        # 7.结合注意力权重返回
        outputs = [output] + attn_output[1:]
        return outputs
