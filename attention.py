import torch
import torch.nn as nn
from torch import Tensor
from conv import Conv1D

n_position = 10


class Attention(nn.Module):

    def __init__(self, config: GPTConfig, scale: bool = False) -> None:
        super().__init__()
        self.n_embd = config.n_embd

        assert config.n_embd % config.n_head == 0

        self.scale = scale
        self.n_head = config.n_head

        self.c_attn = Conv1D(self.n_embd, self.n_embd * 3)
        self.c_proj = Conv1D(self.n_embd, self.n_embd)
        # use flash attention or not
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

    def merge_heads(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor):  (batch_size,  n_head, seq_len, n_embd / n_head)

        Returns:
            Tensor: (batch_size, seq_len, n_embd)
        """
        # x (batch_size,  seq_len, n_head, n_embd / n_head)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (batch_size, seq_len, n_embd)
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    

    def forward(
        self, x: Tensor, attention_mask: Tensor = None, output_attentions: bool = False
    ) -> list[Tensor]:
        """

        Args:
            x (Tensor): (batch_size, seq_len, n_embd)

        Returns:
            Tensor: (batch_size, seq_len, n_embd) attn_output
            Tensor(optional): (batch_size, n_head, seq_len, seq_len) attn_weights

        """
        # calculate query, key ,value for all heads in batch
        # x (batch_size, seq_len, n_embd * 3)
        x = self.c_attn(x)
        #  query, key, value (batch_size, seq_len, n_embd)
        query, key, value = x.split(self.n_embd, dim=2)
        # query (batch_size,  n_head, seq_len, n_embd / n_head)
        query = self.split_heads(query)
        # key (batch_size, n_head, n_embd / n_head, seq_len)
        key = self.split_heads(key, is_key=not self.flash)
        # value (batch_size,  n_head, seq_len, n_embd / n_head)
        value = self.split_heads(value)
        # attn_output (batch_size,  n_head, seq_len, n_embd / n_head)
        attn_outputs = self._attn(query, key, value, attention_mask, output_attentions)
        attn_output = attn_outputs[0]

        del query, key, value

        # output (batch_size, seq_len, n_embd)
        output = self.merge_heads(attn_output)
        # (batch_size, seq_len, n_embd)
        output = self.c_proj(output)

        output = self.proj_dropout(output)

        outputs = [output] + attn_outputs[1:]
        return outputs
