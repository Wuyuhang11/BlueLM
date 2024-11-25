import torch
import torch.nn as nn
from torch import Tensor
import GPTConfig
from attention import Attention
import MLP 


class Block(nn.Module):

    # 初始化Block结构
    def __init__(self, config: GPTConfig, scale: bool = False) -> None:
        super().__init__()
        n_embd = config.n_embd  # 输入序列维度
        self.attn = Attention(config, scale)  # 多头注意力初始化
        self.ln_1 = nn.LayerNorm(n_embd)  # 层归一化
        self.mlp = MLP(config)  # 前向传播初始化
        self.ln_2 = nn.LayerNorm(n_embd)  # 层归一化

        def forward(
            self,
            x: Tensor,
            attention_mask: Tensor = None,
            output_attentions: bool = False,
        ) -> Tensor:
            """Block前向传播

            Args:
                x (Tensor): 输入特征张量
                attention_mask (Tensor, optional): 掩码矩阵，对预测的当前token进行掩码，自回归. Defaults to None.
                output_attentions (bool, optional): 是否输出注意力权重. Defaults to False.

            Returns:
                Tensor: 输出特征张量
            """
            attn_outputs = self.attn(x, attention_mask, output_attentions)  #

            # 1.得到注意力输出：a -> (batch_size, n_head, seq_len, n_embd / n_head)
            a = attn_outputs[0]

            # 2.残差连接和层归一化
            n = self.ln_1(x + a)
            # 3.前向传播
            m = self.mlp(n)
            # 4.残差连接层归一化
            h = self.ln_2(n + m)
            # 5.返回最终输出与注意力权重列表 attn_outputs[1:]
            outputs = [h] + attn_outputs[1:]
            return outputs
