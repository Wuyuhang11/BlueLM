import torch
import torch.nn as nn
from torch import Tensor
import GPTConfig

# Conv1D的实现
class Conv1D(nn.Module):  # 继承神经网络基座
    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Args:
            in_features (int): 输入特征
            out_features (int): 输出特征
        """
        super().__init__()  # 初始化父类对象实例
        self.out_features = out_features  # 输出特征赋值给实例
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )  # 创建一个形状为(in_features,out_features)的权重矩阵作为模型的参数列表
        self.bias = nn.Parameter(torch.zeros(out_features))  # 初始化模型的偏置向量

    def forward(self, x: Tensor) -> Tensor:
        """
        Summary：利用模型Conv1D对输入 x 执行卷积操作
        Args:
            x (Tensor): (batch_size, seq_len, embed_dim)

        Returns:
            Tensor: 返回 (batch_size, se_len, out_features)
        """

        # 输出格式
        size_out = x.size()[:-1] + (self.out_features,)

        # 定义卷积操作
        # x.view(-1, x.size(-1)): 除去最后一个维度即 embed_dim不变 ，然后将前两个维度相乘，即 batch_size*seq_len 作为一个step时间步内的向量（可以理解为将输入平坦展开）
        # -> x = (batch_size*seq_len[一个时间步内的输入内容], embed_dim[输入元素的特征数量]) ->  (batch_size * seq_len,embed_dim) x (embed_dim, out_features)
        # -> (batch_size * seq_len, out_features)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)

        # 返回标准输出格式
        x = x.view(size_out)
        return x


# Test
embed_dim = 768
# 特征变换
conv1d = Conv1D(embed_dim, embed_dim * 3)
# 输入:(batch_size, seq_len, embed_dim)
x = torch.rand(2, 5, embed_dim)
x = conv1d(x)
print(x.shape)
