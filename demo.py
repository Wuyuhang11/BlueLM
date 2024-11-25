import torch
import torch.nn as nn
# 0. 超参数设置
embed_dim = 10  # 每条数据的维度
seq_len = 3  # 一共多少数据
batch_size = 2  # 每次吃多少数据
hidden_size = 5  # 隐藏层大小，即数据输出的特征维度

# 1.定义输入数据
x = torch.randn(batch_size, seq_len, embed_dim)

# 2.定义全连接层和一维卷积：假设，我们有5个输出通道（特征），有10个输入通道（特征），卷积核宽度为1，表示每个输出特征都是通过将一个输入特征加权求和得到的。
fc = nn.Linear(
    embed_dim, hidden_size
)  # 全连接层：将embed_dim维的输入映射到hidden_size维的输出上
conv = nn.Conv1d(
    embed_dim, hidden_size, kernel_size=1
)  # 通过一个一维卷积层，将embed_dim维的输入映射到hidden_size维的输出上，卷积核kernel_size为1(1x1)读取输入的中的每个位置的信息

# 3.参数共享：使一维卷积与全连接层的行为一致，每个输出包含输入的加权求和信息
conv.weight = nn.Parameter(
    fc.weight.reshape(hidden_size, embed_dim, 1)
)  # fc.weight是一个二维张量，其(i, :)代表以i全连接层的输出特征数，(:,i)代表输入特征数，使全连接层fc重塑为(hidden_size, embed_dim, 1)
conv.bias = fc.bias

# 4.计算输出
fc_output = fc(x)
x_conv = x.permute(0, 2, 1)
conv_output = conv(x_conv)
conv_output = conv_output.permute(0, 2, 1)

print(torch.allclose(fc_output, conv_output))
