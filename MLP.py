from torch.nn import functional as F
import torch.nn as nn
from conv import Conv1D
from torch import Tensor
import GPTConfig

class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        embed_dim = config.n_embed
        self.c_fc = Conv1D(embed_dim, embed_dim * 4) # 特征变换: 增强模型非线性表达能力，学习更为复杂的特征
        self.c_proj = Conv1D(embed_dim * 4, embed_dim) # 特征压缩: 压缩信息，既能学习复杂特征。又能使模型更专注于重要信息
        self.act = F.gelu # 激活函数
        self.dropout = nn.Dropout(config.dropout) # 创建Dropout层
        
    def forward(self, x: Tensor) -> Tensor: 
        """前向传播

        Args:
            x (Tensor): (batch_size, seq_len, embed_dim)

        Returns:
            Tensor: (batch_size, seq_len, embed_dim)
        """
        h = self.act(self.c_fc(x)) # 特征变换后经过激活函数
        h = self.c_proj(h) # 特征压缩