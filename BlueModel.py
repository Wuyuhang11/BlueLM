import torch
import torch.nn as nn
from GPTConfig import GPTConfig
from BlueLMPreTrainedModel import BlueLMPreTrainedModel

class BlueModel(BlueLMPreTrainedModel):
    def __init__(self, config: GPTConfig) -> None:
        """继承基座模型生成一个预训练的Transformer模型

        Args:
            config (GPTConfig): 配置类
        """
        
        super().__init__(config)
        self.config = config
        # 1. 定义两个嵌入层：tokens_embed用于将输入的token ID转换为嵌入向量，positions_embedding将位置信息嵌入到相同维度嵌入向量中
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd) # 模型能够处理的最大序列和嵌入维度
        
        # 2.定义多个Block实例和dropout层构建Transformer主题
        self.dropout = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            [Block(config, scale=True) for _ in range(config.n_layer)]
        )
        # 3.注册了一个名为 position_ids 的缓冲区，用于存储位置 ID，这些 ID 用于位置嵌入层。
        self.register_buffer(
            "position_ids", torch.arange(config.n_positions), persistent=False
        )
        self.post_init()
