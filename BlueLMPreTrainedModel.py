from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn
import GPTConfig
from conv import Conv1D
from attention import Attention


class BlueLMPreTrainedModel(PreTrainedModel):
    """继承来自Transformers的PreTrainedModel，定义一个模型基类，用于传入配置文件，定义参数初始化的方法

    Args:
        PreTrainedModel (_type_): 预训练模型基类
    """

    config_class = GPTConfig
    base_model_prefix = "transformer"

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

    def _init_weights(self, module):
        """初始化模型权重

        Args:
            module (_type_): _description_
        """

        # 1. 判断module是否是线性层或者一维卷积层
        if isinstance(module, (nn.Linear, Conv1D)):
            # 以正态分布的形式进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模型的偏置不为空，偏置初始化为0
            if module.bias is not None:
                module.bias.data.zero_
        # 2.判断Embedding，进行初始化
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_
        # 3.判断层归一化
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_
            module.weight.data.fill_(1.0)
            