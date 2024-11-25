import torch
import torch.nn as nn
from torch import Tensor
from GPTConfig import GPTConfig
from BlueLMPreTrainedModel import BlueLMPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from typing import Tuple, Union, Any
from block import Block

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
        self.positions_embed = nn.Embedding(
            config.n_positions, config.n_embd
        )  # 模型能够处理的最大序列和嵌入维度

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

    # 卡你想传播算法
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        
        """
        Args:
            input_ids (torch.LongTensor): 表示输入数据的张量，类型为长整型张量（torch.LongTensor），形状为 (batch_size, seq_len)。这里 batch_size 是批次中的序列数量，seq_len 是每个序列的长度。
            output_attentions (bool, optional): 一个可选的布尔参数，默认值为 False。如果设置为 True，则方法将返回所有注意力层的注意力张量（即注意力权重）。
            output_hidden_states (bool, optional): 另一个可选的布尔参数，默认值为 False。如果设置为 True，则方法将返回所有层的隐藏状态。
            return_dict (bool, optional): 一个可选的布尔参数，默认值为 False。如果设置为 True，则方法将返回一个 BaseModelOutput 对象，而不是一个普通的元组。

            BaseModelOutput：last_hidden_state: 最后一层的隐藏状态，可以用于序列生成任务中的下一个时间步的输入。hidden_states: 所有层的隐藏状态，这在分析模型内部工作时非常有用。attentions: 注意力权重，可以用于解释模型是如何关注输入序列的不同部分的。
            CausalLMOutput 是一个专门用于因果语言模型（Causal Language Model）的输出类。因果语言模型是一种特殊的生成模型，它在生成文本时，每一步只能依赖于当前和之前的词，而不能看到未来的词。
            注：CausalLMOutput 除了包含 BaseModelOutput 的所有属性外，还可能包含：
            -loss: 如果提供了目标标签（即在训练或评估时），则包含计算的损失值。
            -logits: 经过 softmax 函数转换后的对数，表示每个可能的下一个词的概率分布。
            
        Returns:
            Union[Tuple[torch.Tensor], BaseModelOutput]: 方法的返回值可以是一个包含 torch.Tensor 张量的元组，或者是一个 BaseModelOutput 对象。BaseModelOutput 是一个封装了模型输出的结构体，它提供了一种更结构化的方式来访问模型的输出，如最后一层的隐藏状态、所有层的隐藏状态和注意力权重。
        """

        input_shape = input_ids.size()

        inputs_embeds = self.tokens_embed(input_ids)
        # generate position ids
        position_ids = self.position_ids[None, : input_shape[-1]]

        position_embeds = self.positions_embed(position_ids)

        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.dropout(hidden_states)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for _, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(hidden_states, attention_mask, output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        # add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
