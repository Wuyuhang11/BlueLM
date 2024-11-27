import torch
import torch.nn as nn
import GPTConfig
import BlueModel
from torch import Tensor
from typing import Tuple, Union, Any
import BlueLMPreTrainedModel
import BlueModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput


class BlueLMHeadModel(BlueLMPreTrainedModel):
    """输出层

    Args:
        BlueLMPreTrainedModel (_type_): 基座模型,初始化权重
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        # 1.定义BlueLM模型,命名为transformer
        self.transformer = BlueModel(config)
        # 2.定义一个线性层 lm_head ,用于将 Transformer 的输出转化为 vocab 大小的 logits ,以便于词汇输出的预测
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 3.在子类中执行一些额外的初始化工作。
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor = None,  # labels 用于训练正确的token IDs序列，标签label初始化为 None 用于 Pretrain 无监督训练
        attention_mask: torch.FloatTensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[Tensor], CausalLMOutput]:
        """_summary_

        Args:
            input_ids (torch.LongTensor): token IDs
            labels (torch.LongTensor, optional): 可选的标签. Defaults to None.
            attention_mask (torch.FloatTensor, optional): 注意力掩码. Defaults to None.
            output_attentions (bool, optional): 是否输出注意力权重. Defaults to False.
            output_hidden_states (bool, optional): 是否输出隐藏层状态. Defaults to False.
            return_dict (bool, optional): 输出标志. Defaults to True.

        Returns:
            Union[Tuple[Tensor], CausalLMOutput]: _description_
        """
        # 1.注意力掩码的处理 (与应用掩码后的 weights 相加,导致掩码位置元素大小非常小,那么在加权求和的时候,这些位置的贡献将会被忽略 (因为Softmax的作用))
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将掩码转换为与模型权重参数 weights 相同的数据类型
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            # 将掩码中的值转换为负数，这样在计算注意力分数时，被掩码的位置将被赋予非常小的权重
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        print("mask:", attention_mask)

        # 2.Transformer 输出，执行自注意力和前馈神经网络的计算
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print("transformer_output:", transformer_outputs)

        # 3.得到最后一层的隐藏状态
        hidden_states = transformer_outputs[0]
        print("hidden_output:", hidden_states)

        # 4.通过 Linear 线性层计算当前 token 的 logits（token依赖于序列中不同位置）
        lm_logits = self.lm_head(hidden_states)
        print("logits:", lm_logits)

        # 5.计算损失
        loss = None
        if labels is not None:
            # 进行偏移操作: 将logits向右移动一个位置，因为右移一个的token才是我们预测的token，contiguous保证张量移动时候，数据是连续的
            shift_logits = lm_logits[..., :-1, :].contiguous()
            # 同样将标签右移动，以预测下一个token
            shift_labels = labels[..., 1:].contiguous()
            # 定义损失函数，CrossEntropyLoss交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            # 计算偏移后的logits和标签之间的损失
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        # 6.输出
        if not return_dict:
            # 将logits添加到最后一层
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
