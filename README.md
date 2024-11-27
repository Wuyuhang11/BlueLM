# BlueLM
<div align="center">
    <img src="https://github.com/user-attachments/assets/2cf7b081-7392-42f2-9954-316407d83aea" alt="BlueLM" />
</div>


**1.介绍：**
我们从 `0` 到 `1` 实现了一个`Pretrain-model`，名为 `BlueLM` ，能够根据输入对《斗破苍穹》实现文本续写

**2.相关工作：**
1. 训练了一个基于《斗破苍穹》小说的中文分词器： https://huggingface.co/Wuyuhang11/BlueLM-doupo.
2. 利用分词器对文档进行处理，所拆分的训练集和验证集上传至： https://huggingface.co/datasets/Wuyuhang11/doupo-dataset.
3. 在输入层，我们模拟了对输入 `token` 的嵌入与位置嵌入向量的实现.
4. 我们简单实现了 `Transformer` 模块，包括多头注意力机制与残差连接、前向传播。在细节方面，我们实现了 `Q` 、 `K` 、 `V` 的拆分、注意力分数的计算、多头注意力头的合并等等.
5. 输出层面，我们实现了注意力掩码的处理与 `token` 的 `logits` 输出.
6. 定义了一个简单的早停器用于 `BlueLM` 的训练，并实现了 `BlueLM` 的训练过程
7. 完整的展示了 `GPT` 在 `Pretrain` 阶段的实现


**3.演示：**

