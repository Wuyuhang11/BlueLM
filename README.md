# BlueLM
<div align="center">
    <img src="[https://github.com/user-attachments/assets/17dcefe0-378c-4a19-9104-8ee1d540973c](https://github.com/user-attachments/assets/3a2877e0-fb02-4f11-be1f-80d43dd31ab0)" alt="BlueLM" />
</div>

**1.介绍：**
我们从0到1实现了一个Pretrain-model，名为BlueLM，能够根据输入对《斗破苍穹》实现文本续写

**2.相关工作：**
1. 训练了一个基于《斗破苍穹》小说的中文分词器：https://huggingface.co/Wuyuhang11/BlueLM-doupo.
2. 利用分词器对文档进行处理，所拆分的训练集和验证集上传至：https://huggingface.co/datasets/Wuyuhang11/doupo-dataset.
3. 在输入层，我们模拟了对输入token的嵌入与位置嵌入向量的实现.
4. 我们简单实现了Transformer模块，包括多头注意力机制与残差连接、前向传播。在细节方面，我们实现了Q、K、V的拆分、注意力分数的计算、多头注意力头的合并等等.
5. 输出层面，我们实现了注意力掩码的处理与token的logits输出.
6. 定义了一个简单的早停器用于BlueLM的训练，并实现了BlueLM的训练过程
7. 
**3.演示：**

