from transformers import AutoTokenizer  # 导入transformers库中的AutoTokenizer类
import torch  # 导入PyTorch库

from BlueLMHeadModel import BlueLMHeadModel  # 导入自定义的BlueLMHeadModel类

# 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained("Wuyuhang11/BlueLM-doupo")

# 加载预训练的模型，并将其移动到指定GPU
model = BlueLMHeadModel.from_pretrained("Wuyuhang11/BlueLM-doupo").to(device)

# 定义输入的文本前缀
prefix = "肖炎经过不懈修炼，终于达到一个俯视一众强者的高度"

# 使用分词器将文本前缀编码成模型可以理解的输入ID，不添加特殊标记，并将其移动到指定设备
input_ids = tokenizer.encode(prefix, return_tensors="pt", add_special_tokens=False).to(device)

# 进行三次文本生成过程
for i in range(3):
    # 生成文本，设置生成参数
    beam_output = model.generate(
        input_ids,  # 输入的编码ID
        max_length=512,  # 生成文本的最大长度
        num_beams=5,  # 束搜索的宽度
        no_repeat_ngram_size=2,  # 防止生成重复的n-gram
        early_stopping=True,  # 如果达到最大长度则提前停止
        do_sample=True,  # 启用随机采样
        top_k=50,  # 采样时只考虑概率最高的50个词
        top_p=0.95,  # 采样时只考虑累积概率达到95%的词
        repetition_penalty=1.25,  # 惩罚重复词
    )

    # 打印分隔线
    print("-" * 219)
    # 使用分词器将生成的文本ID解码成字符串，并移除所有空格，然后打印
    print(tokenizer.decode(beam_output[0], skip_special_tokens=True).replace(" ", ""))