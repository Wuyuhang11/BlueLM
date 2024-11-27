import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing, BertProcessing
from huggingface_hub import login, snapshot_download
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from trainConfig import train_args


def train(file_path: str, save_path="tokenizer.json", vocab_size: int = 5000) -> None:
    """训练分词器

    Args:
        file_path (str): 文件路径
        save_path (str, optional): 保存路径. Defaults to "tokenizer.json".
        vocab_size (int, optional): 词汇表大小. Defaults to 5000.
    """

    # 1.创建一个分词器tokenizer，使用BPE模型，并设置一个未知（UNK）标记
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))

    # 2.创建一个BPE训练器，指定了特殊标记和词汇表大小
    # - speical_tokens：表示特殊的语法/语义，如开始、结束等；
    # - vocab_size：模型能够学习的最大不同 token 的数量
    trainer = BpeTrainer(special_tokens=["<|endoftext|>"], vocab_size=vocab_size)

    # 3.设置一个预分词器为：BertTokenizer
    # - 作用：将长文本分割为更小的单元，去除不必要的空格
    tokenizer.pre_tokenizer = BertPreTokenizer()

    # 4.使用BPE训练器训练分词器
    tokenizer.train([file_path], trainer)

    # 5.后处理分词器【*重要】
    # - 在tokenizer处理文本后，对生成的token进一步处理，在生成的标记中添加特殊标记，如：<|startoftext|，<|endotext|>>
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",  # 序列结束的特殊标记
        pair="$A <|endoftext|> $B:1 <|endoftext|>:1",  # 字符串模板
        special_tokens=[
            (
                "<|endoftext|>",
                tokenizer.token_to_id("<|endoftext|>"),
            ),  # 一个特殊标记列表：包含特殊标记以及对应 ID（从分词器词汇表中获取）
        ],
    )
    # 6.打印词汇表大小
    print(f"vocab size: {tokenizer.get_vocab_size()}")
    # 7.分词器保存为json文件
    tokenizer.save(save_path)


if __name__ == "__main__":
    # 定义文本结束的特殊标记
    eos_token = "<|endoftext|>"
    # 调用train函数开始训练分词器，传入训练数据的路径和特殊标记
    train("e:/code/BlueLM/BlueLM/data/doupo.txt")
    # 加载训练好的分词器模型，设置最大序列长度为512
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json", model_max_length=512
    )
    # 设置未知标记（UNK）为特殊标记
    tokenizer.unk_token = eos_token
    # 将开始标记（BOS）、结束标记（EOS）和填充标记（PAD）都设置为特殊标记
    tokenizer.bos_token = tokenizer.unk_token
    tokenizer.eos_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.unk_token
    # 如果from_remote为True，则上传分词器到Hugging Face Hub，并从Hub加载分词器
    if train_args.from_remote:
        # tokenizer.push_to_hub(f"{train_args.owner}/{train_args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            f"{train_args.owner}/{train_args.model_name}"
        )
    # 如果from_remote为False，则将分词器保存到本地，并从本地加载分词器
    else:
        tokenizer.save_pretrained(train_args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(train_args.model_name)
    # 使用分词器对两个文本进行编码
    encodes = tokenizer("三十年河东三十年河西，莫欺少年穷！", "突破斗者！")
    # 打印编码后的输出，包括input_ids, attention_mask等
    print(encodes)
    # 将编码后的input_ids转换回文本标记（tokens）
    print(tokenizer.convert_ids_to_tokens(encodes["input_ids"]))
