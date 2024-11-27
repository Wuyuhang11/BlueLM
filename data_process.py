import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import login, snapshot_download
from trainConfig import train_args


def get_tokenized_datasets(text_path: str, tokenizer: AutoTokenizer) -> Dataset:
    # 定义一个字典，指定训练数据集的路径
    data_files = {"train": text_path}
    
    # 1.加载数据集，基于文档区分样本 -> 我们这里就提供一个样本
    raw_datasets = load_dataset("text", data_files=data_files, sample_by="document")
    
    # 2.分词器能够处理的最大序列长度
    max_seq_length = tokenizer.model_max_length
    
    # 3.定义tokenize_function函数，对样本中的文档进行分词处理
    def tokenize_function(examples):
        """
        作用到数据集的每个样本中，将原始文本转换为模型可以理解的格式

        Args:
            examples (_type_): 样本文本

        Returns:
            _type_: 分词、截断、添加特殊标记，返回处理后的标记序列
            {
                "input_ids": [101, 102, 103, ..., 1000, 1001, 1002],  # 标记ID序列
                "attention_mask": [1, 1, 1, ..., 1, 0, 0],  # 注意力掩码，1表示有效标记，0表示填充或截断
                "token_type_ids": [0, 0, 0, ..., 0]  # 标记类型ID，用于区分两个句子（对于单句子输入，通常全为0）
            }
        """
        return tokenizer(
            examples["text"],
            add_special_tokens=True, # 添加特殊标记
            truncation=True, # 是否截断
            max_length=max_seq_length, # 最大序列文本
            return_overflowing_tokens=True, # 是否返回被截断的部分
        )
        
    # 4.返回处理后的数据集结构
    tokenized_datasets = raw_datasets.map(
        tokenize_function, # 对每个样本文本进行分词，输入为文本样本
        batched=True, # 批量方式处理多样本，以并行方式将tokenize_function函数作用多样本上
        remove_columns="text", # 转换完模型可理解的格式后，去除原始的text列
        desc="分词器在样本文本运行中...",
    )
    
    # 5.截断最后一个样本（长度不足）
    tokenized_datasets = tokenized_datasets.filter(
        # 检查样本中的input_ids 长度是否超过最大序列
        lambda example: len(example["input_ids"]) == max_seq_length
    )

    tokenized_datasets = tokenized_datasets.remove_columns("overflow_to_sample_mapping")

    # 6.拆分训练集和验证集
    train_valid = tokenized_datasets["train"].train_test_split(test_size=0.05)
    tokenized_datasets = DatasetDict(
        {
            "train": train_valid["train"],
            "valid": train_valid["test"],
        }
    )
    print("train:",train_valid["train"])
    print("valid:",train_valid["test"])
    return tokenized_datasets


if __name__ == "__main__":
    if train_args.from_remote:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{train_args.owner}/{train_args.model_name}"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(train_args.model_name)

    tokenized_datasets = get_tokenized_datasets(
        text_path="e:/code/BlueLM/BlueLM/data/doupo.txt", tokenizer=tokenizer
    )

    print(tokenized_datasets)
    login(token="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")
    if train_args.from_remote:
        tokenized_datasets.push_to_hub(f"{train_args.owner}/{train_args.dataset_name}")
    else:
        tokenized_datasets.save_to_disk(train_args.dataset_name)
