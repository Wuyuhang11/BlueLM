import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset, load_from_disk # 加载数据集
from transformers import ( # 分词器
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from torch.utils.data.dataloader import DataLoader # 加载数据集
from huggingface_hub import login, snapshot_download
from torch.optim import AdamW # 优化器
import torch
from tqdm import tqdm # 进度条
from log import logger
from utils import EarlyStopper, generate
from trainConfig import train_args # 训练参数
from GPTConfig import GPTConfig
from BlueLMHeadModel import BlueLMHeadModel # 输出层


def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    """
    权重衰减，一种正则化技术，在损失函数中添加一个与参数相关的惩罚项来防止过拟合。
    Args:
        model (_type_): 模型实例
        weight_decay (_type_): 权重衰减
        no_decay (list, optional): 不应用权重衰减的参数列表. 默认为 ["bias", "LayerNorm.weight"]

    Returns:
        _type_: 返回一个包含两个字典的列表，其中一个字典为参数，另一个为对应的权重衰减值
    """
    params_with_wd, params_without_wd = [], [] # 初始化两个空列表，分别存储需要和不需要的权重衰减参数
    for n, p in model.named_parameters(): # 遍历模型的所有参数，n 是参数名称，p 是参数张量
        if any(nd in n for nd in no_decay): # 如果 n 包含在 no_decay 中的任意字符串
            params_without_wd.append(p) # 将该参数添加到不需要权重衰减的列表中
        else:
            params_with_wd.append(p) # 否则，将该参数添加到权重衰减的列表当中
    return [
        {"params": params_with_wd, "weight_decay": weight_decay}, # 返回需要权重衰减的参数，权重衰减系数为 weight_decay
        {"params": params_without_wd, "weight_decay": 0.0}, # 返回不需要权重衰减的参数，权重衰减系数为 0 
    ]


def train(model, train_dataloader, val_dataloader, optimizer, device, scheduler, args):
    """
    训练模型的主要函数。
    
    Args:
        model (torch.nn.Module): 模型实例。
        train_dataloader (DataLoader): 训练数据加载器。
        val_dataloader (DataLoader): 验证数据加载器。
        optimizer (Optimizer): 优化器实例。
        device (torch.device): 设备（如 GPU 或 CPU）。
        scheduler (LR_scheduler): 学习率调度器。
        args (Namespace): 包含训练参数的命名空间。
    """
    max_grad_norm = args.max_grad_norm  # 最大梯度范数，用于梯度裁剪：放梯度范数超过指定阈值，就会进行缩放防止梯度爆炸，有效提高模型收敛速度
    logging_steps = args.logging_steps  # 日志记录的步数间隔
    gradient_accumulation_steps = args.gradient_accumulation_steps  # 梯度累积的步数，每过一次累积步数，就会更新一次模型参数，越小频率越频繁需要大量计算，消耗时间长【另外对噪声也会十分敏感，不自然】

    total_loss = 0.0  # 累计总损失
    logging_loss = 0.0  # 上一次日志记录时的累计损失
    best_loss = 10000  # 最佳验证损失，初始化为一个较大的值【用于早停】
    global_steps = 0  # 全局训练步数

    early_stopper = EarlyStopper()  # 早停机制的实例

    for epoch in range(args.epochs):  # 遍历每个 epoch
        model.train()  # 设置模型为训练模式
        p_bar = tqdm(train_dataloader, disable=False)  # 创建进度条
        
        # 1.多次batch_size数据训练
        for step, batch in enumerate(p_bar):  # 遍历训练数据加载器中的批次,step=0,batch=16为批大小，p_bar为进度条对象
            batch = {k: v.to(device) for k, v in batch.items()}  # 将批次数据中的数据和标签移动到GPU
            outputs = model(batch["input_ids"])  # 前向传播，因为在做pretrain，所以输入数据和和标签是相同的【这里我不传入label】
            loss = outputs.loss  # 获取损失

            total_loss += loss.item()  # 累计损失

            p_bar.set_description(
                f"epoch {epoch + 1:2d} (loss={loss.item():5.3f} | global_steps {global_steps:4d} | lr {scheduler.get_last_lr()[0]:.5f} )"
            )  # 更新进度条描述

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps  # 如果使用梯度累积，平均损失

            loss.backward()  # 反向传播：计算每个参数对损失函数的敏感度，即改变某个参数，敏感度如何变化【求导为梯度】

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪：将模型梯度总范数控制在max_grad_norm=1以内

            if (step + 1) % gradient_accumulation_steps == 0:  # 每累积一定步数后更新参数
                optimizer.step()  # 更新优化器
                scheduler.step()  # 更新学习率调度器
                optimizer.zero_grad()  # 清空梯度
                global_steps += 1  # 增加全局步数

                if logging_steps > 0 and global_steps % logging_steps == 0:  # 如果达到日志记录步数
                    if args.use_wandb:
                        train_loss = (total_loss - logging_loss) / (
                            logging_steps * gradient_accumulation_steps
                        )  # 计算当前的日志记录损失
                        wandb.log(
                            {
                                "global_steps": global_steps,
                                "lr": scheduler.get_lr()[0],
                                "train_loss:": train_loss,
                            }
                        )  # 使用 WandB 记录日志

                    logging_loss = total_loss  # 更新日志记录的损失

        # 2.每经过一次训练，就评估一次（一次训练=多次batch_size的数据训练完）
        eval_loss = evalute(model, val_dataloader, device)  # 在验证集上评估模型
        logger.info(
            f"epoch {epoch} | global_steps {global_steps}  | eval loss {eval_loss:.3f}"
        )  # 记录当前 epoch 的评估损失

        if args.use_wandb:
            wandb.log({"epoch": epoch, "eval_loss:": eval_loss})  # 使用 WandB 记录评估损失

        if eval_loss < best_loss:  # 如果当前评估损失低于历史最佳损失
            best_loss = eval_loss  # 更新最佳损失
            logger.info(
                f"Saving model to {args.model_name} with best eval loss {eval_loss:.3f}"
            )  # 保存模型：每隔一次epoch，如果loss降低就会保存一次model
            model.save_pretrained(f"{args.model_name}")  # 保存到本地磁盘

        torch.cuda.empty_cache()  # 清空 GPU 缓存

        if early_stopper.step(eval_loss):  # 如果满足早停条件：验证集上的表现没有明显提升，就早停
            print(f"Stop from early stopping.")  # 打印早停信息
            break  # 提前终止训练

@torch.no_grad()
def evalute(model, dataloader, device):
    model.eval()
    p_bar = tqdm(dataloader, desc="iter", disable=False)

    total_loss = 0.0

    for batch in p_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(batch["input_ids"], labels=batch["input_ids"])

        total_loss += outputs.loss.item()

    test_loss = total_loss / len(dataloader)

    return test_loss


# 主函数
if __name__ == "__main__":
    # run train_tokenizer.py to get tokenizer
    if train_args.from_remote:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{train_args.owner}/{train_args.tokenizer_name}", use_fast=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{train_args.tokenizer_name}", use_fast=True
        )

    if train_args.use_wandb:
        import wandb

        wandb.init(
            project="simple-gpt",
            config=vars(train_args),
        )

    config = GPTConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BlueLMHeadModel(config)
    model.to(device)
    # run data_process.py to get dataset
    if train_args.from_remote:
        tokenized_dataset = load_dataset(
            f"{train_args.owner}/{train_args.dataset_name}"
        )
    else:
        tokenized_dataset = load_from_disk(f"{train_args.dataset_name}")

    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["valid"]

    batch_size = int(train_args.batch_size / train_args.gradient_accumulation_steps)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    total_training_steps = int(
        train_args.epochs
        * len(train_dataloader)
        / train_args.gradient_accumulation_steps
    )

    print(f"total train steps={total_training_steps}")

    optimizer = AdamW(
        get_grouped_params(model, weight_decay=train_args.weight_decay),
        lr=train_args.learning_rate,
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_args.warmup_proportion * total_training_steps),
        num_training_steps=total_training_steps,
    )

    train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        device,
        lr_scheduler,
        train_args,
    )

    model = BlueLMHeadModel.from_pretrained(f"{train_args.model_name}")
    generated_text = generate(
        model, tokenizer, device, "肖炎经过不懈地修炼，终于突破到了斗帝级别"
    )

    print(f"generated text: {generated_text}")
    if train_args.from_remote:
        logger.info(f"Pushing model to {train_args.owner}/{train_args.model_name}")
        login(token="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")
        model.push_to_hub(f"{train_args.owner}/{train_args.model_name}")
