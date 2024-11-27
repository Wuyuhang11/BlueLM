from dataclasses import dataclass


@dataclass
class TrainArguments:
    batch_size: int = 16 # 训练时每个批次的样本数量，默认为16
    weight_decay: float = 1e-1 # 权重衰减，用于正则化，防止过拟合，默认为0.1
    epochs: int = 50 # 训练的总轮数，默认为50
    warmup_proportion: float = 0.05 # 预热比例，用于学习率预热策略，默认为0.05，意味着在训练的前5%的步骤中逐渐增加学习率
    learning_rate: float = 4e-5 # 学习率，用于控制模型权重更新的步长，默认为4e-5
    logging_steps = 100 # 日志记录的步数间隔，默认每100步记录一次日志
    gradient_accumulation_steps: int = 1 # 梯度累积的步数，默认为1，意味着每步都会更新权重。如果设置为大于1的值，那么会在累积了指定步数的梯度后才更新权重
    max_grad_norm: float = 1.0 # 梯度裁剪的最大范数，默认为1.0，用于防止梯度爆炸
    use_wandb: bool = False # 是否使用 Weights & Biases（wandb）进行实验跟踪，默认为False
    from_remote: bool = True # 是否从远程获取数据或模型，默认为True
    dataset_name: str = "doupo-dataset"
    model_name: str = "BlueLM-doupo" 
    tokenizer_name: str = "BlueLM-doupo" # 分词器的名称
    owner: str = "Wuyuhang11"
    devices: str = "0"


train_args = TrainArguments()
