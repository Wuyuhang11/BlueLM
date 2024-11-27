from log import logger

def generate(model, tokenizer, device, prefix):
    model = model.to(device)
    input_ids = tokenizer.encode(
        prefix, return_tensors="pt", add_special_tokens=False
    ).to(device)
    beam_output = model.generate(
        input_ids,
        max_length=512,
        num_beams=3,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        repetition_penalty=1.25,
    )

    return tokenizer.decode(beam_output[0], skip_special_tokens=True).replace(" ", "")


class EarlyStopper:
    def __init__(self, patience: int = 5, mode: str = "min") -> None:
        # 初始化早停器，设置耐心值（patience）和模式（mode），默认为 min
        self.patience = patience  # 允许没有性能提升的最大连续周期数
        self.counter = patience  # 初始化计数器，用于记录连续没有性能提升的周期数
        # 检查模式是否有效，模式可以是"min"或"max"
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")  # 如果模式无效，抛出异常

        # 根据模式初始化最佳值, max 为寻找最大所以初始化为 0 ，min 寻找最小所以初始化为 inf 无穷大
        self.best_value = 0.0 if mode == "max" else float("inf")  # 如果是最大化问题，初始化为0；如果是最小化问题，初始化为无穷大
        self.mode = mode  # 保存模式

    def step(self, value: float) -> bool:
        # 检查当前值是否比最佳值更好
        if self.is_better(value):
            self.best_value = value  # 更新最佳值
            self.counter = self.patience  # 重置计数器
        else:
            self.counter -= 1  # 如果当前值不比最佳值好，计数器减1

        # 如果计数器归零，表示已经连续patience个周期没有性能提升，返回True触发早停
        if self.counter == 0:
            return True

        # 如果计数器不是patience，记录剩余的早停周期数
        if self.counter != self.patience:
            logger.info(f"early stop left: {self.counter}")

        return False  # 如果没有触发早停，返回False

    def is_better(self, a: float) -> bool:
        # 根据模式判断当前值是否比最佳值更好
        if self.mode == "min":
            return a < self.best_value  # 最小化问题，当前值小于最佳值则更好
        return a > self.best_value  # 最大化问题，当前值大于最佳值则更好