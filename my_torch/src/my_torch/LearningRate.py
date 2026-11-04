__all__ = ["LRScheduler", "ConstantLR", "ExponentialLR", "StepLR"]


class LRScheduler:
    """Base class: the learning rate is a function of the current epoch.

    The network owns one scheduler. It reads `get_learning_rate()` on every
    update and calls `step()` once per epoch.
    """

    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.epoch = 0

    def compute(self, epoch: int) -> float:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_learning_rate(self) -> float:
        return self.compute(self.epoch)

    def step(self) -> None:
        self.epoch += 1

    def reset(self) -> None:
        self.epoch = 0


class ConstantLR(LRScheduler):
    def compute(self, epoch: int) -> float:
        return self.initial_lr


class ExponentialLR(LRScheduler):
    """lr = initial_lr * decay_rate ** epoch"""

    def __init__(self, initial_lr: float, decay_rate: float):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def compute(self, epoch: int) -> float:
        return self.initial_lr * self.decay_rate ** epoch


class StepLR(LRScheduler):
    """lr = initial_lr * gamma ** (epoch // step_size)"""

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def compute(self, epoch: int) -> float:
        return self.initial_lr * self.gamma ** (epoch // self.step_size)
