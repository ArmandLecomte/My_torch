from typing import Any

from ._config import get_finite_number, get_positive_integer, validate_mapping

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
    @classmethod
    def from_config(cls, config: Any, description: str) -> "ConstantLR":
        config = validate_mapping(config, {"type", "initial_lr"}, set(), description)
        initial_lr = get_finite_number(config["initial_lr"], f"{description}.initial_lr")
        if initial_lr <= 0:
            raise ValueError(f"{description}.initial_lr must be greater than zero.")
        return cls(initial_lr)

    def compute(self, epoch: int) -> float:
        return self.initial_lr


class ExponentialLR(LRScheduler):
    """lr = initial_lr * decay_rate ** epoch"""

    @classmethod
    def from_config(cls, config: Any, description: str) -> "ExponentialLR":
        config = validate_mapping(
            config, {"type", "initial_lr", "decay_rate"}, set(), description
        )
        initial_lr = get_finite_number(config["initial_lr"], f"{description}.initial_lr")
        decay_rate = get_finite_number(config["decay_rate"], f"{description}.decay_rate")
        if initial_lr <= 0:
            raise ValueError(f"{description}.initial_lr must be greater than zero.")
        if not 0 < decay_rate <= 1:
            raise ValueError(f"{description}.decay_rate must be in the range (0, 1].")
        return cls(initial_lr, decay_rate)

    def __init__(self, initial_lr: float, decay_rate: float):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def compute(self, epoch: int) -> float:
        return self.initial_lr * self.decay_rate ** epoch


class StepLR(LRScheduler):
    """lr = initial_lr * gamma ** (epoch // step_size)"""

    @classmethod
    def from_config(cls, config: Any, description: str) -> "StepLR":
        config = validate_mapping(
            config,
            {"type", "initial_lr", "step_size"},
            {"gamma"},
            description,
        )
        initial_lr = get_finite_number(config["initial_lr"], f"{description}.initial_lr")
        step_size = get_positive_integer(config["step_size"], f"{description}.step_size")
        gamma = get_finite_number(config.get("gamma", 0.1), f"{description}.gamma")
        if initial_lr <= 0:
            raise ValueError(f"{description}.initial_lr must be greater than zero.")
        if not 0 < gamma <= 1:
            raise ValueError(f"{description}.gamma must be in the range (0, 1].")
        return cls(initial_lr, step_size, gamma)

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def compute(self, epoch: int) -> float:
        return self.initial_lr * self.gamma ** (epoch // self.step_size)
