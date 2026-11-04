from dataclasses import dataclass, field
from . import typing as mtt
from .LearningRate import LRScheduler, ConstantLR
from .builtin_functions import mse_loss, mse_derivative

__all__ = ["LayerSettings", "NNSettings"]


@dataclass
class LayerSettings:
    neurons: int
    default_weight: mtt.weight_function
    default_bias: mtt.bias_function
    activation_function: mtt.activation_function


@dataclass
class NNSettings:
    input_count: int
    hidden_layers: list[LayerSettings]
    output_layer: LayerSettings
    loss_function: mtt.loss_function = mse_loss
    loss_derivative: mtt.loss_derivative = mse_derivative
    lr_scheduler: LRScheduler = field(default_factory=lambda: ConstantLR(0.1))
