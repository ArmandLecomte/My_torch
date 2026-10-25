from dataclasses import dataclass
from my_torch import typing as mtt, mse_loss, mse_derivative

__all__ = ["LayerSettings", "NNSettings"]


@dataclass
class LayerSettings:
    neurons: int
    default_weight: mtt.weight_function
    default_bias: mtt.bias_function
    activation_function: mtt.activation_function
    derivative_activation: mtt.derivative_activation


@dataclass
class NNSettings:
    input_count: int
    hidden_layers: list[LayerSettings]
    output_layer: LayerSettings
    loss_function: mtt.loss_function = mse_loss
    loss_derivative: mtt.loss_derivative = mse_derivative
