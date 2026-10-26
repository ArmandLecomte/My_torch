"""A simple library like PyTorch."""

from . import Layer, NNSettings, NeuralNetwork, typing
from .Layer import Layer
from .NNSettings import LayerSettings, NNSettings
from .NeuralNetwork import NeuralNetwork
from .builtin_functions import sigmoid, sigmoid_derivative, mse_loss, mse_derivative

_modules = [NeuralNetwork, NNSettings, Layer, builtin_functions]

__all__ = [
    "typing",
    *(symbol for module in _modules for symbol in getattr(module, "__all__", [])),
]