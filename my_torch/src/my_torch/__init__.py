"""A simple library like PyTorch."""

from . import builtin_functions, typing
from .ActivationFunction import (
    ActivationFunction,
    Heaviside,
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Softplus,
    Tanh,
)
from .LearningRate import LRScheduler, ExponentialLR, ConstantLR, StepLR
from .Layer import Layer
from .NNSettings import LayerSettings, NNSettings
from .NeuralNetwork import NeuralNetwork
from .NNGenerator import generate_neural_network
from .builtin_functions import (
    mse_derivative,
    mse_loss,
)

__all__ = [
    "Layer",
    "ActivationFunction",
    "Sigmoid",
    "Heaviside",
    "ReLU",
    "LeakyReLU",
    "Linear",
    "Tanh",
    "Softplus",
    "Softmax",
    "LayerSettings",
    "NNSettings",
    "NeuralNetwork",
    "generate_neural_network",
    "typing",
    "ExponentialLR",
    "LRScheduler",
    "ConstantLR",
    "StepLR",
    *builtin_functions.__all__,
]
