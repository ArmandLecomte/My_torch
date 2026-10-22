"""A simple library like PyTorch."""
import numpy as np
from my_torch.typing import Callable
from numbers import Number


__all__ = ["sigmoid", "NeuralNetwork"]

def sigmoid(x):
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def heaviside(x):
    """Compute the Heaviside step function."""
    return np.heaviside(x, 1)


class NeuralNetwork:
    """A simple neural network class."""

    def __init__(self):
        self.weights : list[list[Number]] = []

    def add_layer(self, layer):
        """Add a layer to the neural network."""
        self.weights.append(layer)

    def forward(self, x):
        """Perform a forward pass through the network."""
        
        return x
