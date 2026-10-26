import numpy as np
from . import typing as mtt

__all__ = ["sigmoid", "sigmoid_derivative", "mse_loss", "mse_derivative",
           "heaviside"]


def sigmoid(x: mtt.layer) -> mtt.layer:
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: mtt.layer) -> mtt.layer:
    """Compute the derivative of the sigmoid activation function."""
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))


def mse_loss(predicted: mtt.layer, targets: mtt.layer) -> float:
    """Mean Squared Error (MSE): L = mean((y_hat - y)^2)"""
    return float(np.mean((predicted - targets) ** 2))


def mse_derivative(predicted: mtt.layer, targets: mtt.layer) -> mtt.layer:
    """Derivative of MSE with respect to predictions (dLoss / dPredicted)."""
    return np.divide(2 * (predicted - targets), predicted.size)


def heaviside(x):
    """Compute the Heaviside step function."""
    return np.heaviside(x, 1)
