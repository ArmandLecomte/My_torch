import numpy as np
from . import typing as mtt

__all__ = ["sigmoid", "sigmoid_derivative", "mse_loss", "mse_derivative",
           "heaviside", "relu", "relu_derivative"]


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

def relu(x):
    """Compute the ReLU activation function."""
    return np.fmax(x, 0.0)

def relu_derivative(x):
    """Compute the derivative of the ReLU activation function."""
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu(x):
    """Compute the leaky ReLU activation function with alpha=0.01."""
    return np.where(x > 0, x, 0.01 * x)

def leaky_relu_derivative(x):
    """Compute the derivative of the leaky ReLU activation function with alpha=0.01."""
    return np.where(x > 0, 1.0, 0.01)

def linear(x):
    """Compute the linear activation function."""
    return x

def linear_derivative(x):
    """Compute the derivative of the linear activation function."""
    return np.ones_like(x)

def tanh(x):
    """Compute the hyperbolic tangent activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Compute the derivative of the hyperbolic tangent activation function."""
    return 1 - np.tanh(x) ** 2
