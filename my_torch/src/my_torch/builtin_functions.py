import numpy as np
from . import typing as mtt

__all__ = [
    "sigmoid", "sigmoid_derivative", "mse_loss", "mse_derivative",
    "heaviside", "relu", "relu_derivative", "leaky_relu",
    "leaky_relu_derivative", "linear", "linear_derivative",
    "tanh", "tanh_derivative", "softplus", "softplus_derivative",
    "softmax", "softmax_derivative",
]


def sigmoid(x: mtt.layer) -> mtt.layer:
    """Compute the sigmoid activation function: f(x) = 1 / (1 + exp(-x)).

    Output Layer Only:
        No (can be used in hidden layers, but primarily recommended for output layers).

    When to use:
        - Output Layer: Used for binary classification (single output predicting probability in (0, 1))
          or multi-label classification (multiple independent probability outputs).
        - Hidden Layers: Useful in shallow architectures or gating units, but generally avoided in deep
          networks due to the vanishing gradient problem.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: mtt.layer) -> mtt.layer:
    """Compute the derivative of the sigmoid activation function: f'(x) = f(x) * (1 - f(x)).

    Output Layer Only:
        No.

    When to use:
        - Backpropagation: Used to compute gradients with respect to pre-activation inputs for any layer
          (hidden or output) utilizing the sigmoid activation function.
    """
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))


def mse_loss(predicted: mtt.layer, targets: mtt.layer) -> float:
    """Mean Squared Error (MSE): L = mean((y_hat - y)^2)"""
    return float(np.mean((predicted - targets) ** 2))


def mse_derivative(predicted: mtt.layer, targets: mtt.layer) -> mtt.layer:
    """Derivative of MSE with respect to predictions (dLoss / dPredicted)."""
    return np.divide(2 * (predicted - targets), predicted.size)


def heaviside(x):
    """Compute the Heaviside step function: f(x) = 1 if x >= 0 else 0.

    Output Layer Only:
        No.

    When to use:
        - Binary Thresholding: Used in step-based perceptron models or decision units requiring hard binary
          (0/1) outputs without smooth probability. Not suitable for standard gradient descent training
          because its derivative is zero almost everywhere.
    """
    return np.heaviside(x, 1)


def relu(x):
    """Compute the Rectified Linear Unit (ReLU) activation function: f(x) = max(0, x).

    Output Layer Only:
        No (primarily intended for hidden layers).

    When to use:
        - Hidden Layers: The standard default activation for hidden layers in deep neural networks to
          mitigate vanishing gradients and ensure fast, computationally efficient training.
        - Output Layer: Generally not recommended for output layers, except for specific regression tasks
          where target outputs are strictly non-negative.
    """
    return np.fmax(x, 0.0)


def relu_derivative(x):
    """Compute the derivative of the ReLU activation function: f'(x) = 1 if x > 0 else 0.

    Output Layer Only:
        No.

    When to use:
        - Backpropagation: Used during backward passes to calculate gradients for layers (primarily hidden
          layers) using ReLU activation.
    """
    return np.where(x > 0, 1.0, 0.0)


def leaky_relu(x):
    """Compute the Leaky ReLU activation function: f(x) = x if x > 0 else 0.01 * x.

    Output Layer Only:
        No (primarily intended for hidden layers).

    When to use:
        - Hidden Layers: Used in hidden layers to prevent the "dying ReLU" problem by allowing a small,
          non-zero gradient when the input is negative.
        - Output Layer: Rarely used in output layers; applicable only when piecewise linear unbounded
          positive and scaled negative outputs are specifically desired.
    """
    return np.where(x > 0, x, 0.01 * x)


def leaky_relu_derivative(x):
    """Compute the derivative of the Leaky ReLU activation function with alpha=0.01.

    Output Layer Only:
        No.

    When to use:
        - Backpropagation: Used during backward passes to calculate gradients for layers (primarily hidden
          layers) using Leaky ReLU activation.
    """
    return np.where(x > 0, 1.0, 0.01)


def linear(x):
    """Compute the linear (identity) activation function: f(x) = x.

    Output Layer Only:
        No.

    When to use:
        - Output Layer: Standard activation function for general regression tasks predicting unbounded
          continuous values in (-inf, +inf).
        - Hidden Layers: Used for linear bottleneck projections, dimensionality reduction, or identity
          mappings (not recommended for general stacked hidden layers as multiple linear layers collapse
          to a single linear mapping).
    """
    return x


def linear_derivative(x):
    """Compute the derivative of the linear activation function: f'(x) = 1.

    Output Layer Only:
        No.

    When to use:
        - Backpropagation: Used to compute gradients for layers utilizing linear activation (passes
          gradients through unchanged).
    """
    return np.ones_like(x)


def tanh(x):
    """Compute the hyperbolic tangent (tanh) activation function: f(x) = tanh(x).

    Output Layer Only:
        No.

    When to use:
        - Hidden Layers: Preferred in hidden layers when zero-centered activations in the range (-1, 1) are
          advantageous for convergence.
        - Output Layer: Used for regression or classification tasks where targets are strictly bounded
          within [-1, 1].
    """
    return np.tanh(x)


def tanh_derivative(x):
    """Compute the derivative of the hyperbolic tangent activation function: f'(x) = 1 - tanh(x)^2.

    Output Layer Only:
        No.

    When to use:
        - Backpropagation: Used during backward passes to calculate gradients for layers (hidden or output)
          using tanh activation.
    """
    return 1 - np.tanh(x) ** 2


def softplus(x):
    """Compute the softplus activation function: f(x) = log(1 + exp(x)).

    Output Layer Only:
        No.

    When to use:
        - Hidden Layers: Used as a smooth, continuously differentiable alternative to ReLU.
        - Output Layer: Used for regression tasks where predictions must be strictly positive continuous
          values (e.g., standard deviations, variances, or rate parameters).
    """
    return np.log1p(np.exp(x))


def softplus_derivative(x):
    """Compute the derivative of the softplus activation function: f'(x) = 1 / (1 + exp(-x)).

    Output Layer Only:
        No.

    When to use:
        - Backpropagation: Used during backward passes to calculate gradients for layers using softplus
          activation (evaluates to the sigmoid function).
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Compute the softmax activation function: f(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x))).

    Output Layer Only:
        Yes (almost exclusively used in output layers).

    When to use:
        - Output Layer: Used in the output layer for multi-class classification problems to convert
          unnormalized logits into a mutually exclusive probability distribution summing to 1 across classes.
          Not suitable for hidden layers because it normalizes across all neurons in the layer.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def softmax_derivative(x):
    """Compute the derivative of the softmax activation function.

    Output Layer Only:
        Yes (matches softmax output layer).

    When to use:
        - Backpropagation: Used during backward passes to compute gradients for output layers utilizing
          softmax activation.
    """
    s = softmax(x)
    return s * (1 - s)
