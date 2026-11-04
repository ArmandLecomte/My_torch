from abc import ABC, abstractmethod

import numpy as np

from . import typing as mtt

__all__ = [
    "ActivationFunction", "Sigmoid", "Heaviside", "ReLU", "LeakyReLU",
    "Linear", "Tanh", "Softplus", "Softmax",
]


class ActivationFunction(ABC):
    def __call__(self, x: mtt.layer) -> mtt.layer:
        return self.forward(x)

    @staticmethod
    @abstractmethod
    def forward(x: mtt.layer) -> mtt.layer:
        raise NotImplementedError("Forward method not implemented.")

    @staticmethod
    def backward(x: mtt.layer) -> mtt.layer:
        raise NotImplementedError("Backward method not implemented.")


class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(x: mtt.layer) -> mtt.layer:
        """Compute sigmoid activation: 1 / (1 + exp(-x)).

        Output layer only: No (recommended for output layer).
        Scenario: Binary classification or multi-label classification.
        Solves: Maps unbounded inputs to bounded (0, 1) probabilities for threshold decisions.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x: mtt.layer) -> mtt.layer:
        """Compute derivative of sigmoid activation: s * (1 - s).

        Output layer only: No.
        Scenario: Backpropagation in layers using sigmoid activation.
        Solves: Computes gradients efficiently from forward-pass activations.
        """
        return np.multiply(Sigmoid.forward(x), (1 - Sigmoid.forward(x)))


class Heaviside(ActivationFunction):
    @staticmethod
    def forward(x: mtt.layer) -> mtt.layer:
        """Compute Heaviside step function: 1 if x >= 0 else 0.

        Output layer only: No.
        Scenario: Hard-threshold binary decision units and step perceptrons.
        Solves: Produces crisp discrete binary outputs (non-differentiable; not used in gradient descent).
        """
        return np.heaviside(x, 1)


class ReLU(ActivationFunction):
    @staticmethod
    def forward(x: mtt.layer) -> mtt.layer:
        """Compute Rectified Linear Unit activation: max(0, x).

        Output layer only: No (primarily for hidden layers).
        Scenario: Default hidden layers in deep neural networks, or strictly non-negative outputs.
        Solves: Eliminates vanishing gradients for positive inputs and accelerates training convergence.
        """
        return np.fmax(x, 0.0)

    @staticmethod
    def backward(x: mtt.layer) -> mtt.layer:
        """Compute derivative of ReLU activation: 1 if x > 0 else 0.

        Output layer only: No.
        Scenario: Backpropagation in layers using ReLU activation.
        Solves: Passes full gradient through active neurons while zeroing inactive ones.
        """
        return np.where(x > 0, 1.0, 0.0)


class LeakyReLU(ActivationFunction):
    @staticmethod
    def forward(x: mtt.layer) -> mtt.layer:
        """Compute Leaky ReLU activation with alpha=0.01: x if x > 0 else 0.01 * x.

        Output layer only: No (primarily for hidden layers).
        Scenario: Hidden layers prone to neuron inactivity / deactivation.
        Solves: Prevents the 'dying ReLU' issue by allowing small gradient flow for negative inputs.
        """
        return np.where(x > 0, x, 0.01 * x)

    @staticmethod
    def backward(x: mtt.layer) -> mtt.layer:
        """Compute derivative of Leaky ReLU activation with alpha=0.01.

        Output layer only: No.
        Scenario: Backpropagation in layers using Leaky ReLU activation.
        Solves: Ensures continuous gradient updates across all negative and positive inputs.
        """
        return np.where(x > 0, 1.0, 0.01)


class Linear(ActivationFunction):
    @staticmethod
    def forward(x: mtt.layer) -> mtt.layer:
        """Compute linear (identity) activation: x.

        Output layer only: No (standard for output layer in regression).
        Scenario: Regression output layers, linear bottlenecks, or identity skip connections.
        Solves: Allows unbounded continuous predictions across (-inf, +inf) without compression.
        """
        return x

    @staticmethod
    def backward(x: mtt.layer) -> mtt.layer:
        """Compute derivative of linear activation: 1.

        Output layer only: No.
        Scenario: Backpropagation in layers using linear activation.
        Solves: Passes incoming gradients through unchanged with unit scaling.
        """
        return np.ones_like(x)


class Tanh(ActivationFunction):
    @staticmethod
    def forward(x: mtt.layer) -> mtt.layer:
        """Compute hyperbolic tangent activation: tanh(x).

        Output layer only: No.
        Scenario: Hidden layers needing zero-centered activations, or targets bounded in [-1, 1].
        Solves: Centers activations around zero to avoid systematic gradient bias during optimization.
        """
        return np.tanh(x)

    @staticmethod
    def backward(x: mtt.layer) -> mtt.layer:
        """Compute derivative of hyperbolic tangent activation: 1 - tanh(x)^2.

        Output layer only: No.
        Scenario: Backpropagation in layers using tanh activation.
        Solves: Scales gradients based on saturation level, providing smooth backpropagation updates.
        """
        return 1 - np.tanh(x) ** 2


class Softplus(ActivationFunction):
    @staticmethod
    def forward(x: mtt.layer) -> mtt.layer:
        """Compute softplus activation: log(1 + exp(x)).

        Output layer only: No.
        Scenario: Hidden layers or output layers predicting strictly positive quantities (e.g., variance).
        Solves: Provides a smooth, continuously differentiable approximation of ReLU with non-zero curvature.
        """
        return np.log1p(np.exp(x))

    @staticmethod
    def backward(x: mtt.layer) -> mtt.layer:
        """Compute derivative of softplus activation: 1 / (1 + exp(-x)).

        Output layer only: No.
        Scenario: Backpropagation in layers using softplus activation.
        Solves: Computes smooth sigmoid gradients avoiding sharp derivative discontinuities.
        """
        return 1 / (1 + np.exp(-x))


class Softmax(ActivationFunction):
    @staticmethod
    def forward(x: mtt.layer) -> mtt.layer:
        """Compute softmax activation: exp(x - max(x)) / sum(exp(x - max(x))).

        Output layer only: Yes.
        Scenario: Output layer in multi-class classification tasks.
        Solves: Normalizes class scores into a valid probability distribution summing to 1 across mutually exclusive classes.
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def backward(x: mtt.layer) -> mtt.layer:
        """Compute derivative of softmax activation: s * (1 - s).

        Output layer only: Yes.
        Scenario: Backpropagation in output layers using softmax activation.
        Solves: Computes diagonal gradient response for multi-class probability outputs.
        """
        s = Softmax.forward(x)
        return s * (1 - s)
