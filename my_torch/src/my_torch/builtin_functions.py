import numpy as np

from . import typing as mtt

__all__ = [
    "mse_loss", "mse_derivative",
]

def mse_loss(predicted: mtt.layer, targets: mtt.layer) -> float:
    """Compute Mean Squared Error loss: mean((predicted - targets)^2).

    Output layer only: Yes.
    Scenario: Regression tasks predicting continuous numerical values.
    Solves: Measures discrepancy between predictions and targets, penalizing larger errors quadratically.
    """
    return float(np.mean((predicted - targets) ** 2))


def mse_derivative(predicted: mtt.layer, targets: mtt.layer) -> mtt.layer:
    """Compute derivative of MSE loss with respect to predictions: 2 * (predicted - targets) / N.

    Output layer only: Yes.
    Scenario: Output layer backpropagation in MSE regression models.
    Solves: Provides initial gradient direction and magnitude to initiate backward error propagation.
    """
    return np.divide(2 * (predicted - targets), predicted.size)
