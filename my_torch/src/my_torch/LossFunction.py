from abc import ABC, abstractmethod

import numpy as np

from . import typing as mtt

__all__ = [
    "LossFunction", "MSELoss", "CategoricalCrossEntropyLoss",
]


class LossFunction(ABC):
    def __call__(self, predicted: mtt.layer, targets: mtt.layer) -> float:
        return self.function(predicted, targets)

    @staticmethod
    @abstractmethod
    def function(predicted: mtt.layer, targets: mtt.layer) -> float:
        raise NotImplementedError("Loss function method not implemented.")

    @staticmethod
    def derivative(predicted: mtt.layer, targets: mtt.layer) -> mtt.layer:
        raise NotImplementedError("Loss derivative method not implemented.")


class MSELoss(LossFunction):
    @staticmethod
    def function(predicted: mtt.layer, targets: mtt.layer) -> float:
        """Compute Mean Squared Error loss: mean((predicted - targets)^2).

        Output layer only: Yes.
        Scenario: Regression tasks predicting continuous numerical values.
        Solves: Measures discrepancy between predictions and targets, penalizing larger errors quadratically.
        """
        return float(np.mean((predicted - targets) ** 2))

    @staticmethod
    def derivative(predicted: mtt.layer, targets: mtt.layer) -> mtt.layer:
        """Compute derivative of MSE loss with respect to predictions: 2 * (predicted - targets) / N.

        Output layer only: Yes.
        Scenario: Output layer backpropagation in MSE regression models.
        Solves: Provides initial gradient direction and magnitude to initiate backward error propagation.
        """
        return np.divide(2 * (predicted - targets), predicted.size)


class CategoricalCrossEntropyLoss(LossFunction):
    @staticmethod
    def function(predicted: mtt.layer, targets: mtt.layer) -> float:
        """Compute categorical cross-entropy loss: -sum(targets * log(predicted)).

        Output layer only: Yes.
        Scenario: Multi-class classification with softmax output layer.
        Solves: Measures divergence between true class distribution and predicted probabilities.
        """
        epsilon = 1e-15
        clipped = np.clip(predicted, epsilon, 1 - epsilon)
        return float(-np.sum(targets * np.log(clipped)))

    @staticmethod
    def derivative(predicted: mtt.layer, targets: mtt.layer) -> mtt.layer:
        """Compute derivative of categorical cross-entropy with respect to predictions.

        Output layer only: Yes.
        Scenario: Output layer backpropagation in classification models with softmax.
        Solves: Provides gradient that penalizes confident wrong predictions more strongly.
        """
        epsilon = 1e-15
        clipped = np.clip(predicted, epsilon, 1 - epsilon)
        return -targets / clipped
