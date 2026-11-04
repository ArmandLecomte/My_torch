import copy
import pickle

import numpy as np

from . import typing as mtt
from .Layer import Layer
from .LearningRate import LRScheduler
from .NNSettings import NNSettings

__all__ = ["NeuralNetwork"]

class NeuralNetwork:
    """A simple neural network class."""

    def __init__(self, settings: NNSettings):
        self._settings: NNSettings = settings
        # The network owns its scheduler so the caller's settings stay untouched.
        self.lr_scheduler: LRScheduler = copy.deepcopy(settings.lr_scheduler)
        self._layers: list[Layer] = self._init_layers()
        self.last_loss: float | None = None
        self.loss_history: list[float] = []

    def _init_layers(self) -> list[Layer]:
        """Initialize all hidden layers according to the layer definition."""
        layers: list[Layer] = []
        prev_size = self._settings.input_count

        for layer_setting in self._settings.hidden_layers:
            layers.append(Layer(prev_size, layer_setting))
            prev_size = layer_setting.neurons

        layers.append(Layer(prev_size, self._settings.output_layer))
        return layers

    def _keep_all_forward(self, input_layer: mtt.layer) -> list[mtt.layer]:
        if input_layer.size != self._settings.input_count:
            raise ValueError(
                "Input layer size does not match the defined input size.")

        activations: list[mtt.layer] = [input_layer]
        for layer in self._layers:
            activations.append(layer.apply(activations[-1]))
        return activations

    def forward(self, input_layer: mtt.layer) -> mtt.layer:
        if input_layer.size != self._settings.input_count:
            raise ValueError(
                "Input layer size does not match the defined input size.")
        return self._keep_all_forward(input_layer)[-1]

    def compute_loss(self, predicted: mtt.layer, targets: mtt.layer) -> float:
        """Calculate the scalar loss using the configured loss function."""
        return self._settings.loss_function(predicted, targets)

    def back_propagation(self, inputs: mtt.layer, targets: mtt.layer,
                         learning_rate: float | None = None):
        """Run one update. Uses the scheduler's rate unless `learning_rate` overrides it."""
        if inputs.size != self._settings.input_count:
            raise ValueError(
                "Input layer size does not match the defined input size.")
        if targets.size != self._settings.output_layer.neurons:
            raise ValueError(
                "Expected output size does not match the defined output size.")

        if learning_rate is None:
            learning_rate = self.lr_scheduler.get_learning_rate()
        activations = self._keep_all_forward(inputs)
        delta = activations[-1] - targets
        current_loss = self.compute_loss(activations[-1], targets)
        self.last_loss = current_loss
        self.loss_history.append(current_loss)

        for i in reversed(range(len(self._layers))):
            prev_activation = activations[i]
            current = self._layers[i]
            gradient = prev_activation.T.dot(delta)
            if i > 0:
                hidden_derivative = current.activation_function.backward(prev_activation)
                error = delta.dot(current.weights.T)
                delta = error * hidden_derivative
            current.weights -= learning_rate * gradient

        return current_loss

    def step_lr_scheduler(self) -> None:
        """Advance the scheduler by one epoch (done by `train`; call it in custom loops)."""
        self.lr_scheduler.step()

    def train(self, train_data: mtt.train_data, num_epoch: int = 100):
        accuracy = []
        losses = []
        for j in range(num_epoch):
            epoch_losses = []
            for inputs, target in train_data:
                loss = self.back_propagation(inputs, target)
                epoch_losses.append(loss)
            losses.append(np.mean(epoch_losses))
            accuracy.append((1 - losses[-1]) * 100)
            self.step_lr_scheduler()
        return accuracy, losses

    def save(self, path: str):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> "NeuralNetwork":
        with open(path, "rb") as file:
            return pickle.load(file)
