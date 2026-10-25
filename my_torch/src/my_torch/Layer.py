import numpy as np

from my_torch import typing as mtt, LayerSettings

__all__ = ["Layer"]

class Layer:
    def __init__(self, input_neurons_count: int, settings: LayerSettings):
        self._settings = settings
        self.weights: mtt.layer = self._create_layer_weights(input_neurons_count)
        self.biases: mtt.biases = self._create_biases()
        self._previous_output: mtt.layer | None = None

    def _create_layer_weights(self, in_features: int) -> mtt.layer:
        """Create a weight matrix layer initialized using the default value generator."""
        total_elements = self._settings.neurons * in_features
        weights = np.fromiter(
            (self.default_weight() for _ in
             range(total_elements)),
            dtype=np.float64,
            count=total_elements,
        )
        return weights.reshape(in_features, self._settings.neurons)

    def _create_biases(self) -> mtt.biases:
        bias = np.fromiter(
            (self.default_bias() for _ in
             range(self._settings.neurons)),
            dtype=np.float64,
            count=self._settings.neurons)
        return bias

    def apply(self, layer_input: mtt.layer):
        layer_result = layer_input.dot(self.weights)
        biased_result = layer_result + self.biases
        return self.activation_function(biased_result)

    @property
    def activation_function(self):
        return self._settings.activation_function

    @property
    def derivative_activation(self):
        return self._settings.derivative_activation

    @property
    def default_weight(self):
        return self._settings.default_weight

    @property
    def default_bias(self):
        return self._settings.default_bias
