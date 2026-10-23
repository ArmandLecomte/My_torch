"""A simple library like PyTorch."""
from dataclasses import dataclass

import numpy as np
from my_torch import typing as mtt

__all__ = ["sigmoid", "heaviside", "NeuralNetwork", "LayersDefinition"]

def sigmoid(x):
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def heaviside(x):
    """Compute the Heaviside step function."""
    return np.heaviside(x, 1)


@dataclass
class LayersDefinition:
    input_size: int
    hidden_layers: list[int]
    output_size: int
    default_weight: mtt.weight_function
    default_bias: mtt.bias_function

class NeuralNetwork:
    """A simple neural network class."""

    def __init__(self, layers_definition: LayersDefinition,
                 activation_function: mtt.activation_function = sigmoid):
        self._layers_definition = layers_definition
        self._layers : list[mtt.layer] = self._init_layers()
        self._biases : mtt.biases = self._init_biases()
        self._activation_function = activation_function

    def _create_layer_weights(self, in_features: int,
                              out_features: int) -> mtt.layer:
        """Create a weight matrix layer initialized using the default value generator."""
        total_elements = out_features * in_features
        weights = np.fromiter(
            (self._layers_definition.default_weight() for _ in
             range(total_elements)),
            dtype=np.float64,
            count=total_elements,
        )
        return weights.reshape(in_features, out_features)

    def _init_layers(self) -> list[mtt.layer]:
        """Initialize all hidden layers according to the layer definition."""
        layers: list[mtt.layer] = []
        prev_size = self._layers_definition.input_size

        for layer_size in self._layers_definition.hidden_layers:
            layer = self._create_layer_weights(in_features=prev_size, out_features=layer_size)
            layers.append(layer)
            prev_size = layer_size

        layers.append(self._create_layer_weights(in_features=prev_size, out_features=self._layers_definition.output_size))
        return layers

    def forward(self, input_layer: mtt.layer):
        if input_layer.size != self._layers_definition.input_size:
            raise ValueError("Input layer size does not match the defined input size.")

        layer_output: mtt.layer = input_layer
        for i in range(len(self._layers)):
            layer_output = self.apply_layer(i, layer_output)
        return layer_output

    def apply_layer(self, i: int, prev_output: mtt.layer) -> mtt.layer:
        layer_result = prev_output.dot(self._layers[i])
        biased_result = layer_result + self._biases[i]
        neurons_output = self._activation_function(biased_result)
        return neurons_output

    @property
    def layers(self):
        return self._layers

    @property
    def biases(self):
        return self._biases

    def _init_biases(self) -> mtt.biases:
        biases_count = len(self._layers_definition.hidden_layers) + 1
        biases: mtt.biases = []
        for i in range(biases_count):
            bias_count = self._layers_definition.hidden_layers[i] if i < len(self._layers_definition.hidden_layers) else self._layers_definition.output_size
            bias = np.fromiter(
                (self._layers_definition.default_weight() for _ in
                 range(bias_count)),
                dtype=np.float64,
                count=bias_count)
            biases.append(bias)
        return biases
