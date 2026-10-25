from my_torch import typing as mtt, Layer, NNSettings

__all__ = ["NeuralNetwork"]


class NeuralNetwork:
    """A simple neural network class."""

    def __init__(self, settings: NNSettings):
        self._settings: NNSettings = settings
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
                         learning_rate: float = 0.1):
        if inputs.size != self._settings.input_count:
            raise ValueError(
                "Input layer size does not match the defined input size.")
        if targets.size != self._settings.output_layer.neurons:
            raise ValueError(
                "Expected output size does not match the defined output size.")

        activations = self._keep_all_forward(inputs)

        current_loss = self.compute_loss(activations[-1], targets)
        self.last_loss = current_loss
        self.loss_history.append(current_loss)

        delta = activations[-1] - targets

        for i in reversed(range(len(self._layers))):
            prev_activation = activations[i]
            current = self.layers[i]

            gradient = prev_activation.T.dot(delta)

            if i > 0:
                hidden_derivative = current.derivative_activation(
                    prev_activation)
                error = delta.dot(current.weights.T)
                delta = error * hidden_derivative

            current.weights -= learning_rate * gradient

    @property
    def layers(self):
        return self._layers
