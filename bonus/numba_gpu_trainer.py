"""GPU mini-batch trainer for MyTorch networks, implemented with Numba CUDA."""

from __future__ import annotations

from math import ceil, exp, log1p, tanh
from typing import Callable, Sequence

import numpy as np

try:
    from numba import cuda
except (ImportError, RuntimeError) as error:
    cuda = None
    _CUDA_IMPORT_ERROR = error
else:
    _CUDA_IMPORT_ERROR = None


_ACTIVATIONS = {
    "Sigmoid": 1,
    "ReLU": 2,
    "LeakyReLU": 3,
    "Linear": 4,
    "Tanh": 5,
    "Softplus": 6,
    "Softmax": 7,
}


if cuda is not None:

    @cuda.jit(device=True)
    def _activate(value, preactivations, sample, output_size, activation):
        if activation == 1:
            return 1.0 / (1.0 + exp(-value))
        if activation == 2:
            return value if value > 0.0 else 0.0
        if activation == 3:
            return value if value > 0.0 else 0.01 * value
        if activation == 4:
            return value
        if activation == 5:
            return tanh(value)
        if activation == 6:
            return log1p(exp(value))

        maximum = preactivations[sample, 0]
        for index in range(1, output_size):
            if preactivations[sample, index] > maximum:
                maximum = preactivations[sample, index]
        total = 0.0
        for index in range(output_size):
            total += exp(preactivations[sample, index] - maximum)
        return exp(value - maximum) / total


    @cuda.jit(device=True)
    def _activation_derivative(value, preactivations, sample, output_size, activation):
        if activation == 1:
            result = _activate(value, preactivations, sample, output_size, activation)
            return result * (1.0 - result)
        if activation == 2:
            return 1.0 if value > 0.0 else 0.0
        if activation == 3:
            return 1.0 if value > 0.0 else 0.01
        if activation == 4:
            return 1.0
        if activation == 5:
            result = tanh(value)
            return 1.0 - result * result
        if activation == 6:
            return 1.0 / (1.0 + exp(-value))

        result = _activate(value, preactivations, sample, output_size, activation)
        return result * (1.0 - result)


    @cuda.jit
    def _forward_linear_kernel(inputs, weights, biases, preactivations,
                               input_size, output_size):
        index = cuda.grid(1)
        sample_count = preactivations.shape[0]
        if index < sample_count * output_size:
            sample = index // output_size
            neuron = index % output_size
            total = biases[neuron]
            for input_index in range(input_size):
                total += inputs[sample, input_index] * weights[input_index, neuron]
            preactivations[sample, neuron] = total


    @cuda.jit
    def _apply_activation_kernel(preactivations, outputs, output_size, activation):
        index = cuda.grid(1)
        sample_count = outputs.shape[0]
        if index < sample_count * output_size:
            sample = index // output_size
            neuron = index % output_size
            outputs[sample, neuron] = _activate(
                preactivations[sample, neuron],
                preactivations,
                sample,
                output_size,
                activation,
            )


    @cuda.jit
    def _output_delta_kernel(predictions, targets, preactivations, deltas,
                             output_size, activation):
        index = cuda.grid(1)
        if index < predictions.size:
            sample = index // output_size
            neuron = index % output_size
            delta = 2.0 * (predictions[sample, neuron] - targets[sample, neuron])
            delta /= output_size
            delta *= _activation_derivative(
                preactivations[sample, neuron],
                preactivations,
                sample,
                output_size,
                activation,
            )
            deltas[sample, neuron] = delta


    @cuda.jit
    def _hidden_delta_kernel(next_deltas, weights, preactivations, deltas,
                             batch_size, input_size, output_size, activation):
        index = cuda.grid(1)
        if index < batch_size * input_size:
            sample = index // input_size
            neuron = index % input_size
            error = 0.0
            for next_neuron in range(output_size):
                error += next_deltas[sample, next_neuron] * weights[neuron, next_neuron]
            deltas[sample, neuron] = error * _activation_derivative(
                preactivations[sample, neuron],
                preactivations,
                sample,
                input_size,
                activation,
            )


    @cuda.jit
    def _update_parameters_kernel(inputs, deltas, weights, biases,
                                  batch_size, input_size, output_size, learning_rate):
        index = cuda.grid(1)
        if index < input_size * output_size:
            input_neuron = index // output_size
            output_neuron = index % output_size
            gradient = 0.0
            for sample in range(batch_size):
                gradient += inputs[sample, input_neuron] * deltas[sample, output_neuron]
            weights[input_neuron, output_neuron] -= learning_rate * gradient / batch_size

        if index < output_size:
            gradient = 0.0
            for sample in range(batch_size):
                gradient += deltas[sample, index]
            biases[index] -= learning_rate * gradient / batch_size


    @cuda.jit
    def _loss_kernel(predictions, targets, losses, output_size):
        sample = cuda.grid(1)
        if sample < predictions.shape[0]:
            loss = 0.0
            for neuron in range(output_size):
                difference = predictions[sample, neuron] - targets[sample, neuron]
                loss += difference * difference
            losses[sample] = loss / output_size


class NumbaGPUTrainer:
    """Train a MyTorch network with CUDA mini-batches and MSE loss.

    The trainer updates the supplied network in place and advances its learning
    rate scheduler once per epoch. Heaviside and custom loss functions are not
    differentiable/implementable by these CUDA kernels and are rejected.
    """

    def __init__(self, batch_size: int = 512, threads_per_block: int = 128):
        if cuda is None:
            if _CUDA_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "Numba CUDA could not initialize. Install bonus/requirements.txt "
                    "and check the CUDA runtime/compiler libraries."
                ) from _CUDA_IMPORT_ERROR
            raise RuntimeError(
                "Numba CUDA is unavailable. Install bonus/requirements.txt and "
                "use a system with a supported NVIDIA GPU and CUDA driver."
            )
        try:
            cuda_available = cuda.is_available()
        except (OSError, RuntimeError) as error:
            raise RuntimeError("The CUDA runtime could not be initialized.") from error
        if not cuda_available:
            raise RuntimeError(
                "No CUDA device is available. NumbaGPUTrainer requires a supported "
                "NVIDIA GPU and CUDA driver."
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(threads_per_block, int) or threads_per_block <= 0:
            raise ValueError("threads_per_block must be a positive integer.")
        self.batch_size = batch_size
        self.threads_per_block = threads_per_block

    @staticmethod
    def supports_network(network) -> bool:
        """Return whether this trainer implements the network's loss and activations."""
        loss_function = network._settings.loss_function
        if (
            getattr(loss_function, "__module__", None) != "my_torch.builtin_functions"
            or getattr(loss_function, "__name__", None) != "mse_loss"
        ):
            return False

        layers = network._layers
        if not layers:
            return False
        for index, layer in enumerate(layers):
            activation_name = getattr(layer.activation_function, "__name__", None)
            if activation_name not in _ACTIVATIONS:
                return False
            if (
                layer.weights.ndim != 2
                or layer.biases.shape != (layer.weights.shape[1],)
            ):
                return False
            if index and layers[index - 1].weights.shape[1] != layer.weights.shape[0]:
                return False
        return True

    def _launch(self, kernel, item_count: int, *args) -> None:
        blocks = ceil(item_count / self.threads_per_block)
        kernel[blocks, self.threads_per_block](*args)

    @staticmethod
    def _prepare_data(train_data: Sequence) -> tuple[np.ndarray, np.ndarray]:
        if not train_data:
            raise ValueError("train_data must contain at least one sample.")

        def stack_vectors(values, name: str) -> np.ndarray:
            vectors = []
            for value in values:
                vector = np.asarray(value, dtype=np.float64)
                if vector.ndim == 2 and vector.shape[0] == 1:
                    vector = vector[0]
                if vector.ndim != 1:
                    raise ValueError(
                        f"Each training {name} must be a one-dimensional array "
                        "or a single-row matrix."
                    )
                vectors.append(vector)
            try:
                return np.stack(vectors)
            except ValueError as error:
                raise ValueError(
                    f"All training {name} vectors must have the same size."
                ) from error

        inputs = stack_vectors((sample[0] for sample in train_data), "input")
        targets = stack_vectors((sample[1] for sample in train_data), "target")
        if len(inputs) != len(targets):
            raise ValueError("Training input and target counts must match.")
        return np.ascontiguousarray(inputs), np.ascontiguousarray(targets)

    @staticmethod
    def _activation_code(layer) -> int:
        activation_name = layer.activation_function.__name__
        try:
            return _ACTIVATIONS[activation_name]
        except KeyError as error:
            raise ValueError(
                f"Activation {activation_name!r} is not supported by the GPU trainer."
            ) from error

    def _validate_training(self, network, train_data: Sequence, num_epoch: int):
        if not isinstance(num_epoch, int) or num_epoch < 0:
            raise ValueError("num_epoch must be a non-negative integer.")

        loss_function = network._settings.loss_function
        if (
            getattr(loss_function, "__module__", None) != "my_torch.builtin_functions"
            or getattr(loss_function, "__name__", None) != "mse_loss"
        ):
            raise ValueError("NumbaGPUTrainer currently supports only the MSE loss.")

        inputs, targets = self._prepare_data(train_data)
        layers = network._layers
        if not layers:
            raise ValueError("The network must contain at least one layer.")
        if inputs.shape[1] != network._settings.input_count:
            raise ValueError("Training input size does not match the network.")
        if targets.shape[1] != network._settings.output_layer.neurons:
            raise ValueError("Training target size does not match the network.")

        activation_codes = [self._activation_code(layer) for layer in layers]
        for index, layer in enumerate(layers):
            if (
                layer.weights.ndim != 2
                or layer.biases.shape != (layer.weights.shape[1],)
            ):
                raise ValueError(f"Layer {index} has incompatible weights or biases.")
            if index and layers[index - 1].weights.shape[1] != layer.weights.shape[0]:
                raise ValueError(f"Layer {index} has an incompatible input size.")

        return inputs, targets, layers, activation_codes

    @staticmethod
    def _copy_parameters_to_device(layers):
        device_weights = [cuda.to_device(layer.weights) for layer in layers]
        device_biases = [cuda.to_device(layer.biases) for layer in layers]
        return device_weights, device_biases

    def _forward_batch(self, batch_inputs, layers, device_weights, device_biases,
                       activation_codes):
        batch_count = batch_inputs.shape[0]
        activations = [batch_inputs]
        preactivations = []

        for layer_index, layer in enumerate(layers):
            input_size, output_size = layer.weights.shape
            device_pre = cuda.device_array((batch_count, output_size), dtype=np.float64)
            device_output = cuda.device_array_like(device_pre)
            self._launch(
                _forward_linear_kernel,
                batch_count * output_size,
                activations[-1],
                device_weights[layer_index],
                device_biases[layer_index],
                device_pre,
                input_size,
                output_size,
            )
            self._launch(
                _apply_activation_kernel,
                batch_count * output_size,
                device_pre,
                device_output,
                output_size,
                activation_codes[layer_index],
            )
            preactivations.append(device_pre)
            activations.append(device_output)

        return activations, preactivations

    def _calculate_batch_deltas(self, batch_targets, activations, preactivations,
                                layers, activation_codes):
        batch_count = batch_targets.shape[0]
        output_size = layers[-1].weights.shape[1]
        deltas = [
            cuda.device_array((batch_count, layer.weights.shape[1]), dtype=np.float64)
            for layer in layers
        ]
        self._launch(
            _output_delta_kernel,
            batch_count * output_size,
            activations[-1],
            batch_targets,
            preactivations[-1],
            deltas[-1],
            output_size,
            activation_codes[-1],
        )

        device_losses = cuda.device_array(batch_count, dtype=np.float64)
        self._launch(
            _loss_kernel,
            batch_count,
            activations[-1],
            batch_targets,
            device_losses,
            output_size,
        )
        batch_loss = float(np.sum(device_losses.copy_to_host()))
        return deltas, batch_loss

    def _update_batch(self, batch_count, activations, preactivations, deltas,
                      layers, device_weights, device_biases, activation_codes,
                      learning_rate):
        for layer_index in reversed(range(len(layers))):
            layer = layers[layer_index]
            input_size, output_size = layer.weights.shape
            if layer_index > 0:
                self._launch(
                    _hidden_delta_kernel,
                    batch_count * input_size,
                    deltas[layer_index],
                    device_weights[layer_index],
                    preactivations[layer_index - 1],
                    deltas[layer_index - 1],
                    batch_count,
                    input_size,
                    output_size,
                    activation_codes[layer_index - 1],
                )
            self._launch(
                _update_parameters_kernel,
                max(input_size * output_size, output_size),
                activations[layer_index],
                deltas[layer_index],
                device_weights[layer_index],
                device_biases[layer_index],
                batch_count,
                input_size,
                output_size,
                learning_rate,
            )

    def _train_batch(self, inputs, targets, start, end, layers,
                     device_weights, device_biases, activation_codes, learning_rate):
        batch_inputs = cuda.to_device(inputs[start:end])
        batch_targets = cuda.to_device(targets[start:end])
        batch_count = end - start
        activations, preactivations = self._forward_batch(
            batch_inputs, layers, device_weights, device_biases, activation_codes
        )
        deltas, batch_loss = self._calculate_batch_deltas(
            batch_targets, activations, preactivations, layers, activation_codes
        )
        self._update_batch(
            batch_count,
            activations,
            preactivations,
            deltas,
            layers,
            device_weights,
            device_biases,
            activation_codes,
            learning_rate,
        )
        return batch_loss

    def _train_epoch(self, inputs, targets, layers, device_weights, device_biases,
                     activation_codes, learning_rate):
        epoch_loss = 0.0
        for start in range(0, len(inputs), self.batch_size):
            end = min(start + self.batch_size, len(inputs))
            epoch_loss += self._train_batch(
                inputs,
                targets,
                start,
                end,
                layers,
                device_weights,
                device_biases,
                activation_codes,
                learning_rate,
            )
        return epoch_loss / len(inputs)

    @staticmethod
    def _copy_parameters_to_network(layers, device_weights, device_biases):
        for layer, weights, biases in zip(layers, device_weights, device_biases):
            layer.weights[...] = weights.copy_to_host()
            layer.biases[...] = biases.copy_to_host()

    def train(
        self,
        network,
        train_data: Sequence,
        num_epoch: int = 100,
        epoch_callback: Callable[[int, float], None] | None = None,
    ):
        """Train with mini-batch gradient descent; return accuracy and MSE per epoch."""
        inputs, targets, layers, activation_codes = self._validate_training(
            network, train_data, num_epoch
        )
        device_weights, device_biases = self._copy_parameters_to_device(layers)
        accuracy_history: list[float] = []
        loss_history: list[float] = []

        for _ in range(num_epoch):
            learning_rate = network.lr_scheduler.get_learning_rate()
            mean_loss = self._train_epoch(
                inputs,
                targets,
                layers,
                device_weights,
                device_biases,
                activation_codes,
                learning_rate,
            )
            loss_history.append(mean_loss)
            accuracy_history.append((1.0 - mean_loss) * 100.0)
            network.lr_scheduler.step()
            if epoch_callback is not None:
                self._copy_parameters_to_network(layers, device_weights, device_biases)
                epoch_callback(len(loss_history), mean_loss)

        self._copy_parameters_to_network(layers, device_weights, device_biases)

        if loss_history:
            network.last_loss = loss_history[-1]
            network.loss_history.extend(loss_history)
        return accuracy_history, loss_history
