"""GPU mini-batch trainer for MyTorch networks, implemented with Numba CUDA."""

from __future__ import annotations

import warnings
from math import exp, log1p, tanh
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
_EPSILON = 1e-15
_MSE_LOSS = 0
_CROSS_ENTROPY_LOSS = 1
_SHARED_MEMORY_BUDGET = 16 * 1024


def softmax_cross_entropy_loss(predicted: np.ndarray, targets: np.ndarray) -> float:
    """Mean categorical cross-entropy for softmax probabilities and one-hot targets."""
    probabilities = np.clip(predicted, _EPSILON, 1.0)
    return float(-np.mean(np.sum(targets * np.log(probabilities), axis=-1)))


def softmax_cross_entropy_derivative(
        predicted: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Derivative of categorical cross-entropy with respect to probabilities."""
    probabilities = np.clip(predicted, _EPSILON, 1.0)
    return -targets / probabilities


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


    @cuda.jit(device=True)
    def _output_delta(predictions, targets, preactivations, sample, neuron,
                      output_size, activation, loss_kind):
        prediction = predictions[sample, neuron]
        target = targets[sample, neuron]
        if loss_kind == _CROSS_ENTROPY_LOSS:
            return prediction - target

        if activation == 7:
            projection = 0.0
            for output_neuron in range(output_size):
                difference = predictions[sample, output_neuron] - targets[sample, output_neuron]
                projection += predictions[sample, output_neuron] * difference
            difference = prediction - target
            return 2.0 * prediction * (difference - projection) / output_size

        delta = 2.0 * (prediction - target) / output_size
        return delta * _activation_derivative(preactivations[sample, neuron], preactivations, sample, output_size, activation)


    @cuda.jit
    def _forward_linear_kernel(inputs, weights, biases, preactivations, input_size, output_size):
        sample = cuda.blockIdx.x
        sample_count = preactivations.shape[0]
        if sample < sample_count:
            for neuron in range(cuda.threadIdx.x, output_size, cuda.blockDim.x):
                total = biases[neuron]
                for input_index in range(input_size):
                    total += inputs[sample, input_index] * weights[input_index, neuron]
                preactivations[sample, neuron] = total


    @cuda.jit
    def _apply_activation_kernel(preactivations, outputs, output_size, activation):
        sample = cuda.blockIdx.x
        sample_count = outputs.shape[0]
        if sample < sample_count:
            for neuron in range(cuda.threadIdx.x, output_size, cuda.blockDim.x):
                outputs[sample, neuron] = _activate(preactivations[sample, neuron], preactivations, sample,
                                                    output_size, activation)


    @cuda.jit
    def _output_delta_kernel(predictions, targets, preactivations, deltas,
                             output_size, activation, loss_kind):
        sample = cuda.blockIdx.x
        if sample < predictions.shape[0]:
            for neuron in range(cuda.threadIdx.x, output_size, cuda.blockDim.x):
                deltas[sample, neuron] = _output_delta(predictions, targets, preactivations, sample, neuron,
                                                       output_size, activation, loss_kind)


    @cuda.jit
    def _hidden_delta_kernel(next_deltas, weights, preactivations, deltas,
                             batch_size, input_size, output_size, activation):
        sample = cuda.blockIdx.x
        if sample < batch_size:
            for neuron in range(cuda.threadIdx.x, input_size, cuda.blockDim.x):
                error = 0.0
                for next_neuron in range(output_size):
                    error += next_deltas[sample, next_neuron] * weights[neuron, next_neuron]
                deltas[sample, neuron] = error * _activation_derivative(preactivations[sample, neuron], preactivations,
                                                                        sample, input_size, activation)


    @cuda.jit
    def _update_parameters_kernel(inputs, deltas, weights, biases,
                                  batch_size, input_size, output_size, learning_rate,
                                  parameter_count, parameters_per_block):
        lane = cuda.threadIdx.x
        weight_count = input_size * output_size
        partials = cuda.shared.array(0, dtype=np.float64)
        first_parameter = cuda.blockIdx.x * parameters_per_block

        for parameter_offset in range(parameters_per_block):
            parameter = first_parameter + parameter_offset
            gradient = 0.0
            if parameter < weight_count:
                input_neuron = parameter // output_size
                output_neuron = parameter % output_size
                for sample in range(lane, batch_size, cuda.blockDim.x):
                    gradient += (
                            inputs[sample, input_neuron] * deltas[sample, output_neuron]
                    )
            elif parameter < parameter_count:
                output_neuron = parameter - weight_count
                for sample in range(lane, batch_size, cuda.blockDim.x):
                    gradient += deltas[sample, output_neuron]

            base_offset = parameter_offset * cuda.blockDim.x
            partial_offset = base_offset + lane
            partials[partial_offset] = gradient
        cuda.syncthreads()

        stride = cuda.blockDim.x // 2
        while stride > 0:
            for parameter_offset in range(parameters_per_block):
                base_offset = parameter_offset * cuda.blockDim.x
                if lane < stride:
                    partials[base_offset + lane] += partials[
                        base_offset + lane + stride
                        ]
            cuda.syncthreads()
            stride //= 2

        parameter_offset = lane
        parameter = first_parameter + parameter_offset
        if parameter_offset < parameters_per_block and parameter < parameter_count:
            gradient = partials[parameter_offset * cuda.blockDim.x]
            if parameter < weight_count:
                input_neuron = parameter // output_size
                output_neuron = parameter % output_size
                weights[input_neuron, output_neuron] -= learning_rate * gradient / batch_size
            else:
                output_neuron = parameter - weight_count
                biases[output_neuron] -= learning_rate * gradient / batch_size


class NumbaGPUTrainer:
    """Train a MyTorch network with CUDA mini-batches.

    The trainer updates the supplied network in place and advances its learning
    rate scheduler once per epoch. Heaviside and custom loss functions are not
    differentiable/implementable by these CUDA kernels and are rejected.
    """

    def __init__(
            self,
            batch_size: int = 512,
            threads_per_block: int = 128,
            seed: int = 42,
    ):
        self._ensure_cuda_available()
        self._validate_options(batch_size, threads_per_block, seed)
        self.batch_size = batch_size
        self.threads_per_block = threads_per_block
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _ensure_cuda_available() -> None:
        if cuda is None:
            if _CUDA_IMPORT_ERROR is not None:
                raise RuntimeError("Numba CUDA could not initialize. Install bonus/requirements.txt and check the CUDA runtime/compiler libraries.") from _CUDA_IMPORT_ERROR
            raise RuntimeError("Numba CUDA is unavailable. Install bonus/requirements.txt and use a system with a supported NVIDIA GPU and CUDA driver.")
        try:
            cuda_available = cuda.is_available()
        except (OSError, RuntimeError) as error:
            raise RuntimeError("The CUDA runtime could not be initialized.") from error
        if not cuda_available:
            raise RuntimeError("No CUDA device is available. NumbaGPUTrainer requires a supported NVIDIA GPU and CUDA driver.")

    @staticmethod
    def _validate_options(
            batch_size: int, threads_per_block: int, seed: int
    ) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if (
                not isinstance(threads_per_block, int)
                or threads_per_block <= 0
                or threads_per_block > 1024
                or (threads_per_block & (threads_per_block - 1)) != 0
        ):
            raise ValueError("threads_per_block must be a power of two between 1 and 1024.")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer.")

    @staticmethod
    def _is_supported_loss(loss_function) -> tuple[bool, bool]:
        is_mse = (
                getattr(loss_function, "__module__", None) == "my_torch.builtin_functions"
                and getattr(loss_function, "__name__", None) == "mse_loss"
        )
        is_cross_entropy = loss_function is softmax_cross_entropy_loss

        # Also recognize the class-based loss functions from LossFunction.py
        loss_function_module = getattr(loss_function, "__module__", None)
        loss_function_name = getattr(loss_function, "__name__", None)
        if loss_function_module == "my_torch.LossFunction":
            if loss_function_name == "function":
                # Static methods on LossFunction subclasses lose the class context,
                # so check the qualified name to distinguish MSELoss vs CategoricalCrossEntropyLoss
                qualname = getattr(loss_function, "__qualname__", None)
                if qualname == "MSELoss.function":
                    is_mse = True
                elif qualname == "CategoricalCrossEntropyLoss.function":
                    is_cross_entropy = True

        return is_mse, is_cross_entropy

    @staticmethod
    def _has_supported_layers(layers) -> bool:
        if not layers:
            return False
        for index, layer in enumerate(layers):
            activation_name = getattr(layer.activation_function, "__name__", None)
            if activation_name not in _ACTIVATIONS:
                return False
            if layer.weights.ndim != 2 or layer.biases.shape != (layer.weights.shape[1],):
                return False
            if index and layers[index - 1].weights.shape[1] != layer.weights.shape[0]:
                return False
        return True

    @classmethod
    def supports_network(cls, network) -> bool:
        """Return whether this trainer implements the network's loss and activations."""
        is_mse, is_cross_entropy = cls._is_supported_loss(network._settings.loss_function)
        if not (is_mse or is_cross_entropy):
            return False

        layers = network._layers
        if not cls._has_supported_layers(layers):
            return False
        return not is_cross_entropy or layers[-1].activation_function.__name__ == "Softmax"

    def _launch_rows(self, kernel, row_count: int, row_width: int, *args) -> None:
        width = 1 << (row_width - 1).bit_length()
        threads = min(self.threads_per_block, max(32, width))
        kernel[row_count, threads](*args)

    def _launch_parameter_update(self, parameter_count: int, *args) -> None:
        max_parameters_per_block = max(1, _SHARED_MEMORY_BUDGET // (self.threads_per_block * np.dtype(np.float64).itemsize))
        parameters_per_block = min(16, max_parameters_per_block, max(1, parameter_count // 128))
        launch_config = (
            (parameter_count + parameters_per_block - 1) // parameters_per_block,
            parameters_per_block
            * self.threads_per_block
            * np.dtype(np.float64).itemsize,
        )
        if parameter_count < 128:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Grid size \d+ will likely result in GPU under-utilization due to low occupancy\.", category=Warning)
                self._launch_parameter_update_kernel(launch_config, parameter_count, parameters_per_block, args)
        else:
            self._launch_parameter_update_kernel(launch_config, parameter_count, parameters_per_block, args)

    def _launch_parameter_update_kernel(
            self, launch_config, parameter_count: int, parameters_per_block: int, args
    ) -> None:
        block_count, shared_memory = launch_config
        _update_parameters_kernel[block_count, self.threads_per_block, 0, shared_memory](*args, parameter_count, parameters_per_block)

    @staticmethod
    def _stack_vectors(values, name: str) -> np.ndarray:
        vectors = []
        for value in values:
            vector = np.asarray(value, dtype=np.float64)
            if vector.ndim == 2 and vector.shape[0] == 1:
                vector = vector[0]
            if vector.ndim != 1:
                raise ValueError(f"Each training {name} must be a one-dimensional array or a single-row matrix.")
            vectors.append(vector)
        try:
            return np.stack(vectors)
        except ValueError as error:
            raise ValueError(f"All training {name} vectors must have the same size.") from error

    @classmethod
    def _prepare_data(cls, train_data: Sequence) -> tuple[np.ndarray, np.ndarray]:
        if not train_data:
            raise ValueError("train_data must contain at least one sample.")
        inputs = cls._stack_vectors((sample[0] for sample in train_data), "input")
        targets = cls._stack_vectors((sample[1] for sample in train_data), "target")
        if len(inputs) != len(targets):
            raise ValueError("Training input and target counts must match.")
        return np.ascontiguousarray(inputs), np.ascontiguousarray(targets)

    @staticmethod
    def _validate_epoch_count(num_epoch: int) -> None:
        if not isinstance(num_epoch, int) or num_epoch < 0:
            raise ValueError("num_epoch must be a non-negative integer.")

    @staticmethod
    def _validate_layer_shapes(layers) -> None:
        for index, layer in enumerate(layers):
            if (
                    layer.weights.ndim != 2
                    or layer.biases.shape != (layer.weights.shape[1],)
            ):
                raise ValueError(f"Layer {index} has incompatible weights or biases.")
            if index and layers[index - 1].weights.shape[1] != layer.weights.shape[0]:
                raise ValueError(f"Layer {index} has an incompatible input size.")

    @staticmethod
    def _validate_data_shapes(inputs, targets, network) -> None:
        if inputs.shape[1] != network._settings.input_count:
            raise ValueError("Training input size does not match the network.")
        if targets.shape[1] != network._settings.output_layer.neurons:
            raise ValueError("Training target size does not match the network.")

    @classmethod
    def _validate_training(cls, network, train_data: Sequence, num_epoch: int):
        cls._validate_epoch_count(num_epoch)
        is_mse, is_cross_entropy = cls._is_supported_loss(network._settings.loss_function)
        if not (is_mse or is_cross_entropy):
            raise ValueError("NumbaGPUTrainer supports MSE or softmax cross-entropy loss.")

        inputs, targets = cls._prepare_data(train_data)
        layers = network._layers
        if not layers:
            raise ValueError("The network must contain at least one layer.")
        if is_cross_entropy and layers[-1].activation_function.__name__ != "Softmax":
            raise ValueError("Cross-entropy training requires a Softmax output layer.")
        cls._validate_data_shapes(inputs, targets, network)
        cls._validate_layer_shapes(layers)
        activation_codes = [cls._activation_code(layer) for layer in layers]
        loss_kind = _CROSS_ENTROPY_LOSS if is_cross_entropy else _MSE_LOSS
        return inputs, targets, layers, activation_codes, loss_kind

    @staticmethod
    def _activation_code(layer) -> int:
        activation_name = layer.activation_function.__name__
        try:
            return _ACTIVATIONS[activation_name]
        except KeyError as error:
            raise ValueError(f"Activation {activation_name!r} is not supported by the GPU trainer.") from error

    @staticmethod
    def _copy_parameters_to_device(layers):
        device_weights = [cuda.to_device(layer.weights) for layer in layers]
        device_biases = [cuda.to_device(layer.biases) for layer in layers]
        return device_weights, device_biases

    def _forward_batch(self, batch_inputs, layers, device_weights, device_biases,
                       activation_codes):
        activations = [batch_inputs]
        preactivations = []

        for index, layer in enumerate(layers):
            output, preactivation = self._forward_layer(activations[-1], layer,
                                device_weights[index], device_biases[index], activation_codes[index])
            preactivations.append(preactivation)
            activations.append(output)

        return activations, preactivations

    def _forward_layer(
            self, inputs, layer, weights, biases, activation_code
    ):
        batch_count = inputs.shape[0]
        input_size, output_size = layer.weights.shape
        preactivations = cuda.device_array((batch_count, output_size), dtype=np.float64)
        outputs = cuda.device_array_like(preactivations)
        self._launch_rows(_forward_linear_kernel, batch_count, output_size, inputs, weights, biases, preactivations, input_size, output_size)
        self._launch_rows(_apply_activation_kernel, batch_count, output_size, preactivations, outputs, output_size, activation_code)
        return outputs, preactivations

    def _calculate_batch_deltas(self, batch_targets, activations, preactivations,
                                layers, activation_codes, loss_kind):
        batch_count = batch_targets.shape[0]
        output_size = layers[-1].weights.shape[1]
        deltas = self._allocate_deltas(batch_count, layers)
        self._launch_rows(_output_delta_kernel, batch_count, output_size, activations[-1], batch_targets,
                          preactivations[-1], deltas[-1], output_size, activation_codes[-1], loss_kind)
        batch_metrics = self._calculate_batch_metrics(activations[-1], batch_targets, loss_kind)
        return deltas, batch_metrics

    @staticmethod
    def _allocate_deltas(batch_count, layers):
        return [cuda.device_array((batch_count, layer.weights.shape[1]), dtype=np.float64)for layer in layers]

    @staticmethod
    def _calculate_batch_metrics(device_predictions, device_targets, loss_kind):
        predictions = device_predictions.copy_to_host()
        targets = device_targets.copy_to_host()
        if loss_kind == _CROSS_ENTROPY_LOSS:
            losses = -np.sum(targets * np.log(np.clip(predictions, _EPSILON, 1.0)), axis=1)
        else:
            losses = np.mean(np.square(predictions - targets), axis=1)
        loss = float(np.sum(losses))
        accuracy = float(np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)))
        return loss, accuracy

    def _update_batch(self, batch_count, activations, preactivations, deltas, layers,
            device_weights, device_biases, activation_codes, learning_rate):
        for layer_index in reversed(range(len(layers))):
            self._calculate_hidden_delta(layer_index, batch_count, preactivations, deltas, layers, device_weights, activation_codes)
            self._update_layer_parameters(layer_index, batch_count, activations, deltas, layers, device_weights, device_biases, learning_rate)

    def _calculate_hidden_delta(self, layer_index, batch_count, preactivations, deltas, layers, device_weights, activation_codes):
        if layer_index == 0:
            return
        input_size = layers[layer_index].weights.shape[0]
        output_size = layers[layer_index].weights.shape[1]
        self._launch_rows(_hidden_delta_kernel, batch_count, input_size, deltas[layer_index], device_weights[layer_index],
                          preactivations[layer_index - 1], deltas[layer_index - 1], batch_count, input_size, output_size,
                          activation_codes[layer_index - 1])

    def _update_layer_parameters(
            self, layer_index, batch_count, activations, deltas,
            layers,
            device_weights,
            device_biases,
            learning_rate,
    ):
        input_size, output_size = layers[layer_index].weights.shape
        parameter_count = input_size * output_size + output_size
        self._launch_parameter_update(parameter_count, activations[layer_index], deltas[layer_index],
                                      device_weights[layer_index], device_biases[layer_index], batch_count, input_size,
                                      output_size, learning_rate)

    def _train_batch(
            self,
            inputs,
            targets,
            batch_indices,
            layers,
            device_weights,
            device_biases,
            activation_codes,
            loss_kind,
            learning_rate,
    ):
        batch_inputs = cuda.to_device(inputs[batch_indices])
        batch_targets = cuda.to_device(targets[batch_indices])
        batch_count = len(batch_indices)
        activations, preactivations = self._forward_batch(batch_inputs, layers, device_weights, device_biases, activation_codes)
        deltas, (batch_loss, batch_accuracy) = self._calculate_batch_deltas(batch_targets, activations, preactivations, layers, activation_codes, loss_kind)
        self._update_batch(batch_count, activations, preactivations, deltas, layers, device_weights, device_biases, activation_codes, learning_rate)
        return batch_loss, batch_accuracy

    def _batch_indices(self, sample_count):
        shuffled_indices = self._rng.permutation(sample_count)
        for start in range(0, sample_count, self.batch_size):
            yield shuffled_indices[start: start + self.batch_size]

    def _train_epoch(
            self,
            inputs,
            targets,
            layers,
            device_weights,
            device_biases,
            activation_codes,
            loss_kind,
            learning_rate,
    ):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for batch_indices in self._batch_indices(len(inputs)):
            batch_loss, batch_accuracy = self._train_batch(inputs, targets, batch_indices, layers, device_weights,
                                                           device_biases, activation_codes, loss_kind, learning_rate)
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        return self._average_metrics(epoch_loss, epoch_accuracy, len(inputs))

    @staticmethod
    def _average_metrics(total_loss, total_accuracy, sample_count):
        return total_loss / sample_count, 100.0 * total_accuracy / sample_count

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
        """Train with mini-batch gradient descent; return accuracy and loss per epoch."""
        inputs, targets, layers, activation_codes, loss_kind = self._validate_training(network, train_data, num_epoch)
        device_weights, device_biases = self._copy_parameters_to_device(layers)
        accuracy_history: list[float] = []
        loss_history: list[float] = []
        self._train_epochs(network, inputs, targets, layers, device_weights, device_biases, activation_codes, loss_kind,
                           num_epoch, accuracy_history, loss_history, epoch_callback)
        self._copy_parameters_to_network(layers, device_weights, device_biases)
        self._store_training_history(network, loss_history)
        return accuracy_history, loss_history

    def _train_epochs(self, network, inputs, targets, layers, device_weights, device_biases, activation_codes,
            loss_kind, num_epoch, accuracy_history, loss_history, epoch_callback):
        for epoch in range(num_epoch):
            mean_loss, mean_accuracy = self._train_one_epoch(network, inputs, targets, layers, device_weights,
                                        device_biases, activation_codes, loss_kind)
            loss_history.append(mean_loss)
            accuracy_history.append(mean_accuracy)
            self._notify_epoch(epoch_callback, epoch + 1, mean_loss, layers, device_weights, device_biases)

    def _train_one_epoch(self, network, inputs, targets, layers, device_weights, device_biases,
            activation_codes, loss_kind):
        learning_rate = network.lr_scheduler.get_learning_rate()
        metrics = self._train_epoch(inputs, targets, layers, device_weights, device_biases, activation_codes, loss_kind, learning_rate)
        network.lr_scheduler.step()
        return metrics

    def _notify_epoch(self, callback, epoch, loss, layers, device_weights, device_biases):
        if callback is None:
            return
        self._copy_parameters_to_network(layers, device_weights, device_biases)
        callback(epoch, loss)

    @staticmethod
    def _store_training_history(network, loss_history):
        if loss_history:
            network.last_loss = loss_history[-1]
            network.loss_history.extend(loss_history)
