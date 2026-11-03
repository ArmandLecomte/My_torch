from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["weight", "layer", "bias", "biases", "bias_function",
           "weight_function", "activation_function", "derivative_activation",
           "loss_function", "loss_derivative", "train_data", "DataPair"]

type _1d = tuple[int]
type _2d = tuple[int, int]

type weight = np.float64
type layer = np.ndarray[_2d, np.dtype[weight]]

type bias = np.float64
type biases = np.ndarray[_1d, np.dtype[bias]]

type bias_function = Callable[[], float]
type weight_function = Callable[[], float]

type activation_function = Callable[[layer], layer]
type derivative_activation = activation_function

type loss_function = Callable[[layer, layer], float]
type loss_derivative = Callable[[layer, layer], layer]


@dataclass
class DataPair:
    input: layer
    target: layer

    def __getitem__(self, key):
        if key == 0:
            return self.input
        elif key == 1:
            return self.target
        else:
            raise IndexError("not valid key '{}'".format(key))

type train_data = list[DataPair]