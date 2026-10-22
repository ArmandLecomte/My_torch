import numpy as np
from typing import Callable

type _1d = tuple[int]
type _2d = tuple[int, int]

type weight = np.float64
type hidden_layer = np.ndarray[_2d, np.dtype[weight]]
type input_layer = np.ndarray[_2d, np.dtype[weight]]

type bias = np.float64
type biases = list[np.ndarray[_1d, np.dtype[bias]]]

type bias_function = Callable[[],float]
type weight_function = Callable[[],float]

type activation_function = Callable[[weight], weight]