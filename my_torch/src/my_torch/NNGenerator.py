import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ActivationFunction import (
    ActivationFunction,
    Heaviside,
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Softplus,
    Tanh,
)
from ._config import get_finite_number as _finite_number
from ._config import get_positive_integer as _positive_integer
from ._config import validate_mapping as _validate_mapping
from .LearningRate import ConstantLR, ExponentialLR, LRScheduler, StepLR
from .NNSettings import LayerSettings, NNSettings
from .NeuralNetwork import NeuralNetwork

__all__ = ["generate_neural_network"]


@dataclass(frozen=True)
class UniformInitializer:
    low: float
    high: float

    @classmethod
    def from_config(cls, config: Any, description: str) -> "UniformInitializer":
        config = _validate_mapping(config, {"type", "low", "high"}, set(), description)
        low = _finite_number(config["low"], f"{description}.low")
        high = _finite_number(config["high"], f"{description}.high")
        if low > high:
            raise ValueError(f"{description}.low must not exceed {description}.high.")
        return cls(low, high)

    def __call__(self) -> float:
        return float(np.random.uniform(self.low, self.high))


@dataclass(frozen=True)
class NormalInitializer:
    mean: float
    std: float

    @classmethod
    def from_config(cls, config: Any, description: str) -> "NormalInitializer":
        config = _validate_mapping(config, {"type", "mean", "std"}, set(), description)
        mean = _finite_number(config["mean"], f"{description}.mean")
        std = _finite_number(config["std"], f"{description}.std")
        if std < 0:
            raise ValueError(f"{description}.std must not be negative.")
        return cls(mean, std)

    def __call__(self) -> float:
        return float(np.random.normal(self.mean, self.std))


@dataclass(frozen=True)
class ConstantInitializer:
    value: float

    @classmethod
    def from_config(cls, config: Any, description: str) -> "ConstantInitializer":
        config = _validate_mapping(config, {"type", "value"}, set(), description)
        return cls(_finite_number(config["value"], f"{description}.value"))

    def __call__(self) -> float:
        return self.value



def _apply_factory(factory: dict[str, Callable[[Any, str], Any]], config: Any, description: str):
    if not isinstance(config, dict):
        raise ValueError(f"{description} must be a JSON object.")
    kind = config.get("type")
    if not isinstance(kind, str):
        raise ValueError(f"{description}.type must be a string.")

    kind = kind.lower()
    factory_func = factory.get(kind)
    if factory_func is None:
        names = list(factory)
        supported = ", ".join(repr(name) for name in names[:-1]) + f", or {names[-1]!r}"
        raise ValueError(
            f"Unknown {description}.type {kind!r}; expected {supported}."
        )
    return factory_func(config, description)


_ACTIVATIONS: dict[str, type[ActivationFunction]] = {
    "sigmoid": Sigmoid,
    "heaviside": Heaviside,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "linear": Linear,
    "tanh": Tanh,
    "softplus": Softplus,
    "softmax": Softmax,
}


_Initializer = UniformInitializer | NormalInitializer | ConstantInitializer
_InitializerFactory = Callable[[Any, str], _Initializer]

_INITIALIZERS: dict[str, _InitializerFactory] = {
    "uniform": UniformInitializer.from_config,
    "normal": NormalInitializer.from_config,
    "constant": ConstantInitializer.from_config,
}


def _initializer(config: Any, description: str):
    return _apply_factory(_INITIALIZERS, config, description)


def _layer_settings(config: Any, description: str) -> LayerSettings:
    config = _validate_mapping(
        config,
        {"neurons", "activation_function", "weight_initializer", "bias_initializer"},
        set(),
        description,
    )
    neurons = _positive_integer(config["neurons"], f"{description}.neurons")
    activation_name = config["activation_function"]
    if not isinstance(activation_name, str):
        raise ValueError(f"{description}.activation_function must be a string.")

    activation = _ACTIVATIONS.get(activation_name.lower().replace("_", ""))
    if activation is None:
        supported = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(
            f"Unknown {description}.activation_function {activation_name!r}; "
            f"supported activations are: {supported}."
        )

    return LayerSettings(
        neurons=neurons,
        default_weight=_initializer(
            config["weight_initializer"], f"{description}.weight_initializer"
        ),
        default_bias=_initializer(
            config["bias_initializer"], f"{description}.bias_initializer"
        ),
        activation_function=activation,
    )


_SchedulerFactory = Callable[[Any, str], LRScheduler]

_SCHEDULERS: dict[str, _SchedulerFactory] = {
    "constant": ConstantLR.from_config,
    "exponential": ExponentialLR.from_config,
    "step": StepLR.from_config,
}


def _learning_rate_scheduler(config: Any) -> LRScheduler:
    description = "lr_scheduler"
    return _apply_factory(_SCHEDULERS, config, description)


def _settings(config: Any) -> NNSettings:
    config = _validate_mapping(
        config,
        {"input_count", "hidden_layers", "output_layer"},
        {"lr_scheduler"},
        "network configuration",
    )
    input_count = _positive_integer(config["input_count"], "input_count")
    hidden_layers_config = config["hidden_layers"]
    if not isinstance(hidden_layers_config, list):
        raise ValueError("hidden_layers must be a JSON array.")

    hidden_layers = [
        _layer_settings(layer, f"hidden_layers[{index}]")
        for index, layer in enumerate(hidden_layers_config)
    ]
    output_layer = _layer_settings(config["output_layer"], "output_layer")
    settings = NNSettings(
        input_count=input_count,
        hidden_layers=hidden_layers,
        output_layer=output_layer,
    )
    if "lr_scheduler" in config:
        settings.lr_scheduler = _learning_rate_scheduler(config["lr_scheduler"])
    return settings


def generate_neural_network(config_file: str | Path) -> NeuralNetwork:
    """Load a JSON network configuration and create a fresh neural network.

    Initializer objects are module-level callables so the resulting network
    remains compatible with :meth:`NeuralNetwork.save` and ``pickle``.
    """
    config_path = Path(config_file)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {config_path} at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error

    return NeuralNetwork(_settings(config))