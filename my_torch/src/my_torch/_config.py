import math
from typing import Any


def validate_mapping(
    value: Any,
    required_keys: set[str],
    optional_keys: set[str],
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object.")

    missing = required_keys - value.keys()
    unknown = value.keys() - required_keys - optional_keys
    if missing:
        raise ValueError(f"{description} is missing required fields: {', '.join(sorted(missing))}.")
    if unknown:
        raise ValueError(f"{description} has unknown fields: {', '.join(sorted(unknown))}.")
    return value


def get_positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer.")
    return value


def get_finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{description} must be a finite number.")
    return number
