from typing import Any, TypeVar
from collections.abc import Callable

T = TypeVar("T")
V = TypeVar("V")


def map_leaves(
    fun: Callable[..., V], arg: T | list[Any] | dict[str, Any]
) -> V | list[Any] | dict[str, Any]:
    if isinstance(arg, dict):
        return {k: map_leaves(fun, v) for k, v in arg.items()}
    elif isinstance(arg, list):
        return [map_leaves(fun, it) for it in arg]
    else:
        return fun(arg)
