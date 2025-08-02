from collections.abc import Callable
from typing import Any


def map_leaves[V](fun: Callable[..., V], arg: Any) -> V | list[Any] | dict[str, Any]:
    if isinstance(arg, dict):
        return {k: map_leaves(fun, v) for k, v in arg.items()}
    elif isinstance(arg, list):
        return [map_leaves(fun, it) for it in arg]
    else:
        return fun(arg)
