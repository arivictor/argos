"""Parameter binding service for converting and binding parameters to plugin methods."""

import inspect
import typing
from typing import Any, get_origin, get_args, Optional, Union

from ..plugins.base import PluginBase


class ParameterBinder:
    """Binds parameters (accepting mixed types) to plugin execute method arguments with type coercion.

    The bind method accepts parameter dictionaries with mixed-type values, and will coerce strings to the target types when necessary.
    """

    def bind(self, plugin: PluginBase, params: dict[str, Any]) -> dict[str, Any]:
        """Binds and coerces parameters (accepting mixed-type values) to the plugin's execute method signature.

        Accepts a parameter dictionary with mixed-type values; will coerce strings to the target types when necessary.
        """
        sig = inspect.signature(plugin.execute)
        hints = typing.get_type_hints(plugin.execute, include_extras=False)
        bound: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name not in params:
                continue
            target = hints.get(name, Any)
            bound[name] = self._coerce(params[name], target)
        return bound

    def _coerce(self, value: Any, target_type: Any) -> Any:
        """Coerces a string value to the target type, handling Optional and Union types."""
        # If already the right type, return as-is
        if (
            target_type is Any or isinstance(value, target_type)
            if isinstance(target_type, type)
            else False
        ):
            return value
        # Unwrap Optional[T] and Union[T, ...]
        origin = get_origin(target_type)
        if origin is Optional:
            inner = get_args(target_type)[0]
            return self._coerce(value, inner)
        if origin is Union:
            for t in get_args(target_type):
                try:
                    return self._coerce(value, t)
                except Exception:
                    continue
            return value
        # Primitive coercions from string
        if isinstance(value, str):
            if target_type is int:
                return int(value)
            if target_type is float:
                return float(value)
            if target_type is bool:
                v = value.strip().lower()
                if v in {"true", "1", "yes", "y"}:
                    return True
                if v in {"false", "0", "no", "n"}:
                    return False
            # Leave as string for anything else
            return value
        return value