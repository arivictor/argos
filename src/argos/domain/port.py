from typing import Any


class PluginBase:
    """Base class for all plugins. Enforces 'execute' method and registers subclasses."""

    _plugins = []

    def __init_subclass__(cls, **kwargs):
        """Registers subclass and ensures 'execute' method is defined."""
        super().__init_subclass__(**kwargs)

        if "execute" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a 'execute' method")

        PluginBase._plugins.append(cls)

    def execute(self, *args, **kwargs) -> Any:
        """Abstract execute method to be implemented by plugins."""
        raise NotImplementedError("Plugins must implement the execute method")
