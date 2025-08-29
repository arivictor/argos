from argos.application.adapter import PluginResolver
from argos.domain.port import PluginBase


class InMemoryPluginResolver(PluginResolver):
    """Resolves plugins from an in-memory registry."""

    def __init__(self, plugins: list[type[PluginBase]]):
        """Initializes resolver with optional plugin list."""
        self._registry: dict[str, type[PluginBase]] = {}
        for cls in plugins:
            key = getattr(cls, "plugin_name", cls.__name__)
            self._registry[key] = cls

    def resolve(self, name: str) -> PluginBase:
        """Returns a plugin instance matching the given name, raises KeyError if not found."""
        try:
            cls = self._registry[name]
        except KeyError:
            raise KeyError(f"No plugin registered for operation '{name}'") from None
        return cls()
