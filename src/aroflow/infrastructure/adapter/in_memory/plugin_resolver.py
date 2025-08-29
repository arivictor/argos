from aroflow.application.adapter import PluginResolver
from aroflow.domain.port import PluginBase


class InMemoryPluginResolver(PluginResolver):
    """Resolves plugins from an in-memory registry."""

    def __init__(self, plugins: list[type[PluginBase]]):
        """
        Initializes resolver with optional plugin list.

        :param plugins: List of plugin classes to register
        :type plugins: list[type[PluginBase]]
        """
        self._registry: dict[str, type[PluginBase]] = {}
        for cls in plugins:
            key = getattr(cls, "plugin_name", cls.__name__)
            self._registry[key] = cls

    def resolve(self, name: str) -> PluginBase:
        """
        Returns a plugin instance matching the given name.

        :param name: The name of the plugin to resolve
        :type name: str
        :returns: Plugin instance matching the given name
        :rtype: PluginBase
        :raises KeyError: If no plugin is registered for the given name
        """
        try:
            cls = self._registry[name]
        except KeyError:
            raise KeyError(f"No plugin registered for operation '{name}'") from None
        return cls()
