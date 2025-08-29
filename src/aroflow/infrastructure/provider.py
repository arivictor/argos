from aroflow.domain.port import PluginBase


def load_plugins() -> list[type[PluginBase]]:
    """Returns a list of all registered plugin classes."""
    return PluginBase._plugins
