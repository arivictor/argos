from argos.domain.port import PluginBase
from plugins import NumberAdderPlugin, SayHelloPlugin, SleepyPlugin, ThrowExceptionPlugin  # noqa: F401


def get_plugins() -> list[type[PluginBase]]:
    """Returns a list of all registered plugin classes."""
    return PluginBase._plugins
