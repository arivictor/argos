from typing import Any

from aroflow.application.adapter import TaskRunner
from aroflow.domain.port import PluginBase


class InMemoryTaskRunner(TaskRunner):
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        """
        Execute a plugin with bound parameters.

        :param plugin: The plugin instance to execute
        :type plugin: PluginBase
        :param bound: Dictionary of bound parameters
        :type bound: dict[str, Any]
        :returns: The result of plugin execution
        :rtype: Any
        """
        return plugin.execute(**bound)
