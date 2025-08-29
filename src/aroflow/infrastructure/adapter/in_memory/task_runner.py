from typing import Any

from aroflow.application.adapter import TaskRunner
from aroflow.domain.port import PluginBase


class InMemoryTaskRunner(TaskRunner):
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        return plugin.execute(**bound)
