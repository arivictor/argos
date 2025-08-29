from typing import Any

from argos.application.adapter import TaskRunner
from argos.domain.port import PluginBase


class InMemoryTaskRunner(TaskRunner):
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        return plugin.execute(**bound)
