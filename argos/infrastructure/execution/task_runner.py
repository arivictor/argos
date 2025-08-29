"""Task runner infrastructure for executing plugin operations."""

from abc import ABC, abstractmethod
from typing import Any

from ...domain.plugins.base import PluginBase


class TaskRunner(ABC):
    """Abstract interface for running plugin tasks."""
    
    @abstractmethod
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        """Execute a plugin with bound parameters."""
        ...


class LocalTaskRunner(TaskRunner):
    """Task runner that executes plugins locally in the current process."""
    
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        return plugin.execute(**bound)