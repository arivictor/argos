from abc import ABC, abstractmethod
from typing import Any

from aroflow.domain.entity import Step, WorkflowDSL
from aroflow.domain.port import PluginBase


class WorkflowEngine(ABC):
    """Abstract base class defining the workflow engine interface."""

    @abstractmethod
    def run(self, workflow: WorkflowDSL, workflow_id: str | None = None) -> Any:
        """Runs the given workflow."""
        ...


class TaskRunner(ABC):
    @abstractmethod
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any: ...


class StepExecutor(ABC):
    """Abstract executor interface for executing workflow steps and returning results."""

    @abstractmethod
    def execute(self, step: Any) -> Any:
        """Execute a workflow step and return its result."""
        ...


class ExecutorFactory(ABC):
    @abstractmethod
    def get_executor(self, step: Step) -> StepExecutor: ...


class PluginResolver(ABC):
    """Abstract base class defining plugin resolution interface."""

    @abstractmethod
    def resolve(self, name: str) -> PluginBase:
        """Resolves and returns a plugin instance by its name."""
        ...


class ResultStore(ABC):
    @abstractmethod
    def set(self, key: str, value: Any): ...
    @abstractmethod
    def get(self, key: str) -> Any: ...


class PlaceholderResolver(ABC):
    @abstractmethod
    def resolve_any(self, value: Any) -> Any: ...


class Binder(ABC):
    @abstractmethod
    def bind(self, plugin: PluginBase, params: dict[str, Any]) -> dict[str, Any]: ...


class Context(ABC):
    @abstractmethod
    def get_result(self, step_id: str) -> Any: ...
