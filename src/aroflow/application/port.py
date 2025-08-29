from abc import ABC, abstractmethod
from typing import Any

from aroflow.domain.entity import Step, WorkflowDSL
from aroflow.domain.port import PluginBase


class WorkflowEngine(ABC):
    """Abstract base class defining the workflow engine interface."""

    @abstractmethod
    def run(self, workflow: WorkflowDSL, workflow_id: str | None = None) -> Any:
        """
        Runs the given workflow.

        :param workflow: The workflow to execute
        :type workflow: WorkflowDSL
        :param workflow_id: Optional workflow identifier
        :type workflow_id: str | None
        :returns: The result of executing the workflow
        :rtype: Any
        """
        ...


class TaskRunner(ABC):
    """Abstract interface for running plugin tasks."""

    @abstractmethod
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        """
        Run a plugin with bound parameters.

        :param plugin: The plugin to run
        :type plugin: PluginBase
        :param bound: Dictionary of bound parameters
        :type bound: dict[str, Any]
        :returns: The result of running the plugin
        :rtype: Any
        """


class StepExecutor(ABC):
    """Abstract executor interface for executing workflow steps and returning results."""

    @abstractmethod
    def execute(self, step: Any) -> Any:
        """
        Execute a workflow step and return its result.

        :param step: The workflow step to execute
        :type step: Any
        :returns: The result of executing the step
        :rtype: Any
        """
        ...


class ExecutorFactory(ABC):
    """Abstract factory for creating step executors."""

    @abstractmethod
    def get_executor(self, step: Step) -> StepExecutor:
        """
        Get an executor for the given step.

        :param step: The step to get an executor for
        :type step: Step
        :returns: An executor capable of executing the step
        :rtype: StepExecutor
        """


class PluginResolver(ABC):
    """Abstract base class defining plugin resolution interface."""

    @abstractmethod
    def resolve(self, name: str) -> PluginBase:
        """
        Resolves and returns a plugin instance by its name.

        :param name: The name of the plugin to resolve
        :type name: str
        :returns: The resolved plugin instance
        :rtype: PluginBase
        :raises: KeyError if the plugin is not found
        """
        ...


class ResultStore(ABC):
    """Abstract interface for storing and retrieving workflow results."""

    @abstractmethod
    def set(self, key: str, value: Any):
        """
        Store a value with the given key.

        :param key: The key to store the value under
        :type key: str
        :param value: The value to store
        :type value: Any
        """

    @abstractmethod
    def get(self, key: str) -> Any:
        """
        Retrieve a value by key.

        :param key: The key to retrieve the value for
        :type key: str
        :returns: The stored value
        :rtype: Any
        :raises: KeyError if the key is not found
        """


class PlaceholderResolver(ABC):
    """Abstract interface for resolving placeholders in values."""

    @abstractmethod
    def resolve_any(self, value: Any) -> Any:
        """
        Resolve placeholders in any value type.

        :param value: The value that may contain placeholders
        :type value: Any
        :returns: The value with placeholders resolved
        :rtype: Any
        """


class Binder(ABC):
    """Abstract interface for binding parameters to plugin methods."""

    @abstractmethod
    def bind(self, plugin: PluginBase, params: dict[str, Any]) -> dict[str, Any]:
        """
        Bind parameters to a plugin's execute method.

        :param plugin: The plugin to bind parameters for
        :type plugin: PluginBase
        :param params: The parameters to bind
        :type params: dict[str, Any]
        :returns: Dictionary of bound parameters
        :rtype: dict[str, Any]
        """


class Context(ABC):
    """Abstract interface for workflow execution context."""

    @abstractmethod
    def get_result(self, step_id: str) -> Any:
        """
        Get the result of a previously executed step.

        :param step_id: The ID of the step whose result to retrieve
        :type step_id: str
        :returns: The result of the specified step
        :rtype: Any
        :raises: KeyError if the step result is not found
        """
