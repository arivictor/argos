from typing import TYPE_CHECKING

from argos.application.port import PluginResolver, WorkflowEngine
from argos.application.service import load_workflow
from argos.domain.entity import WorkflowResult
from argos.domain.port import PluginBase

if TYPE_CHECKING:
    pass


class Client:
    """
    Unified client façade for workflow execution.

    The Client is the only thing users interact with. It exposes methods like .plugin(),
    .run(), and .load_plugins(). It holds a reference to the chosen backend implementation
    under the hood.
    """

    def __init__(self, backend: WorkflowEngine, plugin_resolver: PluginResolver, executor_factory=None):
        """
        Initialize the client with a backend engine and plugin resolver.

        Args:
            backend: The workflow engine implementation (e.g., InMemoryWorkflowEngine)
            plugin_resolver: The plugin resolver for registering and resolving plugins
            executor_factory: The executor factory that actually uses the plugin resolver
        """
        self._engine = backend
        self._resolver = plugin_resolver
        self._executor_factory = executor_factory

    def plugin(self, plugin: type[PluginBase]) -> "Client":
        """
        Register a plugin with the client.

        Args:
            plugin: The plugin class to register

        Returns:
            The client instance for method chaining
        """
        # Register with the executor factory's resolver (the one that actually matters)
        if (
            self._executor_factory
            and hasattr(self._executor_factory, "resolver")
            and hasattr(self._executor_factory.resolver, "_registry")
        ):
            plugin_name = getattr(plugin, "plugin_name", plugin.__name__)
            self._executor_factory.resolver._registry[plugin_name] = plugin

        # Also register with our own resolver for consistency
        if hasattr(self._resolver, "_registry"):
            plugin_name = getattr(plugin, "plugin_name", plugin.__name__)
            self._resolver._registry[plugin_name] = plugin
        return self

    def run(self, workflow_dict: dict) -> WorkflowResult:
        """
        Execute a workflow.

        Args:
            workflow_dict: The workflow definition as a dictionary
            workflow_id: Optional workflow identifier

        Returns:
            The workflow execution result
        """
        workflow = load_workflow(workflow_dict)
        return self._engine.run(workflow)

    def load_plugins(self) -> list[type[PluginBase]]:
        """
        Get all registered plugins.

        Returns:
            List of registered plugin classes
        """
        return PluginBase._plugins

    @staticmethod
    def get_available_plugins() -> list[type[PluginBase]]:
        """
        Get all available plugin classes (static method).

        Returns:
            List of available plugin classes
        """
        return PluginBase._plugins
