from typing import TYPE_CHECKING, Any

from aroflow.application.port import PluginResolver, WorkflowEngine
from aroflow.application.service import load_workflow
from aroflow.domain.entity import WorkflowResult
from aroflow.domain.port import PluginBase

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

        :param backend: The workflow engine implementation (e.g., InMemoryWorkflowEngine)
        :type backend: WorkflowEngine
        :param plugin_resolver: The plugin resolver for registering and resolving plugins
        :type plugin_resolver: PluginResolver
        :param executor_factory: The executor factory that actually uses the plugin resolver
        """
        self._engine = backend
        self._resolver = plugin_resolver
        self._executor_factory = executor_factory

    def plugin(self, plugin: type[PluginBase]) -> "Client":
        if self._executor_factory and hasattr(self._executor_factory, "resolver"):
            resolver = self._executor_factory.resolver
            plugin_name = getattr(plugin, "plugin_name", plugin.__name__)
            registry = getattr(resolver, "_registry", None)

            if isinstance(registry, dict):
                registry[plugin_name] = plugin
            else:
                # If no dict registry, just set it up
                resolver._registry = {plugin_name: plugin}
        return self

    def run(self, workflow_dict: dict, workflow_id: str | None = None) -> WorkflowResult:
        """
        Execute a workflow.

        :param workflow_dict: The workflow definition as a dictionary
        :type workflow_dict: dict
        :param workflow_id: Optional workflow identifier
        :type workflow_id: str | None
        :returns: The workflow execution result
        :rtype: WorkflowResult
        """
        workflow = load_workflow(workflow_dict)
        return self._engine.run(workflow, workflow_id)

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        """
        Retrieve all results for a specific workflow by ID.

        :param workflow_id: The workflow identifier
        :type workflow_id: str
        :returns: Dictionary mapping step_id to result values
        :rtype: dict[str, Any]
        :raises KeyError: If no results are found for the workflow
        """
        if hasattr(self._engine, "result_store") and hasattr(self._engine.result_store, "get_workflow_results"):
            return self._engine.result_store.get_workflow_results(workflow_id)
        else:
            raise NotImplementedError("Backend does not support workflow querying")

    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete all results for a specific workflow by ID.

        :param workflow_id: The workflow identifier
        :type workflow_id: str
        :returns: True if any results were deleted, False otherwise
        :rtype: bool
        """
        if hasattr(self._engine, "result_store") and hasattr(self._engine.result_store, "delete_workflow_results"):
            return self._engine.result_store.delete_workflow_results(workflow_id)
        else:
            raise NotImplementedError("Backend does not support workflow deletion")

    def list_workflows(self) -> list[str]:
        """
        Get a list of all workflow IDs that have stored results.

        :returns: List of workflow identifiers
        :rtype: list[str]
        """
        if hasattr(self._engine, "result_store") and hasattr(self._engine.result_store, "list_workflow_ids"):
            return self._engine.result_store.list_workflow_ids()
        else:
            raise NotImplementedError("Backend does not support workflow listing")

    def load_plugins(self) -> list[type[PluginBase]]:
        """
        Get all registered plugins.

        :returns: List of registered plugin classes
        :rtype: list[type[PluginBase]]
        """
        return PluginBase._plugins

    @staticmethod
    def get_available_plugins() -> list[type[PluginBase]]:
        """
        Get all available plugin classes (static method).

        :returns: List of available plugin classes
        :rtype: list[type[PluginBase]]
        """
        return PluginBase._plugins
