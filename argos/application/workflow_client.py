"""High-level workflow client for easy usage."""

from typing import Optional

from ..domain.models.execution_options import ExecutionOptions
from ..domain.models.results import WorkflowResult
from ..domain.plugins.base import PluginBase
from ..domain.services.plugin_resolution import InMemoryPluginResolver
from ..domain.services.parameter_binding import ParameterBinder
from ..domain.services.placeholder_resolution import PlaceholderResolver
from ..infrastructure.storage.result_store import InMemoryResultStore, ExecutionContext, ResultRegistrar
from ..infrastructure.execution.executor_factory import InMemoryExecutorFactory
from .workflow_engine import InMemoryWorkflowEngine
from .workflow_validation import load_workflow


class WorkflowClient:
    """
    Encapsulates setup of plugin resolver, binder, result store, context, values, executor factory, registrar, and engine.
    Provides a high-level interface for loading and running workflows.
    """
    
    def __init__(
        self,
        plugins: list[type[PluginBase]] | None = None,
        result_store=None,
        execution_options: ExecutionOptions | None = None,
    ):
        self.resolver = InMemoryPluginResolver(plugins)
        self.binder = ParameterBinder()
        self.result_store = result_store if result_store is not None else InMemoryResultStore()
        self.ctx = ExecutionContext(self.result_store)
        self.values = PlaceholderResolver(self.ctx)
        self.execution_options = execution_options if execution_options is not None else ExecutionOptions()
        self.executor_factory = InMemoryExecutorFactory(self.resolver, self.binder, self.values, execution_options=self.execution_options)
        self.registrar = ResultRegistrar(self.result_store)
        self.engine = InMemoryWorkflowEngine(self.executor_factory, self.result_store, self.registrar)

    def run(self, workflow_dict: dict, workflow_id: Optional[str] = None) -> WorkflowResult:
        """
        Loads and executes a workflow from a dictionary.
        Returns a WorkflowResult.
        """
        workflow = load_workflow(workflow_dict)
        result = self.engine.run(workflow, workflow_id=workflow_id)
        return result