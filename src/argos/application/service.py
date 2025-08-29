from typing import Any

import msgspec

from argos.application.adapter import VariableResolver
from argos.application.port import (
    Binder,
    Context,
    PluginResolver,
    ResultStore,
    WorkflowEngine,
)
from argos.domain.entity import MapResult, OperationResult, ParallelResult, WorkflowDSL, WorkflowResult
from argos.domain.service import validate_workflow
from argos.domain.value_object import ExecutionOptions


class ResultRegistrar:
    """Handles registration of results into a ResultStore."""

    def __init__(self, result_store: ResultStore):
        self.result_store = result_store

    def register(self, step_result: Any):
        """Register results into the store, including nested results."""
        # Store the main result under its id
        if hasattr(step_result, "id"):
            self.result_store.set(step_result.id, step_result)
        # Register nested operation results so they are accessible by id
        if isinstance(step_result, MapResult):
            nested_id = step_result.id
            # Store the entire list of results under the nested operation id
            self.result_store.set(nested_id, step_result.results)
            # Additionally, register each MapItemResult by its own id
            for item_result in step_result.results:
                self.result_store.set(item_result.id, item_result)
        elif isinstance(step_result, ParallelResult):
            for opres in step_result.results:
                self.result_store.set(opres.id, opres)
        elif isinstance(step_result, OperationResult):
            self.result_store.set(step_result.id, step_result)


def load_workflow(data: dict | WorkflowDSL) -> WorkflowDSL:
    """Decodes and validates a workflow from a Python dictionary.

    Args:
        data: The workflow data as a dictionary.

    Returns:
        A WorkflowDSL instance representing the validated workflow.
    """
    if isinstance(data, WorkflowDSL):
        validate_workflow(data)
        return data

    workflow = msgspec.convert(data, type=WorkflowDSL)
    validate_workflow(workflow)
    return workflow


def execute_workflow(workflow: WorkflowDSL, engine: WorkflowEngine) -> WorkflowResult:
    """Runs the given workflow using the specified workflow engine and returns a WorkflowResult."""
    return engine.run(workflow)


class WorkflowClient:
    """
    Encapsulates setup of plugin resolver, binder, result store, context,
    values, executor factory, registrar, and engine.
    Provides a high-level interface for loading and running workflows.

    Note:
        Infrastructure wiring (e.g., concrete implementations of PluginResolver, ExecutorFactory,
        WorkflowEngine, and ResultStore) should be done in a composition root (e.g., container.py or main.py)
        and injected into this class.
    """

    def __init__(
        self,
        plugin_resolver: PluginResolver,
        executor_factory: Any,
        workflow_engine: WorkflowEngine,
        result_store: ResultStore,
        binder: Binder,
        exectuion_context: Context,
        execution_options: ExecutionOptions | None = None,
    ):
        self.resolver = plugin_resolver
        self.binder = binder
        self.result_store = result_store
        self.ctx = exectuion_context
        self.values = VariableResolver(self.ctx)
        self.execution_options = execution_options if execution_options is not None else ExecutionOptions()
        self.executor_factory = executor_factory
        self.registrar = ResultRegistrar(self.result_store)
        self.engine = workflow_engine

    def run(self, workflow: dict | WorkflowDSL, workflow_id: str | None = None) -> WorkflowResult:
        """
        Loads and executes a workflow from a dictionary.
        Returns a WorkflowResult.
        """
        workflow = load_workflow(workflow)
        result = self.engine.run(workflow)
        return result
