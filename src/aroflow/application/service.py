from typing import Any

import msgspec

from aroflow.application.adapter import VariableResolver
from aroflow.application.port import (
    Binder,
    Context,
    PluginResolver,
    ResultStore,
    WorkflowEngine,
)
from aroflow.domain.entity import MapResult, OperationResult, ParallelResult, WorkflowDSL, WorkflowResult
from aroflow.domain.service import validate_workflow
from aroflow.domain.value_object import ExecutionOptions


class ResultRegistrar:
    """Handles registration of results into a ResultStore."""

    def __init__(self, result_store: ResultStore):
        self.result_store = result_store

    def register(self, step_result: Any):
        """
        Register results into the store, including nested results.

        :param step_result: The result from executing a step
        :type step_result: Any
        """
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
    """
    Decodes and validates a workflow from a Python dictionary.

    :param data: The workflow data as a dictionary
    :type data: dict | WorkflowDSL
    :returns: A WorkflowDSL instance representing the validated workflow
    :rtype: WorkflowDSL
    """
    if isinstance(data, WorkflowDSL):
        validate_workflow(data)
        return data

    workflow = msgspec.convert(data, type=WorkflowDSL)
    validate_workflow(workflow)
    return workflow


def execute_workflow(workflow: WorkflowDSL, engine: WorkflowEngine) -> WorkflowResult:
    """
    Runs the given workflow using the specified workflow engine and returns a WorkflowResult.

    :param workflow: The workflow to execute
    :type workflow: WorkflowDSL
    :param engine: The workflow engine to use for execution
    :type engine: WorkflowEngine
    :returns: The result of the workflow execution
    :rtype: WorkflowResult
    """
    return engine.run(workflow)


class WorkflowClient:
    """
    Encapsulates setup of plugin resolver, binder, result store, context,
    values, executor factory, registrar, and engine.
    Provides a high-level interface for loading and running workflows.

    .. note::
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

        :param workflow: The workflow definition as a dictionary or WorkflowDSL instance
        :type workflow: dict | WorkflowDSL
        :param workflow_id: Optional workflow identifier
        :type workflow_id: str | None
        :returns: A WorkflowResult
        :rtype: WorkflowResult
        """
        workflow = load_workflow(workflow)
        result = self.engine.run(workflow, workflow_id)
        return result

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        """
        Retrieve all results for a specific workflow by ID.

        :param workflow_id: The workflow identifier
        :type workflow_id: str
        :returns: Dictionary mapping step_id to result values
        :rtype: dict[str, Any]
        :raises KeyError: If no results are found for the workflow
        """
        if hasattr(self.result_store, "get_workflow_results"):
            return self.result_store.get_workflow_results(workflow_id)
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
        if hasattr(self.result_store, "delete_workflow_results"):
            return self.result_store.delete_workflow_results(workflow_id)
        else:
            raise NotImplementedError("Backend does not support workflow deletion")

    def list_workflows(self) -> list[str]:
        """
        Get a list of all workflow IDs that have stored results.

        :returns: List of workflow identifiers
        :rtype: list[str]
        """
        if hasattr(self.result_store, "list_workflow_ids"):
            return self.result_store.list_workflow_ids()
        else:
            raise NotImplementedError("Backend does not support workflow listing")
