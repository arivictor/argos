import uuid
from typing import Any

from aroflow.application.adapter import ExecutionContext, VariableResolver
from aroflow.application.port import ExecutorFactory, ResultStore, WorkflowEngine
from aroflow.application.service import ResultRegistrar
from aroflow.domain.entity import WorkflowDSL, WorkflowResult
from aroflow.domain.value_object import ResultStatus, WorkflowResultStatus
from aroflow.infrastructure.adapter.sqlite.result_store import SQLiteResultStore


class UUIDGenerator:
    """Generates unique identifiers using UUID."""

    def generate(self) -> str:
        """
        Generate a unique identifier.

        :returns: A unique identifier string
        :rtype: str
        """
        return uuid.uuid4().hex


class SQLiteWorkflowEngine(WorkflowEngine):
    """Workflow engine that executes steps and stores results in SQLite."""

    def __init__(
        self,
        executor_factory: ExecutorFactory,
        result_store: ResultStore,
        registrar: ResultRegistrar,
    ):
        """
        Initializes with an executor factory and optional result store and registrar.

        :param executor_factory: Factory for creating step executors
        :type executor_factory: ExecutorFactory
        :param result_store: Store for workflow results
        :type result_store: ResultStore
        :param registrar: Registrar for storing step results
        :type registrar: ResultRegistrar
        """
        self.result_store = result_store if result_store is not None else SQLiteResultStore()
        self.registrar = registrar if registrar is not None else ResultRegistrar(self.result_store)
        self.ctx = ExecutionContext(self.result_store)
        self.values = VariableResolver(self.ctx)
        self.executor_factory = executor_factory

    def run(self, workflow: WorkflowDSL, workflow_id: str | None = None) -> WorkflowResult:
        """
        Run a workflow using the SQLite backend.

        :param workflow: The workflow to execute
        :type workflow: WorkflowDSL
        :param workflow_id: Optional workflow identifier
        :type workflow_id: str | None
        :returns: The result of executing the workflow
        :rtype: WorkflowResult
        """
        if workflow_id is None:
            workflow_id = UUIDGenerator().generate()

        results = []
        error_msg = None
        any_failed = False
        any_nonfatal_failed = False

        for step in workflow.steps:
            try:
                # Get the executor for this step
                executor = self.executor_factory.get_executor(step)

                # Execute the step
                step_result = executor.execute(step)

                # Register the result with workflow_id
                self._register_workflow_result(workflow_id, step_result)

                # Check for failures
                failed = False
                nonfatal_failed = False

                if hasattr(step_result, "status") and getattr(step_result, "status", None) == ResultStatus.FAILED:
                    # Try to check fail_workflow
                    fail_workflow = getattr(step, "fail_workflow", True)
                    if fail_workflow:
                        failed = True
                    else:
                        nonfatal_failed = True

                # For MapResult or ParallelResult, check contained item results
                if hasattr(step_result, "results"):
                    contained = getattr(step_result, "results", [])
                    for item in contained:
                        if hasattr(item, "status") and getattr(item, "status", None) == ResultStatus.FAILED:
                            fail_workflow = getattr(step, "fail_workflow", True)
                            if fail_workflow:
                                failed = True
                            else:
                                nonfatal_failed = True

                if failed:
                    any_failed = True
                if nonfatal_failed:
                    any_nonfatal_failed = True

                results.append(step_result)
            except Exception as e:
                # Fatal failure
                any_failed = True
                error_msg = str(e)
                break

        # Determine workflow status
        if any_failed:
            status = WorkflowResultStatus.FAILED
        elif any_nonfatal_failed:
            status = WorkflowResultStatus.PARTIAL_FAILURE
        else:
            status = WorkflowResultStatus.SUCCESS

        return WorkflowResult(
            id=workflow_id,
            status=status,
            results=results,
            error=error_msg,
        )

    def _register_workflow_result(self, workflow_id: str, step_result: Any):
        """
        Register a step result with the workflow ID context.

        :param workflow_id: The workflow identifier
        :type workflow_id: str
        :param step_result: The result from executing a step
        :type step_result: Any
        """
        from aroflow.domain.entity import MapResult, OperationResult, ParallelResult

        # Store the main result under workflow_id.step_id
        if hasattr(step_result, "id"):
            key = f"{workflow_id}.{step_result.id}"
            self.result_store.set(key, step_result)
        
        # Register nested operation results
        if isinstance(step_result, MapResult):
            nested_id = step_result.id
            # Store the entire list of results under the nested operation id
            key = f"{workflow_id}.{nested_id}"
            self.result_store.set(key, step_result.results)
            # Additionally, register each MapItemResult by its own id
            for item_result in step_result.results:
                key = f"{workflow_id}.{item_result.id}"
                self.result_store.set(key, item_result)
        elif isinstance(step_result, ParallelResult):
            for opres in step_result.results:
                key = f"{workflow_id}.{opres.id}"
                self.result_store.set(key, opres)
        elif isinstance(step_result, OperationResult):
            key = f"{workflow_id}.{step_result.id}"
            self.result_store.set(key, step_result)
