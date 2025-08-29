import uuid

from aroflow.application.adapter import ExecutionContext, VariableResolver
from aroflow.application.port import ExecutorFactory, ResultStore, WorkflowEngine
from aroflow.application.service import ResultRegistrar
from aroflow.domain.entity import WorkflowDSL, WorkflowResult
from aroflow.domain.value_object import ResultStatus, WorkflowResultStatus
from aroflow.infrastructure.adapter.in_memory.result_store import InMemoryResultStore


class UUIDGenerator:
    """Generates unique identifiers using UUID."""

    def generate(self) -> str:
        """
        Generate a unique identifier.

        :returns: A unique identifier string
        :rtype: str
        """
        return uuid.uuid4().hex


class InMemoryWorkflowEngine(WorkflowEngine):
    """Workflow engine that executes steps in memory using executors."""

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
        self.result_store = result_store if result_store is not None else InMemoryResultStore()
        self.registrar = registrar if registrar is not None else ResultRegistrar(self.result_store)
        self.ctx = ExecutionContext(self.result_store)
        self.values = VariableResolver(self.ctx)
        self.executor_factory = executor_factory

    def run(self, workflow: WorkflowDSL, workflow_id: str | None = None) -> WorkflowResult:
        """
        Executes each step of the workflow in order and returns a WorkflowResult.

        :param workflow: The workflow to execute
        :type workflow: WorkflowDSL
        :param workflow_id: Optional workflow identifier
        :type workflow_id: str | None
        :returns: The result of executing the workflow
        :rtype: WorkflowResult
        """
        results = []
        any_failed = False
        any_nonfatal_failed = False
        error_msg = None
        for step in workflow.steps:
            executor = self.executor_factory.get_executor(step)
            try:
                step_result = executor.execute(step)
                self.registrar.register(step_result)

                # Determine if this step failed
                # OperationResult, MapResult, ParallelResult
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
        if workflow_id is None:
            workflow_id = UUIDGenerator().generate()
        return WorkflowResult(
            id=workflow_id,
            status=status,
            results=results,
            error=error_msg,
        )
