"""Workflow execution engine."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..domain.models.steps import WorkflowDSL
from ..domain.models.results import WorkflowResult
from ..infrastructure.storage.result_store import ResultStore, InMemoryResultStore, ExecutionContext, ResultRegistrar
from ..infrastructure.execution.executor_factory import ExecutorFactory
from ..domain.services.placeholder_resolution import PlaceholderResolver


class WorkflowEngine(ABC):
    """Abstract base class defining the workflow engine interface."""

    @abstractmethod
    def run(self, workflow: WorkflowDSL) -> Any:
        """Runs the given workflow."""
        ...


class InMemoryWorkflowEngine(WorkflowEngine):
    """Workflow engine that executes steps in memory using executors."""
    
    def __init__(
        self,
        executor_factory: ExecutorFactory,
        result_store: ResultStore | None = None,
        registrar: ResultRegistrar | None = None,
    ):
        """Initializes with an executor factory and optional result store and registrar."""
        self.result_store = result_store if result_store is not None else InMemoryResultStore()
        self.registrar = registrar if registrar is not None else ResultRegistrar(self.result_store)
        self.ctx = ExecutionContext(self.result_store)
        self.values = PlaceholderResolver(self.ctx)
        self.executor_factory = executor_factory
        
    def run(self, workflow: WorkflowDSL, workflow_id: Optional[str] = None) -> WorkflowResult:
        """Executes each step of the workflow in order and returns a WorkflowResult."""
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
                if hasattr(step_result, "status"):
                    if getattr(step_result, "status", None) == "failed":
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
                        if hasattr(item, "status") and getattr(item, "status", None) == "failed":
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
            status = "failed"
        elif any_nonfatal_failed:
            status = "partial"
        else:
            status = "success"
        if workflow_id is None:
            workflow_id = "workflow"
        return WorkflowResult(
            id=workflow_id,
            status=status,
            results=results,
            error=error_msg,
        )


def execute_workflow(workflow: WorkflowDSL, engine: WorkflowEngine, workflow_id: Optional[str] = None) -> WorkflowResult:
    """Runs the given workflow using the specified workflow engine and returns a WorkflowResult."""
    return engine.run(workflow)