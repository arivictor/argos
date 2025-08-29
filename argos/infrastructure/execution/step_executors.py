"""Step executors for different types of workflow steps."""

import time
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any
from msgspec import structs

from ...domain.models.execution_options import ExecutionOptions
from ...domain.models.steps import Step, OperationStep, ParallelStep
from ...domain.models.results import OperationResult, ParallelOpResult, ParallelResult
from ...domain.services.plugin_resolution import PluginResolver
from ...domain.services.parameter_binding import ParameterBinder
from ...domain.services.placeholder_resolution import PlaceholderResolver
from .task_runner import TaskRunner, LocalTaskRunner


class StepExecutor(ABC):
    """Abstract executor interface for executing workflow steps and returning results."""
    
    @abstractmethod
    def execute(self, step: Any) -> Any:
        """Execute a workflow step and return its result."""
        ...


class OperationExecutor(StepExecutor):
    """Executes an operation step by resolving and running the corresponding plugin."""
    
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
        execution_options: ExecutionOptions | None = None,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
        self.execution_options = execution_options if execution_options is not None else ExecutionOptions()

    def execute(self, step: OperationStep):
        """Executes the operation step and returns a structured result."""
        # Resolve placeholders in parameters
        resolved_params = self.values.resolve_any(step.parameters)
        step = structs.replace(step, parameters=resolved_params)
        plugin = self.resolver.resolve(step.operation)
        bound = self.binder.bind(plugin, step.parameters)
        result = None
        last_exc = None
        # Step-level retries/timeout override global options
        retries = step.retries if hasattr(step, "retries") and step.retries is not None else self.execution_options.retries
        timeout = step.timeout if hasattr(step, "timeout") and step.timeout is not None else self.execution_options.timeout
        for attempt in range(retries + 1):
            try:
                if timeout is not None:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self.task_runner.run, plugin, bound)
                        result = future.result(timeout=timeout)
                else:
                    result = self.task_runner.run(plugin, bound)
                # On success
                return OperationResult(
                    id=step.id,
                    kind="operation",
                    operation=step.operation,
                    parameters=step.parameters,
                    result=result,
                    status="success",
                    error=None,
                )
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    # Exponential backoff: 0.1, 0.2, 0.4, ...
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    if getattr(step, "fail_workflow", True):
                        raise
                    # else: capture the error and return as result
                    return OperationResult(
                        id=step.id,
                        kind="operation",
                        operation=step.operation,
                        parameters=step.parameters,
                        result=None,
                        status="failed",
                        error=str(e),
                    )


class ParallelOperationExecutor(StepExecutor):
    """Executes multiple operation steps in parallel using threads."""
    
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
        execution_options: ExecutionOptions | None = None,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
        self.execution_options = execution_options if execution_options is not None else ExecutionOptions()
        
    def execute(self, step: ParallelStep):
        """Executes all operation steps in parallel and returns a structured result."""
        # Use step-level retries/timeout for all inner operations, unless operation itself overrides (not handled here for simplicity).
        retries = step.retries if hasattr(step, "retries") and step.retries is not None else self.execution_options.retries
        timeout = step.timeout if hasattr(step, "timeout") and step.timeout is not None else self.execution_options.timeout
        
        def run_op(op: OperationStep):
            op_retries = op.retries if hasattr(op, "retries") and op.retries is not None else retries
            op_timeout = op.timeout if hasattr(op, "timeout") and op.timeout is not None else timeout
            op_with_opts = structs.replace(op, retries=op_retries, timeout=op_timeout)
            try:
                res = OperationExecutor(
                    self.resolver,
                    self.binder,
                    self.values,
                    self.task_runner,
                    self.execution_options
                ).execute(op_with_opts)
                return ParallelOpResult(
                    id=op.id,
                    operation=op.operation,
                    parameters=op.parameters,
                    result=res.result,
                    status="success",
                    error=None,
                )
            except Exception as e:
                # Decide fail_workflow logic: op.fail_workflow or step.fail_workflow
                fail_workflow = getattr(op, "fail_workflow", getattr(step, "fail_workflow", True))
                if fail_workflow:
                    raise
                # else: capture the error in ParallelOpResult
                return ParallelOpResult(
                    id=op.id,
                    operation=op.operation,
                    parameters=op.parameters,
                    result=None,
                    status="failed",
                    error=str(e),
                )
                
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_op, op): op for op in step.operations}
            results = []
            for future in concurrent.futures.as_completed(futures):
                op = futures[future]
                try:
                    result = future.result(timeout=timeout)
                except Exception as e:
                    raise
                results.append(result)
        # Keep order as in step.operations
        results.sort(key=lambda r: [o.id for o in step.operations].index(r.id))
        return ParallelResult(id=step.id, kind="parallel", results=results)