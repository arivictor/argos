"""Map execution strategies for sequential and parallel processing."""

import concurrent.futures
from abc import ABC, abstractmethod
from msgspec import structs

from ...domain.models.execution_options import ExecutionOptions
from ...domain.models.steps import MapStep
from ...domain.models.results import MapResult, MapItemResult
from ...domain.services.plugin_resolution import PluginResolver
from ...domain.services.parameter_binding import ParameterBinder
from ...domain.services.placeholder_resolution import PlaceholderResolver
from .task_runner import TaskRunner, LocalTaskRunner
from .step_executors import StepExecutor, OperationExecutor


class MapStrategy(ABC):
    """Abstract strategy for executing map steps."""
    
    @abstractmethod
    def execute(self, step: MapStep) -> MapResult:
        """Execute a map step and return the result."""
        ...


class SequentialMapStrategy(MapStrategy):
    """Executes map step items sequentially."""
    
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
        execution_options: ExecutionOptions | None = None,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
        self.execution_options = execution_options if execution_options is not None else ExecutionOptions()
        
    def execute(self, step: MapStep) -> MapResult:
        base_op = step.operation
        base_params = self.values.resolve_any(base_op.parameters)
        retries = step.retries if hasattr(step, "retries") and step.retries is not None else self.execution_options.retries
        timeout = step.timeout if hasattr(step, "timeout") and step.timeout is not None else self.execution_options.timeout
        results = []
        for idx, item in enumerate(step.inputs):
            new_params = {}
            for k, v in base_params.items():
                if v == "{{" + step.iterator + "}}":
                    new_params[k] = item
                else:
                    new_params[k] = v
            op_retries = base_op.retries if hasattr(base_op, "retries") and base_op.retries is not None else retries
            op_timeout = base_op.timeout if hasattr(base_op, "timeout") and base_op.timeout is not None else timeout
            op = structs.replace(base_op, parameters=new_params, retries=op_retries, timeout=op_timeout)
            try:
                res = OperationExecutor(
                    self.resolver,
                    self.binder,
                    self.values,
                    self.task_runner,
                    self.execution_options
                ).execute(op)
                result_val = res.result
                status = "success"
                error = None
            except Exception as e:
                fail_workflow = getattr(op, "fail_workflow", getattr(step, "fail_workflow", True))
                if fail_workflow:
                    raise
                result_val = None
                status = "failed"
                error = str(e)
            results.append(
                MapItemResult(
                    id=f"{op.id}_{idx}",
                    input=item,
                    operation=op.operation,
                    parameters=op.parameters,
                    result=result_val,
                    status=status,
                    error=error,
                )
            )
        return MapResult(
            id=step.id,
            kind="map",
            mode=step.mode,
            iterator=step.iterator,
            inputs=step.inputs,
            results=results,
        )


class ParallelMapStrategy(MapStrategy):
    """Executes map step items in parallel."""
    
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
        execution_options: ExecutionOptions | None = None,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
        self.execution_options = execution_options if execution_options is not None else ExecutionOptions()
        
    def execute(self, step: MapStep) -> MapResult:
        base_op = step.operation
        base_params = self.values.resolve_any(base_op.parameters)
        retries = step.retries if hasattr(step, "retries") and step.retries is not None else self.execution_options.retries
        timeout = step.timeout if hasattr(step, "timeout") and step.timeout is not None else self.execution_options.timeout
        
        def run_op(args):
            idx, item = args
            new_params = {}
            for k, v in base_params.items():
                if v == "{{" + step.iterator + "}}":
                    new_params[k] = item
                else:
                    new_params[k] = v
            op_retries = base_op.retries if hasattr(base_op, "retries") and base_op.retries is not None else retries
            op_timeout = base_op.timeout if hasattr(base_op, "timeout") and base_op.timeout is not None else timeout
            op = structs.replace(base_op, parameters=new_params, retries=op_retries, timeout=op_timeout)
            try:
                res = OperationExecutor(
                    self.resolver,
                    self.binder,
                    self.values,
                    self.task_runner,
                    self.execution_options
                ).execute(op)
                result_val = res.result
                status = "success"
                error = None
            except Exception as e:
                fail_workflow = getattr(op, "fail_workflow", getattr(step, "fail_workflow", True))
                if fail_workflow:
                    raise
                result_val = None
                status = "failed"
                error = str(e)
            return MapItemResult(
                id=f"{op.id}_{idx}",
                input=item,
                operation=op.operation,
                parameters=op.parameters,
                result=result_val,
                status=status,
                error=error,
            )
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_op, (idx, item)): idx for idx, item in enumerate(step.inputs)}
            results = []
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=timeout)
                except Exception as e:
                    raise
                results.append(result)
        results.sort(key=lambda r: int(r.id.split("_")[-1]))
        return MapResult(
            id=step.id,
            kind="map",
            mode=step.mode,
            iterator=step.iterator,
            inputs=step.inputs,
            results=results,
        )


class MapStrategyFactory:
    """Factory to return the correct MapStrategy based on mode."""
    
    @staticmethod
    def get_strategy(
        mode: str,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
        execution_options: ExecutionOptions | None = None,
    ) -> MapStrategy:
        if mode == "parallel":
            return ParallelMapStrategy(resolver, binder, values, task_runner, execution_options)
        else:
            return SequentialMapStrategy(resolver, binder, values, task_runner, execution_options)


class MapExecutor(StepExecutor):
    """Executes a map step using a strategy pattern."""
    
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
        execution_options: ExecutionOptions | None = None,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
        self.execution_options = execution_options if execution_options is not None else ExecutionOptions()
        # strategy is selected per execution
        
    def execute(self, step: MapStep):
        strategy = MapStrategyFactory.get_strategy(
            step.mode, self.resolver, self.binder, self.values, self.task_runner, self.execution_options
        )
        return strategy.execute(step)