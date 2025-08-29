"""Executor factory for creating appropriate executors for different step types."""

from abc import ABC, abstractmethod

from ...domain.models.execution_options import ExecutionOptions
from ...domain.models.steps import Step, OperationStep, MapStep, ParallelStep
from ...domain.services.plugin_resolution import PluginResolver
from ...domain.services.parameter_binding import ParameterBinder
from ...domain.services.placeholder_resolution import PlaceholderResolver
from .task_runner import TaskRunner, LocalTaskRunner
from .step_executors import StepExecutor, OperationExecutor, ParallelOperationExecutor
from .map_strategies import MapExecutor


class ExecutorFactory(ABC):
    """Abstract factory for creating step executors."""
    
    @abstractmethod
    def get_executor(self, step: Step) -> StepExecutor:
        """Get an appropriate executor for the given step."""
        ...


class InMemoryExecutorFactory(ExecutorFactory):
    """Factory that creates executors for in-memory execution."""
    
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
        
    def get_executor(self, step: Step) -> StepExecutor:
        if isinstance(step, OperationStep):
            return OperationExecutor(self.resolver, self.binder, self.values, self.task_runner, self.execution_options)
        elif isinstance(step, MapStep):
            return MapExecutor(self.resolver, self.binder, self.values, self.task_runner, self.execution_options)
        elif isinstance(step, ParallelStep):
            return ParallelOperationExecutor(self.resolver, self.binder, self.values, self.task_runner, self.execution_options)
        else:
            raise ValueError(f"Unknown step type: {type(step)}")