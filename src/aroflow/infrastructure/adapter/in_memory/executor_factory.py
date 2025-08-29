from aroflow.application.adapter import (
    MapExecutor,
    OperationExecutor,
    ParallelOperationExecutor,
    ParameterBinder,
    PlaceholderResolver,
)
from aroflow.application.port import ExecutorFactory, PluginResolver, StepExecutor, TaskRunner
from aroflow.domain.entity import MapStep, OperationStep, ParallelStep, Step
from aroflow.domain.value_object import ExecutionOptions


class InMemoryExecutorFactory(ExecutorFactory):
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner,
        execution_options: ExecutionOptions,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = execution_options

    def get_executor(self, step: Step) -> StepExecutor:
        """
        Get the appropriate executor for the given step type.

        :param step: The step to get an executor for
        :type step: Step
        :returns: The executor for the given step type
        :rtype: StepExecutor
        :raises ValueError: If the step type is unknown
        """
        if isinstance(step, OperationStep):
            return OperationExecutor(self.resolver, self.binder, self.values, self.task_runner, self.execution_options)
        elif isinstance(step, MapStep):
            return MapExecutor(self.resolver, self.binder, self.values, self.task_runner, self.execution_options)
        elif isinstance(step, ParallelStep):
            return ParallelOperationExecutor(
                self.resolver, self.binder, self.values, self.task_runner, self.execution_options
            )
        else:
            raise ValueError(f"Unknown step type: {type(step)}")
