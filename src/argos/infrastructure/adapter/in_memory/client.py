from argos.application.adapter import ExecutionContext, ParameterBinder, VariableResolver
from argos.application.service import ResultRegistrar, WorkflowClient
from argos.domain.port import PluginBase
from argos.domain.value_object import ExecutionOptions
from argos.infrastructure.adapter.in_memory.executor_factory import InMemoryExecutorFactory
from argos.infrastructure.adapter.in_memory.plugin_resolver import InMemoryPluginResolver
from argos.infrastructure.adapter.in_memory.result_store import InMemoryResultStore
from argos.infrastructure.adapter.in_memory.task_runner import InMemoryTaskRunner
from argos.infrastructure.adapter.in_memory.workflow_engine import InMemoryWorkflowEngine


class InMemoryClient(WorkflowClient):
    pass


def create(plugins: list[type[PluginBase]]) -> InMemoryClient:
    result_store = InMemoryResultStore()
    context = ExecutionContext(result_store)
    executor_factory = InMemoryExecutorFactory(
        resolver=InMemoryPluginResolver(plugins),
        binder=ParameterBinder(),
        values=VariableResolver(context),
        task_runner=InMemoryTaskRunner(),
        execution_options=ExecutionOptions(retries=1, timeout=30),
    )

    workflow_engine = InMemoryWorkflowEngine(
        executor_factory=executor_factory,
        result_store=result_store,
        registrar=ResultRegistrar(result_store=result_store),
    )
    plugin_resolver = InMemoryPluginResolver(plugins)

    client = InMemoryClient(
        plugin_resolver=plugin_resolver,
        executor_factory=executor_factory,
        workflow_engine=workflow_engine,
        result_store=result_store,
        execution_options=ExecutionOptions(retries=1, timeout=30),
        binder=ParameterBinder(),
        exectuion_context=context,
    )

    return client
