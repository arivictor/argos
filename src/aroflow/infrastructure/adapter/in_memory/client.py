from aroflow.application.adapter import ExecutionContext, ParameterBinder, VariableResolver
from aroflow.application.service import ResultRegistrar, WorkflowClient
from aroflow.domain.port import PluginBase
from aroflow.domain.value_object import ExecutionOptions
from aroflow.infrastructure.adapter.in_memory.executor_factory import InMemoryExecutorFactory
from aroflow.infrastructure.adapter.in_memory.plugin_resolver import InMemoryPluginResolver
from aroflow.infrastructure.adapter.in_memory.result_store import InMemoryResultStore
from aroflow.infrastructure.adapter.in_memory.task_runner import InMemoryTaskRunner
from aroflow.infrastructure.adapter.in_memory.workflow_engine import InMemoryWorkflowEngine


class InMemoryClient(WorkflowClient):
    pass


def create(plugins: list[type[PluginBase]]) -> InMemoryClient:
    """
    Creates an InMemoryClient with the specified plugins.

    :param plugins: List of plugin classes to register
    :type plugins: list[type[PluginBase]]
    :returns: Configured InMemoryClient instance
    :rtype: InMemoryClient
    """
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
