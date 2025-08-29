from aroflow.application.adapter import ExecutionContext, ParameterBinder, VariableResolver
from aroflow.application.service import ResultRegistrar, WorkflowClient
from aroflow.domain.port import PluginBase
from aroflow.domain.value_object import ExecutionOptions
from aroflow.infrastructure.adapter.in_memory.executor_factory import InMemoryExecutorFactory
from aroflow.infrastructure.adapter.in_memory.plugin_resolver import InMemoryPluginResolver
from aroflow.infrastructure.adapter.in_memory.task_runner import InMemoryTaskRunner
from aroflow.infrastructure.adapter.sqlite.result_store import SQLiteResultStore
from aroflow.infrastructure.adapter.sqlite.workflow_engine import SQLiteWorkflowEngine


class SQLiteClient(WorkflowClient):
    """SQLite-based workflow client."""

    pass


def create(plugins: list[type[PluginBase]], db_path: str = ":memory:") -> SQLiteClient:
    """
    Creates a SQLiteClient with the specified plugins and database path.

    :param plugins: List of plugin classes to register
    :type plugins: list[type[PluginBase]]
    :param db_path: Path to SQLite database file (defaults to in-memory)
    :type db_path: str
    :returns: Configured SQLiteClient instance
    :rtype: SQLiteClient
    """
    result_store = SQLiteResultStore(db_path=db_path)
    context = ExecutionContext(result_store)
    executor_factory = InMemoryExecutorFactory(
        resolver=InMemoryPluginResolver(plugins),
        binder=ParameterBinder(),
        values=VariableResolver(context),
        task_runner=InMemoryTaskRunner(),
        execution_options=ExecutionOptions(retries=1, timeout=30),
    )

    workflow_engine = SQLiteWorkflowEngine(
        executor_factory=executor_factory,
        result_store=result_store,
        registrar=ResultRegistrar(result_store=result_store),
    )
    plugin_resolver = InMemoryPluginResolver(plugins)

    client = SQLiteClient(
        plugin_resolver=plugin_resolver,
        executor_factory=executor_factory,
        workflow_engine=workflow_engine,
        result_store=result_store,
        execution_options=ExecutionOptions(retries=1, timeout=30),
        binder=ParameterBinder(),
        exectuion_context=context,
    )

    return client
