from aroflow.backend import BackendType
from aroflow.client import Client
from aroflow.domain.port import PluginBase
from aroflow.infrastructure.adapter.in_memory.client import create as create_in_memory_client
from aroflow.infrastructure.adapter.sqlite.client import create as create_sqlite_client


def create(backend: BackendType, plugins: list[type[PluginBase]] | None = None, **kwargs) -> Client:
    """
    Factory function to create a Client with the specified backend.

    :param backend: The backend type to use for workflow execution
    :type backend: BackendType
    :param plugins: Optional list of plugin classes to pre-register
    :type plugins: list[type[PluginBase]] | None
    :param kwargs: Additional backend-specific configuration options
    :returns: A configured Client instance
    :rtype: Client
    :raises ValueError: If the backend type is unsupported
    """
    plugins = plugins or []

    if backend == BackendType.IN_MEMORY:
        # Create the in-memory client using the existing factory
        in_memory_client = create_in_memory_client(plugins)

        # Create the unified client façade
        client = Client(
            backend=in_memory_client.engine,
            plugin_resolver=in_memory_client.resolver,
            executor_factory=in_memory_client.executor_factory,
        )

        return client

    elif backend == BackendType.SQLITE:
        # Create the SQLite client using the SQLite factory
        db_path = kwargs.get("db_path", ":memory:")
        sqlite_client = create_sqlite_client(plugins, db_path=db_path)

        # Create the unified client façade
        client = Client(
            backend=sqlite_client.engine,
            plugin_resolver=sqlite_client.resolver,
            executor_factory=sqlite_client.executor_factory,
        )

        return client

    elif backend == BackendType.TEMPORAL:
        # TODO: Implement Temporal backend
        raise ValueError("Temporal backend not yet implemented")

    elif backend == BackendType.CELERY:
        # TODO: Implement Celery backend
        raise ValueError("Celery backend not yet implemented")

    else:
        raise ValueError(f"Unsupported backend: {backend}")
