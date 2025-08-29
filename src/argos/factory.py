from argos.backend import BackendType
from argos.client import Client
from argos.domain.port import PluginBase
from argos.infrastructure.adapter.in_memory.client import create as create_in_memory_client


def create(backend: BackendType, plugins: list[type[PluginBase]] | None = None) -> Client:
    """
    Factory function to create a Client with the specified backend.

    Args:
        backend: The backend type to use for workflow execution
        plugins: Optional list of plugin classes to pre-register

    Returns:
        A configured Client instance

    Raises:
        ValueError: If the backend type is unsupported
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

    elif backend == BackendType.TEMPORAL:
        # TODO: Implement Temporal backend
        raise ValueError("Temporal backend not yet implemented")

    elif backend == BackendType.CELERY:
        # TODO: Implement Celery backend
        raise ValueError("Celery backend not yet implemented")

    else:
        raise ValueError(f"Unsupported backend: {backend}")
