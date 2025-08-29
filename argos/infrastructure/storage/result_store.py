"""Result storage infrastructure for workflow execution context."""

from abc import ABC, abstractmethod
from typing import Any


class ResultStore(ABC):
    """Abstract interface for storing and retrieving step results."""
    
    @abstractmethod
    def set(self, key: str, value: Any):
        """Store a result value with the given key."""
        ...
        
    @abstractmethod
    def get(self, key: str) -> Any:
        """Retrieve a result value by key."""
        ...


class InMemoryResultStore(ResultStore):
    """In-memory implementation of ResultStore."""
    
    def __init__(self):
        self._store: dict[str, Any] = {}
        
    def set(self, key: str, value: Any):
        self._store[key] = value
        
    def get(self, key: str) -> Any:
        return self._store[key]


class ExecutionContext:
    """Holds results of previously executed steps, addressable by step id."""
    
    def __init__(self, result_store: ResultStore | None = None):
        self.results = result_store if result_store is not None else InMemoryResultStore()


class ResultRegistrar:
    """Handles registration of results into a ResultStore."""
    
    def __init__(self, result_store: ResultStore):
        self.result_store = result_store
        
    def register(self, step_result: Any):
        """Register results into the store, including nested results."""
        from ...domain.models.results import MapResult, ParallelResult, OperationResult
        
        # Store the main result under its id
        if hasattr(step_result, "id"):
            self.result_store.set(step_result.id, step_result)
        # Register nested operation results so they are accessible by id
        if isinstance(step_result, MapResult):
            nested_id = step_result.id
            # Store the entire list of results under the nested operation id
            self.result_store.set(nested_id, step_result.results)
            # Additionally, register each MapItemResult by its own id
            for item_result in step_result.results:
                self.result_store.set(item_result.id, item_result)
        elif isinstance(step_result, ParallelResult):
            for opres in step_result.results:
                self.result_store.set(opres.id, opres)
        elif isinstance(step_result, OperationResult):
            self.result_store.set(step_result.id, step_result)