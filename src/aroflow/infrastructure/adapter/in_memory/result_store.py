from typing import Any

from aroflow.application.adapter import ResultStore


class InMemoryResultStore(ResultStore):
    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self._store[key] = value

    def get(self, key: str) -> Any:
        return self._store[key]
