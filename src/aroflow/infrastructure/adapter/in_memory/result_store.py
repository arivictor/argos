from typing import Any

from aroflow.application.adapter import ResultStore


class InMemoryResultStore(ResultStore):
    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any):
        """
        Store a value with the given key.

        :param key: The key to store the value under
        :type key: str
        :param value: The value to store
        :type value: Any
        """
        self._store[key] = value

    def get(self, key: str) -> Any:
        """
        Retrieve a value by key.

        :param key: The key to retrieve the value for
        :type key: str
        :returns: The stored value
        :rtype: Any
        :raises KeyError: If the key is not found
        """
        return self._store[key]
