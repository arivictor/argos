"""
Tests for in-memory result store.

This module tests the InMemoryResultStore implementation.
"""

import pytest

from aroflow.infrastructure.adapter.in_memory.result_store import InMemoryResultStore


class TestInMemoryResultStore:
    """Test cases for InMemoryResultStore."""

    def setup_method(self):
        """Setup test fixtures."""
        self.store = InMemoryResultStore()

    def test_create_result_store(self):
        """Test creating in-memory result store."""
        assert hasattr(self.store, "_store")
        assert isinstance(self.store._store, dict)
        assert len(self.store._store) == 0

    def test_set_and_get_value(self):
        """Test setting and getting a value."""
        self.store.set("key1", "value1")

        result = self.store.get("key1")
        assert result == "value1"

    def test_set_multiple_values(self):
        """Test setting multiple values."""
        self.store.set("key1", "value1")
        self.store.set("key2", "value2")
        self.store.set("key3", 42)

        assert self.store.get("key1") == "value1"
        assert self.store.get("key2") == "value2"
        assert self.store.get("key3") == 42

    def test_overwrite_value(self):
        """Test overwriting existing value."""
        self.store.set("key", "original_value")
        assert self.store.get("key") == "original_value"

        self.store.set("key", "new_value")
        assert self.store.get("key") == "new_value"

    def test_get_nonexistent_key(self):
        """Test getting nonexistent key raises KeyError."""
        with pytest.raises(KeyError):
            self.store.get("nonexistent_key")

    def test_set_none_value(self):
        """Test setting None as value."""
        self.store.set("none_key", None)

        result = self.store.get("none_key")
        assert result is None

    def test_set_complex_objects(self):
        """Test setting complex objects as values."""
        complex_obj = {"nested": {"data": [1, 2, 3]}, "list": ["a", "b", "c"], "number": 42, "boolean": True}

        self.store.set("complex", complex_obj)

        result = self.store.get("complex")
        assert result == complex_obj
        assert result is complex_obj  # Should be the same object reference

    def test_key_types(self):
        """Test different key types."""
        # Keys should be strings
        self.store.set("string_key", "value1")
        self.store.set("123", "value2")
        self.store.set("key_with_underscores", "value3")
        self.store.set("key-with-dashes", "value4")

        assert self.store.get("string_key") == "value1"
        assert self.store.get("123") == "value2"
        assert self.store.get("key_with_underscores") == "value3"
        assert self.store.get("key-with-dashes") == "value4"

    def test_empty_string_key(self):
        """Test empty string as key."""
        self.store.set("", "empty_key_value")

        result = self.store.get("")
        assert result == "empty_key_value"

    def test_value_types(self):
        """Test storing different value types."""
        test_values = {
            "string": "test_string",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3, "four"],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "none": None,
        }

        for key, value in test_values.items():
            self.store.set(key, value)

        for key, expected_value in test_values.items():
            actual_value = self.store.get(key)
            assert actual_value == expected_value
            if expected_value is not None:
                assert type(actual_value) is type(expected_value)

    def test_store_isolation(self):
        """Test that different store instances are isolated."""
        store1 = InMemoryResultStore()
        store2 = InMemoryResultStore()

        store1.set("key", "value1")
        store2.set("key", "value2")

        assert store1.get("key") == "value1"
        assert store2.get("key") == "value2"

        # Should not find key from other store
        store1.set("unique1", "value")
        store2.set("unique2", "value")

        with pytest.raises(KeyError):
            store1.get("unique2")

        with pytest.raises(KeyError):
            store2.get("unique1")

    def test_store_state_persistence(self):
        """Test that store maintains state across operations."""
        # Add multiple values
        values = {f"key_{i}": f"value_{i}" for i in range(10)}

        for key, value in values.items():
            self.store.set(key, value)

        # Verify all values are still there
        for key, expected_value in values.items():
            assert self.store.get(key) == expected_value

        # Modify some values
        self.store.set("key_0", "modified_value_0")
        self.store.set("key_5", "modified_value_5")

        # Check modifications
        assert self.store.get("key_0") == "modified_value_0"
        assert self.store.get("key_5") == "modified_value_5"

        # Check others remain unchanged
        assert self.store.get("key_1") == "value_1"
        assert self.store.get("key_9") == "value_9"
