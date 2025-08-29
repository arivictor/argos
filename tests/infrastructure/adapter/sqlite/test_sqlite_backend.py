"""
Tests for SQLite backend functionality.

This module tests the SQLite backend implementation.
"""

import tempfile
import os
import pytest

from aroflow.backend import BackendType
from aroflow.domain.port import PluginBase
from aroflow.infrastructure.adapter.sqlite.result_store import SQLiteResultStore
import aroflow


class TestSQLiteResultStore:
    """Test cases for SQLiteResultStore."""

    def test_in_memory_sqlite_store(self):
        """Test SQLite result store with in-memory database."""
        store = SQLiteResultStore()
        
        # Test basic set/get
        store.set("test.step1", {"result": "success", "value": 42})
        result = store.get("test.step1")
        
        assert result == {"result": "success", "value": 42}

    def test_file_based_sqlite_store(self):
        """Test SQLite result store with file-based database."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name
        
        try:
            store = SQLiteResultStore(db_path)
            
            # Test storing complex data
            test_data = {
                "workflow_id": "test123",
                "results": [1, 2, 3],
                "metadata": {"timestamp": "2023-01-01", "user": "test"}
            }
            store.set("workflow1.step1", test_data)
            
            # Create new store instance to test persistence
            new_store = SQLiteResultStore(db_path)
            retrieved_data = new_store.get("workflow1.step1")
            
            assert retrieved_data == test_data
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_key_not_found_error(self):
        """Test that KeyError is raised for non-existent keys."""
        store = SQLiteResultStore()
        
        with pytest.raises(KeyError, match="Key 'nonexistent.key' not found"):
            store.get("nonexistent.key")

    def test_key_parsing(self):
        """Test that keys are parsed correctly for workflow_id and step_id."""
        store = SQLiteResultStore()
        
        # Test simple key (no dot)
        store.set("simple_key", "simple_value")
        assert store.get("simple_key") == "simple_value"
        
        # Test complex key (with dot)
        store.set("workflow123.step456", "complex_value")
        assert store.get("workflow123.step456") == "complex_value"


class TestSQLiteBackend:
    """Test cases for SQLite backend integration."""

    def test_sqlite_backend_creation(self):
        """Test creating a client with SQLite backend."""
        class TestPlugin(PluginBase):
            plugin_name = "test_plugin"
            
            def execute(self, value: str) -> str:
                return f"Processed: {value}"
        
        client = aroflow.create(BackendType.SQLITE, plugins=[TestPlugin])
        assert client is not None

    def test_sqlite_workflow_execution(self):
        """Test executing a workflow with SQLite backend."""
        class TestPlugin(PluginBase):
            plugin_name = "test_plugin"
            
            def execute(self, value: str) -> str:
                return f"Processed: {value}"
        
        client = aroflow.create(BackendType.SQLITE, plugins=[TestPlugin])
        
        workflow = {
            "steps": [
                {
                    "id": "step1",
                    "kind": "operation",
                    "operation": "test_plugin",
                    "parameters": {"value": "Hello SQLite!"}
                }
            ]
        }
        
        result = client.run(workflow)
        
        assert result.status.value == "success"
        assert len(result.results) == 1
        assert result.results[0].result == "Processed: Hello SQLite!"
        assert result.results[0].id == "step1"

    def test_sqlite_workflow_with_file_database(self):
        """Test executing a workflow with file-based SQLite database."""
        class TestPlugin(PluginBase):
            plugin_name = "test_plugin"
            
            def execute(self, value: str) -> str:
                return f"Processed: {value}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name
        
        try:
            client = aroflow.create(BackendType.SQLITE, plugins=[TestPlugin], db_path=db_path)
            
            workflow = {
                "steps": [
                    {
                        "id": "step1",
                        "kind": "operation",
                        "operation": "test_plugin",
                        "parameters": {"value": "Persistent Hello!"}
                    }
                ]
            }
            
            result = client.run(workflow)
            
            assert result.status.value == "success"
            assert len(result.results) == 1
            assert result.results[0].result == "Processed: Persistent Hello!"
            
            # Verify database file was created
            assert os.path.exists(db_path)
            assert os.path.getsize(db_path) > 0
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)