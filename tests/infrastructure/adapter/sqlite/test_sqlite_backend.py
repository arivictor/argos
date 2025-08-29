"""
Tests for SQLite backend functionality.

This module tests the SQLite backend implementation.
"""

import os
import tempfile

import pytest

import aroflow
from aroflow.backend import BackendType
from aroflow.domain.port import PluginBase
from aroflow.infrastructure.adapter.sqlite.result_store import SQLiteResultStore


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
                "metadata": {"timestamp": "2023-01-01", "user": "test"},
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

    def test_get_workflow_results(self):
        """Test retrieving all results for a workflow."""
        store = SQLiteResultStore()

        # Store multiple results for a workflow
        store.set("workflow1.step1", {"result": "data1"})
        store.set("workflow1.step2", {"result": "data2"})
        store.set("workflow1.step3", {"result": "data3"})
        # Store results for different workflow
        store.set("workflow2.step1", {"result": "other_data"})

        # Get results for workflow1
        results = store.get_workflow_results("workflow1")

        assert len(results) == 3
        assert results["step1"] == {"result": "data1"}
        assert results["step2"] == {"result": "data2"}
        assert results["step3"] == {"result": "data3"}

    def test_get_workflow_results_not_found(self):
        """Test that KeyError is raised for non-existent workflow."""
        store = SQLiteResultStore()

        with pytest.raises(KeyError, match="No results found for workflow 'nonexistent'"):
            store.get_workflow_results("nonexistent")

    def test_delete_workflow_results(self):
        """Test deleting all results for a workflow."""
        store = SQLiteResultStore()

        # Store multiple results for multiple workflows
        store.set("workflow1.step1", {"result": "data1"})
        store.set("workflow1.step2", {"result": "data2"})
        store.set("workflow2.step1", {"result": "other_data"})

        # Delete workflow1 results
        deleted = store.delete_workflow_results("workflow1")
        assert deleted is True

        # Verify workflow1 results are gone
        with pytest.raises(KeyError):
            store.get_workflow_results("workflow1")

        # Verify workflow2 results still exist
        results = store.get_workflow_results("workflow2")
        assert results["step1"] == {"result": "other_data"}

    def test_delete_workflow_results_not_found(self):
        """Test deleting non-existent workflow returns False."""
        store = SQLiteResultStore()

        deleted = store.delete_workflow_results("nonexistent")
        assert deleted is False

    def test_list_workflow_ids(self):
        """Test listing all workflow IDs."""
        store = SQLiteResultStore()

        # Initially empty
        workflow_ids = store.list_workflow_ids()
        assert workflow_ids == []

        # Add some workflows
        store.set("workflow_c.step1", {"result": "data"})
        store.set("workflow_a.step1", {"result": "data"})
        store.set("workflow_b.step1", {"result": "data"})
        store.set("workflow_a.step2", {"result": "data"})  # Same workflow, different step

        # Should return sorted unique workflow IDs
        workflow_ids = store.list_workflow_ids()
        assert workflow_ids == ["workflow_a", "workflow_b", "workflow_c"]


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
                    "parameters": {"value": "Hello SQLite!"},
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
                        "parameters": {"value": "Persistent Hello!"},
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

    def test_workflow_querying(self):
        """Test querying workflow results by ID."""

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
                    "parameters": {"value": "Hello"},
                },
                {
                    "id": "step2",
                    "kind": "operation",
                    "operation": "test_plugin",
                    "parameters": {"value": "World"},
                },
            ]
        }

        # Execute workflow with specific ID
        workflow_id = "test_workflow_123"
        _ = client.run(workflow, workflow_id=workflow_id)

        # Query the workflow results
        workflow_results = client.get_workflow(workflow_id)

        assert len(workflow_results) == 2
        assert "step1" in workflow_results
        assert "step2" in workflow_results
        assert workflow_results["step1"]["result"] == "Processed: Hello"
        assert workflow_results["step2"]["result"] == "Processed: World"
        assert workflow_results["step1"]["id"] == "step1"
        assert workflow_results["step2"]["id"] == "step2"
        assert workflow_results["step1"]["kind"] == "operation"
        assert workflow_results["step2"]["kind"] == "operation"

    def test_workflow_querying_not_found(self):
        """Test querying non-existent workflow raises KeyError."""

        class TestPlugin(PluginBase):
            plugin_name = "test_plugin"

            def execute(self, value: str) -> str:
                return f"Processed: {value}"

        client = aroflow.create(BackendType.SQLITE, plugins=[TestPlugin])

        with pytest.raises(KeyError, match="No results found for workflow 'nonexistent'"):
            client.get_workflow("nonexistent")

    def test_workflow_deletion(self):
        """Test deleting workflow results by ID."""

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
                    "parameters": {"value": "Hello"},
                }
            ]
        }

        # Execute two workflows
        workflow_id1 = "workflow_to_delete"
        workflow_id2 = "workflow_to_keep"

        client.run(workflow, workflow_id=workflow_id1)
        client.run(workflow, workflow_id=workflow_id2)

        # Verify both workflows exist
        assert len(client.get_workflow(workflow_id1)) == 1
        assert len(client.get_workflow(workflow_id2)) == 1

        # Delete first workflow
        deleted = client.delete_workflow(workflow_id1)
        assert deleted is True

        # Verify first workflow is gone, second still exists
        with pytest.raises(KeyError):
            client.get_workflow(workflow_id1)

        assert len(client.get_workflow(workflow_id2)) == 1

    def test_workflow_deletion_not_found(self):
        """Test deleting non-existent workflow returns False."""

        class TestPlugin(PluginBase):
            plugin_name = "test_plugin"

            def execute(self, value: str) -> str:
                return f"Processed: {value}"

        client = aroflow.create(BackendType.SQLITE, plugins=[TestPlugin])

        deleted = client.delete_workflow("nonexistent")
        assert deleted is False

    def test_list_workflows(self):
        """Test listing all workflow IDs."""

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
                    "parameters": {"value": "Hello"},
                }
            ]
        }

        # Initially no workflows
        workflow_ids = client.list_workflows()
        assert workflow_ids == []

        # Execute some workflows
        client.run(workflow, workflow_id="workflow_c")
        client.run(workflow, workflow_id="workflow_a")
        client.run(workflow, workflow_id="workflow_b")

        # Should return sorted workflow IDs
        workflow_ids = client.list_workflows()
        assert workflow_ids == ["workflow_a", "workflow_b", "workflow_c"]
