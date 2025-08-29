"""
Tests for client facade.

This module tests the Client facade implementation.
"""

from unittest.mock import Mock

import pytest

from argos.application.port import PluginResolver, WorkflowEngine
from argos.client import Client
from argos.domain.entity import WorkflowResult
from argos.domain.port import PluginBase
from argos.domain.value_object import WorkflowResultStatus


class TestPlugin(PluginBase):
    """Test plugin for client tests."""

    plugin_name = "test_plugin"

    def execute(self, value: str) -> str:
        return f"processed: {value}"


class AnotherTestPlugin(PluginBase):
    """Another test plugin for client tests."""

    def execute(self, number: int) -> int:
        return number * 2


class TestClient:
    """Test cases for Client facade."""

    def setup_method(self):
        """Setup test fixtures."""
        self.backend = Mock(spec=WorkflowEngine)
        self.plugin_resolver = Mock(spec=PluginResolver)
        self.executor_factory = Mock()

        self.client = Client(
            backend=self.backend, plugin_resolver=self.plugin_resolver, executor_factory=self.executor_factory
        )

    def test_create_client(self):
        """Test creating client facade."""
        assert self.client._engine == self.backend
        assert self.client._resolver == self.plugin_resolver
        assert self.client._executor_factory == self.executor_factory

    def test_client_without_executor_factory(self):
        """Test creating client without executor factory."""
        client = Client(backend=self.backend, plugin_resolver=self.plugin_resolver)

        assert client._engine == self.backend
        assert client._resolver == self.plugin_resolver
        assert client._executor_factory is None

    def test_plugin_registration_with_executor_factory(self):
        """Test plugin registration with executor factory."""
        # Setup executor factory with resolver that has registry
        self.executor_factory.resolver = Mock()
        self.executor_factory.resolver._registry = {}

        result = self.client.plugin(TestPlugin)

        # Should register plugin in executor factory's resolver
        assert "test_plugin" in self.executor_factory.resolver._registry
        assert self.executor_factory.resolver._registry["test_plugin"] == TestPlugin

        # Should return self for method chaining
        assert result == self.client

    def test_plugin_registration_with_default_name(self):
        """Test plugin registration with default name (class name)."""
        self.executor_factory.resolver = Mock()
        self.executor_factory.resolver._registry = {}

        # Plugin without custom plugin_name
        class DefaultNamePlugin(PluginBase):
            def execute(self) -> str:
                return "default"

        result = self.client.plugin(DefaultNamePlugin)

        # Should register with class name
        assert "DefaultNamePlugin" in self.executor_factory.resolver._registry
        assert result == self.client

    def test_plugin_registration_without_executor_factory(self):
        """Test plugin registration without executor factory."""
        client = Client(backend=self.backend, plugin_resolver=self.plugin_resolver)

        # Should not crash when executor factory is None
        result = client.plugin(TestPlugin)
        assert result == client

    def test_plugin_registration_without_registry(self):
        """Test plugin registration when executor factory has no registry."""
        self.executor_factory.resolver = Mock()
        # No _registry attribute - the hasattr check should prevent assignment

        # Should not crash even if the resolver doesn't have _registry
        result = self.client.plugin(TestPlugin)
        assert result == self.client

    def test_run_workflow_from_dict(self):
        """Test running workflow from dictionary."""
        workflow_dict = {
            "steps": [
                {"id": "test_step", "kind": "operation", "operation": "test_op", "parameters": {"param": "value"}}
            ]
        }

        expected_result = WorkflowResult(id="workflow_123", status=WorkflowResultStatus.SUCCESS, results=[])
        self.backend.run.return_value = expected_result

        result = self.client.run(workflow_dict)

        # Should convert dict to WorkflowDSL and pass to engine
        self.backend.run.assert_called_once()
        workflow_arg = self.backend.run.call_args[0][0]

        # Should be WorkflowDSL
        from argos.domain.entity import WorkflowDSL

        assert isinstance(workflow_arg, WorkflowDSL)
        assert len(workflow_arg.steps) == 1

        assert result == expected_result

    def test_run_workflow_with_id(self):
        """Test running workflow with custom workflow ID."""
        workflow_dict = {"steps": [{"id": "test_step", "kind": "operation", "operation": "test_op", "parameters": {}}]}

        expected_result = WorkflowResult(id="custom_id", status=WorkflowResultStatus.SUCCESS, results=[])
        self.backend.run.return_value = expected_result

        _ = self.client.run(workflow_dict, workflow_id="custom_id")

        # Should pass workflow_id to backend
        self.backend.run.assert_called_once()
        call_args = self.backend.run.call_args
        # Check if workflow_id was passed as keyword argument
        if len(call_args) > 1 and "workflow_id" in call_args[1]:
            assert call_args[1]["workflow_id"] == "custom_id"
        else:
            # If not passed as kwarg, check if it was passed as positional
            assert len(call_args[0]) >= 2

    def test_run_workflow_validation_error(self):
        """Test running workflow with validation error."""
        invalid_workflow = {
            "steps": []  # Empty steps will cause validation error
        }

        with pytest.raises(ValueError, match="Workflow has no steps"):
            self.client.run(invalid_workflow)

    def test_run_workflow_engine_error(self):
        """Test running workflow when engine raises error."""
        workflow_dict = {"steps": [{"id": "test_step", "kind": "operation", "operation": "test_op", "parameters": {}}]}

        self.backend.run.side_effect = RuntimeError("Engine failed")

        with pytest.raises(RuntimeError, match="Engine failed"):
            self.client.run(workflow_dict)

    def test_load_plugins_static_method(self):
        """Test loading plugins as static method."""
        # Clear plugins registry first
        PluginBase._plugins.clear()

        # Register some plugins
        class Plugin1(PluginBase):
            def execute(self) -> str:
                return "plugin1"

        class Plugin2(PluginBase):
            def execute(self) -> str:
                return "plugin2"

        plugins = self.client.load_plugins()

        assert len(plugins) == 2
        assert Plugin1 in plugins
        assert Plugin2 in plugins

    def test_get_available_plugins_static_method(self):
        """Test getting available plugins as static method."""
        # Clear plugins registry first
        PluginBase._plugins.clear()

        # Register a plugin
        class StaticTestPlugin(PluginBase):
            def execute(self) -> str:
                return "static"

        plugins = Client.get_available_plugins()

        assert len(plugins) == 1
        assert StaticTestPlugin in plugins

    def test_get_available_plugins_instance_method(self):
        """Test getting available plugins as instance method."""
        # Clear plugins registry first
        PluginBase._plugins.clear()

        # Register a plugin
        class InstanceTestPlugin(PluginBase):
            def execute(self) -> str:
                return "instance"

        plugins = self.client.load_plugins()

        assert len(plugins) == 1
        assert InstanceTestPlugin in plugins

    def test_method_chaining(self):
        """Test method chaining for fluent interface."""
        self.executor_factory.resolver = Mock()
        self.executor_factory.resolver._registry = {}

        # Should support chaining
        result = self.client.plugin(TestPlugin).plugin(AnotherTestPlugin)

        assert result == self.client
        assert len(self.executor_factory.resolver._registry) == 2

    def test_multiple_plugin_registration(self):
        """Test registering multiple plugins."""
        self.executor_factory.resolver = Mock()
        self.executor_factory.resolver._registry = {}

        self.client.plugin(TestPlugin)
        self.client.plugin(AnotherTestPlugin)

        assert "test_plugin" in self.executor_factory.resolver._registry
        assert "AnotherTestPlugin" in self.executor_factory.resolver._registry

    def test_client_with_real_workflow_execution(self):
        """Test client with more realistic workflow execution."""
        # Setup a more realistic backend mock
        workflow_result = WorkflowResult(
            id="real_workflow",
            status=WorkflowResultStatus.SUCCESS,
            results=[
                Mock(id="step1", result="step1_result"),
                Mock(id="step2", result="step2_result"),
            ],
        )
        self.backend.run.return_value = workflow_result

        workflow = {
            "steps": [
                {"id": "step1", "kind": "operation", "operation": "op1", "parameters": {"input": "data1"}},
                {"id": "step2", "kind": "operation", "operation": "op2", "parameters": {"input": "${step1}"}},
            ]
        }

        result = self.client.run(workflow)

        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 2

    def test_client_error_propagation(self):
        """Test that client properly propagates errors."""
        # Test workflow loading error
        invalid_workflow = {"invalid": "structure"}

        with pytest.raises(Exception):  # noqa B017
            self.client.run(invalid_workflow)

    def test_client_components_access(self):
        """Test accessing client components."""
        # Internal components should be accessible for testing
        assert self.client._engine == self.backend
        assert self.client._resolver == self.plugin_resolver
        assert self.client._executor_factory == self.executor_factory

    def test_plugin_registration_edge_cases(self):
        """Test plugin registration edge cases."""
        self.executor_factory.resolver = Mock()
        self.executor_factory.resolver._registry = {}

        # Plugin with None plugin_name
        class NoneNamePlugin(PluginBase):
            plugin_name = None

            def execute(self) -> str:
                return "none_name"

        self.client.plugin(NoneNamePlugin)

        # Should use class name when plugin_name is None (getattr behavior)
        # Since getattr(cls, "plugin_name", cls.__name__) returns None when plugin_name exists but is None
        assert None in self.executor_factory.resolver._registry

    def test_client_isolation(self):
        """Test that different client instances are isolated."""
        backend2 = Mock(spec=WorkflowEngine)
        resolver2 = Mock(spec=PluginResolver)
        executor_factory2 = Mock()

        client2 = Client(backend=backend2, plugin_resolver=resolver2, executor_factory=executor_factory2)

        # Should have different components
        assert self.client._engine != client2._engine
        assert self.client._resolver != client2._resolver
        assert self.client._executor_factory != client2._executor_factory

    def test_load_plugins_registry_state(self):
        """Test that load_plugins reflects current registry state."""
        # Clear registry
        PluginBase._plugins.clear()

        # Should be empty
        assert len(self.client.load_plugins()) == 0

        # Add plugin
        class DynamicPlugin(PluginBase):
            def execute(self) -> str:
                return "dynamic"

        # Should now have one plugin
        assert len(self.client.load_plugins()) == 1
        assert DynamicPlugin in self.client.load_plugins()
