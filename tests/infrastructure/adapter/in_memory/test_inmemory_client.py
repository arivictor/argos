"""
Tests for in-memory client.

This module tests the InMemoryClient and create function.
"""

import pytest
from unittest.mock import Mock, patch

from argos.infrastructure.adapter.in_memory.client import InMemoryClient, create
from argos.domain.port import PluginBase
from argos.domain.value_object import ExecutionOptions


class TestPlugin(PluginBase):
    """Test plugin for testing."""
    plugin_name = "test_plugin"
    
    def execute(self, value: str) -> str:
        return f"processed: {value}"


class AnotherTestPlugin(PluginBase):
    """Another test plugin."""
    
    def execute(self, number: int) -> int:
        return number * 2


class TestInMemoryClient:
    """Test cases for InMemoryClient."""

    def test_inmemory_client_inheritance(self):
        """Test that InMemoryClient inherits from WorkflowClient."""
        from argos.application.service import WorkflowClient
        
        # Create mock dependencies
        plugin_resolver = Mock()
        executor_factory = Mock()
        workflow_engine = Mock()
        result_store = Mock()
        binder = Mock()
        execution_context = Mock()
        
        client = InMemoryClient(
            plugin_resolver=plugin_resolver,
            executor_factory=executor_factory,
            workflow_engine=workflow_engine,
            result_store=result_store,
            binder=binder,
            exectuion_context=execution_context,  # Note: typo in original
            execution_options=ExecutionOptions()
        )
        
        assert isinstance(client, WorkflowClient)


class TestCreateFunction:
    """Test cases for the create function."""

    def test_create_with_empty_plugins(self):
        """Test creating client with empty plugin list."""
        client = create([])
        
        assert isinstance(client, InMemoryClient)
        assert hasattr(client, 'resolver')
        assert hasattr(client, 'executor_factory')
        assert hasattr(client, 'engine')
        assert hasattr(client, 'result_store')
        assert hasattr(client, 'binder')
        assert hasattr(client, 'ctx')

    def test_create_with_plugins(self):
        """Test creating client with plugin list."""
        plugins = [TestPlugin, AnotherTestPlugin]
        
        client = create(plugins)
        
        assert isinstance(client, InMemoryClient)
        # The plugins should be registered in the resolver
        assert hasattr(client, 'resolver')

    def test_create_sets_execution_options(self):
        """Test that create function sets correct execution options."""
        client = create([])
        
        assert hasattr(client, 'execution_options')
        assert isinstance(client.execution_options, ExecutionOptions)
        assert client.execution_options.retries == 1
        assert client.execution_options.timeout == 30

    def test_create_components_wiring(self):
        """Test that create function properly wires components."""
        plugins = [TestPlugin]
        client = create(plugins)
        
        # Test that components are properly connected
        from argos.infrastructure.adapter.in_memory.result_store import InMemoryResultStore
        from argos.infrastructure.adapter.in_memory.plugin_resolver import InMemoryPluginResolver
        from argos.infrastructure.adapter.in_memory.executor_factory import InMemoryExecutorFactory
        from argos.infrastructure.adapter.in_memory.workflow_engine import InMemoryWorkflowEngine
        from argos.infrastructure.adapter.in_memory.task_runner import InMemoryTaskRunner
        from argos.application.adapter import ParameterBinder, ExecutionContext, VariableResolver
        from argos.application.service import ResultRegistrar
        
        assert isinstance(client.result_store, InMemoryResultStore)
        assert isinstance(client.resolver, InMemoryPluginResolver)
        assert isinstance(client.executor_factory, InMemoryExecutorFactory)
        assert isinstance(client.engine, InMemoryWorkflowEngine)
        assert isinstance(client.binder, ParameterBinder)
        assert isinstance(client.ctx, ExecutionContext)

    def test_create_with_multiple_plugins(self):
        """Test creating client with multiple plugins."""
        plugins = [TestPlugin, AnotherTestPlugin]
        
        client = create(plugins)
        
        # Should be able to access both plugins through resolver
        # Note: This is testing the integration, so we need to check 
        # that the plugins are actually registered
        assert isinstance(client, InMemoryClient)

    def test_create_plugin_registration(self):
        """Test that plugins are properly registered during creation."""
        plugins = [TestPlugin, AnotherTestPlugin]
        
        client = create(plugins)
        
        # The resolver should have the plugins registered
        # We can test this by checking the resolver's registry
        resolver = client.resolver
        
        # TestPlugin should be registered with its custom name
        try:
            test_plugin_instance = resolver.resolve("test_plugin")
            assert isinstance(test_plugin_instance, TestPlugin)
        except KeyError:
            pytest.fail("TestPlugin not registered correctly")
        
        # AnotherTestPlugin should be registered with class name
        try:
            another_plugin_instance = resolver.resolve("AnotherTestPlugin")
            assert isinstance(another_plugin_instance, AnotherTestPlugin)
        except KeyError:
            pytest.fail("AnotherTestPlugin not registered correctly")

    def test_create_result_store_integration(self):
        """Test that result store is properly integrated."""
        client = create([])
        
        # Result store should be shared between components
        # The execution context should use the same result store
        assert client.ctx.results == client.result_store
        
        # The registrar should use the same result store
        assert client.registrar.result_store == client.result_store

    def test_create_execution_context_integration(self):
        """Test that execution context is properly integrated."""
        client = create([])
        
        # Variable resolver should use the execution context
        assert client.values.ctx == client.ctx

    def test_create_workflow_engine_integration(self):
        """Test that workflow engine is properly configured."""
        client = create([])
        
        # Workflow engine should use the executor factory
        assert client.engine.executor_factory == client.executor_factory
        
        # Workflow engine should have a registrar (but not necessarily the same instance)
        assert hasattr(client.engine, 'registrar')
        assert client.engine.registrar is not None

    def test_create_executor_factory_integration(self):
        """Test that executor factory is properly configured."""
        client = create([])
        
        # Executor factory should have all necessary components
        factory = client.executor_factory
        
        assert factory.resolver is not None
        assert factory.binder is not None
        assert factory.values is not None
        assert factory.task_runner is not None
        assert isinstance(factory.execution_options, ExecutionOptions)

    def test_create_duplicate_plugin_resolver(self):
        """Test that create function creates duplicate plugin resolver."""
        # The create function creates two resolvers - one for executor factory
        # and one for the client itself
        plugins = [TestPlugin]
        client = create(plugins)
        
        # Both should exist
        assert hasattr(client, 'resolver')
        assert hasattr(client.executor_factory, 'resolver')
        
        # They should be different instances but configured the same way
        client_resolver = client.resolver
        factory_resolver = client.executor_factory.resolver
        
        # Both should be able to resolve the same plugins
        client_plugin = client_resolver.resolve("test_plugin")
        factory_plugin = factory_resolver.resolve("test_plugin")
        
        assert type(client_plugin) == type(factory_plugin)

    def test_create_with_complex_plugins(self):
        """Test creating client with complex plugin configurations."""
        class ComplexPlugin(PluginBase):
            plugin_name = "complex"
            
            def execute(self, config: dict, multiplier: int = 1, **kwargs) -> dict:
                return {
                    "result": {k: v * multiplier for k, v in config.items() if isinstance(v, (int, float))},
                    "metadata": kwargs
                }
        
        class SimplePlugin(PluginBase):
            def execute(self) -> str:
                return "simple"
        
        plugins = [ComplexPlugin, SimplePlugin]
        client = create(plugins)
        
        # Should be able to resolve both plugins
        complex_instance = client.resolver.resolve("complex")
        simple_instance = client.resolver.resolve("SimplePlugin")
        
        assert isinstance(complex_instance, ComplexPlugin)
        assert isinstance(simple_instance, SimplePlugin)

    def test_create_immutability(self):
        """Test that multiple calls to create return independent clients."""
        plugins1 = [TestPlugin]
        plugins2 = [AnotherTestPlugin]
        
        client1 = create(plugins1)
        client2 = create(plugins2)
        
        # Should be different instances
        assert client1 is not client2
        assert client1.resolver is not client2.resolver
        assert client1.result_store is not client2.result_store
        
        # Should have different plugin registrations
        # client1 should have TestPlugin but not AnotherTestPlugin
        assert client1.resolver.resolve("test_plugin")
        with pytest.raises(KeyError):
            client1.resolver.resolve("AnotherTestPlugin")
        
        # client2 should have AnotherTestPlugin but not TestPlugin  
        assert client2.resolver.resolve("AnotherTestPlugin")
        with pytest.raises(KeyError):
            client2.resolver.resolve("test_plugin")

    def test_create_with_no_plugins_list(self):
        """Test creating client when no plugins provided (should handle gracefully)."""
        # Test with empty list (explicit)
        client = create([])
        
        assert isinstance(client, InMemoryClient)
        
        # Should have empty resolver
        with pytest.raises(KeyError):
            client.resolver.resolve("nonexistent")

    def test_create_return_type(self):
        """Test that create function returns correct type."""
        client = create([TestPlugin])
        
        assert isinstance(client, InMemoryClient)
        # InMemoryClient should also be a WorkflowClient
        from argos.application.service import WorkflowClient
        assert isinstance(client, WorkflowClient)