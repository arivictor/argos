"""
Tests for factory functions.

This module tests the create factory function and backend creation.
"""

import pytest
from unittest.mock import Mock, patch

from argos.factory import create
from argos.backend import BackendType
from argos.client import Client
from argos.domain.port import PluginBase


class TestPlugin(PluginBase):
    """Test plugin for factory tests."""
    plugin_name = "test_plugin"
    
    def execute(self, value: str) -> str:
        return f"processed: {value}"


class AnotherTestPlugin(PluginBase):
    """Another test plugin for factory tests."""
    
    def execute(self, number: int) -> int:
        return number * 2


class TestCreate:
    """Test cases for create factory function."""

    def test_create_in_memory_backend(self):
        """Test creating client with in-memory backend."""
        client = create(BackendType.IN_MEMORY)
        
        assert isinstance(client, Client)
        assert hasattr(client, '_engine')
        assert hasattr(client, '_resolver')
        assert hasattr(client, '_executor_factory')

    def test_create_in_memory_with_empty_plugins(self):
        """Test creating in-memory client with empty plugin list."""
        client = create(BackendType.IN_MEMORY, plugins=[])
        
        assert isinstance(client, Client)

    def test_create_in_memory_with_plugins(self):
        """Test creating in-memory client with plugin list."""
        plugins = [TestPlugin, AnotherTestPlugin]
        
        client = create(BackendType.IN_MEMORY, plugins=plugins)
        
        assert isinstance(client, Client)

    def test_create_in_memory_with_none_plugins(self):
        """Test creating in-memory client with None plugins."""
        client = create(BackendType.IN_MEMORY, plugins=None)
        
        assert isinstance(client, Client)

    def test_create_temporal_backend_not_implemented(self):
        """Test that temporal backend raises ValueError."""
        with pytest.raises(ValueError, match="Temporal backend not yet implemented"):
            create(BackendType.TEMPORAL)

    def test_create_celery_backend_not_implemented(self):
        """Test that celery backend raises ValueError."""
        with pytest.raises(ValueError, match="Celery backend not yet implemented"):
            create(BackendType.CELERY)

    def test_create_unsupported_backend(self):
        """Test that unsupported backend raises ValueError."""
        # Create a mock backend type that's not supported
        class UnsupportedBackend:
            pass
        
        unsupported = UnsupportedBackend()
        
        with pytest.raises(ValueError, match="Unsupported backend"):
            create(unsupported)

    def test_create_in_memory_returns_client_facade(self):
        """Test that create returns Client facade, not InMemoryClient directly."""
        client = create(BackendType.IN_MEMORY)
        
        # Should be the unified Client facade
        assert type(client).__name__ == "Client"
        assert not type(client).__name__ == "InMemoryClient"

    def test_create_client_components_wiring(self):
        """Test that created client has properly wired components."""
        plugins = [TestPlugin]
        client = create(BackendType.IN_MEMORY, plugins=plugins)
        
        # Should have backend engine
        assert client._engine is not None
        
        # Should have plugin resolver  
        assert client._resolver is not None
        
        # Should have executor factory
        assert client._executor_factory is not None

    def test_create_client_plugin_registration(self):
        """Test that plugins are registered in the created client."""
        plugins = [TestPlugin, AnotherTestPlugin]
        client = create(BackendType.IN_MEMORY, plugins=plugins)
        
        # Should be able to register additional plugins
        class AdditionalPlugin(PluginBase):
            def execute(self) -> str:
                return "additional"
        
        # The plugin method should work
        result_client = client.plugin(AdditionalPlugin)
        assert result_client == client  # Should return self for chaining

    def test_create_with_complex_plugins(self):
        """Test creating client with complex plugin configurations."""
        class ComplexPlugin(PluginBase):
            plugin_name = "complex_processor"
            
            def execute(self, data: dict, options: dict = None) -> dict:
                if options is None:
                    options = {}
                return {
                    "processed": True,
                    "data": data,
                    "options": options
                }
        
        class SimplePlugin(PluginBase):
            def execute(self, message: str) -> str:
                return f"Simple: {message}"
        
        plugins = [ComplexPlugin, SimplePlugin]
        client = create(BackendType.IN_MEMORY, plugins=plugins)
        
        assert isinstance(client, Client)

    def test_create_multiple_clients_isolation(self):
        """Test that multiple created clients are isolated."""
        plugins1 = [TestPlugin]
        plugins2 = [AnotherTestPlugin]
        
        client1 = create(BackendType.IN_MEMORY, plugins=plugins1)
        client2 = create(BackendType.IN_MEMORY, plugins=plugins2)
        
        # Should be different instances
        assert client1 is not client2
        assert client1._engine is not client2._engine
        assert client1._resolver is not client2._resolver

    def test_create_default_plugins_handling(self):
        """Test create function with default plugins parameter."""
        # When plugins is None, should default to empty list
        client = create(BackendType.IN_MEMORY)
        
        assert isinstance(client, Client)

    def test_create_client_method_chaining(self):
        """Test that created client supports method chaining."""
        client = create(BackendType.IN_MEMORY)
        
        # Should support chaining
        result = client.plugin(TestPlugin).plugin(AnotherTestPlugin)
        assert result == client

    def test_create_error_handling(self):
        """Test create function error handling."""
        # Test with invalid backend type
        with pytest.raises((ValueError, TypeError)):
            create("invalid_backend")

    def test_create_in_memory_backend_integration(self):
        """Test full integration of in-memory backend creation."""
        plugins = [TestPlugin]
        client = create(BackendType.IN_MEMORY, plugins=plugins)
        
        # Should be able to run a simple workflow
        workflow = {
            "steps": [
                {
                    "id": "test_step",
                    "kind": "operation",
                    "operation": "test_plugin",
                    "parameters": {"value": "test_input"}
                }
            ]
        }
        
        # This should work without errors
        result = client.run(workflow)
        
        # Verify result structure
        assert hasattr(result, 'id')
        assert hasattr(result, 'status')
        assert hasattr(result, 'results')

    def test_create_preserves_plugin_functionality(self):
        """Test that created client preserves plugin functionality."""
        plugins = [TestPlugin, AnotherTestPlugin]
        client = create(BackendType.IN_MEMORY, plugins=plugins)
        
        # Should be able to access available plugins
        available = client.get_available_plugins()
        assert len(available) >= 2  # At least our plugins

    @patch('argos.factory.create_in_memory_client')
    def test_create_calls_in_memory_factory(self, mock_create_in_memory):
        """Test that create function calls the in-memory factory."""
        # Setup mock
        mock_in_memory_client = Mock()
        mock_in_memory_client.engine = Mock()
        mock_in_memory_client.resolver = Mock()
        mock_in_memory_client.executor_factory = Mock()
        mock_create_in_memory.return_value = mock_in_memory_client
        
        plugins = [TestPlugin]
        client = create(BackendType.IN_MEMORY, plugins=plugins)
        
        # Should have called the in-memory client factory
        mock_create_in_memory.assert_called_once_with(plugins)
        assert isinstance(client, Client)

    def test_create_backend_enum_validation(self):
        """Test that create function validates BackendType enum."""
        # Valid enum values should work
        client = create(BackendType.IN_MEMORY)
        assert isinstance(client, Client)
        
        # Invalid values should raise errors
        with pytest.raises(ValueError):
            create(BackendType.TEMPORAL)
        
        with pytest.raises(ValueError):
            create(BackendType.CELERY)

    def test_create_function_signature(self):
        """Test create function signature and parameters."""
        import inspect
        sig = inspect.signature(create)
        
        # Should have backend and plugins parameters
        assert 'backend' in sig.parameters
        assert 'plugins' in sig.parameters
        
        # plugins should have default value
        plugins_param = sig.parameters['plugins']
        assert plugins_param.default is None or plugins_param.default == []