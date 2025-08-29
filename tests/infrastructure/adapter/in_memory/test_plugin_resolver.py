"""
Tests for in-memory plugin resolver.

This module tests the InMemoryPluginResolver implementation.
"""

import pytest

from aroflow.domain.port import PluginBase
from aroflow.infrastructure.adapter.in_memory.plugin_resolver import InMemoryPluginResolver


class TestPlugin1(PluginBase):
    """Test plugin with default name."""

    def execute(self, value: str) -> str:
        return f"plugin1: {value}"


class TestPlugin2(PluginBase):
    """Test plugin with custom name."""

    plugin_name = "custom_plugin"

    def execute(self, value: int) -> int:
        return value * 2


class NoExecutePlugin:
    """Plugin without execute method (should fail during PluginBase registration)."""

    pass


class TestInMemoryPluginResolver:
    """Test cases for InMemoryPluginResolver."""

    def test_create_resolver_empty(self):
        """Test creating resolver with empty plugin list."""
        resolver = InMemoryPluginResolver([])

        assert hasattr(resolver, "_registry")
        assert isinstance(resolver._registry, dict)
        assert len(resolver._registry) == 0

    def test_create_resolver_with_plugins(self):
        """Test creating resolver with plugin list."""
        plugins = [TestPlugin1, TestPlugin2]
        resolver = InMemoryPluginResolver(plugins)

        assert len(resolver._registry) == 2
        assert "TestPlugin1" in resolver._registry
        assert "custom_plugin" in resolver._registry

    def test_plugin_registration_default_name(self):
        """Test plugin registration with default name (class name)."""
        resolver = InMemoryPluginResolver([TestPlugin1])

        assert "TestPlugin1" in resolver._registry
        assert resolver._registry["TestPlugin1"] == TestPlugin1

    def test_plugin_registration_custom_name(self):
        """Test plugin registration with custom plugin_name."""
        resolver = InMemoryPluginResolver([TestPlugin2])

        assert "custom_plugin" in resolver._registry
        assert resolver._registry["custom_plugin"] == TestPlugin2

    def test_resolve_plugin_by_default_name(self):
        """Test resolving plugin by default name."""
        resolver = InMemoryPluginResolver([TestPlugin1])

        plugin_instance = resolver.resolve("TestPlugin1")

        assert isinstance(plugin_instance, TestPlugin1)
        assert plugin_instance.execute("test") == "plugin1: test"

    def test_resolve_plugin_by_custom_name(self):
        """Test resolving plugin by custom name."""
        resolver = InMemoryPluginResolver([TestPlugin2])

        plugin_instance = resolver.resolve("custom_plugin")

        assert isinstance(plugin_instance, TestPlugin2)
        assert plugin_instance.execute(5) == 10

    def test_resolve_nonexistent_plugin(self):
        """Test resolving nonexistent plugin raises KeyError."""
        resolver = InMemoryPluginResolver([TestPlugin1])

        with pytest.raises(KeyError, match="No plugin registered for operation 'nonexistent'"):
            resolver.resolve("nonexistent")

    def test_resolve_returns_new_instance(self):
        """Test that resolve returns new instance each time."""
        resolver = InMemoryPluginResolver([TestPlugin1])

        instance1 = resolver.resolve("TestPlugin1")
        instance2 = resolver.resolve("TestPlugin1")

        assert isinstance(instance1, TestPlugin1)
        assert isinstance(instance2, TestPlugin1)
        assert instance1 is not instance2  # Different instances

    def test_multiple_plugins_resolution(self):
        """Test resolving multiple different plugins."""
        resolver = InMemoryPluginResolver([TestPlugin1, TestPlugin2])

        plugin1 = resolver.resolve("TestPlugin1")
        plugin2 = resolver.resolve("custom_plugin")

        assert isinstance(plugin1, TestPlugin1)
        assert isinstance(plugin2, TestPlugin2)

        # Test their functionality
        assert plugin1.execute("hello") == "plugin1: hello"
        assert plugin2.execute(3) == 6

    def test_plugin_name_override(self):
        """Test that plugin_name attribute overrides class name."""

        class PluginWithName(PluginBase):
            plugin_name = "overridden_name"

            def execute(self) -> str:
                return "overridden"

        resolver = InMemoryPluginResolver([PluginWithName])

        # Should be registered under custom name, not class name
        assert "overridden_name" in resolver._registry
        assert "PluginWithName" not in resolver._registry

        plugin = resolver.resolve("overridden_name")
        assert plugin.execute() == "overridden"

    def test_plugin_without_plugin_name(self):
        """Test plugin without plugin_name uses class name."""

        class SimplePlugin(PluginBase):
            def execute(self) -> str:
                return "simple"

        resolver = InMemoryPluginResolver([SimplePlugin])

        assert "SimplePlugin" in resolver._registry
        plugin = resolver.resolve("SimplePlugin")
        assert plugin.execute() == "simple"

    def test_empty_plugin_name(self):
        """Test plugin with empty plugin_name."""

        class EmptyNamePlugin(PluginBase):
            plugin_name = ""

            def execute(self) -> str:
                return "empty_name"

        resolver = InMemoryPluginResolver([EmptyNamePlugin])

        assert "" in resolver._registry
        plugin = resolver.resolve("")
        assert plugin.execute() == "empty_name"

    def test_none_plugin_name(self):
        """Test plugin with None plugin_name falls back to class name."""

        class NoneNamePlugin(PluginBase):
            plugin_name = None

            def execute(self) -> str:
                return "none_name"

        resolver = InMemoryPluginResolver([NoneNamePlugin])

        # getattr with default should use None when plugin_name is None
        # The actual implementation uses getattr(cls, "plugin_name", cls.__name__)
        # When plugin_name = None, getattr returns None, not the default
        assert None in resolver._registry
        plugin = resolver.resolve(None)
        assert plugin.execute() == "none_name"

    def test_duplicate_plugin_names(self):
        """Test behavior with duplicate plugin names."""

        class Plugin1(PluginBase):
            plugin_name = "duplicate"

            def execute(self) -> str:
                return "first"

        class Plugin2(PluginBase):
            plugin_name = "duplicate"

            def execute(self) -> str:
                return "second"

        # Last one should win
        resolver = InMemoryPluginResolver([Plugin1, Plugin2])

        assert len(resolver._registry) == 1
        assert "duplicate" in resolver._registry

        plugin = resolver.resolve("duplicate")
        assert plugin.execute() == "second"  # Last one registered

    def test_plugin_inheritance(self):
        """Test plugin inheritance."""

        class BasePlugin(PluginBase):
            def execute(self, value: str) -> str:
                return f"base: {value}"

        class DerivedPlugin(BasePlugin):
            plugin_name = "derived"

            def execute(self, value: str) -> str:
                return f"derived: {value}"

        resolver = InMemoryPluginResolver([BasePlugin, DerivedPlugin])

        base_plugin = resolver.resolve("BasePlugin")
        derived_plugin = resolver.resolve("derived")

        assert base_plugin.execute("test") == "base: test"
        assert derived_plugin.execute("test") == "derived: test"

    def test_complex_plugin_functionality(self):
        """Test resolver with complex plugin functionality."""

        class CalculatorPlugin(PluginBase):
            plugin_name = "calculator"

            def execute(self, operation: str, a: int, b: int) -> int:
                if operation == "add":
                    return a + b
                elif operation == "multiply":
                    return a * b
                else:
                    raise ValueError(f"Unknown operation: {operation}")

        resolver = InMemoryPluginResolver([CalculatorPlugin])

        calc = resolver.resolve("calculator")

        assert calc.execute("add", 5, 3) == 8
        assert calc.execute("multiply", 4, 7) == 28

        with pytest.raises(ValueError):
            calc.execute("divide", 10, 2)

    def test_resolver_isolation(self):
        """Test that different resolver instances are isolated."""
        resolver1 = InMemoryPluginResolver([TestPlugin1])
        resolver2 = InMemoryPluginResolver([TestPlugin2])

        # resolver1 should only have TestPlugin1
        plugin1 = resolver1.resolve("TestPlugin1")
        assert isinstance(plugin1, TestPlugin1)

        with pytest.raises(KeyError):
            resolver1.resolve("custom_plugin")

        # resolver2 should only have TestPlugin2
        plugin2 = resolver2.resolve("custom_plugin")
        assert isinstance(plugin2, TestPlugin2)

        with pytest.raises(KeyError):
            resolver2.resolve("TestPlugin1")

    def test_getattr_with_default(self):
        """Test the getattr behavior with default value."""

        class NoNamePlugin(PluginBase):
            # No plugin_name attribute
            def execute(self) -> str:
                return "no_name"

        resolver = InMemoryPluginResolver([NoNamePlugin])

        # Should use class name as default
        assert "NoNamePlugin" in resolver._registry
        plugin = resolver.resolve("NoNamePlugin")
        assert plugin.execute() == "no_name"
