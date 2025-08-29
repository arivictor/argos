"""
Tests for domain port (PluginBase).

This module tests the PluginBase class including:
- Plugin registration mechanism
- Execute method enforcement
- Subclass validation
"""

import pytest

from aroflow.domain.port import PluginBase


class TestPluginBase:
    """Test cases for PluginBase."""

    def setup_method(self):
        """Clear the plugins registry before each test."""
        PluginBase._plugins.clear()

    def teardown_method(self):
        """Clear the plugins registry after each test."""
        PluginBase._plugins.clear()

    def test_plugin_registration(self):
        """Test that plugins are automatically registered."""
        assert len(PluginBase._plugins) == 0

        class TestPlugin(PluginBase):
            def execute(self, param: str) -> str:
                return f"processed: {param}"

        assert len(PluginBase._plugins) == 1
        assert TestPlugin in PluginBase._plugins

    def test_multiple_plugin_registration(self):
        """Test that multiple plugins are registered."""

        class Plugin1(PluginBase):
            def execute(self) -> str:
                return "plugin1"

        class Plugin2(PluginBase):
            def execute(self) -> str:
                return "plugin2"

        assert len(PluginBase._plugins) == 2
        assert Plugin1 in PluginBase._plugins
        assert Plugin2 in PluginBase._plugins

    def test_plugin_without_execute_method_fails(self):
        """Test that plugin without execute method raises TypeError."""
        with pytest.raises(TypeError, match="must define a 'execute' method"):

            class InvalidPlugin(PluginBase):
                def some_other_method(self):
                    pass

    def test_plugin_with_invalid_execute_method_name_fails(self):
        """Test that plugin with wrong method name fails."""
        with pytest.raises(TypeError, match="must define a 'execute' method"):

            class InvalidPlugin(PluginBase):
                def excute(self):  # Typo in method name
                    pass

    def test_plugin_execute_method_not_implemented_raises(self):
        """Test that calling execute on base class raises NotImplementedError."""

        class ValidPlugin(PluginBase):
            def execute(self):
                # Call parent execute to get NotImplementedError
                return super().execute()

        plugin = ValidPlugin()
        with pytest.raises(NotImplementedError, match="Plugins must implement the execute method"):
            plugin.execute()

    def test_plugin_execute_with_custom_implementation(self):
        """Test plugin with custom execute implementation."""

        class CustomPlugin(PluginBase):
            def execute(self, value: int) -> int:
                return value * 2

        plugin = CustomPlugin()
        result = plugin.execute(5)
        assert result == 10

    def test_plugin_execute_with_various_signatures(self):
        """Test plugins with different execute method signatures."""

        class NoArgsPlugin(PluginBase):
            def execute(self) -> str:
                return "no args"

        class SingleArgPlugin(PluginBase):
            def execute(self, value: str) -> str:
                return f"single: {value}"

        class MultiArgsPlugin(PluginBase):
            def execute(self, a: int, b: int) -> int:
                return a + b

        class KwargsPlugin(PluginBase):
            def execute(self, **kwargs) -> dict:
                return kwargs

        # All should be registered successfully
        assert len(PluginBase._plugins) == 4

        # Test execution
        no_args = NoArgsPlugin()
        assert no_args.execute() == "no args"

        single_arg = SingleArgPlugin()
        assert single_arg.execute("test") == "single: test"

        multi_args = MultiArgsPlugin()
        assert multi_args.execute(3, 7) == 10

        kwargs_plugin = KwargsPlugin()
        result = kwargs_plugin.execute(key1="value1", key2="value2")
        assert result == {"key1": "value1", "key2": "value2"}

    def test_plugin_inheritance_chain(self):
        """Test plugin inheritance with multiple levels."""

        class BasePlugin(PluginBase):
            def execute(self) -> str:
                return "base"

        class DerivedPlugin(BasePlugin):
            def execute(self) -> str:
                return "derived"

        class FurtherDerivedPlugin(DerivedPlugin):
            def execute(self) -> str:
                return "further derived"

        # All should be registered
        assert len(PluginBase._plugins) == 3
        assert BasePlugin in PluginBase._plugins
        assert DerivedPlugin in PluginBase._plugins
        assert FurtherDerivedPlugin in PluginBase._plugins

        # Test execution at each level
        base = BasePlugin()
        derived = DerivedPlugin()
        further = FurtherDerivedPlugin()

        assert base.execute() == "base"
        assert derived.execute() == "derived"
        assert further.execute() == "further derived"

    def test_plugin_overriding_execute_method(self):
        """Test plugin that overrides parent's execute method."""

        class ParentPlugin(PluginBase):
            def execute(self, value: str) -> str:
                return f"parent: {value}"

        class ChildPlugin(ParentPlugin):
            def execute(self, value: str) -> str:
                parent_result = super().execute(value)
                return f"child overrides ({parent_result})"

        child = ChildPlugin()
        result = child.execute("test")
        assert result == "child overrides (parent: test)"

    def test_plugin_with_additional_methods(self):
        """Test plugin with additional methods beyond execute."""

        class ExtendedPlugin(PluginBase):
            def execute(self, value: str) -> str:
                return self.process(value)

            def process(self, value: str) -> str:
                return f"processed: {value}"

            def helper_method(self) -> str:
                return "helper"

        plugin = ExtendedPlugin()
        assert plugin.execute("test") == "processed: test"
        assert plugin.helper_method() == "helper"

    def test_plugin_registry_persistence(self):
        """Test that plugin registry persists across instantiation."""

        class PersistentPlugin(PluginBase):
            def execute(self) -> str:
                return "persistent"

        # Create multiple instances
        instance1 = PersistentPlugin()
        instance2 = PersistentPlugin()

        # Registry should still contain the class, not instances
        assert len(PluginBase._plugins) == 1
        assert PersistentPlugin in PluginBase._plugins

        # Both instances should work
        assert instance1.execute() == "persistent"
        assert instance2.execute() == "persistent"

    def test_plugin_registry_is_class_list(self):
        """Test that plugin registry contains classes, not instances."""

        class ClassTestPlugin(PluginBase):
            def execute(self) -> str:
                return "class test"

        # Registry should contain the class
        assert len(PluginBase._plugins) == 1
        assert ClassTestPlugin in PluginBase._plugins

        # Registry should not contain instances
        instance = ClassTestPlugin()
        assert instance not in PluginBase._plugins

    def test_abstract_execute_method_in_base_class(self):
        """Test that PluginBase.execute is properly abstract."""
        base = PluginBase()

        # Should raise NotImplementedError when called directly
        with pytest.raises(NotImplementedError):
            base.execute()

        # Also test with arguments
        with pytest.raises(NotImplementedError):
            base.execute("arg1", "arg2")

        # And with keyword arguments
        with pytest.raises(NotImplementedError):
            base.execute(key="value")

    def test_plugin_with_complex_execute_signature(self):
        """Test plugin with complex execute method signature."""

        class ComplexPlugin(PluginBase):
            def execute(self, required_arg: str, optional_arg: int = 42, *args, **kwargs) -> dict:
                return {"required": required_arg, "optional": optional_arg, "args": args, "kwargs": kwargs}

        plugin = ComplexPlugin()

        # Test with required arg only
        result1 = plugin.execute("test")
        assert result1["required"] == "test"
        assert result1["optional"] == 42
        assert result1["args"] == ()
        assert result1["kwargs"] == {}

        # Test with all types of arguments
        result2 = plugin.execute("test", 100, "extra1", "extra2", key1="value1", key2="value2")
        assert result2["required"] == "test"
        assert result2["optional"] == 100
        assert result2["args"] == ("extra1", "extra2")
        assert result2["kwargs"] == {"key1": "value1", "key2": "value2"}
