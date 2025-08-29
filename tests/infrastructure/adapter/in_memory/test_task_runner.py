"""
Tests for in-memory task runner.

This module tests the InMemoryTaskRunner implementation.
"""

import pytest

from aroflow.domain.port import PluginBase
from aroflow.infrastructure.adapter.in_memory.task_runner import InMemoryTaskRunner


class SimplePlugin(PluginBase):
    """Simple test plugin."""

    def execute(self, value: str) -> str:
        return f"processed: {value}"


class MathPlugin(PluginBase):
    """Math operations plugin."""

    def execute(self, operation: str, a: int, b: int) -> int:
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        elif operation == "subtract":
            return a - b
        else:
            raise ValueError(f"Unknown operation: {operation}")


class NoArgsPlugin(PluginBase):
    """Plugin with no arguments."""

    def execute(self) -> str:
        return "no_args_result"


class FailingPlugin(PluginBase):
    """Plugin that raises an exception."""

    def execute(self, should_fail: bool = True) -> str:
        if should_fail:
            raise RuntimeError("Plugin execution failed")
        return "success"


class ComplexPlugin(PluginBase):
    """Plugin with complex parameter types."""

    def execute(self, data: dict, multiplier: int = 1, **kwargs) -> dict:
        result = {
            "processed_data": {k: v * multiplier for k, v in data.items() if isinstance(v, int | float)},
            "metadata": kwargs,
        }
        return result


class TestInMemoryTaskRunner:
    """Test cases for InMemoryTaskRunner."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = InMemoryTaskRunner()

    def test_create_task_runner(self):
        """Test creating in-memory task runner."""
        assert isinstance(self.runner, InMemoryTaskRunner)

    def test_run_simple_plugin(self):
        """Test running simple plugin."""
        plugin = SimplePlugin()
        bound_params = {"value": "test_input"}

        result = self.runner.run(plugin, bound_params)

        assert result == "processed: test_input"

    def test_run_math_plugin_add(self):
        """Test running math plugin with add operation."""
        plugin = MathPlugin()
        bound_params = {"operation": "add", "a": 5, "b": 3}

        result = self.runner.run(plugin, bound_params)

        assert result == 8

    def test_run_math_plugin_multiply(self):
        """Test running math plugin with multiply operation."""
        plugin = MathPlugin()
        bound_params = {"operation": "multiply", "a": 4, "b": 7}

        result = self.runner.run(plugin, bound_params)

        assert result == 28

    def test_run_math_plugin_subtract(self):
        """Test running math plugin with subtract operation."""
        plugin = MathPlugin()
        bound_params = {"operation": "subtract", "a": 10, "b": 3}

        result = self.runner.run(plugin, bound_params)

        assert result == 7

    def test_run_no_args_plugin(self):
        """Test running plugin with no arguments."""
        plugin = NoArgsPlugin()
        bound_params = {}

        result = self.runner.run(plugin, bound_params)

        assert result == "no_args_result"

    def test_run_plugin_with_extra_params(self):
        """Test running plugin with extra parameters causes TypeError."""
        plugin = NoArgsPlugin()
        bound_params = {"extra_param": "ignored", "another": 42}

        # Should raise TypeError because plugin doesn't accept these parameters
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            self.runner.run(plugin, bound_params)

    def test_run_failing_plugin(self):
        """Test running plugin that raises exception."""
        plugin = FailingPlugin()
        bound_params = {"should_fail": True}

        with pytest.raises(RuntimeError, match="Plugin execution failed"):
            self.runner.run(plugin, bound_params)

    def test_run_failing_plugin_success_case(self):
        """Test running failing plugin in success case."""
        plugin = FailingPlugin()
        bound_params = {"should_fail": False}

        result = self.runner.run(plugin, bound_params)

        assert result == "success"

    def test_run_complex_plugin(self):
        """Test running plugin with complex parameters."""
        plugin = ComplexPlugin()
        bound_params = {
            "data": {"x": 10, "y": 20, "name": "test"},
            "multiplier": 3,
            "metadata": "extra_info",
            "version": "1.0",
        }

        result = self.runner.run(plugin, bound_params)

        expected = {
            "processed_data": {"x": 30, "y": 60},  # Only numeric values multiplied
            "metadata": {"metadata": "extra_info", "version": "1.0"},
        }
        assert result == expected

    def test_run_plugin_with_default_params(self):
        """Test running plugin that uses default parameters."""
        plugin = ComplexPlugin()
        bound_params = {
            "data": {"value": 5}
            # multiplier not provided, should use default of 1
        }

        result = self.runner.run(plugin, bound_params)

        expected = {
            "processed_data": {"value": 5},  # multiplier=1 (default)
            "metadata": {},
        }
        assert result == expected

    def test_run_different_plugins_same_runner(self):
        """Test running different plugins with same runner instance."""
        simple_plugin = SimplePlugin()
        math_plugin = MathPlugin()
        no_args_plugin = NoArgsPlugin()

        # Run different plugins
        result1 = self.runner.run(simple_plugin, {"value": "hello"})
        result2 = self.runner.run(math_plugin, {"operation": "add", "a": 1, "b": 2})
        result3 = self.runner.run(no_args_plugin, {})

        assert result1 == "processed: hello"
        assert result2 == 3
        assert result3 == "no_args_result"

    def test_run_same_plugin_multiple_times(self):
        """Test running same plugin multiple times."""
        plugin = SimplePlugin()

        result1 = self.runner.run(plugin, {"value": "first"})
        result2 = self.runner.run(plugin, {"value": "second"})
        result3 = self.runner.run(plugin, {"value": "third"})

        assert result1 == "processed: first"
        assert result2 == "processed: second"
        assert result3 == "processed: third"

    def test_run_plugin_with_none_values(self):
        """Test running plugin with None values."""

        class NoneHandlingPlugin(PluginBase):
            def execute(self, value, default="default") -> str:
                if value is None:
                    return default
                return str(value)

        plugin = NoneHandlingPlugin()

        result1 = self.runner.run(plugin, {"value": None})
        result2 = self.runner.run(plugin, {"value": None, "default": "custom_default"})
        result3 = self.runner.run(plugin, {"value": "not_none"})

        assert result1 == "default"
        assert result2 == "custom_default"
        assert result3 == "not_none"

    def test_run_plugin_with_empty_dict(self):
        """Test running plugin with empty bound parameters."""
        plugin = NoArgsPlugin()

        result = self.runner.run(plugin, {})

        assert result == "no_args_result"

    def test_run_plugin_return_types(self):
        """Test plugins that return different types."""

        class MultiTypePlugin(PluginBase):
            def execute(self, return_type: str):
                if return_type == "string":
                    return "string_result"
                elif return_type == "int":
                    return 42
                elif return_type == "float":
                    return 3.14
                elif return_type == "bool":
                    return True
                elif return_type == "list":
                    return [1, 2, 3]
                elif return_type == "dict":
                    return {"key": "value"}
                elif return_type == "none":
                    return None
                else:
                    raise ValueError("Unknown return type")

        plugin = MultiTypePlugin()

        assert self.runner.run(plugin, {"return_type": "string"}) == "string_result"
        assert self.runner.run(plugin, {"return_type": "int"}) == 42
        assert self.runner.run(plugin, {"return_type": "float"}) == 3.14
        assert self.runner.run(plugin, {"return_type": "bool"}) is True
        assert self.runner.run(plugin, {"return_type": "list"}) == [1, 2, 3]
        assert self.runner.run(plugin, {"return_type": "dict"}) == {"key": "value"}
        assert self.runner.run(plugin, {"return_type": "none"}) is None

    def test_run_plugin_with_type_errors(self):
        """Test running plugin with wrong parameter types."""
        plugin = MathPlugin()

        # Pass string instead of int
        with pytest.raises(TypeError):
            self.runner.run(plugin, {"operation": "add", "a": "not_int", "b": 3})

    def test_task_runner_isolation(self):
        """Test that task runner instances are isolated."""
        runner1 = InMemoryTaskRunner()
        runner2 = InMemoryTaskRunner()

        plugin1 = SimplePlugin()
        plugin2 = SimplePlugin()

        result1 = runner1.run(plugin1, {"value": "runner1"})
        result2 = runner2.run(plugin2, {"value": "runner2"})

        assert result1 == "processed: runner1"
        assert result2 == "processed: runner2"

    def test_plugin_state_isolation(self):
        """Test that plugin instances maintain their own state."""

        class StatefulPlugin(PluginBase):
            def __init__(self):
                self.counter = 0

            def execute(self, increment: int = 1) -> int:
                self.counter += increment
                return self.counter

        plugin1 = StatefulPlugin()
        plugin2 = StatefulPlugin()

        # Each plugin should maintain its own counter
        assert self.runner.run(plugin1, {"increment": 5}) == 5
        assert self.runner.run(plugin2, {"increment": 3}) == 3
        assert self.runner.run(plugin1, {"increment": 2}) == 7  # 5 + 2
        assert self.runner.run(plugin2, {"increment": 1}) == 4  # 3 + 1
