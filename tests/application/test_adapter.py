"""
Tests for application adapters.

This module tests the application layer adapters including:
- ExecutionContext
- ParameterBinder
- VariableResolver
- Various executor classes
"""

from typing import Any
from unittest.mock import Mock

import pytest

from argos.application.adapter import (
    ExecutionContext,
    MapStrategyFactory,
    OperationExecutor,
    ParallelMapStrategy,
    ParameterBinder,
    SequentialMapStrategy,
    VariableResolver,
)
from argos.application.port import ResultStore
from argos.domain.entity import MapStep, OperationStep
from argos.domain.port import PluginBase
from argos.domain.value_object import ExecutionOptions


class TestExecutionContext:
    """Test cases for ExecutionContext."""

    def test_create_execution_context(self):
        """Test creating an execution context."""
        result_store = Mock(spec=ResultStore)
        context = ExecutionContext(result_store)

        assert context.results == result_store

    def test_get_result_calls_store(self):
        """Test that get_result calls the result store."""
        result_store = Mock(spec=ResultStore)
        result_store.get.return_value = "test_result"

        context = ExecutionContext(result_store)
        result = context.get_result("step_id")

        result_store.get.assert_called_once_with("step_id")
        assert result == "test_result"

    def test_get_result_with_different_step_ids(self):
        """Test getting results for different step IDs."""
        result_store = Mock(spec=ResultStore)
        result_store.get.side_effect = lambda step_id: f"result_for_{step_id}"

        context = ExecutionContext(result_store)

        result1 = context.get_result("step1")
        result2 = context.get_result("step2")

        assert result1 == "result_for_step1"
        assert result2 == "result_for_step2"


class TestParameterBinder:
    """Test cases for ParameterBinder."""

    def setup_method(self):
        """Setup test fixtures."""
        self.binder = ParameterBinder()

    def test_bind_simple_plugin(self):
        """Test binding parameters to simple plugin."""

        class SimplePlugin(PluginBase):
            def execute(self, name: str, age: int) -> str:
                return f"{name} is {age}"

        plugin = SimplePlugin()
        params = {"name": "Alice", "age": "25"}

        bound = self.binder.bind(plugin, params)

        assert bound == {"name": "Alice", "age": 25}

    def test_bind_skips_self_parameter(self):
        """Test that self parameter is skipped."""

        class PluginWithSelf(PluginBase):
            def execute(self, value: str) -> str:
                return value

        plugin = PluginWithSelf()
        params = {"value": "test", "self": "should_be_ignored"}

        bound = self.binder.bind(plugin, params)

        assert bound == {"value": "test"}
        assert "self" not in bound

    def test_bind_missing_parameters(self):
        """Test binding when some parameters are missing."""

        class PluginWithMissingParams(PluginBase):
            def execute(self, required: str, missing: int) -> str:
                return required

        plugin = PluginWithMissingParams()
        params = {"required": "value"}  # missing 'missing' parameter

        bound = self.binder.bind(plugin, params)

        assert bound == {"required": "value"}
        assert "missing" not in bound

    def test_bind_no_type_hints(self):
        """Test binding plugin with no type hints."""

        class NoHintsPlugin(PluginBase):
            def execute(self, value):
                return value

        plugin = NoHintsPlugin()
        params = {"value": "test"}

        bound = self.binder.bind(plugin, params)

        assert bound == {"value": "test"}

    def test_coerce_string_to_int(self):
        """Test coercing string to int."""
        result = self.binder._coerce("42", int)
        assert result == 42
        assert isinstance(result, int)

    def test_coerce_string_to_float(self):
        """Test coercing string to float."""
        result = self.binder._coerce("3.14", float)
        assert result == 3.14
        assert isinstance(result, float)

    def test_coerce_string_to_bool_true_values(self):
        """Test coercing string to bool (true values)."""
        true_values = ["true", "True", "TRUE", "1", "yes", "YES", "y", "Y"]

        for value in true_values:
            result = self.binder._coerce(value, bool)
            assert result is True, f"Failed for value: {value}"

    def test_coerce_string_to_bool_false_values(self):
        """Test coercing string to bool (false values)."""
        false_values = ["false", "False", "FALSE", "0", "no", "NO", "n", "N"]

        for value in false_values:
            result = self.binder._coerce(value, bool)
            assert result is False, f"Failed for value: {value}"

    def test_coerce_string_to_bool_other_values(self):
        """Test coercing string to bool (other values remain as string)."""
        other_values = ["maybe", "unknown", "2", "yes please"]

        for value in other_values:
            result = self.binder._coerce(value, bool)
            assert result == value, f"Failed for value: {value}"

    def test_coerce_already_correct_type(self):
        """Test coercing value that's already the correct type."""
        assert self.binder._coerce(42, int) == 42
        assert self.binder._coerce(3.14, float) == 3.14
        assert self.binder._coerce(True, bool) is True
        assert self.binder._coerce("test", str) == "test"

    def test_coerce_any_type(self):
        """Test coercing to Any type."""
        assert self.binder._coerce("test", Any) == "test"
        assert self.binder._coerce(42, Any) == 42
        assert self.binder._coerce([1, 2, 3], Any) == [1, 2, 3]

    def test_coerce_optional_type(self):
        """Test coercing to Optional type."""
        result = self.binder._coerce("42", int | None)
        assert result == 42
        assert isinstance(result, int)

    def test_coerce_union_type(self):
        """Test coercing to Union type."""
        # Should try int first and succeed
        result = self.binder._coerce("42", int | str)
        assert result == 42
        assert isinstance(result, int)

        # Should try int, fail, then try str and succeed
        result = self.binder._coerce("not_a_number", int | str)
        assert result == "not_a_number"
        assert isinstance(result, str)

    def test_coerce_union_type_no_match(self):
        """Test coercing to Union type with no matching types."""
        # None of the union types will match, should return original
        result = self.binder._coerce("test", int | float)
        assert result == "test"

    def test_coerce_non_string_value(self):
        """Test coercing non-string value."""
        # Should return as-is for non-string values when no type match
        result = self.binder._coerce([1, 2, 3], int)
        assert result == [1, 2, 3]

    def test_bind_complex_plugin(self):
        """Test binding complex plugin with various types."""

        class ComplexPlugin(PluginBase):
            def execute(
                self,
                name: str,
                age: int,
                height: float,
                active: bool,
                optional_param: str | None = None,
                union_param: int | str = "default",
            ) -> dict:
                return {
                    "name": name,
                    "age": age,
                    "height": height,
                    "active": active,
                    "optional": optional_param,
                    "union": union_param,
                }

        plugin = ComplexPlugin()
        params = {
            "name": "Bob",
            "age": "30",
            "height": "5.9",
            "active": "true",
            "optional_param": "test",
            "union_param": "42",
        }

        bound = self.binder.bind(plugin, params)

        assert bound["name"] == "Bob"
        assert bound["age"] == 30
        assert bound["height"] == 5.9
        assert bound["active"] is True
        assert bound["optional_param"] == "test"
        assert bound["union_param"] == 42

    def test_coerce_invalid_int_conversion(self):
        """Test coercing invalid string to int raises exception."""
        with pytest.raises(ValueError):
            self.binder._coerce("not_a_number", int)

    def test_coerce_invalid_float_conversion(self):
        """Test coercing invalid string to float raises exception."""
        with pytest.raises(ValueError):
            self.binder._coerce("not_a_float", float)


class TestVariableResolver:
    """Test cases for VariableResolver."""

    def setup_method(self):
        """Setup test fixtures."""
        self.context = Mock()
        self.resolver = VariableResolver(self.context)

    def test_resolve_simple_string(self):
        """Test resolving string without placeholders."""
        result = self.resolver.resolve_any("simple string")
        assert result == "simple string"

    def test_resolve_exact_placeholder_match(self):
        """Test resolving exact placeholder match returns raw value."""
        self.context.get_result.return_value = {"key": "value"}

        result = self.resolver.resolve_any("${step1}")

        self.context.get_result.assert_called_once_with("step1")
        assert result == {"key": "value"}

    def test_resolve_placeholder_in_string(self):
        """Test resolving placeholder within string."""
        self.context.get_result.return_value = "world"

        result = self.resolver.resolve_any("Hello ${step1}!")

        self.context.get_result.assert_called_once_with("step1")
        assert result == "Hello world!"

    def test_resolve_multiple_placeholders(self):
        """Test resolving multiple placeholders in string."""
        self.context.get_result.side_effect = lambda step_id: {"step1": "Hello", "step2": "world"}[step_id]

        result = self.resolver.resolve_any("${step1} ${step2}!")

        assert result == "Hello world!"

    def test_resolve_nested_dict(self):
        """Test resolving nested dictionary."""
        self.context.get_result.return_value = "resolved_value"

        data = {"key1": "${step1}", "nested": {"key2": "prefix_${step1}_suffix", "key3": "no_placeholder"}}

        result = self.resolver.resolve_any(data)

        expected = {
            "key1": "resolved_value",
            "nested": {"key2": "prefix_resolved_value_suffix", "key3": "no_placeholder"},
        }
        assert result == expected

    def test_resolve_nested_list(self):
        """Test resolving nested list."""
        self.context.get_result.return_value = "resolved"

        data = ["${step1}", "static", ["nested_${step1}", "static"]]

        result = self.resolver.resolve_any(data)

        expected = ["resolved", "static", ["nested_resolved", "static"]]
        assert result == expected

    def test_resolve_complex_nested_structure(self):
        """Test resolving complex nested data structure."""
        self.context.get_result.return_value = "value"

        data = {
            "list": ["${step1}", {"nested": "${step1}"}],
            "dict": {"key": "${step1}", "list": ["item1", "${step1}"]},
            "exact": "${step1}",
        }

        result = self.resolver.resolve_any(data)

        expected = {
            "list": ["value", {"nested": "value"}],
            "dict": {"key": "value", "list": ["item1", "value"]},
            "exact": "value",
        }
        assert result == expected

    def test_resolve_field_access(self):
        """Test resolving field access in placeholders."""
        step_result = {"result": "field_value"}
        self.context.get_result.return_value = step_result

        # Mock msgspec.to_builtins
        import msgspec

        original_to_builtins = msgspec.to_builtins
        msgspec.to_builtins = Mock(return_value=step_result)

        try:
            result = self.resolver._lookup_token("step1.result")
            assert result == "field_value"
        finally:
            msgspec.to_builtins = original_to_builtins

    def test_resolve_array_access(self):
        """Test resolving array access in placeholders."""
        step_result = {"results": ["item1", "item2", "item3"]}
        self.context.get_result.return_value = step_result

        import msgspec

        original_to_builtins = msgspec.to_builtins
        msgspec.to_builtins = Mock(return_value=step_result)

        try:
            result = self.resolver._lookup_token("step1.results[1]")
            assert result == "item2"
        finally:
            msgspec.to_builtins = original_to_builtins

    def test_resolve_unknown_step_id(self):
        """Test resolving unknown step ID raises KeyError."""
        self.context.get_result.side_effect = KeyError("Unknown step")

        with pytest.raises(KeyError, match="Unknown placeholder: unknown_step"):
            self.resolver._lookup_token("unknown_step")

    def test_resolve_non_string_values(self):
        """Test resolving non-string values."""
        assert self.resolver.resolve_any(42) == 42
        assert self.resolver.resolve_any(3.14) == 3.14
        assert self.resolver.resolve_any(True) is True
        assert self.resolver.resolve_any(None) is None

    def test_pattern_regex(self):
        """Test the placeholder pattern regex."""
        pattern = self.resolver._pattern

        # Should match placeholders
        assert pattern.search("${step1}")
        assert pattern.search("text ${step1} more")
        assert pattern.search("${step1.field}")
        assert pattern.search("${step1[0]}")

        # Should not match invalid patterns
        assert not pattern.search("$step1}")
        assert not pattern.search("${}")
        assert not pattern.search("no_placeholder")


class TestOperationExecutor:
    """Test cases for OperationExecutor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = Mock()
        self.binder = Mock()
        self.values = Mock()
        self.task_runner = Mock()
        self.execution_options = ExecutionOptions()

        self.executor = OperationExecutor(
            self.resolver, self.binder, self.values, self.task_runner, self.execution_options
        )

    def test_create_operation_executor(self):
        """Test creating operation executor."""
        assert self.executor.resolver == self.resolver
        assert self.executor.binder == self.binder
        assert self.executor.values == self.values
        assert self.executor.task_runner == self.task_runner
        assert self.executor.execution_options == self.execution_options

    def test_execute_operation_step(self):
        """Test executing operation step."""
        # Setup mocks
        plugin = Mock()
        self.resolver.resolve.return_value = plugin

        resolved_params = {"param": "resolved_value"}
        self.values.resolve_any.return_value = resolved_params

        bound_params = {"param": "bound_value"}
        self.binder.bind.return_value = bound_params

        task_result = "task_result"
        self.task_runner.run.return_value = task_result

        # Create test step
        step = OperationStep(id="test_step", operation="test_op", parameters={"param": "value"})

        # Execute
        result = self.executor.execute(step)

        # Verify calls
        self.resolver.resolve.assert_called_once_with("test_op")
        self.values.resolve_any.assert_called_once_with({"param": "value"})
        self.binder.bind.assert_called_once_with(plugin, resolved_params)
        self.task_runner.run.assert_called_once_with(plugin, bound_params)

        # Verify result
        assert result is not None
        assert result.id == "test_step"
        assert result.kind == "operation"
        assert result.operation == "test_op"
        assert result.parameters == resolved_params
        assert result.result == task_result
        assert result.status == "success"
        assert result.error is None


class TestMapStrategyFactory:
    """Test cases for MapStrategyFactory."""

    def test_get_sequential_strategy(self):
        """Test getting sequential map strategy."""
        resolver = Mock()
        binder = Mock()
        values = Mock()
        task_runner = Mock()
        execution_options = ExecutionOptions()

        strategy = MapStrategyFactory.get_strategy(
            "sequential", resolver, binder, values, task_runner, execution_options
        )

        assert isinstance(strategy, SequentialMapStrategy)

    def test_get_parallel_strategy(self):
        """Test getting parallel map strategy."""
        resolver = Mock()
        binder = Mock()
        values = Mock()
        task_runner = Mock()
        execution_options = ExecutionOptions()

        strategy = MapStrategyFactory.get_strategy("parallel", resolver, binder, values, task_runner, execution_options)

        assert isinstance(strategy, ParallelMapStrategy)

    def test_get_default_strategy(self):
        """Test getting default strategy for unknown mode."""
        resolver = Mock()
        binder = Mock()
        values = Mock()
        task_runner = Mock()
        execution_options = ExecutionOptions()

        strategy = MapStrategyFactory.get_strategy(
            "unknown_mode", resolver, binder, values, task_runner, execution_options
        )

        # Should default to sequential
        assert isinstance(strategy, SequentialMapStrategy)


class MockPlugin(PluginBase):
    """Mock plugin for testing."""

    def execute(self, item: str) -> str:
        return f"processed_{item}"


class TestSequentialMapStrategy:
    """Test cases for SequentialMapStrategy."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = Mock()
        self.binder = Mock()
        self.values = Mock()
        self.task_runner = Mock()
        self.execution_options = ExecutionOptions()

        self.strategy = SequentialMapStrategy(
            self.resolver, self.binder, self.values, self.task_runner, self.execution_options
        )

    def test_execute_map_step(self):
        """Test executing map step sequentially."""
        # Setup mocks
        plugin = MockPlugin()
        self.resolver.resolve.return_value = plugin

        # Mock parameter resolution and binding
        self.values.resolve_any.side_effect = lambda x: x  # Pass through
        self.binder.bind.return_value = {"item": "test_item"}

        # Mock task runner
        self.task_runner.run.side_effect = ["result1", "result2"]

        # Create test step
        operation = OperationStep(id="map_op", operation="test_operation", parameters={"param": "value"})

        step = MapStep(id="map_step", inputs=["input1", "input2"], iterator="item", operation=operation)

        # Execute
        result = self.strategy.execute(step)

        # Verify result structure
        assert result.id == "map_step"
        assert result.kind == "map"
        assert result.mode == "sequential"
        assert result.iterator == "item"
        assert result.inputs == ["input1", "input2"]
        assert len(result.results) == 2


# Note: This is a subset of the comprehensive tests. The full implementation would include:
# - More detailed tests for ParallelMapStrategy
# - Tests for MapExecutor
# - Tests for ParallelOperationExecutor
# - Error handling tests
# - Performance tests
# - Integration tests between components
