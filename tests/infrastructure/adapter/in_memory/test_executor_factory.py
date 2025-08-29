"""
Tests for in-memory executor factory.

This module tests the InMemoryExecutorFactory implementation.
"""

from unittest.mock import Mock

import pytest

from aroflow.application.adapter import (
    MapExecutor,
    OperationExecutor,
    ParallelOperationExecutor,
    ParameterBinder,
    PlaceholderResolver,
)
from aroflow.application.port import PluginResolver, TaskRunner
from aroflow.domain.entity import MapStep, OperationStep, ParallelStep
from aroflow.domain.value_object import ExecutionOptions
from aroflow.infrastructure.adapter.in_memory.executor_factory import InMemoryExecutorFactory


class TestInMemoryExecutorFactory:
    """Test cases for InMemoryExecutorFactory."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = Mock(spec=PluginResolver)
        self.binder = Mock(spec=ParameterBinder)
        self.values = Mock(spec=PlaceholderResolver)
        self.task_runner = Mock(spec=TaskRunner)
        self.execution_options = ExecutionOptions(retries=3, timeout=30.0)

        self.factory = InMemoryExecutorFactory(
            resolver=self.resolver,
            binder=self.binder,
            values=self.values,
            task_runner=self.task_runner,
            execution_options=self.execution_options,
        )

    def test_create_executor_factory(self):
        """Test creating executor factory."""
        assert self.factory.resolver == self.resolver
        assert self.factory.binder == self.binder
        assert self.factory.values == self.values
        assert self.factory.task_runner == self.task_runner
        assert self.factory.execution_options == self.execution_options

    def test_get_executor_for_operation_step(self):
        """Test getting executor for operation step."""
        step = OperationStep(id="test_step", operation="test_op", parameters={"param": "value"})

        executor = self.factory.get_executor(step)

        assert isinstance(executor, OperationExecutor)
        assert executor.resolver == self.resolver
        assert executor.binder == self.binder
        assert executor.values == self.values
        assert executor.task_runner == self.task_runner
        assert executor.execution_options == self.execution_options

    def test_get_executor_for_map_step(self):
        """Test getting executor for map step."""
        inner_operation = OperationStep(id="inner_op", operation="process", parameters={})

        step = MapStep(id="map_step", inputs=["a", "b", "c"], iterator="item", operation=inner_operation)

        executor = self.factory.get_executor(step)

        assert isinstance(executor, MapExecutor)
        assert executor.resolver == self.resolver
        assert executor.binder == self.binder
        assert executor.values == self.values
        assert executor.task_runner == self.task_runner
        assert executor.execution_options == self.execution_options

    def test_get_executor_for_parallel_step(self):
        """Test getting executor for parallel step."""
        operations = [
            OperationStep(id="op1", operation="task1", parameters={}),
            OperationStep(id="op2", operation="task2", parameters={}),
        ]

        step = ParallelStep(id="parallel_step", operations=operations)

        executor = self.factory.get_executor(step)

        assert isinstance(executor, ParallelOperationExecutor)
        assert executor.resolver == self.resolver
        assert executor.binder == self.binder
        assert executor.values == self.values
        assert executor.task_runner == self.task_runner
        assert executor.execution_options == self.execution_options

    def test_get_executor_for_unknown_step_type(self):
        """Test getting executor for unknown step type raises ValueError."""

        class UnknownStep:
            pass

        unknown_step = UnknownStep()

        with pytest.raises(ValueError, match="Unknown step type"):
            self.factory.get_executor(unknown_step)

    def test_get_executor_multiple_calls_same_type(self):
        """Test that multiple calls for same step type return new instances."""
        step1 = OperationStep(id="step1", operation="op1", parameters={})
        step2 = OperationStep(id="step2", operation="op2", parameters={})

        executor1 = self.factory.get_executor(step1)
        executor2 = self.factory.get_executor(step2)

        assert isinstance(executor1, OperationExecutor)
        assert isinstance(executor2, OperationExecutor)
        assert executor1 is not executor2  # Different instances

    def test_get_executor_different_step_types(self):
        """Test getting executors for different step types."""
        operation_step = OperationStep(id="op", operation="test", parameters={})

        inner_op = OperationStep(id="inner", operation="process", parameters={})
        map_step = MapStep(id="map", inputs=["a"], iterator="item", operation=inner_op)

        parallel_step = ParallelStep(id="parallel", operations=[operation_step])

        op_executor = self.factory.get_executor(operation_step)
        map_executor = self.factory.get_executor(map_step)
        parallel_executor = self.factory.get_executor(parallel_step)

        assert isinstance(op_executor, OperationExecutor)
        assert isinstance(map_executor, MapExecutor)
        assert isinstance(parallel_executor, ParallelOperationExecutor)

    def test_executor_factory_with_different_execution_options(self):
        """Test executor factory with different execution options."""
        custom_options = ExecutionOptions(retries=5, timeout=60.0)

        factory = InMemoryExecutorFactory(
            resolver=self.resolver,
            binder=self.binder,
            values=self.values,
            task_runner=self.task_runner,
            execution_options=custom_options,
        )

        step = OperationStep(id="test", operation="test_op", parameters={})
        executor = factory.get_executor(step)

        assert executor.execution_options == custom_options
        assert executor.execution_options.retries == 5
        assert executor.execution_options.timeout == 60.0

    def test_executor_factory_component_isolation(self):
        """Test that executor factory components are properly isolated."""
        resolver1 = Mock(spec=PluginResolver)
        resolver2 = Mock(spec=PluginResolver)

        factory1 = InMemoryExecutorFactory(
            resolver=resolver1,
            binder=self.binder,
            values=self.values,
            task_runner=self.task_runner,
            execution_options=self.execution_options,
        )

        factory2 = InMemoryExecutorFactory(
            resolver=resolver2,
            binder=self.binder,
            values=self.values,
            task_runner=self.task_runner,
            execution_options=self.execution_options,
        )

        step = OperationStep(id="test", operation="test_op", parameters={})

        executor1 = factory1.get_executor(step)
        executor2 = factory2.get_executor(step)

        assert executor1.resolver == resolver1
        assert executor2.resolver == resolver2
        assert executor1.resolver != executor2.resolver

    def test_executor_inheritance_from_step_executor(self):
        """Test that returned executors inherit from StepExecutor."""
        from aroflow.application.port import StepExecutor

        operation_step = OperationStep(id="op", operation="test", parameters={})
        map_step = MapStep(
            id="map",
            inputs=["a"],
            iterator="item",
            operation=OperationStep(id="inner", operation="process", parameters={}),
        )
        parallel_step = ParallelStep(
            id="parallel", operations=[OperationStep(id="pop", operation="test", parameters={})]
        )

        op_executor = self.factory.get_executor(operation_step)
        map_executor = self.factory.get_executor(map_step)
        parallel_executor = self.factory.get_executor(parallel_step)

        assert isinstance(op_executor, StepExecutor)
        assert isinstance(map_executor, StepExecutor)
        assert isinstance(parallel_executor, StepExecutor)

    def test_executor_factory_with_none_components(self):
        """Test executor factory behavior with None components."""
        # This should work since the components are just passed through
        factory = InMemoryExecutorFactory(
            resolver=None, binder=None, values=None, task_runner=None, execution_options=self.execution_options
        )

        step = OperationStep(id="test", operation="test_op", parameters={})
        executor = factory.get_executor(step)

        assert isinstance(executor, OperationExecutor)
        assert executor.resolver is None
        assert executor.binder is None
        assert executor.values is None
        assert executor.task_runner is None

    def test_complex_step_configurations(self):
        """Test factory with complex step configurations."""
        # Operation step with all options
        op_step = OperationStep(
            id="complex_op",
            operation="complex_operation",
            parameters={"complex": "params"},
            retries=5,
            timeout=120.0,
            fail_workflow=False,
        )

        # Map step with parallel mode
        map_step = MapStep(
            id="complex_map",
            inputs=[1, 2, 3, 4, 5],
            iterator="number",
            mode="parallel",
            operation=OperationStep(id="map_op", operation="process_number", parameters={}),
            retries=2,
            timeout=60.0,
            fail_workflow=True,
        )

        # Parallel step with multiple operations
        parallel_step = ParallelStep(
            id="complex_parallel",
            operations=[
                OperationStep(id="task1", operation="task_one", parameters={"p1": "v1"}),
                OperationStep(id="task2", operation="task_two", parameters={"p2": "v2"}),
                OperationStep(id="task3", operation="task_three", parameters={"p3": "v3"}),
            ],
            retries=1,
            timeout=90.0,
            fail_workflow=False,
        )

        # All should create appropriate executors
        op_executor = self.factory.get_executor(op_step)
        map_executor = self.factory.get_executor(map_step)
        parallel_executor = self.factory.get_executor(parallel_step)

        assert isinstance(op_executor, OperationExecutor)
        assert isinstance(map_executor, MapExecutor)
        assert isinstance(parallel_executor, ParallelOperationExecutor)

    def test_step_type_checking_edge_cases(self):
        """Test step type checking with edge cases."""

        # Test with object that has same attributes but wrong type
        class FakeOperationStep:
            def __init__(self):
                self.id = "fake"
                self.operation = "fake_op"
                self.parameters = {}

        fake_step = FakeOperationStep()

        # Should fail because isinstance check should be strict
        with pytest.raises(ValueError, match="Unknown step type"):
            self.factory.get_executor(fake_step)
