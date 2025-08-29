"""
Tests for domain entities.

This module tests all the core domain entities including:
- Step types (OperationStep, MapStep, ParallelStep)
- WorkflowDSL
- Result types (OperationResult, MapResult, ParallelResult, WorkflowResult)
"""

import json

import pytest

from argos.domain.entity import (
    MapResult,
    MapStep,
    OperationResult,
    OperationStep,
    ParallelResult,
    ParallelStep,
    StepTypes,
    WorkflowDSL,
    WorkflowResult,
)
from argos.domain.value_object import (
    MapItemResult,
    ParallelOpResult,
    WorkflowResultStatus,
)


class TestOperationStep:
    """Test cases for OperationStep."""

    def test_create_valid_operation_step(self):
        """Test creating a valid operation step."""
        step = OperationStep(id="test_step", operation="test_op", parameters={"param1": "value1", "param2": 42})

        assert step.id == "test_step"
        assert step.operation == "test_op"
        assert step.parameters == {"param1": "value1", "param2": 42}
        assert step.retries == 0
        assert step.timeout is None
        assert step.fail_workflow is True

    def test_create_operation_step_with_optional_params(self):
        """Test creating operation step with optional parameters."""
        step = OperationStep(
            id="test_step",
            operation="test_op",
            parameters={"param1": "value1"},
            retries=3,
            timeout=30.0,
            fail_workflow=False,
        )

        assert step.retries == 3
        assert step.timeout == 30.0
        assert step.fail_workflow is False

    def test_operation_step_validation_success(self):
        """Test valid operation step passes validation."""
        step = OperationStep(id="valid_step", operation="test_op", parameters={"param": "value"})

        # Should not raise an exception
        step.validate()

    def test_operation_step_validation_invalid_id(self):
        """Test operation step validation with invalid id."""
        step = OperationStep(id="", operation="test_op", parameters={"param": "value"})

        with pytest.raises(ValueError, match="Invalid step id"):
            step.validate()

    def test_operation_step_validation_non_string_id(self):
        """Test operation step validation with non-string id."""
        step = OperationStep(
            id=123,  # This will be caught by msgspec, but let's test our validation
            operation="test_op",
            parameters={"param": "value"},
        )

        with pytest.raises(ValueError, match="Invalid step id"):
            step.validate()

    def test_operation_step_validation_invalid_parameters(self):
        """Test operation step validation with invalid parameters."""
        step = OperationStep(
            id="test_step",
            operation="test_op",
            parameters="not_a_dict",  # Should be dict
        )

        with pytest.raises(ValueError, match="parameters must be a dict"):
            step.validate()

    def test_operation_step_with_empty_parameters(self):
        """Test operation step with empty parameters dict."""
        step = OperationStep(id="test_step", operation="test_op", parameters={})

        step.validate()  # Should pass


class TestParallelStep:
    """Test cases for ParallelStep."""

    def test_create_valid_parallel_step(self):
        """Test creating a valid parallel step."""
        operations = [
            OperationStep(id="op1", operation="test_op1", parameters={"p1": "v1"}),
            OperationStep(id="op2", operation="test_op2", parameters={"p2": "v2"}),
        ]

        step = ParallelStep(id="parallel_step", operations=operations)

        assert step.id == "parallel_step"
        assert len(step.operations) == 2
        assert step.retries == 0
        assert step.timeout is None
        assert step.fail_workflow is True

    def test_parallel_step_with_optional_params(self):
        """Test parallel step with optional parameters."""
        operations = [
            OperationStep(id="op1", operation="test_op1", parameters={"p1": "v1"}),
        ]

        step = ParallelStep(id="parallel_step", operations=operations, retries=2, timeout=60.0, fail_workflow=False)

        assert step.retries == 2
        assert step.timeout == 60.0
        assert step.fail_workflow is False

    def test_parallel_step_validation_success(self):
        """Test valid parallel step passes validation."""
        operations = [
            OperationStep(id="op1", operation="test_op1", parameters={"p1": "v1"}),
            OperationStep(id="op2", operation="test_op2", parameters={"p2": "v2"}),
        ]

        step = ParallelStep(id="parallel_step", operations=operations)

        step.validate()  # Should not raise

    def test_parallel_step_validation_invalid_id(self):
        """Test parallel step validation with invalid id."""
        operations = [
            OperationStep(id="op1", operation="test_op1", parameters={"p1": "v1"}),
        ]

        step = ParallelStep(id="", operations=operations)

        with pytest.raises(ValueError, match="Invalid step id"):
            step.validate()

    def test_parallel_step_validation_no_operations(self):
        """Test parallel step validation with no operations."""
        step = ParallelStep(id="parallel_step", operations=[])

        with pytest.raises(ValueError, match="has no operations"):
            step.validate()

    def test_parallel_step_validation_non_operation_step(self):
        """Test parallel step validation with non-operation step."""

        # Create a mock non-operation step
        class MockStep:
            pass

        step = ParallelStep(
            id="parallel_step",
            operations=[MockStep()],  # Not an OperationStep
        )

        with pytest.raises(ValueError, match="contains non-operation step"):
            step.validate()


class TestMapStep:
    """Test cases for MapStep."""

    def test_create_valid_map_step(self):
        """Test creating a valid map step."""
        operation = OperationStep(id="map_op", operation="test_op", parameters={"param": "value"})

        step = MapStep(id="map_step", inputs=["input1", "input2", "input3"], iterator="item", operation=operation)

        assert step.id == "map_step"
        assert step.inputs == ["input1", "input2", "input3"]
        assert step.iterator == "item"
        assert step.mode == "sequential"  # default
        assert step.operation == operation
        assert step.retries == 0
        assert step.timeout is None
        assert step.fail_workflow is True

    def test_map_step_with_parallel_mode(self):
        """Test map step with parallel mode."""
        operation = OperationStep(id="map_op", operation="test_op", parameters={"param": "value"})

        step = MapStep(
            id="map_step", inputs=["input1", "input2"], iterator="item", mode="parallel", operation=operation
        )

        assert step.mode == "parallel"

    def test_map_step_with_optional_params(self):
        """Test map step with optional parameters."""
        operation = OperationStep(id="map_op", operation="test_op", parameters={"param": "value"})

        step = MapStep(
            id="map_step",
            inputs=["input1"],
            iterator="item",
            operation=operation,
            retries=5,
            timeout=45.0,
            fail_workflow=False,
        )

        assert step.retries == 5
        assert step.timeout == 45.0
        assert step.fail_workflow is False

    def test_map_step_validation_success(self):
        """Test valid map step passes validation."""
        operation = OperationStep(id="map_op", operation="test_op", parameters={"param": "value"})

        step = MapStep(id="map_step", inputs=["input1", "input2"], iterator="valid_identifier", operation=operation)

        step.validate()  # Should not raise

    def test_map_step_validation_invalid_id(self):
        """Test map step validation with invalid id."""
        operation = OperationStep(id="map_op", operation="test_op", parameters={"param": "value"})

        step = MapStep(id="", inputs=["input1"], iterator="item", operation=operation)

        with pytest.raises(ValueError, match="Invalid step id"):
            step.validate()

    def test_map_step_validation_empty_inputs(self):
        """Test map step validation with empty inputs."""
        operation = OperationStep(id="map_op", operation="test_op", parameters={"param": "value"})

        step = MapStep(id="map_step", inputs=[], iterator="item", operation=operation)

        with pytest.raises(ValueError, match="has empty inputs"):
            step.validate()

    def test_map_step_validation_invalid_iterator(self):
        """Test map step validation with invalid iterator name."""
        operation = OperationStep(id="map_op", operation="test_op", parameters={"param": "value"})

        step = MapStep(
            id="map_step",
            inputs=["input1"],
            iterator="invalid-identifier",  # Not a valid Python identifier
            operation=operation,
        )

        with pytest.raises(ValueError, match="Invalid iterator name"):
            step.validate()

    def test_map_step_validation_empty_iterator(self):
        """Test map step validation with empty iterator."""
        operation = OperationStep(id="map_op", operation="test_op", parameters={"param": "value"})

        step = MapStep(id="map_step", inputs=["input1"], iterator="", operation=operation)

        with pytest.raises(ValueError, match="Invalid iterator name"):
            step.validate()

    def test_map_step_validation_non_operation_step(self):
        """Test map step validation with non-operation step."""

        class MockOperation:
            pass

        step = MapStep(
            id="map_step",
            inputs=["input1"],
            iterator="item",
            operation=MockOperation(),  # Not an OperationStep
        )

        with pytest.raises(ValueError, match="operation must be an OperationStep"):
            step.validate()


class TestWorkflowDSL:
    """Test cases for WorkflowDSL."""

    def test_create_workflow_with_operation_steps(self):
        """Test creating workflow with operation steps."""
        steps = [
            OperationStep(id="step1", operation="op1", parameters={"p1": "v1"}),
            OperationStep(id="step2", operation="op2", parameters={"p2": "v2"}),
        ]

        workflow = WorkflowDSL(steps=steps)

        assert len(workflow.steps) == 2
        assert workflow.steps[0].id == "step1"
        assert workflow.steps[1].id == "step2"

    def test_create_workflow_with_mixed_step_types(self):
        """Test creating workflow with different step types."""
        operation = OperationStep(id="inner_op", operation="test", parameters={})

        steps = [
            OperationStep(id="op_step", operation="op1", parameters={"p1": "v1"}),
            MapStep(id="map_step", inputs=["a", "b"], iterator="item", operation=operation),
            ParallelStep(
                id="parallel_step",
                operations=[
                    OperationStep(id="p_op1", operation="pop1", parameters={}),
                    OperationStep(id="p_op2", operation="pop2", parameters={}),
                ],
            ),
        ]

        workflow = WorkflowDSL(steps=steps)

        assert len(workflow.steps) == 3
        assert isinstance(workflow.steps[0], OperationStep)
        assert isinstance(workflow.steps[1], MapStep)
        assert isinstance(workflow.steps[2], ParallelStep)

    def test_empty_workflow(self):
        """Test creating workflow with no steps."""
        workflow = WorkflowDSL(steps=[])

        assert len(workflow.steps) == 0


class TestOperationResult:
    """Test cases for OperationResult."""

    def test_create_operation_result(self):
        """Test creating operation result."""
        result = OperationResult(
            id="test_step",
            kind="operation",
            operation="test_op",
            parameters={"param": "value"},
            result="operation result",
        )

        assert result.id == "test_step"
        assert result.kind == "operation"
        assert result.operation == "test_op"
        assert result.parameters == {"param": "value"}
        assert result.result == "operation result"
        assert result.status == "success"  # default
        assert result.error is None

    def test_operation_result_with_failure(self):
        """Test operation result with failure status."""
        result = OperationResult(
            id="failed_step",
            kind="operation",
            operation="failing_op",
            parameters={},
            result=None,
            status="failed",
            error="Something went wrong",
        )

        assert result.status == "failed"
        assert result.error == "Something went wrong"

    def test_operation_result_with_skipped_status(self):
        """Test operation result with skipped status."""
        result = OperationResult(
            id="skipped_step", kind="operation", operation="skipped_op", parameters={}, result=None, status="skipped"
        )

        assert result.status == "skipped"


class TestMapResult:
    """Test cases for MapResult."""

    def test_create_map_result(self):
        """Test creating map result."""
        map_items = [
            MapItemResult(
                id="item1", input="input1", operation="test_op", parameters={"param": "value"}, result="result1"
            ),
            MapItemResult(
                id="item2", input="input2", operation="test_op", parameters={"param": "value"}, result="result2"
            ),
        ]

        result = MapResult(
            id="map_step",
            kind="map",
            mode="sequential",
            iterator="item",
            inputs=["input1", "input2"],
            results=map_items,
        )

        assert result.id == "map_step"
        assert result.kind == "map"
        assert result.mode == "sequential"
        assert result.iterator == "item"
        assert result.inputs == ["input1", "input2"]
        assert len(result.results) == 2

    def test_map_result_parallel_mode(self):
        """Test map result with parallel mode."""
        result = MapResult(id="map_step", kind="map", mode="parallel", iterator="item", inputs=["input1"], results=[])

        assert result.mode == "parallel"


class TestParallelResult:
    """Test cases for ParallelResult."""

    def test_create_parallel_result(self):
        """Test creating parallel result."""
        parallel_ops = [
            ParallelOpResult(id="op1", operation="test_op1", parameters={"p1": "v1"}, result="result1"),
            ParallelOpResult(id="op2", operation="test_op2", parameters={"p2": "v2"}, result="result2"),
        ]

        result = ParallelResult(id="parallel_step", kind="parallel", results=parallel_ops)

        assert result.id == "parallel_step"
        assert result.kind == "parallel"
        assert len(result.results) == 2


class TestWorkflowResult:
    """Test cases for WorkflowResult."""

    def test_create_workflow_result(self):
        """Test creating workflow result."""
        step_results = [OperationResult(id="step1", kind="operation", operation="op1", parameters={}, result="result1")]

        result = WorkflowResult(id="workflow_123", status=WorkflowResultStatus.SUCCESS, results=step_results)

        assert result.id == "workflow_123"
        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 1
        assert result.error is None

    def test_workflow_result_with_failure(self):
        """Test workflow result with failure."""
        result = WorkflowResult(
            id="workflow_456", status=WorkflowResultStatus.FAILED, results=[], error="Workflow failed due to step error"
        )

        assert result.status == WorkflowResultStatus.FAILED
        assert result.error == "Workflow failed due to step error"

    def test_workflow_result_to_dict(self):
        """Test converting workflow result to dictionary."""
        step_results = [
            OperationResult(
                id="step1", kind="operation", operation="op1", parameters={"param": "value"}, result="result1"
            )
        ]

        result = WorkflowResult(id="workflow_789", status=WorkflowResultStatus.SUCCESS, results=step_results)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["id"] == "workflow_789"
        assert result_dict["status"] == "success"
        assert len(result_dict["results"]) == 1

    def test_workflow_result_to_json(self):
        """Test converting workflow result to JSON."""
        result = WorkflowResult(id="workflow_json", status=WorkflowResultStatus.SUCCESS, results=[])

        json_str = result.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["id"] == "workflow_json"
        assert parsed["status"] == "success"

    def test_workflow_result_to_yaml(self):
        """Test converting workflow result to YAML."""
        result = WorkflowResult(id="workflow_yaml", status=WorkflowResultStatus.SUCCESS, results=[])

        yaml_str = result.to_yaml()

        assert isinstance(yaml_str, str)
        assert "workflow_yaml" in yaml_str
        assert "success" in yaml_str


class TestStepTypes:
    """Test StepTypes union type."""

    def test_step_types_union(self):
        """Test that StepTypes includes all step types."""
        # This is mainly a compile-time check, but we can verify the types
        op_step = OperationStep(id="op", operation="test", parameters={})

        inner_op = OperationStep(id="inner", operation="inner_test", parameters={})
        map_step = MapStep(id="map", inputs=["a"], iterator="item", operation=inner_op)

        parallel_step = ParallelStep(id="parallel", operations=[op_step])

        # All should be valid StepTypes
        steps: list[StepTypes] = [op_step, map_step, parallel_step]

        assert len(steps) == 3
        assert isinstance(steps[0], OperationStep)
        assert isinstance(steps[1], MapStep)
        assert isinstance(steps[2], ParallelStep)
