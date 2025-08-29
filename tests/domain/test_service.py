"""
Tests for domain service.

This module tests the domain service functions including:
- validate_workflow function with various scenarios
"""

import pytest
from argos.domain.service import validate_workflow
from argos.domain.entity import (
    WorkflowDSL,
    OperationStep,
    MapStep,
    ParallelStep,
)


class TestValidateWorkflow:
    """Test cases for validate_workflow function."""

    def test_validate_workflow_with_valid_single_step(self):
        """Test validating workflow with single valid step."""
        steps = [
            OperationStep(
                id="step1",
                operation="test_op",
                parameters={"param": "value"}
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        # Should return True and not raise
        result = validate_workflow(workflow)
        assert result is True

    def test_validate_workflow_with_multiple_valid_steps(self):
        """Test validating workflow with multiple valid steps."""
        steps = [
            OperationStep(
                id="step1",
                operation="op1",
                parameters={"p1": "v1"}
            ),
            OperationStep(
                id="step2",
                operation="op2",
                parameters={"p2": "v2"}
            ),
            OperationStep(
                id="step3",
                operation="op3",
                parameters={"p3": "v3"}
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        result = validate_workflow(workflow)
        assert result is True

    def test_validate_workflow_with_mixed_step_types(self):
        """Test validating workflow with different step types."""
        inner_operation = OperationStep(
            id="inner_op",
            operation="inner_test",
            parameters={}
        )
        
        steps = [
            OperationStep(
                id="operation_step",
                operation="test_op",
                parameters={"param": "value"}
            ),
            MapStep(
                id="map_step",
                inputs=["input1", "input2"],
                iterator="item",
                operation=inner_operation
            ),
            ParallelStep(
                id="parallel_step",
                operations=[
                    OperationStep(id="p_op1", operation="pop1", parameters={}),
                    OperationStep(id="p_op2", operation="pop2", parameters={}),
                ]
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        result = validate_workflow(workflow)
        assert result is True

    def test_validate_workflow_with_no_steps(self):
        """Test validating workflow with no steps."""
        workflow = WorkflowDSL(steps=[])
        
        with pytest.raises(ValueError, match="Workflow has no steps"):
            validate_workflow(workflow)

    def test_validate_workflow_with_duplicate_step_ids(self):
        """Test validating workflow with duplicate step IDs."""
        steps = [
            OperationStep(
                id="duplicate_id",
                operation="op1",
                parameters={"p1": "v1"}
            ),
            OperationStep(
                id="duplicate_id",  # Duplicate ID
                operation="op2",
                parameters={"p2": "v2"}
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        with pytest.raises(ValueError, match="Duplicate step id found: duplicate_id"):
            validate_workflow(workflow)

    def test_validate_workflow_with_multiple_duplicate_ids(self):
        """Test validating workflow with multiple sets of duplicate IDs."""
        steps = [
            OperationStep(id="id1", operation="op1", parameters={}),
            OperationStep(id="id2", operation="op2", parameters={}),
            OperationStep(id="id1", operation="op3", parameters={}),  # Duplicate
            OperationStep(id="id3", operation="op4", parameters={}),
        ]
        workflow = WorkflowDSL(steps=steps)
        
        # Should catch the first duplicate
        with pytest.raises(ValueError, match="Duplicate step id found: id1"):
            validate_workflow(workflow)

    def test_validate_workflow_calls_step_validation(self):
        """Test that validate_workflow calls validate() on each step."""
        # Create a step with invalid parameters to trigger step validation
        steps = [
            OperationStep(
                id="valid_step",
                operation="test_op",
                parameters={"param": "value"}
            ),
            OperationStep(
                id="invalid_step",
                operation="test_op",
                parameters="not_a_dict"  # Invalid parameters
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        # Should fail at step validation
        with pytest.raises(ValueError, match="parameters must be a dict"):
            validate_workflow(workflow)

    def test_validate_workflow_with_invalid_operation_step(self):
        """Test workflow validation with invalid operation step."""
        steps = [
            OperationStep(
                id="",  # Invalid empty ID
                operation="test_op",
                parameters={"param": "value"}
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        with pytest.raises(ValueError, match="Invalid step id"):
            validate_workflow(workflow)

    def test_validate_workflow_with_invalid_map_step(self):
        """Test workflow validation with invalid map step."""
        inner_operation = OperationStep(
            id="inner_op",
            operation="inner_test",
            parameters={}
        )
        
        steps = [
            MapStep(
                id="map_step",
                inputs=[],  # Invalid empty inputs
                iterator="item",
                operation=inner_operation
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        with pytest.raises(ValueError, match="has empty inputs"):
            validate_workflow(workflow)

    def test_validate_workflow_with_invalid_parallel_step(self):
        """Test workflow validation with invalid parallel step."""
        steps = [
            ParallelStep(
                id="parallel_step",
                operations=[]  # Invalid empty operations
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        with pytest.raises(ValueError, match="has no operations"):
            validate_workflow(workflow)

    def test_validate_workflow_with_complex_map_step(self):
        """Test workflow validation with complex nested map step."""
        inner_operation = OperationStep(
            id="inner_op",
            operation="process_item",
            parameters={"multiplier": 2}
        )
        
        steps = [
            MapStep(
                id="complex_map",
                inputs=[1, 2, 3, 4, 5],
                iterator="number",
                mode="parallel",
                operation=inner_operation,
                retries=3,
                timeout=30.0,
                fail_workflow=False
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        result = validate_workflow(workflow)
        assert result is True

    def test_validate_workflow_with_complex_parallel_step(self):
        """Test workflow validation with complex parallel step."""
        operations = [
            OperationStep(
                id="parallel_op1",
                operation="task1",
                parameters={"param1": "value1"},
                retries=2,
                timeout=15.0
            ),
            OperationStep(
                id="parallel_op2",
                operation="task2",
                parameters={"param2": "value2"},
                retries=1,
                timeout=20.0
            ),
            OperationStep(
                id="parallel_op3",
                operation="task3",
                parameters={"param3": "value3"}
            )
        ]
        
        steps = [
            ParallelStep(
                id="complex_parallel",
                operations=operations,
                retries=1,
                timeout=60.0,
                fail_workflow=False
            )
        ]
        workflow = WorkflowDSL(steps=steps)
        
        result = validate_workflow(workflow)
        assert result is True

    def test_validate_workflow_stops_at_first_duplicate(self):
        """Test that validation stops at the first duplicate ID found."""
        steps = [
            OperationStep(id="step1", operation="op1", parameters={}),
            OperationStep(id="step2", operation="op2", parameters={}),
            OperationStep(id="step1", operation="op3", parameters={}),  # First duplicate
            OperationStep(id="step2", operation="op4", parameters={}),  # Second duplicate (not reached)
        ]
        workflow = WorkflowDSL(steps=steps)
        
        # Should only report the first duplicate
        with pytest.raises(ValueError, match="Duplicate step id found: step1"):
            validate_workflow(workflow)

    def test_validate_workflow_with_large_number_of_steps(self):
        """Test workflow validation with many steps."""
        steps = []
        for i in range(100):
            steps.append(
                OperationStep(
                    id=f"step_{i}",
                    operation=f"operation_{i}",
                    parameters={f"param_{i}": f"value_{i}"}
                )
            )
        
        workflow = WorkflowDSL(steps=steps)
        
        result = validate_workflow(workflow)
        assert result is True

    def test_validate_workflow_preserves_step_order(self):
        """Test that validation processes steps in order."""
        # Create steps where only the second one has an invalid ID
        steps = [
            OperationStep(id="valid_step", operation="op1", parameters={}),
            OperationStep(id="", operation="op2", parameters={}),  # Invalid
            OperationStep(id="another_valid", operation="op3", parameters={}),
        ]
        workflow = WorkflowDSL(steps=steps)
        
        # Should fail on the second step
        with pytest.raises(ValueError, match="Invalid step id"):
            validate_workflow(workflow)

    def test_validate_workflow_returns_true_on_success(self):
        """Test that validate_workflow returns True on successful validation."""
        steps = [
            OperationStep(id="step1", operation="op1", parameters={"param": "value"})
        ]
        workflow = WorkflowDSL(steps=steps)
        
        result = validate_workflow(workflow)
        
        # Explicitly test return value
        assert result is True
        assert isinstance(result, bool)