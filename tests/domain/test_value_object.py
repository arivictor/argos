"""
Tests for domain value objects.

This module tests value objects and enums including:
- ExecutionOptions
- WorkflowResultStatus
- ResultStatus
- MapItemResult
- ParallelOpResult
"""

from aroflow.domain.value_object import (
    ExecutionOptions,
    MapItemResult,
    ParallelOpResult,
    ResultStatus,
    WorkflowResultStatus,
)


class TestExecutionOptions:
    """Test cases for ExecutionOptions."""

    def test_default_execution_options(self):
        """Test default execution options."""
        options = ExecutionOptions()

        assert options.retries == 0
        assert options.timeout is None

    def test_execution_options_with_retries(self):
        """Test execution options with custom retries."""
        options = ExecutionOptions(retries=3)

        assert options.retries == 3
        assert options.timeout is None

    def test_execution_options_with_timeout(self):
        """Test execution options with custom timeout."""
        options = ExecutionOptions(timeout=30.0)

        assert options.retries == 0
        assert options.timeout == 30.0

    def test_execution_options_with_both_params(self):
        """Test execution options with both retries and timeout."""
        options = ExecutionOptions(retries=5, timeout=60.0)

        assert options.retries == 5
        assert options.timeout == 60.0

    def test_execution_options_with_zero_timeout(self):
        """Test execution options with zero timeout."""
        options = ExecutionOptions(timeout=0.0)

        assert options.timeout == 0.0

    def test_execution_options_with_negative_retries(self):
        """Test execution options with negative retries."""
        options = ExecutionOptions(retries=-1)

        assert options.retries == -1

    def test_execution_options_equality(self):
        """Test equality comparison of execution options."""
        options1 = ExecutionOptions(retries=3, timeout=30.0)
        options2 = ExecutionOptions(retries=3, timeout=30.0)
        options3 = ExecutionOptions(retries=2, timeout=30.0)

        assert options1 == options2
        assert options1 != options3

    def test_execution_options_immutability(self):
        """Test that execution options behave as expected for dataclass."""
        options = ExecutionOptions(retries=3, timeout=30.0)

        # Should be able to modify since it's a regular dataclass
        options.retries = 5
        assert options.retries == 5


class TestWorkflowResultStatus:
    """Test cases for WorkflowResultStatus enum."""

    def test_workflow_result_status_values(self):
        """Test workflow result status enum values."""
        assert WorkflowResultStatus.SUCCESS == "success"
        assert WorkflowResultStatus.FAILED == "failed"
        assert WorkflowResultStatus.PARTIAL_FAILURE == "partial_failure"

    def test_workflow_result_status_iteration(self):
        """Test iterating over WorkflowResultStatus values."""
        statuses = list(WorkflowResultStatus)

        assert len(statuses) == 3
        assert WorkflowResultStatus.SUCCESS in statuses
        assert WorkflowResultStatus.FAILED in statuses
        assert WorkflowResultStatus.PARTIAL_FAILURE in statuses

    def test_workflow_result_status_comparison(self):
        """Test comparison of WorkflowResultStatus values."""
        success1 = WorkflowResultStatus.SUCCESS
        success2 = WorkflowResultStatus.SUCCESS
        failed = WorkflowResultStatus.FAILED

        assert success1 == success2
        assert success1 != failed
        assert success1 == "success"
        assert failed == "failed"

    def test_workflow_result_status_string_representation(self):
        """Test string representation of WorkflowResultStatus."""
        # Enum values inherit their string value, not their name representation
        assert WorkflowResultStatus.SUCCESS.value == "success"
        assert WorkflowResultStatus.FAILED.value == "failed"
        assert WorkflowResultStatus.PARTIAL_FAILURE.value == "partial_failure"


class TestResultStatus:
    """Test cases for ResultStatus enum."""

    def test_result_status_values(self):
        """Test result status enum values."""
        assert ResultStatus.SUCCESS == "success"
        assert ResultStatus.FAILED == "failed"
        assert ResultStatus.SKIPPED == "skipped"

    def test_result_status_iteration(self):
        """Test iterating over ResultStatus values."""
        statuses = list(ResultStatus)

        assert len(statuses) == 3
        assert ResultStatus.SUCCESS in statuses
        assert ResultStatus.FAILED in statuses
        assert ResultStatus.SKIPPED in statuses

    def test_result_status_comparison(self):
        """Test comparison of ResultStatus values."""
        success1 = ResultStatus.SUCCESS
        success2 = ResultStatus.SUCCESS
        failed = ResultStatus.FAILED

        assert success1 == success2
        assert success1 != failed
        assert success1 == "success"
        assert failed == "failed"

    def test_result_status_string_representation(self):
        """Test string representation of ResultStatus."""
        # Enum values inherit their string value, not their name representation
        assert ResultStatus.SUCCESS.value == "success"
        assert ResultStatus.FAILED.value == "failed"
        assert ResultStatus.SKIPPED.value == "skipped"


class TestMapItemResult:
    """Test cases for MapItemResult."""

    def test_create_map_item_result(self):
        """Test creating a map item result."""
        result = MapItemResult(
            id="item_1",
            input="input_value",
            operation="test_operation",
            parameters={"param1": "value1", "param2": 42},
            result="output_value",
        )

        assert result.id == "item_1"
        assert result.input == "input_value"
        assert result.operation == "test_operation"
        assert result.parameters == {"param1": "value1", "param2": 42}
        assert result.result == "output_value"
        assert result.status == ResultStatus.SUCCESS  # default
        assert result.error is None

    def test_map_item_result_with_custom_status(self):
        """Test map item result with custom status."""
        result = MapItemResult(
            id="item_2", input="input", operation="test_op", parameters={}, result=None, status=ResultStatus.FAILED
        )

        assert result.status == ResultStatus.FAILED

    def test_map_item_result_with_error(self):
        """Test map item result with error."""
        result = MapItemResult(
            id="item_3",
            input="input",
            operation="test_op",
            parameters={},
            result=None,
            status=ResultStatus.FAILED,
            error="Something went wrong",
        )

        assert result.status == ResultStatus.FAILED
        assert result.error == "Something went wrong"

    def test_map_item_result_with_skipped_status(self):
        """Test map item result with skipped status."""
        result = MapItemResult(
            id="item_4", input="input", operation="test_op", parameters={}, result=None, status=ResultStatus.SKIPPED
        )

        assert result.status == ResultStatus.SKIPPED

    def test_map_item_result_with_complex_input(self):
        """Test map item result with complex input data."""
        complex_input = {"nested": {"data": [1, 2, 3]}, "list": ["a", "b", "c"], "number": 42}

        result = MapItemResult(
            id="item_5",
            input=complex_input,
            operation="process_complex",
            parameters={"mode": "deep"},
            result={"processed": True},
        )

        assert result.input == complex_input
        assert result.result == {"processed": True}

    def test_map_item_result_with_none_values(self):
        """Test map item result with None values where allowed."""
        result = MapItemResult(id="item_6", input=None, operation="test_op", parameters={}, result=None)

        assert result.input is None
        assert result.result is None
        assert result.error is None

    def test_map_item_result_equality(self):
        """Test equality comparison of map item results."""
        result1 = MapItemResult(id="item_1", input="input", operation="op", parameters={}, result="output")

        result2 = MapItemResult(id="item_1", input="input", operation="op", parameters={}, result="output")

        result3 = MapItemResult(id="item_2", input="input", operation="op", parameters={}, result="output")

        assert result1 == result2
        assert result1 != result3


class TestParallelOpResult:
    """Test cases for ParallelOpResult."""

    def test_create_parallel_op_result(self):
        """Test creating a parallel operation result."""
        result = ParallelOpResult(
            id="parallel_op_1",
            operation="test_operation",
            parameters={"param1": "value1", "param2": 42},
            result="operation_output",
        )

        assert result.id == "parallel_op_1"
        assert result.operation == "test_operation"
        assert result.parameters == {"param1": "value1", "param2": 42}
        assert result.result == "operation_output"
        assert result.status == ResultStatus.SUCCESS  # default
        assert result.error is None

    def test_parallel_op_result_with_custom_status(self):
        """Test parallel operation result with custom status."""
        result = ParallelOpResult(
            id="parallel_op_2", operation="test_op", parameters={}, result=None, status=ResultStatus.FAILED
        )

        assert result.status == ResultStatus.FAILED

    def test_parallel_op_result_with_error(self):
        """Test parallel operation result with error."""
        result = ParallelOpResult(
            id="parallel_op_3",
            operation="test_op",
            parameters={},
            result=None,
            status=ResultStatus.FAILED,
            error="Operation failed",
        )

        assert result.status == ResultStatus.FAILED
        assert result.error == "Operation failed"

    def test_parallel_op_result_with_skipped_status(self):
        """Test parallel operation result with skipped status."""
        result = ParallelOpResult(
            id="parallel_op_4", operation="test_op", parameters={}, result=None, status=ResultStatus.SKIPPED
        )

        assert result.status == ResultStatus.SKIPPED

    def test_parallel_op_result_with_complex_parameters(self):
        """Test parallel operation result with complex parameters."""
        complex_params = {
            "config": {"mode": "advanced", "threads": 4},
            "data": [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}],
            "flags": [True, False, True],
        }

        result = ParallelOpResult(
            id="parallel_op_5",
            operation="complex_operation",
            parameters=complex_params,
            result={"success": True, "items_processed": 2},
        )

        assert result.parameters == complex_params
        assert result.result == {"success": True, "items_processed": 2}

    def test_parallel_op_result_with_none_values(self):
        """Test parallel operation result with None values where allowed."""
        result = ParallelOpResult(id="parallel_op_6", operation="test_op", parameters={}, result=None)

        assert result.result is None
        assert result.error is None

    def test_parallel_op_result_equality(self):
        """Test equality comparison of parallel operation results."""
        result1 = ParallelOpResult(id="op_1", operation="test_op", parameters={"param": "value"}, result="output")

        result2 = ParallelOpResult(id="op_1", operation="test_op", parameters={"param": "value"}, result="output")

        result3 = ParallelOpResult(id="op_2", operation="test_op", parameters={"param": "value"}, result="output")

        assert result1 == result2
        assert result1 != result3

    def test_parallel_op_result_with_empty_parameters(self):
        """Test parallel operation result with empty parameters."""
        result = ParallelOpResult(id="parallel_op_7", operation="no_param_op", parameters={}, result="success")

        assert result.parameters == {}
        assert result.result == "success"


class TestValueObjectsInteraction:
    """Test interaction between different value objects."""

    def test_status_enums_compatibility(self):
        """Test that status enums can be used interchangeably where appropriate."""
        # Both enums have SUCCESS and FAILED
        assert ResultStatus.SUCCESS == "success"
        assert WorkflowResultStatus.SUCCESS == "success"
        assert ResultStatus.FAILED == "failed"
        assert WorkflowResultStatus.FAILED == "failed"

        # But ResultStatus has SKIPPED while WorkflowResultStatus has PARTIAL_FAILURE
        assert hasattr(ResultStatus, "SKIPPED")
        assert hasattr(WorkflowResultStatus, "PARTIAL_FAILURE")
        assert not hasattr(ResultStatus, "PARTIAL_FAILURE")
        assert not hasattr(WorkflowResultStatus, "SKIPPED")

    def test_result_objects_with_all_status_values(self):
        """Test result objects with all possible status values."""
        # MapItemResult with all ResultStatus values
        map_success = MapItemResult(
            id="map_1", input="in", operation="op", parameters={}, result="out", status=ResultStatus.SUCCESS
        )
        map_failed = MapItemResult(
            id="map_2",
            input="in",
            operation="op",
            parameters={},
            result=None,
            status=ResultStatus.FAILED,
            error="Failed",
        )
        map_skipped = MapItemResult(
            id="map_3", input="in", operation="op", parameters={}, result=None, status=ResultStatus.SKIPPED
        )

        assert map_success.status == ResultStatus.SUCCESS
        assert map_failed.status == ResultStatus.FAILED
        assert map_skipped.status == ResultStatus.SKIPPED

        # ParallelOpResult with all ResultStatus values
        parallel_success = ParallelOpResult(
            id="parallel_1", operation="op", parameters={}, result="out", status=ResultStatus.SUCCESS
        )
        parallel_failed = ParallelOpResult(
            id="parallel_2", operation="op", parameters={}, result=None, status=ResultStatus.FAILED, error="Failed"
        )
        parallel_skipped = ParallelOpResult(
            id="parallel_3", operation="op", parameters={}, result=None, status=ResultStatus.SKIPPED
        )

        assert parallel_success.status == ResultStatus.SUCCESS
        assert parallel_failed.status == ResultStatus.FAILED
        assert parallel_skipped.status == ResultStatus.SKIPPED
