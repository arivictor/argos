"""
Tests for in-memory workflow engine.

This module tests the InMemoryWorkflowEngine implementation.
"""

from unittest.mock import Mock

from aroflow.application.port import ExecutorFactory, ResultStore
from aroflow.application.service import ResultRegistrar
from aroflow.domain.entity import (
    OperationResult,
    OperationStep,
    WorkflowDSL,
    WorkflowResult,
)
from aroflow.domain.value_object import ResultStatus, WorkflowResultStatus
from aroflow.infrastructure.adapter.in_memory.workflow_engine import (
    InMemoryWorkflowEngine,
    UUIDGenerator,
)


class TestUUIDGenerator:
    """Test cases for UUIDGenerator."""

    def test_generate_uuid(self):
        """Test generating UUID."""
        generator = UUIDGenerator()

        generated_id = generator.generate()

        assert isinstance(generated_id, str)
        assert len(generated_id) == 32  # uuid4().hex is 32 characters

        # Should be valid hex
        int(generated_id, 16)  # Should not raise ValueError

    def test_generate_unique_uuids(self):
        """Test that generated UUIDs are unique."""
        generator = UUIDGenerator()

        ids = [generator.generate() for _ in range(100)]

        # All should be unique
        assert len(set(ids)) == 100

    def test_multiple_generators(self):
        """Test that multiple generators produce different UUIDs."""
        gen1 = UUIDGenerator()
        gen2 = UUIDGenerator()

        id1 = gen1.generate()
        id2 = gen2.generate()

        assert id1 != id2


class TestInMemoryWorkflowEngine:
    """Test cases for InMemoryWorkflowEngine."""

    def setup_method(self):
        """Setup test fixtures."""
        self.executor_factory = Mock(spec=ExecutorFactory)
        self.result_store = Mock(spec=ResultStore)
        self.registrar = Mock(spec=ResultRegistrar)

        self.engine = InMemoryWorkflowEngine(
            executor_factory=self.executor_factory,
            result_store=self.result_store,
            registrar=self.registrar,
        )

    def test_create_workflow_engine(self):
        """Test creating workflow engine."""
        assert self.engine.executor_factory == self.executor_factory
        assert self.engine.registrar == self.registrar
        # Note: The engine creates its own result store if none provided
        assert hasattr(self.engine, "ctx")
        assert hasattr(self.engine, "values")

    def test_run_workflow_success(self):
        """Test running workflow successfully."""
        # Setup step
        step = OperationStep(id="test_step", operation="test_op", parameters={"param": "value"})
        workflow = WorkflowDSL(steps=[step])

        # Setup executor mock
        executor = Mock()
        step_result = OperationResult(
            id="test_step",
            kind="operation",
            operation="test_op",
            parameters={"param": "value"},
            result="success_result",
            status="success",
        )
        executor.execute.return_value = step_result
        self.executor_factory.get_executor.return_value = executor

        # Run workflow
        result = self.engine.run(workflow)

        # Verify calls
        self.executor_factory.get_executor.assert_called_once_with(step)
        executor.execute.assert_called_once_with(step)
        self.registrar.register.assert_called_once_with(step_result)

        # Verify result
        assert isinstance(result, WorkflowResult)
        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 1
        assert result.results[0] == step_result
        assert result.error is None

    def test_run_workflow_with_custom_id(self):
        """Test running workflow with custom workflow ID."""
        step = OperationStep(id="step", operation="op", parameters={})
        workflow = WorkflowDSL(steps=[step])

        # Setup executor
        executor = Mock()
        step_result = OperationResult(
            id="step", kind="operation", operation="op", parameters={}, result="result", status="success"
        )
        executor.execute.return_value = step_result
        self.executor_factory.get_executor.return_value = executor

        # Run with custom ID
        custom_id = "custom_workflow_123"
        result = self.engine.run(workflow, workflow_id=custom_id)

        assert result.id == custom_id

    def test_run_workflow_generates_id_when_none(self):
        """Test running workflow generates ID when none provided."""
        step = OperationStep(id="step", operation="op", parameters={})
        workflow = WorkflowDSL(steps=[step])

        # Setup executor
        executor = Mock()
        step_result = OperationResult(
            id="step", kind="operation", operation="op", parameters={}, result="result", status="success"
        )
        executor.execute.return_value = step_result
        self.executor_factory.get_executor.return_value = executor

        # Run without ID
        result = self.engine.run(workflow)

        assert result.id is not None
        assert isinstance(result.id, str)
        assert len(result.id) == 32  # UUID hex

    def test_run_workflow_with_execution_error(self):
        """Test running workflow with execution error."""
        step = OperationStep(id="failing_step", operation="failing_op", parameters={})
        workflow = WorkflowDSL(steps=[step])

        # Setup executor to raise exception
        executor = Mock()
        executor.execute.side_effect = RuntimeError("Execution failed")
        self.executor_factory.get_executor.return_value = executor

        # Run workflow
        result = self.engine.run(workflow)

        # Should have failed status and error message
        assert result.status == WorkflowResultStatus.FAILED
        assert result.error == "Execution failed"
        assert len(result.results) == 0  # No results when exception occurs

    def test_run_workflow_with_failed_step(self):
        """Test running workflow with failed step."""
        step = OperationStep(id="failed_step", operation="failing_op", parameters={})
        workflow = WorkflowDSL(steps=[step])

        # Setup executor with failed result
        executor = Mock()
        step_result = OperationResult(
            id="failed_step",
            kind="operation",
            operation="failing_op",
            parameters={},
            result=None,
            status="failed",
            error="Step failed",
        )
        # Mock the hasattr and getattr calls
        step_result.status = ResultStatus.FAILED
        executor.execute.return_value = step_result
        self.executor_factory.get_executor.return_value = executor

        # Run workflow
        result = self.engine.run(workflow)

        # Should detect failure
        assert result.status == WorkflowResultStatus.FAILED
        assert len(result.results) == 1
        assert result.results[0] == step_result

    def test_run_workflow_with_nonfatal_failure(self):
        """Test running workflow with non-fatal failure."""
        step = OperationStep(
            id="nonfatal_step",
            operation="op",
            parameters={},
            fail_workflow=False,  # Non-fatal
        )
        workflow = WorkflowDSL(steps=[step])

        # Setup executor with failed result
        executor = Mock()
        step_result = OperationResult(
            id="nonfatal_step", kind="operation", operation="op", parameters={}, result=None, status="failed"
        )
        step_result.status = ResultStatus.FAILED
        executor.execute.return_value = step_result
        self.executor_factory.get_executor.return_value = executor

        # Run workflow
        result = self.engine.run(workflow)

        # Should have partial failure status
        assert result.status == WorkflowResultStatus.PARTIAL_FAILURE
        assert len(result.results) == 1

    def test_run_workflow_multiple_steps(self):
        """Test running workflow with multiple steps."""
        steps = [
            OperationStep(id="step1", operation="op1", parameters={}),
            OperationStep(id="step2", operation="op2", parameters={}),
            OperationStep(id="step3", operation="op3", parameters={}),
        ]
        workflow = WorkflowDSL(steps=steps)

        # Setup executors
        executors = [Mock() for _ in range(3)]
        step_results = [
            OperationResult(
                id=f"step{i + 1}",
                kind="operation",
                operation=f"op{i + 1}",
                parameters={},
                result=f"result{i + 1}",
                status="success",
            )
            for i in range(3)
        ]

        for _i, (executor, step_result) in enumerate(zip(executors, step_results, strict=False)):
            executor.execute.return_value = step_result

        self.executor_factory.get_executor.side_effect = executors

        # Run workflow
        result = self.engine.run(workflow)

        # Verify all steps executed
        assert len(result.results) == 3
        assert result.status == WorkflowResultStatus.SUCCESS
        assert [r.id for r in result.results] == ["step1", "step2", "step3"]

    def test_run_workflow_stops_on_fatal_error(self):
        """Test that workflow stops on fatal error."""
        steps = [
            OperationStep(id="step1", operation="op1", parameters={}),
            OperationStep(id="step2", operation="op2", parameters={}),  # This will fail
            OperationStep(id="step3", operation="op3", parameters={}),  # Should not execute
        ]
        workflow = WorkflowDSL(steps=steps)

        # Setup first executor to succeed
        executor1 = Mock()
        step_result1 = OperationResult(
            id="step1", kind="operation", operation="op1", parameters={}, result="result1", status="success"
        )
        executor1.execute.return_value = step_result1

        # Setup second executor to raise exception
        executor2 = Mock()
        executor2.execute.side_effect = RuntimeError("Fatal error")

        # Third executor should not be called
        executor3 = Mock()

        self.executor_factory.get_executor.side_effect = [executor1, executor2, executor3]

        # Run workflow
        result = self.engine.run(workflow)

        # Should have failed and stopped after step2
        assert result.status == WorkflowResultStatus.FAILED
        assert result.error == "Fatal error"
        assert len(result.results) == 1  # Only step1 result
        assert result.results[0] == step_result1

        # Verify step3 executor was not called
        executor3.execute.assert_not_called()

    def test_run_empty_workflow(self):
        """Test running workflow with no steps."""
        workflow = WorkflowDSL(steps=[])

        # This should succeed but return empty results
        result = self.engine.run(workflow)

        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 0
        assert result.error is None

    def test_engine_initialization_with_defaults(self):
        """Test engine initialization with default components."""
        from aroflow.application.service import ResultRegistrar
        from aroflow.infrastructure.adapter.in_memory.result_store import InMemoryResultStore

        # Create engine with minimal setup
        engine = InMemoryWorkflowEngine(
            executor_factory=self.executor_factory,
            result_store=None,  # Should create default
            registrar=None,  # Should create default
        )

        # Should have created default components
        assert isinstance(engine.result_store, InMemoryResultStore)
        assert isinstance(engine.registrar, ResultRegistrar)
        assert engine.registrar.result_store == engine.result_store

    def test_workflow_result_structure(self):
        """Test that workflow result has correct structure."""
        step = OperationStep(id="step", operation="op", parameters={})
        workflow = WorkflowDSL(steps=[step])

        # Setup executor
        executor = Mock()
        step_result = OperationResult(
            id="step", kind="operation", operation="op", parameters={}, result="result", status="success"
        )
        executor.execute.return_value = step_result
        self.executor_factory.get_executor.return_value = executor

        # Run workflow
        result = self.engine.run(workflow, workflow_id="test_workflow")

        # Verify structure
        assert hasattr(result, "id")
        assert hasattr(result, "status")
        assert hasattr(result, "results")
        assert hasattr(result, "error")

        assert result.id == "test_workflow"
        assert isinstance(result.status, WorkflowResultStatus)
        assert isinstance(result.results, list)
        assert result.error is None or isinstance(result.error, str)
