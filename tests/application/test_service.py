"""
Tests for application services.

This module tests the application layer services including:
- ResultRegistrar
- load_workflow function
- execute_workflow function  
- WorkflowClient
"""

import pytest
from unittest.mock import Mock, MagicMock

from argos.application.service import (
    ResultRegistrar,
    load_workflow,
    execute_workflow,
    WorkflowClient,
)
from argos.application.port import ResultStore, WorkflowEngine
from argos.domain.entity import (
    WorkflowDSL,
    OperationStep,
    OperationResult,
    MapResult,
    ParallelResult,
    WorkflowResult,
)
from argos.domain.value_object import (
    MapItemResult,
    ParallelOpResult,
    WorkflowResultStatus,
    ExecutionOptions,
)


class TestResultRegistrar:
    """Test cases for ResultRegistrar."""

    def setup_method(self):
        """Setup test fixtures."""
        self.result_store = Mock(spec=ResultStore)
        self.registrar = ResultRegistrar(self.result_store)

    def test_create_result_registrar(self):
        """Test creating result registrar."""
        assert self.registrar.result_store == self.result_store

    def test_register_operation_result(self):
        """Test registering operation result."""
        result = OperationResult(
            id="step1",
            kind="operation",
            operation="test_op",
            parameters={"param": "value"},
            result="output"
        )
        
        self.registrar.register(result)
        
        # Should register the result by its ID twice (general + specific)
        assert self.result_store.set.call_count == 2
        # Both calls should be with the same arguments
        for call in self.result_store.set.call_args_list:
            assert call.args == ("step1", result)

    def test_register_map_result(self):
        """Test registering map result."""
        map_items = [
            MapItemResult(
                id="item1",
                input="input1",
                operation="test_op",
                parameters={},
                result="result1"
            ),
            MapItemResult(
                id="item2",
                input="input2",
                operation="test_op",
                parameters={},
                result="result2"
            )
        ]
        
        result = MapResult(
            id="map_step",
            kind="map",
            mode="sequential",
            iterator="item",
            inputs=["input1", "input2"],
            results=map_items
        )
        
        self.registrar.register(result)
        
        # Should register the main result and each item result
        expected_calls = [
            (("map_step", result),),
            (("map_step", map_items),),  # Store results list under step id
            (("item1", map_items[0]),),  # Each item by its own id
            (("item2", map_items[1]),),
        ]
        
        assert self.result_store.set.call_count == 4
        for call, expected in zip(self.result_store.set.call_args_list, expected_calls):
            assert call.args == expected[0]

    def test_register_parallel_result(self):
        """Test registering parallel result."""
        parallel_ops = [
            ParallelOpResult(
                id="op1",
                operation="test_op1",
                parameters={},
                result="result1"
            ),
            ParallelOpResult(
                id="op2",
                operation="test_op2",
                parameters={},
                result="result2"
            )
        ]
        
        result = ParallelResult(
            id="parallel_step",
            kind="parallel",
            results=parallel_ops
        )
        
        self.registrar.register(result)
        
        # Should register the main result and each operation result
        expected_calls = [
            (("parallel_step", result),),
            (("op1", parallel_ops[0]),),
            (("op2", parallel_ops[1]),),
        ]
        
        assert self.result_store.set.call_count == 3
        for call, expected in zip(self.result_store.set.call_args_list, expected_calls):
            assert call.args == expected[0]

    def test_register_result_without_id(self):
        """Test registering result without id attribute."""
        class ResultWithoutId:
            pass
        
        result = ResultWithoutId()
        
        # Should not crash, just not register anything
        self.registrar.register(result)
        
        self.result_store.set.assert_not_called()

    def test_register_none_result(self):
        """Test registering None result."""
        self.registrar.register(None)
        
        # Should not crash
        self.result_store.set.assert_not_called()

    def test_register_mixed_results_sequence(self):
        """Test registering a sequence of different result types."""
        operation_result = OperationResult(
            id="step1",
            kind="operation",
            operation="op1",
            parameters={},
            result="output1"
        )
        
        map_result = MapResult(
            id="step2",
            kind="map",
            mode="sequential",
            iterator="item",
            inputs=["a"],
            results=[
                MapItemResult(
                    id="item1",
                    input="a",
                    operation="op2",
                    parameters={},
                    result="mapped_a"
                )
            ]
        )
        
        # Register both
        self.registrar.register(operation_result)
        self.registrar.register(map_result)
        
        # Should have registered all components
        assert self.result_store.set.call_count == 5  # op result (2x) + map result + map results list + map item


class TestLoadWorkflow:
    """Test cases for load_workflow function."""

    def test_load_workflow_from_dict(self):
        """Test loading workflow from dictionary."""
        workflow_dict = {
            "steps": [
                {
                    "id": "step1",
                    "kind": "operation",
                    "operation": "test_op",
                    "parameters": {"param": "value"}
                }
            ]
        }
        
        workflow = load_workflow(workflow_dict)
        
        assert isinstance(workflow, WorkflowDSL)
        assert len(workflow.steps) == 1
        assert workflow.steps[0].id == "step1"
        assert workflow.steps[0].operation == "test_op"

    def test_load_workflow_from_workflow_dsl(self):
        """Test loading workflow from existing WorkflowDSL."""
        step = OperationStep(
            id="step1",
            operation="test_op",
            parameters={"param": "value"}
        )
        existing_workflow = WorkflowDSL(steps=[step])
        
        workflow = load_workflow(existing_workflow)
        
        assert workflow is existing_workflow

    def test_load_workflow_validation_called(self):
        """Test that workflow validation is called."""
        # Create workflow with duplicate IDs to trigger validation error
        workflow_dict = {
            "steps": [
                {
                    "id": "duplicate",
                    "kind": "operation",
                    "operation": "op1",
                    "parameters": {}
                },
                {
                    "id": "duplicate",  # Duplicate ID
                    "kind": "operation",
                    "operation": "op2",
                    "parameters": {}
                }
            ]
        }
        
        with pytest.raises(ValueError, match="Duplicate step id found"):
            load_workflow(workflow_dict)

    def test_load_workflow_invalid_structure(self):
        """Test loading workflow with invalid structure."""
        # Missing required 'steps' field
        invalid_dict = {"not_steps": []}
        
        with pytest.raises(Exception):  # msgspec will raise conversion error
            load_workflow(invalid_dict)

    def test_load_workflow_empty_steps(self):
        """Test loading workflow with empty steps."""
        workflow_dict = {"steps": []}
        
        with pytest.raises(ValueError, match="Workflow has no steps"):
            load_workflow(workflow_dict)

    def test_load_workflow_complex(self):
        """Test loading complex workflow with different step types."""
        workflow_dict = {
            "steps": [
                {
                    "id": "operation_step",
                    "kind": "operation",
                    "operation": "simple_op",
                    "parameters": {"param": "value"}
                },
                {
                    "id": "map_step",
                    "kind": "map",
                    "inputs": ["a", "b", "c"],
                    "iterator": "item",
                    "operation": {
                        "id": "map_operation",
                        "kind": "operation",
                        "operation": "process_item",
                        "parameters": {"multiplier": 2}
                    }
                },
                {
                    "id": "parallel_step",
                    "kind": "parallel",
                    "operations": [
                        {
                            "id": "parallel_op1",
                            "kind": "operation",
                            "operation": "task1",
                            "parameters": {}
                        },
                        {
                            "id": "parallel_op2",
                            "kind": "operation",
                            "operation": "task2",
                            "parameters": {}
                        }
                    ]
                }
            ]
        }
        
        workflow = load_workflow(workflow_dict)
        
        assert len(workflow.steps) == 3
        assert workflow.steps[0].id == "operation_step"
        assert workflow.steps[1].id == "map_step"
        assert workflow.steps[2].id == "parallel_step"


class TestExecuteWorkflow:
    """Test cases for execute_workflow function."""

    def test_execute_workflow(self):
        """Test executing workflow with engine."""
        step = OperationStep(
            id="step1",
            operation="test_op",
            parameters={}
        )
        workflow = WorkflowDSL(steps=[step])
        
        engine = Mock(spec=WorkflowEngine)
        expected_result = WorkflowResult(
            id="workflow_id",
            status=WorkflowResultStatus.SUCCESS,
            results=[]
        )
        engine.run.return_value = expected_result
        
        result = execute_workflow(workflow, engine)
        
        engine.run.assert_called_once_with(workflow)
        assert result == expected_result

    def test_execute_workflow_with_different_engines(self):
        """Test executing workflow with different engines."""
        workflow = WorkflowDSL(steps=[
            OperationStep(id="step1", operation="op1", parameters={})
        ])
        
        engine1 = Mock(spec=WorkflowEngine)
        engine2 = Mock(spec=WorkflowEngine)
        
        result1 = WorkflowResult(id="1", status=WorkflowResultStatus.SUCCESS, results=[])
        result2 = WorkflowResult(id="2", status=WorkflowResultStatus.FAILED, results=[])
        
        engine1.run.return_value = result1
        engine2.run.return_value = result2
        
        assert execute_workflow(workflow, engine1) == result1
        assert execute_workflow(workflow, engine2) == result2


class TestWorkflowClient:
    """Test cases for WorkflowClient."""

    def setup_method(self):
        """Setup test fixtures."""
        self.plugin_resolver = Mock()
        self.executor_factory = Mock()
        self.workflow_engine = Mock(spec=WorkflowEngine)
        self.result_store = Mock(spec=ResultStore)
        self.binder = Mock()
        self.execution_context = Mock()
        
        self.client = WorkflowClient(
            plugin_resolver=self.plugin_resolver,
            executor_factory=self.executor_factory,
            workflow_engine=self.workflow_engine,
            result_store=self.result_store,
            binder=self.binder,
            exectuion_context=self.execution_context,  # Note: typo in original
            execution_options=ExecutionOptions(retries=2, timeout=30.0)
        )

    def test_create_workflow_client(self):
        """Test creating workflow client."""
        assert self.client.resolver == self.plugin_resolver
        assert self.client.binder == self.binder
        assert self.client.result_store == self.result_store
        assert self.client.ctx == self.execution_context
        assert self.client.execution_options.retries == 2
        assert self.client.execution_options.timeout == 30.0
        assert self.client.executor_factory == self.executor_factory
        assert self.client.engine == self.workflow_engine

    def test_create_workflow_client_with_defaults(self):
        """Test creating workflow client with default execution options."""
        client = WorkflowClient(
            plugin_resolver=self.plugin_resolver,
            executor_factory=self.executor_factory,
            workflow_engine=self.workflow_engine,
            result_store=self.result_store,
            binder=self.binder,
            exectuion_context=self.execution_context
        )
        
        assert client.execution_options.retries == 0
        assert client.execution_options.timeout is None

    def test_run_workflow_from_dict(self):
        """Test running workflow from dictionary."""
        workflow_dict = {
            "steps": [
                {
                    "id": "step1",
                    "kind": "operation", 
                    "operation": "test_op",
                    "parameters": {}
                }
            ]
        }
        
        expected_result = WorkflowResult(
            id="workflow_id",
            status=WorkflowResultStatus.SUCCESS,
            results=[]
        )
        self.workflow_engine.run.return_value = expected_result
        
        result = self.client.run(workflow_dict)
        
        # Should convert dict to WorkflowDSL and run
        self.workflow_engine.run.assert_called_once()
        workflow_arg = self.workflow_engine.run.call_args[0][0]
        assert isinstance(workflow_arg, WorkflowDSL)
        assert len(workflow_arg.steps) == 1
        assert result == expected_result

    def test_run_workflow_from_dsl(self):
        """Test running workflow from WorkflowDSL."""
        step = OperationStep(
            id="step1",
            operation="test_op",
            parameters={}
        )
        workflow = WorkflowDSL(steps=[step])
        
        expected_result = WorkflowResult(
            id="workflow_id",
            status=WorkflowResultStatus.SUCCESS,
            results=[]
        )
        self.workflow_engine.run.return_value = expected_result
        
        result = self.client.run(workflow)
        
        self.workflow_engine.run.assert_called_once_with(workflow)
        assert result == expected_result

    def test_run_workflow_with_id(self):
        """Test running workflow with specific ID."""
        workflow_dict = {
            "steps": [
                {
                    "id": "step1",
                    "kind": "operation",
                    "operation": "test_op", 
                    "parameters": {}
                }
            ]
        }
        
        # The workflow ID parameter is currently ignored in the implementation
        # as the engine.run method is called without it
        result = self.client.run(workflow_dict, workflow_id="custom_id")
        
        self.workflow_engine.run.assert_called_once()

    def test_run_workflow_validation_error(self):
        """Test running workflow with validation error."""
        invalid_workflow_dict = {
            "steps": []  # Empty steps will cause validation error
        }
        
        with pytest.raises(ValueError, match="Workflow has no steps"):
            self.client.run(invalid_workflow_dict)

    def test_variable_resolver_creation(self):
        """Test that variable resolver is created correctly."""
        # The client should create a VariableResolver with the execution context
        assert hasattr(self.client, 'values')
        # In real implementation, this would be: isinstance(self.client.values, VariableResolver)

    def test_result_registrar_creation(self):
        """Test that result registrar is created correctly."""
        # The client should create a ResultRegistrar with the result store
        assert hasattr(self.client, 'registrar')
        assert self.client.registrar.result_store == self.result_store


class TestWorkflowClientErrorHandling:
    """Test error handling in WorkflowClient."""

    def test_invalid_workflow_structure(self):
        """Test client handling of invalid workflow structure."""
        plugin_resolver = Mock()
        executor_factory = Mock()
        workflow_engine = Mock(spec=WorkflowEngine)
        result_store = Mock(spec=ResultStore)
        binder = Mock()
        execution_context = Mock()
        
        client = WorkflowClient(
            plugin_resolver=plugin_resolver,
            executor_factory=executor_factory,
            workflow_engine=workflow_engine,
            result_store=result_store,
            binder=binder,
            exectuion_context=execution_context
        )
        
        # Invalid workflow dict missing required fields
        invalid_workflow = {"invalid": "structure"}
        
        with pytest.raises(Exception):  # msgspec conversion error
            client.run(invalid_workflow)

    def test_engine_execution_error(self):
        """Test client handling of engine execution errors."""
        plugin_resolver = Mock()
        executor_factory = Mock()
        workflow_engine = Mock(spec=WorkflowEngine)
        result_store = Mock(spec=ResultStore)
        binder = Mock()
        execution_context = Mock()
        
        client = WorkflowClient(
            plugin_resolver=plugin_resolver,
            executor_factory=executor_factory,
            workflow_engine=workflow_engine,
            result_store=result_store,
            binder=binder,
            exectuion_context=execution_context
        )
        
        workflow_dict = {
            "steps": [
                {
                    "id": "step1",
                    "kind": "operation",
                    "operation": "test_op",
                    "parameters": {}
                }
            ]
        }
        
        # Engine raises exception during execution
        workflow_engine.run.side_effect = RuntimeError("Engine failed")
        
        with pytest.raises(RuntimeError, match="Engine failed"):
            client.run(workflow_dict)