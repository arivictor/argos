"""
Integration tests for the entire AroFlow framework.

This module tests end-to-end functionality across all layers.
"""

from aroflow import BackendType, PluginMixin, WorkflowResult, create
from aroflow.domain.value_object import WorkflowResultStatus


class SimplePlugin(PluginMixin):
    """Simple test plugin."""

    plugin_name = "simple"

    def execute(self, message: str) -> str:
        return f"Hello, {message}!"


class MathPlugin(PluginMixin):
    """Math operations plugin."""

    plugin_name = "math"

    def execute(self, operation: str, a: int, b: int) -> int:
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        elif operation == "subtract":
            return a - b
        else:
            raise ValueError(f"Unknown operation: {operation}")


class DataProcessorPlugin(PluginMixin):
    """Data processing plugin."""

    plugin_name = "data_processor"

    def execute(self, data: list, operation: str = "sum") -> dict:
        if operation == "sum":
            result = sum(data)
        elif operation == "max":
            result = max(data)
        elif operation == "min":
            result = min(data)
        else:
            import logging

            logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.DEBUG)
            logger.error(f"Unknown operation: {operation}")
            raise ValueError(f"Unknown operation: {operation}")

        return {"operation": operation, "input": data, "result": result, "count": len(data)}


class TestIntegration:
    """Integration test cases."""

    def test_simple_workflow_execution(self):
        """Test simple workflow execution end-to-end."""
        # Create client
        client = create(BackendType.IN_MEMORY)

        # Register plugin
        client.plugin(SimplePlugin)

        # Define workflow
        workflow = {
            "steps": [
                {"id": "greeting", "kind": "operation", "operation": "simple", "parameters": {"message": "World"}}
            ]
        }

        # Execute workflow
        result = client.run(workflow)

        # Verify result
        assert isinstance(result, WorkflowResult)
        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 1
        assert result.results[0].id == "greeting"
        assert result.results[0].result == "Hello, World!"

    def test_multi_step_workflow_with_dependencies(self):
        """Test multi-step workflow with step dependencies."""
        # Create client with multiple plugins
        client = create(BackendType.IN_MEMORY).plugin(MathPlugin).plugin(SimplePlugin)

        # Define workflow with dependencies
        workflow = {
            "steps": [
                {
                    "id": "add_numbers",
                    "kind": "operation",
                    "operation": "math",
                    "parameters": {"operation": "add", "a": 10, "b": 5},
                },
                {
                    "id": "multiply_result",
                    "kind": "operation",
                    "operation": "math",
                    "parameters": {"operation": "multiply", "a": "${add_numbers.result}", "b": 2},
                },
                {
                    "id": "announce_result",
                    "kind": "operation",
                    "operation": "simple",
                    "parameters": {"message": "The final result is ${multiply_result.result}"},
                },
            ]
        }

        # Execute workflow
        result = client.run(workflow)

        # Print debug info if failed
        if result.status != WorkflowResultStatus.SUCCESS:
            print(f"Workflow failed with status: {result.status}")
            print(f"Error: {result.error}")
            for i, step_result in enumerate(result.results):
                print(f"Step {i}: {step_result.id} - {getattr(step_result, 'error', 'No error')}")

        # Verify result
        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 3

        # Check intermediate results
        assert result.results[0].result == 15  # 10 + 5
        assert result.results[1].result == 30  # 15 * 2
        assert result.results[2].result == "Hello, The final result is 30!"

    def test_map_step_workflow(self):
        """Test workflow with map step."""
        client = create(BackendType.IN_MEMORY).plugin(DataProcessorPlugin)

        workflow = {
            "steps": [
                {
                    "id": "process_datasets",
                    "kind": "map",
                    "inputs": [[1, 2, 3, 4, 5], [10, 20, 30], [100, 200]],
                    "iterator": "dataset",
                    "operation": {
                        "id": "process_single_dataset",
                        "kind": "operation",
                        "operation": "data_processor",
                        "parameters": {"data": "${dataset}", "operation": "sum"},
                    },
                }
            ]
        }

        # Execute workflow
        result = client.run(workflow)

        print(result.to_dict())

        # Verify result
        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 1

        map_result = result.results[0]
        assert map_result.kind == "map"
        assert len(map_result.results) == 3

        # Check individual map results
        assert map_result.results[0].result["result"] == 15  # sum([1,2,3,4,5])
        assert map_result.results[1].result["result"] == 60  # sum([10,20,30])
        assert map_result.results[2].result["result"] == 300  # sum([100,200])

    def test_parallel_step_workflow(self):
        """Test workflow with parallel step."""
        client = create(BackendType.IN_MEMORY).plugin(MathPlugin)

        workflow = {
            "steps": [
                {
                    "id": "parallel_calculations",
                    "kind": "parallel",
                    "operations": [
                        {
                            "id": "calc1",
                            "kind": "operation",
                            "operation": "math",
                            "parameters": {"operation": "add", "a": 100, "b": 50},
                        },
                        {
                            "id": "calc2",
                            "kind": "operation",
                            "operation": "math",
                            "parameters": {"operation": "multiply", "a": 20, "b": 3},
                        },
                        {
                            "id": "calc3",
                            "kind": "operation",
                            "operation": "math",
                            "parameters": {"operation": "subtract", "a": 1000, "b": 250},
                        },
                    ],
                }
            ]
        }

        # Execute workflow
        result = client.run(workflow)

        # Verify result
        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 1

        parallel_result = result.results[0]
        assert parallel_result.kind == "parallel"
        assert len(parallel_result.results) == 3

        # Check individual parallel results
        results_by_id = {r.id: r.result for r in parallel_result.results}
        assert results_by_id["calc1"] == 150  # 100 + 50
        assert results_by_id["calc2"] == 60  # 20 * 3
        assert results_by_id["calc3"] == 750  # 1000 - 250

    # def test_complex_workflow_with_all_step_types(self):
    #     """Test complex workflow combining all step types."""
    #     client = (create(BackendType.IN_MEMORY)
    #               .plugin(MathPlugin)
    #               .plugin(DataProcessorPlugin)
    #               .plugin(SimplePlugin))

    #     workflow = {
    #         "steps": [
    #             # Step 1: Parallel calculations
    #             {
    #                 "id": "initial_calculations",
    #                 "kind": "parallel",
    #                 "operations": [
    #                     {
    #                         "id": "base_value",
    #                         "kind": "operation",
    #                         "operation": "math",
    #                         "parameters": {
    #                             "operation": "multiply",
    #                             "a": 10,
    #                             "b": 5
    #                         }
    #                     },
    #                     {
    #                         "id": "multiplier",
    #                         "kind": "operation",
    #                         "operation": "math",
    #                         "parameters": {
    #                             "operation": "add",
    #                             "a": 2,
    #                             "b": 3
    #                         }
    #                     }
    #                 ]
    #             },
    #             # Step 2: Map step processing arrays
    #             {
    #                 "id": "process_arrays",
    #                 "kind": "map",
    #                 "inputs": [
    #                     [1, 2, 3],
    #                     [4, 5, 6],
    #                     [7, 8, 9]
    #                 ],
    #                 "iterator": "array",
    #                 "operation": {
    #                     "id": "sum_array",
    #                     "kind": "operation",
    #                     "operation": "data_processor",
    #                     "parameters": {
    #                         "data": "${array}",
    #                         "operation": "sum"
    #                     }
    #                 }
    #             },
    #             # Step 3: Combine results
    #             {
    #                 "id": "combine_results",
    #                 "kind": "operation",
    #                 "operation": "math",
    #                 "parameters": {
    #                     "operation": "multiply",
    #                     "a": "${initial_calculations.results[0].result}",  # base_value
    #                     "b": "${initial_calculations.results[1].result}"   # multiplier
    #                 }
    #             },
    #             # Step 4: Final announcement
    #             {
    #                 "id": "final_message",
    #                 "kind": "operation",
    #                 "operation": "simple",
    #                 "parameters": {
    #                     "message": "Workflow completed with result ${combine_results.result}"
    #                 }
    #             }
    #         ]
    #     }

    #     # Execute workflow
    #     result = client.run(workflow)

    #     # Verify result
    #     assert result.status == WorkflowResultStatus.SUCCESS
    #     assert len(result.results) == 4

    #     # Check results
    #     initial_parallel = result.results[0]
    #     assert initial_parallel.kind == "parallel"

    #     process_map = result.results[1]
    #     assert process_map.kind == "map"
    #     assert len(process_map.results) == 3

    #     combine_result = result.results[2]
    #     assert combine_result.result == 250  # 50 * 5

    #     final_message = result.results[3]
    #     assert "250" in final_message.result

    def test_workflow_error_handling(self):
        """Test workflow error handling."""
        client = create(BackendType.IN_MEMORY).plugin(MathPlugin)

        # Workflow with invalid operation
        workflow = {
            "steps": [
                {
                    "id": "invalid_operation",
                    "kind": "operation",
                    "operation": "math",
                    "parameters": {
                        "operation": "divide",  # Not supported
                        "a": 10,
                        "b": 2,
                    },
                }
            ]
        }

        # Execute workflow
        result = client.run(workflow)

        # Should have failed
        assert result.status == WorkflowResultStatus.FAILED
        assert result.error is not None

    def test_workflow_json_yaml_serialization(self):
        """Test workflow result serialization."""
        client = create(BackendType.IN_MEMORY).plugin(SimplePlugin)

        workflow = {
            "steps": [
                {
                    "id": "test_step",
                    "kind": "operation",
                    "operation": "simple",
                    "parameters": {"message": "Serialization Test"},
                }
            ]
        }

        result = client.run(workflow)

        # Test JSON serialization
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "Serialization Test" in json_str

        # Test YAML serialization
        yaml_str = result.to_yaml()
        assert isinstance(yaml_str, str)
        assert "Serialization Test" in yaml_str

        # Test dict conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "results" in result_dict

    def test_plugin_registration_and_discovery(self):
        """Test plugin registration and discovery."""
        # Clear existing plugins
        from aroflow.domain.port import PluginBase

        original_plugins = PluginBase._plugins.copy()
        PluginBase._plugins.clear()

        try:
            # Register plugins directly with PluginBase (since they auto-register)
            class TempPlugin1(PluginBase):
                def execute(self) -> str:
                    return "temp1"

            class TempPlugin2(PluginBase):
                def execute(self) -> str:
                    return "temp2"

            # Create client after plugins are registered
            client = create(BackendType.IN_MEMORY)

            # Should now have plugins
            available = client.get_available_plugins()
            assert len(available) == 2
            assert TempPlugin1 in available
            assert TempPlugin2 in available

        finally:
            # Restore original plugins
            PluginBase._plugins = original_plugins

    def test_workflow_with_custom_id(self):
        """Test workflow execution with custom workflow ID."""
        client = create(BackendType.IN_MEMORY).plugin(SimplePlugin)

        workflow = {
            "steps": [
                {
                    "id": "test_step",
                    "kind": "operation",
                    "operation": "simple",
                    "parameters": {"message": "Custom ID Test"},
                }
            ]
        }

        custom_id = "my_custom_workflow_123"
        result = client.run(workflow, workflow_id=custom_id)

        assert result.id == custom_id
        assert result.status == WorkflowResultStatus.SUCCESS

    def test_end_to_end_example_from_readme(self):
        """Test the example from the README to ensure it works."""

        # This should match the example.py file
        class SayHelloPlugin(PluginMixin):
            plugin_name = "say_hello"

            def execute(self, name: str) -> str:
                return f"Hello, {name}!"

        # Create client for in-memory execution
        client = create(BackendType.IN_MEMORY)

        # Register plugins
        client.plugin(SayHelloPlugin)

        workflow = {
            "steps": [
                {
                    "id": "step1",
                    "kind": "operation",
                    "operation": "say_hello",
                    "parameters": {"name": "Ari"},
                },
            ]
        }

        workflow_id = "example_workflow_001"
        result = client.run(workflow, workflow_id)

        # Verify it works as expected
        assert result.id == workflow_id
        assert result.status == WorkflowResultStatus.SUCCESS
        assert len(result.results) == 1
        assert result.results[0].result == "Hello, Ari!"

        # Test serialization methods work
        result_dict = result.to_dict()
        json_str = result.to_json()
        yaml_str = result.to_yaml()

        assert isinstance(result_dict, dict)
        assert isinstance(json_str, str)
        assert isinstance(yaml_str, str)
        assert "Hello, Ari!" in json_str
        assert "Hello, Ari!" in yaml_str
