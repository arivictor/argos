#!/usr/bin/env python3
"""
Demo script showing SQLite backend workflow query and delete functionality.
"""

import aroflow
from aroflow.backend import BackendType
from aroflow.domain.port import PluginBase


class DataProcessorPlugin(PluginBase):
    """Example plugin for data processing."""
    
    plugin_name = "data_processor"
    
    def execute(self, operation: str, data: str) -> str:
        if operation == "uppercase":
            return data.upper()
        elif operation == "lowercase":
            return data.lower()
        elif operation == "reverse":
            return data[::-1]
        else:
            return f"Unknown operation: {operation}"


def main():
    print("=== SQLite Backend Workflow Query & Delete Demo ===\n")
    
    # Create client with SQLite backend (in-memory for demo)
    client = aroflow.create(BackendType.SQLITE, plugins=[DataProcessorPlugin])
    
    # Define a workflow
    workflow = {
        "steps": [
            {
                "id": "step1",
                "kind": "operation",
                "operation": "data_processor",
                "parameters": {"operation": "uppercase", "data": "hello world"}
            },
            {
                "id": "step2", 
                "kind": "operation",
                "operation": "data_processor",
                "parameters": {"operation": "reverse", "data": "hello world"}
            }
        ]
    }
    
    print("1. Initially no workflows exist:")
    print(f"   Workflows: {client.list_workflows()}\n")
    
    print("2. Executing workflows with specific IDs...")
    
    # Execute workflows with specific IDs
    result1 = client.run(workflow, workflow_id="workflow_001")
    print(f"   Executed workflow_001: {result1.status.value}")
    
    result2 = client.run(workflow, workflow_id="workflow_002") 
    print(f"   Executed workflow_002: {result2.status.value}")
    
    # Execute another workflow with different data
    workflow2 = {
        "steps": [
            {
                "id": "step1",
                "kind": "operation", 
                "operation": "data_processor",
                "parameters": {"operation": "lowercase", "data": "PYTHON ROCKS"}
            }
        ]
    }
    result3 = client.run(workflow2, workflow_id="workflow_003")
    print(f"   Executed workflow_003: {result3.status.value}\n")
    
    print("3. List all workflow IDs:")
    workflows = client.list_workflows()
    print(f"   Workflows: {workflows}\n")
    
    print("4. Query specific workflow results:")
    for workflow_id in workflows:
        results = client.get_workflow(workflow_id)
        print(f"   {workflow_id}:")
        for step_id, step_result in results.items():
            print(f"     {step_id}: {step_result['result']}")
    print()
    
    print("5. Delete workflow_002:")
    deleted = client.delete_workflow("workflow_002")
    print(f"   Deleted: {deleted}")
    print(f"   Remaining workflows: {client.list_workflows()}\n")
    
    print("6. Try to query deleted workflow:")
    try:
        client.get_workflow("workflow_002")
        print("   ERROR: Should have failed!")
    except KeyError as e:
        print(f"   Expected error: {e}\n")
    
    print("7. Try to delete non-existent workflow:")
    deleted = client.delete_workflow("nonexistent")
    print(f"   Deleted nonexistent workflow: {deleted}\n")
    
    print("8. Final state:")
    print(f"   Remaining workflows: {client.list_workflows()}")
    for workflow_id in client.list_workflows():
        results = client.get_workflow(workflow_id)
        print(f"   {workflow_id} has {len(results)} steps")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()