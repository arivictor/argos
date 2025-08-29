#!/usr/bin/env python3
"""Example usage of the Argos workflow orchestration framework."""

import json
import msgspec

# Import from the new DDD structure
from argos import WorkflowClient, WorkflowResult

if __name__ == "__main__":
    my_workflow = {
        "steps": [
            {
                "id": "step1",
                "kind": "operation",
                "operation": "say_hello",
                "parameters": {"name": "Ari"},
            },
            {
                "id": "step2",
                "kind": "map",
                "inputs": [1, 2, 3, 4, 5],
                "iterator": "item",
                "mode": "parallel",
                "operation": {
                    "id": "step2_op",
                    "kind": "operation",
                    "operation": "add",
                    "parameters": {"a": "{{item}}", "b": 10},
                },
            },
            {
                "id": "step3",
                "kind": "map",
                "inputs": ["ari", "bob", "carol", 1],
                "iterator": "item",
                "mode": "parallel",
                "operation": {
                    "id": "step3_op",
                    "kind": "operation",
                    "operation": "say_hello",
                    "parameters": {"name": "{{item}}"},
                },
            },
            {
                "id": "step4",
                "kind": "parallel",
                "operations": [
                    {
                        "id": "step4_op1",
                        "kind": "operation",
                        "operation": "say_hello",
                        "parameters": {"name": "Parallel 1"},
                    },
                    {
                        "id": "step4_op2",
                        "kind": "operation",
                        "operation": "say_hello",
                        "parameters": {"name": "${step3_op_2.result}"},
                    },
                ],
            },
            {
                "id": "step5",
                "kind": "operation",
                "operation": "throw_exception",
                "parameters": {"message": "Intentional failure to test retries"},
                "retries": 3, 
                "timeout": 5,
                "fail_workflow": False,
            },
            {
                "id": "step6",
                "kind": "operation",
                "operation": "sleep",
                "parameters": {"seconds": 3},
                "retries": 1, 
                "timeout": 10
            }
        ]
    }
    client = WorkflowClient()
    result: WorkflowResult = client.run(my_workflow)
    print("Workflow results:")
    print(json.dumps(msgspec.to_builtins(result), indent=2))
    print(result.status)