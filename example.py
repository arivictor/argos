import json

import argos
from argos.domain.entity import WorkflowResult
from plugins import SayHelloPlugin



if __name__ == "__main__":
        # Create client for in-memory execution
    client = argos.create(argos.BackendType.IN_MEMORY)

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

    result: WorkflowResult = client.run(workflow)
    print(result.to_dict())
    print(result.to_json())
    print(result.to_yaml())