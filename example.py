import argos
from argos import WorkflowResult


class SayHelloPlugin(argos.PluginMixin):
    plugin_name = "say_hello"

    def execute(self, name: str) -> str:
        return f"Hello, {name}!"


if __name__ == "__main__":
    # Create client for in-memory execution
    client = argos.create(argos.BackendType.IN_MEMORY)

    # Register plugins
    client.plugin(SayHelloPlugin)

    workflow = {
        "steps": [
            {
                "id": "step1",
                "kind": "map",
                "mode": "sequential",
                "iterator": "name",
                "inputs": ["Ari", "Budi", "Cici"],
                "operation": {
                    "id": "step1",
                    "kind": "operation",
                    "operation": "say_hello",
                    "parameters": {"name": "${name}"},
                },
            },
        ]
    }

    workflow_id = "example_workflow_001"
    result: WorkflowResult = client.run(workflow, workflow_id)

    print(result.to_dict())
    print(result.to_json())
    print(result.to_yaml())
