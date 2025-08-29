from plugins import SayHelloPlugin, NumberAdderPlugin

# Define a workflow
workflow = {
    "steps": [
        {
            "id": "step1",
            "kind": "operation",
            "operation": "say_hello",
            "parameters": {"name": "Ari"},
        },
        {
            "id": "step2",
            "kind": "operation",
            "operation": "add",
            "parameters": {"a": 3, "b": 5},
        }
    ]
}
