import aroflow


class SayHelloPlugin(aroflow.PluginMixin):
    plugin_name = "say_hello"

    def execute(self, name: str) -> str:
        return f"Hello, {name}!"


if __name__ == "__main__":
    import aroflow
    from aroflow.backend import BackendType

    # File-based SQLite (persistent storage)
    client = aroflow.create(BackendType.SQLITE, plugins=[SayHelloPlugin], db_path="workflows.db")

    # Execute workflows exactly the same way
    workflow = {
        "steps": [{"id": "process_data", "kind": "operation", "operation": "say_hello", "parameters": {"name": "ari"}}]
    }

    my_id = "workflow_1"
    result = client.run(workflow, my_id)
    print(result.to_dict())

    workflows = client.list_workflows()
    print(f"Stored workflows: {workflows}")

    my_workflow = client.get_workflow(my_id)
