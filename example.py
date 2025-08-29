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

    result = client.run(workflow)
    print(result.to_dict())
