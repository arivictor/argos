# AroFlow Documentation

Welcome to AroFlow - a modern workflow orchestration framework that lets you **define workflows once and run them anywhere**.

## What is AroFlow?

AroFlow is a powerful workflow orchestration tool designed to streamline and automate complex processes across various execution environments. Whether you're managing data pipelines, automating deployments, or coordinating tasks across distributed systems, AroFlow provides a unified platform to define, execute, and monitor your workflows with ease and reliability.

## Key Features

- **Backend Agnostic**: Define workflows once, run on multiple backends (in-memory, Temporal, Celery, etc.)
- **Plugin System**: Extensible architecture with custom plugins for any operation
- **Simple API**: Clean, intuitive client API for defining and running workflows
- **Advanced Workflows**: Support for sequential, parallel, and map operations
- **Real-time Monitoring**: Built-in support for monitoring and logging
- **Error Handling**: Robust error handling with retries and graceful failures

## Quick Start

Let's get you up and running with AroFlow in just a few minutes!

### Installation

```bash
pip install aroflow
```

### Your First Workflow

Here's a simple example to get you started:

```python
import aroflow

# 1. Define a Plugin
class SayHelloPlugin(aroflow.PluginMixin):
    plugin_name = "say_hello"

    def execute(self, name: str) -> str:
        return f"Hello, {name}!"

# 2. Create a client and register your plugin
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(SayHelloPlugin)

# 3. Define a workflow
workflow = {
    "steps": [
        {
            "id": "step1",
            "kind": "operation", 
            "operation": "say_hello",
            "parameters": {"name": "AroFlow"},
        },
    ]
}

# 4. Execute the workflow
result = client.run(workflow)
print(result.to_yaml())
```

**Output:**
```yaml
id: workflow
status: success
results:
- id: step1
  kind: operation
  operation: say_hello
  parameters:
    name: AroFlow
  result: Hello, AroFlow!
  status: success
  error: null
error: null
```

## Core Concepts

### Plugins
Plugins are the building blocks of AroFlow. They encapsulate the business logic that your workflows will execute. Each plugin implements a single `execute` method and declares a unique `plugin_name`.

### Workflows
Workflows are declarative definitions of the tasks you want to execute. They consist of one or more steps that can run sequentially, in parallel, or mapped over data.

### Steps
Steps are individual units of work within a workflow. AroFlow supports three types of steps:

- **Operation Steps**: Execute a single plugin
- **Map Steps**: Execute a plugin over a list of inputs
- **Parallel Steps**: Execute multiple plugins concurrently

### Backends
Backends determine where and how your workflows execute. AroFlow abstracts the execution environment, so you can switch backends without changing your workflow definitions.

## Next Steps

Ready to dive deeper? Check out our comprehensive guides:

- **[Writing Plugins](plugins.md)** - Learn how to create custom plugins for your business logic
- **[Creating Workflows](workflows.md)** - Master the art of workflow composition
- **[Examples](examples.md)** - See real-world examples and patterns

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/arivictor/aroflow/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/arivictor/aroflow/discussions)

---

**Ready to orchestrate your workflows?** Let's start with [writing your first plugin](plugins.md)!
