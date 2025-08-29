"""This module implements a plugin-driven workflow DSL engine following SOLID principles.
It provides plugin registration, resolution, parameter binding, and execution of workflow steps.
"""

from abc import ABC, abstractmethod
import json
from typing import Any, Literal, Union, get_origin, get_args, Optional
import inspect
import typing
import msgspec
from msgspec import structs
import re


# domain/plugins/base.py
class PluginBase:
    """Base class for all plugins. Enforces 'execute' method and registers subclasses."""

    _plugins = []

    def __init_subclass__(cls, **kwargs):
        """Registers subclass and ensures 'execute' method is defined."""
        super().__init_subclass__(**kwargs)

        if "execute" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a 'execute' method")

        # print(f"Registering: {cls.__name__}")
        PluginBase._plugins.append(cls)

    def execute(self, *args, **kwargs) -> Any:
        """Abstract execute method to be implemented by plugins."""
        raise NotImplementedError("Plugins must implement the execute method")


class NumberAdderPlugin(PluginBase):
    """Plugin that adds two integers and returns the sum."""

    plugin_name = "add"

    def execute(self, a: int, b: int) -> int:
        """Adds two integers a and b."""
        return a + b


class SayHelloPlugin(PluginBase):
    """Plugin that returns a greeting string for a given name."""

    plugin_name = "say_hello"

    def execute(self, name: str) -> str:
        """Returns a greeting message for the given name."""
        return f"Hello, {name}!"


# domain/plugins/registry.py
def get_plugins() -> list[type[PluginBase]]:
    """Returns a list of all registered plugin classes."""
    return PluginBase._plugins


# domain/plugins/resolver.py
# Plugin Resolution and DI
class PluginResolver(ABC):
    """Abstract base class defining plugin resolution interface."""

    @abstractmethod
    def resolve(self, name: str) -> PluginBase:
        """Resolves and returns a plugin instance by its name."""
        ...


# infrastructure/plugins/in_memory.py
class InMemoryPluginResolver(PluginResolver):
    """Resolves plugins from an in-memory registry."""

    def __init__(self, plugins: list[type[PluginBase]] | None = None):
        """Initializes resolver with optional plugin list."""
        self._registry: dict[str, type[PluginBase]] = {}
        if plugins is None:
            plugins = get_plugins()
        for cls in plugins:
            key = getattr(cls, "plugin_name", cls.__name__)
            self._registry[key] = cls

    def resolve(self, name: str) -> PluginBase:
        """Returns a plugin instance matching the given name, raises KeyError if not found."""
        try:
            cls = self._registry[name]
        except KeyError:
            raise KeyError(f"No plugin registered for operation '{name}'")
        return cls()


# infrastructure/binding/parameter_binder.py
# Parameter binding based on plugin type hints
class ParameterBinder:
    """Binds parameters (accepting mixed types) to plugin execute method arguments with type coercion.

    The bind method accepts parameter dictionaries with mixed-type values, and will coerce strings to the target types when necessary.
    """

    def bind(self, plugin: PluginBase, params: dict[str, Any]) -> dict[str, Any]:
        """Binds and coerces parameters (accepting mixed-type values) to the plugin's execute method signature.

        Accepts a parameter dictionary with mixed-type values; will coerce strings to the target types when necessary.
        """
        sig = inspect.signature(plugin.execute)
        hints = typing.get_type_hints(plugin.execute, include_extras=False)
        bound: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name not in params:
                continue
            target = hints.get(name, Any)
            bound[name] = self._coerce(params[name], target)
        return bound

    def _coerce(self, value: Any, target_type: Any) -> Any:
        """Coerces a string value to the target type, handling Optional and Union types."""
        # If already the right type, return as-is
        if (
            target_type is Any or isinstance(value, target_type)
            if isinstance(target_type, type)
            else False
        ):
            return value
        # Unwrap Optional[T] and Union[T, ...]
        origin = get_origin(target_type)
        if origin is Optional:
            inner = get_args(target_type)[0]
            return self._coerce(value, inner)
        if origin is Union:
            for t in get_args(target_type):
                try:
                    return self._coerce(value, t)
                except Exception:
                    continue
            return value
        # Primitive coercions from string
        if isinstance(value, str):
            if target_type is int:
                return int(value)
            if target_type is float:
                return float(value)
            if target_type is bool:
                v = value.strip().lower()
                if v in {"true", "1", "yes", "y"}:
                    return True
                if v in {"false", "0", "no", "n"}:
                    return False
            # Leave as string for anything else
            return value
        return value


# infrastructure/binding/placeholder_resolver.py
class ExecutionContext:
    """Holds results of previously executed steps, addressable by step id."""

    def __init__(self):
        self.results: dict[str, Any] = {}


class PlaceholderResolver:
    """Resolves ${stepId[.field][[index]].field} placeholders in arbitrarily nested data structures.

    Rules:
    - If a string is exactly a single placeholder like "${step1}", return the referenced value as-is (preserve type).
    - Otherwise, perform string interpolation by converting referenced values to str.
    - Supported paths: `${id}`, `${id.result}`, `${id.results}`, `${id.results[0]}`, `${id.results[0].result}`.
    """

    _pattern = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, ctx: ExecutionContext):
        self.ctx = ctx

    def resolve_any(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self.resolve_any(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve_any(v) for v in value]
        if isinstance(value, str):
            return self._resolve_string(value)
        return value

    def _resolve_string(self, s: str) -> Any:
        # Exact single-token match => return raw value to preserve type
        m = self._pattern.fullmatch(s.strip())
        if m:
            return self._lookup_token(m.group(1))

        # Interpolate within string
        def repl(match: re.Match) -> str:
            val = self._lookup_token(match.group(1))
            return str(val)

        return self._pattern.sub(repl, s)

    def _lookup_token(self, token: str) -> Any:
        # token grammar: id(.field|[index])*
        parts = re.findall(r"[^.\[\]]+|\[\d+\]", token)
        if not parts:
            return token
        step_id = parts[0]
        if step_id not in self.ctx.results:
            raise KeyError(f"Unknown step id in placeholder: {step_id}")
        current = self.ctx.results[step_id]
        # Convert msgspec Structs to builtins for traversal
        current = msgspec.to_builtins(current)
        # Walk remaining parts
        for p in parts[1:]:
            if p.startswith("["):
                idx = int(p[1:-1])
                current = current[idx]
            else:
                current = current[p]
        return current


# domain/model/steps.py
# Workflow Engine Domain


class Step(msgspec.Struct, tag_field="kind"):
    """Base class for workflow steps with a unique identifier."""

    id: str


class OperationStep(Step, kw_only=True, tag="operation"):
    """Represents an operation step with an operation name and parameters.

    The parameters dictionary may contain any JSON value; accepts mixed types.
    """

    operation: str
    parameters: dict[str, Any]


class ParallelStep(Step, kw_only=True, tag="parallel"):
    """Represents a parallel step that runs multiple operation steps concurrently."""

    operations: list[OperationStep]


class MapStep(Step, kw_only=True, tag="map"):
    """Represents a map step that iterates over inputs applying an operation step."""

    inputs: list[Any]
    iterator: str
    mode: Literal["sequential", "parallel"] = "sequential"
    operation: OperationStep


StepTypes = Union[OperationStep, MapStep, ParallelStep]


# domain/model/workflow.py


class WorkflowDSL(msgspec.Struct):
    """Represents a workflow composed of multiple steps."""

    steps: list[StepTypes]


# domain/model/results.py
class OperationResult(msgspec.Struct):
    """Result of a single operation execution."""

    id: str
    kind: Literal["operation"]
    operation: str
    parameters: dict[str, Any]
    result: Any


class MapItemResult(msgspec.Struct):
    """Result of applying an operation to a single input item in a map step."""

    input: Any
    operation: str
    parameters: dict[str, Any]
    result: Any


class MapResult(msgspec.Struct):
    """Aggregated result of a map step."""

    id: str
    kind: Literal["map"]
    mode: Literal["sequential", "parallel"]
    iterator: str
    inputs: list[Any]
    results: list[MapItemResult]


class ParallelOpResult(msgspec.Struct):
    """Result of one operation within a parallel step."""

    id: str
    operation: str
    parameters: dict[str, Any]
    result: Any


class ParallelResult(msgspec.Struct):
    """Aggregated result of a parallel step."""

    id: str
    kind: Literal["parallel"]
    results: list[ParallelOpResult]


class ValidateWorkflowStep:
    def execute(self, step: StepTypes) -> bool:
        """Validates a single workflow step.

        Args:
            step: The workflow step to validate.

        Returns:
            True if the step is valid, raises ValueError otherwise.
        """
        if not step.id or not isinstance(step.id, str):
            raise ValueError(f"Invalid step id: {step.id}")

        if isinstance(step, MapStep):
            if not step.inputs:
                raise ValueError(f"Map step {step.id} has empty inputs")
            if not step.iterator.isidentifier():
                raise ValueError(
                    f"Invalid iterator name in map step {step.id}: {step.iterator}"
                )
            if not isinstance(step.operation, OperationStep):
                raise ValueError(
                    f"Map step {step.id} operation must be an OperationStep"
                )
        if isinstance(step, ParallelStep):
            if not step.operations:
                raise ValueError(f"Parallel step {step.id} has no operations")
            for op in step.operations:
                if not isinstance(op, OperationStep):
                    raise ValueError(
                        f"Parallel step {step.id} contains non-operation step"
                    )
        return True


def validate_workflow(data: WorkflowDSL) -> bool:
    """Validates the workflow structure and contents.

    Args:
        data: The WorkflowDSL instance to validate.

    Returns:
        True if the workflow is valid, raises ValueError otherwise.
    """
    if not data.steps:
        raise ValueError("Workflow has no steps")
    seen_ids = set()
    step_validator = ValidateWorkflowStep()
    for step in data.steps:
        if step.id in seen_ids:
            raise ValueError(f"Duplicate step id found: {step.id}")
        seen_ids.add(step.id)
        step_validator.execute(step)
    return True


# application/services/decoder.py
def load_workflow(data: dict) -> WorkflowDSL:
    """Decodes and validates a workflow from a Python dictionary.

    Args:
        data: The workflow data as a dictionary.

    Returns:
        A WorkflowDSL instance representing the validated workflow.
    """
    workflow = msgspec.convert(data, type=WorkflowDSL)
    validate_workflow(workflow)
    return workflow


# application/engine.py
class WorkflowEngine(ABC):
    """Abstract base class defining the workflow engine interface."""

    @abstractmethod
    def run(self, workflow: WorkflowDSL) -> Any:
        """Runs the given workflow."""
        ...


# application/executors/base.py
class StepExecutor(ABC):
    """Abstract executor interface for executing workflow steps and returning results."""

    @abstractmethod
    def execute(self, step: Any) -> Any:
        """Execute a workflow step and return its result."""
        ...


# application/executors/operation.py
class OperationExecutor(StepExecutor):
    """Executes an operation step by resolving and running the corresponding plugin."""

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values

    def execute(self, step: OperationStep):
        """Executes the operation step and returns a structured result."""
        # Resolve placeholders in parameters
        resolved_params = self.values.resolve_any(step.parameters)
        step = structs.replace(step, parameters=resolved_params)
        print(f"Executing operation {step.operation} with parameters {step.parameters}")
        plugin = self.resolver.resolve(step.operation)
        bound = self.binder.bind(plugin, step.parameters)
        result = plugin.execute(**bound)
        print(f"Result: {result}")
        return OperationResult(
            id=step.id,
            kind="operation",
            operation=step.operation,
            parameters=step.parameters,
            result=result,
        )


class ParallelOperationExecutor(StepExecutor):
    """Executes multiple operation steps in parallel using threads."""

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values

    def execute(self, step: ParallelStep):
        """Executes all operation steps in parallel and returns a structured result."""
        print(
            f"Executing parallel operations: {[op.operation for op in step.operations]}"
        )
        from concurrent.futures import ThreadPoolExecutor

        def run_op(op: OperationStep):
            res = OperationExecutor(self.resolver, self.binder, self.values).execute(op)
            return ParallelOpResult(
                id=op.id,
                operation=op.operation,
                parameters=op.parameters,
                result=res.result,
            )

        with ThreadPoolExecutor() as executor:
            inner = list(executor.map(run_op, step.operations))
        return ParallelResult(id=step.id, kind="parallel", results=inner)


# application/executors/map.py
class SequentialMapExecutor(StepExecutor):
    """Executes a map step sequentially over inputs."""

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values

    def execute(self, step: MapStep):
        """Executes the map step sequentially and returns a structured result."""
        print(
            f"Executing sequential map over {step.inputs} with iterator {step.iterator}"
        )
        base_op = step.operation
        # Resolve placeholders in base operation parameters
        base_params = self.values.resolve_any(base_op.parameters)
        results = []
        for item in step.inputs:
            new_params = {}
            for k, v in base_params.items():
                if v == "{{" + step.iterator + "}}":
                    new_params[k] = item
                else:
                    new_params[k] = v
            op = structs.replace(base_op, parameters=new_params)
            res = OperationExecutor(self.resolver, self.binder, self.values).execute(op)
            results.append(
                MapItemResult(
                    input=item,
                    operation=op.operation,
                    parameters=op.parameters,
                    result=res.result,
                )
            )
        return MapResult(
            id=step.id,
            kind="map",
            mode=step.mode,
            iterator=step.iterator,
            inputs=step.inputs,
            results=results,
        )


# application/executors/map.py
class ParallelMapExecutor(StepExecutor):
    """Executes a map step in parallel over inputs using threads."""

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values

    def execute(self, step: MapStep):
        """Executes the map step in parallel and returns a structured result."""
        print(
            f"Executing parallel map over {step.inputs} with iterator {step.iterator}"
        )
        base_op = step.operation
        # Resolve placeholders in base operation parameters
        base_params = self.values.resolve_any(base_op.parameters)
        from concurrent.futures import ThreadPoolExecutor

        def run_op(item):
            new_params = {}
            for k, v in base_params.items():
                if v == "{{" + step.iterator + "}}":
                    new_params[k] = item
                else:
                    new_params[k] = v
            op = structs.replace(base_op, parameters=new_params)
            res = OperationExecutor(self.resolver, self.binder, self.values).execute(op)
            return MapItemResult(
                input=item,
                operation=op.operation,
                parameters=op.parameters,
                result=res.result,
            )

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(run_op, step.inputs))
        return MapResult(
            id=step.id,
            kind="map",
            mode=step.mode,
            iterator=step.iterator,
            inputs=step.inputs,
            results=results,
        )


# application/executors/map.py
class MapExecutor(StepExecutor):
    """Delegates map step execution to sequential or parallel executors based on mode."""

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values

    def execute(self, step: MapStep):
        """Executes the map step in the configured mode and returns aggregated results."""
        print(f"Executing map over {step.inputs} with iterator {step.iterator}")
        if step.mode == "parallel":
            return ParallelMapExecutor(self.resolver, self.binder, self.values).execute(
                step
            )
        else:
            return SequentialMapExecutor(
                self.resolver, self.binder, self.values
            ).execute(step)


# application/engine.py
class InMemoryWorkflowEngine(WorkflowEngine):
    """Workflow engine that executes steps in memory using executors."""

    def __init__(self, resolver: PluginResolver, binder: ParameterBinder):
        """Initializes with a plugin resolver and parameter binder."""
        self.resolver = resolver
        self.binder = binder
        self.ctx = ExecutionContext()
        self.values = PlaceholderResolver(self.ctx)

    def run(self, workflow: WorkflowDSL) -> list[Any]:
        """Executes each step of the workflow in order and returns a list of step results."""
        results = []
        for step in workflow.steps:
            if isinstance(step, OperationStep):
                step_result = OperationExecutor(
                    self.resolver, self.binder, self.values
                ).execute(step)
            elif isinstance(step, MapStep):
                step_result = MapExecutor(
                    self.resolver, self.binder, self.values
                ).execute(step)
            elif isinstance(step, ParallelStep):
                step_result = ParallelOperationExecutor(
                    self.resolver, self.binder, self.values
                ).execute(step)
            else:
                raise ValueError(f"Unknown step type: {type(step)}")
            self.ctx.results[step.id] = step_result
            results.append(step_result)
        return results


# interfaces/cli/main.py
def execute_workflow(workflow: WorkflowDSL, engine: WorkflowEngine):
    """Runs the given workflow using the specified workflow engine and returns results."""
    return engine.run(workflow)


# interfaces/cli/main.py
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
                        "parameters": {"name": "Parallel 2"},
                    },
                ],
            },
        ]
    }

    resolver = InMemoryPluginResolver()
    binder = ParameterBinder()
    results = execute_workflow(
        load_workflow(my_workflow), InMemoryWorkflowEngine(resolver, binder)
    )
    print("Workflow results:")
    print(json.dumps(msgspec.to_builtins(results), indent=2))
