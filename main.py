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


def get_plugins() -> list[type[PluginBase]]:
    """Returns a list of all registered plugin classes."""
    return PluginBase._plugins


class PluginResolver(ABC):
    """Abstract base class defining plugin resolution interface."""

    @abstractmethod
    def resolve(self, name: str) -> PluginBase:
        """Resolves and returns a plugin instance by its name."""
        ...


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



# --- ResultStore abstraction ---
from typing import Any
class ResultStore(ABC):
    @abstractmethod
    def set(self, key: str, value: Any):
        ...
    @abstractmethod
    def get(self, key: str) -> Any:
        ...

class InMemoryResultStore(ResultStore):
    def __init__(self):
        self._store: dict[str, Any] = {}
    def set(self, key: str, value: Any):
        self._store[key] = value
    def get(self, key: str) -> Any:
        return self._store[key]


class ExecutionContext:
    """Holds results of previously executed steps, addressable by step id."""
    def __init__(self, result_store: ResultStore | None = None):
        self.results = result_store if result_store is not None else InMemoryResultStore()



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
        try:
            current = self.ctx.results.get(step_id)
        except KeyError:
            raise KeyError(f"Unknown step id in placeholder: {step_id}")
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


class Step(msgspec.Struct, tag_field="kind"):
    """Base class for workflow steps with a unique identifier."""

    id: str

    @abstractmethod
    def validate(self) -> None:
        """Validate the step. Raises ValueError if invalid."""
        ...


class OperationStep(Step, kw_only=True, tag="operation"):
    """Represents an operation step with an operation name and parameters.

    The parameters dictionary may contain any JSON value; accepts mixed types.
    """

    operation: str
    parameters: dict[str, Any]

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Invalid step id: {self.id}")
        if not isinstance(self.parameters, dict):
            raise ValueError(f"Operation step {self.id} parameters must be a dict")


class ParallelStep(Step, kw_only=True, tag="parallel"):
    """Represents a parallel step that runs multiple operation steps concurrently."""

    operations: list[OperationStep]

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Invalid step id: {self.id}")
        if not self.operations or not isinstance(self.operations, list):
            raise ValueError(f"Parallel step {self.id} has no operations")
        for op in self.operations:
            if not isinstance(op, OperationStep):
                raise ValueError(
                    f"Parallel step {self.id} contains non-operation step"
                )


class MapStep(Step, kw_only=True, tag="map"):
    """Represents a map step that iterates over inputs applying an operation step."""

    inputs: list[Any]
    iterator: str
    mode: Literal["sequential", "parallel"] = "sequential"
    operation: OperationStep

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Invalid step id: {self.id}")
        if not self.inputs or not isinstance(self.inputs, list):
            raise ValueError(f"Map step {self.id} has empty inputs")
        if not self.iterator or not isinstance(self.iterator, str) or not self.iterator.isidentifier():
            raise ValueError(
                f"Invalid iterator name in map step {self.id}: {self.iterator}"
            )
        if not isinstance(self.operation, OperationStep):
            raise ValueError(
                f"Map step {self.id} operation must be an OperationStep"
            )


StepTypes = Union[OperationStep, MapStep, ParallelStep]




class WorkflowDSL(msgspec.Struct):
    """Represents a workflow composed of multiple steps."""

    steps: list[StepTypes]


class OperationResult(msgspec.Struct):
    """Result of a single operation execution."""

    id: str
    kind: Literal["operation"]
    operation: str
    parameters: dict[str, Any]
    result: Any


class MapItemResult(msgspec.Struct):
    """Result of applying an operation to a single input item in a map step."""

    id: str
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
    for step in data.steps:
        if step.id in seen_ids:
            raise ValueError(f"Duplicate step id found: {step.id}")
        seen_ids.add(step.id)
        step.validate()
    return True


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


class WorkflowEngine(ABC):
    """Abstract base class defining the workflow engine interface."""

    @abstractmethod
    def run(self, workflow: WorkflowDSL) -> Any:
        """Runs the given workflow."""
        ...



# --- TaskRunner abstraction ---
class TaskRunner(ABC):
    @abstractmethod
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        ...

class LocalTaskRunner(TaskRunner):
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        return plugin.execute(**bound)


class StepExecutor(ABC):
    """Abstract executor interface for executing workflow steps and returning results."""
    @abstractmethod
    def execute(self, step: Any) -> Any:
        """Execute a workflow step and return its result."""
        ...


class OperationExecutor(StepExecutor):
    """Executes an operation step by resolving and running the corresponding plugin."""
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
    def execute(self, step: OperationStep):
        """Executes the operation step and returns a structured result."""
        # Resolve placeholders in parameters
        resolved_params = self.values.resolve_any(step.parameters)
        step = structs.replace(step, parameters=resolved_params)
        print(f"Executing operation {step.operation} with parameters {step.parameters}")
        plugin = self.resolver.resolve(step.operation)
        bound = self.binder.bind(plugin, step.parameters)
        result = self.task_runner.run(plugin, bound)
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
        task_runner: TaskRunner | None = None,
    ):
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
    def execute(self, step: ParallelStep):
        """Executes all operation steps in parallel and returns a structured result."""
        print(
            f"Executing parallel operations: {[op.operation for op in step.operations]}"
        )
        from concurrent.futures import ThreadPoolExecutor
        def run_op(op: OperationStep):
            res = OperationExecutor(self.resolver, self.binder, self.values, self.task_runner).execute(op)
            return ParallelOpResult(
                id=op.id,
                operation=op.operation,
                parameters=op.parameters,
                result=res.result,
            )
        with ThreadPoolExecutor() as executor:
            inner = list(executor.map(run_op, step.operations))
        return ParallelResult(id=step.id, kind="parallel", results=inner)





# --- MapStrategy pattern ---
class MapStrategy(ABC):
    @abstractmethod
    def execute(self, step: MapStep) -> MapResult:
        ...


class SequentialMapStrategy(MapStrategy):
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
    def execute(self, step: MapStep) -> MapResult:
        print(
            f"Executing sequential map over {step.inputs} with iterator {step.iterator}"
        )
        base_op = step.operation
        base_params = self.values.resolve_any(base_op.parameters)
        results = []
        for idx, item in enumerate(step.inputs):
            new_params = {}
            for k, v in base_params.items():
                if v == "{{" + step.iterator + "}}":
                    new_params[k] = item
                else:
                    new_params[k] = v
            op = structs.replace(base_op, parameters=new_params)
            res = OperationExecutor(self.resolver, self.binder, self.values, self.task_runner).execute(op)
            results.append(
                MapItemResult(
                    id=f"{op.id}_{idx}",
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


class ParallelMapStrategy(MapStrategy):
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
    def execute(self, step: MapStep) -> MapResult:
        print(
            f"Executing parallel map over {step.inputs} with iterator {step.iterator}"
        )
        base_op = step.operation
        base_params = self.values.resolve_any(base_op.parameters)
        from concurrent.futures import ThreadPoolExecutor
        def run_op(args):
            idx, item = args
            new_params = {}
            for k, v in base_params.items():
                if v == "{{" + step.iterator + "}}":
                    new_params[k] = item
                else:
                    new_params[k] = v
            op = structs.replace(base_op, parameters=new_params)
            res = OperationExecutor(self.resolver, self.binder, self.values, self.task_runner).execute(op)
            return MapItemResult(
                id=f"{op.id}_{idx}",
                input=item,
                operation=op.operation,
                parameters=op.parameters,
                result=res.result,
            )
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(run_op, enumerate(step.inputs)))
        return MapResult(
            id=step.id,
            kind="map",
            mode=step.mode,
            iterator=step.iterator,
            inputs=step.inputs,
            results=results,
        )


class MapStrategyFactory:
    """Factory to return the correct MapStrategy based on mode."""
    @staticmethod
    def get_strategy(
        mode: str,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
    ) -> MapStrategy:
        if mode == "parallel":
            return ParallelMapStrategy(resolver, binder, values, task_runner)
        else:
            return SequentialMapStrategy(resolver, binder, values, task_runner)


class MapExecutor(StepExecutor):
    """Executes a map step using a strategy pattern."""
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
        # strategy is selected per execution
    def execute(self, step: MapStep):
        print(f"Executing map over {step.inputs} with iterator {step.iterator}")
        strategy = MapStrategyFactory.get_strategy(
            step.mode, self.resolver, self.binder, self.values, self.task_runner
        )
        return strategy.execute(step)


# --- ExecutorFactory abstraction ---
class ExecutorFactory(ABC):
    @abstractmethod
    def get_executor(self, step: Step) -> StepExecutor:
        ...

class InMemoryExecutorFactory(ExecutorFactory):
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner | None = None,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner if task_runner is not None else LocalTaskRunner()
    def get_executor(self, step: Step) -> StepExecutor:
        if isinstance(step, OperationStep):
            return OperationExecutor(self.resolver, self.binder, self.values, self.task_runner)
        elif isinstance(step, MapStep):
            return MapExecutor(self.resolver, self.binder, self.values, self.task_runner)
        elif isinstance(step, ParallelStep):
            return ParallelOperationExecutor(self.resolver, self.binder, self.values, self.task_runner)
        else:
            raise ValueError(f"Unknown step type: {type(step)}")




# --- ResultRegistrar abstraction ---
class ResultRegistrar:
    """Handles registration of results into a ResultStore."""
    def __init__(self, result_store: ResultStore):
        self.result_store = result_store
    def register(self, step_result: Any):
        """Register results into the store, including nested results."""
        # Store the main result under its id
        if hasattr(step_result, "id"):
            self.result_store.set(step_result.id, step_result)
        # Register nested operation results so they are accessible by id
        if isinstance(step_result, MapResult):
            nested_id = step_result.id
            # Store the entire list of results under the nested operation id
            self.result_store.set(nested_id, step_result.results)
            # Additionally, register each MapItemResult by its own id
            for item_result in step_result.results:
                self.result_store.set(item_result.id, item_result)
        elif isinstance(step_result, ParallelResult):
            for opres in step_result.results:
                self.result_store.set(opres.id, opres)
        elif isinstance(step_result, OperationResult):
            self.result_store.set(step_result.id, step_result)


class InMemoryWorkflowEngine(WorkflowEngine):
    """Workflow engine that executes steps in memory using executors."""
    def __init__(
        self,
        executor_factory: ExecutorFactory,
        result_store: ResultStore | None = None,
        registrar: 'ResultRegistrar' = None,
    ):
        """Initializes with an executor factory and optional result store and registrar."""
        self.result_store = result_store if result_store is not None else InMemoryResultStore()
        self.registrar = registrar if registrar is not None else ResultRegistrar(self.result_store)
        self.ctx = ExecutionContext(self.result_store)
        self.values = PlaceholderResolver(self.ctx)
        self.executor_factory = executor_factory
    def run(self, workflow: WorkflowDSL) -> list[Any]:
        """Executes each step of the workflow in order and returns a list of step results."""
        results = []
        for step in workflow.steps:
            executor = self.executor_factory.get_executor(step)
            step_result = executor.execute(step)
            self.registrar.register(step_result)
            results.append(step_result)
        return results


def execute_workflow(workflow: WorkflowDSL, engine: WorkflowEngine):
    """Runs the given workflow using the specified workflow engine and returns results."""
    return engine.run(workflow)




# --- WorkflowClient abstraction ---
class WorkflowClient:
    """
    Encapsulates setup of plugin resolver, binder, result store, context, values, executor factory, registrar, and engine.
    Provides a high-level interface for loading and running workflows.
    """
    def __init__(
        self,
        plugins: list[type[PluginBase]] | None = None,
        result_store: ResultStore | None = None,
    ):
        self.resolver = InMemoryPluginResolver(plugins)
        self.binder = ParameterBinder()
        self.result_store = result_store if result_store is not None else InMemoryResultStore()
        self.ctx = ExecutionContext(self.result_store)
        self.values = PlaceholderResolver(self.ctx)
        self.executor_factory = InMemoryExecutorFactory(self.resolver, self.binder, self.values)
        self.registrar = ResultRegistrar(self.result_store)
        self.engine = InMemoryWorkflowEngine(self.executor_factory, self.result_store, self.registrar)

    def run(self, workflow_dict: dict):
        """
        Loads and executes a workflow from a dictionary.
        Returns the list of step results.
        """
        workflow = load_workflow(workflow_dict)
        results = self.engine.run(workflow)
        return results


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
        ]
    }
    client = WorkflowClient()
    results = client.run(my_workflow)
    print("Workflow results:")
    print(json.dumps(msgspec.to_builtins(results), indent=2))
