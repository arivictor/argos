from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import re
import types
from typing import Any, Literal, Union, get_args, get_origin
import typing
import msgspec
from msgspec import structs
import time
import inspect
import uuid

"""
Utils / Common / Shared
"""


class UUIDGenerator:
    """Generates unique identifiers using UUID."""

    def generate(self) -> str:
        """
        Generate a unique identifier.

        :returns: A unique identifier string
        :rtype: str
        """
        return uuid.uuid4().hex


"""
Domain - Value Objects
"""


@dataclass
class ExecutionOptions:
    retries: int = 0
    timeout: float | None = None


class WorkflowResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_FAILURE = "partial_failure"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class MapItemResult(msgspec.Struct, forbid_unknown_fields=True):
    """Result of applying an operation to a single input item in a map step."""

    id: str
    input: Any
    operation: str
    parameters: dict[str, Any]
    result: Any
    status: ResultStatus = ResultStatus.SUCCESS
    error: str | None = None


class ParallelOpResult(msgspec.Struct, forbid_unknown_fields=True):
    """Result of one operation within a parallel step."""

    id: str
    operation: str
    parameters: dict[str, Any]
    result: Any
    status: ResultStatus = ResultStatus.SUCCESS
    error: str | None = None


"""
Domain - Entities
"""


class Step(msgspec.Struct, tag_field="kind", forbid_unknown_fields=True):
    """Base class for workflow steps with a unique identifier."""

    id: str

    @abstractmethod
    def validate(self) -> None:
        """
        Validate the step.

        :raises ValueError: If the step is invalid
        """
        ...
class StepExecution(msgspec.Struct, forbid_unknown_fields=True):
    """Represents the execution of a single step in a workflow."""
    id: str
    step_kind: str
    started_at: float
    finished_at: float | None = None
    status: ResultStatus = ResultStatus.SUCCESS
    result: Any = None
    error: str | None = None


class WorkflowExecution(msgspec.Struct, forbid_unknown_fields=True):
    """Represents the execution lifecycle of a workflow."""
    id: str
    started_at: float
    finished_at: float | None = None
    step_executions: list[StepExecution] = []
    status: WorkflowResultStatus | None = None
    error: str | None = None


class OperationStep(Step, kw_only=True, tag="operation", forbid_unknown_fields=True):
    """
    Represents an operation step with an operation name and parameters.

    The parameters dictionary may contain any JSON value; accepts mixed types.
    Optional: retries and timeout for this step.
    """

    operation: str
    parameters: dict[str, Any]
    retries: int = 0
    timeout: float | None = None
    fail_workflow: bool = True

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Invalid step id: {self.id}")
        if not isinstance(self.parameters, dict):
            raise ValueError(f"Operation step {self.id} parameters must be a dict")


class ParallelStep(Step, kw_only=True, tag="parallel", forbid_unknown_fields=True):
    """
    Represents a parallel step that runs multiple operation steps concurrently.

    Optional: retries and timeout for all inner operations.
    """

    operations: list[OperationStep]
    retries: int = 0
    timeout: float | None = None
    fail_workflow: bool = True

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Invalid step id: {self.id}")
        if not self.operations or not isinstance(self.operations, list):
            raise ValueError(f"Parallel step {self.id} has no operations")
        for op in self.operations:
            if not isinstance(op, OperationStep):
                raise ValueError(f"Parallel step {self.id} contains non-operation step")


class MapStep(Step, kw_only=True, tag="map", forbid_unknown_fields=True):
    """
    Represents a map step that iterates over inputs applying an operation step.

    Optional: retries and timeout for each mapped operation.
    """

    inputs: list[Any]
    iterator: str
    mode: Literal["sequential", "parallel"] = "sequential"
    operation: OperationStep
    retries: int = 0
    timeout: float | None = None
    fail_workflow: bool = True

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Invalid step id: {self.id}")
        if not self.inputs or not isinstance(self.inputs, list):
            raise ValueError(f"Map step {self.id} has empty inputs")
        if (
            not self.iterator
            or not isinstance(self.iterator, str)
            or not self.iterator.isidentifier()
        ):
            raise ValueError(
                f"Invalid iterator name in map step {self.id}: {self.iterator}"
            )
        if not isinstance(self.operation, OperationStep):
            raise ValueError(f"Map step {self.id} operation must be an OperationStep")


StepTypes = OperationStep | MapStep | ParallelStep


class WorkflowDSL(msgspec.Struct, forbid_unknown_fields=True):
    """Represents a workflow composed of multiple steps."""

    steps: list[StepTypes]


class OperationResult(msgspec.Struct, forbid_unknown_fields=True):
    """Result of a single operation execution."""

    id: str
    kind: Literal["operation"]
    operation: str
    parameters: dict[str, Any]
    result: Any
    status: ResultStatus = ResultStatus.SUCCESS
    error: str | None = None


class MapResult(msgspec.Struct, forbid_unknown_fields=True):
    """Aggregated result of a map step."""

    id: str
    kind: Literal["map"]
    mode: Literal["sequential", "parallel"]
    iterator: str
    inputs: list[Any]
    results: list[MapItemResult]


class ParallelResult(msgspec.Struct, forbid_unknown_fields=True):
    """Aggregated result of a parallel step."""

    id: str
    kind: Literal["parallel"]
    results: list[ParallelOpResult]


class WorkflowResult(msgspec.Struct, forbid_unknown_fields=True):
    """Result of running a workflow, including status and all step results."""

    id: str
    status: WorkflowResultStatus
    results: list[Any]
    error: str | None = None


"""
Domain - Ports
"""


class PluginBase:
    """Base class for all plugins. Enforces 'execute' method and registers subclasses."""

    _plugins = []

    def __init_subclass__(cls, **kwargs):
        """
        Registers subclass and ensures 'execute' method is defined.

        :param kwargs: Additional keyword arguments passed to super().__init_subclass__
        :raises TypeError: If the subclass doesn't define an 'execute' method
        """
        super().__init_subclass__(**kwargs)

        if "execute" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a 'execute' method")

        PluginBase._plugins.append(cls)

    def execute(self, *args, **kwargs) -> Any:
        """
        Abstract execute method to be implemented by plugins.

        :param args: Positional arguments for the plugin execution
        :param kwargs: Keyword arguments for the plugin execution
        :returns: Result of the plugin execution
        :rtype: Any
        :raises NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Plugins must implement the execute method")


"""
Domain - Services
"""


def validate_workflow(data: WorkflowDSL) -> bool:
    """
    Validates the workflow structure and contents.

    :param data: The WorkflowDSL instance to validate
    :type data: WorkflowDSL
    :returns: True if the workflow is valid, raises ValueError otherwise
    :rtype: bool
    :raises ValueError: If the workflow structure or contents are invalid
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


# ---------------------------------------------------------------------------

"""
Application - Ports
"""


class WorkflowEngine(ABC):
    """Abstract base class defining the workflow engine interface."""

    @abstractmethod
    def run(self, workflow: WorkflowDSL, workflow_id: str | None = None) -> Any:
        """
        Runs the given workflow.

        :param workflow: The workflow to execute
        :type workflow: WorkflowDSL
        :param workflow_id: Optional workflow identifier
        :type workflow_id: str | None
        :returns: The result of executing the workflow
        :rtype: Any
        """
        ...


class TaskRunner(ABC):
    """Abstract interface for running plugin tasks."""

    @abstractmethod
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        """
        Run a plugin with bound parameters.

        :param plugin: The plugin to run
        :type plugin: PluginBase
        :param bound: Dictionary of bound parameters
        :type bound: dict[str, Any]
        :returns: The result of running the plugin
        :rtype: Any
        """


class StepExecutor(ABC):
    """Abstract executor interface for executing workflow steps and returning results."""

    @abstractmethod
    def execute(self, step: Any) -> Any:
        """
        Execute a workflow step and return its result.

        :param step: The workflow step to execute
        :type step: Any
        :returns: The result of executing the step
        :rtype: Any
        """
        ...


class ExecutorFactory(ABC):
    """Abstract factory for creating step executors."""

    @abstractmethod
    def get_executor(self, step: Step) -> StepExecutor:
        """
        Get an executor for the given step.

        :param step: The step to get an executor for
        :type step: Step
        :returns: An executor capable of executing the step
        :rtype: StepExecutor
        """


class PluginResolver(ABC):
    """Abstract base class defining plugin resolution interface."""

    @abstractmethod
    def resolve(self, name: str) -> PluginBase:
        """
        Resolves and returns a plugin instance by its name.

        :param name: The name of the plugin to resolve
        :type name: str
        :returns: The resolved plugin instance
        :rtype: PluginBase
        :raises: KeyError if the plugin is not found
        """
        ...


class ResultStore(ABC):
    """Abstract interface for storing and retrieving workflow results."""

    @abstractmethod
    def set(self, key: str, value: Any):
        """
        Store a value with the given key.

        :param key: The key to store the value under
        :type key: str
        :param value: The value to store
        :type value: Any
        """

    @abstractmethod
    def get(self, key: str) -> Any:
        """
        Retrieve a value by key.

        :param key: The key to retrieve the value for
        :type key: str
        :returns: The stored value
        :rtype: Any
        :raises: KeyError if the key is not found
        """


class PlaceholderResolver(ABC):
    """Abstract interface for resolving placeholders in values."""

    @abstractmethod
    def resolve_any(self, value: Any) -> Any:
        """
        Resolve placeholders in any value type.

        :param value: The value that may contain placeholders
        :type value: Any
        :returns: The value with placeholders resolved
        :rtype: Any
        """


class Binder(ABC):
    """Abstract interface for binding parameters to plugin methods."""

    @abstractmethod
    def bind(self, plugin: PluginBase, params: dict[str, Any]) -> dict[str, Any]:
        """
        Bind parameters to a plugin's execute method.

        :param plugin: The plugin to bind parameters for
        :type plugin: PluginBase
        :param params: The parameters to bind
        :type params: dict[str, Any]
        :returns: Dictionary of bound parameters
        :rtype: dict[str, Any]
        """


class Context(ABC):
    """Abstract interface for workflow execution context."""

    @abstractmethod
    def get_result(self, step_id: str) -> Any:
        """
        Get the result of a previously executed step.

        :param step_id: The ID of the step whose result to retrieve
        :type step_id: str
        :returns: The result of the specified step
        :rtype: Any
        :raises: KeyError if the step result is not found
        """


"""
Application - Services
"""


class ExecutionContext(Context):
    """Holds results of previously executed steps and generic workflow variables."""

    def __init__(self, result_store: ResultStore):
        self.results = result_store
        self.variables: dict[str, Any] = {}

    def get_result(self, step_id: str) -> Any:
        """
        Retrieves the result of a previously executed step by its id.

        :param step_id: The identifier of the step whose result to retrieve
        :type step_id: str
        :returns: The result of the specified step
        :rtype: Any
        """
        return self.results.get(step_id)

    def set_var(self, name: str, value: Any) -> None:
        """
        Sets a generic workflow variable.

        :param name: The name of the variable
        :type name: str
        :param value: The value to assign to the variable
        :type value: Any
        """
        self.variables[name] = value

    def get_var(self, name: str) -> Any:
        """
        Gets a generic workflow variable.

        :param name: The name of the variable to retrieve
        :type name: str
        :returns: The value of the specified variable
        :rtype: Any
        """
        return self.variables[name]



class ParameterBinder(Binder):
    """
    Binds parameters (accepting mixed types) to plugin execute method arguments with type coercion.

    The bind method accepts parameter dictionaries with mixed-type values,
    and will coerce strings to the target types when necessary.
    """

    def bind(self, plugin: PluginBase, params: dict[str, Any]) -> dict[str, Any]:
        """
        Binds and coerces parameters (accepting mixed-type values) to the plugin's execute method signature.

        Accepts a parameter dictionary with mixed-type values; will coerce strings to the target types when necessary.

        :param plugin: The plugin instance to bind parameters for
        :type plugin: PluginBase
        :param params: The parameter dictionary with mixed-type values
        :type params: dict[str, Any]
        :returns: Dictionary of bound and coerced parameters
        :rtype: dict[str, Any]
        """
        sig = inspect.signature(plugin.execute)
        hints = typing.get_type_hints(plugin.execute, include_extras=False)
        bound: dict[str, Any] = {}
        for name, _ in sig.parameters.items():
            if name == "self":
                continue
            if name not in params:
                continue
            target = hints.get(name, Any)
            bound[name] = self._coerce(params[name], target)
        return bound

    def _coerce(self, value: Any, target_type: Any) -> Any:
        """
        Coerces a string value to the target type, handling Optional and Union types, including PEP 604 unions.

        :param value: The value to coerce
        :type value: Any
        :param target_type: The target type to coerce to
        :type target_type: Any
        :returns: The coerced value
        :rtype: Any
        """
        # If already the right type, return as-is
        if target_type is Any or (
            isinstance(target_type, type) and isinstance(value, target_type)
        ):
            return value
        origin = get_origin(target_type)
        # Handle typing.Union and PEP 604 UnionType
        if isinstance(target_type, types.UnionType) or origin is Union:
            args = get_args(target_type) if origin is Union else target_type.__args__
            none_types = [t for t in args if t is type(None)]
            if (
                none_types
                and value is None
                or (isinstance(value, str) and value.strip().lower() in {"none", ""})
            ):
                return None
            for t in args:
                if t is type(None):
                    continue
                try:
                    return self._coerce(value, t)
                except Exception:
                    continue
            return value
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
            return value
        return value


class RetryPolicy:
    """Simple retry with exponential backoff."""

    def __init__(self, base_sleep: float = 0.1):
        self.base_sleep = base_sleep

    def run(self, func, retries: int):
        last_exc = None
        for attempt in range(retries + 1):
            try:
                return func()
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    time.sleep(self.base_sleep * (2**attempt))
                else:
                    raise last_exc


"""
Infrastructure - Adapters
"""

class VariableResolver(PlaceholderResolver):
    """
    Resolves ${stepId[.field][[index]].field} placeholders in arbitrarily nested data structures.

    Rules:
    - If a string is exactly a single placeholder like "${step1}", return the referenced value as-is.
    - Otherwise, interpolate within the string via str().
    """

    _pattern = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, ctx: Context):
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
        m = self._pattern.fullmatch(s.strip())
        if m:
            return self._lookup_token(m.group(1))

        def repl(match: re.Match) -> str:
            val = self._lookup_token(match.group(1))
            return str(val)

        return self._pattern.sub(repl, s)

    def _lookup_token(self, token: str) -> Any:
        if (
            hasattr(self.ctx, "variables")
            and isinstance(self.ctx.variables, dict)
            and token in self.ctx.variables
        ):
            return self.ctx.get_var(token)
        parts = re.findall(r"[^.\[\]]+|\[\d+\]", token)
        if not parts:
            return token
        step_id = parts[0]
        try:
            current = self.ctx.get_result(step_id)
        except KeyError:
            raise KeyError(f"Unknown placeholder: {token}") from None
        current = msgspec.to_builtins(current)
        try:
            for p in parts[1:]:
                if p.startswith("["):
                    idx = int(p[1:-1])
                    current = current[idx]
                else:
                    current = current[p]
        except (KeyError, IndexError, TypeError):
            raise KeyError(f"Unknown placeholder: {token}") from None
        return current


class OperationExecutor(StepExecutor):
    """Executes an operation step by resolving and running the corresponding plugin."""

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner,
        execution_options: ExecutionOptions,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = execution_options

    def execute(self, step: OperationStep):
        resolved_params = self.values.resolve_any(step.parameters)
        step = structs.replace(step, parameters=resolved_params)
        plugin = self.resolver.resolve(step.operation)
        bound = self.binder.bind(plugin, step.parameters)

        retries = (
            step.retries
            if hasattr(step, "retries") and step.retries is not None
            else self.execution_options.retries
        )
        timeout = (
            step.timeout
            if hasattr(step, "timeout") and step.timeout is not None
            else self.execution_options.timeout
        )
        policy = RetryPolicy()

        def call_once():
            if timeout is not None:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.task_runner.run, plugin, bound)
                    return future.result(timeout=timeout)
            return self.task_runner.run(plugin, bound)

        try:
            result = policy.run(call_once, retries)
            return OperationResult(
                id=step.id,
                kind="operation",
                operation=step.operation,
                parameters=step.parameters,
                result=result,
                status=ResultStatus.SUCCESS,
                error=None,
            )
        except Exception as e:
            if getattr(step, "fail_workflow", True):
                raise
            return OperationResult(
                id=step.id,
                kind="operation",
                operation=step.operation,
                parameters=step.parameters,
                result=None,
                status=ResultStatus.FAILED,
                error=str(e),
            )


class WorkflowResultSerializer:
    @staticmethod
    def to_dict(result: WorkflowResult) -> dict:
        return msgspec.to_builtins(result)

    @staticmethod
    def to_json(result: WorkflowResult) -> str:
        return msgspec.json.encode(result).decode()

    @staticmethod
    def to_yaml(result: WorkflowResult) -> str:
        return msgspec.yaml.encode(result).decode()


class ParallelOperationExecutor(StepExecutor):
    """Executes multiple operation steps in parallel using threads."""

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner,
        execution_options: ExecutionOptions,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = (
            execution_options if execution_options is not None else ExecutionOptions()
        )

    def execute(self, step: ParallelStep):
        import concurrent.futures

        retries = (
            step.retries
            if hasattr(step, "retries") and step.retries is not None
            else self.execution_options.retries
        )
        timeout = (
            step.timeout
            if hasattr(step, "timeout") and step.timeout is not None
            else self.execution_options.timeout
        )

        def run_op(op: OperationStep):
            op_retries = (
                op.retries
                if hasattr(op, "retries") and op.retries is not None
                else retries
            )
            op_timeout = (
                op.timeout
                if hasattr(op, "timeout") and op.timeout is not None
                else timeout
            )
            op_with_opts = structs.replace(op, retries=op_retries, timeout=op_timeout)
            try:
                res = OperationExecutor(
                    self.resolver,
                    self.binder,
                    self.values,
                    self.task_runner,
                    self.execution_options,
                ).execute(op_with_opts)

                if res is None:
                    raise ValueError(
                        f"OperationExecutor returned None for operation {op.id}"
                    )

                return ParallelOpResult(
                    id=op.id,
                    operation=op.operation,
                    parameters=op.parameters,
                    result=res.result,
                    status=ResultStatus.SUCCESS,
                    error=None,
                )
            except Exception as e:
                fail_workflow = getattr(
                    op, "fail_workflow", getattr(step, "fail_workflow", True)
                )
                if fail_workflow:
                    raise
                return ParallelOpResult(
                    id=op.id,
                    operation=op.operation,
                    parameters=op.parameters,
                    result=None,
                    status=ResultStatus.FAILED,
                    error=str(e),
                )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_op, op): op for op in step.operations}
            results = []
            for future in concurrent.futures.as_completed(futures):
                _ = futures[future]
                try:
                    result = future.result(timeout=timeout)
                except Exception:
                    raise
                results.append(result)
        results.sort(key=lambda r: [o.id for o in step.operations].index(r.id))
        return ParallelResult(id=step.id, kind="parallel", results=results)


class MapStrategy(ABC):
    @abstractmethod
    def execute(self, step: MapStep) -> MapResult: ...


class SequentialMapStrategy(MapStrategy):
    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner,
        execution_options: ExecutionOptions,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = execution_options

    def execute(self, step: MapStep) -> MapResult:
        base_op = step.operation
        retries = (
            step.retries
            if hasattr(step, "retries") and step.retries is not None
            else self.execution_options.retries
        )
        timeout = (
            step.timeout
            if hasattr(step, "timeout") and step.timeout is not None
            else self.execution_options.timeout
        )
        results = []
        for idx, item in enumerate(step.inputs):
            self.values.ctx.set_var(step.iterator, item)
            new_params = self.values.resolve_any(base_op.parameters)
            op_retries = (
                base_op.retries
                if hasattr(base_op, "retries") and base_op.retries is not None
                else retries
            )
            op_timeout = (
                base_op.timeout
                if hasattr(base_op, "timeout") and base_op.timeout is not None
                else timeout
            )
            op = structs.replace(
                base_op, parameters=new_params, retries=op_retries, timeout=op_timeout
            )
            try:
                res = OperationExecutor(
                    self.resolver,
                    self.binder,
                    self.values,
                    self.task_runner,
                    self.execution_options,
                ).execute(op)

                if res is None:
                    raise ValueError(
                        f"OperationExecutor returned None for operation {op.id}"
                    )

                result_val = res.result
                status = ResultStatus.SUCCESS
                error = None
            except Exception as e:
                fail_workflow = getattr(
                    op, "fail_workflow", getattr(step, "fail_workflow", True)
                )
                if fail_workflow:
                    raise
                result_val = None
                status = ResultStatus.FAILED
                error = str(e)
            results.append(
                MapItemResult(
                    id=f"{op.id}_{idx}",
                    input=item,
                    operation=op.operation,
                    parameters=op.parameters,
                    result=result_val,
                    status=status,
                    error=error,
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
        task_runner: TaskRunner,
        execution_options: ExecutionOptions,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = execution_options

    def execute(self, step: MapStep) -> MapResult:
        base_op = step.operation
        retries = (
            step.retries
            if hasattr(step, "retries") and step.retries is not None
            else self.execution_options.retries
        )
        timeout = (
            step.timeout
            if hasattr(step, "timeout") and step.timeout is not None
            else self.execution_options.timeout
        )
        import concurrent.futures

        def run_op(args):
            idx, item = args
            self.values.ctx.set_var(step.iterator, item)
            new_params = self.values.resolve_any(base_op.parameters)
            op_retries = (
                base_op.retries
                if hasattr(base_op, "retries") and base_op.retries is not None
                else retries
            )
            op_timeout = (
                base_op.timeout
                if hasattr(base_op, "timeout") and base_op.timeout is not None
                else timeout
            )
            op = structs.replace(
                base_op, parameters=new_params, retries=op_retries, timeout=op_timeout
            )
            try:
                res = OperationExecutor(
                    self.resolver,
                    self.binder,
                    self.values,
                    self.task_runner,
                    self.execution_options,
                ).execute(op)

                if res is None:
                    raise ValueError(
                        f"OperationExecutor returned None for operation {op.id}"
                    )

                result_val = res.result
                status = ResultStatus.SUCCESS
                error = None
            except Exception as e:
                fail_workflow = getattr(
                    op, "fail_workflow", getattr(step, "fail_workflow", True)
                )
                if fail_workflow:
                    raise
                result_val = None
                status = ResultStatus.FAILED
                error = str(e)
            return MapItemResult(
                id=f"{op.id}_{idx}",
                input=item,
                operation=op.operation,
                parameters=op.parameters,
                result=result_val,
                status=status,
                error=error,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(run_op, (idx, item)): idx
                for idx, item in enumerate(step.inputs)
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                _ = futures[future]
                try:
                    result = future.result(timeout=timeout)
                except Exception:
                    raise
                results.append(result)
        results.sort(key=lambda r: int(r.id.split("_")[-1]))
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
        task_runner: TaskRunner,
        execution_options: ExecutionOptions,
    ) -> MapStrategy:
        if mode == "parallel":
            return ParallelMapStrategy(
                resolver, binder, values, task_runner, execution_options
            )
        else:
            return SequentialMapStrategy(
                resolver, binder, values, task_runner, execution_options
            )


class MapExecutor(StepExecutor):
    """Executes a map step using a strategy pattern."""

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner,
        execution_options: ExecutionOptions,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = execution_options

    def execute(self, step: MapStep):
        strategy = MapStrategyFactory.get_strategy(
            step.mode,
            self.resolver,
            self.binder,
            self.values,
            self.task_runner,
            self.execution_options,
        )
        return strategy.execute(step)


"""
Application - Services
"""


class ResultRegistrar:
    """Handles registration of results into a ResultStore."""

    def __init__(self, result_store: ResultStore):
        self.result_store = result_store

    def register(self, step_result: Any):
        if hasattr(step_result, "id"):
            self.result_store.set(step_result.id, step_result)
        if isinstance(step_result, MapResult):
            nested_id = step_result.id
            self.result_store.set(nested_id, step_result.results)
            for item_result in step_result.results:
                self.result_store.set(item_result.id, item_result)
        elif isinstance(step_result, ParallelResult):
            for opres in step_result.results:
                self.result_store.set(opres.id, opres)
        elif isinstance(step_result, OperationResult):
            self.result_store.set(step_result.id, step_result)


def load_workflow(data: dict | WorkflowDSL) -> WorkflowDSL:
    if isinstance(data, WorkflowDSL):
        validate_workflow(data)
        return data

    workflow = msgspec.convert(data, type=WorkflowDSL)
    validate_workflow(workflow)
    return workflow


def execute_workflow(workflow: WorkflowDSL, engine: WorkflowEngine) -> WorkflowResult:
    return engine.run(workflow)


class WorkflowClient:
    """
    High level façade. Dependencies are injected, not re-created.
    """

    def __init__(
        self,
        plugin_resolver: PluginResolver,
        executor_factory: Any,
        workflow_engine: WorkflowEngine,
        result_store: ResultStore,
        binder: Binder,
        exectuion_context: Context,
        values: PlaceholderResolver,
        registrar: ResultRegistrar,
        execution_options: ExecutionOptions | None = None,
    ):
        self.resolver = plugin_resolver
        self.binder = binder
        self.result_store = result_store
        self.ctx = exectuion_context
        self.values = values
        self.execution_options = (
            execution_options if execution_options is not None else ExecutionOptions()
        )
        self.executor_factory = executor_factory
        self.registrar = registrar
        self.engine = workflow_engine

    def run(
        self, workflow: dict | WorkflowDSL, workflow_id: str | None = None
    ) -> WorkflowResult:
        workflow = load_workflow(workflow)
        result = self.engine.run(workflow, workflow_id)
        return result


# ---------------------------------------------------------------------------

"""
Infrastructure - Adapters
"""


class InMemoryPluginResolver(PluginResolver):
    """Resolves plugins from an in-memory registry."""

    def __init__(self, plugins: list[type[PluginBase]]):
        self._registry: dict[str, type[PluginBase]] = {}
        for cls in plugins:
            key = getattr(cls, "plugin_name", cls.__name__)
            self._registry[key] = cls

    def resolve(self, name: str) -> PluginBase:
        try:
            cls = self._registry[name]
        except KeyError:
            raise KeyError(f"No plugin registered for operation '{name}'") from None
        return cls()


class InMemoryExecutorFactory(ExecutorFactory):
    """
    Executor factory that returns executors based on step type.
    """

    def __init__(
        self,
        resolver: PluginResolver,
        binder: ParameterBinder,
        values: PlaceholderResolver,
        task_runner: TaskRunner,
        execution_options: ExecutionOptions,
    ):
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = execution_options

    def get_executor(self, step: Step) -> StepExecutor:
        if isinstance(step, OperationStep):
            return OperationExecutor(
                self.resolver,
                self.binder,
                self.values,
                self.task_runner,
                self.execution_options,
            )
        elif isinstance(step, MapStep):
            return MapExecutor(
                self.resolver,
                self.binder,
                self.values,
                self.task_runner,
                self.execution_options,
            )
        elif isinstance(step, ParallelStep):
            return ParallelOperationExecutor(
                self.resolver,
                self.binder,
                self.values,
                self.task_runner,
                self.execution_options,
            )
        else:
            raise ValueError(f"Unknown step type: {type(step)}")


class InMemoryResultStore(ResultStore):
    """
    Simple in-memory result store using a dictionary.
    Stores and retrieves workflow results by key.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self._store[key] = value

    def get(self, key: str) -> Any:
        return self._store[key]


class InMemoryTaskRunner(TaskRunner):
    def run(self, plugin: PluginBase, bound: dict[str, Any]) -> Any:
        return plugin.execute(**bound)


class InMemoryWorkflowEngine(WorkflowEngine):
    """Workflow engine that executes steps in memory using executors."""

    def __init__(
        self,
        executor_factory: ExecutorFactory,
        result_store: ResultStore,
        registrar: ResultRegistrar,
    ):
        self.result_store = (
            result_store if result_store is not None else InMemoryResultStore()
        )
        self.registrar = (
            registrar if registrar is not None else ResultRegistrar(self.result_store)
        )
        self.executor_factory = executor_factory

    def run(
        self, workflow: WorkflowDSL, workflow_id: str | None = None
    ) -> WorkflowResult:
        if workflow_id is None:
            workflow_id = UUIDGenerator().generate()

        wf_exec = WorkflowExecution(id=workflow_id, started_at=time.time())
        results = []
        error_msg = None
        any_failed = False
        any_nonfatal_failed = False

        for step in workflow.steps:
            step_exec = StepExecution(
                id=step.id,
                step_kind=type(step).__name__,
                started_at=time.time(),
            )
            try:
                executor = self.executor_factory.get_executor(step)
                step_result = executor.execute(step)
                self.registrar.register(step_result)

                step_exec.result = step_result
                step_exec.status = ResultStatus.SUCCESS

                failed = False
                nonfatal_failed = False
                if (
                    hasattr(step_result, "status")
                    and getattr(step_result, "status", None) == ResultStatus.FAILED
                ):
                    fail_workflow = getattr(step, "fail_workflow", True)
                    if fail_workflow:
                        failed = True
                    else:
                        nonfatal_failed = True

                if hasattr(step_result, "results"):
                    contained = getattr(step_result, "results", [])
                    for item in contained:
                        if (
                            hasattr(item, "status")
                            and getattr(item, "status", None) == ResultStatus.FAILED
                        ):
                            fail_workflow = getattr(step, "fail_workflow", True)
                            if fail_workflow:
                                failed = True
                            else:
                                nonfatal_failed = True

                if failed:
                    any_failed = True
                if nonfatal_failed:
                    any_nonfatal_failed = True

                results.append(step_result)
            except Exception as e:
                step_exec.error = str(e)
                step_exec.status = ResultStatus.FAILED
                any_failed = True
                error_msg = str(e)
                step_exec.finished_at = time.time()
                wf_exec.step_executions.append(step_exec)
                break
            finally:
                step_exec.finished_at = time.time()
                wf_exec.step_executions.append(step_exec)

        wf_exec.finished_at = time.time()
        if any_failed:
            wf_exec.status = WorkflowResultStatus.FAILED
        elif any_nonfatal_failed:
            wf_exec.status = WorkflowResultStatus.PARTIAL_FAILURE
        else:
            wf_exec.status = WorkflowResultStatus.SUCCESS
        wf_exec.error = error_msg

        return WorkflowResult(
            id=wf_exec.id,
            status=wf_exec.status,
            results=results,
            error=wf_exec.error,
        )


if __name__ == "__main__":

    class AddPlugin(PluginBase):
        plugin_name = "add"

        def execute(self, a: int, b: int) -> int:
            return a + b

    class ConcatPlugin(PluginBase):
        plugin_name = "concat"

        def execute(self, s1: str, s2: str) -> str:
            if type(s1) is not str:
                s1 = str(s1)
            if type(s2) is not str:
                s2 = str(s2)
            return s1 + s2

    # Setup infrastructure components, instantiate once
    plugins = [AddPlugin, ConcatPlugin]
    resolver = InMemoryPluginResolver(plugins)
    result_store = InMemoryResultStore()
    binder = ParameterBinder()
    ctx = ExecutionContext(result_store)
    values = VariableResolver(ctx)
    task_runner = InMemoryTaskRunner()
    execution_options = ExecutionOptions(retries=1, timeout=5.0)
    executor_factory = InMemoryExecutorFactory(
        resolver, binder, values, task_runner, execution_options
    )
    registrar = ResultRegistrar(result_store)
    engine = InMemoryWorkflowEngine(executor_factory, result_store, registrar)
    client = WorkflowClient(
        resolver,
        executor_factory,
        engine,
        result_store,
        binder,
        ctx,
        values,
        registrar,
        execution_options,
    )

    # Define a sample workflow
    sample_workflow = {
        "steps": [
            {
                "id": "step1",
                "kind": "operation",
                "operation": "add",
                "parameters": {"a": 5, "b": 10},
            },
            {
                "id": "step2",
                "kind": "operation",
                "operation": "concat",
                "parameters": {"s1": "Result is: ", "s2": "${step1.result}"},
            },
            {
                "id": "step3",
                "kind": "map",
                "mode": "parallel",
                "iterator": "item",
                "inputs": [1, 2, 3],
                "operation": {
                    "id": "map_op",
                    "kind": "operation",
                    "operation": "add",
                    "parameters": {"a": "${item}", "b": 100},
                },
            },
        ]
    }

    # Run the workflow
    result = client.run(sample_workflow, workflow_id="workflow1")
    print(WorkflowResultSerializer.to_yaml(result))