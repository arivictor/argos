import inspect
import re
import time
import types
import typing
from abc import ABC, abstractmethod
from typing import Any, Union, get_args, get_origin

import msgspec
from msgspec import structs

from argos.application.port import (
    Binder,
    Context,
    PlaceholderResolver,
    PluginResolver,
    ResultStore,
    StepExecutor,
    TaskRunner,
)
from argos.domain.entity import (
    MapResult,
    MapStep,
    OperationResult,
    OperationStep,
    ParallelResult,
    ParallelStep,
)
from argos.domain.port import PluginBase
from argos.domain.value_object import ExecutionOptions, MapItemResult, ParallelOpResult, ResultStatus


class ExecutionContext(Context):
    """Holds results of previously executed steps and generic workflow variables."""

    def __init__(self, result_store: ResultStore):
        self.results = result_store
        self.variables: dict[str, Any] = {}

    def get_result(self, step_id: str) -> Any:
        """Retrieves the result of a previously executed step by its id."""
        return self.results.get(step_id)

    def set_var(self, name: str, value: Any) -> None:
        """Sets a generic workflow variable."""
        self.variables[name] = value

    def get_var(self, name: str) -> Any:
        """Gets a generic workflow variable."""
        return self.variables[name]


class ParameterBinder(Binder):
    """Binds parameters (accepting mixed types) to plugin execute method arguments with type coercion.

    The bind method accepts parameter dictionaries with mixed-type values,
    and will coerce strings to the target types when necessary.
    """

    def bind(self, plugin: PluginBase, params: dict[str, Any]) -> dict[str, Any]:
        """Binds and coerces parameters (accepting mixed-type values) to the plugin's execute method signature.

        Accepts a parameter dictionary with mixed-type values; will coerce strings to the target types when necessary.
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
        """Coerces a string value to the target type, handling Optional and Union types, including PEP 604 unions."""
        # If already the right type, return as-is
        if target_type is Any or (isinstance(target_type, type) and isinstance(value, target_type)):
            return value
        origin = get_origin(target_type)
        # Handle typing.Union and PEP 604 UnionType (e.g. int | None)
        if isinstance(target_type, types.UnionType) or origin is Union:
            args = get_args(target_type) if origin is Union else target_type.__args__
            # If NoneType is in the union, handle None-like values
            none_types = [t for t in args if t is type(None)]
            if none_types and value is None or (isinstance(value, str) and value.strip().lower() in {"none", ""}):
                return None
            # Try to coerce to each type in the union (except NoneType)
            for t in args:
                if t is type(None):
                    continue
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


class VariableResolver(PlaceholderResolver):
    """Resolves ${stepId[.field][[index]].field} placeholders in arbitrarily nested data structures.
    Rules:
    - If a string is exactly a single placeholder like "${step1}", return the referenced value as-is (preserve type).
    - Otherwise, perform string interpolation by converting referenced values to str.
    - Supported paths: `${id}`, `${id.result}`, `${id.results}`, `${id.results[0]}`, `${id.results[0].result}`.
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
        # First, check for generic workflow variable safely
        if hasattr(self.ctx, "variables") and isinstance(self.ctx.variables, dict) and token in self.ctx.variables:
            return self.ctx.get_var(token)
        # token grammar: id(.field|[index])*
        parts = re.findall(r"[^.\[\]]+|\[\d+\]", token)
        if not parts:
            return token
        step_id = parts[0]
        try:
            current = self.ctx.get_result(step_id)
        except KeyError:
            # If not found in results and not a variable, raise generic unknown placeholder error
            raise KeyError(f"Unknown placeholder: {token}") from None
        # Convert msgspec Structs to builtins for traversal
        current = msgspec.to_builtins(current)
        # Walk remaining parts
        try:
            for p in parts[1:]:
                if p.startswith("["):
                    idx = int(p[1:-1])
                    current = current[idx]
                else:
                    current = current[p]
        except (KeyError, IndexError, TypeError):
            # If any part of the traversal fails, treat as unknown placeholder
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
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = execution_options

    def execute(self, step: OperationStep):
        """Executes the operation step and returns a structured result."""
        # Resolve placeholders in parameters
        resolved_params = self.values.resolve_any(step.parameters)
        step = structs.replace(step, parameters=resolved_params)
        plugin = self.resolver.resolve(step.operation)
        bound = self.binder.bind(plugin, step.parameters)
        result = None

        # Step-level retries/timeout override global options
        retries = (
            step.retries if hasattr(step, "retries") and step.retries is not None else self.execution_options.retries
        )
        timeout = (
            step.timeout if hasattr(step, "timeout") and step.timeout is not None else self.execution_options.timeout
        )
        for attempt in range(retries + 1):
            try:
                if timeout is not None:
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self.task_runner.run, plugin, bound)
                        result = future.result(timeout=timeout)
                else:
                    result = self.task_runner.run(plugin, bound)
                # On success
                return OperationResult(
                    id=step.id,
                    kind="operation",
                    operation=step.operation,
                    parameters=step.parameters,
                    result=result,
                    status="success",
                    error=None,
                )
            except Exception as e:
                if attempt < retries:
                    # Exponential backoff: 0.1, 0.2, 0.4, ...
                    time.sleep(0.1 * (2**attempt))
                else:
                    if getattr(step, "fail_workflow", True):
                        raise
                    # else: capture the error and return as result
                    return OperationResult(
                        id=step.id,
                        kind="operation",
                        operation=step.operation,
                        parameters=step.parameters,
                        result=None,
                        status="failed",
                        error=str(e),
                    )


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
        """Initializes with a plugin resolver, parameter binder, and placeholder resolver."""
        self.resolver = resolver
        self.binder = binder
        self.values = values
        self.task_runner = task_runner
        self.execution_options = execution_options if execution_options is not None else ExecutionOptions()

    def execute(self, step: ParallelStep):
        """Executes all operation steps in parallel and returns a structured result."""
        import concurrent.futures

        # Use step-level retries/timeout for all inner operations, unless operation itself
        # overrides (not handled here for simplicity).
        retries = (
            step.retries if hasattr(step, "retries") and step.retries is not None else self.execution_options.retries
        )
        timeout = (
            step.timeout if hasattr(step, "timeout") and step.timeout is not None else self.execution_options.timeout
        )

        def run_op(op: OperationStep):
            op_retries = op.retries if hasattr(op, "retries") and op.retries is not None else retries
            op_timeout = op.timeout if hasattr(op, "timeout") and op.timeout is not None else timeout
            op_with_opts = structs.replace(op, retries=op_retries, timeout=op_timeout)
            try:
                res = OperationExecutor(
                    self.resolver, self.binder, self.values, self.task_runner, self.execution_options
                ).execute(op_with_opts)

                if res is None:
                    raise ValueError(f"OperationExecutor returned None for operation {op.id}")

                return ParallelOpResult(
                    id=op.id,
                    operation=op.operation,
                    parameters=op.parameters,
                    result=res.result,
                    status=ResultStatus.SUCCESS,
                    error=None,
                )
            except Exception as e:
                # Decide fail_workflow logic: op.fail_workflow or step.fail_workflow
                fail_workflow = getattr(op, "fail_workflow", getattr(step, "fail_workflow", True))
                if fail_workflow:
                    raise
                # else: capture the error in ParallelOpResult
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
        # Keep order as in step.operations
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
            step.retries if hasattr(step, "retries") and step.retries is not None else self.execution_options.retries
        )
        timeout = (
            step.timeout if hasattr(step, "timeout") and step.timeout is not None else self.execution_options.timeout
        )
        results = []
        for idx, item in enumerate(step.inputs):
            # Set iterator variable in context before resolving parameters
            self.values.ctx.set_var(step.iterator, item)
            # Resolve parameters fresh for each item (allowing for ${iterator} placeholders)
            new_params = self.values.resolve_any(base_op.parameters)
            op_retries = base_op.retries if hasattr(base_op, "retries") and base_op.retries is not None else retries
            op_timeout = base_op.timeout if hasattr(base_op, "timeout") and base_op.timeout is not None else timeout
            op = structs.replace(base_op, parameters=new_params, retries=op_retries, timeout=op_timeout)
            try:
                res = OperationExecutor(
                    self.resolver, self.binder, self.values, self.task_runner, self.execution_options
                ).execute(op)

                if res is None:
                    raise ValueError(f"OperationExecutor returned None for operation {op.id}")

                result_val = res.result
                status = ResultStatus.SUCCESS
                error = None
            except Exception as e:
                fail_workflow = getattr(op, "fail_workflow", getattr(step, "fail_workflow", True))
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
            step.retries if hasattr(step, "retries") and step.retries is not None else self.execution_options.retries
        )
        timeout = (
            step.timeout if hasattr(step, "timeout") and step.timeout is not None else self.execution_options.timeout
        )
        import concurrent.futures

        def run_op(args):
            idx, item = args
            # Set iterator variable in context before resolving parameters
            self.values.ctx.set_var(step.iterator, item)
            # Resolve parameters fresh for each item (allowing for ${iterator} placeholders)
            new_params = self.values.resolve_any(base_op.parameters)
            op_retries = base_op.retries if hasattr(base_op, "retries") and base_op.retries is not None else retries
            op_timeout = base_op.timeout if hasattr(base_op, "timeout") and base_op.timeout is not None else timeout
            op = structs.replace(base_op, parameters=new_params, retries=op_retries, timeout=op_timeout)
            try:
                res = OperationExecutor(
                    self.resolver, self.binder, self.values, self.task_runner, self.execution_options
                ).execute(op)

                if res is None:
                    raise ValueError(f"OperationExecutor returned None for operation {op.id}")

                result_val = res.result
                status = ResultStatus.SUCCESS
                error = None
            except Exception as e:
                fail_workflow = getattr(op, "fail_workflow", getattr(step, "fail_workflow", True))
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
            futures = {executor.submit(run_op, (idx, item)): idx for idx, item in enumerate(step.inputs)}
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
            return ParallelMapStrategy(resolver, binder, values, task_runner, execution_options)
        else:
            return SequentialMapStrategy(resolver, binder, values, task_runner, execution_options)


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
        # strategy is selected per execution

    def execute(self, step: MapStep):
        strategy = MapStrategyFactory.get_strategy(
            step.mode, self.resolver, self.binder, self.values, self.task_runner, self.execution_options
        )
        return strategy.execute(step)
