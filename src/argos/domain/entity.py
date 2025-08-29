from abc import abstractmethod
from typing import Any, Literal

import msgspec

from argos.domain.value_object import MapItemResult, ParallelOpResult


class Step(msgspec.Struct, tag_field="kind", forbid_unknown_fields=True):
    """Base class for workflow steps with a unique identifier."""

    id: str

    @abstractmethod
    def validate(self) -> None:
        """Validate the step. Raises ValueError if invalid."""
        ...


class OperationStep(Step, kw_only=True, tag="operation", forbid_unknown_fields=True):
    """Represents an operation step with an operation name and parameters.

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
    """Represents a parallel step that runs multiple operation steps concurrently.
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
    """Represents a map step that iterates over inputs applying an operation step.
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
        if not self.iterator or not isinstance(self.iterator, str) or not self.iterator.isidentifier():
            raise ValueError(f"Invalid iterator name in map step {self.id}: {self.iterator}")
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
    status: Literal["success", "failed", "skipped"] = "success"
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
    status: Literal["success", "failed", "partial"]
    results: list[Any]
    error: str | None = None

    def to_dict(self):
        """Convert the WorkflowResult to a dictionary."""
        return msgspec.to_builtins(self)

    def to_json(self) -> str:
        """Convert the WorkflowResult to a JSON string."""
        return msgspec.json.encode(self).decode()

    def to_yaml(self) -> str:
        """Convert the WorkflowResult to a YAML string."""
        return msgspec.yaml.encode(self).decode()
