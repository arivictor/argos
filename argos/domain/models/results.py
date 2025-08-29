"""Domain models for workflow execution results."""

from typing import Any, Literal, Optional
import msgspec


class OperationResult(msgspec.Struct, forbid_unknown_fields=True):
    """Result of a single operation execution."""

    id: str
    kind: Literal["operation"]
    operation: str
    parameters: dict[str, Any]
    result: Any
    status: Literal["success", "failed", "skipped"] = "success"
    error: Optional[str] = None


class MapItemResult(msgspec.Struct, forbid_unknown_fields=True):
    """Result of applying an operation to a single input item in a map step."""

    id: str
    input: Any
    operation: str
    parameters: dict[str, Any]
    result: Any
    status: Literal["success", "failed", "skipped"] = "success"
    error: Optional[str] = None


class MapResult(msgspec.Struct, forbid_unknown_fields=True):
    """Aggregated result of a map step."""

    id: str
    kind: Literal["map"]
    mode: Literal["sequential", "parallel"]
    iterator: str
    inputs: list[Any]
    results: list[MapItemResult]


class ParallelOpResult(msgspec.Struct, forbid_unknown_fields=True):
    """Result of one operation within a parallel step."""

    id: str
    operation: str
    parameters: dict[str, Any]
    result: Any
    status: Literal["success", "failed", "skipped"] = "success"
    error: Optional[str] = None


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
    error: Optional[str] = None