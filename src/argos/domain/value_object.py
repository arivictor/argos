from dataclasses import dataclass
from enum import Enum
from typing import Any

import msgspec


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
