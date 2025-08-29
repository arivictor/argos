"""Domain models for execution configuration."""

from dataclasses import dataclass


@dataclass
class ExecutionOptions:
    """Configuration options for step execution."""
    retries: int = 0
    timeout: float | None = None