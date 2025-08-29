from enum import Enum


class BackendType(Enum):
    """Supported workflow execution backends."""

    IN_MEMORY = "in_memory"
    TEMPORAL = "temporal"
    CELERY = "celery"
