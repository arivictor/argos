from enum import Enum


class BackendType(Enum):
    """Supported workflow execution backends."""

    IN_MEMORY = "in_memory"
    SQLITE = "sqlite"
    TEMPORAL = "temporal"
    CELERY = "celery"
