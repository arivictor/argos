"""
Argos - Workflow Orchestration Framework

A powerful abstraction over various execution backends, allowing you to define
your workflows once and run them anywhere without rewriting code.
"""

from argos.backend import BackendType
from argos.client import Client
from argos.factory import create

__all__ = [
    "Client",
    "BackendType",
    "create",
]
