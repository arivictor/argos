"""
AroFlow - Workflow Orchestration Framework

A powerful abstraction over various execution backends, allowing you to define
your workflows once and run them anywhere without rewriting code.
"""

from aroflow.backend import BackendType
from aroflow.client import Client
from aroflow.domain.entity import WorkflowResult
from aroflow.domain.port import PluginBase as PluginMixin
from aroflow.factory import create

__all__ = [
    "Client",
    "BackendType",
    "create",
    "PluginMixin",
    "WorkflowResult",
]
