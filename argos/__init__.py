"""Argos: Workflow Orchestration Framework

This module implements a plugin-driven workflow DSL engine following SOLID principles.
It provides plugin registration, resolution, parameter binding, and execution of workflow steps.
"""

# Re-export main public APIs for backward compatibility
from .application.workflow_client import WorkflowClient
from .application.workflow_engine import WorkflowEngine, InMemoryWorkflowEngine
from .domain.models.execution_options import ExecutionOptions
from .domain.models.steps import Step, OperationStep, MapStep, ParallelStep, StepTypes
from .domain.models.results import (
    OperationResult, MapResult, MapItemResult, ParallelResult, ParallelOpResult, WorkflowResult
)
from .domain.plugins.base import PluginBase
from .domain.plugins.builtin import NumberAdderPlugin, SayHelloPlugin, ThrowExceptionPlugin, SleepyPlugin

__all__ = [
    'WorkflowClient',
    'WorkflowEngine', 'InMemoryWorkflowEngine',
    'ExecutionOptions',
    'Step', 'OperationStep', 'MapStep', 'ParallelStep', 'StepTypes',
    'OperationResult', 'MapResult', 'MapItemResult', 'ParallelResult', 'ParallelOpResult', 'WorkflowResult',
    'PluginBase',
    'NumberAdderPlugin', 'SayHelloPlugin', 'ThrowExceptionPlugin', 'SleepyPlugin',
]