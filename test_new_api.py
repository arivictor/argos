#!/usr/bin/env python3
"""
Test script for the new unified Client façade
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argos
from plugins import NumberAdderPlugin, SayHelloPlugin


def test_new_api():
    """Test the new unified Client API"""
    print("Testing new unified Client API...")

    # Check if plugin is auto-registered
    print(f"Available plugins: {[cls.__name__ for cls in argos.Client.get_available_plugins()]}")

    # Create client for in-memory execution
    client = argos.create(argos.BackendType.IN_MEMORY)

    # Register plugins
    client.plugin(SayHelloPlugin).plugin(NumberAdderPlugin)

    print(f"Plugin registry after registration: {getattr(client._resolver, '_registry', {}).keys()}")

    # Simple workflow
    workflow = {
        "steps": [
            {"id": "s1", "kind": "operation", "operation": "say_hello", "parameters": {"name": "Ari"}},
            {"id": "s2", "kind": "operation", "operation": "add", "parameters": {"a": 5, "b": 10}},
        ]
    }

    # Run workflow
    result = client.run(workflow)
    print(f"Status: {result.status}")
    print(f"Results length: {len(result.results)}")

    if result.results:
        print(f"Result 1: {result.results[0].result}")
        print(f"Result 2: {result.results[1].result}")
        return result.status == "success"
    else:
        print("No results found")
        return False


def test_backward_compatibility():
    """Test that we can still use the old patterns"""
    print("\nTesting backward compatibility...")
    from argos.infrastructure.adapter.in_memory.client import create as old_create
    from argos.infrastructure.provider import load_plugins

    plugins = load_plugins()
    old_client = old_create(plugins)

    workflow = {
        "steps": [{"id": "test1", "kind": "operation", "operation": "say_hello", "parameters": {"name": "Old API"}}]
    }

    result = old_client.run(workflow)
    print(f"Old API Status: {result.status}")
    print(f"Old API Result: {result.results[0].result if result.results else 'No results'}")
    return result.status == "success"


if __name__ == "__main__":
    success1 = test_new_api()
    success2 = test_backward_compatibility()
    overall_success = success1 and success2
    print(f"\nOverall Test {'PASSED' if overall_success else 'FAILED'}")
    sys.exit(0 if overall_success else 1)
