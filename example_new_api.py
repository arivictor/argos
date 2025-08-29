#!/usr/bin/env python3
"""
Example using the new unified Client façade pattern.
This demonstrates the desired user experience where users only interact with the Client.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argos
from plugins import SayHelloPlugin


def main():
    """Example demonstrating the new unified API"""

    # Create client for in-memory execution
    client = argos.create(argos.BackendType.IN_MEMORY)

    # Register plugins using method chaining
    client.plugin(SayHelloPlugin)

    # Run workflow
    result = client.run(
        {"steps": [{"id": "s1", "kind": "operation", "operation": "say_hello", "parameters": {"name": "Ari"}}]}
    )

    print(result.status, result.results[0].result)

    # Later, if the user wants Temporal or Celery, they change one line:
    # client = argos.create(argos.BackendType.TEMPORAL)
    # Everything else stays the same.


if __name__ == "__main__":
    main()
