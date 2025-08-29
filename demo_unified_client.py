#!/usr/bin/env python3
"""
Example demonstrating the new unified Client façade pattern.

This example shows the exact user experience described in the problem statement:
- Users only interact with Client
- Backends are pluggable via BackendType  
- Plugin registration via .plugin()
- create() is the composition root
- Switching backends requires changing only one line
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argos
from plugins import SayHelloPlugin

def main():
    """Demonstrate the unified Client API"""
    
    print("🚀 Argos Unified Client Façade Demo")
    print("=" * 40)
    
    # Create client for in-memory execution
    client = argos.create(argos.BackendType.IN_MEMORY)

    # Register plugins using fluent interface
    client.plugin(SayHelloPlugin)

    # Run workflow  
    result = client.run({
        "steps": [
            {"id": "s1", "kind": "operation", "operation": "say_hello", "parameters": {"name": "Ari"}}
        ]
    })
    
    print(f"Status: {result.status}")
    print(f"Result: {result.results[0].result}")
    
    print("\n✨ To switch to Temporal or Celery later:")
    print("   Just change: client = argos.create(argos.BackendType.TEMPORAL)")
    print("   Everything else stays the same!")
    
    print(f"\n🎯 Available backends: {list(argos.BackendType)}")
    print("   ✅ IN_MEMORY (implemented)")
    print("   🚧 TEMPORAL (planned)")  
    print("   🚧 CELERY (planned)")

if __name__ == "__main__":
    main()