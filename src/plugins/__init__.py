import time

from argos.domain.port import PluginBase

"""
Example plugins for testing and demonstration purposes.
"""

__all__ = [
    "NumberAdderPlugin",
    "SayHelloPlugin",
    "ThrowExceptionPlugin",
    "SleepyPlugin",
]


class NumberAdderPlugin(PluginBase):
    """Plugin that adds two integers and returns the sum."""

    plugin_name = "add"

    def execute(self, a: int, b: int) -> int:
        """Adds two integers a and b."""
        return a + b


class SayHelloPlugin(PluginBase):
    """Plugin that returns a greeting string for a given name."""

    plugin_name = "say_hello"

    def execute(self, name: str) -> str:
        """Returns a greeting message for the given name."""
        return f"Hello, {name}!"


class ThrowExceptionPlugin(PluginBase):
    """Plugin that throws an exception to test retry logic."""

    plugin_name = "throw_exception"

    def execute(self, message: str) -> str:
        """Raises an exception with the given message."""
        raise RuntimeError(message)


class SleepyPlugin(PluginBase):
    """Plugin that sleeps for a given number of seconds."""

    plugin_name = "sleep"

    def execute(self, seconds: int) -> str:
        """Sleeps for the specified number of seconds."""
        time.sleep(seconds)
        return f"Slept for {seconds} seconds"
