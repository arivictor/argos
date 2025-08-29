"""Placeholder resolution service for resolving ${stepId[.field][[index]].field} placeholders."""

import re
from typing import Any
import msgspec


class PlaceholderResolver:
    """Resolves ${stepId[.field][[index]].field} placeholders in arbitrarily nested data structures.
    Rules:
    - If a string is exactly a single placeholder like "${step1}", return the referenced value as-is (preserve type).
    - Otherwise, perform string interpolation by converting referenced values to str.
    - Supported paths: `${id}`, `${id.result}`, `${id.results}`, `${id.results[0]}`, `${id.results[0].result}`.
    """
    _pattern = re.compile(r"\$\{([^}]+)\}")
    
    def __init__(self, ctx):
        """Initialize with an execution context that has a results attribute."""
        self.ctx = ctx
        
    def resolve_any(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self.resolve_any(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve_any(v) for v in value]
        if isinstance(value, str):
            return self._resolve_string(value)
        return value
        
    def _resolve_string(self, s: str) -> Any:
        # Exact single-token match => return raw value to preserve type
        m = self._pattern.fullmatch(s.strip())
        if m:
            return self._lookup_token(m.group(1))
        # Interpolate within string
        def repl(match: re.Match) -> str:
            val = self._lookup_token(match.group(1))
            return str(val)
        return self._pattern.sub(repl, s)
        
    def _lookup_token(self, token: str) -> Any:
        # token grammar: id(.field|[index])*
        parts = re.findall(r"[^.\[\]]+|\[\d+\]", token)
        if not parts:
            return token
        step_id = parts[0]
        try:
            current = self.ctx.results.get(step_id)
        except KeyError:
            raise KeyError(f"Unknown step id in placeholder: {step_id}")
        # Convert msgspec Structs to builtins for traversal
        current = msgspec.to_builtins(current)
        # Walk remaining parts
        for p in parts[1:]:
            if p.startswith("["):
                idx = int(p[1:-1])
                current = current[idx]
            else:
                current = current[p]
        return current