"""
Tests for backend enumeration.

This module tests the BackendType enum.
"""

import pytest
from argos.backend import BackendType


class TestBackendType:
    """Test cases for BackendType enum."""

    def test_backend_type_values(self):
        """Test backend type enum values."""
        assert BackendType.IN_MEMORY.value == "in_memory"
        assert BackendType.TEMPORAL.value == "temporal"
        assert BackendType.CELERY.value == "celery"

    def test_backend_type_membership(self):
        """Test membership in BackendType enum."""
        assert "in_memory" in BackendType
        assert "temporal" in BackendType
        assert "celery" in BackendType
        assert "unknown" not in BackendType

    def test_backend_type_iteration(self):
        """Test iterating over BackendType values."""
        backends = list(BackendType)
        
        assert len(backends) == 3
        assert BackendType.IN_MEMORY in backends
        assert BackendType.TEMPORAL in backends
        assert BackendType.CELERY in backends

    def test_backend_type_comparison(self):
        """Test comparison of BackendType values."""
        in_memory1 = BackendType.IN_MEMORY
        in_memory2 = BackendType.IN_MEMORY
        temporal = BackendType.TEMPORAL
        
        assert in_memory1 == in_memory2
        assert in_memory1 != temporal
        assert in_memory1.value == "in_memory"
        assert temporal.value == "temporal"

    def test_backend_type_string_representation(self):
        """Test string representation of BackendType."""
        assert BackendType.IN_MEMORY.value == "in_memory"
        assert BackendType.TEMPORAL.value == "temporal"
        assert BackendType.CELERY.value == "celery"

    def test_backend_type_from_string(self):
        """Test creating BackendType from string."""
        assert BackendType("in_memory") == BackendType.IN_MEMORY
        assert BackendType("temporal") == BackendType.TEMPORAL
        assert BackendType("celery") == BackendType.CELERY

    def test_backend_type_invalid_string(self):
        """Test creating BackendType from invalid string."""
        with pytest.raises(ValueError):
            BackendType("invalid_backend")

    def test_backend_type_name_attribute(self):
        """Test name attribute of BackendType values."""
        assert BackendType.IN_MEMORY.name == "IN_MEMORY"
        assert BackendType.TEMPORAL.name == "TEMPORAL"
        assert BackendType.CELERY.name == "CELERY"

    def test_backend_type_unique_values(self):
        """Test that all BackendType values are unique."""
        values = [backend.value for backend in BackendType]
        assert len(values) == len(set(values))

    def test_backend_type_enum_properties(self):
        """Test BackendType enum properties."""
        # Should be an enum
        from enum import Enum
        assert issubclass(BackendType, Enum)
        
        # Should have exactly 3 members
        assert len(BackendType) == 3
        
        # All members should be BackendType instances
        for backend in BackendType:
            assert isinstance(backend, BackendType)