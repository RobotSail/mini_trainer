"""
Test utilities for mini_trainer tests.

This package contains shared utilities used across multiple test files.
"""

from .orthogonality import (
    OrthogonalityTracker,
    check_gradient_orthogonality,
    check_parameter_orthogonality,
    compute_angle_differences,
)

__all__ = [
    "OrthogonalityTracker",
    "check_gradient_orthogonality",
    "check_parameter_orthogonality",
    "compute_angle_differences",
]
