"""
Test utilities for mini_trainer tests.

This package contains shared utilities used across multiple test files.
"""

from .orthogonality import (
    compute_angle_differences,
    OrthogonalityTracker,
    check_gradient_orthogonality,
    check_parameter_orthogonality,
)

__all__ = [
    'compute_angle_differences',
    'OrthogonalityTracker',
    'check_gradient_orthogonality',
    'check_parameter_orthogonality',
]
