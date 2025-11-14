"""
Utilities for testing OSFT orthogonality constraints.

This module provides reusable utilities for verifying that OSFT gradient
projection and parameter updates maintain orthogonality constraints.

Extracted from regression_tests/test_osft_orthogonalization.py for use
in unit tests that don't require full-scale distributed training.
"""

import torch
import math
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class OrthogonalityMetrics:
    """Stores metrics for a single orthogonality check."""
    param_name: str
    check_type: str  # 'U_grad', 'V_grad', 'U_param', 'V_param'
    max_angle_diff: float  # in degrees
    step: int


class OrthogonalityTracker:
    """Tracks orthogonality metrics across training steps."""

    def __init__(self, margin_deg: float = 1.0):
        self.margin_deg = margin_deg
        self.metrics: Dict[str, Dict[str, OrthogonalityMetrics]] = {}
        self.total_checks = 0
        self.failed_checks = 0

    def update(self, param_name: str, check_type: str, max_angle_diff: float, step: int):
        """Update tracker with new measurement."""
        self.total_checks += 1

        if max_angle_diff > self.margin_deg:
            self.failed_checks += 1

        key = f"{param_name}:{check_type}"

        if key not in self.metrics:
            self.metrics[key] = {
                'param_name': param_name,
                'check_type': check_type,
                'max_angle_diff': max_angle_diff,
                'step': step
            }
        else:
            # Update if this is worse
            if max_angle_diff > self.metrics[key]['max_angle_diff']:
                self.metrics[key]['max_angle_diff'] = max_angle_diff
                self.metrics[key]['step'] = step

    def get_top_violations(self, n: int = 5) -> List[Dict]:
        """Get top N worst violations."""
        sorted_metrics = sorted(
            self.metrics.values(),
            key=lambda x: x['max_angle_diff'],
            reverse=True
        )
        return sorted_metrics[:n]

    def is_successful(self) -> bool:
        """Check if all measurements passed."""
        for metric in self.metrics.values():
            if metric['max_angle_diff'] > self.margin_deg:
                return False
        return True

    def get_summary(self) -> str:
        """Generate summary report."""
        lines = []
        lines.append("=" * 80)
        lines.append("OSFT ORTHOGONALITY TEST SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total checks performed: {self.total_checks}")
        lines.append(f"Failed checks (>{self.margin_deg}°): {self.failed_checks}")
        lines.append(f"Pass rate: {100 * (1 - self.failed_checks / max(self.total_checks, 1)):.2f}%")
        lines.append("")

        if self.is_successful():
            lines.append("✅ RESULT: PASSED - All orthogonality constraints satisfied!")
        else:
            lines.append("❌ RESULT: FAILED - Orthogonality violations detected!")

        lines.append("")
        lines.append("Top 5 Largest Angle Deviations:")
        lines.append("-" * 80)
        lines.append(f"{'Rank':<6}{'Parameter':<40}{'Check Type':<15}{'Max Diff (°)':<15}{'Step':<10}")
        lines.append("-" * 80)

        for i, metric in enumerate(self.get_top_violations(5), 1):
            lines.append(
                f"{i:<6}{metric['param_name']:<40}{metric['check_type']:<15}"
                f"{metric['max_angle_diff']:<15.4f}{metric['step']:<10}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


def compute_angle_differences(A: torch.Tensor, B: torch.Tensor = None, top_n: int = 5) -> List[float]:
    """
    Compute angle differences between matrices A and B, returning the top N worst deviations from orthogonality.

    Args:
        A: First matrix
        B: Second matrix (if None, assumes A)
        top_n: Number of worst angle deviations to return

    Returns:
        List of top N angle differences in degrees, or empty list if matrices are invalid
    """
    try:
        solo_matrix = False
        if B is None:
            B = A
            solo_matrix = True

        # Handle DTensor conversion for distributed training
        if isinstance(A, torch.distributed._tensor.api.DTensor):
            A = A.to_local()
        if isinstance(B, torch.distributed._tensor.api.DTensor):
            B = B.to_local()

        if A.dim() != 2 or B.dim() != 2:
            return []

        ma, na = A.shape
        mb, nb = B.shape

        if ma != mb:
            return []

        # Compute orthogonality angles
        Mag_A = A.pow(2).sum(dim=0, keepdim=True).sqrt()
        Mag_B = B.pow(2).sum(dim=0, keepdim=True).sqrt()
        Mag = Mag_B.T @ Mag_A

        Proj = B.T @ A
        angles = (Proj / Mag).abs().clamp(0.0, 1.0).acos()

        # Expected angles (should be pi/2 for orthogonality)
        correct = torch.ones_like(angles) * (math.pi / 2)

        # For same matrix, don't count self-angles
        if solo_matrix:
            correct[torch.arange(nb), torch.arange(na)] = 0.0

        # Compute angle differences in degrees
        diff = (angles - correct).abs() * 180 / math.pi

        # Get top N worst deviations
        diff_flat = diff.flatten()
        if solo_matrix:
            # Exclude diagonal elements for same matrix
            mask = torch.ones(diff.shape, dtype=torch.bool, device=diff.device)
            mask[torch.arange(min(nb, na)), torch.arange(min(nb, na))] = False
            diff_flat = diff[mask]

        if len(diff_flat) == 0:
            return []

        top_diffs, _ = torch.topk(diff_flat, min(top_n, len(diff_flat)))
        return top_diffs.cpu().tolist()

    except Exception as e:
        print(f"Error computing angle differences: {e}")
        return []


def check_gradient_orthogonality(
    model,
    module,
    step: int,
    tracker: OrthogonalityTracker
) -> None:
    """
    Check if gradients of U_low and V_low are orthogonal to U_high and V_high.

    Args:
        model: OSFT model
        module: Module with OSFT parameters attached
        step: Current training step
        tracker: OrthogonalityTracker to update
    """
    svd_dict = model.get_svd_dict_for_module(module)
    if svd_dict["U_low"].grad is None or svd_dict["V_low"].grad is None:
        return

    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    V_low = svd_dict["V_low"]

    # get the safe_name for tracking
    safe_name = module.osft_params.safe_name

    # we need to pull the gradients out before casting these variables to full_tensor,
    # since `.full_tensor` doesn't return a tensor with the .grad attribute populated
    dU_low = U_low.grad.full_tensor() if hasattr(U_low.grad, 'full_tensor') else U_low.grad
    dV_low = V_low.grad.full_tensor() if hasattr(V_low.grad, 'full_tensor') else V_low.grad

    if hasattr(U_high, 'full_tensor'):
        U_high = U_high.full_tensor()
    if hasattr(V_high, 'full_tensor'):
        V_high = V_high.full_tensor()
    if hasattr(U_low, 'full_tensor'):
        U_low = U_low.full_tensor()
    if hasattr(V_low, 'full_tensor'):
        V_low = V_low.full_tensor()


    # Check U gradient orthogonality
    u_grad_diffs = compute_angle_differences(U_high, dU_low, top_n=1)
    if u_grad_diffs:
        tracker.update(safe_name, 'U_grad', u_grad_diffs[0], step)

    # Check V gradient orthogonality
    v_grad_diffs = compute_angle_differences(V_high.T, dV_low.T, top_n=1)
    if v_grad_diffs:
        tracker.update(safe_name, 'V_grad', v_grad_diffs[0], step)


def check_parameter_orthogonality(
    model,
    module,
    step: int,
    tracker: OrthogonalityTracker
) -> None:
    """
    Check if post-update U_low and V_low are orthogonal to U_high and V_high.

    Args:
        model: OSFT model
        module: Module with OSFT parameters attached
        step: Current training step
        tracker: OrthogonalityTracker to update
    """
    svd_dict = model.get_svd_dict_for_module(module)

    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    V_low = svd_dict["V_low"]

    # get the safe_name for tracking
    safe_name = module.osft_params.safe_name

    if hasattr(U_high, 'full_tensor'):
        U_high = U_high.full_tensor()
    if hasattr(V_high, 'full_tensor'):
        V_high = V_high.full_tensor()
    if hasattr(U_low, 'full_tensor'):
        U_low = U_low.full_tensor()
    if hasattr(V_low, 'full_tensor'):
        V_low = V_low.full_tensor()

    # Check U parameter orthogonality
    u_param_diffs = compute_angle_differences(U_high, U_low, top_n=1)
    if u_param_diffs:
        tracker.update(safe_name, 'U_param', u_param_diffs[0], step)

    # Check V parameter orthogonality
    v_param_diffs = compute_angle_differences(V_high.T, V_low.T, top_n=1)
    if v_param_diffs:
        tracker.update(safe_name, 'V_param', v_param_diffs[0], step)
