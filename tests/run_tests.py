#!/usr/bin/env python3
"""
Simple test runner for batch_lengths_to_minibatches performance comparison.

This script runs both the correctness tests and the performance comparison
between the greedy and LPT algorithms.
"""

import sys
import subprocess
import os

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_correctness_tests():
    """Run pytest for correctness verification."""
    print("=" * 60)
    print("RUNNING CORRECTNESS TESTS")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_batch_lengths_to_minibatches.py::TestBatchLengthsToMinibatches",
            "-v",
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def run_performance_comparison():
    """Run the performance comparison between algorithms."""
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE COMPARISON")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "test_batch_lengths_to_minibatches.py"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    """Run all tests."""
    print("Batch Lengths to Minibatches - Test Suite")
    print("=========================================")

    # Run correctness tests first
    correctness_passed = run_correctness_tests()

    if not correctness_passed:
        print("\n❌ CORRECTNESS TESTS FAILED!")
        print("Please fix the issues before running performance comparison.")
        return 1

    print("\n✅ ALL CORRECTNESS TESTS PASSED!")

    # Run performance comparison
    performance_passed = run_performance_comparison()

    if not performance_passed:
        print("\n❌ PERFORMANCE COMPARISON FAILED!")
        return 1

    print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
