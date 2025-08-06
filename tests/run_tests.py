#!/usr/bin/env python3
"""
Comprehensive test runner for mini_trainer test suite.

This script runs all unit tests for the training pipeline including:
- Batch packing algorithms
- Data loading and sampling
- Model initialization
- Training components
- Main training loop
"""

import sys
import subprocess
import os

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_correctness_tests():
    """Run pytest for all unit tests."""
    print("=" * 60)
    print("RUNNING ALL UNIT TESTS")
    print("=" * 60)
    
    # Run all test files
    test_files = [
        "test_batch_lengths_to_minibatches.py::TestBatchLengthsToMinibatches",
        "test_data_loader.py",
        "test_model_initialization.py", 
        "test_training_components.py",
        "test_training_loop.py",
        "test_async_logger.py",
        "test_integration_small_models.py"
    ]
    
    all_passed = True
    for test_file in test_files:
        print(f"\n>>> Running {test_file}")
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            all_passed = False
            print(f"❌ {test_file} FAILED")
        else:
            print(f"✅ {test_file} PASSED")
    
    return all_passed

def run_performance_comparison():
    """Run the performance comparison between algorithms."""
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE COMPARISON")
    print("=" * 60)
    
    result = subprocess.run([
        sys.executable, "test_batch_lengths_to_minibatches.py"
    ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    """Run all tests."""
    print("Mini Trainer - Comprehensive Test Suite")
    print("========================================")
    print("Testing critical components:")
    print("  • Data loading and sampling")
    print("  • Batch packing algorithms") 
    print("  • Model initialization")
    print("  • Training utilities")
    print("  • Main training loop")
    print()
    
    # Run correctness tests first
    correctness_passed = run_correctness_tests()
    
    if not correctness_passed:
        print("\n❌ UNIT TESTS FAILED!")
        print("Please fix the issues before running performance comparison.")
        return 1
    
    print("\n✅ ALL UNIT TESTS PASSED!")
    
    # Run performance comparison for batch packing
    performance_passed = run_performance_comparison()
    
    if not performance_passed:
        print("\n❌ PERFORMANCE COMPARISON FAILED!")
        return 1
    
    print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
