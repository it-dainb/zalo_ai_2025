"""
Test Runner Script
==================

Run all tests from smallest to biggest components.

Usage:
    # Run all tests
    python run_all_tests.py

    # Run specific test category
    python run_all_tests.py --category data
    python run_all_tests.py --category model
    python run_all_tests.py --category training
    python run_all_tests.py --category evaluation
    python run_all_tests.py --category inference
    python run_all_tests.py --category e2e

    # Run with verbose output
    python run_all_tests.py --verbose

    # Run specific test file
    python run_all_tests.py --file test_data_loading.py
"""

import sys
import subprocess
from pathlib import Path
import argparse


# Test categories organized from smallest to biggest
TEST_CATEGORIES = {
    'data': [
        'test_data_loading.py',
    ],
    'model': [
        'test_model_components.py',
    ],
    'training': [
        'test_training_components.py',
        'test_training_full.py',
    ],
    'evaluation': [
        'test_evaluation.py',
    ],
    'inference': [
        'test_inference.py',
    ],
    'e2e': [
        'test_e2e_complete_pipeline.py',
    ],
}


def run_tests(test_files, verbose=False, stop_on_error=False):
    """
    Run specified test files.
    
    Args:
        test_files: List of test file names
        verbose: Enable verbose output
        stop_on_error: Stop on first failure
    
    Returns:
        bool: True if all tests passed
    """
    tests_dir = Path(__file__).parent
    all_passed = True
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Running: {test_file}")
        print(f"{'='*70}\n")
        
        # Build pytest command
        cmd = ['pytest', str(test_path)]
        
        if verbose:
            cmd.append('-v')
        
        cmd.extend(['-s', '--tb=short'])  # Show print statements and short traceback
        
        # Run test
        try:
            result = subprocess.run(cmd, cwd=tests_dir.parent.parent)
            
            if result.returncode != 0:
                all_passed = False
                print(f"\n❌ {test_file} FAILED\n")
                
                if stop_on_error:
                    print("Stopping on error (--stop-on-error flag)")
                    return False
            else:
                print(f"\n✅ {test_file} PASSED\n")
                
        except Exception as e:
            print(f"\n❌ Error running {test_file}: {str(e)}\n")
            all_passed = False
            
            if stop_on_error:
                return False
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Run YOLOv8n-RefDet tests from smallest to biggest components'
    )
    parser.add_argument(
        '--category',
        choices=['data', 'model', 'training', 'evaluation', 'inference', 'e2e', 'all'],
        default='all',
        help='Test category to run (default: all)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Run specific test file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop on first test failure'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tests'
    )
    
    args = parser.parse_args()
    
    # List tests
    if args.list:
        print("\nAvailable test categories (from smallest to biggest):\n")
        for category, files in TEST_CATEGORIES.items():
            print(f"  {category}:")
            for f in files:
                print(f"    - {f}")
        print()
        return
    
    # Determine which tests to run
    if args.file:
        test_files = [args.file]
    elif args.category == 'all':
        # Run all tests in order from smallest to biggest
        test_files = []
        for category in ['data', 'model', 'training', 'evaluation', 'inference', 'e2e']:
            test_files.extend(TEST_CATEGORIES[category])
    else:
        test_files = TEST_CATEGORIES[args.category]
    
    # Print header
    print("\n" + "="*70)
    print("YOLOv8n-RefDet Test Suite")
    print("Testing from smallest to biggest components")
    print("="*70)
    print(f"\nRunning {len(test_files)} test file(s):")
    for f in test_files:
        print(f"  - {f}")
    print()
    
    # Run tests
    all_passed = run_tests(test_files, verbose=args.verbose, stop_on_error=args.stop_on_error)
    
    # Print summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
