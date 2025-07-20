#!/usr/bin/env python3
"""
Test runner for the nano Qwen3 serving engine.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run the test suite."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    cmd.append("tests/")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.append("--cov=nano_qwen3_serving")
        cmd.append("--cov-report=html")
        cmd.append("--cov-report=term")
    
    # Filter by test type
    if test_type == "unit":
        cmd.append("-m")
        cmd.append("unit")
    elif test_type == "integration":
        cmd.append("-m")
        cmd.append("integration")
    elif test_type == "fast":
        cmd.append("-m")
        cmd.append("not slow")
    elif test_type == "slow":
        cmd.append("-m")
        cmd.append("slow")
    
    # Add additional options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings
    ])
    
    print(f"Running tests: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the tests
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode


def run_specific_test(test_file, verbose=False):
    """Run a specific test file."""
    cmd = ["python", "-m", "pytest", f"tests/{test_file}"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings",
    ])
    
    print(f"Running specific test: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "pytest",
        "torch",
        "transformers",
        "loguru",
        "pydantic",
        "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True


def check_mps_availability():
    """Check if MPS is available."""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ MPS is available")
            return True
        else:
            print("⚠️  MPS is not available (tests will run on CPU)")
            return False
    except ImportError:
        print("❌ PyTorch not available")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for nano Qwen3 serving engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all tests
  %(prog)s --type unit        # Run only unit tests
  %(prog)s --type integration # Run only integration tests
  %(prog)s --type fast        # Run fast tests (exclude slow)
  %(prog)s --file test_sampling_params.py  # Run specific test file
  %(prog)s --verbose          # Verbose output
  %(prog)s --coverage         # Run with coverage report
        """
    )
    
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "fast", "slow"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--file",
        help="Run a specific test file"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    parser.add_argument(
        "--check-mps",
        action="store_true",
        help="Check MPS availability and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        return 0 if check_dependencies() else 1
    
    # Check MPS if requested
    if args.check_mps:
        return 0 if check_mps_availability() else 1
    
    # Check dependencies before running tests
    if not check_dependencies():
        return 1
    
    # Check MPS availability
    check_mps_availability()
    
    print()
    
    # Run tests
    if args.file:
        return run_specific_test(args.file, args.verbose)
    else:
        return run_tests(args.type, args.verbose, args.coverage)


if __name__ == "__main__":
    sys.exit(main()) 