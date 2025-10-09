#!/usr/bin/env python3
"""
Verification script to check if the load testing environment is properly set up.

This script checks:
- Required Python packages are installed
- Configuration files exist
- Environment variables are set
- Scripts are executable
"""

import os
import sys
from pathlib import Path


def check_package(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False


def check_file(file_path: Path) -> bool:
    """Check if a file exists."""
    if file_path.exists():
        print(f"✓ {file_path} exists")
        return True
    else:
        print(f"✗ {file_path} does NOT exist")
        return False


def check_executable(file_path: Path) -> bool:
    """Check if a file is executable."""
    if file_path.exists() and os.access(file_path, os.X_OK):
        print(f"✓ {file_path} is executable")
        return True
    else:
        print(f"✗ {file_path} is NOT executable")
        return False


def check_env_var(var_name: str) -> bool:
    """Check if an environment variable is set."""
    value = os.environ.get(var_name)
    if value:
        print(f"✓ {var_name} is set")
        return True
    else:
        print(f"✗ {var_name} is NOT set")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("TEXT2SQL LOAD TEST SETUP VERIFICATION")
    print("=" * 70)

    all_checks_passed = True

    # Check Python packages
    print("\n1. Checking Python packages...")
    packages = ["psutil", "requests", "aiohttp", "nat"]
    for package in packages:
        if not check_package(package):
            all_checks_passed = False

    # Check files
    print("\n2. Checking files...")
    base_dir = Path(__file__).parent
    files = [
        base_dir / "load_test_text2sql.py",
        base_dir / "run_text2sql_memory_leak_test.py",
        base_dir / "configs" / "config_text2sql_mcp.yml",
        base_dir / "LOAD_TESTING.md",
    ]
    for file_path in files:
        if not check_file(file_path):
            all_checks_passed = False

    # Check executability
    print("\n3. Checking script permissions...")
    scripts = [
        base_dir / "load_test_text2sql.py",
        base_dir / "run_text2sql_memory_leak_test.py",
    ]
    for script in scripts:
        if not check_executable(script):
            all_checks_passed = False

    # Check environment variables
    print("\n4. Checking environment variables...")
    env_vars = ["NVIDIA_API_KEY"]
    for var in env_vars:
        if not check_env_var(var):
            all_checks_passed = False
            print(f"   Hint: Set {var} in your .env file or export it")

    # Optional checks
    print("\n5. Checking optional environment variables...")
    optional_vars = ["MILVUS_HOST", "MILVUS_PORT", "MILVUS_USERNAME"]
    optional_set = []
    for var in optional_vars:
        if check_env_var(var):
            optional_set.append(var)

    if not optional_set:
        print("   Note: Using local Milvus (no cloud variables set)")

    # Check monitor script
    print("\n6. Checking debug tools...")
    monitor_script = base_dir.parent.parent / "debug_tools" / "monitor_memory.py"
    if not check_file(monitor_script):
        all_checks_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✓ ALL REQUIRED CHECKS PASSED")
        print("\nYou're ready to run load tests!")
        print("\nQuick start:")
        print("  cd examples/text2sql")
        print("  python run_text2sql_memory_leak_test.py")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running load tests.")
        print("\nSetup instructions:")
        print("1. Install packages: pip install psutil requests aiohttp")
        print("2. Install text2sql: cd examples/text2sql && uv pip install -e .")
        print("3. Set environment: cp .env.example .env and edit")
        print("4. Train Vanna: Set train_on_startup: true in config, then run once")
    print("=" * 70)

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
