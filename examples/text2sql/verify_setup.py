#!/usr/bin/env python3
"""
Verify that the text2sql MCP server setup is correct.

This script checks:
1. Required Python packages are installed
2. Configuration file is valid
3. Environment variables are set
4. All required files exist
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def print_section(title: str):
    """Print a section header."""
    print(f"\n{BLUE}{'=' * 70}{NC}")
    print(f"{BLUE}{title}{NC}")
    print(f"{BLUE}{'=' * 70}{NC}")


def check_pass(message: str):
    """Print a passing check."""
    print(f"{GREEN}✓{NC} {message}")


def check_fail(message: str):
    """Print a failing check."""
    print(f"{RED}✗{NC} {message}")


def check_warn(message: str):
    """Print a warning."""
    print(f"{YELLOW}⚠{NC} {message}")


def check_python_packages():
    """Check if required Python packages are installed."""
    print_section("Checking Python Packages")

    packages = {
        'aiohttp': 'Load testing HTTP client',
        'psutil': 'Memory monitoring',
        'pydantic': 'Configuration validation',
        'fastapi': 'NAT dependencies',
        'langchain': 'LLM framework',
        'langchain_nvidia_ai_endpoints': 'NVIDIA NIM integration',
        'vanna': 'Text-to-SQL framework',
        'pymilvus': 'Milvus vector store',
    }

    all_installed = True
    for package, description in packages.items():
        try:
            __import__(package)
            check_pass(f"{package:35s} - {description}")
        except ImportError:
            check_fail(f"{package:35s} - {description} (NOT INSTALLED)")
            all_installed = False

    if not all_installed:
        print(f"\n{YELLOW}Install missing packages:{NC}")
        print("  pip install aiohttp psutil")
        print("  pip install -e .[mcp]")
        print("  pip install -e talk-to-supply-chain-tools/")

    return all_installed


def check_environment_variables():
    """Check if required environment variables are set."""
    print_section("Checking Environment Variables")

    required = {
        'NVIDIA_API_KEY': 'NVIDIA NIM API key',
    }

    optional = {
        'MILVUS_HOST': 'Milvus cloud host (if using vanna_remote: true)',
        'MILVUS_PORT': 'Milvus cloud port',
        'MILVUS_USERNAME': 'Milvus cloud username',
        'MILVUS_PASSWORD': 'Milvus cloud password',
    }

    all_set = True

    # Check required
    for var, description in required.items():
        value = os.environ.get(var)
        if value:
            # Mask the value for security
            masked = value[:8] + '...' if len(value) > 8 else '***'
            check_pass(f"{var:20s} = {masked:20s} - {description}")
        else:
            check_fail(f"{var:20s} NOT SET - {description}")
            all_set = False

    # Check optional
    print(f"\n{BLUE}Optional (for remote Milvus):{NC}")
    for var, description in optional.items():
        value = os.environ.get(var)
        if value:
            masked = value[:8] + '...' if len(value) > 8 else '***'
            check_pass(f"{var:20s} = {masked:20s} - {description}")
        else:
            check_warn(f"{var:20s} NOT SET - {description}")

    if not all_set:
        print(f"\n{YELLOW}Set environment variables:{NC}")
        print("  export NVIDIA_API_KEY='your-api-key'")
        print("  export MILVUS_HOST='your-host'")
        print("  export MILVUS_PORT='19530'")
        print("  export MILVUS_USERNAME='your-username'")
        print("  export MILVUS_PASSWORD='your-password'")

    return all_set


def check_files():
    """Check if required files exist."""
    print_section("Checking Required Files")

    base_dir = Path("examples/text2sql")

    files = {
        'config_text2sql_mcp.yml': 'MCP server configuration',
        'text2sql_standalone.py': 'Standalone text2sql function',
        'text2sql_function.py': 'Production text2sql function',
        'sql_utils.py': 'Vanna utilities',
        'text2sql_load_test.py': 'Load testing script',
        'monitor_server_memory.py': 'Memory monitoring script',
        'run_text2sql_load_test.sh': 'Automated test runner',
        'README.md': 'Main documentation',
        'QUICKSTART.md': 'Quick start guide',
        '__init__.py': 'Python package file',
    }

    all_exist = True
    for filename, description in files.items():
        filepath = base_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            check_pass(f"{filename:30s} ({size:>6d} bytes) - {description}")
        else:
            check_fail(f"{filename:30s} MISSING - {description}")
            all_exist = False

    return all_exist


def check_config_file():
    """Check if configuration file is valid."""
    print_section("Checking Configuration File")

    config_file = Path("examples/text2sql/config_text2sql_mcp.yml")

    if not config_file.exists():
        check_fail("Configuration file not found")
        return False

    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Check key sections
        if 'functions' in config:
            check_pass("'functions' section found")
            if 'text2sql_standalone' in config['functions']:
                check_pass("'text2sql_standalone' function configured")
            else:
                check_fail("'text2sql_standalone' function not found")
        else:
            check_fail("'functions' section missing")

        if 'llms' in config:
            check_pass("'llms' section found")
        else:
            check_warn("'llms' section missing")

        if 'embedders' in config:
            check_pass("'embedders' section found")
        else:
            check_warn("'embedders' section missing")

        if 'workflow' in config:
            check_pass("'workflow' section found")
        else:
            check_warn("'workflow' section missing")

        return True

    except ImportError:
        check_warn("PyYAML not installed, skipping config validation")
        print("  Install with: pip install pyyaml")
        return True  # Don't fail on this
    except Exception as e:
        check_fail(f"Configuration file is invalid: {e}")
        return False


def check_nat_command():
    """Check if nat command is available."""
    print_section("Checking NAT Command")

    import subprocess
    try:
        result = subprocess.run(['nat', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            check_pass(f"'nat' command available: {version}")
            return True
        else:
            check_fail("'nat' command failed")
            return False
    except FileNotFoundError:
        check_fail("'nat' command not found")
        print("\n  Install NAT with: pip install -e .[mcp]")
        return False
    except Exception as e:
        check_fail(f"Error checking 'nat' command: {e}")
        return False


def main():
    """Run all verification checks."""
    print(f"{BLUE}")
    print("=" * 70)
    print("  Text2SQL MCP Server Setup Verification")
    print("=" * 70)
    print(f"{NC}")

    results = {
        'Python Packages': check_python_packages(),
        'Environment Variables': check_environment_variables(),
        'Files': check_files(),
        'Configuration': check_config_file(),
        'NAT Command': check_nat_command(),
    }

    # Print summary
    print_section("Summary")

    all_passed = True
    for check, passed in results.items():
        if passed:
            check_pass(f"{check:25s} PASSED")
        else:
            check_fail(f"{check:25s} FAILED")
            all_passed = False

    print(f"\n{BLUE}{'=' * 70}{NC}\n")

    if all_passed:
        print(f"{GREEN}✓ All checks passed! Setup is complete.{NC}\n")
        print("You can now run the load test:")
        print(f"  {BLUE}./examples/text2sql/run_text2sql_load_test.sh{NC}\n")
        print("Or start the server manually:")
        print(f"  {BLUE}nat mcp serve --config_file examples/text2sql/config_text2sql_mcp.yml{NC}\n")
        return 0
    else:
        print(f"{RED}✗ Some checks failed. Please fix the issues above.{NC}\n")
        print("See the following for help:")
        print(f"  {BLUE}examples/text2sql/README.md{NC}")
        print(f"  {BLUE}examples/text2sql/QUICKSTART.md{NC}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
