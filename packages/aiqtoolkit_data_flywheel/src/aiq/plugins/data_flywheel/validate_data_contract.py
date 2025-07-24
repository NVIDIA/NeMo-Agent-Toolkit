#!/usr/bin/env python3
"""
Data Flywheel Contract Validation Script

This script validates all data files in the 'data' directory against the
NemoDFWRecord contract defined in dfw_record.py.

Features:
- Validates JSONL files against the contract schema only
- Strict contract-based validation using Pydantic
- Provides detailed error reporting
- Generates validation summary report

Usage:
    uv run python validate_data_contract.py [--data-dir DATA_DIR] [--output OUTPUT_FILE]

Examples:
    # Validate all files in default 'data' directory
    uv run python validate_data_contract.py

    # Validate files in custom directory
    uv run python validate_data_contract.py --data-dir ./my_data

    # Save validation report to file
    uv run python validate_data_contract.py --output validation_report.json

Requirements:
    - Python 3.8+
    - uv package manager
    - pydantic
    - jsonlines (for JSONL file handling)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonlines

# Import the contract classes
from aiq.plugins.data_flywheel.schemas.dfw_record import NemoDFWRecord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("validation.log")],
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.total_records = 0
        self.valid_records = 0
        self.invalid_records = 0
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.contract_versions: dict[str, int] = {}
        self.workload_ids: dict[str, int] = {}
        self.client_ids: dict[str, int] = {}

    def add_error(self, record_index: int, error_type: str, message: str, details: dict | None = None):
        """Add an error to the validation result."""
        self.errors.append({
            "record_index": record_index,
            "error_type": error_type,
            "message": message,
            "details": details or {},
        })
        self.invalid_records += 1

    def add_warning(self, record_index: int, warning_type: str, message: str, details: dict | None = None):
        """Add a warning to the validation result."""
        self.warnings.append({
            "record_index": record_index,
            "warning_type": warning_type,
            "message": message,
            "details": details or {},
        })

    def add_record(self, record: NemoDFWRecord):
        """Add a valid record and track statistics."""
        self.valid_records += 1

        # Track contract versions
        version = record.contract_version.value
        self.contract_versions[version] = self.contract_versions.get(version, 0) + 1

        # Track workload IDs
        workload_id = record.workload_id
        self.workload_ids[workload_id] = self.workload_ids.get(workload_id, 0) + 1

        # Track client IDs
        client_id = record.client_id
        self.client_ids[client_id] = self.client_ids.get(client_id, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "file_path": self.file_path,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "success_rate": (self.valid_records / self.total_records * 100) if self.total_records > 0 else 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "contract_versions": self.contract_versions,
            "workload_ids": self.workload_ids,
            "client_ids": self.client_ids,
        }


class DataContractValidator:
    """Validator for Data Flywheel contract compliance."""

    def __init__(self):
        pass

    def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate a single JSONL file against the contract."""
        result = ValidationResult(str(file_path))

        logger.info(f"Validating file: {file_path}")

        try:
            with jsonlines.open(file_path) as reader:
                for record_index, raw_record in enumerate(reader):
                    result.total_records += 1

                    try:
                        # Validate against contract only
                        record = NemoDFWRecord(**raw_record)
                        result.add_record(record)

                    except Exception as e:
                        # Handle validation errors
                        error_type = type(e).__name__
                        result.add_error(
                            record_index=record_index,
                            error_type=error_type,
                            message=str(e),
                            details={"raw_record": raw_record},
                        )

        except Exception as e:
            # Handle file reading errors
            result.add_error(record_index=0, error_type="FileError", message=f"Failed to read file: {e!s}")

        return result

    def validate_directory(self, data_dir: Path) -> list[ValidationResult]:
        """Validate all JSONL files in a directory."""
        results = []

        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return results

        # Find all JSONL files
        jsonl_files = list(data_dir.glob("*.jsonl"))

        if not jsonl_files:
            logger.warning(f"No JSONL files found in {data_dir}")
            return results

        logger.info(f"Found {len(jsonl_files)} JSONL files to validate")

        for file_path in jsonl_files:
            result = self.validate_file(file_path)
            results.append(result)

        return results


def generate_summary_report(results: list[ValidationResult]) -> dict[str, Any]:
    """Generate a summary report from validation results."""
    total_files = len(results)
    total_records = sum(r.total_records for r in results)
    total_valid = sum(r.valid_records for r in results)
    total_invalid = sum(r.invalid_records for r in results)
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)

    # Aggregate statistics
    all_contract_versions = {}
    all_workload_ids = {}
    all_client_ids = {}

    for result in results:
        for version, count in result.contract_versions.items():
            all_contract_versions[version] = all_contract_versions.get(version, 0) + count
        for workload_id, count in result.workload_ids.items():
            all_workload_ids[workload_id] = all_workload_ids.get(workload_id, 0) + count
        for client_id, count in result.client_ids.items():
            all_client_ids[client_id] = all_client_ids.get(client_id, 0) + count

    return {
        "validation_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": total_files,
            "total_records": total_records,
            "total_valid_records": total_valid,
            "total_invalid_records": total_invalid,
            "overall_success_rate": (total_valid / total_records * 100) if total_records > 0 else 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
        },
        "statistics": {
            "contract_versions": all_contract_versions,
            "workload_ids": all_workload_ids,
            "client_ids": all_client_ids,
        },
        "file_results": [result.to_dict() for result in results],
    }


def print_summary_report(report: dict[str, Any]):
    """Print a human-readable summary report."""
    summary = report["summary"]
    stats = report["statistics"]

    print("\n" + "=" * 80)
    print("DATA FLYWHEEL CONTRACT VALIDATION REPORT")
    print("=" * 80)
    print(f"Validation Time: {report['validation_timestamp']}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Total Records: {summary['total_records']}")
    print(f"Valid Records: {summary['total_valid_records']}")
    print(f"Invalid Records: {summary['total_invalid_records']}")
    print(f"Success Rate: {summary['overall_success_rate']:.2f}%")
    print(f"Total Errors: {summary['total_errors']}")
    print(f"Total Warnings: {summary['total_warnings']}")

    print("\n" + "-" * 80)
    print("CONTRACT VERSIONS")
    print("-" * 80)
    for version, count in stats["contract_versions"].items():
        print(f"  {version}: {count} records")

    print("\n" + "-" * 80)
    print("WORKLOAD IDs")
    print("-" * 80)
    for workload_id, count in stats["workload_ids"].items():
        print(f"  {workload_id}: {count} records")

    print("\n" + "-" * 80)
    print("CLIENT IDs")
    print("-" * 80)
    for client_id, count in stats["client_ids"].items():
        print(f"  {client_id}: {count} records")

    print("\n" + "-" * 80)
    print("FILE DETAILS")
    print("-" * 80)
    for file_result in report["file_results"]:
        print(f"\nFile: {file_result['file_path']}")
        print(f"  Records: {file_result['total_records']}")
        print(f"  Valid: {file_result['valid_records']}")
        print(f"  Invalid: {file_result['invalid_records']}")
        print(f"  Success Rate: {file_result['success_rate']:.2f}%")

        if file_result["errors"]:
            print(f"  Errors: {len(file_result['errors'])}")
            for error in file_result["errors"][:3]:  # Show first 3 errors
                print(f"    - Record {error['record_index']}: {error['error_type']} - {error['message']}")
            if len(file_result["errors"]) > 3:
                print(f"    ... and {len(file_result['errors']) - 3} more errors")

        if file_result["warnings"]:
            print(f"  Warnings: {len(file_result['warnings'])}")

    print("\n" + "=" * 80)


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate Data Flywheel data files against the enhanced contract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing JSONL files to validate (default: ./data)",
    )

    parser.add_argument("--output", type=Path, help="Output file for detailed validation report (JSON format)")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Data Flywheel contract validation")

    # Validate data files
    validator = DataContractValidator()
    results = validator.validate_directory(args.data_dir)

    if not results:
        logger.error("No validation results generated")
        sys.exit(1)

    # Generate and display report
    report = generate_summary_report(results)
    print_summary_report(report)

    # Save detailed report if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Detailed report saved to: {args.output}")

    # Exit with error code if there are validation failures
    total_invalid = sum(r.invalid_records for r in results)
    if total_invalid > 0:
        logger.error(f"Validation completed with {total_invalid} invalid records")
        sys.exit(1)
    else:
        logger.info("All records passed validation successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
