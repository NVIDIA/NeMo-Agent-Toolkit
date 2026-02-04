"""
Test script to validate the MCP enum caching fix.
Based on the reproduction code from issue #1441.
"""
import sys
import typing
from enum import Enum
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "nvidia_nat_mcp" / "src"))

from nat.plugins.mcp.utils import model_from_mcp_schema


def test_enum_caching_fix():
    """Test that enum classes are cached and reused."""
    print("=" * 60)
    print("Testing MCP Enum Caching Fix")
    print("=" * 60)

    # Create a simple MCP schema with an enum
    mcp_schema = {
        "type": "object",
        "properties": {
            "sortBy": {
                "type": "string",
                "enum": ["Relevance", "Votes", "DownloadCount"]
            }
        },
        "required": []
    }

    # Create schema twice (simulating what happens in the bug)
    print("\n1. Creating first schema...")
    Schema1 = model_from_mcp_schema("search_datasets", mcp_schema)
    field1_annotation = Schema1.model_fields["sortBy"].annotation

    # Extract the actual enum class (might be in a Union)
    import typing
    if hasattr(field1_annotation, '__args__'):
        # It's a Union, extract the enum
        enum1 = [arg for arg in typing.get_args(field1_annotation) if isinstance(arg, type) and issubclass(arg, Enum)][0]
    else:
        enum1 = field1_annotation
    print(f"   Schema1 enum class: {enum1}")

    print("\n2. Creating second schema (should use cached enum)...")
    Schema2 = model_from_mcp_schema("search_datasets", mcp_schema)
    field2_annotation = Schema2.model_fields["sortBy"].annotation

    # Extract the actual enum class
    if hasattr(field2_annotation, '__args__'):
        enum2 = [arg for arg in typing.get_args(field2_annotation) if isinstance(arg, type) and issubclass(arg, Enum)][0]
    else:
        enum2 = field2_annotation
    print(f"   Schema2 enum class: {enum2}")

    print("\n3. Checking if enum classes are the same...")
    if enum1 is enum2:
        print("   ✓ SUCCESS: Enum classes are identical (cached)")
    else:
        print("   ✗ FAILURE: Enum classes are different (not cached)")
        return False

    # Test validation across schemas
    print("\n4. Testing validation across schemas...")
    try:
        # Create an instance with Schema1's enum
        instance1 = Schema1(sortBy=enum1.Relevance)
        print(f"   Created instance with Schema1: {instance1.sortBy}")

        # Try to validate with Schema2
        validated = Schema2.model_validate({"sortBy": instance1.sortBy})
        print(f"   ✓ SUCCESS: Validation passed with Schema2: {validated.sortBy}")
        return True
    except Exception as e:
        print(f"   ✗ FAILURE: Validation failed: {e}")
        return False


def test_different_enums_not_cached():
    """Test that different enums get different cache entries."""
    print("\n" + "=" * 60)
    print("Testing Different Enums Get Different Cache Entries")
    print("=" * 60)

    schema1 = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["Active", "Inactive"]
            }
        }
    }

    schema2 = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["Open", "Closed"]
            }
        }
    }

    print("\n1. Creating schema with ['Active', 'Inactive']...")
    Model1 = model_from_mcp_schema("workflow", schema1)
    enum1 = Model1.model_fields["status"].annotation

    print("2. Creating schema with ['Open', 'Closed']...")
    Model2 = model_from_mcp_schema("task", schema2)
    enum2 = Model2.model_fields["status"].annotation

    print("\n3. Checking that different enums are NOT the same...")
    if enum1 is not enum2:
        print("   ✓ SUCCESS: Different enum values get different classes")
        return True
    else:
        print("   ✗ FAILURE: Different enums incorrectly cached as same")
        return False


if __name__ == "__main__":
    print("\nMCP Enum Caching Fix Validation\n")

    test1_passed = test_enum_caching_fix()
    test2_passed = test_different_enums_not_cached()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test 1 (Enum Caching):          {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Test 2 (Different Enums):       {'PASSED ✓' if test2_passed else 'FAILED ✗'}")

    if test1_passed and test2_passed:
        print("\n✓ All tests passed! The fix works correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Fix needs adjustment.")
        sys.exit(1)
