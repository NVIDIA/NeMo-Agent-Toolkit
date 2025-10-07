#!/usr/bin/env python3
"""
Simple script to test the MCP server debug endpoint and verify it's working correctly.
"""

import argparse
import asyncio
import json

import aiohttp


async def test_debug_endpoint(url: str):
    """Test the MCP server debug endpoint."""
    print(f"Testing MCP server at: {url}")
    print("=" * 70)

    # Derive debug URL
    debug_url = url.replace('/mcp', '/debug/tools/list')
    print(f"Debug endpoint: {debug_url}")
    print()

    try:
        async with aiohttp.ClientSession() as session:
            # Test debug endpoint
            print("Fetching tools list...")
            async with session.get(debug_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                print()

                if response.status == 200:
                    text = await response.text()
                    print(f"Raw response (first 500 chars):")
                    print(text[:500])
                    print()

                    try:
                        data = json.loads(text)
                        print(f"Response type: {type(data)}")
                        print(f"Parsed JSON:")
                        print(json.dumps(data, indent=2))
                        print()

                        # Try to extract tool names
                        if isinstance(data, list):
                            print(f"✓ Response is a list with {len(data)} items")
                            for i, item in enumerate(data[:3]):  # Show first 3
                                print(f"  Item {i}: {type(item)} - {item}")
                        elif isinstance(data, dict):
                            print(f"✓ Response is a dict with keys: {list(data.keys())}")
                            if 'tools' in data:
                                print(f"  Found 'tools' key with {len(data['tools'])} items")
                        else:
                            print(f"⚠ Unexpected response type: {type(data)}")

                        print()
                        print("✓ MCP server is responding correctly")
                        return True

                    except json.JSONDecodeError as e:
                        print(f"✗ Failed to parse JSON: {e}")
                        return False
                else:
                    print(f"✗ Server returned error status: {response.status}")
                    text = await response.text()
                    print(f"Response: {text[:500]}")
                    return False

    except aiohttp.ClientConnectorError:
        print("✗ Cannot connect to MCP server")
        print("  Make sure the server is running:")
        print(f"  nat mcp serve --config_file <your_config.yml>")
        return False
    except asyncio.TimeoutError:
        print("✗ Request timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MCP server debug endpoint")
    parser.add_argument("--url",
                        default="http://localhost:9901/mcp",
                        help="MCP server URL (default: http://localhost:9901/mcp)")

    args = parser.parse_args()

    success = asyncio.run(test_debug_endpoint(args.url))

    if success:
        print("\n" + "=" * 70)
        print("SUCCESS: MCP server is ready for load testing")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("FAILED: Fix the issues above before running load tests")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
