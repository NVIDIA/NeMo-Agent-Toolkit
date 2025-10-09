#!/usr/bin/env python3
"""
Load testing script for Text2SQL MCP Server to detect memory leaks.

This script simulates multiple concurrent users making text2sql queries
to the MCP server to observe memory usage patterns over time.
"""

import argparse
import asyncio
import logging
import random
import time
from typing import Any

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Realistic text2sql queries based on supply chain use cases
TEXT2SQL_QUERIES = [
    # Shortage queries
    "Show me the top 10 components with highest shortages",
    "Show me the top 20 NVPNs with highest shortages",
    "List components with shortages more than 50 units",
    "What are the components with critical shortages?",
    "Display items with shortages for build id PB-61738",

  # Lead time queries
    "What components have lead time greater than 50 days?",
    "Show components without lead time for project E2425",
    "Display average lead time by component type",
    "List all components with lead time exceeding 100 days",

  # Inventory queries
    "Display components with nettable inventory above 1000 units",
    "Show latest demand in next 26 weeks for NVPN 681-24287-0012.A",
    "What is the current inventory status for NVPN 315-1157-000?",
    "Show supply and demand trend for critical components",

  # Build request queries
    "Show me all components without lead time for build id PB-60506",
    "Give me a list of components with insufficient quantity for PB-55330",
    "Show all consigned parts for build id PB-61738 that are green",
    "List all red items for build request PB-61738",

  # Material cost queries
    "Display the latest material cost by CM for NVPN 316-0899-000",
    "Show cost analysis for top 10 most expensive components",
    "What are the components with highest material costs?",

  # CM and site queries
    "Show shortage breakdown by CM site",
    "Display inventory levels across all CM sites",
    "List components by manufacturing site",

  # Trend analysis queries
    "Show demand forecast for next quarter",
    "Display historical shortage trends",
    "What are the trending supply issues?",
]


class Text2SQLLoadTester:
    """Load tester for Text2SQL MCP server that simulates multiple concurrent users."""

    def __init__(
        self,
        server_url: str = "http://localhost:9901/mcp",
        num_users: int = 40,
        calls_per_user: int = 10,
        delay_between_calls: float = 0.5,
        use_cli: bool = True,
    ):
        """
        Initialize the load tester.

        Args:
            server_url: URL of the MCP server
            num_users: Number of concurrent users to simulate
            calls_per_user: Number of tool calls each user should make
            delay_between_calls: Delay in seconds between consecutive calls per user
            use_cli: If True, use nat mcp client CLI (proper protocol). If False, use HTTP (gets 406 errors)
        """
        self.server_url = server_url
        self.num_users = num_users
        self.calls_per_user = calls_per_user
        self.delay_between_calls = delay_between_calls
        self.use_cli = use_cli
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.start_time = None
        self.end_time = None

    async def get_available_tools(self, session: aiohttp.ClientSession) -> list[str]:
        """
        Retrieve list of available tools from the MCP server.

        Args:
            session: aiohttp client session

        Returns:
            List of tool names
        """
        try:
            debug_url = self.server_url.replace('/mcp', '/debug/tools/list')
            async with session.get(debug_url) as response:
                if response.status == 200:
                    tools = await response.json()
                    # Handle different response formats
                    if isinstance(tools, dict):
                        # Response might be wrapped in a dict with a 'tools' key
                        if 'tools' in tools:
                            tools = tools['tools']
                        else:
                            logger.error(f"Unexpected response format: {tools}")
                            return []
                    if not isinstance(tools, list):
                        logger.error(f"Expected list of tools, got: {type(tools)}")
                        return []
                    # Extract tool names
                    tool_names = []
                    for tool in tools:
                        if isinstance(tool, dict) and 'name' in tool:
                            tool_names.append(tool['name'])
                        elif isinstance(tool, str):
                            tool_names.append(tool)
                        else:
                            logger.warning(f"Skipping unexpected tool format: {tool}")
                    logger.info(f"Found {len(tool_names)} tools: {tool_names}")
                    return tool_names
                else:
                    logger.error(f"Failed to get tools: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    async def call_tool_via_cli(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: int,
    ) -> bool:
        """
        Call an MCP tool using the nat mcp client CLI (proper MCP protocol).

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            user_id: ID of the simulated user making the call

        Returns:
            True if call was successful, False otherwise
        """
        try:
            import json

            # Build the CLI command
            cmd = [
                "nat",
                "mcp",
                "client",
                "tool",
                "call",
                tool_name,
                "--url",
                self.server_url,
                "--json-args",
                json.dumps(arguments)
            ]

            # Run the command
            process = await asyncio.create_subprocess_exec(*cmd,
                                                           stdout=asyncio.subprocess.DEVNULL,
                                                           stderr=asyncio.subprocess.DEVNULL)

            returncode = await process.wait()

            if returncode == 0:
                logger.debug(f"User {user_id}: Tool {tool_name} succeeded")
                return True
            else:
                logger.debug(f"User {user_id}: Tool {tool_name} failed with code {returncode}")
                return False

        except Exception as e:
            logger.debug(f"User {user_id}: Tool {tool_name} error: {e}")
            return False

    async def call_tool_via_http(
        self,
        session: aiohttp.ClientSession,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: int,
    ) -> bool:
        """
        Call an MCP tool via HTTP using the streamable-http transport.

        NOTE: This method sends raw HTTP POST requests which may get 406 Not Acceptable
        errors from the MCP server. Use call_tool_via_cli() for proper MCP protocol.

        Args:
            session: aiohttp client session
            tool_name: Name of the tool to call
            arguments: Tool arguments
            user_id: ID of the simulated user making the call

        Returns:
            True if call was successful, False otherwise
        """
        try:
            # Construct MCP request following the protocol
            mcp_request = {
                "jsonrpc": "2.0",
                "id": f"user-{user_id}-{int(time.time() * 1000)}",
                "method": "tools/call",
                "params": {
                    "name": tool_name, "arguments": arguments
                }
            }

            headers = {
                "Content-Type": "application/json",
            }

            async with session.post(self.server_url,
                                    json=mcp_request,
                                    headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    result = await response.json()
                    if "error" in result:
                        logger.debug(f"User {user_id}: Tool {tool_name} returned error: {result['error']}")
                        return False
                    logger.debug(f"User {user_id}: Tool {tool_name} succeeded")
                    return True
                else:
                    logger.debug(f"User {user_id}: Tool {tool_name} failed with status {response.status}")
                    return False
        except TimeoutError:
            logger.debug(f"User {user_id}: Tool {tool_name} timed out")
            return False
        except Exception as e:
            logger.debug(f"User {user_id}: Tool {tool_name} error: {e}")
            return False

    def generate_text2sql_query(self) -> dict[str, Any]:
        """
        Generate a random text2sql query from the predefined list.

        Returns:
            Dictionary with question and optional analysis_type
        """
        question = random.choice(TEXT2SQL_QUERIES)

        # Randomly decide whether to include analysis_type (30% of the time)
        if random.random() < 0.3:
            analysis_type = random.choice(["pbr", "supply_gap"])
            return {"question": question, "analysis_type": analysis_type}
        else:
            return {"question": question}

    async def simulate_user(
        self,
        user_id: int,
        session: aiohttp.ClientSession,
        tools: list[str],
    ):
        """
        Simulate a single user making multiple tool calls.

        Args:
            user_id: Unique identifier for this user
            session: aiohttp client session (used only for HTTP mode)
            tools: List of available tools to call
        """
        logger.info(f"User {user_id} starting {self.calls_per_user} calls")

        # Find the text2sql tool
        text2sql_tool = None
        for tool in tools:
            if "text2sql" in tool.lower():
                text2sql_tool = tool
                break

        if not text2sql_tool:
            logger.error(f"User {user_id}: No text2sql tool found in available tools")
            return

        for call_num in range(self.calls_per_user):
            # Generate a random text2sql query
            arguments = self.generate_text2sql_query()

            # Log the query being made
            logger.info(f"User {user_id} call {call_num + 1}: {arguments.get('question', 'unknown')[:50]}...")

            # Make the tool call using selected method
            if self.use_cli:
                success = await self.call_tool_via_cli(text2sql_tool, arguments, user_id)
            else:
                success = await self.call_tool_via_http(session, text2sql_tool, arguments, user_id)

            self.total_calls += 1
            if success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1

            # Add delay between calls
            if call_num < self.calls_per_user - 1:
                await asyncio.sleep(self.delay_between_calls)

        logger.info(f"User {user_id} completed all calls")

    async def run_load_test(self):
        """Execute the load test with multiple concurrent users."""
        logger.info(f"Starting Text2SQL load test: {self.num_users} users, {self.calls_per_user} calls each")
        logger.info(f"Target MCP server: {self.server_url}")

        self.start_time = time.time()

        # Create a single session for all requests
        async with aiohttp.ClientSession() as session:
            # Get available tools
            tools = await self.get_available_tools(session)
            if not tools:
                logger.error("No tools available, cannot proceed with load test")
                return

            # Create tasks for all simulated users
            tasks = [self.simulate_user(user_id, session, tools) for user_id in range(1, self.num_users + 1)]

            # Run all user simulations concurrently
            await asyncio.gather(*tasks)

        self.end_time = time.time()
        self.print_summary()

    def print_summary(self):
        """Print a summary of the load test results."""
        if self.end_time is None or self.start_time is None:
            logger.error("Cannot print summary: timing data not available")
            return
        duration = self.end_time - self.start_time
        calls_per_second = self.total_calls / duration if duration > 0 else 0

        logger.info("=" * 70)
        logger.info("TEXT2SQL LOAD TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total users:        {self.num_users}")
        logger.info(f"Calls per user:     {self.calls_per_user}")
        logger.info(f"Total calls:        {self.total_calls}")
        logger.info(f"Successful calls:   {self.successful_calls}")
        logger.info(f"Failed calls:       {self.failed_calls}")
        logger.info(f"Success rate:       {self.successful_calls / self.total_calls * 100:.2f}%")
        logger.info(f"Duration:           {duration:.2f} seconds")
        logger.info(f"Calls per second:   {calls_per_second:.2f}")
        logger.info("=" * 70)


async def run_multiple_rounds(
    server_url: str,
    num_users: int,
    calls_per_user: int,
    num_rounds: int,
    delay_between_rounds: float,
    use_cli: bool = True,
):
    """
    Run multiple rounds of load testing to observe memory behavior over time.

    Args:
        server_url: URL of the MCP server
        num_users: Number of concurrent users per round
        calls_per_user: Number of calls per user per round
        num_rounds: Number of rounds to execute
        delay_between_rounds: Delay in seconds between rounds
        use_cli: If True, use nat mcp client CLI (proper protocol)
    """
    logger.info(f"Starting {num_rounds} rounds of Text2SQL load testing")
    logger.info(f"Method: {'CLI (proper MCP protocol)' if use_cli else 'HTTP (may get 406 errors)'}")

    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"ROUND {round_num} of {num_rounds}")
        logger.info(f"{'=' * 70}\n")

        tester = Text2SQLLoadTester(
            server_url=server_url,
            num_users=num_users,
            calls_per_user=calls_per_user,
            use_cli=use_cli,
        )

        await tester.run_load_test()

        if round_num < num_rounds:
            logger.info(f"\nWaiting {delay_between_rounds} seconds before next round...\n")
            await asyncio.sleep(delay_between_rounds)

    logger.info("\nAll rounds completed!")


def main():
    """Main entry point for the load testing script."""
    parser = argparse.ArgumentParser(description="Load test Text2SQL MCP Server to detect memory leaks")
    parser.add_argument("--url",
                        default="http://localhost:9901/mcp",
                        help="URL of the MCP server (default: http://localhost:9901/mcp)")
    parser.add_argument("--users", type=int, default=40, help="Number of concurrent users to simulate (default: 40)")
    parser.add_argument("--calls", type=int, default=10, help="Number of tool calls per user (default: 10)")
    parser.add_argument("--rounds", type=int, default=1, help="Number of load test rounds to run (default: 1)")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds between rounds (default: 5.0)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--use-cli",
                        action="store_true",
                        default=True,
                        help="Use nat mcp client CLI for proper MCP protocol (default: True)")
    parser.add_argument("--use-http", action="store_true", help="Use raw HTTP POST (gets 406 errors, for testing only)")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine which method to use
    use_cli = not args.use_http  # Default to CLI unless --use-http is specified

    # Run the load test
    asyncio.run(
        run_multiple_rounds(
            server_url=args.url,
            num_users=args.users,
            calls_per_user=args.calls,
            num_rounds=args.rounds,
            delay_between_rounds=args.delay,
            use_cli=use_cli,
        ))


if __name__ == "__main__":
    main()
