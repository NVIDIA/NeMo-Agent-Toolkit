#!/usr/bin/env python3
"""
Load testing script specifically for text2sql MCP server.

This script simulates realistic supply chain text-to-SQL queries to test
for memory leaks and performance issues in the text2sql_standalone function.
"""

import argparse
import asyncio
import json
import logging
import random
import time
from typing import Any

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample supply chain questions for realistic load testing
SAMPLE_QUESTIONS = [
    # Supplier queries
    "Show me the top 10 suppliers by total revenue",
    "List all suppliers located in Asia",
    "Find suppliers with more than 100 parts",
    "Which suppliers have the highest on-time delivery rate?",
    "Show suppliers with pending orders",

  # Parts queries
    "What are the most expensive parts in inventory?",
    "List parts with inventory below 50 units",
    "Show all electronic components",
    "Find parts with no recent orders",
    "Which parts have the highest demand?",

  # Order queries
    "Show pending orders from last month",
    "List orders with delivery delays",
    "What is the total order value by customer?",
    "Find orders with status 'shipped'",
    "Show orders exceeding $10,000",

  # Inventory queries
    "List all parts with low stock levels",
    "Show inventory turnover by category",
    "Find parts that need reordering",
    "What is the total inventory value?",
    "Show parts in warehouse location A1",

  # Demand queries
    "What is the demand forecast for Q1 2024?",
    "Show parts with increasing demand trend",
    "List critical shortage items",
    "Find parts with demand spike",
    "Show seasonal demand patterns",

  # Analytics queries
    "Calculate average lead time by supplier",
    "Show revenue trend over last 6 months",
    "Find top 5 customers by order volume",
    "What is the average order size?",
    "Show supplier performance metrics",
]

# Analysis types for filtering few-shot examples
ANALYSIS_TYPES = ["pbr", "supply_gap", None]  # None for no filtering


class Text2SQLLoadTester:
    """Load tester specifically for text2sql MCP server."""

    def __init__(
        self,
        server_url: str = "http://localhost:9901/mcp",
        num_users: int = 20,
        calls_per_user: int = 10,
        delay_between_calls: float = 0.5,
        use_cli: bool = True,
    ):
        """
        Initialize the text2sql load tester.

        Args:
            server_url: URL of the MCP server
            num_users: Number of concurrent users to simulate
            calls_per_user: Number of text2sql calls each user should make
            delay_between_calls: Delay in seconds between consecutive calls per user
            use_cli: If True, use nat mcp client CLI (proper protocol)
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
        self.response_times = []

    async def verify_server(self, session: aiohttp.ClientSession) -> bool:
        """
        Verify the MCP server is running and has text2sql_standalone tool.

        Args:
            session: aiohttp client session

        Returns:
            True if server is ready, False otherwise
        """
        try:
            # Check health endpoint
            health_url = self.server_url.replace('/mcp', '/health')
            async with session.get(health_url) as response:
                if response.status != 200:
                    logger.error(f"Health check failed: {response.status}")
                    return False

            # Check for text2sql_standalone tool
            tools_url = self.server_url.replace('/mcp', '/debug/tools/list')
            async with session.get(tools_url) as response:
                if response.status == 200:
                    tools = await response.json()
                    # Extract tool names
                    if isinstance(tools, dict) and 'tools' in tools:
                        tools = tools['tools']

                    tool_names = []
                    for tool in tools:
                        if isinstance(tool, dict) and 'name' in tool:
                            tool_names.append(tool['name'])
                        elif isinstance(tool, str):
                            tool_names.append(tool)

                    if 'text2sql_standalone' in tool_names:
                        logger.info("✅ Server verified: text2sql_standalone tool is available")
                        return True
                    else:
                        logger.error(f"text2sql_standalone not found. Available tools: {tool_names}")
                        return False
                else:
                    logger.error(f"Failed to get tools list: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Server verification failed: {e}")
            return False

    async def call_text2sql_via_cli(
        self,
        question: str,
        analysis_type: str | None,
        user_id: int,
    ) -> tuple[bool, float]:
        """
        Call text2sql using nat mcp client CLI (proper MCP protocol).

        Args:
            question: Natural language question
            analysis_type: Optional analysis type filter
            user_id: ID of the simulated user

        Returns:
            Tuple of (success: bool, response_time: float)
        """
        start_time = time.time()

        try:
            # Build arguments
            arguments = {"question": question}
            if analysis_type:
                arguments["analysis_type"] = analysis_type

            # Build the CLI command
            cmd = [
                "nat",
                "mcp",
                "client",
                "tool",
                "call",
                "text2sql_standalone",
                "--url",
                self.server_url,
                "--json-args",
                json.dumps(arguments)
            ]

            # Run the command
            process = await asyncio.create_subprocess_exec(*cmd,
                                                           stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.PIPE)

            stdout, stderr = await process.communicate()
            returncode = process.returncode

            elapsed = time.time() - start_time

            if returncode == 0:
                logger.debug(f"User {user_id}: text2sql succeeded in {elapsed:.2f}s "
                             f"(question: {question[:50]}...)")
                return True, elapsed
            else:
                logger.debug(f"User {user_id}: text2sql failed with code {returncode} "
                             f"(stderr: {stderr.decode()[:100]})")
                return False, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            logger.debug(f"User {user_id}: text2sql error: {e}")
            return False, elapsed

    async def call_text2sql_via_http(
        self,
        session: aiohttp.ClientSession,
        question: str,
        analysis_type: str | None,
        user_id: int,
    ) -> tuple[bool, float]:
        """
        Call text2sql via HTTP (for testing only, may get 406 errors).

        Args:
            session: aiohttp client session
            question: Natural language question
            analysis_type: Optional analysis type filter
            user_id: ID of the simulated user

        Returns:
            Tuple of (success: bool, response_time: float)
        """
        start_time = time.time()

        try:
            # Build arguments
            arguments = {"question": question}
            if analysis_type:
                arguments["analysis_type"] = analysis_type

            # Construct MCP request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": f"user-{user_id}-{int(time.time() * 1000)}",
                "method": "tools/call",
                "params": {
                    "name": "text2sql_standalone", "arguments": arguments
                }
            }

            headers = {"Content-Type": "application/json"}

            async with session.post(self.server_url,
                                    json=mcp_request,
                                    headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=60)) as response:
                elapsed = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    if "error" in result:
                        logger.debug(f"User {user_id}: text2sql returned error: {result['error']}")
                        return False, elapsed
                    logger.debug(f"User {user_id}: text2sql succeeded in {elapsed:.2f}s")
                    return True, elapsed
                else:
                    logger.debug(f"User {user_id}: text2sql failed with status {response.status}")
                    return False, elapsed

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.debug(f"User {user_id}: text2sql timed out after {elapsed:.2f}s")
            return False, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            logger.debug(f"User {user_id}: text2sql error: {e}")
            return False, elapsed

    async def simulate_user(
        self,
        user_id: int,
        session: aiohttp.ClientSession,
    ):
        """
        Simulate a single user making multiple text2sql calls.

        Args:
            user_id: Unique identifier for this user
            session: aiohttp client session (used only for HTTP mode)
        """
        logger.info(f"User {user_id} starting {self.calls_per_user} text2sql calls")

        for call_num in range(self.calls_per_user):
            # Randomly select a question and analysis type
            question = random.choice(SAMPLE_QUESTIONS)
            analysis_type = random.choice(ANALYSIS_TYPES)

            # Make the text2sql call
            if self.use_cli:
                success, response_time = await self.call_text2sql_via_cli(question, analysis_type, user_id)
            else:
                success, response_time = await self.call_text2sql_via_http(session, question, analysis_type, user_id)

            # Record metrics
            self.total_calls += 1
            self.response_times.append(response_time)
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
        logger.info("=" * 70)
        logger.info("TEXT2SQL LOAD TEST")
        logger.info("=" * 70)
        logger.info(f"Target MCP server: {self.server_url}")
        logger.info(f"Concurrent users: {self.num_users}")
        logger.info(f"Calls per user: {self.calls_per_user}")
        logger.info(f"Total calls: {self.num_users * self.calls_per_user}")
        logger.info(f"Method: {'CLI (proper MCP)' if self.use_cli else 'HTTP'}")
        logger.info("=" * 70)

        self.start_time = time.time()

        # Create a single session for all requests
        async with aiohttp.ClientSession() as session:
            # Verify server is ready
            if not await self.verify_server(session):
                logger.error("Server verification failed, aborting test")
                return

            # Create tasks for all simulated users
            tasks = [self.simulate_user(user_id, session) for user_id in range(1, self.num_users + 1)]

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

        # Calculate response time statistics
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        min_response_time = min(self.response_times) if self.response_times else 0
        max_response_time = max(self.response_times) if self.response_times else 0

        logger.info("=" * 70)
        logger.info("TEXT2SQL LOAD TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total users:          {self.num_users}")
        logger.info(f"Calls per user:       {self.calls_per_user}")
        logger.info(f"Total calls:          {self.total_calls}")
        logger.info(f"Successful calls:     {self.successful_calls}")
        logger.info(f"Failed calls:         {self.failed_calls}")
        logger.info(f"Success rate:         {self.successful_calls / self.total_calls * 100:.2f}%")
        logger.info(f"Duration:             {duration:.2f} seconds")
        logger.info(f"Throughput:           {calls_per_second:.2f} calls/sec")
        logger.info(f"Avg response time:    {avg_response_time:.2f} seconds")
        logger.info(f"Min response time:    {min_response_time:.2f} seconds")
        logger.info(f"Max response time:    {max_response_time:.2f} seconds")
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
        use_cli: If True, use nat mcp client CLI
    """
    logger.info("🚀 Starting text2sql MCP server load testing")
    logger.info(f"Total rounds: {num_rounds}")
    logger.info(f"Method: {'CLI (proper MCP protocol)' if use_cli else 'HTTP (may get 406 errors)'}")
    logger.info("")

    round_summaries = []

    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"🔄 ROUND {round_num} of {num_rounds}")
        logger.info(f"{'=' * 70}\n")

        tester = Text2SQLLoadTester(
            server_url=server_url,
            num_users=num_users,
            calls_per_user=calls_per_user,
            use_cli=use_cli,
        )

        await tester.run_load_test()

        # Store round summary for trend analysis
        round_summaries.append({
            "round":
                round_num,
            "total_calls":
                tester.total_calls,
            "successful":
                tester.successful_calls,
            "failed":
                tester.failed_calls,
            "duration":
                tester.end_time - tester.start_time if tester.end_time and tester.start_time else 0,
            "avg_response_time":
                sum(tester.response_times) / len(tester.response_times) if tester.response_times else 0,
        })

        if round_num < num_rounds:
            logger.info(f"\n⏳ Waiting {delay_between_rounds} seconds before next round...\n")
            await asyncio.sleep(delay_between_rounds)

    # Print overall summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 OVERALL TEST SUMMARY")
    logger.info("=" * 70)

    for summary in round_summaries:
        logger.info(f"Round {summary['round']}: "
                    f"{summary['successful']}/{summary['total_calls']} succeeded, "
                    f"avg response time: {summary['avg_response_time']:.2f}s")

    # Check for performance degradation
    if len(round_summaries) >= 2:
        first_round_time = round_summaries[0]['avg_response_time']
        last_round_time = round_summaries[-1]['avg_response_time']

        if last_round_time > first_round_time * 1.2:  # 20% slower
            logger.warning(f"⚠️  Performance degradation detected: "
                           f"Round 1 avg: {first_round_time:.2f}s, "
                           f"Round {num_rounds} avg: {last_round_time:.2f}s "
                           f"({(last_round_time/first_round_time - 1) * 100:.1f}% slower)")
        else:
            logger.info(f"✅ Performance stable across rounds "
                        f"(Round 1: {first_round_time:.2f}s, Round {num_rounds}: {last_round_time:.2f}s)")

    logger.info("=" * 70)
    logger.info("\n✨ All rounds completed!")


def main():
    """Main entry point for the text2sql load testing script."""
    parser = argparse.ArgumentParser(description="Load test text2sql MCP server to detect memory leaks")
    parser.add_argument("--url",
                        default="http://localhost:9901/mcp",
                        help="URL of the MCP server (default: http://localhost:9901/mcp)")
    parser.add_argument("--users", type=int, default=20, help="Number of concurrent users to simulate (default: 20)")
    parser.add_argument("--calls", type=int, default=10, help="Number of text2sql calls per user (default: 10)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of load test rounds to run (default: 3)")
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
