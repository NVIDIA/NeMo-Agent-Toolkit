#!/usr/bin/env python3
"""
Integrated test runner for Text2SQL MCP server memory leak testing.

This script coordinates:
1. Starting a Text2SQL MCP server
2. Monitoring its memory usage
3. Running load tests with realistic text2sql queries
4. Analyzing results for memory leaks
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Text2SQLMemoryLeakTest:
    """Coordinates Text2SQL MCP server testing and memory monitoring."""

    def __init__(
        self,
        config_file: str,
        server_host: str = "localhost",
        server_port: int = 9901,
        num_users: int = 40,
        calls_per_user: int = 10,
        num_rounds: int = 3,
        delay_between_rounds: float = 10.0,
        output_dir: str = "test_results",
    ):
        """
        Initialize the test runner.

        Args:
            config_file: Path to NAT workflow config file
            server_host: MCP server host
            server_port: MCP server port
            num_users: Number of concurrent users to simulate
            calls_per_user: Number of calls per user
            num_rounds: Number of load test rounds
            delay_between_rounds: Delay between rounds in seconds
            output_dir: Directory for test results
        """
        self.config_file = config_file
        self.server_host = server_host
        self.server_port = server_port
        self.num_users = num_users
        self.calls_per_user = calls_per_user
        self.num_rounds = num_rounds
        self.delay_between_rounds = delay_between_rounds
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.server_process = None
        self.monitor_process = None
        self.server_pid = None
        self.monitor_log_file = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output files
        self.memory_csv = self.output_dir / f"text2sql_memory_{self.timestamp}.csv"
        self.load_test_log = self.output_dir / f"text2sql_load_test_{self.timestamp}.log"
        self.server_log = self.output_dir / f"text2sql_server_{self.timestamp}.log"
        self.monitor_log = self.output_dir / f"text2sql_monitor_{self.timestamp}.log"

    def start_mcp_server(self) -> bool:
        """
        Start the MCP server process.

        Returns:
            True if server started successfully
        """
        logger.info("Starting Text2SQL MCP server...")

        # Build command
        cmd = [
            "nat",
            "mcp",
            "serve",
            "--config_file",
            self.config_file,
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Start server with output redirected to log file
            server_log_file = open(self.server_log, 'w')
            self.server_process = subprocess.Popen(
                cmd,
                stdout=server_log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if sys.platform != 'win32' else None,
            )

            # Wait for server to start
            logger.info("Waiting for server to start...")
            time.sleep(8)  # Text2SQL needs more time to initialize Vanna

            # Check if server is running
            if self.server_process.poll() is not None:
                logger.error("Server process terminated unexpectedly")
                return False

            self.server_pid = self.server_process.pid
            logger.info(f"Text2SQL MCP server started with PID: {self.server_pid}")

            # Wait a bit more to ensure it's fully initialized
            time.sleep(5)

            # Verify server is responding
            if not self.verify_server_health():
                logger.error("Server health check failed")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False

    def verify_server_health(self) -> bool:
        """
        Verify that the MCP server is healthy and responding.

        Returns:
            True if server is healthy
        """
        try:
            import requests
            url = f"http://{self.server_host}:{self.server_port}/health"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Server health check passed: {data}")
                return True
            else:
                logger.error(f"Server health check failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Server health check error: {e}")
            return False

    def start_memory_monitor(self) -> bool:
        """
        Start the memory monitoring process.

        Returns:
            True if monitor started successfully
        """
        logger.info("Starting memory monitor...")

        # Build command - use the monitor script from the same directory
        monitor_script = Path(__file__).parent / "monitor_memory.py"

        # Check if monitor script exists
        if not monitor_script.exists():
            logger.error(f"Monitor script not found at: {monitor_script}")
            logger.error("Please ensure monitor_memory.py exists in examples/text2sql/")
            return False

        cmd = [
            sys.executable,
            str(monitor_script),
            "--pid",
            str(self.server_pid),
            "--interval",
            "1.0",
            "--output",
            str(self.memory_csv),
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Create a log file for monitor output
            self.monitor_log_file = open(self.monitor_log, 'w')

            self.monitor_process = subprocess.Popen(
                cmd,
                stdout=self.monitor_log_file,
                stderr=subprocess.STDOUT,
            )

            logger.info(f"Memory monitor started with PID: {self.monitor_process.pid}")
            logger.info(f"Monitor log: {self.monitor_log}")
            time.sleep(2)

            # Check if monitor process is still running
            if self.monitor_process.poll() is not None:
                logger.error("Monitor process terminated unexpectedly")
                # Close the log file and read its contents
                self.monitor_log_file.close()
                if self.monitor_log.exists():
                    with open(self.monitor_log) as f:
                        error_output = f.read()
                        if error_output:
                            logger.error(f"Monitor output:\n{error_output}")
                return False

            # Give it a bit more time and verify CSV is being created
            time.sleep(3)
            if not self.memory_csv.exists():
                logger.warning(f"Memory CSV not yet created at: {self.memory_csv}")
                logger.warning("Monitor may be having issues, but continuing...")
            else:
                logger.info(f"Memory CSV created successfully: {self.memory_csv}")

            return True

        except Exception as e:
            logger.error(f"Failed to start memory monitor: {e}")
            if self.monitor_log_file:
                try:
                    self.monitor_log_file.close()
                except:
                    pass
            return False

    async def run_load_tests(self):
        """Run the load testing rounds."""
        logger.info("Starting Text2SQL load tests...")

        # Build command - use the load test script we just created
        load_test_script = Path(__file__).parent / "load_test_text2sql.py"

        cmd = [
            sys.executable,
            str(load_test_script),
            "--url",
            f"http://{self.server_host}:{self.server_port}/mcp",
            "--users",
            str(self.num_users),
            "--calls",
            str(self.calls_per_user),
            "--rounds",
            str(self.num_rounds),
            "--delay",
            str(self.delay_between_rounds),
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Run load test with output to log file
            with open(self.load_test_log, 'w') as log_file:  # noqa: UP015
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

                await process.wait()

                if process.returncode == 0:
                    logger.info("Load tests completed successfully")
                else:
                    logger.error(f"Load tests failed with return code {process.returncode}")

        except Exception as e:
            logger.error(f"Error running load tests: {e}")

    def stop_processes(self):
        """Stop all running processes."""
        logger.info("Stopping processes...")

        # Stop memory monitor
        if self.monitor_process:
            try:
                logger.info("Stopping memory monitor...")
                self.monitor_process.send_signal(signal.SIGINT)
                self.monitor_process.wait(timeout=10)
                logger.info("Memory monitor stopped")
            except Exception as e:
                logger.error(f"Error stopping memory monitor: {e}")
                try:
                    self.monitor_process.kill()
                except:
                    pass
            finally:
                # Close monitor log file
                if self.monitor_log_file:
                    try:
                        self.monitor_log_file.close()
                    except:
                        pass

        # Stop MCP server
        if self.server_process:
            try:
                logger.info("Stopping MCP server (allowing cleanup to complete)...")
                if sys.platform != 'win32':
                    # Kill the process group with SIGTERM (allows graceful shutdown)
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                else:
                    self.server_process.terminate()

                # Give server extra time to clean up connections
                self.server_process.wait(timeout=15)
                logger.info("MCP server stopped (cleanup completed)")
            except Exception as e:
                logger.error(f"Error stopping MCP server: {e}")
                try:
                    if sys.platform != 'win32':
                        os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                    else:
                        self.server_process.kill()
                except:
                    pass

    def analyze_results(self):
        """Analyze the test results and print summary."""
        logger.info("\n" + "=" * 70)
        logger.info("TEXT2SQL MEMORY LEAK TEST RESULTS")
        logger.info("=" * 70)

        # Check if memory CSV exists and analyze
        if self.memory_csv.exists():
            try:
                import csv  # noqa: PLC0415
                with open(self.memory_csv) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                    if rows:
                        initial_memory = float(rows[0]['rss_total_mb'])
                        final_memory = float(rows[-1]['rss_total_mb'])
                        max_memory = max(float(row['rss_total_mb']) for row in rows)

                        logger.info("\nMemory Analysis:")
                        logger.info(f"  Initial memory:    {initial_memory:.2f} MB")
                        logger.info(f"  Final memory:      {final_memory:.2f} MB")
                        logger.info(f"  Peak memory:       {max_memory:.2f} MB")
                        logger.info(f"  Memory growth:     {final_memory - initial_memory:.2f} MB")
                        logger.info(
                            f"  Growth percentage: {((final_memory - initial_memory) / initial_memory) * 100:.2f}%")

                        # Check for potential memory leak
                        if final_memory > initial_memory * 1.5:
                            logger.warning("⚠️  POTENTIAL MEMORY LEAK DETECTED!")
                            logger.warning("   Memory increased by >50% during test")
                        elif final_memory > initial_memory * 1.2:
                            logger.warning("⚠️ Significant memory growth detected (>20%)")
                        else:
                            logger.info("✓ Memory growth appears normal (<20%)")

            except Exception as e:
                logger.error(f"Error analyzing memory data: {e}")

        logger.info("\nOutput files:")
        logger.info(f"  Memory data:       {self.memory_csv}")
        logger.info(f"  Load test log:     {self.load_test_log}")
        logger.info(f"  Server log:        {self.server_log}")
        logger.info(f"  Monitor log:       {self.monitor_log}")
        logger.info("=" * 70 + "\n")

    async def run(self):
        """Execute the complete test suite."""
        logger.info("=" * 70)
        logger.info("TEXT2SQL MCP SERVER MEMORY LEAK TEST")
        logger.info("=" * 70)
        logger.info(f"Config file:           {self.config_file}")
        logger.info(f"Server:                {self.server_host}:{self.server_port}")
        logger.info(f"Simulated users:       {self.num_users}")
        logger.info(f"Calls per user:        {self.calls_per_user}")
        logger.info(f"Test rounds:           {self.num_rounds}")
        logger.info(f"Output directory:      {self.output_dir}")
        logger.info("=" * 70 + "\n")

        try:
            # Start MCP server
            if not self.start_mcp_server():
                logger.error("Failed to start MCP server")
                return

            # Start memory monitor
            if not self.start_memory_monitor():
                logger.error("Failed to start memory monitor")
                self.stop_processes()
                return

            # Wait a bit to get baseline memory
            logger.info("Collecting baseline memory data...")
            await asyncio.sleep(10)

            # Run load tests
            await self.run_load_tests()

            # Wait a bit after load tests to see if memory is released
            logger.info("Waiting for post-test memory observation...")
            await asyncio.sleep(30)

        except KeyboardInterrupt:
            logger.info("\nTest interrupted by user")
        except Exception as e:
            logger.error(f"Test error: {e}")
        finally:
            # Stop all processes
            self.stop_processes()

            # Analyze results
            self.analyze_results()


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run integrated Text2SQL MCP server memory leak tests")
    parser.add_argument("--config_file",
                        default="src/text2sql/configs/config_text2sql_mcp.yml",
                        help="Path to NAT workflow config file (default: src/text2sql/configs/config_text2sql_mcp.yml)")
    parser.add_argument("--host", default="localhost", help="MCP server host (default: localhost)")
    parser.add_argument("--port", type=int, default=9901, help="MCP server port (default: 9901)")
    parser.add_argument("--users", type=int, default=40, help="Number of concurrent users to simulate (default: 40)")
    parser.add_argument("--calls", type=int, default=10, help="Number of calls per user (default: 10)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of load test rounds (default: 3)")
    parser.add_argument("--delay", type=float, default=10.0, help="Delay between rounds in seconds (default: 10.0)")
    parser.add_argument("--output_dir",
                        default="test_results",
                        help="Output directory for results (default: test_results)")

    args = parser.parse_args()

    # Check if required packages are available
    try:
        import psutil  # noqa: F401, PLC0415
    except ImportError:
        logger.error("psutil is required for memory monitoring. Install with: pip install psutil")
        sys.exit(1)

    try:
        import requests  # noqa: F401, PLC0415
    except ImportError:
        logger.error("requests is required. Install with: pip install requests")
        sys.exit(1)

    try:
        import aiohttp  # noqa: F401, PLC0415
    except ImportError:
        logger.error("aiohttp is required. Install with: pip install aiohttp")
        sys.exit(1)

    # Run the test
    test = Text2SQLMemoryLeakTest(
        config_file=args.config_file,
        server_host=args.host,
        server_port=args.port,
        num_users=args.users,
        calls_per_user=args.calls,
        num_rounds=args.rounds,
        delay_between_rounds=args.delay,
        output_dir=args.output_dir,
    )

    asyncio.run(test.run())


if __name__ == "__main__":
    main()
