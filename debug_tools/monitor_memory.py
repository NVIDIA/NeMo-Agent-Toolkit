#!/usr/bin/env python3
"""
Memory monitoring script for NAT MCP Server processes.

This script monitors the memory usage of MCP server processes over time,
logging data to help identify memory leaks.
"""

import argparse
import csv
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage of a process over time."""

    def __init__(
        self,
        pid: Optional[int] = None,
        process_name: str = "uvicorn",
        interval: float = 1.0,
        output_file: Optional[str] = None,
    ):
        """
        Initialize the memory monitor.

        Args:
            pid: Process ID to monitor (if None, will search by process_name)
            process_name: Name of process to monitor if pid is not provided
            interval: Sampling interval in seconds
            output_file: Path to CSV file for logging memory data
        """
        self.pid = pid
        self.process_name = process_name
        self.interval = interval
        self.output_file = output_file
        self.process = None
        self.csv_writer = None
        self.csv_file = None
        self.running = True
        self.samples = []

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, stopping monitoring...")
        self.running = False

    def find_process(self) -> Optional[psutil.Process]:
        """
        Find the target process to monitor.

        Returns:
            psutil.Process object if found, None otherwise
        """
        if self.pid:
            try:
                process = psutil.Process(self.pid)
                logger.info(f"Found process with PID {self.pid}: {process.name()}")
                return process
            except psutil.NoSuchProcess:
                logger.error(f"No process found with PID {self.pid}")
                return None
        else:
            # Search for process by name
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check process name
                    if self.process_name.lower() in proc.info['name'].lower():
                        # Verify it's an MCP server by checking command line
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any('mcp' in arg.lower() for arg in cmdline):
                            logger.info(f"Found MCP server process: PID={proc.info['pid']}, Name={proc.info['name']}")
                            logger.info(f"Command line: {' '.join(cmdline)}")
                            return proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            logger.error(f"No process found matching '{self.process_name}' with MCP in command line")
            return None

    def get_memory_info(self, process: psutil.Process) -> dict:
        """
        Get detailed memory information for a process.

        Args:
            process: psutil.Process object

        Returns:
            Dictionary containing memory metrics
        """
        try:
            mem_info = process.memory_info()
            mem_percent = process.memory_percent()

            # Get child processes memory too
            children = process.children(recursive=True)
            total_rss = mem_info.rss
            total_vms = mem_info.vms

            for child in children:
                try:
                    child_mem = child.memory_info()
                    total_rss += child_mem.rss
                    total_vms += child_mem.vms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                'timestamp': datetime.now().isoformat(),
                'pid': process.pid,
                'rss_mb': mem_info.rss / (1024 * 1024),  # RSS in MB
                'vms_mb': mem_info.vms / (1024 * 1024),  # VMS in MB
                'rss_total_mb': total_rss / (1024 * 1024),  # Including children
                'vms_total_mb': total_vms / (1024 * 1024),  # Including children
                'percent': mem_percent,
                'num_children': len(children),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Error getting memory info: {e}")
            return None

    def initialize_csv(self):
        """Initialize CSV file for logging memory data."""
        if not self.output_file:
            return

        self.csv_file = open(self.output_file, 'w', newline='')
        fieldnames = [
            'timestamp',
            'pid',
            'rss_mb',
            'vms_mb',
            'rss_total_mb',
            'vms_total_mb',
            'percent',
            'num_children',
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
        logger.info(f"Logging memory data to: {self.output_file}")

    def log_memory_sample(self, mem_info: dict):
        """
        Log a memory sample to CSV and console.

        Args:
            mem_info: Dictionary containing memory metrics
        """
        if self.csv_writer:
            self.csv_writer.writerow(mem_info)
            self.csv_file.flush()

        self.samples.append(mem_info)

        # Log to console
        logger.info(f"Memory: RSS={mem_info['rss_mb']:.2f}MB, "
                    f"RSS+Children={mem_info['rss_total_mb']:.2f}MB, "
                    f"Percent={mem_info['percent']:.2f}%, "
                    f"Children={mem_info['num_children']}")

    def print_summary(self):
        """Print summary statistics of the monitoring session."""
        if not self.samples:
            logger.warning("No samples collected")
            return

        rss_values = [s['rss_mb'] for s in self.samples]
        rss_total_values = [s['rss_total_mb'] for s in self.samples]

        logger.info("\n" + "=" * 70)
        logger.info("MEMORY MONITORING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Process PID:           {self.samples[0]['pid']}")
        logger.info(f"Total samples:         {len(self.samples)}")
        logger.info(f"Duration:              {len(self.samples) * self.interval:.2f} seconds")
        logger.info("\nProcess Memory (RSS):")
        logger.info(f"  Initial:             {rss_values[0]:.2f} MB")
        logger.info(f"  Final:               {rss_values[-1]:.2f} MB")
        logger.info(f"  Min:                 {min(rss_values):.2f} MB")
        logger.info(f"  Max:                 {max(rss_values):.2f} MB")
        logger.info(f"  Avg:                 {sum(rss_values) / len(rss_values):.2f} MB")
        logger.info(f"  Growth:              {rss_values[-1] - rss_values[0]:.2f} MB")
        logger.info("\nTotal Memory (Process + Children):")
        logger.info(f"  Initial:             {rss_total_values[0]:.2f} MB")
        logger.info(f"  Final:               {rss_total_values[-1]:.2f} MB")
        logger.info(f"  Min:                 {min(rss_total_values):.2f} MB")
        logger.info(f"  Max:                 {max(rss_total_values):.2f} MB")
        logger.info(f"  Avg:                 {sum(rss_total_values) / len(rss_total_values):.2f} MB")
        logger.info(f"  Growth:              {rss_total_values[-1] - rss_total_values[0]:.2f} MB")

        if self.output_file:
            logger.info(f"\nDetailed data saved to: {self.output_file}")

        logger.info("=" * 70 + "\n")

    def monitor(self):
        """Main monitoring loop."""
        # Find the process
        self.process = self.find_process()
        if not self.process:
            logger.error("Failed to find target process")
            return

        # Initialize CSV logging
        self.initialize_csv()

        logger.info(f"Starting memory monitoring (interval: {self.interval}s)")
        logger.info("Press Ctrl+C to stop monitoring and see summary")

        try:
            while self.running:
                mem_info = self.get_memory_info(self.process)

                if mem_info is None:
                    logger.error("Process no longer exists or accessible")
                    break

                self.log_memory_sample(mem_info)
                time.sleep(self.interval)

        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            if self.csv_file:
                self.csv_file.close()

            self.print_summary()


def main():
    """Main entry point for the memory monitoring script."""
    parser = argparse.ArgumentParser(description="Monitor memory usage of NAT MCP Server process")
    parser.add_argument("--pid", type=int, help="Process ID to monitor")
    parser.add_argument("--name",
                        default="uvicorn",
                        help="Process name to search for if PID not provided (default: uvicorn)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--output", help="Output CSV file for memory data (default: auto-generated with timestamp)")

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"memory_monitor_{timestamp}.csv"

    monitor = MemoryMonitor(
        pid=args.pid,
        process_name=args.name,
        interval=args.interval,
        output_file=args.output,
    )

    monitor.monitor()


if __name__ == "__main__":
    main()
