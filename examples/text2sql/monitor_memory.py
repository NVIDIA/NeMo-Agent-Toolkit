#!/usr/bin/env python3
"""
Memory monitoring script for tracking process memory usage over time.

This script monitors a process and its children, recording memory usage
statistics to a CSV file at regular intervals.
"""

import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required. Install with: pip install psutil", file=sys.stderr)
    sys.exit(1)


class MemoryMonitor:
    """Monitor memory usage of a process and its children."""

    def __init__(self, pid: int, interval: float, output_file: str):
        """
        Initialize the memory monitor.

        Args:
            pid: Process ID to monitor
            interval: Sampling interval in seconds
            output_file: Output CSV file path
        """
        self.pid = pid
        self.interval = interval
        self.output_file = Path(output_file)
        self.running = True
        self.csv_file = None
        self.csv_writer = None

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}, shutting down gracefully...", file=sys.stderr)
        self.running = False

    def get_process_memory(self, proc: psutil.Process) -> dict:
        """
        Get memory statistics for a process and its children.

        Args:
            proc: psutil Process object

        Returns:
            Dictionary with memory statistics
        """
        try:
            # Get main process memory info
            mem_info = proc.memory_info()
            cpu_percent = proc.cpu_percent(interval=0.1)

            # Get children processes
            children = proc.children(recursive=True)
            num_children = len(children)

            # Calculate total memory including children
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
                'pid': self.pid,
                'rss_mb': mem_info.rss / (1024 * 1024),
                'vms_mb': mem_info.vms / (1024 * 1024),
                'rss_total_mb': total_rss / (1024 * 1024),
                'vms_total_mb': total_vms / (1024 * 1024),
                'cpu_percent': cpu_percent,
                'num_children': num_children,
            }

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            raise RuntimeError(f"Cannot access process {self.pid}: {e}")

    def monitor(self):
        """Start monitoring the process."""
        try:
            # Get the process
            proc = psutil.Process(self.pid)
            print(f"Monitoring process {self.pid} ({proc.name()})", file=sys.stderr)
            print(f"Writing to: {self.output_file}", file=sys.stderr)
            print(f"Sampling interval: {self.interval}s", file=sys.stderr)

            # Create output directory if needed
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Open CSV file and write header
            self.csv_file = open(self.output_file, 'w', newline='')
            fieldnames = [
                'timestamp', 'pid', 'rss_mb', 'vms_mb', 'rss_total_mb', 'vms_total_mb', 'cpu_percent', 'num_children'
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()

            print("Monitoring started. Press Ctrl+C to stop.", file=sys.stderr)

            # Monitor loop
            sample_count = 0
            while self.running:
                try:
                    # Check if process still exists
                    if not proc.is_running():
                        print(f"Process {self.pid} is no longer running.", file=sys.stderr)
                        break

                    # Get and write memory stats
                    stats = self.get_process_memory(proc)
                    self.csv_writer.writerow(stats)
                    self.csv_file.flush()

                    sample_count += 1
                    if sample_count % 60 == 0:  # Print status every 60 samples
                        print(f"Samples collected: {sample_count}, "
                              f"RSS: {stats['rss_total_mb']:.2f} MB",
                              file=sys.stderr)

                    # Wait for next interval
                    time.sleep(self.interval)

                except RuntimeError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    break
                except Exception as e:
                    print(f"Unexpected error: {e}", file=sys.stderr)
                    time.sleep(self.interval)
                    continue

            print(f"\nMonitoring stopped. Total samples: {sample_count}", file=sys.stderr)

        except psutil.NoSuchProcess:
            print(f"ERROR: Process {self.pid} does not exist", file=sys.stderr)
            sys.exit(1)
        except psutil.AccessDenied:
            print(f"ERROR: Access denied to process {self.pid}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            if self.csv_file:
                self.csv_file.close()
                print(f"Output written to: {self.output_file}", file=sys.stderr)


def main():
    """Main entry point for the memory monitor."""
    parser = argparse.ArgumentParser(description="Monitor process memory usage")
    parser.add_argument("--pid", type=int, required=True, help="Process ID to monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--output", required=True, help="Output CSV file path")

    args = parser.parse_args()

    # Validate arguments
    if args.interval <= 0:
        print("ERROR: Interval must be positive", file=sys.stderr)
        sys.exit(1)

    if args.pid <= 0:
        print("ERROR: PID must be positive", file=sys.stderr)
        sys.exit(1)

    # Create and run monitor
    monitor = MemoryMonitor(args.pid, args.interval, args.output)
    monitor.monitor()


if __name__ == "__main__":
    main()
