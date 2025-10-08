#!/usr/bin/env python3
"""
Simple memory monitor for text2sql MCP server.

Monitors memory usage of a process by PID and logs it periodically.
"""

import argparse
import logging
import sys
import time

try:
    import psutil
except ImportError:
    print("Error: psutil is required. Install it with: pip install psutil")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
)
logger = logging.getLogger(__name__)


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def monitor_process(pid: int, interval: float = 2.0, duration: float | None = None):
    """
    Monitor memory usage of a process.

    Args:
        pid: Process ID to monitor
        interval: Sampling interval in seconds
        duration: Total monitoring duration in seconds (None for infinite)
    """
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.error(f"Process with PID {pid} not found")
        return

    logger.info(f"Monitoring process: {process.name()} (PID: {pid})")
    logger.info(f"Sampling interval: {interval}s")
    if duration:
        logger.info(f"Duration: {duration}s")
    logger.info("-" * 70)

    start_time = time.time()
    sample_count = 0

    initial_memory = None
    peak_memory = 0

    try:
        while True:
            # Check if we should stop
            if duration and (time.time() - start_time) > duration:
                break

            # Check if process still exists
            if not process.is_running():
                logger.warning("Process has terminated")
                break

            # Get memory info
            mem_info = process.memory_info()
            rss = mem_info.rss  # Resident Set Size (physical memory)
            vms = mem_info.vms  # Virtual Memory Size

            # Get CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)

            # Track initial and peak memory
            if initial_memory is None:
                initial_memory = rss
            peak_memory = max(peak_memory, rss)

            # Calculate memory increase
            mem_increase = rss - initial_memory
            mem_increase_percent = (mem_increase / initial_memory * 100) if initial_memory > 0 else 0

            # Log the metrics
            logger.info(f"Sample {sample_count:4d} | "
                        f"RSS: {format_bytes(rss):>10s} | "
                        f"VMS: {format_bytes(vms):>10s} | "
                        f"CPU: {cpu_percent:5.1f}% | "
                        f"Increase: {format_bytes(mem_increase):>10s} ({mem_increase_percent:+.1f}%)")

            sample_count += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("\nMonitoring interrupted by user")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")

    # Print summary
    logger.info("-" * 70)
    logger.info("MONITORING SUMMARY")
    logger.info("-" * 70)
    logger.info(f"Samples collected:  {sample_count}")
    logger.info(f"Duration:           {time.time() - start_time:.2f}s")
    logger.info(f"Initial memory:     {format_bytes(initial_memory) if initial_memory else 'N/A'}")
    logger.info(f"Peak memory:        {format_bytes(peak_memory)}")

    if initial_memory:
        total_increase = peak_memory - initial_memory
        increase_percent = (total_increase / initial_memory * 100)
        logger.info(f"Total increase:     {format_bytes(total_increase)} ({increase_percent:+.1f}%)")

        # Check for potential memory leak
        if increase_percent > 50:
            logger.warning("⚠️  Significant memory increase detected (>50%)!")
            logger.warning("This may indicate a memory leak.")
        elif increase_percent > 20:
            logger.warning("⚠️  Moderate memory increase detected (>20%)")
        else:
            logger.info("✓ Memory usage appears stable")

    logger.info("-" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor memory usage of a process")
    parser.add_argument("--pid", type=int, required=True, help="Process ID to monitor")
    parser.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds (default: 2.0)")
    parser.add_argument("--duration",
                        type=float,
                        default=None,
                        help="Total monitoring duration in seconds (default: infinite)")

    args = parser.parse_args()

    monitor_process(
        pid=args.pid,
        interval=args.interval,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
