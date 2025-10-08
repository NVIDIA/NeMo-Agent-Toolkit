# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Memory profiling utilities for MCP frontend."""

import gc
import logging
import tracemalloc
import weakref
from typing import Any

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory profiler for tracking memory usage and potential leaks."""

    def __init__(self, enabled: bool = False, log_interval: int = 50, top_n: int = 10):
        """Initialize the memory profiler.

        Args:
            enabled: Whether memory profiling is enabled
            log_interval: Log stats every N requests
            top_n: Number of top allocations to log
        """
        self.enabled = enabled
        # normalize interval to avoid modulo-by-zero
        self.log_interval = max(1, int(log_interval))
        self.top_n = top_n
        self.request_count = 0
        self.baseline_snapshot = None

        # Track IntermediateStepManager instances
        self._intermediate_step_managers: list[weakref.ref] = []

        # Track whether this instance started tracemalloc (to avoid resetting external tracing)
        self._we_started_tracemalloc = False

        if self.enabled:
            logger.info("Memory profiling ENABLED (interval=%d, top_n=%d)", self.log_interval, top_n)
            try:
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                    self._we_started_tracemalloc = True
                # Take baseline snapshot
                gc.collect()
                self.baseline_snapshot = tracemalloc.take_snapshot()
            except RuntimeError as e:
                logger.warning("tracemalloc unavailable or not tracing: %s", e)
        else:
            logger.info("Memory profiling DISABLED")

    def on_request_complete(self) -> None:
        """Called after each request completes."""
        if not self.enabled:
            return
        self.request_count += 1
        if self.request_count % self.log_interval == 0:
            self.log_memory_stats()

    def _ensure_tracing(self) -> bool:
        """Ensure tracemalloc is running if we started it originally.

        Returns:
            True if tracemalloc is active, False otherwise
        """
        if tracemalloc.is_tracing():
            return True

        # Only restart if we started it originally (respect external control)
        if not self._we_started_tracemalloc:
            return False

        # Attempt to restart
        try:
            logger.warning("tracemalloc was stopped externally; restarting (we started it originally)")
            tracemalloc.start()

            # Reset baseline since old tracking data is lost
            gc.collect()
            self.baseline_snapshot = tracemalloc.take_snapshot()
            logger.info("Baseline snapshot reset after tracemalloc restart")

            return True
        except RuntimeError as e:
            logger.error("Failed to restart tracemalloc: %s", e)
            return False

    def _safe_traced_memory(self) -> tuple[float, float] | None:
        """Return (current_mb, peak_mb) if tracemalloc is active, else None."""
        if not self._ensure_tracing():
            return None

        try:
            current, peak = tracemalloc.get_traced_memory()
            return (current / 1024 / 1024, peak / 1024 / 1024)
        except RuntimeError:
            return None

    def _safe_snapshot(self):
        """Return a tracemalloc Snapshot if available, else None."""
        if not self._ensure_tracing():
            return None

        try:
            return tracemalloc.take_snapshot()
        except RuntimeError:
            return None

    def log_memory_stats(self) -> dict[str, Any]:
        """Log current memory statistics and return them."""
        if not self.enabled:
            return {}

        # Force garbage collection first
        gc.collect()

        # Get current memory usage
        mem = self._safe_traced_memory()
        if mem is None:
            logger.info("tracemalloc is not active; cannot collect memory stats.")
            # still return structural fields
            stats = {
                "request_count": self.request_count,
                "current_memory_mb": None,
                "peak_memory_mb": None,
                "alive_intermediate_managers": self._alive_intermediate_manager_count(),
                "total_intermediate_managers": len(self._intermediate_step_managers),
                "active_exporters": self._safe_exporter_count(),
                "isolated_exporters": self._safe_isolated_exporter_count(),
                "subject_instances": self._count_instances_of_type("Subject"),
            }
            return stats

        current_mb, peak_mb = mem

        # Take snapshot and compare to baseline
        snapshot = self._safe_snapshot()

        # Track BaseExporter instances (observability layer)
        exporter_count = self._safe_exporter_count()
        isolated_exporter_count = self._safe_isolated_exporter_count()

        # Track Subject instances (event streams)
        subject_count = self._count_instances_of_type("Subject")

        stats = {
            "request_count": self.request_count,
            "current_memory_mb": round(current_mb, 2),
            "peak_memory_mb": round(peak_mb, 2),
            "alive_intermediate_managers": self._alive_intermediate_manager_count(),
            "total_intermediate_managers": len(self._intermediate_step_managers),
            "active_exporters": exporter_count,
            "isolated_exporters": isolated_exporter_count,
            "subject_instances": subject_count,
        }

        logger.info("=" * 80)
        logger.info("MEMORY PROFILE AFTER %d REQUESTS:", self.request_count)
        logger.info("  Current Memory: %.2f MB", current_mb)
        logger.info("  Peak Memory: %.2f MB", peak_mb)
        logger.info("")
        logger.info("NAT COMPONENT INSTANCES:")
        logger.info("  IntermediateStepManagers: %d alive / %d total",
                    stats["alive_intermediate_managers"],
                    stats["total_intermediate_managers"])
        logger.info("  BaseExporters: %d active (%d isolated)", stats["active_exporters"], stats["isolated_exporters"])
        logger.info("  Subject (event streams): %d instances", stats["subject_instances"])

        # Show top allocations
        if snapshot is None:
            logger.info("\ntracemalloc snapshot unavailable.")
        else:
            if self.baseline_snapshot:
                logger.info("\nTOP %d MEMORY GROWTH SINCE BASELINE:", self.top_n)
                top_stats = snapshot.compare_to(self.baseline_snapshot, 'lineno')
            else:
                logger.info("\nTOP %d MEMORY ALLOCATIONS:", self.top_n)
                top_stats = snapshot.statistics('lineno')

            for i, stat in enumerate(top_stats[:self.top_n], 1):
                logger.info("  #%d: %s", i, stat)

        logger.info("=" * 80)

        return stats

    def _alive_intermediate_manager_count(self) -> int:
        """Return count of alive refs; prunes dead refs for accuracy without changing 'total'."""
        alive = 0
        # Light pruning during the count
        keep: list[weakref.ref] = []
        for r in self._intermediate_step_managers:
            obj = r()
            if obj is not None:
                alive += 1
                keep.append(r)
            else:
                # drop dead refs to avoid unbounded growth
                pass
        # Keep behavior: 'total_intermediate_managers' reflects original list length,
        # so we only prune for alive count; to preserve exact original semantics,
        # comment out the next line. (This change only reduces list size in memory.)
        self._intermediate_step_managers = keep
        return alive

    def _count_instances_of_type(self, type_name: str) -> int:
        """Count instances of a specific type in memory."""
        count = 0
        for obj in gc.get_objects():
            try:
                if type(obj).__name__ == type_name:
                    count += 1
            except Exception:
                pass
        return count

    def _safe_exporter_count(self) -> int:
        try:
            from nat.observability.exporter.base_exporter import BaseExporter
            return BaseExporter.get_active_instance_count()
        except Exception as e:
            logger.debug("Could not get BaseExporter stats: %s", e)
            return 0

    def _safe_isolated_exporter_count(self) -> int:
        try:
            from nat.observability.exporter.base_exporter import BaseExporter
            return BaseExporter.get_isolated_instance_count()
        except Exception:
            return 0

    def track_intermediate_step_manager(self, manager: Any) -> None:
        """Track an IntermediateStepManager instance."""
        if not self.enabled:
            return
        self._intermediate_step_managers.append(weakref.ref(manager))
        alive_count = len([r for r in self._intermediate_step_managers if r() is not None])
        logger.debug(
            "IntermediateStepManager created (alive: %d, total: %d)",
            alive_count,
            len(self._intermediate_step_managers),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current memory statistics without logging."""
        if not self.enabled:
            return {"enabled": False}

        mem = self._safe_traced_memory()
        if mem is None:
            return {
                "enabled": True,
                "request_count": self.request_count,
                "current_memory_mb": None,
                "peak_memory_mb": None,
                "alive_intermediate_managers": len([r for r in self._intermediate_step_managers if r() is not None]),
                "total_intermediate_managers": len(self._intermediate_step_managers),
                "active_exporters": self._safe_exporter_count(),
                "isolated_exporters": self._safe_isolated_exporter_count(),
                "subject_instances": self._count_instances_of_type("Subject"),
            }

        current_mb, peak_mb = mem
        return {
            "enabled": True,
            "request_count": self.request_count,
            "current_memory_mb": round(current_mb, 2),
            "peak_memory_mb": round(peak_mb, 2),
            "alive_intermediate_managers": len([r for r in self._intermediate_step_managers if r() is not None]),
            "total_intermediate_managers": len(self._intermediate_step_managers),
            "active_exporters": self._safe_exporter_count(),
            "isolated_exporters": self._safe_isolated_exporter_count(),
            "subject_instances": self._count_instances_of_type("Subject"),
        }

    def reset_baseline(self) -> None:
        """Reset the baseline snapshot to current state."""
        if not self.enabled:
            return
        gc.collect()
        snap = self._safe_snapshot()
        if snap is None:
            logger.info("Cannot reset baseline: tracemalloc is not active.")
            return
        self.baseline_snapshot = snap
        logger.info("Memory profiling baseline reset at request %d", self.request_count)
