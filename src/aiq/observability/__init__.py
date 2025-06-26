# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

logger = logging.getLogger(__name__)


def log_exporter_stats():
    """Log statistics about active BaseExporter instances for debugging memory leaks."""
    try:
        from aiq.observability.exporter.base_exporter import BaseExporter
        BaseExporter.log_instance_stats()
    except ImportError as e:
        logger.warning("Could not import BaseExporter for stats logging: %s", e)
    except Exception as e:
        logger.error("Error logging exporter stats: %s", e)


def get_exporter_counts() -> dict[str, int]:
    """Get counts of active BaseExporter instances.

    Returns:
        dict[str, int]: Dictionary with 'total', 'isolated', and 'original' counts
    """
    try:
        from aiq.observability.exporter.base_exporter import BaseExporter

        total = BaseExporter.get_active_instance_count()
        isolated = BaseExporter.get_isolated_instance_count()
        original = total - isolated

        return {'total': total, 'isolated': isolated, 'original': original}
    except ImportError:
        return {'total': -1, 'isolated': -1, 'original': -1}
    except Exception as e:
        logger.error("Error getting exporter counts: %s", e)
        return {'total': -1, 'isolated': -1, 'original': -1}
