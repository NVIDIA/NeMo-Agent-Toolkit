# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock
from unittest.mock import patch

from nat.plugins.data_flywheel.observability.register import DFWElasticsearchTelemetryExporter
from nat.plugins.data_flywheel.observability.register import dfw_elasticsearch_telemetry_exporter


async def test_dfw_elasticsearch_exporter_passes_plain_password():
    """The Elasticsearch client needs the password as a plain string, not a SecretStr."""
    config = DFWElasticsearchTelemetryExporter(client_id="client",
                                               index="index",
                                               endpoint="http://localhost:9200",
                                               username="elastic",
                                               password="changeme")

    with patch("nat.plugins.data_flywheel.observability.exporter.dfw_elasticsearch_exporter.DFWElasticsearchExporter"
               ) as mock_exporter:
        async with dfw_elasticsearch_telemetry_exporter(config, MagicMock()):
            pass

    assert mock_exporter.call_args.kwargs["elasticsearch_auth"] == ("elastic", "changeme")
