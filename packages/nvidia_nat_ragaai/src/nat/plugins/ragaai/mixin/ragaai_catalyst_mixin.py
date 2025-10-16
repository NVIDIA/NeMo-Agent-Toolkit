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

import asyncio
import json
import logging
import os
from dataclasses import asdict

import ragaai_catalyst
from ragaai_catalyst.tracers.agentic_tracing.utils.trace_utils import format_interactions
from ragaai_catalyst.tracers.agentic_tracing.utils.zip_list_of_unique_files import zip_list_of_unique_files
from ragaai_catalyst.tracers.exporters import DynamicTraceExporter
from ragaai_catalyst.tracers.exporters.ragaai_trace_exporter import RAGATraceExporter
from ragaai_catalyst.tracers.exporters.ragaai_trace_exporter import TracerJSONEncoder
from ragaai_catalyst.tracers.utils.trace_json_converter import convert_json_format

from nat.plugins.opentelemetry.otel_span import OtelSpan

logger = logging.getLogger(__name__)


class RAGATraceExporterOptWrite(RAGATraceExporter):
    """Custom RAGATraceExporter that provides optional local file writing.

    This subclass of RAGATraceExporter allows control over whether the
    rag_agent_traces.json file is written to the current directory.

    Args:
        debug_mode: When False (default), creates local rag_agent_traces.json file.
                   When True, skips local file creation for cleaner operation.
    """

    def __init__(self, *args, debug_mode: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_mode = debug_mode

    def prepare_trace(self, spans, trace_id):
        try:
            try:
                ragaai_trace = convert_json_format(spans,
                                                   self.custom_model_cost,
                                                   self.user_context,
                                                   self.user_gt,
                                                   self.external_id)
            except Exception as e:
                print(f"Error in convert_json_format function: {trace_id}: {e}")
                return None

            try:
                interactions = format_interactions(ragaai_trace)
                if interactions and 'workflow' in interactions:
                    ragaai_trace["workflow"] = interactions['workflow']
            except Exception as e:
                print(f"Error in format_interactions function: {trace_id}: {e}")
                return None

            try:
                # Add source code hash
                files_to_zip = self.files_to_zip or []
                hash_id, zip_path = zip_list_of_unique_files(files_to_zip, output_dir=self.tmp_dir)
            except Exception as e:
                print(f"Error in zip_list_of_unique_files function: {trace_id}: {e}")
                return None

            try:
                ragaai_trace["metadata"]["system_info"] = asdict(self.system_monitor.get_system_info())
                ragaai_trace["metadata"]["resources"] = asdict(self.system_monitor.get_resources())
            except Exception as e:
                print(f"Error in get_system_info or get_resources function: {trace_id}: {e}")
                return None

            try:
                ragaai_trace["metadata"]["system_info"]["source_code"] = hash_id
            except Exception as e:
                print(f"Error in adding source code hash: {trace_id}: {e}")
                return None

            try:
                if "data" in ragaai_trace and ragaai_trace["data"] and len(ragaai_trace["data"]) > 0:
                    if "start_time" in ragaai_trace:
                        ragaai_trace["data"][0]["start_time"] = ragaai_trace["start_time"]
                    if "end_time" in ragaai_trace:
                        ragaai_trace["data"][0]["end_time"] = ragaai_trace["end_time"]
            except Exception as e:
                print(f"Error in adding start_time or end_time: {trace_id}: {e}")
                return None

            try:
                if hasattr(self, 'project_name'):
                    ragaai_trace["project_name"] = self.project_name
            except Exception as e:
                print(f"Error in adding project name: {trace_id}: {e}")
                return None

            try:
                # Add tracer type to the trace
                if hasattr(self, 'tracer_type'):
                    ragaai_trace["tracer_type"] = self.tracer_type
            except Exception as e:
                print(f"Error in adding tracer type: {trace_id}: {e}")
                return None

            # Add user passed metadata to the trace
            try:
                logger.debug("Started adding user passed metadata")

                metadata = (self.user_details.get("trace_user_detail", {}).get("metadata", {})
                            if self.user_details else {})

                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if key not in {"log_source", "recorded_on"}:
                            ragaai_trace.setdefault("metadata", {})[key] = value

                logger.debug("Completed adding user passed metadata")
            except Exception as e:
                print(f"Error in adding metadata: {trace_id}: {e}")
                return None

            try:
                # Save the trace_json
                trace_file_path = os.path.join(self.tmp_dir, f"{trace_id}.json")
                with open(trace_file_path, "w", encoding="utf-8") as file:
                    json.dump(ragaai_trace, file, cls=TracerJSONEncoder, indent=2)

                if self.debug_mode:
                    with open(os.path.join(os.getcwd(), 'rag_agent_traces.json'), 'w', encoding="utf-8") as f:
                        json.dump(ragaai_trace, f, cls=TracerJSONEncoder, indent=2)
            except Exception as e:
                print(f"Error in saving trace json: {trace_id}: {e}")
                return None

            return {'trace_file_path': trace_file_path, 'code_zip_path': zip_path, 'hash_id': hash_id}
        except Exception as e:
            print(f"Error converting trace {trace_id}: {str(e)}")
            return None


class DynamicTraceExporterOptWrite(DynamicTraceExporter):
    """Custom DynamicTraceExporter that uses RAGATraceExporterOptWrite internally.

    This subclass of DynamicTraceExporter creates a RAGATraceExporterOptWrite
    instance instead of the default RAGATraceExporter, providing control over
    local file creation.

    Args:
        debug_mode: When False (default), creates local rag_agent_traces.json file.
                   When True, skips local file creation for cleaner operation.
    """

    def __init__(self, *args, debug_mode: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._exporter = RAGATraceExporterOptWrite(*args, debug_mode=debug_mode, **kwargs)


class RagaAICatalystMixin:
    """Mixin for RagaAI Catalyst exporters.

    This mixin provides RagaAI Catalyst-specific functionality for OpenTelemetry span exporters.
    It handles RagaAI Catalyst project and dataset configuration and uses custom subclassed
    exporters to control local file creation behavior.

    Key Features:
    - RagaAI Catalyst authentication with access key and secret key
    - Project and dataset scoping for trace organization
    - Integration with custom DynamicTraceExporter for telemetry transmission
    - Automatic initialization of RagaAI Catalyst client
    - Configurable local file creation via debug_mode parameter

    This mixin uses subclassed exporters (RAGATraceExporterOptWrite and DynamicTraceExporterOptWrite)
    to provide clean control over whether the rag_agent_traces.json file is created locally.

    This mixin is designed to be used with OtelSpanExporter as a base class:

    Example::

        class MyCatalystExporter(OtelSpanExporter, RagaAICatalystMixin):
            def __init__(self, base_url, access_key, secret_key, project, dataset, **kwargs):
                super().__init__(base_url=base_url, access_key=access_key,
                                 secret_key=secret_key, project=project, dataset=dataset, **kwargs)
    """

    def __init__(self,
                 *args,
                 base_url: str,
                 access_key: str,
                 secret_key: str,
                 project: str,
                 dataset: str,
                 tracer_type: str,
                 debug_mode: bool = False,
                 **kwargs):
        """Initialize the RagaAI Catalyst exporter.

        Args:
            base_url: RagaAI Catalyst base URL.
            access_key: RagaAI Catalyst access key.
            secret_key: RagaAI Catalyst secret key.
            project: RagaAI Catalyst project name.
            dataset: RagaAI Catalyst dataset name.
            tracer_type: RagaAI Catalyst tracer type.
            debug_mode: When False (default), creates local rag_agent_traces.json file. When True, skips local file
            creation for cleaner operation.
            kwargs: Additional keyword arguments passed to parent classes.
        """
        logger.info("RagaAICatalystMixin initialized with debug_mode=%s", debug_mode)

        ragaai_catalyst.RagaAICatalyst(access_key=access_key, secret_key=secret_key, base_url=base_url)

        # Create the DynamicTraceExporter (this will trigger our hook)
        self._exporter = DynamicTraceExporterOptWrite(project, dataset, base_url, tracer_type, debug_mode=debug_mode)

        super().__init__(*args, **kwargs)

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OtelSpans using the custom RagaAI Catalyst exporter.

        This method uses the DynamicTraceExporterOptWrite instance to export spans,
        with local file creation controlled by the debug_mode setting.

        Args:
            spans (list[OtelSpan]): The list of spans to export.

        Raises:
            Exception: If there's an error during span export (logged but not re-raised).
        """
        try:
            # Run the blocking export operation in a thread pool to make it non-blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._exporter.export(spans))  # type: ignore[arg-type]
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
