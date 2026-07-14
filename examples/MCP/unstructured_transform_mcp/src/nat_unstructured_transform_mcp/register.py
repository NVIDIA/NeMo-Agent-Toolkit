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
"""Register the ``transform_document`` function.

The Unstructured Transform MCP server exposes an asynchronous, multi-step protocol
(request an upload URL, upload the raw bytes over plain HTTP, start a transform job,
poll for completion, download the result). This module composes those MCP tools into a
single deterministic function so the agent only needs one reliable tool call.
"""

import asyncio
import json
import logging
import mimetypes
import typing
from pathlib import Path

import httpx
from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# Maximum file size accepted by the Unstructured Transform service (50 MB).
MAX_FILE_SIZE_BYTES = 52_428_800

# Job states reported by the check_transform_status MCP tool which mean "keep waiting".
_PENDING_JOB_STATES = frozenset({"SCHEDULED", "IN_PROGRESS"})
_COMPLETED_JOB_STATE = "COMPLETED"

# The MCP client converts transport-level failures into plain-text tool output; the
# idempotent polling calls tolerate this many consecutive such failures before giving up.
_MAX_CONSECUTIVE_POLL_FAILURES = 3


class TransformDocumentConfig(FunctionBaseConfig, name="transform_document"):
    """Configuration for the ``transform_document`` function."""

    mcp_group: FunctionGroupRef = Field(
        default=FunctionGroupRef("unstructured_transform"),
        description="Reference to the mcp_client function group connected to the Unstructured Transform MCP server.")
    poll_interval_seconds: float = Field(default=5.0, gt=0, description="Delay between transform job status checks.")
    transform_timeout_seconds: float = Field(
        default=900.0,
        gt=0,
        description="Maximum time to wait for a transform job to complete. Large documents take longer.")
    http_timeout_seconds: float = Field(
        default=120.0, gt=0, description="Timeout for the plain HTTP file upload and result download requests.")
    max_file_size_bytes: int = Field(
        default=MAX_FILE_SIZE_BYTES,
        gt=0,
        description="Maximum document size accepted by the Transform service (50 MB at the time of writing).")
    max_output_characters: int = Field(
        default=50_000,
        gt=0,
        description="Truncate the returned Markdown body beyond this length to protect the context window of the "
        "agent. A short truncation notice is appended to truncated output.")


class TransformResult(BaseModel):
    """Outcome of a single document transform."""

    markdown: str
    element_count: int
    character_count: int
    filename: str


class TransformTools(typing.NamedTuple):
    """The four Unstructured Transform MCP tools, resolved from the function group."""

    request_file_upload_url: Function
    transform_files: Function
    check_transform_status: Function
    get_transform_results: Function


def resolve_tools(group_functions: dict[str, Function]) -> TransformTools:
    """Look up the Transform MCP tools in a function group by their MCP tool names.

    Function groups expose members as ``<group_instance>__<tool_name>``; the tool name is
    recovered with the framework ``FunctionGroup.decompose`` helper.

    Args:
        group_functions: Accessible functions of the mcp_client function group.

    Returns:
        The resolved Transform MCP tools.

    Raises:
        ValueError: If a required tool is not present in the function group.
    """
    by_tool_name = {FunctionGroup.decompose(full_name)[1]: fn for full_name, fn in group_functions.items()}

    missing = [name for name in TransformTools._fields if name not in by_tool_name]
    if missing:
        raise ValueError(f"Required MCP tools {missing} were not found in the function group. "
                         f"Available tools: {sorted(by_tool_name)}")

    return TransformTools(**{name: by_tool_name[name] for name in TransformTools._fields})


class MCPToolError(RuntimeError):
    """Raised when a Transform MCP tool returns an error payload."""

    def __init__(self, tool_name: str, code: str, message: str):
        self.code = code
        super().__init__(f"The '{tool_name}' MCP tool returned an error ({code}): {message}")


async def _invoke_tool(tool: Function, **tool_args: typing.Any) -> dict[str, typing.Any]:
    """Invoke an MCP tool and parse its JSON text content into a dictionary."""
    raw = await tool.ainvoke(tool.input_schema(**tool_args))
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(f"Expected a JSON payload from the '{tool.instance_name}' MCP tool but received: "
                           f"{str(raw)[:300]!r}") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from the '{tool.instance_name}' MCP tool but received: "
                           f"{str(raw)[:300]!r}")
    error = payload.get("error")
    if isinstance(error, dict):
        raise MCPToolError(tool.instance_name, str(error.get("code", "unknown")), str(error.get("message", error)))
    return payload


def _require(payload: dict[str, typing.Any], key: str, tool_name: str) -> typing.Any:
    """Return a required key from an MCP tool payload, or raise a descriptive error."""
    try:
        return payload[key]
    except KeyError as e:
        raise RuntimeError(f"The '{tool_name}' MCP tool response is missing the required '{key}' field: "
                           f"{str(payload)[:300]!r}") from e


def _check_http_response(response: httpx.Response, description: str) -> None:
    """Raise a descriptive error for a failed HTTP transfer without leaking the pre-signed URL.

    The query string of a pre-signed URL carries its signature, which is a credential. This
    error message reaches logs and the agent, so it includes only the scheme, host, and path.
    """
    if response.status_code >= 400:
        safe_url = response.request.url.copy_with(query=None, fragment=None)
        raise RuntimeError(f"The {description} failed with HTTP {response.status_code} from {safe_url}")


async def _upload_source(tools: TransformTools, http_client: httpx.AsyncClient, source: str,
                         max_file_size_bytes: int) -> str:
    """Return a file reference for the document, uploading local files first.

    Public HTTP or HTTPS URLs are passed through unchanged because the transform_files
    MCP tool accepts them directly. Local files are uploaded to a pre-signed URL minted
    by the request_file_upload_url MCP tool.
    """
    if source.startswith(("http://", "https://")):
        return source

    path = Path(source).expanduser()
    # Filesystem access is offloaded to a worker thread so the reads (up to the size limit)
    # do not block the event loop, per the async I/O guideline.
    if not await asyncio.to_thread(path.is_file):
        raise FileNotFoundError(f"Document not found: {source}")

    size_bytes = (await asyncio.to_thread(path.stat)).st_size
    if size_bytes > max_file_size_bytes:
        raise ValueError(f"Document is {size_bytes} bytes which exceeds the "
                         f"{max_file_size_bytes} byte limit of the Transform service.")

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    upload = await _invoke_tool(tools.request_file_upload_url,
                                filename=path.name,
                                content_type=content_type,
                                size_bytes=size_bytes)

    # Upload the raw bytes to the pre-signed URL. This is a plain HTTP request, not an
    # MCP call, and the pre-signed URL must not receive the bearer token.
    file_bytes = await asyncio.to_thread(path.read_bytes)
    response = await http_client.request(str(upload.get("method", "PUT")),
                                         _require(upload, "upload_url", "request_file_upload_url"),
                                         content=file_bytes,
                                         headers=upload.get("headers") or {})
    _check_http_response(response, "document upload")

    return _require(upload, "file_ref", "request_file_upload_url")


def _timeout_error(job_id: str, timeout_seconds: float) -> TimeoutError:
    return TimeoutError(f"Transform job {job_id} did not complete within {timeout_seconds} seconds. "
                        "Large documents can take several minutes; consider raising transform_timeout_seconds.")


async def _wait_for_job(tools: TransformTools,
                        job_id: str,
                        poll_interval_seconds: float,
                        deadline: float,
                        timeout_seconds: float) -> None:
    """Poll the transform job status until it completes, fails, or times out.

    Transport-level blips (which the MCP client surfaces as plain-text tool output rather
    than exceptions) are retried a few times so a momentary hiccup does not abandon a
    long-running server-side job.
    """
    loop = asyncio.get_running_loop()
    consecutive_failures = 0

    while True:
        try:
            status_payload = await _invoke_tool(tools.check_transform_status, job_id=job_id)
        except MCPToolError:
            raise
        except RuntimeError:
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_POLL_FAILURES or loop.time() >= deadline:
                raise
            logger.warning("Status check for transform job %s failed (%d consecutive); retrying",
                           job_id,
                           consecutive_failures)
            await asyncio.sleep(poll_interval_seconds)
            continue
        consecutive_failures = 0

        status = str(status_payload.get("status", "")).upper()

        if status == _COMPLETED_JOB_STATE:
            return
        if status not in _PENDING_JOB_STATES:
            raise RuntimeError(f"Transform job {job_id} ended in unexpected state '{status}': "
                               f"{str(status_payload)[:300]!r}")
        if loop.time() >= deadline:
            raise _timeout_error(job_id, timeout_seconds)

        logger.debug("Transform job %s is %s; polling again in %.1f seconds", job_id, status, poll_interval_seconds)
        await asyncio.sleep(poll_interval_seconds)


async def _fetch_results(tools: TransformTools,
                         job_id: str,
                         poll_interval_seconds: float,
                         deadline: float,
                         timeout_seconds: float) -> dict[str, typing.Any]:
    """Fetch the transform results, tolerating the brief window after the job reports completion.

    The status endpoint can report ``COMPLETED`` slightly before the results are
    materialized, in which case get_transform_results returns a ``job_not_complete``
    error. Retry within the overall transform deadline. Transport-level blips are retried
    the same way as in the status loop.
    """
    loop = asyncio.get_running_loop()
    consecutive_failures = 0

    while True:
        try:
            return await _invoke_tool(tools.get_transform_results, job_id=job_id)
        except MCPToolError as e:
            if e.code != "job_not_complete":
                raise
            # The server reports job_not_complete as a JSON error envelope inside a
            # successful (isError=false) tool result; this is the observed behavior of the
            # brief status-vs-results consistency window and is safe to retry.
            if loop.time() >= deadline:
                raise _timeout_error(job_id, timeout_seconds) from e
            consecutive_failures = 0
        except RuntimeError:
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_POLL_FAILURES or loop.time() >= deadline:
                raise
            logger.warning("Results fetch for transform job %s failed (%d consecutive); retrying",
                           job_id,
                           consecutive_failures)

        logger.debug("Transform job %s results are not ready yet; polling again in %.1f seconds",
                     job_id,
                     poll_interval_seconds)
        await asyncio.sleep(poll_interval_seconds)


async def transform_source(tools: TransformTools, config: TransformDocumentConfig, source: str) -> TransformResult:
    """Run the full upload, transform, poll, and download flow for one document.

    Args:
        tools: The resolved Transform MCP tools.
        config: The function configuration.
        source: Local file path or public HTTP or HTTPS URL of the document.

    Returns:
        The transformed document as Markdown plus transform metadata.
    """
    async with httpx.AsyncClient(timeout=config.http_timeout_seconds) as http_client:
        file_ref = await _upload_source(tools, http_client, source, config.max_file_size_bytes)

        job = await _invoke_tool(tools.transform_files, file_refs=[file_ref])
        job_id = _require(job, "job_id", "transform_files")
        logger.info("Started transform job %s for '%s'", job_id, source)

        # The status wait and the results fetch share one overall deadline.
        deadline = asyncio.get_running_loop().time() + config.transform_timeout_seconds
        await _wait_for_job(tools, job_id, config.poll_interval_seconds, deadline, config.transform_timeout_seconds)

        results = await _fetch_results(tools,
                                       job_id,
                                       config.poll_interval_seconds,
                                       deadline,
                                       config.transform_timeout_seconds)
        files = results.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"Transform job {job_id} completed but returned no files.")
        first_file = files[0]
        if not isinstance(first_file, dict):
            raise RuntimeError(f"Transform job {job_id} returned an unexpected files entry: "
                               f"{str(first_file)[:300]!r}")

        # The download URL is pre-signed as well, so no bearer token is sent here either.
        download = await http_client.get(_require(first_file, "download_url", "get_transform_results"))
        _check_http_response(download, "result download")

        return TransformResult(markdown=download.text,
                               element_count=int(first_file.get("element_count") or 0),
                               character_count=int(first_file.get("character_count") or len(download.text)),
                               filename=str(first_file.get("filename") or Path(source).name))


@register_function(config_type=TransformDocumentConfig)
async def transform_document(config: TransformDocumentConfig, builder: Builder):
    """Register the ``transform_document`` function.

    Resolves the four Unstructured Transform MCP tools from the configured mcp_client
    function group and exposes the whole flow to the agent as one tool.
    """
    group = await builder.get_function_group(config.mcp_group)
    tools = resolve_tools(await group.get_accessible_functions())

    async def _transform(source: str) -> str:
        """Transform a document into Markdown.

        Args:
            source: Local file path or public HTTP or HTTPS URL of the document.

        Returns:
            The document content converted to Markdown, or an error description.
        """
        try:
            result = await transform_source(tools, config, source)
        except (OSError, ValueError, RuntimeError, TimeoutError, httpx.HTTPError, httpx.InvalidURL) as e:
            logger.exception("Failed to transform '%s'", source)
            return f"ERROR: failed to transform '{source}': {e}"

        logger.info("Transformed '%s': %d elements, %d characters",
                    result.filename,
                    result.element_count,
                    result.character_count)

        markdown = result.markdown
        if len(markdown) > config.max_output_characters:
            markdown = (f"{markdown[:config.max_output_characters]}\n\n"
                        f"[Output truncated at {config.max_output_characters} characters. The full document has "
                        f"{result.character_count} characters.]")
        return markdown

    yield FunctionInfo.from_fn(
        _transform,
        description=("Convert a document (PDF, DOCX, PPTX, XLSX, HTML, images, and 40+ other formats) into "
                     "clean Markdown using the Unstructured Transform service. The input must be either the path to "
                     "a local file or a public HTTP or HTTPS URL of the document. Returns the extracted content as "
                     "Markdown. Transformation is asynchronous on the server and can take from a few seconds up to "
                     "several minutes depending on document size."))
