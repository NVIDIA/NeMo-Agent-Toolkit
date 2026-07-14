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
"""Unit tests for the transform_document orchestration. No network access is required."""

import json
import typing
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic import ConfigDict
from pytest_httpserver import HTTPServer

from nat_unstructured_transform_mcp.register import TransformDocumentConfig
from nat_unstructured_transform_mcp.register import TransformTools
from nat_unstructured_transform_mcp.register import resolve_tools
from nat_unstructured_transform_mcp.register import transform_document
from nat_unstructured_transform_mcp.register import transform_source

_GROUP = "unstructured_transform"


class _ToolArgs(BaseModel):
    """Permissive input-schema stand-in that accepts any tool arguments."""
    model_config = ConfigDict(extra="allow")


class _FakeTool:
    """Mimics an MCP tool exposed through a function group: JSON text in, JSON text out."""

    def __init__(self, name: str, handler: typing.Callable[..., typing.Awaitable[str]]):
        self.instance_name = f"{_GROUP}__{name}"
        self.input_schema = _ToolArgs
        self.calls: list[dict[str, typing.Any]] = []
        self._handler = handler

    async def ainvoke(self, value: _ToolArgs) -> str:
        """Record the call arguments and return the handler's canned response."""
        args = value.model_dump()
        self.calls.append(args)
        return await self._handler(**args)


class _FakeTransformService:
    """Canned Transform MCP tool responses backed by a local HTTP server for the raw byte transfers."""

    def __init__(self, httpserver: HTTPServer, statuses: list[str]):
        self.statuses = list(statuses)
        self.file_ref = "u10d://file/test-file-ref"
        self.job_id = "job-123"
        self.markdown = "# Sample Document\n\nHello **Markdown** world."
        self.upload_url = httpserver.url_for("/upload/test-file-ref")
        self.download_url = httpserver.url_for("/download/test-file-ref")

        httpserver.expect_request("/upload/test-file-ref", method="PUT").respond_with_data("")
        httpserver.expect_request("/download/test-file-ref",
                                  method="GET").respond_with_data(self.markdown, content_type="text/markdown")

        async def _request_file_upload_url(**_kwargs) -> str:
            return json.dumps({
                "upload_url": self.upload_url,
                "method": "PUT",
                "headers": {
                    "Content-Type": _kwargs.get("content_type", "application/octet-stream")
                },
                "file_ref": self.file_ref,
            })

        async def _transform_files(**_kwargs) -> str:
            return json.dumps({"job_id": self.job_id, "status": "SCHEDULED"})

        async def _check_transform_status(**_kwargs) -> str:
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return json.dumps({"job_id": self.job_id, "status": status})

        async def _get_transform_results(**_kwargs) -> str:
            return json.dumps({
                "job_id":
                    self.job_id,
                "files": [{
                    "filename": "sample.pdf",
                    "download_url": self.download_url,
                    "element_count": 12,
                    "character_count": len(self.markdown),
                }],
            })

        self.group_functions = {
            f"{_GROUP}__request_file_upload_url": _FakeTool("request_file_upload_url", _request_file_upload_url),
            f"{_GROUP}__transform_files": _FakeTool("transform_files", _transform_files),
            f"{_GROUP}__check_transform_status": _FakeTool("check_transform_status", _check_transform_status),
            f"{_GROUP}__get_transform_results": _FakeTool("get_transform_results", _get_transform_results),
        }

    @property
    def tools(self) -> TransformTools:
        """Resolve the fake group into the TransformTools the code under test expects."""
        return resolve_tools(self.group_functions)  # type: ignore[arg-type]

    def tool(self, name: str) -> _FakeTool:
        """Return the fake tool registered under the given MCP tool name."""
        return typing.cast(_FakeTool, self.group_functions[f"{_GROUP}__{name}"])


@pytest.fixture(name="fast_config")
def fast_config_fixture() -> TransformDocumentConfig:
    """Config with tiny timeouts so the polling paths run quickly in tests."""
    return TransformDocumentConfig(poll_interval_seconds=0.01, transform_timeout_seconds=5.0, http_timeout_seconds=5.0)


@pytest.fixture(name="sample_document")
def sample_document_fixture(tmp_path: Path) -> Path:
    """Write a small local file for the upload path to read."""
    document = tmp_path / "sample.pdf"
    document.write_bytes(b"%PDF-1.4 fake test document bytes")
    return document


async def test_local_file_transform(httpserver: HTTPServer, fast_config: TransformDocumentConfig,
                                    sample_document: Path):
    """A local file is uploaded and transformed, with the expected MCP call shapes and no bearer token on the PUT."""
    service = _FakeTransformService(httpserver, statuses=["SCHEDULED", "IN_PROGRESS", "COMPLETED"])

    result = await transform_source(service.tools, fast_config, str(sample_document))

    assert result.markdown == service.markdown
    assert result.element_count == 12
    assert result.character_count == len(service.markdown)
    assert result.filename == "sample.pdf"

    upload_request_args = service.tool("request_file_upload_url").calls[0]
    assert upload_request_args["filename"] == "sample.pdf"
    assert upload_request_args["content_type"] == "application/pdf"
    assert upload_request_args["size_bytes"] == sample_document.stat().st_size

    assert service.tool("transform_files").calls == [{"file_refs": [service.file_ref]}]
    assert all(call == {"job_id": service.job_id} for call in service.tool("check_transform_status").calls)
    assert service.tool("get_transform_results").calls == [{"job_id": service.job_id}]

    put_requests = [request for request, _ in httpserver.log if request.method == "PUT"]
    assert len(put_requests) == 1
    # The pre-signed upload URL must receive the raw bytes without the bearer token.
    assert "Authorization" not in put_requests[0].headers
    assert put_requests[0].headers["Content-Type"] == "application/pdf"
    assert put_requests[0].get_data() == sample_document.read_bytes()


async def test_public_url_skips_upload(httpserver: HTTPServer, fast_config: TransformDocumentConfig):
    """A public URL is passed straight to transform_files, skipping the upload step."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    url = "https://example.com/whitepaper.pdf"

    result = await transform_source(service.tools, fast_config, url)

    assert result.markdown == service.markdown
    assert service.tool("request_file_upload_url").calls == []
    assert service.tool("transform_files").calls == [{"file_refs": [url]}]


async def test_failed_job_raises(httpserver: HTTPServer, fast_config: TransformDocumentConfig, sample_document: Path):
    """A job that ends in a non-success terminal state raises an error naming the state."""
    service = _FakeTransformService(httpserver, statuses=["FAILED"])

    with pytest.raises(RuntimeError, match="unexpected state 'FAILED'"):
        await transform_source(service.tools, fast_config, str(sample_document))


async def test_poll_timeout_raises(httpserver: HTTPServer, sample_document: Path):
    """Polling stops with a timeout when the job never completes."""
    service = _FakeTransformService(httpserver, statuses=["IN_PROGRESS"])
    config = TransformDocumentConfig(poll_interval_seconds=0.01,
                                     transform_timeout_seconds=0.05,
                                     http_timeout_seconds=5.0)

    with pytest.raises(TimeoutError, match="did not complete"):
        await transform_source(service.tools, config, str(sample_document))


async def test_oversized_file_rejected(httpserver: HTTPServer, sample_document: Path):
    """A file above the size limit is rejected before any upload URL is requested."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    config = TransformDocumentConfig(poll_interval_seconds=0.01,
                                     transform_timeout_seconds=5.0,
                                     http_timeout_seconds=5.0,
                                     max_file_size_bytes=4)

    with pytest.raises(ValueError, match="exceeds"):
        await transform_source(service.tools, config, str(sample_document))

    assert service.tool("request_file_upload_url").calls == []


async def test_missing_file_rejected(httpserver: HTTPServer, fast_config: TransformDocumentConfig, tmp_path: Path):
    """A missing local path raises FileNotFoundError."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    with pytest.raises(FileNotFoundError):
        await transform_source(service.tools, fast_config, str(tmp_path / "does-not-exist.pdf"))


async def test_results_not_ready_retries_until_available(httpserver: HTTPServer,
                                                         fast_config: TransformDocumentConfig,
                                                         sample_document: Path):
    """The status endpoint can report COMPLETED slightly before the results exist."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    real_results_tool = service.tool("get_transform_results")
    not_ready_responses = 2
    attempts = []

    async def _flaky_results(**kwargs) -> str:
        if len(attempts) < not_ready_responses:
            attempts.append(kwargs)
            return json.dumps({
                "error": {
                    "code": "job_not_complete", "message": "results are available only after the job completes"
                }
            })
        return await real_results_tool._handler(**kwargs)

    service.group_functions[f"{_GROUP}__get_transform_results"] = _FakeTool("get_transform_results", _flaky_results)

    result = await transform_source(service.tools, fast_config, str(sample_document))

    assert len(attempts) == not_ready_responses
    assert result.markdown == service.markdown


async def test_tool_error_payload_raises(httpserver: HTTPServer,
                                         fast_config: TransformDocumentConfig,
                                         sample_document: Path):
    """An error envelope returned by an MCP tool is surfaced as an error."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _unauthorized(**_kwargs) -> str:
        return json.dumps({"error": {"code": "unauthorized", "message": "Invalid API key", "status": 401}})

    service.group_functions[f"{_GROUP}__transform_files"] = _FakeTool("transform_files", _unauthorized)

    with pytest.raises(RuntimeError, match="unauthorized"):
        await transform_source(service.tools, fast_config, str(sample_document))


async def test_non_json_tool_output_raises(httpserver: HTTPServer,
                                           fast_config: TransformDocumentConfig,
                                           sample_document: Path):
    """Non-JSON tool output raises a descriptive error."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _error_text(**_kwargs) -> str:
        return "MCPToolClient tool call failed: upstream error"

    service.group_functions[f"{_GROUP}__request_file_upload_url"] = _FakeTool("request_file_upload_url", _error_text)

    with pytest.raises(RuntimeError, match="Expected a JSON payload"):
        await transform_source(service.tools, fast_config, str(sample_document))


def test_resolve_tools_reports_missing(httpserver: HTTPServer):
    """resolve_tools reports which required tool is absent from the group."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    del service.group_functions[f"{_GROUP}__get_transform_results"]

    with pytest.raises(ValueError, match="get_transform_results"):
        resolve_tools(service.group_functions)  # type: ignore[arg-type]


def test_workflow_config_loads(monkeypatch: pytest.MonkeyPatch):
    """The shipped config loads and wires the expected function, group, and workflow types."""
    from nat.runtime.loader import load_config
    from nat.test.utils import locate_example_config

    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-key-0123456789")

    config = load_config(locate_example_config(TransformDocumentConfig))

    function_config = config.functions["transform_document"]
    assert isinstance(function_config, TransformDocumentConfig)
    assert str(function_config.mcp_group) == "unstructured_transform"

    group_config = config.function_groups["unstructured_transform"]
    assert group_config.server.transport == "streamable-http"
    assert str(group_config.server.url).rstrip("/") == "https://mcp.transform.unstructured.io"
    assert str(group_config.server.auth_provider) == "unstructured_auth"

    assert config.workflow.type == "react_agent"


# --- Additional error-path coverage for the orchestration core ---


async def test_results_error_not_retried(httpserver: HTTPServer,
                                         fast_config: TransformDocumentConfig,
                                         sample_document: Path):
    """A non-retryable error at results-fetch time must raise immediately, not time out."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _unauthorized(**_kwargs) -> str:
        return json.dumps({"error": {"code": "unauthorized", "message": "Invalid API key", "status": 401}})

    service.group_functions[f"{_GROUP}__get_transform_results"] = _FakeTool("get_transform_results", _unauthorized)

    with pytest.raises(RuntimeError, match="unauthorized"):
        await transform_source(service.tools, fast_config, str(sample_document))


async def test_results_fetch_deadline(httpserver: HTTPServer, sample_document: Path):
    """Results that never materialize after COMPLETED status hit the shared deadline."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _never_ready(**_kwargs) -> str:
        return json.dumps({"error": {"code": "job_not_complete", "message": "not yet"}})

    service.group_functions[f"{_GROUP}__get_transform_results"] = _FakeTool("get_transform_results", _never_ready)
    config = TransformDocumentConfig(poll_interval_seconds=0.01,
                                     transform_timeout_seconds=0.05,
                                     http_timeout_seconds=5.0)

    with pytest.raises(TimeoutError, match="did not complete"):
        await transform_source(service.tools, config, str(sample_document))


async def test_empty_files_list_raises(httpserver: HTTPServer,
                                       fast_config: TransformDocumentConfig,
                                       sample_document: Path):
    """A completed job that returns no files raises an error."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _no_files(**_kwargs) -> str:
        return json.dumps({"job_id": service.job_id, "files": []})

    service.group_functions[f"{_GROUP}__get_transform_results"] = _FakeTool("get_transform_results", _no_files)

    with pytest.raises(RuntimeError, match="returned no files"):
        await transform_source(service.tools, fast_config, str(sample_document))


async def test_missing_required_field_raises(httpserver: HTTPServer,
                                             fast_config: TransformDocumentConfig,
                                             sample_document: Path):
    """A response missing a required field names that field in the error."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _no_file_ref(**kwargs) -> str:
        return json.dumps({"upload_url": service.upload_url, "method": "PUT", "headers": {}})

    service.group_functions[f"{_GROUP}__request_file_upload_url"] = _FakeTool("request_file_upload_url", _no_file_ref)

    with pytest.raises(RuntimeError, match="'file_ref'"):
        await transform_source(service.tools, fast_config, str(sample_document))


async def test_non_dict_json_payload_raises(httpserver: HTTPServer,
                                            fast_config: TransformDocumentConfig,
                                            sample_document: Path):
    """A JSON payload that is not an object raises a descriptive error."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _json_array(**_kwargs) -> str:
        return json.dumps([1, 2, 3])

    service.group_functions[f"{_GROUP}__request_file_upload_url"] = _FakeTool("request_file_upload_url", _json_array)

    with pytest.raises(RuntimeError, match="Expected a JSON object"):
        await transform_source(service.tools, fast_config, str(sample_document))


async def test_transient_status_blip_is_retried(httpserver: HTTPServer,
                                                fast_config: TransformDocumentConfig,
                                                sample_document: Path):
    """A momentary transport failure during status polling must not abandon the job."""
    service = _FakeTransformService(httpserver, statuses=["IN_PROGRESS", "COMPLETED"])
    real_status_tool = service.tool("check_transform_status")
    blips = []

    async def _flaky_status(**kwargs) -> str:
        if not blips:
            blips.append(True)
            return "MCPToolClient tool call failed: connection reset"
        return await real_status_tool._handler(**kwargs)

    service.group_functions[f"{_GROUP}__check_transform_status"] = _FakeTool("check_transform_status", _flaky_status)

    result = await transform_source(service.tools, fast_config, str(sample_document))

    assert result.markdown == service.markdown
    assert blips == [True]


async def test_persistent_status_blips_raise(httpserver: HTTPServer,
                                             fast_config: TransformDocumentConfig,
                                             sample_document: Path):
    """Repeated transport failures during polling give up after the retry limit."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    attempts = []

    async def _always_broken(**_kwargs) -> str:
        attempts.append(True)
        return "MCPToolClient tool call failed: connection reset"

    service.group_functions[f"{_GROUP}__check_transform_status"] = _FakeTool("check_transform_status", _always_broken)

    with pytest.raises(RuntimeError, match="Expected a JSON payload"):
        await transform_source(service.tools, fast_config, str(sample_document))

    assert len(attempts) == 3


async def test_upload_http_error_sanitized(httpserver: HTTPServer,
                                           fast_config: TransformDocumentConfig,
                                           sample_document: Path):
    """A failed pre-signed transfer must not leak the URL signature into the error message."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    httpserver.clear()
    httpserver.expect_request("/upload/test-file-ref", method="PUT").respond_with_data("denied", status=403)
    service.upload_url = httpserver.url_for("/upload/test-file-ref") + "?X-Signature=secret-credential"

    async def _signed_upload(**_kwargs) -> str:
        return json.dumps({
            "upload_url": service.upload_url, "method": "PUT", "headers": {}, "file_ref": service.file_ref
        })

    service.group_functions[f"{_GROUP}__request_file_upload_url"] = _FakeTool("request_file_upload_url", _signed_upload)

    with pytest.raises(RuntimeError, match="document upload failed with HTTP 403") as exc_info:
        await transform_source(service.tools, fast_config, str(sample_document))

    assert "secret-credential" not in str(exc_info.value)
    assert "/upload/test-file-ref" in str(exc_info.value)


async def test_status_error_envelope_raises(httpserver: HTTPServer,
                                            fast_config: TransformDocumentConfig,
                                            sample_document: Path):
    """A non-retryable status error envelope is surfaced immediately."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _status_error(**_kwargs) -> str:
        return json.dumps({"error": {"code": "job_not_found", "message": "Job not found", "status": 404}})

    service.group_functions[f"{_GROUP}__check_transform_status"] = _FakeTool("check_transform_status", _status_error)

    with pytest.raises(RuntimeError, match="job_not_found"):
        await transform_source(service.tools, fast_config, str(sample_document))


async def test_transient_results_blip_is_retried(httpserver: HTTPServer,
                                                 fast_config: TransformDocumentConfig,
                                                 sample_document: Path):
    """A momentary transport failure during the results fetch is retried."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    real_results_tool = service.tool("get_transform_results")
    blips = []

    async def _flaky_results(**kwargs) -> str:
        if not blips:
            blips.append(True)
            return "MCPToolClient tool call failed: connection reset"
        return await real_results_tool._handler(**kwargs)

    service.group_functions[f"{_GROUP}__get_transform_results"] = _FakeTool("get_transform_results", _flaky_results)

    result = await transform_source(service.tools, fast_config, str(sample_document))

    assert result.markdown == service.markdown
    assert blips == [True]


async def test_non_dict_files_entry_raises(httpserver: HTTPServer,
                                           fast_config: TransformDocumentConfig,
                                           sample_document: Path):
    """A files entry that is not a mapping raises a descriptive error."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])

    async def _string_files(**_kwargs) -> str:
        return json.dumps({"job_id": service.job_id, "files": ["not-a-mapping"]})

    service.group_functions[f"{_GROUP}__get_transform_results"] = _FakeTool("get_transform_results", _string_files)

    with pytest.raises(RuntimeError, match="unexpected files entry"):
        await transform_source(service.tools, fast_config, str(sample_document))


# --- The registered function: agent-facing wrapper contract ---


class _StubGroup:
    """Minimal function group that exposes the fake tools to the registered function."""

    def __init__(self, group_functions: dict[str, _FakeTool]):
        self._group_functions = group_functions

    async def get_accessible_functions(self) -> dict[str, _FakeTool]:
        """Return the fake group's functions."""
        return self._group_functions


class _StubBuilder:
    """Minimal builder that hands the registered function its function group."""

    def __init__(self, group: _StubGroup):
        self._group = group

    async def get_function_group(self, _name) -> _StubGroup:
        """Return the stub group regardless of the requested name."""
        return self._group


@asynccontextmanager
async def _registered_transform_fn(service: _FakeTransformService, config: TransformDocumentConfig):
    """Drive the registered async generator exactly as the workflow builder would."""
    builder = _StubBuilder(_StubGroup(service.group_functions))
    async with transform_document(config, builder) as info:  # type: ignore[arg-type]
        yield info.single_fn


async def test_registered_function_happy_path(httpserver: HTTPServer,
                                              fast_config: TransformDocumentConfig,
                                              sample_document: Path):
    """The registered tool returns the transformed Markdown on success."""
    service = _FakeTransformService(httpserver, statuses=["SCHEDULED", "COMPLETED"])

    async with _registered_transform_fn(service, fast_config) as transform_fn:
        output = await transform_fn(str(sample_document))

    assert output == service.markdown


async def test_registered_function_returns_error_string(httpserver: HTTPServer,
                                                        fast_config: TransformDocumentConfig,
                                                        tmp_path: Path):
    """Failures must reach the agent as a readable tool result, never as an exception."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    missing = tmp_path / "does-not-exist.pdf"

    async with _registered_transform_fn(service, fast_config) as transform_fn:
        output = await transform_fn(str(missing))

    assert output.startswith("ERROR: failed to transform")
    assert str(missing) in output


async def test_registered_function_truncates_output(httpserver: HTTPServer, sample_document: Path):
    """Output longer than the limit is truncated and a notice is appended."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    config = TransformDocumentConfig(poll_interval_seconds=0.01,
                                     transform_timeout_seconds=5.0,
                                     http_timeout_seconds=5.0,
                                     max_output_characters=10)

    async with _registered_transform_fn(service, config) as transform_fn:
        output = await transform_fn(str(sample_document))

    assert output.startswith(service.markdown[:10])
    assert "[Output truncated at 10 characters" in output


async def test_registered_function_no_truncation_at_boundary(httpserver: HTTPServer, sample_document: Path):
    """Output exactly at the limit is returned untruncated."""
    service = _FakeTransformService(httpserver, statuses=["COMPLETED"])
    config = TransformDocumentConfig(poll_interval_seconds=0.01,
                                     transform_timeout_seconds=5.0,
                                     http_timeout_seconds=5.0,
                                     max_output_characters=len(service.markdown))

    async with _registered_transform_fn(service, config) as transform_fn:
        output = await transform_fn(str(sample_document))

    assert output == service.markdown
