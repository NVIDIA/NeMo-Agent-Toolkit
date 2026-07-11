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
"""Integration tests against the live Unstructured Transform MCP server.

These tests require the `UNSTRUCTURED_API_KEY` environment variable (get a key from
https://transform.unstructured.io/get-started) and are skipped otherwise. The full
workflow test additionally requires `NVIDIA_API_KEY` for the NIM LLM.

Run with:
    pytest --run_integration --run_slow examples/MCP/unstructured_transform_mcp/tests
"""

from pathlib import Path

import pytest

from nat.test.plugin import require_env_variables

_MAGIC_SENTENCE = "The magic word is xylophone."


@pytest.fixture(name="unstructured_api_key")
def unstructured_api_key_fixture(fail_missing: bool):
    """Skip the test unless an Unstructured Transform API key is available."""
    yield require_env_variables(
        varnames=["UNSTRUCTURED_API_KEY"],
        reason="Unstructured integration tests require the `UNSTRUCTURED_API_KEY` environment variable",
        fail_missing=fail_missing)


def _build_minimal_pdf(text: str) -> bytes:
    """Build a tiny single-page PDF containing the given text, with no extra dependencies."""
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(pdf)


@pytest.fixture(name="sample_pdf")
def sample_pdf_fixture(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "magic_word.pdf"
    pdf_path.write_bytes(_build_minimal_pdf(_MAGIC_SENTENCE))
    return pdf_path


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(700)
@pytest.mark.usefixtures("unstructured_api_key")
async def test_transform_document_function_live(sample_pdf: Path):
    """Exercise the transform_document function against the live server, without an LLM.

    The authentication provider and MCP client group are taken from the shipped workflow
    configuration so this test cannot drift from the example (server URL included);
    loading the configuration also interpolates ``UNSTRUCTURED_API_KEY`` from the
    environment exactly as `nat run` would.
    """
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.loader import load_config
    from nat.test.utils import locate_example_config
    from nat_unstructured_transform_mcp.register import TransformDocumentConfig
    from nat_unstructured_transform_mcp.register import resolve_tools
    from nat_unstructured_transform_mcp.register import transform_source

    shipped_config = load_config(locate_example_config(TransformDocumentConfig))

    async with WorkflowBuilder() as builder:
        await builder.add_auth_provider("unstructured_auth", shipped_config.authentication["unstructured_auth"])
        group = await builder.add_function_group("unstructured_transform",
                                                 shipped_config.function_groups["unstructured_transform"])

        tools = resolve_tools(await group.get_accessible_functions())
        config = TransformDocumentConfig(poll_interval_seconds=3.0, transform_timeout_seconds=600.0)

        result = await transform_source(tools, config, str(sample_pdf))

    assert result.markdown.strip(), "Expected non-empty Markdown output"
    assert result.element_count > 0, "Expected a non-zero element count"
    assert "xylophone" in result.markdown.lower()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(1000)
@pytest.mark.usefixtures("nvidia_api_key", "unstructured_api_key")
async def test_full_workflow_live(sample_pdf: Path):
    """Run the complete agent workflow end-to-end against the live server."""
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_unstructured_transform_mcp.register import TransformDocumentConfig

    config_file = locate_example_config(TransformDocumentConfig)

    await run_workflow(config_file=config_file,
                       question=(f"Use the transform_document tool to convert the document at {sample_pdf} "
                                 "to Markdown, then answer: what is the magic word in the document?"),
                       expected_answer="xylophone")
