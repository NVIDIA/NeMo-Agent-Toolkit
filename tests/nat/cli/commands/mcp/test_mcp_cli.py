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

import asyncio
import json
from types import ModuleType
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
from click.testing import CliRunner

from nat.cli.commands.mcp.mcp import MCPPingResult
from nat.cli.commands.mcp.mcp import call_tool_and_print
from nat.cli.commands.mcp.mcp import call_tool_direct
from nat.cli.commands.mcp.mcp import format_tool
from nat.cli.commands.mcp.mcp import list_tools_direct
from nat.cli.commands.mcp.mcp import mcp_client_ping
from nat.cli.commands.mcp.mcp import mcp_client_tool_call
from nat.cli.commands.mcp.mcp import mcp_client_tool_list
from nat.cli.commands.mcp.mcp import ping_mcp_server
from nat.cli.commands.mcp.mcp import print_tool
from nat.cli.commands.mcp.mcp import validate_transport_cli_args


@pytest.fixture(name="mock_tools")
def fixture_mock_tools():
    return [
        {
            "name": "tool_a",
            "description": "Tool A description",
            "input_schema": None,
        },
        {
            "name": "tool_b",
            "description": "Tool B description",
            "input_schema": '{"type": "object", "properties": {"x": {"type": "number"}}}',
        },
    ]


@pytest.fixture(name="cli_runner")
def fixture_cli_runner():
    return CliRunner()


@pytest.mark.parametrize(
    "cli_args, expect_json, expected_substrings",
    [
        ([], False, ["tool_a", "tool_b"]),
        (["--detail"], False, ["Description: Tool A description", "Input Schema:"]),
        (["--json-output"], True, None),
        (["--json-output", "--detail"], True, None),
    ],
)
@patch("nat.cli.commands.mcp.mcp.list_tools_via_function_group", new_callable=AsyncMock)
def test_mcp_client_tool_list_variants(
    mock_fetcher,
    mock_tools,
    cli_runner,
    cli_args,
    expect_json,
    expected_substrings,
):
    mock_fetcher.return_value = mock_tools
    result = cli_runner.invoke(mcp_client_tool_list, cli_args)
    assert result.exit_code == 0
    if expect_json:
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert parsed[0]["name"] == "tool_a"
    else:
        for text in expected_substrings:
            assert text in result.output


@patch("nat.cli.commands.mcp.mcp.list_tools_via_function_group", new_callable=AsyncMock)
def test_mcp_client_tool_list_specific_tool(mock_fetcher, mock_tools):
    mock_fetcher.return_value = [mock_tools[1]]
    runner = CliRunner()
    result = runner.invoke(mcp_client_tool_list, ["--tool", "tool_b"])
    assert result.exit_code == 0
    assert "Tool: tool_b" in result.output
    assert "Description: Tool B description" in result.output


@pytest.mark.parametrize("json_flag", [False, True])
@patch("nat.cli.commands.mcp.mcp.ping_mcp_server", new_callable=AsyncMock)
def test_mcp_client_ping_output(mock_ping, cli_runner, json_flag):
    mock_ping.return_value = MCPPingResult(url="http://localhost:9901/mcp",
                                           status="healthy",
                                           response_time_ms=4.2,
                                           error=None)
    args = ["--json-output"] if json_flag else []
    result = cli_runner.invoke(mcp_client_ping, args)
    assert result.exit_code == 0
    if json_flag:
        data = json.loads(result.output)
        assert data["status"] == "healthy"
        assert data["url"].endswith("/mcp")
    else:
        assert "healthy" in result.output


@pytest.mark.parametrize("with_direct, expected_direct", [(False, False), (True, True)])
@patch("nat.cli.commands.mcp.mcp.call_tool_and_print", new_callable=AsyncMock)
def test_mcp_client_tool_call_direct_variants(mock_call, cli_runner, with_direct, expected_direct):
    mock_call.return_value = "OK"
    args = [
        "my_tool",
        "--json-args",
        "{}",
    ]
    if with_direct:
        args.insert(1, "--direct")
    result = cli_runner.invoke(mcp_client_tool_call, args)
    assert result.exit_code == 0
    assert "OK" in result.output
    assert mock_call.await_args is not None
    _, kwargs = mock_call.await_args
    assert kwargs.get("direct") is expected_direct


@patch("nat.cli.commands.mcp.mcp.list_tools_direct", new_callable=AsyncMock)
def test_mcp_client_tool_list_direct_fetcher_called(mock_fetcher, mock_tools):
    mock_fetcher.return_value = mock_tools
    runner = CliRunner()
    result = runner.invoke(mcp_client_tool_list, ["--direct"])  # default transport streamable-http
    assert result.exit_code == 0
    assert "tool_a" in result.output and "tool_b" in result.output
    assert mock_fetcher.await_args is not None
    args, kwargs = mock_fetcher.await_args
    # Check positional args: (command, url)
    assert args[0] is None  # command
    assert args[1] == "http://localhost:9901/mcp"  # url
    # Check keyword args
    assert kwargs['tool_name'] is None
    assert kwargs['transport'] == "streamable-http"
    assert kwargs['args'] == []
    assert kwargs['env'] is None


def test_mcp_client_tool_call_invalid_json_args():
    runner = CliRunner()
    result = runner.invoke(
        mcp_client_tool_call,
        [
            "my_tool",
            "--json-args",
            "{",  # invalid JSON
        ])
    assert result.exit_code == 0
    assert "[ERROR] Failed to parse --json-args" in result.output


@patch("nat.cli.commands.mcp.mcp.call_tool_and_print", new_callable=AsyncMock)
def test_mcp_client_tool_call_args_env_parsing(mock_call):
    mock_call.return_value = "OK"
    runner = CliRunner()
    result = runner.invoke(mcp_client_tool_call,
                           [
                               "my_tool",
                               "--transport",
                               "stdio",
                               "--command",
                               "server",
                               "--args",
                               "-v --port 1",
                               "--env",
                               "A=1 B=2",
                               "--json-args",
                               "{}",
                           ])
    assert result.exit_code == 0
    assert "OK" in result.output
    assert mock_call.await_args is not None
    _, kwargs = mock_call.await_args
    assert kwargs.get("transport") == "stdio"
    assert kwargs.get("command") == "server"
    assert kwargs.get("args") == ["-v", "--port", "1"]
    assert kwargs.get("env") == {"A": "1", "B": "2"}
    assert kwargs.get("direct") is False


@patch("nat.cli.commands.mcp.mcp.ping_mcp_server", new_callable=AsyncMock)
def test_mcp_client_ping_unreachable(mock_ping):
    mock_ping.return_value = MCPPingResult(url="http://localhost:9901/mcp",
                                           status="unhealthy",
                                           response_time_ms=None,
                                           error="Timeout after 1 seconds")
    runner = CliRunner()
    result = runner.invoke(mcp_client_ping, [])
    assert result.exit_code == 0
    assert "unhealthy" in result.output
    assert "Timeout" in result.output


@patch("nat.cli.commands.mcp.mcp.call_tool_and_print", new_callable=AsyncMock)
@patch("nat.cli.commands.mcp.mcp.format_mcp_error")
def test_mcp_client_tool_call_mcp_error_formatted(mock_format, mock_call):

    class _FakeMCPError(Exception):
        pass

    # Rebind MCPError symbol used in the module to our fake
    import nat.cli.commands.mcp.mcp as mcp_mod
    mcp_mod.MCPError = _FakeMCPError  # type: ignore

    mock_call.side_effect = _FakeMCPError("boom")
    runner = CliRunner()
    result = runner.invoke(mcp_client_tool_call, [
        "my_tool",
        "--json-args",
        "{}",
    ])
    assert result.exit_code == 0
    assert mock_format.called


class _DummySchema:

    def schema_json(self, indent=2):
        del indent
        return json.dumps({"type": "object", "properties": {"a": {"type": "string"}}}, indent=2)


def _install_fake_mcp(monkeypatch, *, list_tools_response=None, call_tool_result=None, ping_ok=True):
    fake_mcp = ModuleType("mcp")

    class _FakeClientSession:

        def __init__(self, read, write):  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        async def initialize(self):
            return None

        async def list_tools(self):

            class _Resp:

                def __init__(self, tools):
                    self.tools = tools

            tools = list_tools_response or []
            return _Resp(tools)

        async def call_tool(self, tool_name, tool_args):
            del tool_name, tool_args
            return call_tool_result

        async def send_ping(self):
            if not ping_ok:
                raise RuntimeError("ping failed")

    fake_mcp.ClientSession = _FakeClientSession

    fake_mcp_client_session = ModuleType("mcp.client.session")
    fake_mcp_client_session.ClientSession = _FakeClientSession

    class _Ctx:

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        async def __aenter__(self):
            return (object(), object())

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

    fake_mcp_client_sse = ModuleType("mcp.client.sse")

    def _sse_client(url=None):
        del url
        return _Ctx()

    fake_mcp_client_sse.sse_client = _sse_client

    fake_mcp_client_stdio = ModuleType("mcp.client.stdio")

    class _StdioServerParameters:

        def __init__(self, command=None, args=None, env=None):
            self.command = command
            self.args = args
            self.env = env

    def _stdio_client(server=None):
        del server
        return _Ctx()

    fake_mcp_client_stdio.StdioServerParameters = _StdioServerParameters
    fake_mcp_client_stdio.stdio_client = _stdio_client

    fake_mcp_client_stream = ModuleType("mcp.client.streamable_http")

    def _stream_client(url=None):
        del url
        return _Ctx()

    fake_mcp_client_stream.streamablehttp_client = _stream_client

    fake_mcp_types = ModuleType("mcp.types")

    class _TextContent:

        def __init__(self, text):
            self.text = text

    fake_mcp_types.TextContent = _TextContent

    import sys

    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.client.session", fake_mcp_client_session)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", fake_mcp_client_sse)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", fake_mcp_client_stdio)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", fake_mcp_client_stream)
    monkeypatch.setitem(sys.modules, "mcp.types", fake_mcp_types)

    return fake_mcp


@pytest.mark.parametrize(
    "name,input_schema,expect_none,expect_contains,expect_startswith_json",
    [
        ("t1", None, True, None, False),
        ("t2", _DummySchema(), False, "properties", False),
        ("t3", {
            "type": "object"
        }, False, None, True),
        ("t4", object(), False, "raw", False),
    ],
)
def test_format_tool(name, input_schema, expect_none, expect_contains, expect_startswith_json):

    class _Tool:

        def __init__(self, name, description, input_schema):
            self.name = name
            self.description = description
            self.input_schema = input_schema

    out = format_tool(_Tool(name, "d", input_schema))
    assert out["name"] == name
    if expect_none:
        assert out["input_schema"] is None
    else:
        assert isinstance(out["input_schema"], str)
        if expect_contains is not None:
            assert expect_contains in out["input_schema"]
        if expect_startswith_json:
            assert out["input_schema"].strip().startswith("{")


@pytest.mark.parametrize(
    "tool,detail,expected_present,expected_absent",
    [
        (
            {
                "name": "x", "description": "", "input_schema": None
            },
            False,
            ["Tool: x"],
            ["Description:"],
        ),
        (
            {
                "name": "x", "description": "desc", "input_schema": None
            },
            True,
            ["Tool: x", "Description: desc", "Input Schema: None"],
            [],
        ),
    ],
)
def test_print_tool_cases(capsys, tool, detail, expected_present, expected_absent):
    print_tool(tool, detail=detail)
    out = capsys.readouterr().out
    for txt in expected_present:
        assert txt in out
    for txt in expected_absent:
        assert txt not in out


@pytest.mark.parametrize(
    "transport,command",
    [
        ("sse", None),
        ("streamable-http", None),
        ("stdio", "server"),
    ],
)
def test_list_tools_direct_success_transports(monkeypatch, transport, command):

    class _Tool:

        def __init__(self, name, description, input_schema=None):
            self.name = name
            self.description = description
            self.input_schema = input_schema

    _install_fake_mcp(monkeypatch, list_tools_response=[_Tool("a", "da"), _Tool("b", "db", {"type": "object"})])
    tools = asyncio.run(
        list_tools_direct(
            command=command,
            url="http://u",
            tool_name=None,
            transport=transport,
            args=None,
            env=None,
        ))
    assert [t["name"] for t in tools] == ["a", "b"]


@pytest.mark.parametrize("transport,command", [("sse", None), ("streamable-http", None), ("stdio", "server")])
def test_list_tools_direct_tool_not_found_prints(monkeypatch, capsys, transport, command):

    class _Tool:

        def __init__(self, name, description, input_schema=None):  # noqa: ARG002
            self.name = name
            self.description = description
            self.input_schema = input_schema

    _install_fake_mcp(monkeypatch, list_tools_response=[_Tool("a", "da")])
    tools = asyncio.run(
        list_tools_direct(
            command=command,
            url="http://u",
            tool_name="missing",
            transport=transport,
            args=None,
            env=None,
        ))
    assert tools == []
    captured = capsys.readouterr()
    assert "[INFO] Tool 'missing' not found." in captured.out


@pytest.mark.parametrize("transport,command", [("sse", None), ("streamable-http", None), ("stdio", "server")])
def test_list_tools_direct_error_is_formatted(monkeypatch, transport, command):

    class _Tool:

        def __init__(self, name):  # noqa: ARG002
            self.name = name

    def _broken_list_tools(*args, **kwargs):
        raise RuntimeError("boom")

    _install_fake_mcp(monkeypatch, list_tools_response=[_Tool("x")])
    import mcp as _mcp  # type: ignore
    monkeypatch.setattr(_mcp.ClientSession, "list_tools", _broken_list_tools)

    tools = asyncio.run(
        list_tools_direct(
            command=command,
            url="http://u",
            tool_name=None,
            transport=transport,
            args=None,
            env=None,
        ))
    assert tools == []


@pytest.mark.parametrize(
    "transport,command",
    [
        ("sse", None),
        ("streamable-http", None),
        ("stdio", "server"),
    ],
)
def test_ping_mcp_server_healthy_transports(monkeypatch, transport, command):
    _install_fake_mcp(monkeypatch)
    res = asyncio.run(ping_mcp_server(url="http://u", timeout=5, transport=transport, command=command))
    assert isinstance(res, MCPPingResult)
    assert res.status == "healthy"
    assert res.response_time_ms is not None


@pytest.mark.parametrize("transport", ["sse", "streamable-http", "stdio"])
def test_ping_mcp_server_timeout(monkeypatch, transport):

    async def _raise_timeout(coro, timeout=None, **_kwargs):
        del timeout, _kwargs
        # Dispose the passed coroutine to avoid "never awaited" warnings
        try:
            coro.close()
        except Exception:
            pass
        # Simulate asyncio.wait_for timing out
        raise TimeoutError

    monkeypatch.setattr("nat.cli.commands.mcp.mcp.asyncio.wait_for", _raise_timeout)
    res = asyncio.run(ping_mcp_server(url="http://u", timeout=0, transport=transport))
    assert res.status == "unhealthy"
    assert res.error and "Timeout" in res.error


def test_ping_mcp_server_stdio_missing_command(monkeypatch):
    _install_fake_mcp(monkeypatch)
    res = asyncio.run(ping_mcp_server(url="ignored", timeout=5, transport="stdio", command=None))
    assert res.status == "unhealthy"
    assert "--command is required" in (res.error or "")


@pytest.mark.parametrize(
    "transport,command,url",
    [
        ("streamable-http", None, "http://u"),
        ("sse", None, "http://u"),
        ("stdio", "server", None),
    ],
)
def test_call_tool_direct_success_transports(monkeypatch, transport, command, url):

    class _Text:

        def __init__(self, text):
            self.text = text

    class _Result:

        def __init__(self):
            self.content = []
            self.isError = False

    _install_fake_mcp(monkeypatch, call_tool_result=_Result())

    import sys

    sys.modules["mcp.types"].TextContent = _Text  # type: ignore[attr-defined]

    async def _call_tool(self, tool_name, tool_args):
        del self, tool_name, tool_args
        r = _Result()
        r.content = [_Text("Hello"), "ignored-non-text"]
        return r

    sys.modules["mcp"].ClientSession.call_tool = _call_tool  # type: ignore

    out = asyncio.run(
        call_tool_direct(
            command=command,
            url=url,
            tool_name="echo",
            transport=transport,
            args=None,
            env=None,
            tool_args={"x": 1},
        ))
    assert out.splitlines()[0] == "Hello"


@pytest.mark.parametrize(
    "transport,command,url",
    [
        ("sse", None, "http://u"),
        ("streamable-http", None, "http://u"),
        ("stdio", "server", None),
    ],
)
def test_call_tool_direct_tool_error_converted(monkeypatch, transport, command, url):

    class _Result:

        def __init__(self):
            self.content = ["problem"]
            self.isError = True

    _install_fake_mcp(monkeypatch, call_tool_result=_Result())

    with pytest.raises(Exception) as excinfo:  # noqa: BLE001
        asyncio.run(
            call_tool_direct(
                command=command,
                url=url,
                tool_name="bad",
                transport=transport,
                args=None,
                env=None,
                tool_args=None,
            ))
    err = str(excinfo.value)
    assert "Unexpected error:" in err and "problem" in err


@pytest.mark.parametrize(
    "transport,url,command,expected",
    [
        ("sse", None, None, "--url is required"),
        ("streamable-http", None, None, "--url is required"),
    ],
)
def test_call_tool_direct_missing_required_config(monkeypatch, transport, url, command, expected):
    _install_fake_mcp(monkeypatch)
    with pytest.raises(Exception) as excinfo:  # noqa: BLE001
        asyncio.run(
            call_tool_direct(
                command=command,
                url=url,
                tool_name="x",
                transport=transport,
                args=None,
                env=None,
                tool_args=None,
            ))
    err = str(excinfo.value)
    assert "Unexpected error:" in err and expected in err


@pytest.mark.parametrize(
    "transport,command,args,env,expected_ok,expected_err",
    [
        ("sse", "cmd", "-v", "A=1", False, "--command, --args, and --env are not allowed"),
        ("streamable-http", None, None, None, True, None),
        ("stdio", "mcp", "", "", True, None),
        ("stdio", None, None, None, False, "--command is required when using stdio client type"),
    ],
)
def test_validate_transport_cli_args(capsys, transport, command, args, env, expected_ok, expected_err):
    ok = validate_transport_cli_args(transport, command, args, env)
    assert ok is expected_ok
    err = capsys.readouterr().err
    if expected_err:
        assert expected_err in err
    else:
        assert err == ""


def test_call_tool_and_print_group_success(monkeypatch):

    class _Fn:

        async def acall_invoke(self, **kwargs):
            del kwargs
            return "OK"

    class _Group:

        async def get_accessible_functions(self):
            return {"mcp_client.echo": _Fn()}

    fake_builder_mod = ModuleType("nat.builder.workflow_builder")

    class _WorkflowBuilder:

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        async def add_function_group(self, *_args, **_kwargs):  # noqa: ARG002
            return _Group()

    fake_builder_mod.WorkflowBuilder = _WorkflowBuilder

    fake_client_impl = ModuleType("nat.plugins.mcp.client_impl")

    class _Cfg:

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            _ = (args, kwargs)

    fake_client_impl.MCPClientConfig = _Cfg
    fake_client_impl.MCPServerConfig = _Cfg

    import sys

    monkeypatch.setitem(sys.modules, "nat.builder.workflow_builder", fake_builder_mod)
    monkeypatch.setitem(sys.modules, "nat.plugins.mcp.client_impl", fake_client_impl)

    out = asyncio.run(
        call_tool_and_print(
            command=None,
            url="http://u",
            tool_name="echo",
            transport="sse",
            args=None,
            env=None,
            tool_args=None,
            direct=False,
        ))
    assert out == "OK"


def test_call_tool_and_print_group_tool_not_found(monkeypatch):

    class _Group:

        async def get_accessible_functions(self):
            return {"mcp_client.other": object()}

    fake_builder_mod = ModuleType("nat.builder.workflow_builder")

    class _WorkflowBuilder:

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        async def add_function_group(self, *_args, **_kwargs):  # noqa: ARG002
            return _Group()

    fake_builder_mod.WorkflowBuilder = _WorkflowBuilder

    fake_client_impl = ModuleType("nat.plugins.mcp.client_impl")

    class _Cfg:

        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

    fake_client_impl.MCPClientConfig = _Cfg
    fake_client_impl.MCPServerConfig = _Cfg

    import sys

    monkeypatch.setitem(sys.modules, "nat.builder.workflow_builder", fake_builder_mod)
    monkeypatch.setitem(sys.modules, "nat.plugins.mcp.client_impl", fake_client_impl)

    err = None
    try:
        asyncio.run(
            call_tool_and_print(
                command=None,
                url="http://u",
                tool_name="echo",
                transport="sse",
                args=None,
                env=None,
                tool_args=None,
                direct=False,
            ))
    except RuntimeError as exc:  # noqa: BLE001
        err = str(exc)
    assert err is not None and "Tool 'echo' not found" in err
