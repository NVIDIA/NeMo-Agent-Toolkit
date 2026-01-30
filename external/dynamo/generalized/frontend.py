# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import csv
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import uvicorn
import uvloop
from dynamo.runtime import DistributedRuntime
from dynamo.runtime import dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import field_validator
from text_utils import strip_think_tags
from transformers import AutoTokenizer

configure_dynamo_logging()
logger = logging.getLogger(__name__)


# ----------------- Tool Call Parsing -----------------
@dataclass
class ParsedToolCall:
    """Represents a parsed tool call from model output."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool_calls format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            }
        }


class ToolCallParser:
    """
    Parses tool calls from model output in the format:
    <tool_call>
      <function=function_name>
        <parameter=param_name>
          value
        </parameter>
      </function>
    </tool_call>
    """

    # Pattern to match complete tool_call blocks
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>\s*(.*?)\s*</tool_call>',
        re.DOTALL
    )

    # Pattern to match function within a tool_call
    FUNCTION_PATTERN = re.compile(
        r'<function=([^>]+)>\s*(.*?)\s*</function>',
        re.DOTALL
    )

    # Pattern to match parameters within a function
    PARAMETER_PATTERN = re.compile(
        r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>',
        re.DOTALL
    )

    @classmethod
    def parse(cls, text: str) -> tuple[str, list[ParsedToolCall]]:
        """
        Parse tool calls from text.

        Returns:
            tuple of (text_without_tool_calls, list_of_parsed_tool_calls)
        """
        tool_calls: list[ParsedToolCall] = []

        # Find all tool_call blocks
        for match in cls.TOOL_CALL_PATTERN.finditer(text):
            tool_call_content = match.group(1)

            # Find function within this tool_call
            func_match = cls.FUNCTION_PATTERN.search(tool_call_content)
            if func_match:
                func_name = func_match.group(1).strip()
                func_content = func_match.group(2)

                # Parse parameters
                arguments: dict[str, Any] = {}
                for param_match in cls.PARAMETER_PATTERN.finditer(func_content):
                    param_name = param_match.group(1).strip()
                    param_value = param_match.group(2).strip()

                    # Try to parse as JSON for complex types, otherwise keep as string
                    try:
                        arguments[param_name] = json.loads(param_value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[param_name] = param_value

                tool_calls.append(ParsedToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    name=func_name,
                    arguments=arguments,
                ))

        # Remove tool_call blocks from text
        text_without_tools = cls.TOOL_CALL_PATTERN.sub('', text).strip()

        return text_without_tools, tool_calls

    @classmethod
    def has_complete_tool_call(cls, text: str) -> bool:
        """Check if text contains at least one complete tool_call block."""
        return bool(cls.TOOL_CALL_PATTERN.search(text))

    @classmethod
    def has_partial_tool_call(cls, text: str) -> bool:
        """Check if text contains a potentially incomplete tool_call (started but not closed)."""
        # Count opening and closing tags
        open_count = text.count('<tool_call>')
        close_count = text.count('</tool_call>')
        return open_count > close_count

    @classmethod
    def extract_partial_tool_call_info(cls, text: str) -> dict[str, Any] | None:
        """
        Extract partial tool call information for streaming.
        Returns partial info if we can determine function name, even if not complete.
        """
        # Look for the last incomplete tool_call
        last_open = text.rfind('<tool_call>')
        if last_open == -1:
            return None

        partial_content = text[last_open:]

        # Check if this tool_call is already complete
        if '</tool_call>' in partial_content:
            return None

        # Try to extract function name
        func_match = re.search(r'<function=([^>]+)>', partial_content)
        if not func_match:
            return None

        func_name = func_match.group(1).strip()

        # Try to extract any complete parameters so far
        arguments: dict[str, Any] = {}
        for param_match in cls.PARAMETER_PATTERN.finditer(partial_content):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            try:
                arguments[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                arguments[param_name] = param_value

        # Also check for partial parameter (opened but not closed)
        partial_param_match = re.search(
            r'<parameter=([^>]+)>\s*([^<]*?)$',
            partial_content,
            re.DOTALL
        )
        partial_param_name = None
        partial_param_value = None
        if partial_param_match:
            partial_param_name = partial_param_match.group(1).strip()
            partial_param_value = partial_param_match.group(2)

        return {
            "function_name": func_name,
            "complete_arguments": arguments,
            "partial_param_name": partial_param_name,
            "partial_param_value": partial_param_value,
        }


# ----------------- Pydantic request models -----------------
class Message(BaseModel):
    role: str
    # content may be None (assistant with tool_calls) or structured list
    content: Any | None = None
    # Optional fields for tool and assistant messages
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StreamOptions(BaseModel):
    include_usage: bool | None = False


class PrefixHints(BaseModel):
    session_id: str
    latency_priority: str  # LOW | MEDIUM | HIGH

    @field_validator('latency_priority')
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate and normalize priority to LOW/MEDIUM/HIGH."""
        if not v:
            return "MEDIUM"
        normalized = v.strip().upper()
        if normalized not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"latency_priority must be LOW/MEDIUM/HIGH, got: {v}")
        return normalized


class ChatCompletionRequest(BaseModel):
    model: str | None = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    messages: list[Message]
    max_tokens: int | None = 1024
    temperature: float | None = 0.6
    top_p: float | None = 0.999
    top_k: int | None = 1
    ignore_eos: bool | None = False

    # OpenAI-style streaming controls
    stream: bool | None = False
    stream_options: StreamOptions | None = None

    # New generalized hints (filled by frontend from headers)
    prefix_hints: PrefixHints | None = None

    # OpenAI-native tool calling support (pass-through to processor/engine)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None


# ----------------- Frontend handler -----------------
class FrontendRequestHandler:

    def __init__(self, runtime: DistributedRuntime) -> None:
        self.runtime = runtime
        self.processor_client = None
        self.app = None
        self.tokenizers: dict[str, AutoTokenizer] = {}
        # Regex to find one or more JSON objects optionally separated by semicolons

        # Load model mapping from environment (model_name -> model_path)
        # e.g., FRONTEND_MODEL_MAPPING='{"llama-3.3-70b": "/workspace/models/Llama-3.3-70B-Instruct"}'
        self.model_mapping: dict[str, str] = {}
        try:
            mapping_str = os.environ.get("FRONTEND_MODEL_MAPPING", "{}")
            self.model_mapping = json.loads(mapping_str)
            if self.model_mapping:
                logger.info("Loaded model mapping: %s", self.model_mapping)
        except Exception as e:
            logger.warning("Failed to parse FRONTEND_MODEL_MAPPING: %s", e)

        # Throughput (requests/sec) tracking
        self._tps_lock = asyncio.Lock()
        self._tps_count = 0
        try:
            self._tps_interval = float(os.environ.get("FRONTEND_TPS_INTERVAL", "5"))
        except Exception:
            self._tps_interval = 5.0
        self._tps_csv_path = os.environ.get("FRONTEND_TPS_CSV", "frontend_throughput.csv")
        self._tps_task = None

    async def initialize(self) -> None:
        """Initialize the frontend handler.

        Sets up the processor client, FastAPI application, routes, and background
        TPS tracking task.
        """
        self.processor_client = (await
                                 self.runtime.namespace("dynamo").component("processor").endpoint("process").client())
        logger.info("Processor client created, waiting for instances...")
        await self.processor_client.wait_for_instances()
        logger.info("Processor client ready")

        self.app = FastAPI(title="Dynamo")
        self.setup_routes()
        logging.info("Frontend initialized successfully")

        # Initialize TPS CSV and start background writer
        try:
            csv_dir = os.path.dirname(self._tps_csv_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
            if not os.path.exists(self._tps_csv_path):
                async with self._tps_lock:
                    with open(self._tps_csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["ts_epoch_ms", "requests", "interval_s", "req_per_sec"])
        except Exception as e:
            logger.warning("Failed to initialize TPS CSV %s: %s", self._tps_csv_path, e)

        # Start background task
        self._tps_task = asyncio.create_task(self._tps_writer())

    # ----- helpers -----
    def _get_tokenizer(self, model: str) -> AutoTokenizer:
        tok = self.tokenizers.get(model)
        if tok is None:
            # Use model mapping to resolve model name to path
            model_path = self.model_mapping.get(model, model)
            tok = AutoTokenizer.from_pretrained(model_path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.tokenizers[model] = tok
        return tok

    def _messages_to_text(self, messages: list[dict[str, str]], tokenizer) -> str:
        # Try chat template first; fall back to a plain transcript
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def setup_routes(self):

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            try:
                body = await request.body()
                logger.error("422 Unprocessable Entity. Errors: %s", exc.errors())
                logger.info("422 payload: %s", body.decode("utf-8", errors="ignore"))
            except Exception as e:  # pragma: no cover
                logger.exception("Failed to log 422 payload: %s", e)
            return JSONResponse(status_code=422, content={"detail": exc.errors()})

        @self.app.post("/v1/chat/completions")
        async def chat_completions(
                request: ChatCompletionRequest,
                # ---- New generalized prefix headers ----
                hdr_session_id: str | None = Header(None, alias="x-nat-session-id"),
                hdr_latency_priority: str | None = Header(None, alias="x-nat-prefix-latency-priority"),
        ):
            """
            OpenAI-compatible /v1/chat/completions:
            - Non-streaming: returns a single JSON completion.
            - Streaming: returns SSE 'chat.completion.chunk' events, then [DONE].
            - Passes per-prefix hints (ID/Total/OSL/IAT) to the processor.
            - Supports function/tool calling with XML-format tool calls from model.
            """

            try:
                # Convert to dict once; we may augment it with prefix hints
                req_dict: dict[str, Any] = request.model_dump()
                logger.info("Got full request: %s", req_dict)

                # ---- Build prefix_hints from headers (with robust defaults) ----
                session_id = hdr_session_id or f"auto-{uuid.uuid4().hex}"
                latency_priority = hdr_latency_priority or "MEDIUM"

                # PrefixHints will validate and normalize priority via Pydantic validator
                req_dict["prefix_hints"] = {
                    "session_id": session_id,
                    "latency_priority": latency_priority,
                }

                # Build the processor payload (includes stream fields)
                processor_req: dict[str, Any] = dict(req_dict)

                # Fast path: non-streaming -> JSON response
                if not request.stream:
                    processor_stream = await self.processor_client.generate(processor_req)
                    full_text = ""
                    finish_reason = "stop"

                    async for chunk in processor_stream:
                        data = chunk.data()
                        if "error" in data:
                            raise HTTPException(status_code=500, detail=data["error"])
                        # Prefer incremental deltas; accept cumulative 'content' if provided
                        if isinstance(data.get("delta"), str):
                            full_text += data["delta"]
                        elif isinstance(data.get("token"), str):
                            full_text += data["token"]
                        elif isinstance(data.get("text"), str):
                            full_text += data["text"]
                        elif isinstance(data.get("content"), str):
                            full_text = data["content"]

                    # Parse tool calls from the response
                    text_content, tool_calls = ToolCallParser.parse(full_text)

                    # Filter out <think> tags from text content
                    text_content = strip_think_tags(text_content)

                    tok = self._get_tokenizer(request.model or "nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
                    prompt_text = self._messages_to_text(processor_req["messages"], tok)
                    prompt_tokens = len(tok.encode(prompt_text, add_special_tokens=True))
                    # Count tokens from full response (including tool call markup)
                    completion_tokens = len(tok.encode(full_text, add_special_tokens=False))

                    message_payload: dict[str, Any] = {"role": "assistant"}

                    if tool_calls:
                        # Response contains tool calls
                        finish_reason = "tool_calls"
                        message_payload["tool_calls"] = [tc.to_openai_format() for tc in tool_calls]
                        # Include content only if there's meaningful text outside tool calls
                        if text_content:
                            message_payload["content"] = text_content
                        else:
                            message_payload["content"] = None
                    else:
                        # Regular text response
                        message_payload["content"] = text_content

                    # Count completed request
                    await self._inc_tps()

                    return {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": message_payload,
                            "finish_reason": finish_reason,
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }

                # ------------- streaming path (SSE) -------------
                include_usage = bool(getattr(request.stream_options or StreamOptions(), "include_usage", False))

                async def sse_stream() -> AsyncGenerator[str, None]:
                    created = int(time.time())
                    resp_id = f"chatcmpl-{uuid.uuid4().hex}"
                    model_name = request.model or "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

                    # Prepare tokenizer & prompt token count (for usage if requested)
                    prompt_tokens = 0
                    tok = None
                    if include_usage:
                        tok = self._get_tokenizer(model_name)
                        prompt_text = self._messages_to_text(processor_req["messages"], tok)
                        prompt_tokens = len(tok.encode(prompt_text, add_special_tokens=True))

                    def sse_packet(payload: dict[str, Any]) -> str:
                        return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"

                    def make_chunk(
                        delta: dict[str, Any],
                        finish_reason: str | None,
                        tool_calls_delta: list[dict[str, Any]] | None = None
                    ) -> dict[str, Any]:
                        chunk_delta = dict(delta)
                        if tool_calls_delta:
                            chunk_delta["tool_calls"] = tool_calls_delta
                        return {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": chunk_delta,
                                "finish_reason": finish_reason,
                            }],
                        }

                    # 1) Send the role chunk first
                    yield sse_packet(make_chunk(delta={"role": "assistant"}, finish_reason=None))

                    # 2) Stream content chunks from processor
                    processor_stream = await self.processor_client.generate(processor_req)
                    full_text = ""
                    filtered_sent = ""  # Track how much filtered content we've sent
                    finish_reason: str | None = None

                    # Tool call streaming state
                    tool_call_mode = False
                    current_tool_call_index = 0
                    sent_tool_calls: set[int] = set()  # Track which tool calls we've started streaming
                    last_streamed_args: dict[int, str] = {}  # Track streamed args per tool call index

                    try:
                        async for chunk in processor_stream:
                            data = chunk.data()
                            if "error" in data:
                                raise HTTPException(status_code=500, detail=data["error"])

                            piece: str | None = None
                            if isinstance(data.get("delta"), str):
                                piece = data["delta"]
                            elif isinstance(data.get("token"), str):
                                piece = data["token"]
                            elif isinstance(data.get("text"), str):
                                piece = data["text"]
                            elif isinstance(data.get("content"), str):
                                # If cumulative content, stream only the unseen suffix
                                cum = data["content"]
                                start = len(full_text)
                                if len(cum) > start:
                                    piece = cum[start:]

                            if piece:
                                full_text += piece

                                # Check for tool call patterns
                                if ToolCallParser.has_partial_tool_call(full_text) or ToolCallParser.has_complete_tool_call(full_text):
                                    # We're in tool call territory
                                    if not tool_call_mode:
                                        tool_call_mode = True
                                        # Send any remaining text content before tool calls
                                        text_before_tool = full_text.split('<tool_call>')[0]
                                        filtered_before = strip_think_tags(text_before_tool)
                                        if len(filtered_before) > len(filtered_sent):
                                            new_piece = filtered_before[len(filtered_sent):]
                                            filtered_sent = filtered_before
                                            if new_piece.strip():
                                                yield sse_packet(make_chunk(delta={"content": new_piece}, finish_reason=None))

                                    # Parse complete tool calls and stream them
                                    _, complete_tool_calls = ToolCallParser.parse(full_text)

                                    for idx, tc in enumerate(complete_tool_calls):
                                        if idx not in sent_tool_calls:
                                            # Send tool call start (name)
                                            tool_call_delta = [{
                                                "index": idx,
                                                "id": tc.id,
                                                "type": "function",
                                                "function": {
                                                    "name": tc.name,
                                                    "arguments": "",
                                                }
                                            }]
                                            yield sse_packet(make_chunk(delta={}, finish_reason=None, tool_calls_delta=tool_call_delta))
                                            sent_tool_calls.add(idx)
                                            last_streamed_args[idx] = ""

                                        # Stream arguments incrementally
                                        full_args = json.dumps(tc.arguments)
                                        already_sent = last_streamed_args.get(idx, "")
                                        if len(full_args) > len(already_sent):
                                            args_delta = full_args[len(already_sent):]
                                            tool_call_delta = [{
                                                "index": idx,
                                                "function": {
                                                    "arguments": args_delta,
                                                }
                                            }]
                                            yield sse_packet(make_chunk(delta={}, finish_reason=None, tool_calls_delta=tool_call_delta))
                                            last_streamed_args[idx] = full_args

                                    # Also handle partial tool calls (streaming args as they come)
                                    partial_info = ToolCallParser.extract_partial_tool_call_info(full_text)
                                    if partial_info:
                                        idx = len(complete_tool_calls)  # Next index
                                        if idx not in sent_tool_calls:
                                            # Send tool call start
                                            tool_call_delta = [{
                                                "index": idx,
                                                "id": f"call_{uuid.uuid4().hex[:24]}",
                                                "type": "function",
                                                "function": {
                                                    "name": partial_info["function_name"],
                                                    "arguments": "",
                                                }
                                            }]
                                            yield sse_packet(make_chunk(delta={}, finish_reason=None, tool_calls_delta=tool_call_delta))
                                            sent_tool_calls.add(idx)
                                            last_streamed_args[idx] = ""

                                        # Stream partial arguments
                                        partial_args = partial_info["complete_arguments"]
                                        if partial_args:
                                            full_args = json.dumps(partial_args)
                                            already_sent = last_streamed_args.get(idx, "")
                                            if len(full_args) > len(already_sent):
                                                # For partial, we stream what we have so far
                                                # This is tricky because JSON isn't complete yet
                                                # Just stream the partial JSON representation
                                                args_delta = full_args[len(already_sent):]
                                                tool_call_delta = [{
                                                    "index": idx,
                                                    "function": {
                                                        "arguments": args_delta,
                                                    }
                                                }]
                                                yield sse_packet(make_chunk(delta={}, finish_reason=None, tool_calls_delta=tool_call_delta))
                                                last_streamed_args[idx] = full_args

                                else:
                                    # Regular text content (no tool calls detected yet)
                                    filtered_full = strip_think_tags(full_text)
                                    if len(filtered_full) > len(filtered_sent):
                                        new_piece = filtered_full[len(filtered_sent):]
                                        filtered_sent = filtered_full
                                        yield sse_packet(make_chunk(delta={"content": new_piece}, finish_reason=None))

                            if "finish_reason" in data and data["finish_reason"] is not None:
                                finish_reason = data["finish_reason"]

                    except HTTPException:
                        raise
                    except Exception as e:
                        logging.exception("Streaming error: %s", e)
                        yield sse_packet(make_chunk(delta={}, finish_reason="error"))
                        yield "data: [DONE]\n\n"
                        return

                    # 3) Final finish chunk
                    _, final_tool_calls = ToolCallParser.parse(full_text)
                    if final_tool_calls:
                        final_reason = "tool_calls"
                    else:
                        final_reason = finish_reason if finish_reason else "stop"
                    yield sse_packet(make_chunk(delta={}, finish_reason=final_reason))

                    # 4) Optional usage chunk
                    if include_usage:
                        if tok is None:
                            tok = self._get_tokenizer(model_name)
                        completion_tokens = len(tok.encode(full_text, add_special_tokens=False))
                        usage_chunk = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        }
                        yield sse_packet(usage_chunk)

                    # 5) Terminator
                    yield "data: [DONE]\n\n"
                    # Count completed request
                    await self._inc_tps()

                return StreamingResponse(
                    sse_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            except HTTPException:
                raise
            except Exception as e:
                logging.error("Error in chat completions: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

    async def run_server(self, host: str = "0.0.0.0", port: int = 8099) -> None:
        """Start the FastAPI server.

        Args:
            host: Host address to bind to. Defaults to all interfaces.
            port: Port number to listen on. Defaults to 8099.
        """
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        logging.info("Starting FastAPI server on %s:%s", host, port)
        await server.serve()

    # ----------------- throughput helpers -----------------
    async def _inc_tps(self):
        try:
            async with self._tps_lock:
                self._tps_count += 1
        except Exception:
            pass

    async def _tps_writer(self):
        interval = max(0.5, float(self._tps_interval))
        while True:
            try:
                await asyncio.sleep(interval)
                async with self._tps_lock:
                    count = int(self._tps_count)
                    self._tps_count = 0
                rps = float(count) / interval
                ts_ms = int(time.time() * 1000)
                try:
                    with open(self._tps_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([ts_ms, count, f"{interval:.3f}", f"{rps:.6f}"])
                except Exception as e:
                    logger.debug("Failed to append TPS CSV: %s", e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("TPS writer loop error: %s", e)


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime) -> None:
    """Dynamo worker entry point for the frontend service.
    Args:
        runtime: The distributed runtime for inter-service communication.
    """
    frontend = FrontendRequestHandler(runtime)
    await frontend.initialize()
    await frontend.run_server()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())  # pylint: disable=no-value-for-parameter