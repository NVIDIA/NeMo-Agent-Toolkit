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
import time
import uuid
from collections.abc import AsyncGenerator
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
from transformers import AutoTokenizer

configure_dynamo_logging()
logger = logging.getLogger(__name__)


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
    prefix_id: str
    total_requests: int
    osl: str  # LOW | MEDIUM | HIGH
    iat: str  # LOW | MEDIUM | HIGH  (estimated time between requests)


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
                hdr_prefix_id: str | None = Header(None, alias="x-prefix-id"),
                hdr_prefix_total: str | None = Header(None, alias="x-prefix-total-requests"),
                hdr_prefix_osl: str | None = Header(None, alias="x-prefix-osl"),
                hdr_prefix_iat: str | None = Header(None, alias="x-prefix-iat"),
        ):
            """
            OpenAI-compatible /v1/chat/completions:
            - Non-streaming: returns a single JSON completion.
            - Streaming: returns SSE 'chat.completion.chunk' events, then [DONE].
            - Passes per-prefix hints (ID/Total/OSL/IAT) to the processor.
            """

            # No support for tool calling. Raise error if request.tools
            if request.tools:
                raise HTTPException(status_code=400, detail="Tool calling is not supported by this frontend.")

            try:
                # Convert to dict once; we may augment it with prefix hints
                req_dict: dict[str, Any] = request.model_dump()
                logger.info("Got full request: %s", req_dict)

                # ---- Build prefix_hints from headers (with robust defaults) ----
                prefix_id = hdr_prefix_id or f"auto-{uuid.uuid4().hex}"
                try:
                    total_requests = int(hdr_prefix_total) if hdr_prefix_total is not None else 1
                except Exception:
                    total_requests = 1

                def norm_level(v: str | None, default: str = "MEDIUM") -> str:
                    if not v:
                        return default
                    v = str(v).strip().upper()
                    return v if v in ("LOW", "MEDIUM", "HIGH") else default

                osl = norm_level(hdr_prefix_osl, "MEDIUM")
                iat = norm_level(hdr_prefix_iat, "MEDIUM")

                req_dict["prefix_hints"] = {
                    "prefix_id": prefix_id,
                    "total_requests": total_requests,
                    "osl": osl,
                    "iat": iat,
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

                    tok = self._get_tokenizer(request.model)
                    prompt_text = self._messages_to_text(processor_req["messages"], tok)
                    prompt_tokens = len(tok.encode(prompt_text, add_special_tokens=True))
                    completion_tokens = len(tok.encode(full_text, add_special_tokens=False))

                    message_payload: dict[str, Any]
                    message_payload = {"role": "assistant", "content": full_text}

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
                    model_name = request.model

                    # Prepare tokenizer & prompt token count (for usage if requested)
                    prompt_tokens = 0
                    tok = None
                    if include_usage:
                        tok = self._get_tokenizer(model_name)
                        prompt_text = self._messages_to_text(processor_req["messages"], tok)
                        prompt_tokens = len(tok.encode(prompt_text, add_special_tokens=True))

                    def sse_packet(payload: dict[str, Any]) -> str:
                        return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"

                    def make_chunk(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
                        return {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }],
                        }

                    # 1) Send the role chunk first
                    yield sse_packet(make_chunk(delta={"role": "assistant"}, finish_reason=None))

                    # 2) Stream content chunks from processor
                    processor_stream = await self.processor_client.generate(processor_req)
                    full_text = ""
                    finish_reason: str | None = None

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
                                yield sse_packet(make_chunk(delta={"content": piece}, finish_reason=None))

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
