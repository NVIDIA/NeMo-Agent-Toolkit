# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import uvicorn
import uvloop
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
import os
import csv
import re

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


# ----------------- Pydantic request models -----------------
class Message(BaseModel):
    role: str
    # content may be None (assistant with tool_calls) or structured list
    content: Any | None = None
    # Optional fields for tool and assistant messages
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class PrefixHints(BaseModel):
    prefix_id: str
    total_requests: int
    osl: str  # LOW | MEDIUM | HIGH
    iat: str  # LOW | MEDIUM | HIGH  (estimated time between requests)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.999
    top_k: Optional[int] = 1
    ignore_eos: Optional[bool] = False

    # OpenAI-style streaming controls
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None

    # New generalized hints (filled by frontend from headers)
    prefix_hints: Optional[PrefixHints] = None

    # OpenAI-native tool calling support (pass-through to processor/engine)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None


# ----------------- Frontend handler -----------------
class FrontendRequestHandler:
    def __init__(self, runtime: DistributedRuntime):
        self.runtime = runtime
        self.processor_client = None
        self.app = None
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        # Model name mapping: served_name -> actual_model_path
        # Can be configured via FRONTEND_MODEL_MAPPING env var (JSON format)
        # Example: FRONTEND_MODEL_MAPPING='{"llama-3.1-8b": "/path/to/model"}'
        self.model_mapping: Dict[str, str] = {}
        model_mapping_str = os.environ.get("FRONTEND_MODEL_MAPPING", "{}")
        try:
            self.model_mapping = json.loads(model_mapping_str)
            logger.info("Frontend model mapping: %s", self.model_mapping)
        except Exception as e:
            logger.warning("Failed to parse FRONTEND_MODEL_MAPPING: %s", e)
        # Regex to find one or more JSON objects optionally separated by semicolons
        self._tool_json_regex = re.compile(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}(?:\s*;\s*{[^{}]*(?:{[^{}]*}[^{}]*)*})*', re.DOTALL)

        # Throughput (requests/sec) tracking
        self._tps_lock = asyncio.Lock()
        self._tps_count = 0
        try:
            self._tps_interval = float(os.environ.get("FRONTEND_TPS_INTERVAL", "5"))
        except Exception:
            self._tps_interval = 5.0
        self._tps_csv_path = os.environ.get("FRONTEND_TPS_CSV", "frontend_throughput.csv")
        self._tps_task = None

    async def initialize(self):
        self.processor_client = (
            await self.runtime.namespace("dynamo")
            .component("processor")
            .endpoint("process")
            .client()
        )
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
        # Map served model name to actual model path if configured
        actual_model = self.model_mapping.get(model, model)
        
        tok = self.tokenizers.get(actual_model)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(actual_model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.tokenizers[actual_model] = tok
        return tok

    def _messages_to_text(self, messages: List[Dict[str, str]], tokenizer) -> str:
        # Try chat template first; fall back to a plain transcript
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
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
            hdr_prefix_id: Optional[str] = Header(None, alias="x-prefix-id"),
            hdr_prefix_total: Optional[str] = Header(None, alias="x-prefix-total-requests"),
            hdr_prefix_osl: Optional[str] = Header(None, alias="x-prefix-osl"),
            hdr_prefix_iat: Optional[str] = Header(None, alias="x-prefix-iat"),
        ):
            """
            OpenAI-compatible /v1/chat/completions:
            - Non-streaming: returns a single JSON completion.
            - Streaming: returns SSE 'chat.completion.chunk' events, then [DONE].
            - Passes per-prefix hints (ID/Total/OSL/IAT) to the processor.
            """
            try:
                # Convert to dict once; we may augment it with prefix hints
                req_dict: Dict[str, Any] = request.model_dump()
                logger.info("Got full request: %s", req_dict)

                # ---- Build prefix_hints from headers (with robust defaults) ----
                prefix_id = hdr_prefix_id or f"auto-{uuid.uuid4().hex}"
                try:
                    total_requests = int(hdr_prefix_total) if hdr_prefix_total is not None else 1
                except Exception:
                    total_requests = 1

                def norm_level(v: Optional[str], default: str = "MEDIUM") -> str:
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

                # Build the processor payload (includes stream fields and any tool-calling params)
                processor_req: Dict[str, Any] = dict(req_dict)

                # Fast path: non-streaming -> JSON response
                if not request.stream:
                    processor_stream = await self.processor_client.generate(processor_req)
                    full_text = ""
                    finish_reason = "stop"
                    collected_tool_calls: List[Dict[str, Any]] = []

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
                        # Tool-calls pass-through from processor/engine
                        if isinstance(data.get("tool_calls"), list):
                            collected_tool_calls.extend(data["tool_calls"])  # type: ignore[arg-type]
                        if "finish_reason" in data and data["finish_reason"] is not None:
                            finish_reason = data["finish_reason"]

                    # Normalize any explicit tool_calls the processor/engine emitted
                    if collected_tool_calls:
                        collected_tool_calls = self._normalize_tool_calls(collected_tool_calls)

                    # If the backend didn't surface tool_calls explicitly, try to parse from text
                    if not collected_tool_calls and (request.tools is not None and len(request.tools) > 0):
                        parsed_calls = self._extract_tool_calls_from_text(full_text)
                        if parsed_calls:
                            collected_tool_calls = parsed_calls

                    tok = self._get_tokenizer(request.model)
                    prompt_text = self._messages_to_text(processor_req["messages"], tok)
                    prompt_tokens = len(tok.encode(prompt_text, add_special_tokens=True))
                    completion_tokens = len(tok.encode(full_text, add_special_tokens=False))

                    message_payload: Dict[str, Any]
                    if collected_tool_calls:
                        message_payload = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": collected_tool_calls,
                        }
                        finish_reason = "tool_calls"
                    else:
                        message_payload = {"role": "assistant", "content": full_text}

                    # Count completed request
                    await self._inc_tps()

                    return {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": message_payload,
                                "finish_reason": finish_reason,
                            }
                        ],
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

                    def sse_packet(payload: Dict[str, Any]) -> str:
                        return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"

                    def make_chunk(delta: Dict[str, Any], finish_reason: Optional[str]) -> Dict[str, Any]:
                        return {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }

                    # 1) Send the role chunk first
                    yield sse_packet(make_chunk(delta={"role": "assistant"}, finish_reason=None))

                    # 2) Stream content chunks from processor
                    processor_stream = await self.processor_client.generate(processor_req)
                    full_text = ""
                    finish_reason: Optional[str] = None
                    saw_tool_calls = False
                    # Incremental tool-call parsing state
                    incremental_tool_mode = bool(request.tools)
                    tool_buffer = ""
                    tool_calls_emitted = False

                    try:
                        async for chunk in processor_stream:
                            data = chunk.data()
                            if "error" in data:
                                raise HTTPException(status_code=500, detail=data["error"])

                            piece: Optional[str] = None
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
                                # If tools requested, try to parse incremental tool-calls
                                if incremental_tool_mode and not tool_calls_emitted:
                                    tool_buffer += piece
                                    calls, remainder, found = self._extract_tool_calls_incremental(tool_buffer)
                                    if found and calls:
                                        # Normalize then emit one tool_calls delta
                                        calls = self._normalize_tool_calls(calls)
                                        saw_tool_calls = True
                                        tool_calls_emitted = True
                                        yield sse_packet(make_chunk(delta={"tool_calls": calls}, finish_reason=None))
                                        tool_buffer = remainder
                                    # Suppress normal content deltas while parsing tool JSON
                                else:
                                    yield sse_packet(make_chunk(delta={"content": piece}, finish_reason=None))

                            # Tool-calls pass-through
                            if isinstance(data.get("tool_calls"), list):
                                saw_tool_calls = True
                                delta_obj = {"tool_calls": self._normalize_tool_calls(data["tool_calls"])}
                                yield sse_packet(make_chunk(delta=delta_obj, finish_reason=None))

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
                    final_reason = finish_reason or ("tool_calls" if saw_tool_calls else "stop")
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

    # ----------------- tool call parsing (non-streaming) -----------------
    def _extract_tool_calls_from_text(self, model_output: str) -> List[Dict[str, Any]]:
        """
        Parse Llama-style JSON tool calls from a full assistant message.
        Supports multiple JSON objects separated by semicolons.
        Returns OpenAI-compatible tool_calls list or empty list.
        """
        try:
            if not model_output or ('{' not in model_output):
                return []
            m = self._tool_json_regex.search(model_output)
            if not m:
                return []
            json_str = m.group(0)
            objects = [s.strip() for s in json_str.split(';') if s.strip()]
            tool_calls: List[Dict[str, Any]] = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                except Exception:
                    # If surrounding tokens exist (like special tags), try to trim to braces window
                    left = obj_str.find('{')
                    right = obj_str.rfind('}')
                    if left != -1 and right != -1 and right > left:
                        try:
                            obj = json.loads(obj_str[left:right + 1])
                        except Exception:
                            continue
                    else:
                        continue
                name = obj.get("name")
                raw_args = obj.get("arguments", obj.get("parameters"))
                if not name or raw_args is None:
                    continue
                # Coerce args to a dict (handles double-encoded strings)
                args_dict = self._coerce_args_to_dict(raw_args)
                try:
                    args_str = json.dumps(args_dict, ensure_ascii=False)
                except Exception:
                    args_str = str(args_dict)
                tool_calls.append({
                    "id": f"chatcmpl-tool-{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                    # Add LangChain-friendly fields as well
                    "name": name,
                    "args": args_dict,
                })
            # Final pass to normalize shapes in case downstream expects both
            return self._normalize_tool_calls(tool_calls)
        except Exception as e:
            logger.exception("Tool-call parsing failed: %s", e)
            return []

    def _normalize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure each tool call includes both OpenAI fields (function.arguments as string)
        and LangChain-friendly fields (name, args as dict).
        """
        normalized: List[Dict[str, Any]] = []
        for tc in tool_calls:
            tc = dict(tc)  # shallow copy
            name = tc.get("name")
            args_obj: Any = tc.get("args")
            fn = tc.get("function") or {}
            fn_name = fn.get("name") if isinstance(fn, dict) else None
            fn_args = fn.get("arguments") if isinstance(fn, dict) else None

            if not name and fn_name:
                name = fn_name

            # Derive args dict (handle nested/double-encoded JSON strings)
            candidate = args_obj if args_obj is not None else fn_args
            args_obj = self._coerce_args_to_dict(candidate)

            # Ensure OpenAI shape present with arguments as string
            if isinstance(fn, dict):
                fn = dict(fn)
                try:
                    fn["arguments"] = json.dumps(args_obj, ensure_ascii=False)
                except Exception:
                    fn["arguments"] = str(args_obj)
                if name and not fn.get("name"):
                    fn["name"] = name
                tc["function"] = fn

            if name:
                tc["name"] = name
            tc["args"] = args_obj
            normalized.append(tc)
        return normalized

    def _coerce_args_to_dict(self, value: Any) -> Dict[str, Any]:
        """
        Convert a possibly nested/double-encoded JSON string into a dict.
        Repeatedly json.loads while the value is a string; stop when dict/list
        or parsing fails. If final is dict -> return; if list -> wrap; else {}.
        """
        if isinstance(value, dict):
            return value
        cur = value
        for _ in range(3):  # cap nesting depth
            if isinstance(cur, str):
                s = cur.strip()
                # quick check for likely JSON object/array
                if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                    try:
                        cur = json.loads(s)
                        continue
                    except Exception:
                        break
                else:
                    break
            break
        if isinstance(cur, dict):
            return cur
        if isinstance(cur, list):
            return {"items": cur}
        return {}

    # Incremental tool-call extraction from growing buffer (balanced braces; semicolon-separated)
    def _extract_tool_calls_incremental(self, buffer: str) -> tuple[list[Dict[str, Any]], str, bool]:
        calls: list[Dict[str, Any]] = []
        idx = 0
        n = len(buffer)
        found_any = False
        while idx < n:
            # find next '{'
            start = buffer.find('{', idx)
            if start == -1:
                break
            brace = 0
            i = start
            end = -1
            while i < n:
                ch = buffer[i]
                if ch == '{':
                    brace += 1
                elif ch == '}':
                    brace -= 1
                    if brace == 0:
                        end = i
                        break
                i += 1
            if end == -1:
                # incomplete JSON, keep remainder
                break
            candidate = buffer[start:end + 1]
            try:
                obj = json.loads(candidate)
                name = obj.get("name")
                raw_args = obj.get("arguments", obj.get("parameters"))
                if name is not None and raw_args is not None:
                    calls.append({"name": name, "args": self._coerce_args_to_dict(raw_args)})
                    found_any = True
            except Exception:
                # ignore malformed candidate
                pass
            # Move past this object; allow optional semicolon separators
            idx = end + 1
            while idx < n and buffer[idx] in (' ', '\n', '\r', '\t', ';'):
                idx += 1
        remainder = buffer[idx:] if idx < n else ""
        return calls, remainder, found_any

    async def run_server(self, host: str = "0.0.0.0", port: int = 8099):
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
async def worker(runtime: DistributedRuntime):
    frontend = FrontendRequestHandler(runtime)
    await frontend.initialize()
    await frontend.run_server()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())  # pylint: disable=no-value-for-parameter
