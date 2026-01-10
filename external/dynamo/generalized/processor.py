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

import argparse
import asyncio
import csv
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import uvloop
from dynamo.runtime import DistributedRuntime
from dynamo.runtime import dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from pydantic import BaseModel
from transformers import AutoTokenizer

configure_dynamo_logging()
logger = logging.getLogger(__name__)


# ----------------------- request / response models ----------------------- #
class Message(BaseModel):
    role: str
    # Allow None or structured content for assistant tool-calls and tool messages
    content: Any | None = None
    # Optional fields for tool and assistant messages
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StreamOptions(BaseModel):
    include_usage: bool | None = False


class PrefixHints(BaseModel):
    prefix_id: str
    total_requests: int  # same value on every call for this prefix
    osl: str  # LOW | MEDIUM | HIGH  (output sequence length)
    iat: str  # LOW | MEDIUM | HIGH  (inter-arrival time)


class ChatCompletionRequest(BaseModel):
    model: str | None = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: list[Message]
    max_tokens: int | None = 1024
    temperature: float | None = 0.6
    top_p: float | None = 0.999
    top_k: int | None = 1
    ignore_eos: bool | None = False

    # Passed through from frontend
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    prefix_hints: PrefixHints | None = None

    # Native tool-calling support (pass-through to engine)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None


class RouterRequest(BaseModel):
    tokens: list[int]
    prefix_id: str
    reuse_budget: int = 0  # remaining *after this request*
    expected_osl: str | None = None
    interarrival: str | None = None


class RouterFeedbackRequest(BaseModel):
    decision_id: str
    latency_ms: float
    success: bool | None = True
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None


# -------------------------- processor handler -------------------------- #
class ProcessorRequestHandler:

    def __init__(
        self,
        runtime: DistributedRuntime,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        enable_router: bool = True,
    ):
        self.runtime = runtime
        self.model_name = model_name
        self.enable_router = enable_router

        self.tokenizer: AutoTokenizer | None = None
        self.router_pick_client = None
        self.router_feedback_client = None
        self.engine_client = None

        # Prefix-level state: {prefix_id: {"total": int, "processed": int}}
        self._prefix_state: dict[str, dict[str, int]] = {}
        self._prefix_lock = asyncio.Lock()

        # CSV metrics logging
        self._metrics_lock = asyncio.Lock()
        # Allow override with env; default to local file
        self._metrics_csv_path = os.environ.get("PROCESSOR_METRICS_CSV", "processor_requests.csv")
        # Cap number of rows to avoid unbounded growth
        try:
            self._metrics_log_cap = int(os.environ.get("PROCESSOR_METRICS_MAX_ROWS", "2048"))
        except Exception:
            self._metrics_log_cap = 2048
        self._metrics_written_count = 0

        # def _raise_exception(msg: str):  # pragma: no cover - template guard
        #     raise ValueError(msg)

    # ---- init ----
    async def initialize(self):
        """Initialize processor by polling for router and backend."""
        logger.info(f"Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.enable_router:
            ns = self.runtime.namespace("dynamo").component("router")
            self.router_pick_client = await ns.endpoint("find_worker").client()
            self.router_feedback_client = await ns.endpoint("feedback").client()
            logger.info("Router clients created, waiting for instances...")
            await self.router_pick_client.wait_for_instances()
            logger.info("Router clients initialized successfully")

        # Engine client
        self.engine_client = await self.runtime.namespace("dynamo").component("backend").endpoint("generate").client()
        logger.info("Engine client created, waiting for backend instances...")
        await self.engine_client.wait_for_instances()
        logger.info("Processor initialized successfully")

        # Initialize metrics CSV with header if file doesn't exist
        try:
            csv_dir = os.path.dirname(self._metrics_csv_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
            async with self._metrics_lock:
                if not os.path.exists(self._metrics_csv_path):
                    with open(self._metrics_csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["num_tokens", "latency_ms", "latency_ms_per_token"])
                        self._metrics_written_count = 0
                else:
                    # Count existing data rows (exclude header)
                    try:
                        with open(self._metrics_csv_path, newline="") as f:
                            # subtract header if present
                            lines = sum(1 for _ in f)
                            self._metrics_written_count = max(lines - 1, 0)
                    except Exception:
                        self._metrics_written_count = 0
        except Exception as e:
            logger.warning("Failed to initialize metrics CSV %s: %s", self._metrics_csv_path, e)

    # ---- helpers ----
    def _render_prompt(self, messages: list[Message]) -> str:
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(message_dicts, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using simple format")

        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"

    def tokenize(self, text: str) -> list[int]:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")
        return self.tokenizer.encode(text, add_special_tokens=True)

    def detokenize(self, token_ids: list[int]) -> str:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    async def _update_prefix_state(self, hints: PrefixHints) -> tuple[int, str, str]:
        """Updates prefix counters and returns (remaining_after, osl, iat)."""
        pid = hints.prefix_id
        total = max(1, int(hints.total_requests))
        osl = (hints.osl or "MEDIUM").strip().upper()
        if osl not in ("LOW", "MEDIUM", "HIGH"):
            osl = "MEDIUM"
        iat = (hints.iat or "MEDIUM").strip().upper()
        if iat not in ("LOW", "MEDIUM", "HIGH"):
            iat = "MEDIUM"

        async with self._prefix_lock:
            s = self._prefix_state.get(pid)
            if s is None:
                s = {"total": total, "processed": 0}
                self._prefix_state[pid] = s
            else:
                s["total"] = max(s["total"], total)
            s["processed"] += 1
            remaining_after = max(s["total"] - s["processed"], 0)
            if remaining_after == 0:
                # Drop state immediately when finished
                self._prefix_state.pop(pid, None)
        return remaining_after, osl, iat

    async def _pick_worker(self, token_ids: list[int], prefix_id: str, reuse_budget: int, osl: str,
                           iat: str) -> tuple[int | None, str | None]:
        """Pick a worker via the router."""
        if not self.router_pick_client:
            return None, None

        req = RouterRequest(
            tokens=token_ids,
            prefix_id=prefix_id,
            reuse_budget=max(int(reuse_budget), 0),
            expected_osl=osl,
            interarrival=iat,
        )
        stream = await self.router_pick_client.generate(req.model_dump())

        worker_id: int | None = None
        decision_id: str | None = None
        async for chunk in stream:
            data = chunk.data()
            if "error" in data:
                logger.error("Router error: %s", data['error'])
                break
            wid = data.get("worker_id", -1)
            if wid == -1:
                break
            worker_id = int(wid)
            decision_id = data.get("decision_id")
            break

        if worker_id is None:
            logger.warning("Router stream ended without worker_id; falling back to engine load balancing.")
        return worker_id, decision_id

    async def _send_feedback_safely(self,
                                    decision_id: str | None,
                                    latency_ms: float,
                                    success: bool,
                                    tokens_in: int,
                                    tokens_out: int,
                                    finish_reason: str | None):
        if not decision_id or not self.router_feedback_client:
            return
        try:
            fb = RouterFeedbackRequest(
                decision_id=decision_id,
                latency_ms=float(latency_ms),
                success=bool(success),
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                finish_reason=finish_reason or "",
            )
            stream = await self.router_feedback_client.generate(fb.model_dump())
            async for _ in stream:
                pass
        except Exception:
            logger.exception("Failed to send router feedback")

    async def _stream_from_engine(
        self,
        token_ids: list[int],
        model: str,
        temperature: float,
        top_p: float,
        top_k: int,
        ignore_eos: bool,
        max_tokens: int,
        prefix_id: str,
        reuse_budget: int,
        osl: str,
        iat: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming generator: yields {'delta': str} tokens and finally {'finish_reason': <str>}."""
        worker_id, decision_id = await self._pick_worker(token_ids, prefix_id, reuse_budget, osl, iat)
        engine_request: dict[str, Any] = {
            "token_ids": token_ids,
            "sampling_options": {
                "temperature": temperature, "top_p": top_p, "top_k": top_k
            },
            "stop_conditions": {
                "max_tokens": max_tokens, "ignore_eos": ignore_eos
            },
            "model": model,
        }

        if worker_id is not None:
            stream = await self.engine_client.direct(engine_request, worker_id)
        else:
            stream = await self.engine_client.generate(engine_request)

        t0 = time.perf_counter()
        all_tokens: list[int] = []
        text_so_far: str = ""
        finish_reason: str | None = None

        try:
            async for chunk in stream:
                data = chunk.data()
                if "error" in data:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    await self._send_feedback_safely(decision_id,
                                                     latency_ms,
                                                     False,
                                                     len(token_ids),
                                                     len(all_tokens),
                                                     "error")
                    yield {"error": data["error"]}
                    return

                if "token_ids" in data and isinstance(data["token_ids"], list):
                    for tid in data["token_ids"]:
                        all_tokens.append(tid)
                        new_text = self.detokenize(all_tokens) if all_tokens else ""
                        if len(new_text) > len(text_so_far):
                            piece = new_text[len(text_so_far):]
                            text_so_far = new_text
                            if piece:
                                yield {"delta": piece}

                if "finish_reason" in data and data["finish_reason"] is not None:
                    finish_reason = data["finish_reason"]
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    await self._send_feedback_safely(decision_id,
                                                     latency_ms,
                                                     True,
                                                     len(token_ids),
                                                     len(all_tokens),
                                                     finish_reason)
                    # Persist per-request metrics to CSV (concurrency-safe)
                    await self._log_request_metrics(num_tokens=len(all_tokens), latency_ms=latency_ms)
                    yield {"finish_reason": finish_reason}
                    return

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            await self._send_feedback_safely(decision_id,
                                             latency_ms,
                                             False,
                                             len(token_ids),
                                             len(all_tokens),
                                             "exception")
            logger.exception("Engine stream exception")
            yield {"error": str(e)}
            return

    # ---- main generation ----
    async def generate(self, raw: dict[str, Any]):
        """Processor endpoint: always yields a stream of dicts."""
        chat_req = ChatCompletionRequest(**raw)
        logger.info("Chat completion request was %s with %d messages", chat_req.model, len(chat_req.messages))
        hints = chat_req.prefix_hints or PrefixHints(
            prefix_id=f"auto-{uuid.uuid4().hex}", total_requests=1, osl="MEDIUM", iat="MEDIUM")

        # Update prefix state and compute reuse_budget := remaining AFTER this request
        reuse_budget, osl, iat = await self._update_prefix_state(hints)

        # Build input text for the model
        messages = chat_req.messages.copy()
        text = self._render_prompt(messages)
        tokens = self.tokenize(text)

        # Stream from engine (frontend can aggregate if non-streaming)
        async for resp in self._stream_from_engine(tokens,
                                                   chat_req.model,
                                                   chat_req.temperature,
                                                   chat_req.top_p,
                                                   chat_req.top_k,
                                                   chat_req.ignore_eos,
                                                   chat_req.max_tokens,
                                                   hints.prefix_id,
                                                   reuse_budget,
                                                   osl,
                                                   iat):
            yield resp

    async def _log_request_metrics(self, *, num_tokens: int, latency_ms: float):
        """Append a CSV line with (num_tokens, latency_ms, latency_ms_per_token).
        Uses an asyncio lock for concurrency safety across concurrent requests.
        """
        # Guard against divide-by-zero
        denom = max(1, int(num_tokens))
        latency_per_token = float(latency_ms) / float(denom)
        try:
            async with self._metrics_lock:
                # Respect cap if configured
                if self._metrics_log_cap is not None and self._metrics_written_count >= self._metrics_log_cap:
                    return
                with open(self._metrics_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(num_tokens), f"{latency_ms:.3f}", f"{latency_per_token:.6f}"])
                    self._metrics_written_count += 1
        except Exception as e:
            logger.warning("Failed to write metrics CSV %s: %s", self._metrics_csv_path, e)


# -------------------------- worker entry point -------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    p.add_argument("--enable-router", action="store_true", default=True)
    return p.parse_args()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    args = parse_args()
    component = runtime.namespace("dynamo").component("processor")
    await component.create_service()

    handler = ProcessorRequestHandler(runtime, model_name=args.model, enable_router=args.enable_router)
    await handler.initialize()
    await component.endpoint("process").serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())  # pylint: disable=no-value-for-parameter
