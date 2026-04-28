import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

def _parse_sse_block(block: bytes) -> tuple[str | None, dict | None]:
    """Parse one SSE event block (bytes between '\\n\\n' separators).

    Returns (event_type, payload). `payload` is the JSON-decoded `data:` content, or None
    if the block had no data lines (e.g. a bare `event: done`).
    """
    event_type: str | None = None
    data_lines: list[str] = []
    for raw_line in block.decode("utf-8", errors="replace").splitlines():
        line = raw_line.lstrip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    if not data_lines:
        return event_type, None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        logger.debug("tavily research: failed to parse SSE data block: %r", block)
        return event_type, None
    return event_type, payload


def _finalize_research(
    *,
    content_parts: list[str],
    structured_content: dict | None,
    sources: list[dict],
    model: str | None,
    trace: list[dict],
    include_trace: bool,
) -> dict:
    content: str | dict = structured_content if structured_content is not None else "".join(content_parts)
    result: dict = {"content": content, "sources": sources, "model": model}
    if include_trace:
        result["trace"] = trace
    return result


async def _accumulate_research_stream(
    chunks: AsyncIterator[bytes],
    *,
    include_trace: bool,
) -> dict:
    """Consume Tavily's /research SSE stream and return an aggregated result.

    Accumulates `delta.content` string chunks (or captures a single dict when `output_schema`
    was supplied), captures the final `delta.sources` list, and optionally retains the
    sequence of `delta.tool_calls` events as a debug trace. Stops on `event: done`.
    """
    buffer = b""
    content_parts: list[str] = []
    structured_content: dict | None = None
    sources: list[dict] = []
    trace: list[dict] = []
    model: str | None = None

    async for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk
        while b"\n\n" in buffer:
            block, buffer = buffer.split(b"\n\n", 1)
            event_type, payload = _parse_sse_block(block)
            if event_type == "done":
                return _finalize_research(
                    content_parts=content_parts,
                    structured_content=structured_content,
                    sources=sources,
                    model=model,
                    trace=trace,
                    include_trace=include_trace,
                )
            if payload is None:
                continue
            if payload.get("object") == "error":
                raise RuntimeError(payload.get("error") or "tavily research stream error")
            if model is None:
                model = payload.get("model")
            delta = (payload.get("choices") or [{}])[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                content_parts.append(content)
            elif isinstance(content, dict):
                structured_content = content
            if "sources" in delta and "tool_calls" not in delta:
                sources = delta["sources"] or []
            if include_trace and "tool_calls" in delta:
                trace.append(delta["tool_calls"])

    return _finalize_research(
        content_parts=content_parts,
        structured_content=structured_content,
        sources=sources,
        model=model,
        trace=trace,
        include_trace=include_trace,
    )
