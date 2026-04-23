from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any

from azure.storage.blob import BlobServiceClient
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("TRANSCRIPT_MCP_PORT", "8111"))

mcp = FastMCP("transcripts", host="0.0.0.0", port=PORT)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "transcript-mcp"})


def _store():
    container_name = os.environ.get("TRANSCRIPT_BLOB_CONTAINER", "call-transcripts")
    account_name = os.environ.get("TRANSCRIPT_BLOB_ACCOUNT_NAME", "").strip()
    account_key = os.environ.get("TRANSCRIPT_BLOB_ACCOUNT_KEY", "").strip()
    if account_name and account_key:
        service = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=account_key,
        )
    else:
        connection_string = os.environ.get("TRANSCRIPT_BLOB_CONNECTION_STRING", "").strip()
        if not connection_string:
            raise RuntimeError(
                "Set TRANSCRIPT_BLOB_CONNECTION_STRING or "
                "TRANSCRIPT_BLOB_ACCOUNT_NAME + TRANSCRIPT_BLOB_ACCOUNT_KEY."
            )
        service = BlobServiceClient.from_connection_string(connection_string)
    return service.get_container_client(container_name)


def _iter_metadata_blobs() -> list[dict[str, Any]]:
    container = _store()
    transcripts: list[dict[str, Any]] = []
    for blob in container.list_blobs():
        if not blob.name.endswith("/metadata.json"):
            continue
        metadata = json.loads(container.download_blob(blob.name).readall().decode("utf-8"))
        metadata["_metadata_blob"] = blob.name
        transcripts.append(metadata)
    return transcripts


def _download_text_blob(blob_name: str) -> str:
    container = _store()
    return container.download_blob(blob_name).readall().decode("utf-8")


def _speaker_segments_from_vtt(transcript_vtt: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"<v\s+([^>]+)>(.*?)</v>")
    segments: list[tuple[str, str]] = []
    for line in transcript_vtt.splitlines():
        if "-->" in line or not line.strip():
            continue
        match = pattern.search(line)
        if match:
            speaker = match.group(1).strip()
            text = re.sub(r"\s+", " ", match.group(2)).strip()
            if text:
                segments.append((speaker, text))
    return segments


def _format_metadata(metadata: dict[str, Any]) -> str:
    organizer = metadata.get("meetingOrganizer", {}).get("user", {})
    return (
        f"Transcript [{metadata.get('id', '')}]\n"
        f"  Source: {metadata.get('sourceKind', 'unknown')}\n"
        f"  Meeting ID: {metadata.get('meetingId', '(none)')}\n"
        f"  Call ID: {metadata.get('callId', '(none)')}\n"
        f"  Created: {metadata.get('createdDateTime', '(unknown)')}\n"
        f"  Ended: {metadata.get('endDateTime', '(unknown)')}\n"
        f"  Organizer: {organizer.get('id', '(unknown)')}\n"
        f"  Text blob: {metadata.get('transcriptTextBlob', '(missing)')}\n"
    )


@mcp.tool()
async def list_transcripts(
    date_prefix: str = "",
    source_kind: str = "",
    meeting_id: str = "",
    call_id: str = "",
    max_results: int = 20,
) -> str:
    """List transcripts saved in Azure Blob Storage.

    Args:
        date_prefix: Optional YYYY-MM or YYYY-MM-DD prefix to filter createdDateTime.
        source_kind: Optional source filter: onlineMeetings or adhocCalls.
        meeting_id: Optional exact meeting id filter.
        call_id: Optional exact call id filter.
        max_results: Maximum number of transcripts to return.

    Returns:
        A formatted list of matching transcript artifacts.
    """
    results: list[dict[str, Any]] = []
    for metadata in _iter_metadata_blobs():
        if date_prefix and not str(metadata.get("createdDateTime", "")).startswith(date_prefix):
            continue
        if source_kind and metadata.get("sourceKind") != source_kind:
            continue
        if meeting_id and metadata.get("meetingId") != meeting_id:
            continue
        if call_id and metadata.get("callId") != call_id:
            continue
        results.append(metadata)

    results.sort(key=lambda item: item.get("createdDateTime", ""), reverse=True)
    results = results[: min(max_results, 50)]
    if not results:
        return "No transcripts found."

    lines = [f"Found {len(results)} transcript(s):\n"]
    for metadata in results:
        lines.append(_format_metadata(metadata))
    return "\n".join(lines)


@mcp.tool()
async def get_transcript(transcript_id: str, include_vtt: bool = False) -> str:
    """Get transcript metadata and stored text for a transcript id.

    Args:
        transcript_id: The saved transcript id.
        include_vtt: Whether to include the raw VTT content.

    Returns:
        Transcript metadata and stored text content.
    """
    for metadata in _iter_metadata_blobs():
        if metadata.get("id") != transcript_id:
            continue

        text_blob = metadata.get("transcriptTextBlob")
        if not text_blob:
            raise RuntimeError(f"Transcript {transcript_id} is missing transcriptTextBlob metadata")
        transcript_text = _download_text_blob(text_blob)
        response = f"{_format_metadata(metadata)}\nTranscript Text:\n{transcript_text}"

        if include_vtt:
            vtt_blob = metadata.get("transcriptVttBlob")
            if vtt_blob:
                transcript_vtt = _download_text_blob(vtt_blob)
                response += f"\n\nRaw VTT:\n{transcript_vtt}"
        return response

    raise RuntimeError(f"Transcript {transcript_id} not found")


@mcp.tool()
async def summarize_transcript(transcript_id: str, preview_lines: int = 12) -> str:
    """Produce a structural summary of a stored transcript.

    This is intentionally heuristic rather than LLM-based. It provides metadata,
    speaker counts, rough speaking turn counts, and a short preview to help the
    calling agent form a grounded response.

    Args:
        transcript_id: The saved transcript id.
        preview_lines: Number of transcript lines to include as preview.

    Returns:
        A compact, tool-grounded summary of the transcript.
    """
    for metadata in _iter_metadata_blobs():
        if metadata.get("id") != transcript_id:
            continue

        vtt_blob = metadata.get("transcriptVttBlob")
        text_blob = metadata.get("transcriptTextBlob")
        if not vtt_blob or not text_blob:
            raise RuntimeError(f"Transcript {transcript_id} is missing stored blobs")

        transcript_vtt = _download_text_blob(vtt_blob)
        transcript_text = _download_text_blob(text_blob)
        segments = _speaker_segments_from_vtt(transcript_vtt)
        speaker_counts = Counter(speaker for speaker, _ in segments)
        preview = "\n".join(transcript_text.splitlines()[: max(preview_lines, 1)])

        lines = [_format_metadata(metadata)]
        lines.append(f"Utterance count: {len(segments)}")
        if speaker_counts:
            lines.append("Speakers:")
            for speaker, count in speaker_counts.most_common():
                lines.append(f"  {speaker}: {count} turn(s)")
        else:
            lines.append("Speakers: not detected from VTT markup")

        action_candidates = []
        for line in transcript_text.splitlines():
            lower = line.lower()
            if any(marker in lower for marker in ("follow up", "action item", "next step", "we should", "i'll", "i will")):
                action_candidates.append(line)
        if action_candidates:
            lines.append("\nPotential action items:")
            lines.extend(f"  - {item}" for item in action_candidates[:8])

        lines.append("\nPreview:")
        lines.append(preview)
        return "\n".join(lines)

    raise RuntimeError(f"Transcript {transcript_id} not found")


if __name__ == "__main__":
    logger.info("Starting transcript MCP server on port %d", PORT)
    mcp.run(transport="streamable-http")
