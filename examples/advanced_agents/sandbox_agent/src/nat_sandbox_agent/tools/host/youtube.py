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

"""Host-side YouTube transcript tool.

⚠️  LEGAL WARNING ⚠️
This tool uses unofficial methods to access YouTube transcripts.
Use for educational and research purposes ONLY.
Commercial use may violate YouTube's Terms of Service.
NVIDIA is not responsible for any consequences of using this tool.

For production use, please use the official YouTube Data API v3:
https://developers.google.com/youtube/v3

This tool runs on the host machine (not in the sandbox) for:
- Lower latency (no Docker exec overhead)
- No sandbox dependencies required
"""

import asyncio
import logging
import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from nat_sandbox_agent.tools.common import DEFAULT_MAX_OUTPUT_CHARS

logger = logging.getLogger(__name__)


class YouTubeTranscriptInput(BaseModel):
    """Input schema for YouTube transcript extraction."""

    url: str = Field(description="YouTube video URL or video ID.")
    language: str = Field(
        default="en",
        description="Preferred language for transcript (e.g., 'en', 'es', 'fr').",
    )


class HostYouTubeTool:
    """YouTube transcript tool that runs on host (not in sandbox)."""

    def __init__(self, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS):
        """Initialize the YouTube tool.

        Args:
            max_output_chars: Maximum characters for transcript output.
        """
        self._max_output_chars = max_output_chars

    def _extract_video_id(self, url: str) -> str | None:
        """Extract video ID from YouTube URL.

        Args:
            url: YouTube URL or video ID.

        Returns:
            Video ID or None if not found.
        """
        patterns = [
            r"(?:v=|/v/|youtu\.be/|/embed/)([^&?/]+)",
            r"^([a-zA-Z0-9_-]{11})$",  # Direct video ID
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    async def get_transcript(
        self, url: str, language: str = "en"
    ) -> dict[str, Any]:
        """Get transcript from a YouTube video.

        Args:
            url: YouTube video URL or video ID.
            language: Preferred language for transcript.

        Returns:
            Dict with status and transcript text.
        """
        logger.info(f"Getting YouTube transcript: {url}")

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import NoTranscriptFound
            from youtube_transcript_api._errors import TranscriptsDisabled

            video_id = self._extract_video_id(url)
            if not video_id:
                return {
                    "status": "error",
                    "error": "Could not extract video ID from URL",
                }

            # Try to get transcript in preferred language
            # Use asyncio.to_thread to avoid blocking the event loop
            transcript = None
            try:
                transcript_list = await asyncio.to_thread(
                    YouTubeTranscriptApi.list_transcripts, video_id
                )

                try:
                    transcript = transcript_list.find_transcript([language])
                except NoTranscriptFound:
                    # Fall back to auto-generated or any available
                    try:
                        transcript = transcript_list.find_generated_transcript(
                            [language]
                        )
                    except NoTranscriptFound:
                        # Get first available
                        for t in transcript_list:
                            transcript = t
                            break

            except TranscriptsDisabled:
                return {
                    "status": "error",
                    "error": "Transcripts are disabled for this video",
                }

            if transcript:
                data = await asyncio.to_thread(transcript.fetch)
                full_text = " ".join([entry["text"] for entry in data])

                # Create timestamped version
                timestamped = []
                for entry in data:
                    start = int(entry["start"])
                    mins, secs = divmod(start, 60)
                    timestamped.append(f"[{mins:02d}:{secs:02d}] {entry['text']}")

                # Calculate duration
                duration = 0
                if data:
                    last_entry = data[-1]
                    duration = int(
                        last_entry["start"] + last_entry.get("duration", 0)
                    )

                logger.info(f"Got transcript for video {video_id}")

                return {
                    "status": "success",
                    "video_id": video_id,
                    "language": transcript.language,
                    "transcript": full_text[: self._max_output_chars],
                    "timestamped": "\n".join(timestamped[:500]),
                    "duration_seconds": duration,
                }
            else:
                return {
                    "status": "error",
                    "error": "No transcript available for this video",
                }

        except ImportError:
            return {
                "status": "error",
                "error": "youtube-transcript-api not installed",
            }
        except Exception as e:
            logger.exception("YouTube transcript error")
            return {
                "status": "error",
                "error": str(e),
            }


def create_youtube_tool(
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> StructuredTool:
    """Create the YouTube transcript tool.

    Args:
        max_output_chars: Maximum characters for transcript output.

    Returns:
        LangChain StructuredTool instance.
    """
    tool = HostYouTubeTool(max_output_chars=max_output_chars)

    return StructuredTool.from_function(
        coroutine=tool.get_transcript,
        name="youtube_transcript",
        description=(
            "Get the transcript (subtitles/captions) from a YouTube video. "
            "Returns the full text transcript and a timestamped version. "
            "Use this to analyze YouTube video content without watching it."
        ),
        args_schema=YouTubeTranscriptInput,
    )
