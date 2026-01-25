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
"""Tests for host-side tools (web_search, youtube_transcript)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat_sandbox_agent.tools.host.web_search import HostWebSearchTool
from nat_sandbox_agent.tools.host.web_search import create_web_search_tool
from nat_sandbox_agent.tools.host.youtube import HostYouTubeTool
from nat_sandbox_agent.tools.host.youtube import create_youtube_tool


class TestHostWebSearchTool:
    """Tests for HostWebSearchTool."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        tool = HostWebSearchTool(api_key="test-key")

        assert tool._api_key == "test-key"
        assert tool._client is None

    def test_init_uses_env_var(self, monkeypatch):
        """Test that API key is read from environment."""
        monkeypatch.setenv("TAVILY_API_KEY", "env-api-key")
        tool = HostWebSearchTool()

        assert tool._api_key == "env-api-key"

    def test_get_client_raises_without_api_key(self):
        """Test that getting client without API key raises error."""
        tool = HostWebSearchTool(api_key=None)
        tool._api_key = None  # Ensure no key

        with pytest.raises(ValueError, match="TAVILY_API_KEY not set"):
            tool._get_client()

    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful web search."""
        tool = HostWebSearchTool(api_key="test-key")

        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value={
            "results": [{
                "title": "Test Result",
                "url": "https://example.com",
                "content": "This is a test snippet",
                "score": 0.95,
            }],
            "answer": "The answer is 42",
        })
        tool._client = mock_client

        result = await tool.search("test query", num_results=5)

        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Result"
        assert result["results"][0]["url"] == "https://example.com"
        assert result["results"][0]["snippet"] == "This is a test snippet"
        assert result["answer"] == "The answer is 42"

        mock_client.search.assert_called_once_with(
            query="test query",
            max_results=5,
            include_answer=True,
        )

    @pytest.mark.asyncio
    async def test_search_limits_max_results(self):
        """Test that max_results is capped at 10."""
        tool = HostWebSearchTool(api_key="test-key")

        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value={"results": [], "answer": None})
        tool._client = mock_client

        await tool.search("test", num_results=15)

        mock_client.search.assert_called_once_with(
            query="test",
            max_results=10,  # Should be capped
            include_answer=True,
        )

    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test error handling in search."""
        tool = HostWebSearchTool(api_key="test-key")

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=Exception("API error"))
        tool._client = mock_client

        result = await tool.search("test query")

        assert result["status"] == "error"
        assert "API error" in result["error"]


class TestCreateWebSearchTool:
    """Tests for create_web_search_tool function."""

    def test_creates_structured_tool(self):
        """Test that function creates a StructuredTool."""
        tool = create_web_search_tool(api_key="test-key")

        assert tool.name == "web_search"
        assert "Search the web" in tool.description

    def test_tool_has_correct_schema(self):
        """Test that tool has correct input schema."""
        tool = create_web_search_tool(api_key="test-key")

        # Check that the tool has expected input fields
        schema = tool.args_schema.schema()
        assert "query" in schema["properties"]
        assert "num_results" in schema["properties"]


class TestHostYouTubeTool:
    """Tests for HostYouTubeTool."""

    def test_init_with_custom_max_chars(self):
        """Test initialization with custom max output chars."""
        tool = HostYouTubeTool(max_output_chars=5000)

        assert tool._max_output_chars == 5000

    @pytest.mark.parametrize(
        "url,expected_id",
        [
            # Standard watch URLs
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # With additional parameters
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLtest", "dQw4w9WgXcQ"),
            # Short URLs
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Embed URLs
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Direct video ID
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ],
    )
    def test_extract_video_id_various_formats(self, url, expected_id):
        """Test video ID extraction from various URL formats."""
        tool = HostYouTubeTool()

        video_id = tool._extract_video_id(url)

        assert video_id == expected_id

    def test_extract_video_id_invalid_url(self):
        """Test that invalid URL returns None."""
        tool = HostYouTubeTool()

        video_id = tool._extract_video_id("https://example.com/not-youtube")

        assert video_id is None

    @pytest.mark.asyncio
    async def test_get_transcript_invalid_url(self):
        """Test error handling for invalid video URL."""
        # Mock the import to ensure consistent behavior
        mock_api = MagicMock()
        mock_errors = MagicMock()
        mock_errors.NoTranscriptFound = Exception
        mock_errors.TranscriptsDisabled = Exception

        with patch.dict(
                "sys.modules",
            {
                "youtube_transcript_api": mock_api,
                "youtube_transcript_api._errors": mock_errors,
            },
        ):
            tool = HostYouTubeTool()
            result = await tool.get_transcript("not-a-valid-url-at-all")

        assert result["status"] == "error"
        assert "Could not extract video ID" in result["error"]

    @pytest.mark.asyncio
    async def test_get_transcript_success(self):
        """Test successful transcript retrieval with mocked API."""
        mock_transcript_data = [
            {
                "text": "Hello world", "start": 0.0, "duration": 2.0
            },
            {
                "text": "This is a test", "start": 2.0, "duration": 3.0
            },
            {
                "text": "Video transcript", "start": 5.0, "duration": 2.0
            },
        ]

        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = mock_transcript_data
        mock_transcript.language = "en"

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_transcript.return_value = mock_transcript

        mock_api = MagicMock()
        mock_api.YouTubeTranscriptApi.list_transcripts.return_value = mock_transcript_list

        mock_errors = MagicMock()
        mock_errors.NoTranscriptFound = type("NoTranscriptFound", (Exception, ), {})
        mock_errors.TranscriptsDisabled = type("TranscriptsDisabled", (Exception, ), {})

        with patch.dict(
                "sys.modules",
            {
                "youtube_transcript_api": mock_api,
                "youtube_transcript_api._errors": mock_errors,
            },
        ):
            tool = HostYouTubeTool(max_output_chars=1000)
            result = await tool.get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert result["status"] == "success"
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["language"] == "en"
        assert "Hello world" in result["transcript"]
        assert "This is a test" in result["transcript"]
        assert result["duration_seconds"] == 7

    @pytest.mark.asyncio
    async def test_get_transcript_disabled(self):
        """Test handling when transcripts are disabled."""

        # Create a proper exception class
        class TranscriptsDisabled(Exception):

            def __init__(self, video_id):
                self.video_id = video_id
                super().__init__(f"Transcripts disabled for {video_id}")

        mock_api = MagicMock()
        mock_api.YouTubeTranscriptApi.list_transcripts.side_effect = TranscriptsDisabled("test-video-id")

        mock_errors = MagicMock()
        mock_errors.NoTranscriptFound = type("NoTranscriptFound", (Exception, ), {})
        mock_errors.TranscriptsDisabled = TranscriptsDisabled

        with patch.dict(
                "sys.modules",
            {
                "youtube_transcript_api": mock_api,
                "youtube_transcript_api._errors": mock_errors,
            },
        ):
            tool = HostYouTubeTool()
            result = await tool.get_transcript("dQw4w9WgXcQ")

        assert result["status"] == "error"
        assert "disabled" in result["error"].lower()


class TestCreateYouTubeTool:
    """Tests for create_youtube_tool function."""

    def test_creates_structured_tool(self):
        """Test that function creates a StructuredTool."""
        tool = create_youtube_tool()

        assert tool.name == "youtube_transcript"
        assert "transcript" in tool.description.lower()

    def test_tool_has_correct_schema(self):
        """Test that tool has correct input schema."""
        tool = create_youtube_tool()

        schema = tool.args_schema.schema()
        assert "url" in schema["properties"]
        assert "language" in schema["properties"]

    def test_accepts_custom_max_chars(self):
        """Test that max_output_chars parameter is accepted."""
        # Should not raise
        tool = create_youtube_tool(max_output_chars=5000)
        assert tool is not None
