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
"""Unit tests for ATIF multimodal content helpers and media type inference."""

from __future__ import annotations

import pydantic
import pytest

from nat.atif.content import SUPPORTED_IMAGE_MEDIA_TYPES
from nat.atif.content import ContentPart
from nat.atif.content import ImageSource
from nat.atif.content import infer_image_media_type
from nat.atif.content import normalize_content_part
from nat.atif.content import normalize_content_parts
from nat.atif.step import Step


def test_supported_image_media_types_set() -> None:
    expected = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    assert set(SUPPORTED_IMAGE_MEDIA_TYPES) == expected


@pytest.mark.parametrize(
    ("uri", "expected_mime"),
    [
        (
            ("data:image/png;base64,"
             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="),
            "image/png",
        ),
        ("data:image/jpeg;base64,/9j/4AAQSkZJRg==", "image/jpeg"),
        ("data:image/jpg;base64,/9j/4AAQSkZJRg==", "image/jpeg"),
        ("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7", "image/gif"),
        ("data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==", "image/webp"),
        ("data:image/png;charset=utf-8;base64,iVBOR...", "image/png"),
    ],
)
def test_infer_image_media_type_data_uris(uri: str, expected_mime: str) -> None:
    assert infer_image_media_type(uri) == expected_mime


@pytest.mark.parametrize(
    ("url_or_path", "expected_mime"),
    [
        ("https://example.com/assets/banner.png", "image/png"),
        ("https://example.com/photos/avatar.jpg", "image/jpeg"),
        ("https://example.com/photos/avatar.jpeg", "image/jpeg"),
        ("https://example.com/anim/loader.gif", "image/gif"),
        ("https://example.com/stickers/hero.webp", "image/webp"),
        ("http://localhost:8000/media/sample.JPG", "image/jpeg"),
        ("/var/data/uploads/chart.webp", "image/webp"),
        (r"C:\images\sample.gif", "image/gif"),
    ],
)
def test_infer_image_media_type_urls_and_paths(url_or_path: str, expected_mime: str) -> None:
    assert infer_image_media_type(url_or_path) == expected_mime


@pytest.mark.parametrize(
    ("url_with_query_or_frag", "expected_mime"),
    [
        ("https://example.com/assets/diagram.png?version=2&auth=token#section", "image/png"),
        ("http://localhost:8000/image.jpeg?w=100#view", "image/jpeg"),
        ("https://example.com/photo.webp?download=1", "image/webp"),
        ("https://example.com/anim.gif#view", "image/gif"),
        ("https://example.com/image.PNG?version=2&token=abc#view", "image/png"),
    ],
)
def test_infer_image_media_type_strips_queries_and_fragments_from_urls(
    url_with_query_or_frag: str,
    expected_mime: str,
) -> None:
    assert infer_image_media_type(url_with_query_or_frag) == expected_mime


@pytest.mark.parametrize(
    ("local_path", "expected_mime"),
    [
        ("/tmp/chart#1.png", "image/png"),
        ("local/path#foo.jpg", "image/jpeg"),
        (r"C:\images\photo?1.jpg", "image/jpeg"),
        ("relative/chart?name.webp", "image/webp"),
        ("/var/log/snapshot?2.gif", "image/gif"),
        (r"C:\data\report#final?v1.png", "image/png"),
    ],
)
def test_infer_image_media_type_preserves_hashes_and_questions_in_local_paths(
    local_path: str,
    expected_mime: str,
) -> None:
    assert infer_image_media_type(local_path) == expected_mime


@pytest.mark.parametrize(
    "invalid_input",
    [
        "https://example.com/document.pdf",
        "https://example.com/api/get_image",
        "https://example.com/image#fake.png",
        "https://example.com/get_image?file=test.png",
        "data:application/json;base64,eyJmb28iOiAiYmFyIn0=",
        "data:image/bmp;base64,Qk0=",
        "data:text/plain;charset=utf-8,hello",
    ],
)
def test_infer_image_media_type_invalid_raises(invalid_input: str) -> None:
    with pytest.raises(ValueError, match="supported image media_type"):
        infer_image_media_type(invalid_input)


@pytest.mark.parametrize(
    "empty_or_non_string",
    [
        "",
        "   ",
        None,
    ],
)
def test_infer_image_media_type_empty_or_non_string_raises(empty_or_non_string: str | None) -> None:
    with pytest.raises(ValueError, match="empty or non-string input"):
        infer_image_media_type(empty_or_non_string)  # type: ignore[arg-type]


def test_normalize_content_part_text() -> None:
    part = {"type": "text", "text": "Describe this architecture"}
    normalized = normalize_content_part(part)
    assert normalized == {"type": "text", "text": "Describe this architecture"}

    validated = ContentPart(**normalized)
    assert validated.type == "text"
    assert validated.text == "Describe this architecture"
    assert validated.source is None


def test_normalize_content_part_text_default_empty_string() -> None:
    # A text part missing the 'text' key is returned unchanged so ContentPart validation can reject it
    part = {"type": "text"}
    normalized = normalize_content_part(part)
    assert normalized == part
    with pytest.raises(pydantic.ValidationError, match="'text' field is required when type='text'"):
        ContentPart(**normalized)

    # A text part with explicit text=None is also returned unchanged
    part_none = {"type": "text", "text": None}
    normalized_none = normalize_content_part(part_none)
    assert normalized_none == part_none
    with pytest.raises(pydantic.ValidationError, match="'text' field is required when type='text'"):
        ContentPart(**normalized_none)

    # Explicit empty string is preserved and accepted
    part_empty = {"type": "text", "text": ""}
    normalized_empty = normalize_content_part(part_empty)
    assert normalized_empty == {"type": "text", "text": ""}
    validated = ContentPart(**normalized_empty)
    assert validated.text == ""


def test_normalize_content_part_openai_image_url_dict() -> None:
    part = {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/diagram.png?size=large",
            "detail": "high",
        },
    }
    normalized = normalize_content_part(part)
    assert normalized == {
        "type": "image",
        "source": {
            "media_type": "image/png",
            "path": "https://example.com/diagram.png?size=large",
        },
    }

    validated = ContentPart(**normalized)
    assert validated.type == "image"
    assert isinstance(validated.source, ImageSource)
    assert validated.source.media_type == "image/png"
    assert validated.source.path == "https://example.com/diagram.png?size=large"
    assert validated.text is None


def test_normalize_content_part_openai_image_url_string() -> None:
    part = {
        "type": "image_url",
        "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
    }
    normalized = normalize_content_part(part)
    assert normalized == {
        "type": "image",
        "source": {
            "media_type": "image/jpeg",
            "path": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        },
    }
    validated = ContentPart(**normalized)
    assert validated.source.media_type == "image/jpeg"


def test_normalize_content_part_existing_atif_image_passthrough() -> None:
    part = {
        "type": "image",
        "source": {
            "media_type": "image/webp",
            "path": "https://example.com/photo.webp",
        },
    }
    normalized = normalize_content_part(part)
    assert normalized == part
    validated = ContentPart(**normalized)
    assert validated.source.media_type == "image/webp"


def test_normalize_content_part_image_with_image_url_key() -> None:
    part = {
        "type": "image",
        "image_url": {
            "url": "https://example.com/photo.gif"
        },
    }
    normalized = normalize_content_part(part)
    assert normalized == {
        "type": "image",
        "source": {
            "media_type": "image/gif",
            "path": "https://example.com/photo.gif",
        },
    }


def test_normalize_content_part_missing_or_invalid_image_url_raises() -> None:
    with pytest.raises(ValueError, match="Missing or invalid 'url' in image_url content part"):
        normalize_content_part({"type": "image_url", "image_url": {}})

    with pytest.raises(ValueError, match="Missing or invalid 'url' in image_url content part"):
        normalize_content_part({"type": "image_url", "image_url": 123})

    with pytest.raises(ValueError, match="Missing or invalid 'url' in image_url content part"):
        normalize_content_part({"type": "image_url"})


def test_normalize_content_part_unsupported_image_url_raises() -> None:
    with pytest.raises(ValueError, match="Cannot infer supported image media_type"):
        normalize_content_part({"type": "image_url", "image_url": "https://example.com/no_extension_id"})


def test_normalize_content_part_non_dict_passthrough() -> None:
    assert normalize_content_part("lost prompt") == "lost prompt"
    assert normalize_content_part(12345) == 12345


def test_normalize_content_parts_preserves_non_dict_and_invalid_entries() -> None:
    # All invalid entries must not be silently dropped
    all_invalid = ["lost prompt"]
    assert normalize_content_parts(all_invalid) == ["lost prompt"]

    with pytest.raises(pydantic.ValidationError):
        Step(step_id=1, source="user", message=all_invalid)

    # Mixed list must not silently drop non-dict entries
    mixed = ["lost prompt", {"type": "text", "text": "Valid question"}]
    assert normalize_content_parts(mixed) == ["lost prompt", {"type": "text", "text": "Valid question"}]

    with pytest.raises(pydantic.ValidationError):
        Step(step_id=2, source="user", message=mixed)

    # All non-string / non-dict entries
    all_ints = [123, 456]
    assert normalize_content_parts(all_ints) == [123, 456]

    with pytest.raises(pydantic.ValidationError):
        Step(step_id=3, source="user", message=all_ints)

    # Valid message case isolates content-part validation from Step.step_id validation
    valid_step = Step(step_id=1, source="user", message=[{"type": "text", "text": "Valid question"}])
    assert valid_step.step_id == 1


def test_normalize_content_parts_list() -> None:
    parts = [
        {
            "type": "text", "text": "Question 1"
        },
        {
            "type": "image_url", "image_url": {
                "url": "data:image/png;base64,iVBOR..."
            }
        },
        {
            "type": "text", "text": "Question 2"
        },
    ]
    normalized = normalize_content_parts(parts)
    assert len(normalized) == 3
    assert normalized[0] == {"type": "text", "text": "Question 1"}
    assert normalized[1] == {
        "type": "image",
        "source": {
            "media_type": "image/png",
            "path": "data:image/png;base64,iVBOR...",
        },
    }
    assert normalized[2] == {"type": "text", "text": "Question 2"}

    # Validates against list[ContentPart]
    validated = [ContentPart(**p) for p in normalized]
    assert len(validated) == 3


def test_normalize_content_parts_empty_and_invalid() -> None:
    assert normalize_content_parts([]) == []
    assert normalize_content_parts(None) == []  # type: ignore[arg-type]
