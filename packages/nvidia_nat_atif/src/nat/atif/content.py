# SPDX-FileCopyrightText: Copyright (c) 2025, Harbor Framework Contributors (https://github.com/harbor-framework/harbor)
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
"""Content models for multimodal ATIF trajectories (ATIF v1.6+)."""

from __future__ import annotations

from typing import Any
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class ImageSource(BaseModel):
    """Image source specification for images stored as files or at remote URLs."""

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        ...,
        description="MIME type of the image",
    )
    path: str = Field(
        ...,
        description="Location of the image. Can be a relative or absolute file path, or a URL.",
    )

    model_config = ConfigDict(extra="forbid")


class ContentPart(BaseModel):
    """A single content part within a multimodal message.

    Used when a message or observation contains mixed content types (text and images).
    For text-only content, a plain string can still be used instead of a ContentPart array.
    """

    type: Literal["text", "image"] = Field(
        ...,
        description="The type of content",
    )
    text: str | None = Field(
        default=None,
        description="Text content. Required when type='text'.",
    )
    source: ImageSource | None = Field(
        default=None,
        description="Image source (file reference). Required when type='image'.",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_content_type(self) -> ContentPart:
        """Validate that the correct fields are present for each content type."""
        if self.type == "text":
            if self.text is None:
                raise ValueError("'text' field is required when type='text'")
            if self.source is not None:
                raise ValueError("'source' field is not allowed when type='text'")
        elif self.type == "image":
            if self.source is None:
                raise ValueError("'source' field is required when type='image'")
            if self.text is not None:
                raise ValueError("'text' field is not allowed when type='image'")
        return self


SUPPORTED_IMAGE_MEDIA_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
})


def infer_image_media_type(url_or_path: str) -> str:
    """Infer a supported image media type from a data URI, URL, or file path.

    Returns one of ``"image/jpeg"``, ``"image/png"``, ``"image/gif"``, or ``"image/webp"``.
    Normalizes ``"image/jpg"`` to ``"image/jpeg"``.

    For URLs (starting with ``http://``, ``https://``, or ``ftp://``), query strings
    and fragments are stripped before determining the media type. For local file paths,
    the full filename is evaluated so characters such as ``#`` or ``?`` (valid in local
    filenames) are preserved.

    Raises:
        ValueError: if the media type cannot be inferred or is unsupported.
    """
    if not isinstance(url_or_path, str) or not url_or_path.strip():
        raise ValueError(f"Cannot infer image media_type from empty or non-string input: {url_or_path!r}")

    trimmed = url_or_path.strip()
    if trimmed.startswith("data:"):
        header = trimmed[5:].split(",", 1)[0]
        mime = header.split(";", 1)[0].strip().lower()
        if mime == "image/jpg":
            mime = "image/jpeg"
        if mime in SUPPORTED_IMAGE_MEDIA_TYPES:
            return mime
        supported_str = ", ".join(sorted(SUPPORTED_IMAGE_MEDIA_TYPES))
        raise ValueError(f"Unsupported image media_type '{mime}' in data URL. Supported types: {supported_str}")

    if trimmed.startswith(("http://", "https://", "ftp://")):
        path_to_check = urlsplit(trimmed).path.lower()
    else:
        path_to_check = trimmed.lower()

    if path_to_check.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if path_to_check.endswith(".png"):
        return "image/png"
    if path_to_check.endswith(".gif"):
        return "image/gif"
    if path_to_check.endswith(".webp"):
        return "image/webp"

    supported_str = ", ".join(sorted(SUPPORTED_IMAGE_MEDIA_TYPES))
    raise ValueError(f"Cannot infer supported image media_type from URL or path '{url_or_path}'. "
                     f"Supported types: {supported_str}")


def normalize_content_part(part: Any) -> Any:
    """Normalize a raw or vendor-specific content part to an ATIF ContentPart dict.

    Handles:

    * Text parts (``{"type": "text", "text": "..."}``).
    * OpenAI image parts (``{"type": "image_url", "image_url": ...}``) mapped to ATIF
      image parts with inferred media_type.
    * Standard ATIF image parts (``{"type": "image", "source": ...}``).

    Leaves non-dict or unhandled parts intact so downstream ContentPart validation
    or callers can reject them appropriately without silently dropping data.
    """
    if not isinstance(part, dict):
        return part

    part_type = part.get("type")
    if part_type == "text":
        text_val = part.get("text")
        if text_val is None:
            return part
        return {"type": "text", "text": text_val}

    if part_type == "image_url":
        image_url_val = part.get("image_url")
        if isinstance(image_url_val, dict):
            url = image_url_val.get("url")
        elif isinstance(image_url_val, str):
            url = image_url_val
        else:
            url = None

        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"Missing or invalid 'url' in image_url content part: {part!r}")

        media_type = infer_image_media_type(url)
        return {
            "type": "image",
            "source": {
                "media_type": media_type,
                "path": url,
            },
        }

    if part_type == "image":
        if "source" in part and isinstance(part["source"], dict):
            return part
        if "image_url" in part:
            return normalize_content_part({**part, "type": "image_url"})
        return part

    return part


def normalize_content_parts(parts: list[Any]) -> list[Any]:
    """Normalize a list of content parts to ATIF-compliant ContentPart dicts.

    Leaves invalid entries in the returned list so that ContentPart validation
    rejects them rather than silently dropping content or hiding missing prompt data.
    """
    if not isinstance(parts, list):
        return []
    return [normalize_content_part(p) for p in parts]
