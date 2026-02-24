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
"""Skill definition conforming to the Agent Skills specification (https://agentskills.io)."""

import base64
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

# ---------------------------------------------------------------------------
# Name validation regex (spec: 1-64 chars, lowercase a-z, digits, hyphens,
# no leading/trailing/consecutive hyphens)
# ---------------------------------------------------------------------------
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$")
_CONSECUTIVE_HYPHENS_RE = re.compile(r"--")


class Skill(BaseModel):
    """An Agent Skill per the `agentskills.io <https://agentskills.io/specification>`_ spec.

    Required frontmatter
    --------------------
    name : str
        1-64 chars, lowercase alphanumeric + hyphens. No leading/trailing/
        consecutive hyphens. Must match the parent directory name when loaded.
    description : str
        1-1024 chars. Describes what the skill does and when to use it.

    Optional frontmatter
    --------------------
    license, compatibility, allowed_tools, metadata

    Body content
    ------------
    content : str
        Markdown body with skill instructions (loaded on activation).
    """

    model_config = ConfigDict(frozen=True)

    # Required frontmatter
    name: str
    description: str = Field(min_length=1, max_length=1024)

    # Optional frontmatter
    license: str | None = None
    compatibility: str | None = Field(default=None, max_length=500)
    allowed_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    # Body content
    content: str = ""

    # In-memory resource storage (keys = relative paths, values = raw bytes)
    resources: dict[str, bytes] = Field(default_factory=dict, repr=False, exclude=True)

    # Runtime context (not serialised to SKILL.md)
    source_dir: Path | None = Field(default=None, repr=False, exclude=True)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("name is required")
        if len(v) > 64:
            raise ValueError(f"name must be <= 64 chars, got {len(v)}")
        if not _NAME_RE.match(v):
            raise ValueError(f"name {v!r} must be lowercase alphanumeric + hyphens, "
                             "no leading/trailing hyphens")
        if _CONSECUTIVE_HYPHENS_RE.search(v):
            raise ValueError(f"name {v!r} must not contain consecutive hyphens")
        return v

    @model_validator(mode="after")
    def _validate_source_dir_match(self) -> "Skill":
        if self.source_dir and self.source_dir.name != self.name:
            raise ValueError(f"name {self.name!r} must match parent directory {self.source_dir.name!r}")
        return self

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, skill_dir: str | Path) -> "Skill":
        """Load a skill from a directory containing ``SKILL.md``.

        Parses YAML frontmatter (delimited by ``---``) and the remaining
        markdown body.  Also reads files from ``scripts/``, ``references/``,
        and ``assets/`` subdirectories into ``resources``.
        """
        skill_dir = Path(skill_dir)
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")

        raw = skill_md.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw)

        # allowed-tools comes as a space-delimited string in the spec
        allowed_tools_raw = frontmatter.get("allowed-tools", "")
        allowed_tools = allowed_tools_raw.split() if isinstance(allowed_tools_raw, str) and allowed_tools_raw else []

        # Capture resource files
        resources: dict[str, bytes] = {}
        for subdir in ("scripts", "references", "assets"):
            sub_path = skill_dir / subdir
            if sub_path.is_dir():
                for file_path in sorted(sub_path.rglob("*")):
                    if file_path.is_file():
                        rel = str(file_path.relative_to(skill_dir))
                        resources[rel] = file_path.read_bytes()

        return cls(
            name=frontmatter.get("name", skill_dir.name),
            description=frontmatter.get("description", ""),
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            allowed_tools=allowed_tools,
            metadata=frontmatter.get("metadata") or {},
            content=body,
            resources=resources,
            source_dir=skill_dir.resolve(),
        )

    # ------------------------------------------------------------------
    # Progressive disclosure helpers
    # ------------------------------------------------------------------

    def to_metadata_prompt(self) -> str:
        """Compact metadata for system prompt injection (~50-100 tokens)."""
        return f"{self.name}: {self.description}"

    def to_full_prompt(self) -> str:
        """Full SKILL.md content for activation."""
        return self.content

    def to_prompt_xml(self, *, include_content: bool = False, location: str | None = None) -> str:
        """Generate a ``<skill>`` XML block per the spec recommendation.

        Parameters
        ----------
        include_content
            If True, include the full body content (for activated skills).
        location
            Optional absolute path to SKILL.md for filesystem-based agents.
        """
        parts = [
            "  <skill>",
            f"    <name>{_xml_escape(self.name)}</name>",
            f"    <description>{_xml_escape(self.description)}</description>",
        ]
        if location:
            parts.append(f"    <location>{_xml_escape(location)}</location>")
        elif self.source_dir:
            parts.append(f"    <location>{_xml_escape(str(self.source_dir / 'SKILL.md'))}</location>")
        if include_content and self.content:
            parts.append(f"    <content>{_xml_escape(self.content)}</content>")
        parts.append("  </skill>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Resource access
    # ------------------------------------------------------------------

    def get_resource_path(self, relative_path: str) -> Path | None:
        """Resolve a relative path from the skill root directory.

        Returns ``None`` if the skill has no ``source_dir`` or if the
        resolved path does not exist.
        """
        if self.source_dir is None:
            return None
        full = (self.source_dir / relative_path).resolve()
        # Guard against path traversal
        if not str(full).startswith(str(self.source_dir)):
            return None
        return full if full.exists() else None

    def get_resource(self, relative_path: str) -> bytes | None:
        """Get resource bytes by relative path.

        Checks ``self.resources`` first, then falls back to filesystem
        read via ``source_dir``.
        """
        if relative_path in self.resources:
            return self.resources[relative_path]
        if self.source_dir is not None:
            full = (self.source_dir / relative_path).resolve()
            if not str(full).startswith(str(self.source_dir)):
                return None
            if full.is_file():
                return full.read_bytes()
        return None

    def list_resources(self, subdir: str | None = None) -> list[Path]:
        """List files in the skill directory.

        Parameters
        ----------
        subdir
            Optional subdirectory name (e.g. ``"scripts"``, ``"references"``,
            ``"assets"``).  If ``None``, lists all files recursively.

        When ``source_dir`` is available, reads from the filesystem.
        Otherwise, filters keys from the in-memory ``resources`` dict.
        """
        if self.source_dir is not None:
            root = self.source_dir / subdir if subdir else self.source_dir
            if not root.is_dir():
                return []
            return sorted(p for p in root.rglob("*") if p.is_file())

        # In-memory mode: filter resource keys
        if not self.resources:
            return []
        prefix = f"{subdir}/" if subdir else ""
        return sorted(Path(k) for k in self.resources if k.startswith(prefix))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_portable_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict with base64-encoded resources.

        Excludes ``source_dir`` (runtime-only).

        Resource files are base64-encoded for safe JSON transport.
        """
        d = self.model_dump(exclude={"source_dir", "resources"})
        d["resources"] = {k: base64.b64encode(v).decode("ascii") for k, v in self.resources.items()}
        return d

    @classmethod
    def from_portable_dict(cls, data: dict[str, Any]) -> "Skill":
        """Reconstruct a Skill from a ``to_portable_dict()`` output.

        Decodes base64-encoded resources back to bytes.
        """
        data = dict(data)  # shallow copy to avoid mutating input
        raw_resources = data.pop("resources", {})
        resources = {k: base64.b64decode(v) for k, v in raw_resources.items()}
        return cls(resources=resources, **data)

    def save(self, directory: str | Path) -> Path:
        """Write SKILL.md and resource files to disk.

        Parameters
        ----------
        directory
            Parent directory in which to create the skill folder (named
            after ``self.name``).

        Returns
        -------
        Path
            The skill directory path.
        """
        directory = Path(directory)
        skill_dir = directory / self.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Build SKILL.md with YAML frontmatter
        lines = ["---"]
        lines.append(f"name: {self.name}")
        lines.append(f"description: {self.description}")
        if self.license:
            lines.append(f"license: {self.license}")
        if self.compatibility:
            lines.append(f"compatibility: {self.compatibility}")
        if self.allowed_tools:
            lines.append(f"allowed-tools: {' '.join(self.allowed_tools)}")
        if self.metadata:
            lines.append("metadata:")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: \"{v}\"")
        lines.append("---")
        if self.content:
            lines.append(self.content)

        (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")

        # Write resource files
        for rel_path, data in self.resources.items():
            file_path = skill_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(data)

        return skill_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _xml_escape(text: str) -> str:
    """Minimal XML escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter delimited by ``---`` lines.

    Returns ``(frontmatter_dict, body_string)``.
    """
    lines = text.split("\n")

    # The file must start with ---
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, text

    yaml_block = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1:]).strip()

    frontmatter = yaml.safe_load(yaml_block) or {}

    return frontmatter, body
