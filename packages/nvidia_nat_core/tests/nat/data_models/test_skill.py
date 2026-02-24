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
"""Tests for nat.data_models.skill — Skill (Agent Skills spec compliance)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from nat.data_models.skill import Skill

# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


class TestSkillConstruction:

    def test_minimal(self) -> None:
        s = Skill(name="my-skill", description="Does things")
        assert s.name == "my-skill"
        assert s.description == "Does things"
        assert s.license is None
        assert s.compatibility is None
        assert s.allowed_tools == []
        assert s.metadata == {}
        assert s.content == ""
        assert s.source_dir is None
        assert s.resources == {}

    def test_full(self) -> None:
        s = Skill(
            name="pdf-processing",
            description="Extract text from PDFs",
            license="Apache-2.0",
            compatibility="Requires poppler",
            allowed_tools=["Bash(pdftotext:*)", "Read"],
            metadata={
                "author": "nvidia", "version": "1.0"
            },
            content="# Instructions\nDo the thing.",
            source_dir=Path("/tmp/pdf-processing"),
        )
        assert s.license == "Apache-2.0"
        assert s.compatibility == "Requires poppler"
        assert len(s.allowed_tools) == 2
        assert s.metadata["author"] == "nvidia"
        assert "Instructions" in s.content

    def test_frozen(self) -> None:
        s = Skill(name="test", description="test")
        with pytest.raises(ValidationError):
            s.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Name validation (per Agent Skills spec)
# ---------------------------------------------------------------------------


class TestSkillNameValidation:

    def test_valid_names(self) -> None:
        for name in ["a", "my-skill", "data-analysis", "a1b2c3", "x" * 64]:
            Skill(name=name, description="ok")  # should not raise

    def test_empty_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            Skill(name="", description="ok")

    def test_too_long(self) -> None:
        with pytest.raises(ValidationError, match="64"):
            Skill(name="a" * 65, description="ok")

    def test_uppercase(self) -> None:
        with pytest.raises(ValidationError, match="lowercase"):
            Skill(name="PDF-Processing", description="ok")

    def test_leading_hyphen(self) -> None:
        with pytest.raises(ValidationError):
            Skill(name="-pdf", description="ok")

    def test_trailing_hyphen(self) -> None:
        with pytest.raises(ValidationError):
            Skill(name="pdf-", description="ok")

    def test_consecutive_hyphens(self) -> None:
        with pytest.raises(ValidationError, match="consecutive"):
            Skill(name="pdf--tool", description="ok")

    def test_special_characters(self) -> None:
        with pytest.raises(ValidationError):
            Skill(name="my_skill", description="ok")  # underscores not allowed


# ---------------------------------------------------------------------------
# Description validation
# ---------------------------------------------------------------------------


class TestSkillDescriptionValidation:

    def test_empty_description(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            Skill(name="test", description="")

    def test_too_long_description(self) -> None:
        with pytest.raises(ValidationError, match="1024"):
            Skill(name="test", description="x" * 1025)


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------


class TestSkillCompatibilityValidation:

    def test_too_long_compatibility(self) -> None:
        with pytest.raises(ValidationError, match="500"):
            Skill(name="test", description="ok", compatibility="x" * 501)

    def test_valid_compatibility(self) -> None:
        Skill(name="test", description="ok", compatibility="Requires docker")  # should not raise


# ---------------------------------------------------------------------------
# Directory name match validation
# ---------------------------------------------------------------------------


class TestSkillDirectoryNameValidation:

    def test_name_matches_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        Skill(name="my-skill", description="ok", source_dir=skill_dir.resolve())  # should not raise

    def test_name_mismatch(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "other-name"
        skill_dir.mkdir()
        with pytest.raises(ValidationError, match="match"):
            Skill(name="my-skill", description="ok", source_dir=skill_dir.resolve())


# ---------------------------------------------------------------------------
# Load from SKILL.md
# ---------------------------------------------------------------------------


class TestSkillLoad:

    def test_load_minimal(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir /
         "SKILL.md").write_text("---\nname: my-skill\ndescription: A test skill.\n---\n\n# Instructions\nDo stuff.\n")
        s = Skill.load(skill_dir)
        assert s.name == "my-skill"
        assert s.description == "A test skill."
        assert "Instructions" in s.content
        assert s.source_dir == skill_dir.resolve()

    def test_load_with_optional_fields(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "pdf-tool"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\n"
                                            "name: pdf-tool\n"
                                            "description: Process PDFs.\n"
                                            "license: MIT\n"
                                            "compatibility: Needs poppler\n"
                                            "allowed-tools: Bash(pdf:*) Read\n"
                                            "metadata:\n"
                                            "  author: test\n"
                                            "  version: \"2.0\"\n"
                                            "---\n\n"
                                            "Body content here.\n")
        s = Skill.load(skill_dir)
        assert s.name == "pdf-tool"
        assert s.license == "MIT"
        assert s.compatibility == "Needs poppler"
        assert s.allowed_tools == ["Bash(pdf:*)", "Read"]
        assert s.metadata["author"] == "test"
        assert s.metadata["version"] == "2.0"

    def test_load_missing_skill_md(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Skill.load(tmp_path)

    def test_load_no_frontmatter(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bare"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Just some content.\n")
        with pytest.raises(ValidationError, match="description"):
            Skill.load(skill_dir)

    def test_load_uses_dir_name_as_default(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "fallback"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: ok\n---\nBody\n")
        s = Skill.load(skill_dir)
        assert s.name == "fallback"

    def test_load_captures_resources(self, tmp_path: Path) -> None:
        """load() should read scripts/, references/, assets/ into resources."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: ok\n---\nBody\n")

        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "run.py").write_bytes(b"print('hello')")

        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "REFERENCE.md").write_bytes(b"# Reference")

        assets = skill_dir / "assets"
        assets.mkdir()
        (assets / "logo.png").write_bytes(b"\x89PNG")

        s = Skill.load(skill_dir)
        assert "scripts/run.py" in s.resources
        assert s.resources["scripts/run.py"] == b"print('hello')"
        assert "references/REFERENCE.md" in s.resources
        assert s.resources["references/REFERENCE.md"] == b"# Reference"
        assert "assets/logo.png" in s.resources
        assert s.resources["assets/logo.png"] == b"\x89PNG"

    def test_load_no_resource_dirs(self, tmp_path: Path) -> None:
        """load() with no scripts/references/assets should yield empty resources."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: ok\n---\nBody\n")
        s = Skill.load(skill_dir)
        assert s.resources == {}


# ---------------------------------------------------------------------------
# Progressive disclosure helpers
# ---------------------------------------------------------------------------


class TestSkillPrompts:

    def test_to_metadata_prompt(self) -> None:
        s = Skill(name="search", description="Search the web")
        assert s.to_metadata_prompt() == "search: Search the web"

    def test_to_full_prompt(self) -> None:
        s = Skill(name="test", description="d", content="Full body here")
        assert s.to_full_prompt() == "Full body here"

    def test_to_prompt_xml_minimal(self) -> None:
        s = Skill(name="test", description="A test")
        xml = s.to_prompt_xml()
        assert "<name>test</name>" in xml
        assert "<description>A test</description>" in xml
        assert "<content>" not in xml

    def test_to_prompt_xml_with_content(self) -> None:
        s = Skill(name="test", description="d", content="Body")
        xml = s.to_prompt_xml(include_content=True)
        assert "<content>Body</content>" in xml

    def test_to_prompt_xml_with_location(self) -> None:
        s = Skill(name="test", description="d")
        xml = s.to_prompt_xml(location="/path/to/SKILL.md")
        assert "<location>/path/to/SKILL.md</location>" in xml

    def test_to_prompt_xml_with_source_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test"
        skill_dir.mkdir()
        s = Skill(name="test", description="d", source_dir=skill_dir)
        xml = s.to_prompt_xml()
        assert f"<location>{skill_dir}/SKILL.md</location>" in xml

    def test_xml_escaping(self) -> None:
        s = Skill(name="test", description="Use <b>bold</b> & stuff")
        xml = s.to_prompt_xml()
        assert "&lt;b&gt;bold&lt;/b&gt;" in xml
        assert "&amp;" in xml


# ---------------------------------------------------------------------------
# Resource access
# ---------------------------------------------------------------------------


class TestSkillResources:

    def test_get_resource_no_source_dir(self) -> None:
        s = Skill(name="test", description="d")
        assert s.get_resource_path("scripts/run.py") is None

    def test_get_resource_exists(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        scripts = skill_dir / "scripts"
        scripts.mkdir(parents=True)
        script = scripts / "run.py"
        script.write_text("print('hi')")

        s = Skill(name="my-skill", description="d", source_dir=skill_dir)
        result = s.get_resource_path("scripts/run.py")
        assert result is not None
        assert result.name == "run.py"

    def test_get_resource_not_found(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        s = Skill(name="my-skill", description="d", source_dir=skill_dir)
        assert s.get_resource_path("nonexistent.py") is None

    def test_get_resource_path_traversal_blocked(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        s = Skill(name="my-skill", description="d", source_dir=skill_dir)
        assert s.get_resource_path("../../etc/passwd") is None

    def test_list_resources_no_source_dir(self) -> None:
        s = Skill(name="test", description="d")
        assert s.list_resources() == []

    def test_list_resources_all(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("test")
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "run.py").write_text("code")

        s = Skill(name="my-skill", description="d", source_dir=skill_dir)
        files = s.list_resources()
        names = [f.name for f in files]
        assert "SKILL.md" in names
        assert "run.py" in names

    def test_list_resources_subdir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        scripts = skill_dir / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "a.py").write_text("a")
        (scripts / "b.sh").write_text("b")
        (skill_dir / "SKILL.md").write_text("top")

        s = Skill(name="my-skill", description="d", source_dir=skill_dir)
        files = s.list_resources(subdir="scripts")
        names = [f.name for f in files]
        assert "a.py" in names
        assert "b.sh" in names
        assert "SKILL.md" not in names

    def test_list_resources_missing_subdir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        s = Skill(name="my-skill", description="d", source_dir=skill_dir)
        assert s.list_resources(subdir="nonexistent") == []


# ---------------------------------------------------------------------------
# get_resource (in-memory)
# ---------------------------------------------------------------------------


class TestGetResource:

    def test_get_resource_from_memory(self) -> None:
        s = Skill(
            name="test",
            description="d",
            resources={"scripts/run.py": b"print('hi')"},
        )
        assert s.get_resource("scripts/run.py") == b"print('hi')"

    def test_get_resource_from_filesystem(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        scripts = skill_dir / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "run.py").write_bytes(b"code here")

        s = Skill(name="my-skill", description="d", source_dir=skill_dir)
        assert s.get_resource("scripts/run.py") == b"code here"

    def test_get_resource_memory_takes_priority(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        scripts = skill_dir / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "run.py").write_bytes(b"filesystem version")

        s = Skill(
            name="my-skill",
            description="d",
            source_dir=skill_dir,
            resources={"scripts/run.py": b"memory version"},
        )
        assert s.get_resource("scripts/run.py") == b"memory version"

    def test_get_resource_not_found(self) -> None:
        s = Skill(name="test", description="d")
        assert s.get_resource("nonexistent") is None

    def test_get_resource_traversal_blocked(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        s = Skill(name="my-skill", description="d", source_dir=skill_dir)
        assert s.get_resource("../../etc/passwd") is None


# ---------------------------------------------------------------------------
# list_resources (in-memory, no source_dir)
# ---------------------------------------------------------------------------


class TestListResourcesInMemory:

    def test_list_all(self) -> None:
        s = Skill(
            name="test",
            description="d",
            resources={
                "scripts/a.py": b"a",
                "references/REF.md": b"ref",
                "assets/img.png": b"img",
            },
        )
        paths = s.list_resources()
        names = [str(p) for p in paths]
        assert "scripts/a.py" in names
        assert "references/REF.md" in names
        assert "assets/img.png" in names

    def test_list_subdir(self) -> None:
        s = Skill(
            name="test",
            description="d",
            resources={
                "scripts/a.py": b"a",
                "scripts/b.sh": b"b",
                "references/REF.md": b"ref",
            },
        )
        paths = s.list_resources(subdir="scripts")
        names = [str(p) for p in paths]
        assert "scripts/a.py" in names
        assert "scripts/b.sh" in names
        assert "references/REF.md" not in names

    def test_list_empty(self) -> None:
        s = Skill(name="test", description="d")
        assert s.list_resources() == []
        assert s.list_resources(subdir="scripts") == []


# ---------------------------------------------------------------------------
# Portable dict round-trip
# ---------------------------------------------------------------------------


class TestPortableDict:

    def test_round_trip(self) -> None:
        original = Skill(
            name="my-skill",
            description="A test skill",
            license="MIT",
            compatibility="python>=3.11",
            allowed_tools=["Bash", "Read"],
            metadata={"author": "test"},
            content="# Instructions",
            resources={
                "scripts/run.py": b"print('hi')",
                "assets/logo.png": b"\x89PNG\r\n",
            },
        )
        d = original.to_portable_dict()

        # Resources should be base64-encoded strings
        assert isinstance(d["resources"]["scripts/run.py"], str)

        # source_dir should not be in the dict
        assert "source_dir" not in d

        # Round-trip
        restored = Skill.from_portable_dict(d)
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.license == original.license
        assert restored.compatibility == original.compatibility
        assert restored.allowed_tools == original.allowed_tools
        assert restored.metadata == original.metadata
        assert restored.content == original.content
        assert restored.resources == original.resources
        assert restored.source_dir is None

    def test_round_trip_no_resources(self) -> None:
        original = Skill(name="simple", description="ok")
        d = original.to_portable_dict()
        restored = Skill.from_portable_dict(d)
        assert restored.name == "simple"
        assert restored.resources == {}


# ---------------------------------------------------------------------------
# save() to disk
# ---------------------------------------------------------------------------


class TestSkillSave:

    def test_save_creates_skill_dir(self, tmp_path: Path) -> None:
        s = Skill(name="my-skill", description="A test", content="# Body")
        result = s.save(tmp_path)
        assert result == tmp_path / "my-skill"
        assert (result / "SKILL.md").is_file()

    def test_save_writes_frontmatter(self, tmp_path: Path) -> None:
        s = Skill(
            name="my-skill",
            description="A test",
            license="MIT",
            compatibility="python>=3.11",
            allowed_tools=["Bash", "Read"],
            metadata={"author": "test"},
            content="# Body",
        )
        s.save(tmp_path)
        text = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "name: my-skill" in text
        assert "description: A test" in text
        assert "license: MIT" in text
        assert "compatibility: python>=3.11" in text
        assert "allowed-tools: Bash Read" in text
        assert "author:" in text
        assert "# Body" in text

    def test_save_writes_resources(self, tmp_path: Path) -> None:
        s = Skill(
            name="my-skill",
            description="ok",
            resources={
                "scripts/run.py": b"print('hello')",
                "assets/data.json": b'{"key": "value"}',
            },
        )
        s.save(tmp_path)
        assert (tmp_path / "my-skill" / "scripts" / "run.py").read_bytes() == b"print('hello')"
        assert (tmp_path / "my-skill" / "assets" / "data.json").read_bytes() == b'{"key": "value"}'

    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        original = Skill(
            name="my-skill",
            description="A test skill",
            content="# Instructions\nDo stuff.",
            resources={"scripts/run.py": b"print('hi')"},
        )
        skill_dir = original.save(tmp_path)
        loaded = Skill.load(skill_dir)
        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.content == original.content
        assert loaded.resources == original.resources


# ---------------------------------------------------------------------------
# Pydantic model_dump / model_validate
# ---------------------------------------------------------------------------


class TestPydanticIntegration:

    def test_model_dump_excludes_resources_and_source_dir(self) -> None:
        s = Skill(
            name="test",
            description="ok",
            resources={"scripts/a.py": b"code"},
            source_dir=Path("/tmp/test"),
        )
        d = s.model_dump()
        assert "resources" not in d
        assert "source_dir" not in d
        assert d["name"] == "test"

    def test_model_validate(self) -> None:
        s = Skill.model_validate({"name": "test", "description": "ok"})
        assert s.name == "test"
        assert s.resources == {}
