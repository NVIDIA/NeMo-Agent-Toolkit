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

import importlib.util
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for attr_name, attr_value in attributes.items():
        setattr(module, attr_name, attr_value)
    return module


def _build_ingest_dependency_stubs() -> dict[str, ModuleType]:

    class Placeholder:

        def __init__(self, *args, **kwargs):
            pass

    langchain_community = _build_module("langchain_community")
    document_loaders = _build_module("langchain_community.document_loaders", BSHTMLLoader=Placeholder)
    langchain_community.document_loaders = document_loaders
    langchain_milvus = _build_module("langchain_milvus", Milvus=Placeholder)
    langchain_nvidia = _build_module("langchain_nvidia_ai_endpoints", NVIDIAEmbeddings=Placeholder)
    langchain_splitters = _build_module("langchain_text_splitters", RecursiveCharacterTextSplitter=Placeholder)
    pymilvus = _build_module("pymilvus", MilvusClient=Placeholder)

    return {
        "langchain_community": langchain_community,
        "langchain_community.document_loaders": document_loaders,
        "langchain_milvus": langchain_milvus,
        "langchain_nvidia_ai_endpoints": langchain_nvidia,
        "langchain_text_splitters": langchain_splitters,
        "pymilvus": pymilvus,
    }


@pytest.fixture(name="ingest_modules")
def ingest_modules_fixture() -> Iterator[tuple[ModuleType, ModuleType]]:
    stub_modules = _build_ingest_dependency_stubs()
    managed_modules = {
        "web_utils",
        "langchain_web_ingest",
        *stub_modules.keys(),
    }
    prior_modules = {name: sys.modules.pop(name, None) for name in managed_modules}
    sys.modules.update(stub_modules)

    try:
        web_utils = _load_module("web_utils", SCRIPTS_DIR / "web_utils.py")
        langchain_web_ingest = _load_module("langchain_web_ingest", SCRIPTS_DIR / "langchain_web_ingest.py")
        yield web_utils, langchain_web_ingest
    finally:
        for name in managed_modules:
            sys.modules.pop(name, None)
        for name, module in prior_modules.items():
            if module is not None:
                sys.modules[name] = module


def _stub_milvus_dependencies(monkeypatch: pytest.MonkeyPatch,
                              ingest_module: ModuleType,
                              loaded_paths: list[str | None]) -> None:

    class DummyEmbeddings:

        def __init__(self, *args, **kwargs):
            pass

    class DummyMilvus:

        def __init__(self, *args, **kwargs):
            self.col = None

        async def aadd_documents(self, documents, ids):
            return ids

    class DummyLoader:

        def __init__(self, filename: str | None):
            loaded_paths.append(filename)
            self._filename = filename

        def load(self) -> list[str]:
            return [] if self._filename is None else [self._filename]

    class DummySplitter:

        def split_documents(self, docs):
            return docs

    monkeypatch.setattr(ingest_module, "NVIDIAEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(ingest_module, "Milvus", DummyMilvus)
    monkeypatch.setattr(ingest_module, "BSHTMLLoader", DummyLoader)
    monkeypatch.setattr(ingest_module, "RecursiveCharacterTextSplitter", DummySplitter)


def test_cache_html_invalid_input_logs_payload_not_builtin_input(caplog: pytest.LogCaptureFixture,
                                                                 ingest_modules: tuple[ModuleType, ModuleType]) -> None:
    web_utils, _ = ingest_modules
    invalid_input = {"url": "https://example.com/missing", "content": None}

    with caplog.at_level(logging.ERROR):
        cached_input, file_path = web_utils.cache_html(invalid_input, base_path=".")

    assert cached_input == invalid_input
    assert file_path is None
    assert str(invalid_input) in caplog.text
    assert "<built-in function input>" not in caplog.text


@pytest.mark.asyncio
async def test_scrape_empty_body_returns_failure_and_follows_same_origin_redirects(
        monkeypatch: pytest.MonkeyPatch, ingest_modules: tuple[ModuleType, ModuleType]) -> None:
    web_utils, _ = ingest_modules
    request_calls: list[tuple[str, bool]] = []

    class DummyResponse:

        def __init__(self, text: str, status_code: int = 200, headers: dict[str, str] | None = None):
            self.text = text
            self.status_code = status_code
            self.headers = headers or {}

    class DummyAsyncClient:

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, headers=None, follow_redirects: bool = False):
            request_calls.append((url, follow_redirects))
            if url.endswith("/redirect"):
                return DummyResponse("", status_code=301, headers={"location": "/redirected"})
            if url.endswith("/empty"):
                return DummyResponse("")
            return DummyResponse("<html>ok</html>")

    monkeypatch.setattr(web_utils.httpx, "AsyncClient", DummyAsyncClient)

    responses, failures = await web_utils.scrape([
        "https://example.com/redirect",
        "https://example.com/ok",
        "https://example.com/empty",
    ])

    response_urls = {response["url"] for response in responses}

    assert response_urls == {
        "https://example.com/redirect",
        "https://example.com/ok",
    }
    assert {failure["url"] for failure in failures} == {"https://example.com/empty"}
    assert set(request_calls) == {
        ("https://example.com/redirect", False),
        ("https://example.com/redirected", False),
        ("https://example.com/ok", False),
        ("https://example.com/empty", False),
    }


@pytest.mark.asyncio
async def test_scrape_blocks_cross_origin_redirects(monkeypatch: pytest.MonkeyPatch,
                                                    ingest_modules: tuple[ModuleType, ModuleType]) -> None:
    web_utils, _ = ingest_modules
    request_urls: list[str] = []

    class DummyResponse:

        def __init__(self):
            self.text = ""
            self.status_code = 302
            self.headers = {"location": "http://127.0.0.1/latest/meta-data"}

    class DummyAsyncClient:

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, headers=None, follow_redirects: bool = False):
            request_urls.append(url)
            return DummyResponse()

    monkeypatch.setattr(web_utils.httpx, "AsyncClient", DummyAsyncClient)

    responses, failures = await web_utils.scrape("https://example.com/redirect-to-localhost")

    assert responses == []
    assert failures[0]["url"] == "https://example.com/redirect-to-localhost"
    assert "not same-origin" in (failures[0]["exception"] or "")
    assert request_urls == ["https://example.com/redirect-to-localhost"]


@pytest.mark.asyncio
async def test_main_scrapes_uncached_urls_and_skips_scrape_failures(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingest_modules: tuple[ModuleType, ModuleType]) -> None:
    web_utils, ingest = ingest_modules
    loaded_paths: list[str | None] = []
    _stub_milvus_dependencies(monkeypatch, ingest, loaded_paths)

    cached_url = "https://cached.example.com/reference"
    uncached_url = "https://fresh.example.com/reference"
    failed_url = "https://failed.example.com/reference"

    cached_file, cached_dir = web_utils.get_file_path_from_url(cached_url, str(tmp_path))
    fresh_file, fresh_dir = web_utils.get_file_path_from_url(uncached_url, str(tmp_path))
    Path(cached_dir).mkdir(parents=True, exist_ok=True)
    Path(cached_file).write_text("<html>cached</html>", encoding="utf-8")

    scraped_urls: list[list[str]] = []

    async def fake_scrape(urls: list[str]):
        scraped_urls.append(list(urls))
        fresh_page = {"url": uncached_url, "content": "<html>fresh</html>"}
        failed_scrape = {"url": failed_url, "exception": "timeout"}
        return [fresh_page], [failed_scrape]

    def fake_cache_html(input_dict: dict, base_path: str = "."):
        Path(fresh_dir).mkdir(parents=True, exist_ok=True)
        Path(fresh_file).write_text(input_dict["content"], encoding="utf-8")
        return input_dict, fresh_file

    monkeypatch.setattr(ingest, "scrape", fake_scrape)
    monkeypatch.setattr(ingest, "cache_html", fake_cache_html)

    doc_ids = await ingest.main(urls=[cached_url, uncached_url, failed_url],
                                milvus_uri="http://milvus.test:19530",
                                collection_name="cuda_docs",
                                clean_cache=False,
                                base_path=str(tmp_path))

    assert len(doc_ids) == 2
    assert scraped_urls == [[uncached_url, failed_url]]
    assert loaded_paths == [cached_file, fresh_file]
