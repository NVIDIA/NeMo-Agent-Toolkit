# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

from haystack.components.converters.pypdf import PyPDFToDocument
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.core.pipeline import Pipeline
from haystack.document_stores.types import DuplicatePolicy


def _gather_sources(base_dir: Path) -> tuple[list[Path], list[Path]]:
    pdfs = list(base_dir.glob("**/*.pdf"))
    texts = list(base_dir.glob("**/*.txt")) + list(base_dir.glob("**/*.md"))
    return pdfs, texts


def _build_indexing_pipeline(document_store) -> Pipeline:
    p = Pipeline()
    p.add_component("cleaner", DocumentCleaner())
    p.add_component(
        "splitter",
        DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2),
    )
    p.add_component(
        "writer",
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP),
    )
    return p


def run_startup_indexing(document_store, data_dir: str, logger) -> None:
    try:
        data_dir_path = Path(data_dir).expanduser()
        if not data_dir_path.is_absolute():
            data_dir_path = (Path.cwd() / data_dir_path).resolve()

        fallback_data_dir = (Path(__file__).resolve().parents[2] / "data").resolve()

        pdf_sources: list[Path] = []
        text_sources: list[Path] = []
        used_dir = data_dir_path
        if data_dir_path.exists() and data_dir_path.is_dir():
            pdf_sources, text_sources = _gather_sources(data_dir_path)

        if (not pdf_sources and not text_sources and fallback_data_dir.exists() and fallback_data_dir.is_dir()):
            logger.info(
                "Data directory '%s' is missing or empty. Falling back to example data at '%s'",
                str(data_dir_path),
                str(fallback_data_dir),
            )
            used_dir = fallback_data_dir
            pdf_sources, text_sources = _gather_sources(fallback_data_dir)

        total_written = 0
        if pdf_sources or text_sources:
            logger.info(
                "Indexing local files into OpenSearch from '%s' (pdf=%d, text/md=%d)",
                str(used_dir),
                len(pdf_sources),
                len(text_sources),
            )

            indexing_pipeline = _build_indexing_pipeline(document_store)
            indexing_pipeline.add_component("pdf_converter", PyPDFToDocument())
            indexing_pipeline.add_component("text_converter", TextFileToDocument(encoding="utf-8"))

            indexing_pipeline.connect("pdf_converter.documents", "cleaner.documents")
            indexing_pipeline.connect("text_converter.documents", "cleaner.documents")
            indexing_pipeline.connect("cleaner.documents", "splitter.documents")
            indexing_pipeline.connect("splitter.documents", "writer.documents")

            indexing_pipeline.warm_up()

            if pdf_sources:
                res_pdf = indexing_pipeline.run({"pdf_converter": {"sources": pdf_sources}})
                total_written += int(res_pdf.get("writer", {}).get("documents_written", 0))
            if text_sources:
                res_text = indexing_pipeline.run({"text_converter": {"sources": text_sources}})
                total_written += int(res_text.get("writer", {}).get("documents_written", 0))

            logger.info("Indexing complete. Documents written: %s", total_written)
        else:
            logger.info(
                "No indexable files found in '%s' (or fallback '%s'). Skipping indexing.",
                str(data_dir_path),
                str(fallback_data_dir),
            )

    except Exception as e:  # pragma: no cover
        logger.warning("Indexing pipeline failed or was skipped due to an error: %s", str(e))
