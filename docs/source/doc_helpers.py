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

import glob
import os
import shutil
import typing
from urllib.parse import urlparse

from markdown_it import MarkdownIt
from mdformat.renderer import MDRenderer

if typing.TYPE_CHECKING:
    from autoapi._objects import PythonObject
    from markdown_it.token import Token

# The defauylt docstring for Pydantic models contains some docstrings that cause parsing warnings for docutils.
# While this string is tightly tied to a specific version of Pydantic, it is hoped that this will be resolved in future
# versions of Pydantic.
PYDANTIC_DEFAULT_DOCSTRING = 'Usage docs: https://docs.pydantic.dev/2.10/concepts/models/\n'

DIRECT_LINK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.md', '.rst')


def _create_dir(dest_dir: str):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    os.makedirs(dest_dir)


def copy_api_tree(src_dir, dest_dir: str):
    # Work-around for https://github.com/readthedocs/sphinx-autoapi/issues/298
    # AutoAPI support for implicit namespaces is broken, so we need to manually
    # construct an aiq package with an __init__.py file
    _create_dir(dest_dir)
    shutil.copytree(src_dir, os.path.join(dest_dir, "aiq"))
    with open(os.path.join(dest_dir, "aiq", "__init__.py"), "w", encoding="utf-8") as f:
        f.write("")


def url_has_scheme(path: str) -> bool:
    """Check if the path has a scheme (e.g., http, https)."""
    parsed_url = urlparse(path)
    return parsed_url.scheme != ''


def path_updater(doc_path: str, path: str, root_dir: str, github_file_url: str) -> str:
    # only re-write relative urls without a scheme (https://)
    if not url_has_scheme(path) and not path.startswith('#'):
        (_, ext) = os.path.splitext(path)
        if '/docs/source' in path:
            # Transform links like `../../docs/source/components/react-agent.md` to `../../components/react-agent.md`
            path = path.replace('/docs/source', '', 1)
        elif ext not in DIRECT_LINK_EXTENSIONS:
            # Re-write links to source code files to point to the GitHub repo. In MD we can link to files in the
            # source tree directly, but since these are not a part of the documentation, we need to link to the
            # GitHub repo instead.
            # First normalize the path
            if not os.path.isabs(path):
                dir_name = os.path.dirname(doc_path)
                norm_path = os.path.normpath(os.path.join(dir_name, path))
                if os.path.isabs(norm_path):
                    norm_path = os.path.relpath(norm_path, start=root_dir)

                path = norm_path

            # replace with github link
            path = os.path.join(github_file_url, path)

    return path


def token_updater(doc_path: str, token: "Token", root_dir: str, github_file_url: str):
    for attr_key in ('href', 'src'):
        attr = token.attrs.get(attr_key)
        if attr is not None:
            # Update the path to remove the '/docs/source' prefix
            token.attrs[attr_key] = path_updater(doc_path,
                                                 str(attr),
                                                 root_dir=root_dir,
                                                 github_file_url=github_file_url)


def token_checker(doc_path: str, token: "Token", root_dir: str, github_file_url: str, prefix: str = ''):
    # The prefix and loc variables are just used for debugging and error reporting
    loc = f"{prefix}.{token.type}"
    try:
        if token.type in ('link_open', 'image'):
            token_updater(doc_path, token, root_dir=root_dir, github_file_url=github_file_url)

        if token.children is not None:
            for child in token.children:
                token_checker(doc_path, child, prefix=loc, root_dir=root_dir, github_file_url=github_file_url)

    except Exception as e:
        raise RuntimeError(f"Markdown parsing error at {loc}: {e}") from e


def rewrite_markdown(doc_path: str, dest_path: str, root_dir: str, github_file_url: str):
    """
    This method serves two purposes:
        1. It rewrites the markdown file updating links and image paths.
        2. Performs a copy by writing the updated markdown to the destination path.
    """
    md = MarkdownIt()

    with open(doc_path, encoding="utf-8") as f:
        tokens = md.parse(f.read())

    for token in tokens:
        token_checker(doc_path, token, root_dir=root_dir, github_file_url=github_file_url)

    with open(dest_path, "w", encoding="utf-8") as f:
        renderer = MDRenderer()
        f.write(renderer.render(tokens, {}, {}))


def write_examples_index(destination_docs: list[str], index_path: str, doc_examples_dir: str):
    skipped_docs = set()
    with open(index_path, "a", encoding="utf-8") as f:
        f.write("\n\n```{toctree}\n:maxdepth: 1\n")

        for doc in destination_docs:
            relative_path = os.path.relpath(doc, doc_examples_dir)

            if relative_path.count(os.sep) == 1:
                f.write(relative_path + "\n")
            else:
                # Skip files that are not in the top-level directory
                skipped_docs.add(doc)

        f.write("\n```\n")

    while len(skipped_docs) > 0:
        skipped_doc = skipped_docs.pop()

        parent_example_dir = os.path.relpath(skipped_doc, doc_examples_dir)
        while parent_example_dir.count(os.sep) > 0:
            parent_example_dir = os.path.dirname(parent_example_dir)

        print(f"\n----------------------------")
        parent_example_dir = os.path.join(doc_examples_dir, parent_example_dir)
        parent_example = os.path.join(parent_example_dir, "README.md")
        child_docs = [skipped_doc]
        for doc in skipped_docs:
            if doc.startswith(parent_example_dir):
                child_docs.append(doc)

        skipped_docs.difference_update(child_docs)

        write_examples_index(child_docs, index_path=parent_example, doc_examples_dir=doc_examples_dir)



def copy_examples(src_dir: str, dest_dir: str, ignore_files: tuple[str], root_dir: str,
                  github_file_url: str) -> list[str]:
    example_readmes = glob.glob(f'{src_dir}/**/*.md', recursive=True)
    examples_index = os.path.join(dest_dir, "index.md")

    _create_dir(dest_dir)

    destination_docs = []
    for example_readme in example_readmes:
        if example_readme in ignore_files:
            continue

        rel_path = os.path.relpath(example_readme, src_dir)
        if rel_path == "README.md":
            # make the top-level README.md file the index
            dest_doc_path = examples_index
        else:
            dest_doc_path = os.path.join(dest_dir, rel_path)
            destination_docs.append(dest_doc_path)

        dest_doc_dir = os.path.dirname(dest_doc_path)
        os.makedirs(dest_doc_dir, exist_ok=True)
        rewrite_markdown(example_readme, dest_doc_path, root_dir=root_dir, github_file_url=github_file_url)

    write_examples_index(destination_docs, index_path=examples_index, doc_examples_dir=dest_dir)

    return destination_docs


def skip_pydantic_special_attrs(app: object, what: str, name: str, obj: "PythonObject", skip: bool,
                                options: list[str]) -> bool:

    if not skip:
        bases = getattr(obj, 'bases', [])
        if (not skip and ('pydantic.BaseModel' in bases or 'EndpointBase' in bases)
                and obj.docstring.startswith(PYDANTIC_DEFAULT_DOCSTRING)):
            obj.docstring = ""

    return skip
