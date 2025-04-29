# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import glob
import os
import shutil
import subprocess
import typing
from urllib.parse import urlparse

from markdown_it import MarkdownIt
from mdformat.renderer import MDRenderer

if typing.TYPE_CHECKING:
    from autoapi._objects import PythonObject
    from markdown_it.token import Token

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.dirname(CUR_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(CUR_DIR))
AIQ_DIR = os.path.join(ROOT_DIR, "src", "aiq")
DOC_EXAMPLES = os.path.join(DOC_DIR, "source", "examples")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROJECT_URL = "https://github.com/NVIDIA/AIQToolkit"
FILE_URL = f"{PROJECT_URL}/blob/main"

BUILD_DIR = os.path.join(DOC_DIR, "build")
API_TREE = os.path.join(BUILD_DIR, "_api_tree")


def _create_dir(dest_dir: str):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    os.makedirs(dest_dir)


def copy_api_tree():
    # Work-around for https://github.com/readthedocs/sphinx-autoapi/issues/298
    # AutoAPI support for implicit namespaces is broken, so we need to manually
    # construct an aiq package with an __init__.py file
    _create_dir(API_TREE)
    shutil.copytree(AIQ_DIR, os.path.join(API_TREE, "aiq"))
    with open(os.path.join(API_TREE, "aiq", "__init__.py"), "w") as f:
        f.write("")


# Copy example Markdown files into the documentation tree
IGNORE_EXAMPLES = (os.path.join(EXAMPLES_DIR, 'documentation_guides/README.md'), )
example_readmes = glob.glob(f'{EXAMPLES_DIR}/**/*.md', recursive=True)
EXAMPLES_INDEX = os.path.join(DOC_EXAMPLES, "index.md")
DIRECT_LINK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.md', '.rst')


def url_has_scheme(path: str) -> bool:
    """Check if the path has a scheme (e.g., http, https)."""
    parsed_url = urlparse(path)
    return parsed_url.scheme != ''


# re-write links
def path_updater(doc_path: str, path: str) -> str:
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
                    norm_path = os.path.relpath(norm_path, start=ROOT_DIR)

                path = norm_path

            # replace with github link
            path = os.path.join(FILE_URL, path)

    return path


def token_updater(doc_path: str, token: "Token"):
    for attr_key in ('href', 'src'):
        attr = token.attrs.get(attr_key)
        if attr is not None:
            # Update the path to remove the '/docs/source' prefix
            token.attrs[attr_key] = path_updater(doc_path, str(attr))


def token_checker(doc_path: str, token: "Token", prefix: str = ''):
    # The prefix and loc variables are just used for debugging and error reporting
    loc = f"{prefix}.{token.type}"
    try:
        if token.type in ('link_open', 'image'):
            token_updater(doc_path, token)

        if token.children is not None:
            for child in token.children:
                token_checker(doc_path, child, prefix=loc)

    except Exception as e:
        raise RuntimeError(f"Markdown parsing error at {loc}: {e}") from e


def rewrite_markdown(doc_path: str, dest_path: str):
    """
    This method serves two purposes:
        1. It rewrites the markdown file updating links and image paths.
        2. Performs a copy by writing the updated markdown to the destination path.
    """
    md = MarkdownIt()

    with open(doc_path) as f:
        tokens = md.parse(f.read())

    for token in tokens:
        token_checker(doc_path, token)

    with open(dest_path, "w") as f:
        renderer = MDRenderer()
        f.write(renderer.render(tokens, {}, {}))


def copy_examples() -> list[str]:
    _create_dir(DOC_EXAMPLES)

    destination_docs = []
    for example_readme in example_readmes:
        if example_readme in IGNORE_EXAMPLES:
            continue

        rel_path = os.path.relpath(example_readme, EXAMPLES_DIR)
        if rel_path == "README.md":
            # make the top-level README.md file the index
            dest_path = EXAMPLES_INDEX
        else:
            dest_path = os.path.join(DOC_EXAMPLES, rel_path)
            destination_docs.append(dest_path)

        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        rewrite_markdown(example_readme, dest_path)

    return destination_docs


def write_examples_index(destination_docs: list[str]):
    with open(EXAMPLES_INDEX, "a") as f:
        f.write("\n\n```{toctree}\n:maxdepth: 1\n")

        for doc in destination_docs:
            relative_path = os.path.relpath(doc, DOC_EXAMPLES)
            f.write(relative_path + "\n")

        f.write("\n```\n")


copy_api_tree()
destination_docs = copy_examples()
write_examples_index(destination_docs)

# -- Project information -----------------------------------------------------

project = 'Agent Intelligence Toolkit'
copyright = '2025, NVIDIA'
author = 'NVIDIA Corporation'

# Retrieve the version number from git via setuptools_scm
called_proc = subprocess.run('python -m setuptools_scm', shell=True, capture_output=True, check=True)
release = called_proc.stdout.strip().decode('utf-8')
version = '.'.join(release.split('.')[:3])

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    "sphinxmermaid"
]

autoapi_dirs = [API_TREE]

autoapi_root = "api"
autoapi_python_class_content = "both"
autoapi_options = [
    'members',
    'undoc-members',
    'private-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]

# set to true once https://github.com/readthedocs/sphinx-autoapi/issues/298 is fixed
autoapi_python_use_implicit_namespaces = False

# Enable this for debugging
autoapi_keep_files = False

myst_enable_extensions = ["colon_fence"]

html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
add_module_names = False  # Remove namespaces from class/method signatures
myst_heading_anchors = 4  # Generate links for markdown headers
copybutton_prompt_text = ">>> |$ |# "  # characters to be stripped from the copied text

suppress_warnings = [
    "myst.header"  # Allow header increases from h2 to h4 (skipping h3)
]

# Config numpydoc
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False

# Config linkcheck
# Ignore localhost and url prefix fragments
# Ignore openai.com links, as these always report a 403 when requested by the linkcheck agent
linkcheck_ignore = [
    r'http://localhost:\d+/',
    r'https://localhost:\d+/',
    r'^http://$',
    r'^https://$',
    r'https://(platform\.)?openai.com',
    r'https://code.visualstudio.com'
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["build", "dist"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "nvidia_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_logo = '_static/main_nv_logo_square.png'
html_title = f'{project} ({version})'

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 6,
    'extra_head': [  # Adding Adobe Analytics
        '''
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    '''
    ],
    'extra_footer': [
        '''
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    '''
    ],
    "show_nav_level": 2
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'aiqdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'aiq.tex', 'Agent Intelligence Toolkit Documentation', 'NVIDIA', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'aiq', 'Agent Intelligence Toolkit Documentation', [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc,
     'aiq',
     'Agent Intelligence Toolkit Documentation',
     author,
     'aiq',
     'One line description of project.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ('https://docs.python.org/', None), "scipy": ('https://docs.scipy.org/doc/scipy/reference', None)
}

# Set the default role for interpreted code (anything surrounded in `single
# backticks`) to be a python object. See
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role
default_role = "py:obj"

# The defauylt docstring for Pydantic models contains some docstrings that cause parsing warnings for docutils.
# While this string is tightly tied to a specific version of Pydantic, it is hoped that this will be resolved in future
# versions of Pydantic.
PYDANTIC_DEFAULT_DOCSTRING = 'Usage docs: https://docs.pydantic.dev/2.10/concepts/models/\n'


def skip_pydantic_special_attrs(app: object, what: str, name: str, obj: "PythonObject", skip: bool,
                                options: list[str]) -> bool:

    if not skip:
        bases = getattr(obj, 'bases', [])
        if (not skip and ('pydantic.BaseModel' in bases or 'EndpointBase' in bases)
                and obj.docstring.startswith(PYDANTIC_DEFAULT_DOCSTRING)):
            obj.docstring = ""

    return skip


def setup(sphinx):
    # Work-around for for Pydantic docstrings that trigger parsing warnings
    sphinx.connect("autoapi-skip-member", skip_pydantic_special_attrs)
