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
from pathlib import Path

if typing.TYPE_CHECKING:
    from autoapi._objects import PythonObject


def _build_api_tree() -> Path:
    # Work-around for https://github.com/readthedocs/sphinx-autoapi/issues/298
    # AutoAPI support for implicit namespaces is broken, so we need to manually

    cur_dir = Path(os.path.abspath(__file__)).parent
    docs_dir = cur_dir.parent
    root_dir = docs_dir.parent
    nat_dir = root_dir / "src" / "nat"
    plugins_dir = root_dir / "packages"

    build_dir = docs_dir / "build"
    api_tree = build_dir / "_api_tree"
    dest_dir = api_tree / "nat"

    if api_tree.exists():
        shutil.rmtree(api_tree.absolute())

    os.makedirs(api_tree.absolute())
    shutil.copytree(nat_dir, dest_dir)
    dest_plugins_dir = dest_dir / "plugins"

    for sub_dir in (dest_dir, dest_plugins_dir):
        with open(sub_dir / "__init__.py", "w", encoding="utf-8") as f:
            f.write("")

    plugin_dirs = [Path(p) for p in glob.glob(f'{plugins_dir}/nvidia_nat_*')]
    for plugin_dir in plugin_dirs:
        src_dir = plugin_dir / 'src/nat/plugins'
        if src_dir.exists():
            for plugin_subdir in src_dir.iterdir():
                if plugin_subdir.is_dir():
                    dest_subdir = dest_plugins_dir / plugin_subdir.name
                    shutil.copytree(plugin_subdir, dest_subdir)
                    package_file = dest_subdir / "__init__.py"
                    if not package_file.exists():
                        with open(package_file, "w", encoding="utf-8") as f:
                            f.write("")

    return api_tree


API_TREE = _build_api_tree()
print(f"API tree built at {API_TREE}")

# -- Project information -----------------------------------------------------

project = 'NVIDIA NeMo Agent Toolkit'
copyright = '2025, NVIDIA'
author = 'NVIDIA Corporation'

# Retrieve the version number from git via setuptools_scm
called_proc = subprocess.run('python -m setuptools_scm', shell=True, capture_output=True, check=True)
release = called_proc.stdout.strip().decode('utf-8')
version = '.'.join(release.split('.')[:2])

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

autoapi_dirs = [str(API_TREE.absolute())]

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
copybutton_prompt_text = ">>> |$ "  # characters to be stripped from the copied text

# Allow GitHub-style mermaid fence code blocks to be used in markdown files
# see https://myst-parser.readthedocs.io/en/latest/configuration.html
myst_fence_as_directive = ["mermaid"]

suppress_warnings = [
    "myst.header"  # Allow header increases from h2 to h4 (skipping h3)
]

# Config numpydoc
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False

# Config linkcheck
# Ignore localhost and url prefix fragments
# Ignore openai.com links, as these always report a 403 when requested by the linkcheck agent
# mysql.com  reports a 403 when requested by linkcheck
# api.service.com is a placeholder for a service example
# Ignore example.com/mcp as it is inaccessible when building the docs
# Once v1.2 is merged into main, remove the ignore for the banner.png
linkcheck_ignore = [
    r'http://localhost:\d+/',
    r'https://localhost:\d+/',
    r'^http://$',
    r'^https://$',
    r'https://(platform\.)?openai.com',
    r'https://code.visualstudio.com',
    r'https://www.mysql.com',
    r'https://api.service.com',
    r'https?://example\.com/mcp/?',
    r'http://custom-server'
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The root toctree document.
root_doc = 'index'

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

# Setting check_switcher to False, since we are building the version switcher for the first time, the json_url will
# return 404s, which will then cause the build to fail.
html_theme_options = {
    'collapse_navigation':
        False,
    'navigation_depth':
        6,
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
    "show_nav_level":
        2,
    "switcher": {
        "json_url": "../versions1.json", "version_match": version
    },
    "check_switcher":
        False,
    "icon_links": [{
        "name": "GitHub",
        "url": "https://github.com/NVIDIA/NeMo-Agent-Toolkit",
        "icon": "fa-brands fa-github",
    }],
}

html_extra_path = ["versions1.json"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/custom.css']

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
htmlhelp_basename = 'natdoc'

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
    (root_doc, 'nat.tex', 'NeMo Agent Toolkit Documentation', 'NVIDIA', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, 'nat', 'NeMo Agent Toolkit Documentation', [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (root_doc,
     'nat',
     'NeMo Agent Toolkit Documentation',
     author,
     'nat',
     'One line description of project.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ('https://docs.python.org/', None)}

# Set the default role for interpreted code (anything surrounded in `single
# backticks`) to be a python object. See
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role
default_role = "py:obj"

# The defauylt docstring for Pydantic models contains some docstrings that cause parsing warnings for docutils.
# While this string is tightly tied to a specific version of Pydantic, it is hoped that this will be resolved in future
# versions of Pydantic.
PYDANTIC_DEFAULT_DOCSTRING = "A base class for creating Pydantic models."


def skip_pydantic_special_attrs(app: object, what: str, name: str, obj: "PythonObject", skip: bool,
                                options: list[str]) -> bool:

    if not skip:
        bases = getattr(obj, 'bases', [])
        if (not skip and ('pydantic.BaseModel' in bases or 'EndpointBase' in bases)
                and PYDANTIC_DEFAULT_DOCSTRING in obj.docstring):
            obj.docstring = ""

    return skip


def setup(sphinx):
    # Work-around for for Pydantic docstrings that trigger parsing warnings
    sphinx.connect("autoapi-skip-member", skip_pydantic_special_attrs)
