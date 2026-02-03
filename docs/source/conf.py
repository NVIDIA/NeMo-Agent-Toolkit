# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import textwrap
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from autoapi._objects import PythonObject

# API builds take about 4 minutes, while the rest of the build process takes about 30 seconds.
build_api_docs = os.getenv('NAT_DISABLE_API_BUILD', '0') != '1'
cur_dir = Path(os.path.abspath(__file__)).parent


def _build_api_tree() -> Path:
    # Work-around for https://github.com/readthedocs/sphinx-autoapi/issues/298
    # AutoAPI support for implicit namespaces is broken, so we need to manually

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
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_reredirects',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    "sphinxmermaid"
]

if build_api_docs:
    api_tree = _build_api_tree()
    print(f"API tree built at {api_tree}")

    extensions.append('autoapi.extension')

    autoapi_dirs = [str(api_tree.absolute())]

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
    autoapi_keep_files = os.getenv('NAT_AUTOAPI_KEEP_FILES', '0') == '1'

else:
    # Create an empty 'api' directory to avoid build errors when API docs are disabled
    api_stub_path = cur_dir / 'api'
    api_stub_path.mkdir(exist_ok=True)
    with open(api_stub_path / "index.rst", "w", encoding="utf-8") as f:
        index_rst = """
                   ==========
                   Python API
                   ==========

                   Placeholder for API documentation build with NAT_DISABLE_API_BUILD=1.
                   """
        f.write(textwrap.dedent(index_rst))

myst_enable_extensions = ["attrs_inline", "colon_fence"]

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
linkcheck_ignore = [
    r'http://localhost:\d+',
    r'https://localhost:\d+',
    r'^http://$',
    r'^https://$',
    r'https://(platform\.)?openai.com',
    r'https://code.visualstudio.com',
    r'https://www.mysql.com',
    r'https://api.service.com',
    r'https?://example\.com/mcp/?',
    r'http://custom-server',
    r'^\?provider=',
    r'https://agent\.example\.com'
]

templates_path = ['_templates']

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
        1,
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

# Configuration for sphinx-reredirects
# Mapping of old document paths to new document paths, the key is the old path relative to the docs/source directory
# without any extensions, and the value is the new path relative to the source or absolute, but with an html extension.
# When adding new redirects, please add a new comment explaining the reason for the redirect followed by a block of
# redirects related to that reason.
redirects = {
    # These redirects cover the documentation restructuring that happened between versions 1.3 and 1.4
    'extend/adding-a-retriever':
        '/extend/custom-components/adding-a-retriever.html',
    'extend/adding-an-authentication-provider':
        '/extend/custom-components/adding-an-authentication-provider.html',
    'extend/adding-an-llm-provider':
        '/extend/custom-components/adding-an-llm-provider.html',
    'extend/cursor-rules-developer-guide':
        '/resources/contributing/cursor/cursor-rules-developer-guide.html',
    'extend/custom-evaluator':
        '/extend/custom-components/custom-evaluator.html',
    'extend/function-groups':
        '/extend/custom-components/custom-functions/function-groups.html',
    'extend/functions':
        '/extend/custom-components/custom-functions/functions.html',
    'extend/gated-fields':
        '/extend/custom-components/gated-fields.html',
    'extend/integrating-aws-bedrock-models':
        '/components/integrations/integrating-aws-bedrock-models.html',
    'extend/memory':
        '/extend/custom-components/memory.html',
    'extend/object-store':
        '/extend/custom-components/object-store.html',
    'extend/sharing-components':
        '/components/sharing-components.html',
    'extend/telemetry-exporters':
        '/extend/custom-components/telemetry-exporters.html',
    'quick-start/index':
        '/get-started/quick-start.html',
    'quick-start/installing':
        '/get-started/installation.html',
    'quick-start/launching-ui':
        '/run-workflows/launching-ui.html',
    'reference/api-authentication':
        '/components/auth/api-authentication.html',
    'reference/api-server-endpoints':
        '/reference/rest-api/api-server-endpoints.html',
    'reference/cursor-rules-reference':
        '/resources/contributing/cursor/cursor-rules-reference.html',
    'reference/evaluate-api':
        '/reference/rest-api/evaluate-api.html',
    'reference/evaluate':
        '/improve-workflows/evaluate.html',
    'reference/frameworks-overview':
        '/components/integrations/frameworks.html',
    'reference/interactive-models':
        '/build-workflows/advanced/interactive-workflows.html',
    'reference/optimizer':
        '/improve-workflows/optimizer.html',
    'reference/test-time-compute':
        '/improve-workflows/test-time-compute.html',
    'reference/websockets':
        '/reference/rest-api/websockets.html',
    'resources/code-of-conduct':
        '/resources/contributing/code-of-conduct.html',
    'resources/contributing':
        '/resources/contributing/index.html',
    'resources/licensing':
        '/resources/contributing/licensing.html',
    'resources/running-ci-locally':
        '/resources/contributing/testing/running-ci-locally.html',
    'resources/running-tests':
        '/resources/contributing/testing/running-tests.html',
    'store-and-retrieve/memory':
        '/build-workflows/memory.html',
    'store-and-retrieve/object-store':
        '/build-workflows/object-store.html',
    'store-and-retrieve/retrievers':
        '/build-workflows/retrievers.html',
    'support':
        '/resources/support.html',
    'troubleshooting':
        '/resources/troubleshooting.html',
    'tutorials/add-tools-to-a-workflow':
        '/get-started/tutorials/add-tools-to-a-workflow.html',
    'tutorials/build-a-demo-agent-workflow-using-cursor-rules':
        '/get-started/tutorials/build-a-demo-agent-workflow-using-cursor-rules.html',
    'tutorials/create-a-new-workflow':
        '/get-started/tutorials/create-a-new-workflow.html',
    'tutorials/customize-a-workflow':
        '/get-started/tutorials/customize-a-workflow.html',
    'tutorials/index':
        '/get-started/tutorials/index.html',
    'tutorials/test-with-nat-test-llm':
        '/extend/testing/test-with-nat-test-llm.html',
    'workflows/about/index':
        '/build-workflows/about-building-workflows.html',
    'workflows/about/react-agent':
        '/components/agents/react-agent/index.html',
    'workflows/about/reasoning-agent':
        '/components/agents/reasoning-agent/index.html',
    'workflows/about/rewoo-agent':
        '/components/agents/rewoo-agent/index.html',
    'workflows/about/router-agent':
        '/components/agents/router-agent/index.html',
    'workflows/about/sequential-executor':
        '/components/agents/sequential-executor/index.html',
    'workflows/about/tool-calling-agent':
        '/components/agents/tool-calling-agent/index.html',
    'workflows/add-unit-tests-for-tools':
        '/extend/testing/add-unit-tests-for-tools.html',
    'workflows/embedders':
        '/build-workflows/embedders.html',
    'workflows/evaluate':
        '/improve-workflows/evaluate.html',
    'workflows/function-groups':
        '/build-workflows/functions-and-function-groups/function-groups.html',
    'workflows/functions/code-execution':
        '/components/functions/code-execution.html',
    'workflows/functions/index':
        '/build-workflows/functions-and-function-groups/functions.html',
    'workflows/llms/index':
        '/build-workflows/llms/index.html',
    'workflows/llms/using-local-llms':
        '/build-workflows/llms/using-local-llms.html',
    'workflows/mcp/index':
        '/build-workflows/mcp-client.html',
    'workflows/mcp/mcp-auth-token-storage':
        '/components/auth/mcp-auth/mcp-auth-token-storage.html',
    'workflows/mcp/mcp-auth':
        '/components/auth/mcp-auth/index.html',
    'workflows/mcp/mcp-client':
        '/build-workflows/mcp-client.html',
    'workflows/mcp/mcp-server':
        '/run-workflows/mcp-server.html',
    'workflows/observe/index':
        '/run-workflows/observe/observe.html',
    'workflows/observe/observe-workflow-with-catalyst':
        '/run-workflows/observe/observe.html?provider=Catalyst#provider-integration-guides',
    'workflows/observe/observe-workflow-with-data-flywheel':
        '/run-workflows/observe/observe.html?provider=Data-Flywheel#provider-integration-guides',
    'workflows/observe/observe-workflow-with-dynatrace':
        '/run-workflows/observe/observe.html?provider=Dynatrace#provider-integration-guides',
    'workflows/observe/observe-workflow-with-galileo':
        '/run-workflows/observe/observe.html?provider=Galileo#provider-integration-guides',
    'workflows/observe/observe-workflow-with-otel-collector':
        '/run-workflows/observe/observe.html?provider=OTel-collector#provider-integration-guides',
    'workflows/observe/observe-workflow-with-phoenix':
        '/run-workflows/observe/observe.html?provider=Phoenix#provider-integration-guides',
    'workflows/observe/observe-workflow-with-weave':
        '/run-workflows/observe/observe.html?provider=Wandb-Weave#provider-integration-guides',
    'workflows/profiler':
        '/improve-workflows/profiler.html',
    'workflows/retrievers':
        '/build-workflows/retrievers.html',
    'workflows/run-workflows':
        '/run-workflows/about-running-workflows.html',
    'workflows/sizing-calc':
        '/improve-workflows/sizing-calc.html',
    'workflows/workflow-configuration':
        '/build-workflows/workflow-configuration.html'
    # End of v1.3 -> v1.4 documentation restructuring redirects
}

if build_api_docs:

    def skip_pydantic_special_attrs(app: object,
                                    what: str,
                                    name: str,
                                    obj: "PythonObject",
                                    skip: bool,
                                    options: list[str]) -> bool:

        if not skip:
            bases = getattr(obj, 'bases', [])
            if (not skip and ('pydantic.BaseModel' in bases or 'EndpointBase' in bases)
                    and PYDANTIC_DEFAULT_DOCSTRING in obj.docstring):
                obj.docstring = ""

        return skip

    def clean_markdown_from_docstrings(app: object, docname: str, source: list[str]) -> None:
        """Clean up Markdown syntax that doesn't work in RST.

        Some inherited docstrings (for example, from LangChain) use Markdown syntax like
        triple backticks for code blocks and !!! for admonitions. These cause RST
        parsing warnings. This function converts or removes such patterns.
        """
        import re
        if not docname.startswith('api/'):
            return

        content = source[0]

        # Remove MkDocs-style admonition blocks: !!! type "title"\n    content
        # These span multiple lines and are complex to convert, so we remove them
        content = re.sub(r'^\s*!!!\s+\w+\s+"[^"]*"\s*\n(?:\s{4,}.*\n)*', '', content, flags=re.MULTILINE)

        # Convert Markdown code fences to RST code blocks
        # Match ```language\n...code...\n``` and convert to :: block
        def convert_code_fence(match: re.Match[str]) -> str:
            indent = match.group(1)
            lang = match.group(2) or ''
            code = match.group(3)
            # Create RST code block with proper indentation
            if lang:
                header = f"{indent}.. code-block:: {lang}\n\n"
            else:
                header = f"{indent}::\n\n"
            # Indent the code content
            indented_code = '\n'.join(f"{indent}   {line}" if line.strip() else '' for line in code.split('\n'))
            return header + indented_code + '\n'

        # Handle code fences with optional language - match ``` at any indentation
        content = re.sub(r'^(\s*)```(\w*)\n(.*?)^\s*```\s*$',
                         convert_code_fence,
                         content,
                         flags=re.MULTILINE | re.DOTALL)

        # Escape **kwargs and **args patterns that appear in function signatures
        # These get interpreted as RST bold/strong markup
        content = re.sub(r'\*\*(kwargs|args|kw)', r'\\*\\*\1', content)

        source[0] = content

    def setup(sphinx):
        # Work-around for for Pydantic docstrings that trigger parsing warnings
        sphinx.connect("autoapi-skip-member", skip_pydantic_special_attrs)
        # Clean up Markdown syntax in auto-generated API docs
        sphinx.connect("source-read", clean_markdown_from_docstrings)
