<!--
    SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
-->

# Contributing to NVIDIA NeMo Agent toolkit

Contributions to NeMo Agent toolkit fall into the following three categories.

* To report a bug, request a new feature, or report a problem with
   documentation, file a [bug](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/new/choose)
    describing in detail the problem or new feature. The NeMo Agent toolkit team evaluates
    and triages bugs and schedules them for a release. If you believe the
    bug needs priority attention, comment on the bug to notify the
    team.
* To propose and implement a new Feature, file a new feature request
    [issue](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/new/choose). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan is good, go ahead and
    implement it, using the [code contributions](#code-contributions) guide below.
* To implement a feature or bug-fix for an existing outstanding issue,
    follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, ask in a comment.

As contributors and maintainers of NeMo Agent toolkit, you are expected to abide by the NeMo Agent toolkit code of conduct. More information can be found at: [Contributor Code of Conduct](./code-of-conduct.md).

## Set Up Your Development Environment
### Prerequisites

- Install [Git](https://git-scm.com/)
- Install [Git Large File Storage](https://git-lfs.github.com/) (LFS)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Install [Visual Studio Code](https://code.visualstudio.com/) (recommended)

NeMo Agent toolkit is a Python library that doesn’t require a GPU to run the workflow by default. You can deploy the core workflows using one of the following:
- Ubuntu or other Linux distributions, including WSL, in a Python virtual environment.

### Creating the Environment

1. Fork the NeMo Agent toolkit repository choosing **Fork** on the [NeMo Agent toolkit repository page](https://github.com/NVIDIA/NeMo-Agent-Toolkit).

1. Clone your personal fork of the NeMo Agent toolkit repository to your local machine.
    ```bash
    git clone <your fork url> nemo-agent-toolkit
    cd nemo-agent-toolkit
    ```

    Then, set the upstream to the main repository and fetch the latest changes:
    ```bash
    git remote add upstream git@github.com:NVIDIA/NeMo-Agent-Toolkit.git
    git fetch --all
    ```


1. Initialize, fetch, and update submodules in the Git repository.
    ```bash
    git submodule update --init --recursive
    ```

1. Fetch the data sets by downloading the LFS files.
    ```bash
    git lfs install
    git lfs fetch
    git lfs pull
    ```

1. Create a Python environment.
    ```bash
    uv venv --seed .venv
    source .venv/bin/activate
    uv sync --all-groups --all-extras
    ```

1. Install and configure pre-commit hooks (optional these can also be run manually).

    ```bash
    pre-commit install
    ```
    **NOTE**: Running pre-commit for the first time will take longer than normal.

1. Open the NeMo Agent toolkit Workspace in Visual Studio Code.
    ```bash
    code ./nat.code-workspace
    ```

### Install the NeMo Agent toolkit Library

1. Install the NeMo Agent toolkit Examples by doing the following.
   - Install NeMo Agent toolkit examples.

     ```bash
     uv sync --extra examples
     ```
   - Install a single example by running `uv pip install -e ./examples/<example_name>`.
   For example, install the Simple Calculator example with the following command.

     ```bash
     uv pip install -e ./examples/getting_started/simple_web_query
     ```

1. Verify that you've installed the NeMo Agent toolkit library.

     ```bash
     nat --help
     nat --version
     ```

     If the installation succeeded, the `nat` command will log the help message and its current version.


## Code contributions

### Your first issue

1. Find an issue to work on. The best way is to search for issues with the [good first issue](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) label.
1. Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.
1. Comment on the issue stating that you are going to work on it.
1. [Fork the NeMo Agent toolkit repository](https://github.com/NVIDIA/NeMo-Agent-Toolkit/fork)
1. Code!
    - Make sure to update unit tests!
    - Ensure the [license headers are set properly](./licensing.md).
1. Verify your changes:
    * Run the style and lint checks, from the root of the repository run:
        ```bash
        ./ci/scripts/checks.sh
        ```
    * Run all unittests and verify that they are passing, from the root of the repository run:
        ```bash
        pytest
        ```
    * Optionally [run the entire CI pipeline locally](./running-ci-locally.md) with the `./ci/scripts/run_ci_local.sh all` command. This is useful if CI is failing in GitHub Actions and you want to debug the issue locally.
1. When done, [create your pull request](https://github.com/NVIDIA/NeMo-Agent-Toolkit/compare). Select `develop` as the `Target branch` of your pull request.
    - Ensure the body of the pull request references the issue you are working on in the form of `Closes #<issue number>`.
1. Wait for other developers to review your code and update code as needed.
1. Once reviewed and approved, a NeMo Agent toolkit developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO is available at [Developer Certificate of Origin](https://developercertificate.org/)

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you can review the prioritized issues for our next release in our [project boards](https://github.com/NVIDIA/NeMo-Agent-Toolkit/projects).

> **Pro Tip:** Always review the release board with the highest number for issues to work on. This is where NeMo Agent toolkit developers also focus their efforts.

Review the unassigned issues and choose an issue that you are comfortable contributing. Ensure you comment on the issue before you begin to inform others that you are working on it. If you have questions about implementing the issue, comment your questions in the issue instead of the PR.

## Developing with NeMo Agent toolkit

Refer to the [Get Started](../quick-start/installing.md) guide to quickly begin development.

## Documentation

All NeMo Agent toolkit documentation should be written in Markdown format. The documentation located under the `docs/source` directory is included in the documentation builds, refer to `docs/README.md` for information on how to build the documentation. In addition to this, each example should contain a `README.md` file that describes the example.

### Checks
All documentation is checked using [Vale](https://vale.sh/). In documentation the name of a command, variable, class, or function should be surrounded by backticks. For example referring `nat` should always be surrounded by backticks. Vale will not perform a check against anything surrounded by backticks or by a code block.

The spelling of a project name should use the casing of the project, for example [PyPI](https://pypi.org/) should always be spelled as `PyPI` and not `pypi` or `PYPI`. If needed new words can be added to the `ci/vale/styles/config/vocabularies/nat/accept.txt` and `ci/vale/styles/config/vocabularies/nat/reject.txt` files.

### Path Checks

All documentation and files which match certain criteria are checked using a custom path check script.

Path checks are used to ensure:
* all symbolic links are valid
* all paths within files are relative paths
* all paths within files are valid (they exist)

#### Adding to the path allowlist

In the case of referential paths, the checker will fail if the path is outside of the outer-level directory. To allowlist a path, add the path to the `ALLOWLISTED_FILE_PATH_PAIRS` set in the `ci/scripts/path_checks.py` file. Paths in the allowlist are always checked for existence.

#### Adding to the word allowlist

In the case of common word groups such as `input/output`, `and/or`, `N/A`, the checker will fail if the word group is not added to the allowlist. To allowlist a word group, add the word group to the `ALLOWLISTED_WORDS` set in the `ci/scripts/path_checks.py` file.

#### Ignoring paths

Ignoring paths is not recommended and should be used as a last resort. If a path is ignored, it will not be checked for existence. It is intended to be used for paths that are not valid or do not exist under source control.

If an exception is needed for a specific path, consider modifying the `ci/scripts/path_checks.py` file to add the path to one of the following sets:
* `IGNORED_PATHS` - a list of paths to ignore (regular expressions)
* `IGNORED_FILES` - a list of files to ignore (regular expressions).
* `IGNORED_FILE_PATH_PAIRS` - a tuple of two regular expressions, the first is the file path and the second is the path to check.

#### Skipping regions of files

The check can be quite aggressive and may detect false positives. If a path is detected as invalid but is actually valid, such as a path to a file that is generated by a tool or a model name, you can add comment(s) to the file to skip the check.

* To skip the **entire file**, ensure `path-check-skip-file` (as a comment) is present near the top of the file.
* To skip a **section of the file**, ensure `path-check-skip` (as a comment) is present on the line above the section and `path-check-skip-end` (as a comment) is present on the line below the section.
* To skip the **next line** in the file, ensure `path-check-skip-next-line` (as a comment) is present on the line above the line to skip.

##### YAML

To skip an entire YAML file, add the following comment to the top of the file:
```yaml
# path-check-skip-file
```

Or to skip sections of a YAML file see the following example:
```yaml
# path-check-skip-begin
this-will-be-skipped: /path/to/skip
so-will-this: /path/to/skip/too
# path-check-skip-end
...
# path-check-skip-next-line
this-will-be-skipped: /path/to/skip
but-this-will-not: /path/to/not/skip
```

##### Markdown

To skip an entire Markdown file, add the following comment to the top of the file:
```markdown
<!-- path-check-skip-file -->
```

To skip a section of a Markdown file, add the following bookend comments:
```markdown
<!-- path-check-skip-begin -->
Here is a list of generated files:
* /path/to/skip
* /path/to/skip/too
<!-- path-check-skip-end -->
...
<!-- path-check-skip-next-line -->
For example, the path mentioned here: `/path/to/skip` will be skipped.
But this path will not be skipped: `/path/to/not/skip`
```

#### File-type specific checks

The path checker is designed to be file-type specific. For example, the checker will check for valid paths in YAML files, JSON files, or Markdown files.

There is logic within the checker to support per-line checks. For example, within a YAML file, the checker will automatically skip lines that contain `model_name` or `_type` since these are often used to indicate the model or tool name which is not a path.

If you are expanding the checker to support a new file type or adding a new per-line check, you can add a new file-type specific checker by adding a new function to the `ci/scripts/path_checks.py` file.

### NVIDIA NeMo Agent toolkit Name Guidelines

* Full Name: `NVIDIA NeMo Agent toolkit`  - Use for document titles, webpage headers, any public descriptions
  - In situations where all words are capitalized (ex: document titles and headings), 'Toolkit' should be capitalized, in all other situations 'toolkit' should not be.

* Short Name: `NeMo Agent toolkit`
  - Use after `NVIDIA NeMo Agent toolkit` has been referenced in blogs, documents, and other public locations
  - Note that the 't' is lowercase in toolkit unless used in a title or heading
* Uppercase No Space: `NeMo-Agent-Toolkit`
  - Use for situations where capitalization will be preserved like the GitHub URL, directories, etc.
  - Do not use dashes or underscores
  - Note that the 't' is lowercase in toolkit unless used in a title or heading
