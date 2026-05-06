<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notice: pathspec 1.1.0

This notice applies to the `pathspec` package resolved for `nvidia-nat-harbor`.

## Component

- Name: `pathspec`
- Version: `1.1.0`
- License: Mozilla Public License Version 2.0 (`MPL-2.0`)
- Upstream author metadata: Caleb P. Burns
- Upstream copyright metadata: Copyright (c) 2013-2026 Caleb P. Burns
- Upstream project: <https://github.com/cpburnz/python-pathspec>
- PyPI package: <https://pypi.org/project/pathspec/1.1.0/>

## Use In `nvidia-nat-harbor`

`nvidia-nat-harbor` declares `harbor>=0.5.0` in
`packages/nvidia_nat_harbor/pyproject.toml`. The resolved Harbor dependency set
pulls in `pathspec==1.1.0`, as recorded in
`packages/nvidia_nat_harbor/uv.lock`.

Resolved dependency parents in the Harbor lockfile:

- `harbor==0.5.0` depends on `pathspec`
- `harbor==0.5.0` depends on `dirhash==0.5.0`
- `dirhash==0.5.0` depends on `scantree==0.0.4`
- `scantree==0.0.4` depends on `pathspec`

No NVIDIA modifications to `pathspec` are included in `nvidia-nat-harbor`.

## Source Package

The as-received `pathspec` source distribution is stored at:

```text
packages/nvidia_nat_harbor/third_party/source/pathspec-1.1.0.tar.gz
```

It was copied from the PyPI source distribution URL recorded in
`packages/nvidia_nat_harbor/uv.lock`:

```text
https://files.pythonhosted.org/packages/2e/17/9c3094b822982b9f1ea666d8580ce59000f61f87c1663556fb72031ad9ec/pathspec-1.1.0.tar.gz
```

Verified SHA256:

```text
f5d7c555da02fd8dde3e4a2354b6aba817a89112fa8f333f7917a2a4834dd080
```

The source distribution includes the upstream `LICENSE` file containing the
Mozilla Public License Version 2.0 text.
