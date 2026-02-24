<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

<!-- path-check-skip-begin -->

# Installing Dynamo from Source

This guide walks through building and installing Dynamo from source on a
fresh machine. Every command is explicit so you can copy-paste your way
through it. If you already have some of the prerequisites installed, skip
the corresponding section.

Tested on Ubuntu 22.04 and 24.04 (x86_64).

---

## Prerequisites

### Required

| Dependency | Why |
|---|---|
| Python 3.10+ | Runtime language |
| Rust (via `rustup`) | Core runtime is written in Rust |
| `uv` | Python package manager (recommended by the Dynamo team) |
| `maturin` | Builds the Rust-to-Python bindings |
| System libraries | C/C++ compiler, `protobuf` compiler, `libclang`, etc. |

### Optional

| Dependency | Why |
|---|---|
| NIXL native library | GPU-to-GPU memory transfers (RDMA). Without it the build succeeds but NIXL functions are stubbed out |
| CUDA toolkit | Required if you plan to run GPU inference backends (FlashInfer JIT compilation needs `nvcc`) |
| etcd / NATS | Required only for distributed or KV-aware routing setups. For local dev you can pass `--discovery-backend file` |

---

## Step 0 — Clone the repository

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
```

---

## Step 1 — Install system libraries

These are needed by the Rust build (`protobuf` `codegen`, C bindings, linking).

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  python3-dev \
  libclang-dev \
  protobuf-compiler \
  libhwloc-dev \
  libudev-dev
```

> **Already have these?** Run `protoc --version` and `dpkg -l libclang-dev`.
> If both succeed you can skip this step.

---

## Step 2 — Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

Verify:

```bash
rustc --version   # e.g. rustc 1.90.0
cargo --version
```

> **Already have Rust?** As long as `rustc --version` prints 1.80+ you
> should be fine. Run `rustup update` if you need a newer toolchain.

---

## Step 3 — Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

> **Already have uv?** Skip this step.

---

## Step 4 — Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
```

From this point on, every command assumes the `venv` is active **and** Rust is
on `PATH`. If you open a new terminal, re-run:

```bash
source .venv/bin/activate
source "$HOME/.cargo/env"   # or: export PATH="$HOME/.cargo/bin:$PATH"
```

---

## Step 5 — Install Python build tools

```bash
uv pip install pip maturin
```

---

## Step 6 — Build the Rust ↔ Python bindings

```bash
cd lib/bindings/python
maturin develop --uv
cd ../../..
```

This compiles the Rust core and installs the `ai-dynamo-runtime` Python
package into your `venv`. On a first build expect this to download and
compile several hundred crates.

### Troubleshooting this step

| Error | Fix |
|---|---|
| `Could not find protoc` | Install `protobuf-compiler` (step 1) |
| `fatal error: 'stdbool.h' file not found` | Install `libclang-dev` (step 1) |
| `rustc ... is not installed or not in PATH` | Run `source "$HOME/.cargo/env"` before `maturin develop` |
| `NIXL build failed ... falling back to stub API` then a `bindgen` error | Install `libclang-dev`. The NIXL headers warning itself is harmless — it just means the NIXL native library is not present and a stub will be used |
| `Failed to set rpath ... patchelf` | Non-fatal warning. Fix with `uv pip install patchelf` if desired |
| `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | Install the CUDA toolkit and create the symlink (see step 1) |

---

## Step 7 — Install the GPU Memory Service

```bash
uv pip install -e lib/gpu_memory_service
```

This is a Python package with a C++ extension. It requires only a C++
compiler (`g++`) and Python development headers, both installed in step 1.

---

## Step 8 — Install Dynamo

```bash
uv pip install -e .
```

This installs the `ai-dynamo` Python package in editable mode so changes
you make to the Python source are picked up immediately.

---

## Step 9 — Verify the installation

```bash
python -c "from dynamo.runtime import DistributedRuntime; print('OK')"
```

You should see `OK` with no errors.

---

## Step 10 — Run something

Start the frontend (no external dependencies needed):

```bash
python -m dynamo.frontend --discovery-backend file
```

In another terminal (with the `venv` activated), start a worker. Pick the
backend you have installed:

```bash
# SGLang
python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file

# vLLM
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'

# TensorRT-LLM
python -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --discovery-backend file
```

> **Note:** The backend frameworks (vLLM, SGLang, TensorRT-LLM) are
> **not** installed by the base `uv pip install -e .` command.

### Installing SGLang from source (recommended)

Install SGLang from the main branch to get the latest fixes and features:

```bash
git clone https://github.com/sgl-project/sglang.git
uv pip install -e sglang/python
```

For other backends, install via extras, e.g. `uv pip install -e ".[vllm]"`.

Send a test request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

---

## Quick reference — all steps on a clean Ubuntu machine

```bash
# System packages
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config python3-dev \
  libclang-dev protobuf-compiler libhwloc-dev libudev-dev

# CUDA toolkit (needed for GPU inference backends)
sudo apt-get install -y nvidia-cuda-toolkit
sudo ln -sf /usr /usr/local/cuda

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter repo
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

# Virtual environment
uv venv
source .venv/bin/activate

# Build
uv pip install pip maturin
cd lib/bindings/python && maturin develop --uv && cd ../../..
uv pip install -e lib/gpu_memory_service
uv pip install -e .

# SGLang from source (recommended)
git clone https://github.com/sgl-project/sglang.git
uv pip install -e sglang/python

# Verify
python -c "from dynamo.runtime import DistributedRuntime; print('OK')"
```

<!-- path-check-skip-end -->
