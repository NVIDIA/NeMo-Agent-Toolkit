#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

mkdir -p /workspace
cat > /workspace/answer.txt <<'EOF'
The computed value is __EXPECTED_VALUE__.
EOF

