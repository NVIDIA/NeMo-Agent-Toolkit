#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build Dynamo vLLM Image with MultiLruBackend from ryan/kvbm-next branch
#
# This script builds the Dynamo vLLM runtime image from source using the
# ryan/kvbm-next branch, which includes the 4-pool MultiLruBackend for
# frequency-based KV cache eviction.
#
# The build uses the branch's native container/build.sh with:
#   - Framework: VLLM
#   - KVBM enabled (includes MultiLruBackend)
#   - vLLM v0.14.0
#   - CUDA 12.9
#   - Python 3.12
#
# Usage:
#   ./build_multi_lru_image.sh [options]
#
# Options:
#   --no-cache          Build without Docker cache
#   --skip-clone        Skip cloning/updating the branch (use existing source)
#   --source-dir DIR    Source directory (default: auto-detect kvbm_next_source or kvbm_next_build)
#   --target TARGET     Docker build target (default: runtime)
#   --tag TAG           Custom image tag (default: dynamo-multi-lru:latest)
#   --dry-run           Print commands without executing
#   --help              Show this help message
#
# Environment Variables:
#   DYNAMO_SOURCE_DIR   Source directory (alternative to --source-dir)
#   DYNAMO_BUILD_JOBS   Cargo build parallelism (default: 4, reduce if OOM)
#   DYNAMO_MAX_JOBS     vLLM compilation parallelism (default: 8)

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BRANCH="ryan/kvbm-next"
REPO_URL="https://github.com/ai-dynamo/dynamo.git"

# Build options (can be overridden by command line args)
KVBM_NEXT_DIR=""  # Will be set after arg parsing
IMAGE_TAG="${DYNAMO_IMAGE_TAG:-dynamo-multi-lru:latest}"
BUILD_TARGET="${DYNAMO_BUILD_TARGET:-runtime}"
NO_CACHE=""
SKIP_CLONE=false
DRY_RUN=""
CARGO_BUILD_JOBS="${DYNAMO_BUILD_JOBS:-4}"
MAX_JOBS="${DYNAMO_MAX_JOBS:-8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --skip-clone)
            SKIP_CLONE=true
            shift
            ;;
        --source-dir)
            KVBM_NEXT_DIR="$2"
            shift 2
            ;;
        --target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --help|-h)
            head -42 "$0" | tail -37
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect source directory if not specified
if [ -z "$KVBM_NEXT_DIR" ]; then
    if [ -n "${DYNAMO_SOURCE_DIR:-}" ]; then
        KVBM_NEXT_DIR="$DYNAMO_SOURCE_DIR"
    elif [ -d "${SCRIPT_DIR}/kvbm_next_source" ] && [ -f "${SCRIPT_DIR}/kvbm_next_source/container/build.sh" ]; then
        KVBM_NEXT_DIR="${SCRIPT_DIR}/kvbm_next_source"
        echo "Auto-detected existing source: $KVBM_NEXT_DIR"
    else
        KVBM_NEXT_DIR="${SCRIPT_DIR}/kvbm_next_build"
    fi
fi

echo "========================================================="
echo "Building Dynamo vLLM Image with MultiLruBackend"
echo "========================================================="
echo ""
echo "Configuration:"
echo "  Branch:        $BRANCH"
echo "  Source Dir:    $KVBM_NEXT_DIR"
echo "  Image Tag:     $IMAGE_TAG"
echo "  Build Target:  $BUILD_TARGET"
echo "  Cargo Jobs:    $CARGO_BUILD_JOBS"
echo "  vLLM Jobs:     $MAX_JOBS"
echo "  Skip Clone:    $SKIP_CLONE"
echo "  No Cache:      ${NO_CACHE:-false}"
echo ""

# Step 1: Clone or update the ryan/kvbm-next branch
if [ "$SKIP_CLONE" = false ]; then
    if [ -d "$KVBM_NEXT_DIR" ]; then
        echo "Updating existing $BRANCH branch..."
        cd "$KVBM_NEXT_DIR"
        git fetch origin
        git checkout "$BRANCH"
        git pull origin "$BRANCH"
        git submodule update --init --recursive
    else
        echo "Cloning $BRANCH branch..."
        git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$KVBM_NEXT_DIR"
        cd "$KVBM_NEXT_DIR"
        git submodule update --init --recursive
    fi
    echo "✓ Source code ready at $KVBM_NEXT_DIR"
else
    if [ ! -d "$KVBM_NEXT_DIR" ]; then
        echo "ERROR: --skip-clone specified but source directory doesn't exist: $KVBM_NEXT_DIR"
        exit 1
    fi
    echo "Using existing source at $KVBM_NEXT_DIR"
    cd "$KVBM_NEXT_DIR"
fi
echo ""

# Step 2: Apply MultiLruBackend patch (if needed)
# The scheduler at lib/bindings/kvbm/src/v2/scheduler/mod.rs may use LineageBackend by default.
# We patch it to use MultiLruBackend for frequency-based eviction.
SCHEDULER_FILE="lib/bindings/kvbm/src/v2/scheduler/mod.rs"

if [ -f "$SCHEDULER_FILE" ]; then
    if grep -q "with_lineage_backend" "$SCHEDULER_FILE"; then
        echo "Patching scheduler to enable MultiLruBackend..."
        sed -i 's/\.with_lineage_backend()/.with_multi_lru_backend()/g' "$SCHEDULER_FILE"
        
        if grep -q "with_multi_lru_backend" "$SCHEDULER_FILE"; then
            echo "✓ Scheduler patched: LineageBackend → MultiLruBackend"
            grep -n "with_multi_lru_backend" "$SCHEDULER_FILE" | head -3
        else
            echo "WARNING: Patch may have failed - check $SCHEDULER_FILE"
        fi
    elif grep -q "with_multi_lru_backend" "$SCHEDULER_FILE"; then
        echo "✓ Scheduler already uses MultiLruBackend"
    else
        echo "WARNING: Could not find backend configuration in $SCHEDULER_FILE"
        echo "         The scheduler may use a different configuration method."
    fi
else
    echo "WARNING: Scheduler file not found at $SCHEDULER_FILE"
    echo "         This is expected if the branch structure has changed."
fi
echo ""

# Step 3: Build the image using the branch's build.sh
echo "========================================================="
echo "Building Docker image..."
echo "========================================================="
echo ""
echo "Build command:"
echo "  ./container/build.sh \\"
echo "    --framework VLLM \\"
echo "    --target $BUILD_TARGET \\"
echo "    --tag $IMAGE_TAG \\"
echo "    --enable-kvbm \\"
echo "    --build-arg CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS \\"
echo "    --vllm-max-jobs $MAX_JOBS \\"
echo "    $NO_CACHE $DRY_RUN"
echo ""

# Make build.sh executable
chmod +x container/build.sh

# Run the build
# Note: --enable-kvbm is automatically set for VLLM framework, but we set it explicitly for clarity
./container/build.sh \
    --framework VLLM \
    --target "$BUILD_TARGET" \
    --tag "$IMAGE_TAG" \
    --enable-kvbm \
    --build-arg "CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS" \
    --vllm-max-jobs "$MAX_JOBS" \
    $NO_CACHE \
    $DRY_RUN

BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================="
    echo "✓ Build successful!"
    echo "========================================================="
    echo ""
    echo "Image: $IMAGE_TAG"
    echo ""
    
    # Verify the image has KVBM installed
    echo "Verifying image contents..."
    if docker run --rm "$IMAGE_TAG" python3 -c "import kvbm; print('✓ KVBM module installed')" 2>/dev/null; then
        echo ""
    else
        echo "⚠ Warning: Could not verify KVBM installation in image"
    fi
    
    # Check for DynamoScheduler
    if docker run --rm "$IMAGE_TAG" python3 -c "from kvbm.v2.vllm.schedulers.dynamo import DynamoScheduler; print('✓ DynamoScheduler available')" 2>/dev/null; then
        echo ""
    else
        echo "⚠ Warning: Could not verify DynamoScheduler in image"
    fi
    
    echo "Features:"
    echo "  - vLLM v0.14.0 backend"
    echo "  - KVBM with MultiLruBackend (4-pool frequency-based eviction)"
    echo "  - CUDA 12.9"
    echo "  - Python 3.12"
    echo "  - NIXL 0.9.0 for KV transfer"
    echo ""
    echo "MultiLruBackend Configuration:"
    echo "  - 4 priority pools: Cold → Warm → Hot → VeryHot"
    echo "  - Default promotion thresholds: [2, 6, 15] accesses"
    echo "  - Frequently accessed blocks protected from eviction"
    echo ""
    echo "To use this image, update your startup script:"
    echo "  IMAGE=\"$IMAGE_TAG\""
    echo ""
    echo "Or set the environment variable:"
    echo "  export DYNAMO_VLLM_IMAGE=\"$IMAGE_TAG\""
    echo ""
    echo "Then run:"
    echo "  ./start_dynamo_optimized_thompson_hints_vllm_multilru.sh"
    echo ""
else
    echo ""
    echo "========================================================="
    echo "✗ Build failed with exit code: $BUILD_EXIT_CODE"
    echo "========================================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check Docker daemon is running"
    echo "  2. Ensure sufficient disk space (needs ~50GB)"
    echo "  3. Try reducing parallelism:"
    echo "     DYNAMO_BUILD_JOBS=2 DYNAMO_MAX_JOBS=4 ./build_multi_lru_image.sh"
    echo "  4. Check build logs above for specific errors"
    echo ""
    exit $BUILD_EXIT_CODE
fi

