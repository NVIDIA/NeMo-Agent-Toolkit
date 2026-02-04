#!/usr/bin/env python3
"""
Live integration test for HuggingFace providers.
Tests actual API connections to validate implementations work.

REQUIRES:
- HF_TOKEN environment variable set
- huggingface_hub package installed
- sentence-transformers package installed (for embedder)
- langchain packages installed
"""
import os
import sys
import asyncio
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "nvidia_nat_core" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "nvidia_nat_langchain" / "src"))


def check_environment():
    """Check that required environment is set up."""
    print("\n" + "=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    # Check HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("✗ HF_TOKEN environment variable not set")
        print("  Set it with: export HF_TOKEN=your_token_here")
        return False
    print(f"✓ HF_TOKEN is set (length: {len(hf_token)})")

    # Check huggingface_hub
    try:
        import huggingface_hub
        print(f"✓ huggingface_hub installed (version: {huggingface_hub.__version__})")
    except ImportError:
        print("✗ huggingface_hub not installed")
        print("  Install with: pip install huggingface_hub")
        return False

    # Check sentence-transformers
    try:
        import sentence_transformers
        print(f"✓ sentence-transformers installed (version: {sentence_transformers.__version__})")
    except ImportError:
        print("⚠ sentence-transformers not installed (optional for local embeddings)")
        print("  Install with: pip install sentence-transformers")

    # Check langchain packages
    try:
        import langchain_core
        print(f"✓ langchain-core installed")
    except ImportError:
        print("✗ langchain-core not installed")
        print("  Install with: pip install langchain-core")
        return False

    return True


async def test_llm_inference_api():
    """Test HuggingFace Inference API LLM provider with real API call."""
    print("\n" + "=" * 70)
    print("TEST: HuggingFace Inference API LLM (LIVE)")
    print("=" * 70)

    from unittest.mock import MagicMock
    from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
    from nat.builder.builder import Builder
    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.plugins.langchain.llm import huggingface_inference_langchain

    # Create config for a small, fast model
    config = HuggingFaceInferenceConfig(
        model_name="HuggingFaceH4/zephyr-7b-beta",  # Small, fast model
        api_key=os.getenv("HF_TOKEN"),
        max_new_tokens=50,
        temperature=0.7,
    )
    print(f"✓ Created config for model: {config.model_name}")

    # Create mock builder (Builder is abstract)
    builder = MagicMock(spec=Builder)
    print("✓ Created mock Builder")

    try:
        async with huggingface_inference_langchain(config, builder) as client:
            print(f"✓ LangChain client created: {type(client).__name__}")

            # Test basic invocation
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content="Say hello in one sentence.")]

            print("\n  Making API call...")
            result = await client.ainvoke(messages)
            print(f"✓ API call successful!")
            print(f"  Response: {result.content[:100]}...")

            # Verify response is not empty
            assert len(result.content) > 0, "Response should not be empty"
            print("✓ Response validation passed")

            return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_streaming():
    """Test HuggingFace Inference API streaming."""
    print("\n" + "=" * 70)
    print("TEST: HuggingFace Inference API Streaming (LIVE)")
    print("=" * 70)

    from unittest.mock import MagicMock
    from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
    from nat.builder.builder import Builder
    from nat.plugins.langchain.llm import huggingface_inference_langchain
    from langchain_core.messages import HumanMessage

    config = HuggingFaceInferenceConfig(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        api_key=os.getenv("HF_TOKEN"),
        max_new_tokens=30,
        temperature=0.7,
    )

    builder = MagicMock(spec=Builder)

    try:
        async with huggingface_inference_langchain(config, builder) as client:
            print(f"✓ Client created for streaming test")

            messages = [HumanMessage(content="Count to 3.")]

            print("\n  Streaming response:")
            print("  ", end="", flush=True)

            chunk_count = 0
            async for chunk in client.astream(messages):
                print(chunk.content, end="", flush=True)
                chunk_count += 1

            print()  # New line

            assert chunk_count > 0, "Should receive at least one chunk"
            print(f"✓ Streaming successful ({chunk_count} chunks received)")

            return True
    except Exception as e:
        print(f"✗ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_embedder_remote():
    """Test HuggingFace Embedder with Inference API (remote)."""
    print("\n" + "=" * 70)
    print("TEST: HuggingFace Embedder Remote API (LIVE)")
    print("=" * 70)

    from unittest.mock import MagicMock
    from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig
    from nat.builder.builder import Builder
    from nat.plugins.langchain.embedder import huggingface_langchain

    # Use Inference API endpoint
    config = HuggingFaceEmbedderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        endpoint_url="https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        api_key=os.getenv("HF_TOKEN"),
        normalize_embeddings=True,
    )
    print(f"✓ Created embedder config (remote mode)")

    builder = MagicMock(spec=Builder)

    try:
        async with huggingface_langchain(config, builder) as client:
            print(f"✓ Embedder client created: {type(client).__name__}")

            # Test embedding
            texts = ["Hello world", "Test embedding"]
            print(f"\n  Embedding {len(texts)} texts...")

            embeddings = await asyncio.to_thread(client.embed_documents, texts)

            print(f"✓ Embeddings generated!")
            print(f"  Shape: {len(embeddings)} texts x {len(embeddings[0])} dimensions")

            # Validate embeddings
            assert len(embeddings) == len(texts), "Should have one embedding per text"
            assert len(embeddings[0]) > 0, "Embeddings should not be empty"
            print("✓ Embedding validation passed")

            return True
    except Exception as e:
        print(f"✗ Remote embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedder_local():
    """Test HuggingFace Embedder with local model (if sentence-transformers available)."""
    print("\n" + "=" * 70)
    print("TEST: HuggingFace Embedder Local Model")
    print("=" * 70)

    try:
        import sentence_transformers
    except ImportError:
        print("⚠ Skipping local embedder test (sentence-transformers not installed)")
        return None

    from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig

    # Create config for local model
    config = HuggingFaceEmbedderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",  # Use CPU for testing
        normalize_embeddings=True,
        batch_size=2,
    )
    print(f"✓ Created embedder config (local mode)")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")

    # Note: Full local test would require downloading model
    # Just validate config for now
    print("✓ Local embedder config validated (model download not tested)")

    return None  # Neutral result (not pass/fail)


async def main():
    """Run all live integration tests."""
    print("\n" + "=" * 70)
    print("HUGGINGFACE PROVIDERS LIVE INTEGRATION TEST")
    print("=" * 70)
    print("\nThis test makes REAL API calls to HuggingFace.")
    print("Ensure HF_TOKEN is set and you have API access.")

    # Check environment
    if not check_environment():
        print("\n✗ Environment check failed. Please install required packages.")
        return 1

    # Run tests
    tests = [
        ("LLM Inference API", test_llm_inference_api()),
        ("LLM Streaming", test_llm_streaming()),
        ("Embedder Remote API", test_embedder_remote()),
    ]

    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Add non-async test
    local_result = test_embedder_local()
    if local_result is not None:
        results.append(("Embedder Local Model", local_result))

    # Print summary
    print("\n" + "=" * 70)
    print("LIVE INTEGRATION TEST SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        if result is True:
            status = "PASSED ✓"
        elif result is False:
            status = "FAILED ✗"
        else:
            status = "SKIPPED ⊘"
        print(f"{test_name:40s} {status}")

    # Count results
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)

    print("\n" + "=" * 70)
    if failed == 0 and passed > 0:
        print(f"✓ ALL {passed} TESTS PASSED - Integrations validated!")
        print("=" * 70)
        return 0
    elif failed > 0:
        print(f"✗ {failed} TEST(S) FAILED, {passed} PASSED")
        print("=" * 70)
        return 1
    else:
        print("⊘ No tests could run - check environment")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
