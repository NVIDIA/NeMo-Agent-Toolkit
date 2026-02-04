#!/usr/bin/env python3
"""
Integration test for HuggingFace providers.
Validates that configurations can be created and providers can be registered.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "packages" / "nvidia_nat_core" / "src"))

def test_provider_config_creation():
    """Test that provider configs can be created with various parameters."""
    print("\n" + "=" * 70)
    print("TEST: Provider Configuration Creation")
    print("=" * 70)

    from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
    from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig

    # Test LLM config with various parameter combinations
    test_cases = [
        {
            "name": "Serverless API (minimal)",
            "config": HuggingFaceInferenceConfig(model_name="test-model"),
        },
        {
            "name": "Serverless API (full params)",
            "config": HuggingFaceInferenceConfig(
                model_name="meta-llama/Llama-3.2-8B-Instruct",
                api_key="test_key",
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                seed=42,
                timeout=180.0,
            ),
        },
        {
            "name": "Custom Endpoint",
            "config": HuggingFaceInferenceConfig(
                model_name="custom-model",
                endpoint_url="https://api.example.com",
                api_key="test_key",
            ),
        },
    ]

    for test_case in test_cases:
        config = test_case["config"]
        print(f"\n✓ {test_case['name']}")
        print(f"  model_name: {config.model_name}")
        print(f"  endpoint_url: {config.endpoint_url}")
        print(f"  max_new_tokens: {config.max_new_tokens}")
        print(f"  temperature: {config.temperature}")

    # Test Embedder config with various parameter combinations
    embedder_test_cases = [
        {
            "name": "Local embedder (minimal)",
            "config": HuggingFaceEmbedderConfig(model_name="test-embedder"),
        },
        {
            "name": "Local embedder (full params)",
            "config": HuggingFaceEmbedderConfig(
                model_name="BAAI/bge-large-en-v1.5",
                device="cuda",
                normalize_embeddings=True,
                batch_size=64,
                max_seq_length=512,
                trust_remote_code=False,
            ),
        },
        {
            "name": "Remote TEI embedder",
            "config": HuggingFaceEmbedderConfig(
                model_name="remote-embedder",
                endpoint_url="http://localhost:8081",
                api_key="test_key",
                timeout=60.0,
            ),
        },
    ]

    for test_case in embedder_test_cases:
        config = test_case["config"]
        print(f"\n✓ {test_case['name']}")
        print(f"  model_name: {config.model_name}")
        print(f"  endpoint_url: {config.endpoint_url}")
        print(f"  device: {config.device}")
        print(f"  normalize_embeddings: {config.normalize_embeddings}")

    return True


def test_mixin_features():
    """Test that mixin features work correctly."""
    print("\n" + "=" * 70)
    print("TEST: Mixin Features")
    print("=" * 70)

    from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
    from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig

    # Test RetryMixin on LLM
    llm_config = HuggingFaceInferenceConfig(
        model_name="test",
        num_retries=5,
        retry_on_status_codes=[429, 503],
    )
    assert llm_config.num_retries == 5
    assert 429 in llm_config.retry_on_status_codes
    print("✓ LLM RetryMixin works")

    # Test OptimizableMixin on LLM
    llm_config_opt = HuggingFaceInferenceConfig(
        model_name="test",
        temperature=0.5,
        max_new_tokens=512,
    )
    # Check that optimizable fields exist
    assert hasattr(llm_config_opt, 'temperature')
    assert hasattr(llm_config_opt, 'max_new_tokens')
    print("✓ LLM OptimizableMixin works")

    # Test ThinkingMixin on LLM
    llm_config_thinking = HuggingFaceInferenceConfig(
        model_name="test",
        thinking_system_prompt="Think step by step",
    )
    assert llm_config_thinking.thinking_system_prompt == "Think step by step"
    print("✓ LLM ThinkingMixin works")

    # Test RetryMixin on Embedder
    embedder_config = HuggingFaceEmbedderConfig(
        model_name="test",
        num_retries=3,
        retry_on_status_codes=[500],
    )
    assert embedder_config.num_retries == 3
    print("✓ Embedder RetryMixin works")

    return True


def test_config_serialization():
    """Test that configs can be serialized and deserialized."""
    print("\n" + "=" * 70)
    print("TEST: Configuration Serialization")
    print("=" * 70)

    from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
    from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig

    # Test LLM serialization
    llm_config = HuggingFaceInferenceConfig(
        model_name="test-model",
        max_new_tokens=512,
        temperature=0.7,
        seed=42,
    )

    llm_dict = llm_config.model_dump()
    llm_restored = HuggingFaceInferenceConfig(**llm_dict)

    assert llm_restored.model_name == llm_config.model_name
    assert llm_restored.max_new_tokens == llm_config.max_new_tokens
    assert llm_restored.temperature == llm_config.temperature
    assert llm_restored.seed == llm_config.seed
    print("✓ LLM config serialization works")

    # Test Embedder serialization
    embedder_config = HuggingFaceEmbedderConfig(
        model_name="test-embedder",
        device="cuda",
        batch_size=32,
    )

    embedder_dict = embedder_config.model_dump()
    embedder_restored = HuggingFaceEmbedderConfig(**embedder_dict)

    assert embedder_restored.model_name == embedder_config.model_name
    assert embedder_restored.device == embedder_config.device
    assert embedder_restored.batch_size == embedder_config.batch_size
    print("✓ Embedder config serialization works")

    return True


def test_provider_registration():
    """Test that providers are properly registered."""
    print("\n" + "=" * 70)
    print("TEST: Provider Registration")
    print("=" * 70)

    # Import registration modules to trigger side effects
    from nat.llm import register as llm_register
    from nat.embedder import register as embedder_register

    print("✓ LLM register module imported")
    print("✓ Embedder register module imported")

    # Check that our providers are imported
    import nat.llm.huggingface_inference
    import nat.embedder.huggingface_embedder

    print("✓ HuggingFace LLM provider module imported")
    print("✓ HuggingFace Embedder provider module imported")

    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("HUGGINGFACE PROVIDERS INTEGRATION TEST")
    print("=" * 70)

    tests = [
        ("Provider Configuration Creation", test_provider_config_creation),
        ("Mixin Features", test_mixin_features),
        ("Configuration Serialization", test_config_serialization),
        ("Provider Registration", test_provider_registration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        status = "PASSED ✓" if result else "FAILED ✗"
        print(f"{test_name:40s} {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME INTEGRATION TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
