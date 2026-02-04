#!/usr/bin/env python3
"""
Validation script for HuggingFace Inference API and Embedder providers.
Tests configuration, imports, and basic functionality without requiring API keys.
"""
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "nvidia_nat_core" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "nvidia_nat_langchain" / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 70)
    print("TEST 1: Import Validation")
    print("=" * 70)

    try:
        # Core imports
        from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
        from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig
        print("✓ Core provider configs imported successfully")

        # Registration imports
        from nat.llm import register as llm_register
        from nat.embedder import register as embedder_register
        print("✓ Registration modules imported successfully")

        # LangChain imports (may fail if langchain not installed, that's OK)
        try:
            from nat.plugins.langchain.llm import huggingface_inference_langchain
            from nat.plugins.langchain.embedder import huggingface_langchain
            print("✓ LangChain clients imported successfully")
        except ImportError as e:
            print(f"⚠ LangChain clients not available (OK if langchain not installed): {e}")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_creation():
    """Test that config objects can be created with valid parameters."""
    print("\n" + "=" * 70)
    print("TEST 2: Configuration Creation")
    print("=" * 70)

    try:
        from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
        from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig

        # Test LLM config - Serverless API
        llm_config_serverless = HuggingFaceInferenceConfig(
            model_name="meta-llama/Llama-3.2-8B-Instruct",
            max_new_tokens=512,
            temperature=0.7,
        )
        print(f"✓ Created LLM config (Serverless): {llm_config_serverless.model_name}")
        assert llm_config_serverless.endpoint_url is None
        assert llm_config_serverless.max_new_tokens == 512
        assert llm_config_serverless.temperature == 0.7

        # Test LLM config - Custom endpoint
        llm_config_endpoint = HuggingFaceInferenceConfig(
            model_name="custom-model",
            endpoint_url="https://api.example.com",
            api_key="test_key",
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.9,
            seed=42,
        )
        print(f"✓ Created LLM config (Endpoint): {llm_config_endpoint.model_name}")
        assert llm_config_endpoint.endpoint_url == "https://api.example.com"
        assert llm_config_endpoint.seed == 42

        # Test Embedder config - Local
        embedder_config_local = HuggingFaceEmbedderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="auto",
            normalize_embeddings=True,
            batch_size=32,
        )
        print(f"✓ Created Embedder config (Local): {embedder_config_local.model_name}")
        assert embedder_config_local.endpoint_url is None
        assert embedder_config_local.normalize_embeddings is True
        assert embedder_config_local.batch_size == 32

        # Test Embedder config - Remote
        embedder_config_remote = HuggingFaceEmbedderConfig(
            model_name="remote-embedder",
            endpoint_url="http://localhost:8081",
            api_key="test_key",
            timeout=60.0,
        )
        print(f"✓ Created Embedder config (Remote): {embedder_config_remote.model_name}")
        assert embedder_config_remote.endpoint_url == "http://localhost:8081"
        assert embedder_config_remote.timeout == 60.0

        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """Test that invalid configurations are rejected."""
    print("\n" + "=" * 70)
    print("TEST 3: Configuration Validation")
    print("=" * 70)

    try:
        from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
        from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig
        from pydantic import ValidationError

        # Test invalid max_new_tokens (must be >= 1)
        try:
            HuggingFaceInferenceConfig(
                model_name="test",
                max_new_tokens=0,
            )
            print("✗ Should have rejected max_new_tokens=0")
            return False
        except ValidationError:
            print("✓ Correctly rejected max_new_tokens=0")

        # Test invalid temperature (must be >= 0.0 and <= 2.0)
        try:
            HuggingFaceInferenceConfig(
                model_name="test",
                temperature=3.0,
            )
            print("✗ Should have rejected temperature=3.0")
            return False
        except ValidationError:
            print("✓ Correctly rejected temperature=3.0")

        # Test invalid top_p (must be >= 0.0 and <= 1.0)
        try:
            HuggingFaceInferenceConfig(
                model_name="test",
                top_p=1.5,
            )
            print("✗ Should have rejected top_p=1.5")
            return False
        except ValidationError:
            print("✓ Correctly rejected top_p=1.5")

        # Test invalid batch_size (must be >= 1)
        try:
            HuggingFaceEmbedderConfig(
                model_name="test",
                batch_size=0,
            )
            print("✗ Should have rejected batch_size=0")
            return False
        except ValidationError:
            print("✓ Correctly rejected batch_size=0")

        # Test invalid timeout (must be >= 1.0)
        try:
            HuggingFaceEmbedderConfig(
                model_name="test",
                timeout=0.5,
            )
            print("✗ Should have rejected timeout=0.5")
            return False
        except ValidationError:
            print("✓ Correctly rejected timeout=0.5")

        return True
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixin_inheritance():
    """Test that configs properly inherit from mixins."""
    print("\n" + "=" * 70)
    print("TEST 4: Mixin Inheritance")
    print("=" * 70)

    try:
        from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
        from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig
        from nat.data_models.retry_mixin import RetryMixin
        from nat.data_models.optimizable import OptimizableMixin
        from nat.data_models.thinking_mixin import ThinkingMixin

        # Test LLM config mixins
        llm_config = HuggingFaceInferenceConfig(model_name="test")
        assert isinstance(llm_config, RetryMixin), "LLM config should inherit RetryMixin"
        assert isinstance(llm_config, OptimizableMixin), "LLM config should inherit OptimizableMixin"
        assert isinstance(llm_config, ThinkingMixin), "LLM config should inherit ThinkingMixin"
        print("✓ LLM config has RetryMixin, OptimizableMixin, ThinkingMixin")

        # Test Embedder config mixins
        embedder_config = HuggingFaceEmbedderConfig(model_name="test")
        assert isinstance(embedder_config, RetryMixin), "Embedder config should inherit RetryMixin"
        print("✓ Embedder config has RetryMixin")

        # Check that retry config attributes exist
        assert hasattr(llm_config, 'num_retries')
        assert hasattr(embedder_config, 'num_retries')
        print("✓ Retry attributes available on configs")

        return True
    except Exception as e:
        print(f"✗ Mixin inheritance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_serialization():
    """Test that configs can be serialized and deserialized."""
    print("\n" + "=" * 70)
    print("TEST 5: Configuration Serialization")
    print("=" * 70)

    try:
        from nat.llm.huggingface_inference import HuggingFaceInferenceConfig
        from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig

        # Test LLM config serialization
        llm_config = HuggingFaceInferenceConfig(
            model_name="meta-llama/Llama-3.2-8B-Instruct",
            endpoint_url="https://api.example.com",
            max_new_tokens=512,
            temperature=0.7,
            seed=42,
        )

        config_dict = llm_config.model_dump()
        print(f"✓ LLM config serialized: {len(config_dict)} fields")
        assert config_dict['model_name'] == "meta-llama/Llama-3.2-8B-Instruct"
        assert config_dict['max_new_tokens'] == 512

        # Test deserialization
        llm_config_restored = HuggingFaceInferenceConfig(**config_dict)
        assert llm_config_restored.model_name == llm_config.model_name
        assert llm_config_restored.max_new_tokens == llm_config.max_new_tokens
        print("✓ LLM config deserialized correctly")

        # Test Embedder config serialization
        embedder_config = HuggingFaceEmbedderConfig(
            model_name="BAAI/bge-large-en-v1.5",
            device="cuda",
            normalize_embeddings=True,
            batch_size=32,
        )

        config_dict = embedder_config.model_dump()
        print(f"✓ Embedder config serialized: {len(config_dict)} fields")
        assert config_dict['model_name'] == "BAAI/bge-large-en-v1.5"
        assert config_dict['batch_size'] == 32

        embedder_config_restored = HuggingFaceEmbedderConfig(**config_dict)
        assert embedder_config_restored.model_name == embedder_config.model_name
        assert embedder_config_restored.normalize_embeddings == embedder_config.normalize_embeddings
        print("✓ Embedder config deserialized correctly")

        return True
    except Exception as e:
        print(f"✗ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("HUGGINGFACE PROVIDERS VALIDATION")
    print("=" * 70)

    tests = [
        ("Import Validation", test_imports),
        ("Configuration Creation", test_config_creation),
        ("Configuration Validation", test_config_validation),
        ("Mixin Inheritance", test_mixin_inheritance),
        ("Configuration Serialization", test_config_serialization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        status = "PASSED ✓" if result else "FAILED ✗"
        print(f"{test_name:40s} {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Implementation validated successfully!")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please review issues above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
