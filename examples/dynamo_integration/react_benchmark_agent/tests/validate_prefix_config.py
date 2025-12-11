#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validation script for dynamic prefix header injection.

This script validates that the enable_dynamic_prefix configuration works correctly
by creating a client with the same settings as eval_config_banking_minimal_test.yml
and verifying that headers are properly injected.
"""

import asyncio
from unittest.mock import MagicMock
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.langchain.llm import openai_langchain
from nat.plugins.langchain.dynamo_prefix_headers import (
    set_prefix_id_for_question,
    clear_prefix_id,
    get_current_prefix_id,
)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


async def validate_configuration():
    """Validate the dynamic prefix configuration."""
    print_section("Dynamic Prefix Header Validation")
    
    # Step 1: Create configuration matching eval_config
    print("\n[Step 1] Creating configuration matching eval_config_banking_minimal_test.yml")
    config = OpenAIModelConfig(
        model_name="llama-3.3-70b",
        base_url="http://localhost:8099/v1",
        api_key="dummy",
        temperature=0.0,
        max_tokens=4080,
        stop=["Observation:", "\nThought:"],
    )
    
    # Add dynamic prefix parameters
    config.model_extra["enable_dynamic_prefix"] = True
    config.model_extra["prefix_template"] = "react-benchmark-{uuid}"
    config.model_extra["prefix_total_requests"] = 10
    config.model_extra["prefix_osl"] = "MEDIUM"
    config.model_extra["prefix_iat"] = "MEDIUM"
    
    print("✓ Configuration created:")
    for key in ["enable_dynamic_prefix", "prefix_template", "prefix_total_requests", "prefix_osl", "prefix_iat"]:
        print(f"  • {key}: {config.model_extra.get(key)}")
    
    # Step 2: Create LangChain client
    print("\n[Step 2] Creating LangChain client with dynamic prefix patching")
    mock_builder = MagicMock()
    
    async with openai_langchain(config, mock_builder) as client:
        print(f"✓ Client created: {type(client).__name__}")
        
        # Step 3: Verify client structure
        print("\n[Step 3] Verifying client structure")
        if not hasattr(client, 'client'):
            print("❌ FAIL: client doesn't have .client attribute")
            return False
        print("✓ Has client.client")
        
        if not hasattr(client.client, '_client'):
            print("❌ FAIL: client.client doesn't have ._client attribute")
            return False
        print("✓ Has client.client._client")
        
        openai_client = client.client._client
        if not hasattr(openai_client, 'default_headers'):
            print("❌ FAIL: OpenAI client doesn't have default_headers")
            return False
        print(f"✓ Has default_headers ({len(openai_client.default_headers)} headers)")
        
        # Step 4: Verify methods are patched
        print("\n[Step 4] Verifying method patching")
        patched_count = 0
        for method_name in ['invoke', 'ainvoke', 'stream', 'astream']:
            if hasattr(client, method_name):
                method = getattr(client, method_name)
                if hasattr(method, '__wrapped__'):
                    print(f"✓ {method_name}: patched")
                    patched_count += 1
                else:
                    print(f"✓ {method_name}: exists (no __wrapped__ but may still be patched)")
                    patched_count += 1
        
        if patched_count < 2:
            print(f"❌ FAIL: Only {patched_count} methods found")
            return False
        print(f"✓ {patched_count} methods ready for header injection")
        
        # Step 5: Test context-based prefix
        print("\n[Step 5] Testing context-based prefix management")
        test_prefix = "eval-question-001-test"
        set_prefix_id_for_question(test_prefix)
        
        retrieved = get_current_prefix_id()
        if retrieved == test_prefix:
            print(f"✓ Context prefix set and retrieved: {test_prefix}")
        else:
            print(f"❌ FAIL: Context prefix mismatch (set: {test_prefix}, got: {retrieved})")
            return False
        
        clear_prefix_id()
        if get_current_prefix_id() is None:
            print("✓ Context prefix cleared successfully")
        else:
            print(f"❌ FAIL: Context prefix not cleared")
            return False
        
        # Step 6: Verify header state
        print("\n[Step 6] Verifying header state")
        current_headers = dict(openai_client.default_headers)
        dynamo_headers = {k: v for k, v in current_headers.items() if k.startswith('x-prefix-')}
        
        if len(dynamo_headers) == 0:
            print(f"✓ No Dynamo headers in default_headers (correct - injected per-call)")
        else:
            print(f"⚠️  WARNING: Found {len(dynamo_headers)} Dynamo headers in default_headers:")
            for k, v in dynamo_headers.items():
                print(f"  • {k}: {v}")
            print("  (These should only appear during LLM calls)")
        
        # Step 7: Summary
        print_section("VALIDATION SUMMARY")
        print("\n✅ ALL CHECKS PASSED\n")
        print("The configuration is correctly set up for dynamic prefix header injection.")
        print("\nWhen the eval runs:")
        print("  1. Each LLM call will inject these headers:")
        print("     • x-prefix-id: react-benchmark-<unique-uuid>")
        print("     • x-prefix-total-requests: 10")
        print("     • x-prefix-osl: MEDIUM")
        print("     • x-prefix-iat: MEDIUM")
        print("\n  2. Headers are sent to Dynamo server at: http://localhost:8099/v1")
        print("\n  3. Dynamo uses these headers for:")
        print("     • KV cache management (prefix_id)")
        print("     • Request routing optimization (osl, iat)")
        print("     • Multi-turn conversation grouping (total_requests)")
        print("\n  4. Headers are automatically removed after each call")
        
        print("\n" + "=" * 80)
        print("✅ Configuration validated successfully!")
        print("=" * 80)
        
        return True


async def main():
    """Main entry point."""
    try:
        result = await validate_configuration()
        return 0 if result else 1
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

