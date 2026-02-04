#!/usr/bin/env python3
"""
Static audit script for HuggingFace providers.
Checks code quality, consistency, and potential issues without requiring dependencies.
"""
import ast
import re
from pathlib import Path

def audit_file_structure(filepath):
    """Audit a Python file for structure and quality."""
    print(f"\nAuditing: {filepath}")
    print("-" * 70)

    with open(filepath) as f:
        content = f.read()

    issues = []
    warnings = []

    # Check 1: License header
    if not content.startswith("# SPDX-FileCopyrightText:"):
        issues.append("Missing SPDX license header")
    else:
        print("✓ License header present")

    # Check 2: Parse Python syntax
    try:
        tree = ast.parse(content)
        print("✓ Valid Python syntax")
    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")
        return issues, warnings

    # Check 3: Check for proper docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not ast.get_docstring(node):
                warnings.append(f"Missing docstring for {node.name}")

    if not warnings:
        print("✓ All classes/functions have docstrings")
    else:
        print(f"⚠ {len(warnings)} missing docstrings")

    # Check 4: Check imports are organized
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    print(f"✓ {len(imports)} import statements found")

    # Check 5: Check Field descriptions
    field_count = content.count("Field(")
    desc_count = content.count('description="')
    if field_count > 0:
        if desc_count >= field_count:
            print(f"✓ All {field_count} Fields have descriptions")
        else:
            warnings.append(f"Some Fields missing descriptions: {field_count} fields, {desc_count} descriptions")

    # Check 6: Check for TODO/FIXME/HACK
    for keyword in ["TODO", "FIXME", "HACK"]:
        if keyword in content:
            warnings.append(f"Contains {keyword} marker")

    # Check 7: Check async/await usage
    async_defs = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]
    if async_defs:
        print(f"✓ {len(async_defs)} async functions defined")

    return issues, warnings


def check_consistency():
    """Check consistency across implementations."""
    print("\n" + "=" * 70)
    print("CONSISTENCY CHECKS")
    print("=" * 70)

    base_dir = Path("/Users/bledden/Documents/nemo-agent-toolkit/packages/nvidia_nat_core/src/nat")

    # Check LLM provider config
    llm_file = base_dir / "llm" / "huggingface_inference.py"
    with open(llm_file) as f:
        llm_content = f.read()

    # Check Embedder provider config
    embedder_file = base_dir / "embedder" / "huggingface_embedder.py"
    with open(embedder_file) as f:
        embedder_content = f.read()

    issues = []

    # Check 1: Both should have RetryMixin
    if "RetryMixin" in llm_content and "RetryMixin" in embedder_content:
        print("✓ Both providers inherit RetryMixin")
    else:
        issues.append("RetryMixin not consistent across providers")

    # Check 2: Both should use ConfigDict(protected_namespaces=(), extra="allow")
    if 'ConfigDict(protected_namespaces=(), extra="allow")' in llm_content:
        print("✓ LLM config uses proper ConfigDict")
    else:
        issues.append("LLM config missing proper ConfigDict")

    if 'ConfigDict(protected_namespaces=(), extra="allow")' in embedder_content:
        print("✓ Embedder config uses proper ConfigDict")
    else:
        issues.append("Embedder config missing proper ConfigDict")

    # Check 3: api_key should be OptionalSecretStr
    if "api_key: OptionalSecretStr" in llm_content and "api_key: OptionalSecretStr" in embedder_content:
        print("✓ Both use OptionalSecretStr for api_key")
    else:
        issues.append("api_key type not consistent")

    # Check 4: Check register decorators
    if "@register_llm_provider" in llm_content:
        print("✓ LLM provider registration decorator present")
    else:
        issues.append("LLM provider missing registration decorator")

    if "@register_embedder_provider" in embedder_content:
        print("✓ Embedder provider registration decorator present")
    else:
        issues.append("Embedder provider missing registration decorator")

    # Check 5: Check yield LLMProviderInfo/EmbedderProviderInfo
    if "yield LLMProviderInfo" in llm_content:
        print("✓ LLM provider yields LLMProviderInfo")
    else:
        issues.append("LLM provider doesn't yield LLMProviderInfo")

    if "yield EmbedderProviderInfo" in embedder_content:
        print("✓ Embedder provider yields EmbedderProviderInfo")
    else:
        issues.append("Embedder provider doesn't yield EmbedderProviderInfo")

    return issues


def check_langchain_clients():
    """Check LangChain client implementations."""
    print("\n" + "=" * 70)
    print("LANGCHAIN CLIENT CHECKS")
    print("=" * 70)

    langchain_dir = Path("/Users/bledden/Documents/nemo-agent-toolkit/packages/nvidia_nat_langchain/src/nat/plugins/langchain")

    # Check LLM client
    llm_client_file = langchain_dir / "llm.py"
    with open(llm_client_file) as f:
        llm_client_content = f.read()

    # Check Embedder client
    embedder_client_file = langchain_dir / "embedder.py"
    with open(embedder_client_file) as f:
        embedder_client_content = f.read()

    issues = []

    # Check 1: Import of our configs
    if "from nat.llm.huggingface_inference import HuggingFaceInferenceConfig" in llm_client_content:
        print("✓ LLM client imports HuggingFaceInferenceConfig")
    else:
        issues.append("LLM client missing HuggingFaceInferenceConfig import")

    if "from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig" in embedder_client_content:
        print("✓ Embedder client imports HuggingFaceEmbedderConfig")
    else:
        issues.append("Embedder client missing HuggingFaceEmbedderConfig import")

    # Check 2: Registration decorators
    if "@register_llm_client(config_type=HuggingFaceInferenceConfig" in llm_client_content:
        print("✓ LLM client has registration decorator")
    else:
        issues.append("LLM client missing registration decorator")

    if "@register_embedder_client(config_type=HuggingFaceEmbedderConfig" in embedder_client_content:
        print("✓ Embedder client has registration decorator")
    else:
        issues.append("Embedder client missing registration decorator")

    # Check 3: validate_no_responses_api for LLM
    if "validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)" in llm_client_content:
        print("✓ LLM client validates responses API")
    else:
        issues.append("LLM client missing responses API validation")

    # Check 4: RetryMixin patching
    llm_retry_check = "if isinstance(embedder_config, RetryMixin)" in llm_client_content or "_patch_llm_based_on_config" in llm_client_content
    embedder_retry_check = "if isinstance(embedder_config, RetryMixin)" in embedder_client_content

    if llm_retry_check:
        print("✓ LLM client handles RetryMixin")
    else:
        issues.append("LLM client doesn't handle RetryMixin")

    if embedder_retry_check:
        print("✓ Embedder client handles RetryMixin")
    else:
        issues.append("Embedder client doesn't handle RetryMixin")

    # Check 5: Async context manager (yield)
    if "yield" in llm_client_content.split("async def huggingface_inference_langchain")[1].split("@register")[0]:
        print("✓ LLM client yields properly")
    else:
        issues.append("LLM client missing yield")

    if "yield client" in embedder_client_content.split("async def huggingface_langchain")[1]:
        print("✓ Embedder client yields properly")
    else:
        issues.append("Embedder client missing yield")

    return issues


def check_registration():
    """Check that providers are registered in register.py files."""
    print("\n" + "=" * 70)
    print("REGISTRATION CHECKS")
    print("=" * 70)

    base_dir = Path("/Users/bledden/Documents/nemo-agent-toolkit/packages/nvidia_nat_core/src/nat")

    # Check LLM registration
    llm_register = base_dir / "llm" / "register.py"
    with open(llm_register) as f:
        llm_register_content = f.read()

    # Check Embedder registration
    embedder_register = base_dir / "embedder" / "register.py"
    with open(embedder_register) as f:
        embedder_register_content = f.read()

    issues = []

    if "from . import huggingface_inference" in llm_register_content:
        print("✓ huggingface_inference registered in llm/register.py")
    else:
        issues.append("huggingface_inference not registered in llm/register.py")

    if "from . import huggingface_embedder" in embedder_register_content:
        print("✓ huggingface_embedder registered in embedder/register.py")
    else:
        issues.append("huggingface_embedder not registered in embedder/register.py")

    return issues


def main():
    """Run all audits."""
    print("\n" + "=" * 70)
    print("HUGGINGFACE PROVIDERS CODE AUDIT")
    print("=" * 70)

    all_issues = []
    all_warnings = []

    # Audit each file
    files_to_audit = [
        "/Users/bledden/Documents/nemo-agent-toolkit/packages/nvidia_nat_core/src/nat/llm/huggingface_inference.py",
        "/Users/bledden/Documents/nemo-agent-toolkit/packages/nvidia_nat_core/src/nat/embedder/huggingface_embedder.py",
    ]

    for filepath in files_to_audit:
        issues, warnings = audit_file_structure(filepath)
        all_issues.extend(issues)
        all_warnings.extend(warnings)

    # Run consistency checks
    issues = check_consistency()
    all_issues.extend(issues)

    # Run LangChain client checks
    issues = check_langchain_clients()
    all_issues.extend(issues)

    # Run registration checks
    issues = check_registration()
    all_issues.extend(issues)

    # Print summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    if all_issues:
        print(f"\n✗ CRITICAL ISSUES FOUND ({len(all_issues)}):")
        for issue in all_issues:
            print(f"  - {issue}")

    if all_warnings:
        print(f"\n⚠ WARNINGS ({len(all_warnings)}):")
        for warning in all_warnings:
            print(f"  - {warning}")

    if not all_issues and not all_warnings:
        print("\n✓ NO ISSUES FOUND - Code quality looks excellent!")
    elif not all_issues:
        print(f"\n✓ NO CRITICAL ISSUES - Only {len(all_warnings)} minor warnings")
    else:
        print(f"\n✗ FOUND {len(all_issues)} CRITICAL ISSUES")

    print("=" * 70)

    return 1 if all_issues else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
