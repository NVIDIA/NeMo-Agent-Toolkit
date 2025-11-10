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

# Security Considerations

## Overview

NVIDIA NeMo Agent Toolkit is a framework that enables you to build complex agentic systems that can interact with external resources such as file systems, databases, APIs, and other tools. This new level of autonomy and capability brings new security considerations that are important to understand as you build and deploy your applications.

Building secure agentic applications depends on understanding the security implications of your implementation decisions in the areas outlined below.

For additional guidance, see the references below and consult the security best practices blogs published by NVIDIA.

## What to Be Aware Of

When building applications with NeMo Agent Toolkit, it's helpful to be aware of these potential risks:

### Tool Abuse and Misuse

Agentic systems can act with significant autonomy, which means understanding potential failure modes is important. Large Language Models (LLMs) can make mistakes or can be manipulated to take unintended actions. When agents have access to powerful tools, these mistakes can result in:

- **Unauthorized data access**: Agents reading files or database records they should not access
- **Data modification or deletion**: Agents writing, updating, or deleting data inappropriately
- **Unintended API calls**: Agents making external API calls that were not authorized
- **Command execution**: Agents executing system commands that compromise security
- **Resource exhaustion or consumption**: Agents making excessive requests that degrade service availability or accrue excessive costs

### Loss of System Integrity

When agents are capable of writing data to external systems, it's important to consider how this could affect your infrastructure's integrity:

- **Configuration changes**: Unauthorized modifications to system configurations
- **File system corruption**: Deletion or modification of critical files
- **Malicious code injection**: Writing malicious scripts or code to accessible locations
- **Remote code execution**: Writing malicious scripts or code to locations where it might be automatically executed
- **Service disruption**: Actions that cause services to fail or become unavailable

### Loss of Confidentiality

Agents can inadvertently or intentionally expose sensitive information, you should take care to segregate sensitive data from agents wherever possible.  When that is not possible, it's worth understanding these scenarios:

- **Data leakage or exfiltration**: Sensitive data being written to logs, external APIs, or publicly accessible locations
- **Credential exposure**: API keys, passwords, or tokens being logged or transmitted insecurely or to external or adversarially controlled endpoints
- **PII leakage**: Personally identifiable information being shared inappropriately to unauthorized users, services, or third parties
- **Intellectual property disclosure**: Proprietary information being exposed to unauthorized parties, either directly or via indirect mechanisms such as logs or data derived from logs
- **Cross-tenant data leakage**: In multi-tenant systems, data from one tenant being accessible to another

### Logging and Observability Security Considerations

The observability and profiling features in NeMo Agent Toolkit capture detailed information about agent behavior, including the ability to capture LLM prompts and responses, which brings its own considerations:

- **Sensitive data in logs**: User inputs, API responses, and intermediate results may contain sensitive, personal, confidential, or regulated information depending on user input and systems that the agent is permitted to access
- **Credential logging**: API keys, tokens, or credentials may be written to log files
- **Audit trail exposure**: Logs revealing system architecture or security measures
- **Log storage security**: Insufficient protection of stored logs containing sensitive data
- **Log retention policies**: Keeping logs longer than necessary, increasing exposure risk
- **Log access control policies:** Making logs available may inadvertently violate access control on source data, including data manually entered into prompts, or data collected and inserted into prompts by tools, such as an MCP tool, that use delegated authorization from the user.

### Supply Chain and Third Party Security Aspects

When integrating with external dependencies and services, consider these supply chain security aspects:

- **Vulnerable dependencies**: Third party packages and libraries with known CVEs that can be exploited
- **Outdated or unmaintained software**: Using components that no longer receive security updates or patches
- **Malicious third party tools**: Integrating with external tools or plugins that contain backdoors or malicious functionality
- **Insecure third party APIs**: External services with insufficient authentication, authorization, or data protection
- **Model supply chain risks**: Using models from untrusted sources that may contain backdoors or biases – consider using only [signed models](https://developer.nvidia.com/blog/bringing-verifiable-trust-to-ai-models-model-signing-in-ngc/) from trusted sources

## Example Security Approaches

**Tool Abuse and Misuse**

- Guardrails - Input/output validation and content filtering to prevent misuse
- RBAC (Role Based Access Control) - Limit agent permissions to specific resources
- Rate Limiting/Throttling - Prevent resource exhaustion
- Sandboxing - Isolate agent execution

**Loss of System Integrity**

- Sandboxing/Containerization - WebAssembly, containers, container runtime sandboxing for isolated execution
- Least Privilege Access Controls - Minimize write permissions

**Loss of Confidentiality**

- Secret Management - Secure use of secrets, e.g. Key Vault
- Encryption - At rest, in transit

**Logging and Observability Security**

- Log Sanitization/Scrubbing - Remove sensitive data before logging
- Secret Scanning - Detect credentials in logs
- SIEM (Security Information and Event Management) - Secure log management
- Log Encryption - Protect stored logs
- Access Control for Logs - RBAC for log viewing

**Supply Chain and Third Party Security**

- SBOM (Software Bill of Materials) - Track and verify components
- Model Signing & Verification - Ensure model integrity and authenticity
- Vulnerability Scanners - Detect CVEs

## References

* *Practical LLM Security Advice from the NVIDIA AI Red Team*, [https://developer.nvidia.com/blog/practical-llm-security-advice-from-the-nvidia-ai-red-team/](https://developer.nvidia.com/blog/practical-llm-security-advice-from-the-nvidia-ai-red-team/)
* *Modeling Attacks on AI-Powered Apps with the AI Kill Chain Framework*, [https://developer.nvidia.com/blog/modeling-attacks-on-ai-powered-apps-with-the-ai-kill-chain-framework/](https://developer.nvidia.com/blog/modeling-attacks-on-ai-powered-apps-with-the-ai-kill-chain-framework/)
* *Agentic Autonomy Levels and Security*, [https://developer.nvidia.com/blog/agentic-autonomy-levels-and-security/](https://developer.nvidia.com/blog/agentic-autonomy-levels-and-security/)
* *Sandboxing Agentic AI Workflows with WebAssembly*, [https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/](https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/)
* *Bringing Verifiable Trust to AI Models: Model Signing in NGC*, [https://developer.nvidia.com/blog/bringing-verifiable-trust-to-ai-models-model-signing-in-ngc/](https://developer.nvidia.com/blog/bringing-verifiable-trust-to-ai-models-model-signing-in-ngc/)
* *Securing Generative AI Deployments with NVIDIA NIM and NVIDIA NeMo Guardrails*, [https://developer.nvidia.com/blog/securing-generative-ai-deployments-with-nvidia-nim-and-nvidia-nemo-guardrails/](https://developer.nvidia.com/blog/securing-generative-ai-deployments-with-nvidia-nim-and-nvidia-nemo-guardrails/)
