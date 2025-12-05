# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Content Safety Guard Middleware.

This middleware uses specialized safety guard models (like Qwen3Guard) to classify
content safety with structured outputs including severity levels and safety categories.
"""

import logging
import re
from typing import Any, AsyncIterator, Optional

from pydantic import Field

from nat.middleware.defense_middleware import DefenseMiddleware, DefenseMiddlewareConfig
from nat.middleware.function_middleware import CallNext, CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class ContentSafetyGuardMiddlewareConfig(DefenseMiddlewareConfig, name="content_safety_guard"):
    """Configuration for Content Safety Guard middleware.
    
    This middleware uses specialized safety guard models (like Qwen3Guard) to classify
    content safety with structured outputs.
    """
    
    llm_name: str = Field(
        description="Name of the guard model LLM (must be defined in llms section)"
    )
    
    severity_threshold: str = Field(
        default="Unsafe",
        description="Minimum severity to trigger action: 'Safe', 'Controversial', or 'Unsafe'"
    )
    
    blocked_categories: list[str] = Field(
        default_factory=lambda: [
            "Violent",
            "Non-violent Illegal Acts",
            "Sexual Content or Sexual Acts",
            "PII",
            "Suicide & Self-Harm",
            "Jailbreak"
        ],
        description="Safety categories that should be blocked"
    )


class ContentSafetyGuardMiddleware(DefenseMiddleware):
    """Safety guard middleware using specialized guard models.
    
    This middleware uses guard models like Qwen3Guard to perform safety classification
    with structured outputs including severity levels and safety categories.
    """
    
    # Severity levels in order
    SEVERITY_LEVELS = ["Safe", "Controversial", "Unsafe"]
    
    def __init__(self, config: ContentSafetyGuardMiddlewareConfig, builder):
        """Initialize content safety guard middleware.
        
        Args:
            config: Configuration for content safety guard middleware
            builder: Builder instance for loading LLMs
        """
        super().__init__(config, builder)
        # Store config with correct type for linter
        self.config: ContentSafetyGuardMiddlewareConfig = config
        self._llm = None  # Lazy loaded LLM
    
    async def _get_llm(self):
        """Lazy load the guard model LLM when first needed.
        
        Tries HuggingFace direct loading first for performance, then falls back
        to standard LLM loading via builder.
        """
        if self._llm is None:
            # Try to get HuggingFace model directly first (no framework overhead)
            try:
                from nat.llm.huggingface_transformers_llm import get_huggingface_model
                llm_config = self.builder.get_llm_config(self.config.llm_name)
                if hasattr(llm_config, 'model_name'):
                    self._llm = get_huggingface_model(llm_config.model_name)
                    if self._llm:
                        return self._llm
            except Exception as e:
                logger.debug(f"Could not get HuggingFace model directly: {e}")
            
            # Fallback to standard LLM loading (works with any LLM type)
            self._llm = await self._get_llm_for_defense(self.config.llm_name)
        return self._llm
    
    def _parse_guard_response(self, response_text: str) -> dict:
        """Parse the guard model response.
        
        Supports two formats:
        1. Qwen3Guard structured format:
           Safety: Safe|Unsafe|Controversial
           Categories: Violent|...
           Refusal: Yes|No
        
        2. Simple Yes/No answer:
           "Yes" = harmful (should block)
           "No" = safe (should pass)
        
        Also detects implicit refusals where LLM refuses to engage.
        
        Returns:
            Dictionary with parsed safety information
        """
        # Strip markdown formatting (**, __, etc.)
        cleaned_text = re.sub(r'[*_]+', '', response_text).strip()
        response_lower = cleaned_text.lower()
        
        # Try Qwen3Guard structured format first
        safety_pattern = r"Safety:\s*(Safe|Unsafe|Controversial)"
        category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
        refusal_pattern = r"Refusal:\s*(Yes|No)"
        
        safety_match = re.search(safety_pattern, response_text, re.IGNORECASE)
        refusal_match = re.search(refusal_pattern, response_text, re.IGNORECASE)
        
        if safety_match or refusal_match:
            # Qwen3Guard format detected
            safety_label = safety_match.group(1) if safety_match else "Safe"
            refusal = refusal_match.group(1) if refusal_match else None
            categories = re.findall(category_pattern, response_text)
            categories = [cat for cat in categories if cat != "None"]
        else:
            # Simple Yes/No format
            # "Yes" means content IS harmful
            if cleaned_text.lower().startswith("yes"):
                safety_label = "Unsafe"
                refusal = "Yes"
                categories = []
            # "No" means content is NOT harmful
            elif cleaned_text.lower().startswith("no"):
                safety_label = "Safe"
                refusal = "No"
                categories = []
            else:
                # Detect implicit refusals (LLM refuses to engage = harmful detected)
                refusal_indicators = [
                    "i cannot", "i can't", "i won't", "i will not",
                    "i'm unable", "i am unable", "i refuse",
                    "cannot provide", "can't provide", "won't provide",
                    "illegal", "harmful", "dangerous", "inappropriate",
                    "against", "violates", "unethical"
                ]
                
                if any(indicator in response_lower for indicator in refusal_indicators):
                    safety_label = "Unsafe"
                    refusal = "Yes"
                    categories = []
                else:
                    # Default to safe if unclear
                    safety_label = "Safe"
                    refusal = "No"
                    categories = []
            
            # Infer categories from content (optional, for logging)
            if "bomb" in response_lower or "weapon" in response_lower or "explosive" in response_lower:
                if "Violent" not in categories:
                    categories.append("Violent")
            if "suicide" in response_lower or "self-harm" in response_lower:
                if "Suicide & Self-Harm" not in categories:
                    categories.append("Suicide & Self-Harm")
        
        return {
            "safety": safety_label,
            "categories": categories,
            "refusal": refusal,
            "raw_response": response_text
        }
    
    def _should_block(self, parsed_result: dict) -> bool:
        """Determine if content should be blocked based on config."""
        safety = parsed_result["safety"]
        categories = parsed_result["categories"]
        refusal = parsed_result.get("refusal")
        
        # Block if LLM refused to engage (indicates harmful content)
        if refusal == "Yes":
            logger.info(f"Content Safety Guard: Blocking due to LLM refusal (detected harmful content)")
            return True
        
        # Check severity threshold
        threshold_index = self.SEVERITY_LEVELS.index(self.config.severity_threshold)
        content_index = self.SEVERITY_LEVELS.index(safety)
        
        # Block if severity is at or above threshold
        if content_index < threshold_index:
            return False
        
        # Check if any category matches blocked categories
        if any(cat in self.config.blocked_categories for cat in categories):
            return True
        
        # If Unsafe but no specific blocked category, still block
        if safety == "Unsafe":
            return True
        
        return False
    
    async def _analyze_content(
        self, 
        content: Any, 
        content_type: str, 
        original_input: Any = None,
        context: Optional[FunctionMiddlewareContext] = None
    ) -> dict:
        """Check content safety using guard model.
        
        Args:
            content: The content to analyze
            content_type: Either 'input' or 'output'
            original_input: The original input to the function (for context)
            context: Function metadata
        
        Returns:
            Safety classification result with should_block flag
        """
        try:
            # Get the guard model LLM
            llm = await self._get_llm()
            
            # Prepare simple prompt - just ask if content is harmful
            content_str = str(content)
            
            if content_type == "output" and original_input is not None:
                prompt = f"""Is this AI response harmful or dangerous?

User Question: {str(original_input)}
AI Response: {content_str}

Answer Yes or No."""
            else:
                prompt = f"""Is this content harmful or dangerous?

Content: {content_str}

Answer Yes or No."""
            
            messages = [{"role": "user", "content": prompt}]
            
            # Call the guard model
            response = await llm.ainvoke(messages)
            
            # Extract text from response
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            elif isinstance(response, str):
                response_text = response.strip()
            else:
                response_text = str(response).strip()
            
            # Parse the guard model response
            parsed = self._parse_guard_response(response_text)
            parsed["content_type"] = content_type
            parsed["should_block"] = self._should_block(parsed)
            
            return parsed
        
        except Exception as e:
            logger.error(f"Content Safety Guard analysis failed: {e}")
            return {
                "safety": "Safe",
                "refusal": "No",
                "content_type": content_type,
                "should_block": False,
                "error": True,
                "error_message": str(e)
            }
    
    async def _handle_threat(
        self,
        content: Any,
        analysis_result: dict,
        context: FunctionMiddlewareContext
    ) -> Any:
        """Handle unsafe content based on configured action.
        
        Args:
            content: The unsafe content
            analysis_result: Safety classification result
            context: Function context
        
        Returns:
            Handled content (blocked, sanitized, or original)
        """
        action = self.config.action
        
        logger.warning(
            f"Content Safety Guard detected {analysis_result['safety']} content in {context.name}: "
            f"Categories={analysis_result['categories']}"
        )
        
        if action == "block":
            return {
                "error": "Content blocked by safety policy",
                "safety_violation": True,
                "severity": analysis_result["safety"],
                "categories": analysis_result["categories"]
            }
        
        elif action == "sanitize":
            return "[CONTENT BLOCKED BY SAFETY POLICY]"
        
        else:  # action == "log"
            logger.warning(
                f"Safety violation logged but content passed through: "
                f"{analysis_result['safety']} - {analysis_result['categories']}"
            )
            return content
    
    async def function_middleware_invoke(
        self, value: Any, call_next: CallNext, context: FunctionMiddlewareContext
    ) -> Any:
        """Apply content safety guard check to function invocation.
        
        This is the core logic for content safety guard defense - each defense implements
        its own invoke/stream based on its specific strategy.
        
        Args:
            value: Function input (original_input from context)
            call_next: Next middleware/function to call
            context: Function metadata (provides context state)
        
        Returns:
            Function output (potentially blocked or sanitized)
        """
        # Check if defense should apply to this function
        if not self._should_apply_defense(context.name):
            logger.info(f"ContentSafetyGuardMiddleware: Skipping {context.name} (target={self.config.target_function_or_group}, not targeted)")
            return await call_next(value)
        
        logger.info(f"ContentSafetyGuardMiddleware: APPLYING defense to {context.name}")
        
        # Check input if configured
        if self.config.check_input:
            logger.debug(f"ContentSafetyGuardMiddleware: Checking input for {context.name}")
            input_result = await self._analyze_content(value, "input", context=context)
            if input_result["should_block"]:
                return await self._handle_threat(value, input_result, context)
        
        # Call the function (modified_input would be in context if needed)
        output = await call_next(value)
        
        # Check output if configured (pass original input for context)
        if self.config.check_output:
            logger.info(f"ContentSafetyGuardMiddleware: Checking OUTPUT for {context.name}, output={str(output)[:100]}")
            output_result = await self._analyze_content(
                output, "output", original_input=value, context=context
            )
            logger.info(f"ContentSafetyGuardMiddleware: Analysis result: {output_result}")
            if output_result["should_block"]:
                logger.info(f"ContentSafetyGuardMiddleware: BLOCKING output for {context.name}")
                return await self._handle_threat(output, output_result, context)
        
        logger.info(f"ContentSafetyGuardMiddleware: Output passed all checks for {context.name}")
        return output
    
    async def function_middleware_stream(
        self, value: Any, call_next: CallNextStream, context: FunctionMiddlewareContext
    ) -> AsyncIterator[Any]:
        """Apply content safety guard check to streaming function.
        
        Args:
            value: Function input
            call_next: Next middleware/function to call
            context: Function metadata
        
        Yields:
            Function output chunks (potentially blocked or sanitized)
        """
        # Check if defense should apply to this function
        if not self._should_apply_defense(context.name):
            logger.debug(f"ContentSafetyGuardMiddleware: Skipping {context.name} (not targeted)")
            async for chunk in call_next(value):
                yield chunk
            return
        
        # Check input if configured
        if self.config.check_input:
            input_result = await self._analyze_content(value, "input", context=context)
            if input_result["should_block"]:
                handled = await self._handle_threat(value, input_result, context)
                yield handled
                return
        
        # Stream and optionally check output
        if self.config.check_output:
            # Accumulate chunks to analyze after streaming
            accumulated_output = []
            async for chunk in call_next(value):
                accumulated_output.append(str(chunk))
                yield chunk
            
            # Final check after streaming completes
            if accumulated_output:
                full_output = "".join(accumulated_output)
                output_result = await self._analyze_content(
                    full_output, "output", original_input=value, context=context
                )
                if output_result["should_block"]:
                    logger.warning(
                        f"Streaming output violated safety policy: "
                        f"{output_result['safety']} - {output_result['categories']}"
                    )
        else:
            # Just pass through without checking
            async for chunk in call_next(value):
                yield chunk

