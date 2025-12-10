# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Content Safety Guard Middleware.

This middleware uses guard models to classify content as safe or harmful
with simple Yes/No answers.
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
    
    This middleware uses guard models to classify content as safe or harmful.
    
    Actions:
    - 'log': Pass through with warning
    - 'block': Return error dict (hard stop)
    - 'sanitize': Return polite refusal message (user-friendly)
    """
    
    llm_name: str = Field(
        description="Name of the guard model LLM (must be defined in llms section)"
    )


class ContentSafetyGuardMiddleware(DefenseMiddleware):
    """Safety guard middleware using guard models for Yes/No classification.
    
    This middleware asks a guard model: "Is this content harmful?"
    - Yes = block
    - No = allow
    """
    
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
        """Lazy load the guard model LLM when first needed."""
        if self._llm is None:
            self._llm = await self._get_llm_for_defense(self.config.llm_name)
        return self._llm
    
    def _parse_guard_response(self, response_text: str) -> dict:
        """Parse guard model response - simple Yes/No classification.
        
        Expected response:
        - "Yes" = content is harmful/unsafe (should block)
        - "No" = content is safe (should pass)
        
        Also detects implicit refusals where model refuses to engage.
        
        Returns:
            Dictionary with is_safe boolean and raw response
        """
        # Strip markdown formatting
        cleaned_text = re.sub(r'[*_]+', '', response_text).strip()
        response_lower = cleaned_text.lower()
        
        # Check for explicit Yes (harmful) or No (safe)
        if cleaned_text.lower().startswith("yes"):
            is_safe = False
        elif cleaned_text.lower().startswith("no"):
            is_safe = True
        else:
            # Detect implicit refusals (model refuses = harmful content detected)
            refusal_indicators = [
                "i cannot", "i can't", "i won't", "i will not",
                "i'm unable", "i am unable", "i refuse",
                "cannot provide", "can't provide", "won't provide",
                "illegal", "harmful", "dangerous", "inappropriate"
            ]
            
            if any(indicator in response_lower for indicator in refusal_indicators):
                is_safe = False  # Model refused = harmful detected
            else:
                is_safe = True  # Default to safe if unclear
        
        return {
            "is_safe": is_safe,
            "raw_response": response_text
        }
    
    def _should_block(self, parsed_result: dict) -> bool:
        """Determine if content should be blocked.
        
        Args:
            parsed_result: Result from _parse_guard_response
            
        Returns:
            True if content should be blocked
        """
        return not parsed_result.get("is_safe", True)
    
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
            logger.error("Content Safety Guard analysis failed for %s: %s", content_type, e, exc_info=True)
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
            "Content Safety Guard detected unsafe content in %s",
            context.name
        )
        
        if action == "block":
            return {
                "error": "Content blocked by safety policy",
                "safety_violation": True
            }
        
        elif action == "sanitize":
            return "I'm sorry, I cannot help you with that request."
        
        else:  # action == "log"
            logger.warning("Safety violation logged but content passed through")
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
            logger.debug("ContentSafetyGuardMiddleware: Skipping %s (not targeted)", context.name)
            return await call_next(value)
        
        logger.debug("ContentSafetyGuardMiddleware: Applying defense to %s", context.name)
        
        # Call the function
        output = await call_next(value)
        
        # Check output
        logger.debug("ContentSafetyGuardMiddleware: Checking output for %s", context.name)
        output_result = await self._analyze_content(
            output, "output", original_input=value, context=context
        )
        if output_result.get("should_block", False):
                logger.warning(
                    "ContentSafetyGuardMiddleware: Blocking output for %s (unsafe content detected)",
                    context.name
                )
                return await self._handle_threat(output, output_result, context)
        
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
            logger.debug("ContentSafetyGuardMiddleware: Skipping %s (not targeted)", context.name)
            async for chunk in call_next(value):
                yield chunk
            return
        
        # Stream and check output
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
            if output_result.get("should_block", False):
                logger.warning(
                    "Streaming output violated safety policy (unsafe content detected)"
                )

