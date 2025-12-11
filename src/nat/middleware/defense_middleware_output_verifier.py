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
Output Verifier Defense Middleware.

This middleware uses an LLM to verify function outputs for correctness and security.
It can detect incorrect results, malicious content, and provide corrections automatically.
"""

import json
import logging
import re
from typing import Any, AsyncIterator, Optional

from pydantic import Field

from nat.middleware.defense_middleware import DefenseMiddleware, DefenseMiddlewareConfig
from nat.middleware.function_middleware import CallNext, CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext

logger = logging.getLogger(__name__)


class OutputVerifierMiddlewareConfig(DefenseMiddlewareConfig, name="output_verifier"):
    """Configuration for Output Verifier middleware.
    
    This middleware analyzes function outputs using an LLM to verify correctness,
    detect security threats, and provide corrections when needed.
    
    Actions:
    - 'partial_compliance': Detect and log threats, but allow content to pass through
    - 'refusal': Block output if threat detected (hard stop)
    - 'redirection': Replace incorrect output with correct answer from LLM
    
    Note: Only output analysis is currently supported (target_location='output').
    """
    
    llm_name: str = Field(
        description="Name of the LLM to use for verification (must be defined in llms section)"
    )
    
    threshold: float = Field(
        default=0.7,
        description="Confidence threshold for threat detection (0.0-1.0)"
    )
    
    tool_description: Optional[str] = Field(
        default=None,
        description="Description of what the tool/function does (optional, helps LLM verify correctness)"
    )


class OutputVerifierMiddleware(DefenseMiddleware):
    """Verification middleware using an LLM for correctness and security.
    
    This middleware uses NAT's LLM system to verify function outputs for:
    - Correctness and reasonableness
    - Security validation (detecting malicious content and manipulated values)
    - Providing automatic corrections when errors are detected
    
    Only output analysis is currently supported (target_location='output').
    """
    
    def __init__(self, config: OutputVerifierMiddlewareConfig, builder):
        """Initialize output verifier middleware.
        
        Args:
            config: Configuration for output verifier middleware
            builder: Builder instance for loading LLMs
        """
        super().__init__(config, builder)
        # Store config with correct type for linter
        self.config: OutputVerifierMiddlewareConfig = config
        
        # Output Verifier only supports output analysis
        if config.target_location == "input":
            raise ValueError(
                "OutputVerifierMiddleware only supports target_location='output'. "
                "Input analysis is not yet supported."
            )
        
        self._llm = None  # Lazy loaded LLM
    
    async def _get_llm(self):
        """Lazy load the LLM when first needed."""
        if self._llm is None:
            self._llm = await self._get_llm_for_defense(self.config.llm_name)
        return self._llm
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks.
        
        Args:
            response_text: Raw response from LLM
        
        Returns:
            Extracted JSON string
        """
        # Remove markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Extract JSON object (handles nested braces)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return response_text
    
    async def _analyze_content(
        self, 
        content: Any, 
        content_type: str, 
        inputs: Any = None,
        function_name: str | None = None
    ) -> dict:
        """Check content for threats using the configured LLM.
        
        Args:
            content: The content to analyze
            content_type: Either 'input' or 'output' (for logging only)
            inputs: Optional function inputs for context (helps LLM calculate correct answers)
            function_name: Name of the function being verified (for context)
        
        Returns:
            Detection result with threat info and should_refuse flag
        """
        content_str = str(content)
        
        # Build structured system prompt
        system_prompt = """You are a function output verifier that validates correctness and detects security threats.

Your task:
1. Verify if the output is correct and reasonable given the function name and inputs
2. Detect suspicious or manipulated values (extreme values, unexpected types, malicious content)
3. If incorrect, provide the correct output
4. Only report high confidence when certain

Respond ONLY with valid JSON in this exact format:
{"threat_detected": true/false, "confidence": 0.0-1.0, "reason": "brief explanation", "correct_answer": "correct value if wrong, null if correct"}"""
        
        # Build user prompt with function context
        user_prompt_parts = []
        
        if function_name:
            user_prompt_parts.append(f"Function: {function_name}")
        
        if self.config.tool_description:
            user_prompt_parts.append(f"Description: {self.config.tool_description}")
        
        if inputs is not None:
            user_prompt_parts.append(f"Inputs: {inputs}")
        
        user_prompt_parts.append(f"{content_type.capitalize()}: {content_str}")
        
        prompt = "\n".join(user_prompt_parts)
        
        response_text = None
        try:
            # Get the LLM (lazy loaded)
            llm = await self._get_llm()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm.ainvoke(messages)
            
            # Extract text from response
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            elif isinstance(response, str):
                response_text = response.strip()
            else:
                response_text = str(response).strip()
            
            # Extract and parse JSON
            json_str = self._extract_json_from_response(response_text)
            result = json.loads(json_str)
            
            threat_detected = result.get("threat_detected", False)
            confidence = float(result.get("confidence", 0.0))
            
            return {
                "threat_detected": threat_detected,
                "confidence": confidence,
                "reason": result.get("reason", "Unknown"),
                "correct_answer": result.get("correct_answer"),
                "content_type": content_type,
                "should_refuse": threat_detected and confidence >= self.config.threshold
            }
        
        except Exception as e:
            logger.error("Output Verifier analysis failed for %s: %s", content_type, e, exc_info=True)
            logger.debug("Failed response: %s", response_text if response_text else "N/A")
            return {
                "threat_detected": False,
                "confidence": 0.0,
                "reason": f"Analysis failed: {e}",
                "content_type": content_type,
                "should_refuse": False,
                "error": True
            }
    
    async def _handle_threat(
        self,
        content: Any,
        analysis_result: dict,
        context: FunctionMiddlewareContext
    ) -> Any:
        """Handle detected threat based on configured action.
        
        Args:
            content: The threatening content
            analysis_result: Detection result from LLM
            context: Function context
        
        Returns:
            Handled content (blocked, sanitized/corrected, or original)
        """
        logger.warning(
            "Output Verifier detected threat in %s: %s (confidence=%s)",
            context.name,
            analysis_result.get('reason', 'Unknown'),
            analysis_result.get('confidence', 0.0)
        )
        
        action = self.config.action
        
        if action == "refusal":
            logger.error("Output Verifier refusing output of %s: %s", context.name, analysis_result.get('reason', 'Unknown'))
            raise ValueError(f"Content blocked by security policy: {analysis_result.get('reason', 'Unknown')}")
        
        elif action == "redirection":
            # Redirection = Replace with correct answer if available
            correct_answer = analysis_result.get("correct_answer")
            
            if correct_answer is not None:
                # Try to convert to same type as original content
                if isinstance(content, (int, float)):
                    try:
                        correct_answer = float(correct_answer)
                    except (ValueError, TypeError):
                        logger.warning("Could not convert '%s' to number", correct_answer)
                
                logger.info(
                    "Output Verifier redirecting %s: Incorrect: %s → Corrected: %s",
                    context.name,
                    content,
                    correct_answer
                )
                return correct_answer
            else:
                # No correction available, return safe placeholder
                logger.info("Redirecting %s (no correction available)", context.name)
                return {"error": "Content sanitized by security policy", "original_blocked": True}
        
        else:  # action == "partial_compliance"
            logger.warning("Threat logged for %s: %s", context.name, analysis_result.get('reason', 'Unknown'))
            return content
    
    async def function_middleware_invoke(
        self, value: Any, call_next: CallNext, context: FunctionMiddlewareContext
    ) -> Any:
        """Apply output verifier to function invocation.
        
        Analyzes function outputs for correctness and security, with auto-correction.
        
        Args:
            value: Function input
            call_next: Next middleware/function to call
            context: Function metadata
        
        Returns:
            Function output (potentially corrected, blocked, or sanitized)
        """
        # Check if defense should apply to this function
        if not self._should_apply_defense(context.name):
            logger.debug("OutputVerifierMiddleware: Skipping %s (not targeted)", context.name)
            return await call_next(value)
        
        try:
            # Call the function
            output = await call_next(value)
            
            # Extract field from output if target_field is specified
            content_to_analyze, field_info = self._extract_field_from_value(output)
            
            # Check the output (either extracted field or entire output)
            logger.debug(
                "OutputVerifierMiddleware: Checking %s for %s",
                f"field '{self.config.target_field}'" if field_info else "output",
                context.name
            )
            output_result = await self._analyze_content(
                content_to_analyze, 
                "output", 
                inputs=value,
                function_name=context.name
            )
            
            if output_result.get("should_refuse", False):
                # Handle threat - get sanitized/corrected value
                sanitized_content = await self._handle_threat(content_to_analyze, output_result, context)
                
                # If field was extracted, apply sanitized value back to original structure
                if field_info is not None:
                    return self._apply_field_result_to_value(output, field_info, sanitized_content)
                else:
                    # No field extraction - return sanitized content directly
                    return sanitized_content
            
            return output
        
        except Exception as e:
            logger.error("Failed to apply output verification to function %s: %s", context.name, e, exc_info=True)
            raise
    
    async def function_middleware_stream(
        self, value: Any, call_next: CallNextStream, context: FunctionMiddlewareContext
    ) -> AsyncIterator[Any]:
        """Apply output verifier to streaming function.
        
        Args:
            value: Function input
            call_next: Next middleware/function to call
            context: Function metadata
        
        Yields:
            Function output chunks (potentially corrected, blocked, or sanitized)
        """
        # Check if defense should apply to this function
        if not self._should_apply_defense(context.name):
            logger.debug("OutputVerifierMiddleware: Skipping %s (not targeted)", context.name)
            async for chunk in call_next(value):
                yield chunk
            return
        
        try:
            # Accumulate chunks to analyze after streaming
            accumulated_output = []
            async for chunk in call_next(value):
                accumulated_output.append(str(chunk))
                yield chunk
            
            # Final check after streaming completes
            if accumulated_output:
                full_output = "".join(accumulated_output)
                output_result = await self._analyze_content(
                    full_output, 
                    "output", 
                    inputs=value,
                    function_name=context.name
                )
                if output_result.get("should_refuse", False):
                    logger.warning(
                        "Streaming output failed verification: %s",
                        output_result.get('reason', 'Unknown')
                    )
        
        except Exception as e:
            logger.error(
                "Failed to apply output verification to streaming function %s: %s", context.name, e, exc_info=True
            )
            raise

