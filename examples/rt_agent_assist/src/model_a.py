import json
import re
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from .data_models import (
    ModelAResponse, SessionState, FUNCTION_INTENTS, DATA_INTENTS,
    DetectedDataIntent, DetectedFunctionIntent
)
from .mcp_servers import MCPServerManager


class ModelAPrompts:
    """Prompts for Model A (Classifier/Filter)"""
    
    @staticmethod
    def get_classification_prompt(
        transcript: list[str],
        previously_detected_function_intents: list[str],
        previously_detected_data_intents: list[str],
        has_form_submission: bool = False
    ) -> str:
        """Generate the classification prompt for Model A"""
        
        # Build function intents section
        function_lines = []
        for name, intent in FUNCTION_INTENTS.items():
            function_lines.append(f"{name}: {intent.description}")
        
        # Build data intents section
        data_lines = []
        for data_type, intent in DATA_INTENTS.items():
            data_lines.append(f"{data_type.value}: {intent.description}")
        
        # Build requirements mapping
        requirements_lines = []
        for name, intent in FUNCTION_INTENTS.items():
            required = ", ".join([data.value for data in intent.required_data])
            requirements_lines.append(f"{name} -> requires: {required}")
        
        transcript_text = "\n".join(transcript) if transcript else "No transcript available"
        
        previously_triggered_functions = (
            ", ".join(previously_detected_function_intents) 
            if previously_detected_function_intents else "None"
        )
        previously_triggered_data = (
            ", ".join(previously_detected_data_intents) 
            if previously_detected_data_intents else "None"
        )
        
        form_status = (
            "A form submission is ready for processing." 
            if has_form_submission else "No form submissions are ready."
        )
        
        prompt = f"""
You are an Intent Classification AI for analyzing conversation transcripts and detecting new intents.

FUNCTION INTENTS:
{chr(10).join(function_lines)}

DATA INTENTS:
{chr(10).join(data_lines)}

REQUIREMENTS MAPPING:
{chr(10).join(requirements_lines)}

CURRENT TRANSCRIPT:
{transcript_text}

PREVIOUSLY DETECTED:
- Function Intents: {previously_triggered_functions}
- Data Intents: {previously_triggered_data}

FORM SUBMISSION STATUS:
{form_status}

TASK:
1. Analyze the transcript for any NEW function intents that haven't been previously detected
2. For any active function intents (previously detected OR newly detected), identify ALL required data intents that haven't been previously detected
3. Note if there is a form submission ready for processing

RULES:
- Only return NEW intents that haven't been previously detected
- Extract actual VALUES from the transcript for data intents
- Consider the entire transcript context, not just new portions
- If nothing new is detected AND no form submission is ready, the workflow should bypass Model B
- Return JSON ONLY with exactly this structure:

{{
    "new_function_intents": [
        {{"name": "function_intent_name"}}
    ],
    "new_data_intents": [
        {{"name": "data_intent_name", "value": "extracted_value"}}
    ],
    "has_form_submission": {str(has_form_submission).lower()},
    "bypass_model_b": true
}}

- "new_function_intents": list of objects with newly detected function intent names
- "new_data_intents": list of objects with data intent names AND extracted values from transcript
- "has_form_submission": boolean indicating if a form submission is ready
- "bypass_model_b": set to false only if new intents are detected OR form submission is ready

IMPORTANT: For data intents, extract the actual value from the transcript:
- account_id: extract account numbers, IDs mentioned
- customer_name: extract full names mentioned
- transaction_date: extract dates mentioned
- product_type: extract specific product types mentioned

Return JSON only, no explanations.
"""
        return prompt.strip()


class ModelA:
    """Model A: Classifier/Filter for intent detection"""
    
    def __init__(self, builder: Builder, llm_name: str = "llama3.1-8b"):
        self.builder = builder
        self.llm_name = llm_name
        self.llm = None
        self.mcp_manager = MCPServerManager(builder=builder)
    
    async def initialize(self):
        """Initialize the LLM and MCP servers"""
        self.llm = await self.builder.get_llm(self.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        await self.mcp_manager.initialize()
    
    async def classify_intents(
        self,
        session_state: SessionState,
        previously_detected: dict[str, list[str]]
    ) -> ModelAResponse:
        """
        Classify the transcript for new intents and determine if Model B should be bypassed
        """
        if not self.llm:
            await self.initialize()
        
        # Check if there's a form submission ready
        has_form_submission = (
            session_state.latest_form_submission is not None and 
            session_state.latest_form_submission.status == "ready"
        )
        
        # Generate the prompt
        prompt = ModelAPrompts.get_classification_prompt(
            transcript=session_state.transcript,
            previously_detected_function_intents=previously_detected.get("function_intents", []),
            previously_detected_data_intents=previously_detected.get("data_intents", []),
            has_form_submission=has_form_submission
        )
        
        # Get LLM response
        from langchain_core.messages import HumanMessage
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Parse JSON response
            response_data = json.loads(response_text)
            
            # Convert raw response to structured objects
            function_intents = [
                DetectedFunctionIntent(name=intent["name"]) 
                for intent in response_data.get("new_function_intents", [])
            ]
            
            data_intents = [
                DetectedDataIntent(
                    name=intent["name"], 
                    value=intent.get("value")
                ) 
                for intent in response_data.get("new_data_intents", [])
            ]
            
            # Create ModelAResponse
            model_a_response = ModelAResponse(
                new_function_intents=function_intents,
                new_data_intents=data_intents,
                form_submission=session_state.latest_form_submission if has_form_submission else None,
                bypass_model_b=response_data.get("bypass_model_b", True)
            )
            
            # Override bypass_model_b logic: only bypass if no new intents AND no form submission
            if (model_a_response.new_function_intents or
                model_a_response.new_data_intents or
                has_form_submission):
                model_a_response.bypass_model_b = False
            
            # Post-process data intents for date conversion
            await self._convert_relative_dates(model_a_response.new_data_intents)
            
            return model_a_response
            
        except json.JSONDecodeError:
            # Fallback in case of JSON parsing error
            return ModelAResponse(
                new_function_intents=[],
                new_data_intents=[],
                form_submission=session_state.latest_form_submission if has_form_submission else None,
                bypass_model_b=not has_form_submission  # Don't bypass if there's a form submission
            )
        except Exception:
            # Fallback for any other errors
            return ModelAResponse(
                new_function_intents=[],
                new_data_intents=[],
                form_submission=None,
                bypass_model_b=True
            )
    
    async def _convert_relative_dates(self, data_intents: list[DetectedDataIntent]):
        """Convert relative dates in data intents to absolute dates using MCP server"""
        for data_intent in data_intents:
            # Only process transaction_date intents that contain relative date expressions
            if (data_intent.name == "transaction_date" and 
                data_intent.value and 
                self._is_relative_date(data_intent.value)):
                
                try:
                    # Call DateTime conversion MCP server
                    conversion_result = await self.mcp_manager.convert_relative_date(data_intent.value)
                    
                    if (conversion_result.get("status") == "success" and 
                        conversion_result.get("output", {}).get("formatted_date")):
                        
                        # Update the data intent with the converted date
                        original_value = data_intent.value
                        converted_date = conversion_result["output"]["formatted_date"]
                        
                        # Keep both original and converted for reference
                        data_intent.value = converted_date
                        
                        # Add metadata about the conversion
                        output = conversion_result.get("output", {})
                        data_intent.metadata = {
                            "original_value": original_value,
                            "conversion_source": "datetime_conversion_server",
                            "absolute_date": output.get("absolute_date"),
                            "iso_format": output.get("iso_format"),
                            "confidence": output.get("confidence", 0.0),
                            "reasoning": output.get("reasoning", ""),
                            "day_of_week": output.get("day_of_week")
                        }
                        
                except Exception:
                    # If conversion fails, keep the original value
                    pass
    
    def _is_relative_date(self, date_text: str) -> bool:
        """Check if a date expression might need LLM conversion"""
        if not date_text:
            return False
            
        date_lower = date_text.lower().strip()
        
        # Check if it's already a standard date format
        standard_date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\w+ \d{1,2}, \d{4}',  # Month DD, YYYY
        ]
        
        import re
        for pattern in standard_date_patterns:
            if re.match(pattern, date_text):
                return False  # Already in standard format, no conversion needed
        
        # If it's not in standard format, it likely needs conversion
        # This includes: relative dates, partial dates, informal expressions, etc.
        return True
