from typing import Any
import json
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from .data_models import ModelAResponse, ModelBResponse, UIEvent, EventType, FormSubmission
from .mcp_servers import MCPServerManager


class ModelBPrompts:
    """Prompts for Model B (Passthrough/MCP interaction)"""
    
    @staticmethod
    def get_passthrough_prompt(
        new_function_intents: list,
        new_data_intents: list
    ) -> str:
        """Generate prompt for passthrough of intent information"""
        
        function_intent_names = [intent.name for intent in new_function_intents]
        data_intent_details = [f"{intent.name}: {intent.value}" for intent in new_data_intents]
        
        prompt = f"""
You are a structured JSON passthrough system. Your job is to convert detected intents into structured UI events.

DETECTED INTENTS:
- New Function Intents: {', '.join(function_intent_names) if function_intent_names else 'None'}
- New Data Intents: {', '.join(data_intent_details) if data_intent_details else 'None'}

TASK:
Convert these detected intents into a structured JSON response for UI updates.

RULES:
- Return JSON ONLY with exactly this structure
- No explanations, no additional content
- Each intent becomes a separate event

{{
    "events": [
        // For each function intent:
        {{
            "type": "function_intent",
            "data": {{
                "name": "function_intent_name"
            }}
        }},
        // For each data intent:
        {{
            "type": "data_intent", 
            "data": {{
                "name": "data_intent_name",
                "value": "extracted_value"
            }}
        }}
    ]
}}

Return JSON only.
"""
        return prompt.strip()


class ModelB:
    """Model B: Passthrough and MCP server interaction"""
    
    def __init__(self, builder: Builder, llm_name: str = "llama3.1-8b"):
        self.builder = builder
        self.llm_name = llm_name
        self.llm = None
        self.mcp_manager = MCPServerManager()
    
    async def initialize(self):
        """Initialize the LLM"""
        self.llm = await self.builder.get_llm(self.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def process_model_a_response(self, model_a_response: ModelAResponse) -> ModelBResponse:
        """
        Process the response from Model A and generate UI events
        """
        if not self.llm:
            await self.initialize()
        
        events = []
        mcp_results = None
        
        # Handle form submission if present
        if model_a_response.form_submission:
            mcp_results = await self._handle_form_submission(model_a_response.form_submission)
            
            # Create form submission event
            form_event = UIEvent(
                type=EventType.FORM_SUBMISSION,
                data={
                    "function_name": model_a_response.form_submission.function_name,
                    "status": "processed",
                    "mcp_result": mcp_results
                }
            )
            events.append(form_event)
        
        # Handle new intents if present
        if model_a_response.new_function_intents or model_a_response.new_data_intents:
            intent_events = await self._handle_new_intents(
                model_a_response.new_function_intents,
                model_a_response.new_data_intents
            )
            events.extend(intent_events)
        
        return ModelBResponse(
            events=events,
            mcp_results=mcp_results
        )
    
    async def _handle_form_submission(self, form_submission: FormSubmission) -> dict[str, Any]:
        """Handle form submission by calling appropriate MCP server"""
        try:
            result = await self.mcp_manager.process_form_submission(form_submission)
            return result
        except Exception as e:
            return {
                "error": f"Failed to process form submission: {str(e)}",
                "status": "error"
            }
    
    async def _handle_new_intents(
        self,
        new_function_intents: list[str],
        new_data_intents: list[str]
    ) -> list[UIEvent]:
        """Handle new intents by generating passthrough events"""
        
        # Generate prompt for passthrough
        prompt = ModelBPrompts.get_passthrough_prompt(
            new_function_intents,
            new_data_intents
        )
        
        try:
            from langchain_core.messages import HumanMessage
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Parse JSON response
            response_data = json.loads(response_text)
            
            # Convert to UIEvent objects
            events = []
            for event_data in response_data.get("events", []):
                event = UIEvent(
                    type=EventType(event_data["type"]),
                    data=event_data["data"]
                )
                events.append(event)
            
            return events
            
        except json.JSONDecodeError:
            # Fallback: create events manually
            return self._create_fallback_events(new_function_intents, new_data_intents)
        except Exception:
            # Fallback: create events manually
            return self._create_fallback_events(new_function_intents, new_data_intents)
    
    def _create_fallback_events(
        self,
        new_function_intents: list,
        new_data_intents: list
    ) -> list[UIEvent]:
        """Create events manually as fallback"""
        events = []
        
        # Create function intent events
        for function_intent in new_function_intents:
            event = UIEvent(
                type=EventType.FUNCTION_INTENT,
                data={"name": function_intent.name}
            )
            events.append(event)
        
        # Create data intent events
        for data_intent in new_data_intents:
            event_data = {
                "name": data_intent.name,
                "value": data_intent.value
            }
            
            # Include metadata if available (e.g., date conversion info)
            if data_intent.metadata:
                event_data["metadata"] = data_intent.metadata
            
            event = UIEvent(
                type=EventType.DATA_INTENT,
                data=event_data
            )
            events.append(event)
        
        return events
