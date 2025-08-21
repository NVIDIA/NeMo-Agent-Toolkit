import typing
from typing import Any
from pydantic import BaseModel, Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.workflow import WorkflowBaseConfig
from .database_utils import SessionStateDB
from .model_a import ModelA
from .model_b import ModelB


class WorkflowInput(BaseModel):
    """Input schema for the workflow - empty dict as requested"""
    session_id: str = Field(default="default_session", description="Session identifier for database lookup")


class WorkflowOutput(BaseModel):
    """Output schema for the workflow - JSON object with list of UI events"""
    events: list[dict[str, Any]] = Field(default_factory=list, description="List of events to update the UI with")


class RTAgentAssistWorkflowConfig(WorkflowBaseConfig[typing.Literal["rt_agent_assist"]]):
    """Configuration for the RT Agent Assist workflow"""
    
    model_a_llm: str = Field(default="llama3.1-8b", description="LLM to use for Model A (Classifier/Filter)")
    model_b_llm: str = Field(default="llama3.1-8b", description="LLM to use for Model B (Passthrough/MCP)")
    database_path: str = Field(default="session_state.db", description="Path to SQLite database")
    description: str = Field(default="Real-time agent assist workflow for intent detection and UI updates")


@register_function(config_type=RTAgentAssistWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rt_agent_assist_workflow(config: RTAgentAssistWorkflowConfig, builder: Builder):
    """
    Main workflow function for RT Agent Assist
    
    This workflow:
    1. Gets latest session state from database (transcript, events, form submissions)
    2. Uses Model A to classify/filter for new intents
    3. If new intents or form submissions detected, passes to Model B
    4. Model B handles passthrough of intents and MCP server interactions
    5. Returns structured JSON with UI events
    """
    
    # Initialize components
    db = SessionStateDB(config.database_path)
    model_a = ModelA(builder, config.model_a_llm)
    model_b = ModelB(builder, config.model_b_llm)
    
    # Initialize models
    await model_a.initialize()
    await model_b.initialize()
    
    async def _execute_workflow(input_data: dict[str, Any]) -> dict[str, Any]:
        """Main workflow execution function"""
        
        # Parse input
        workflow_input = WorkflowInput(**input_data)
        session_id = workflow_input.session_id
        
        try:
            # Step 1: Get latest session state from database
            session_state = db.get_session_state(session_id)
            
            # Get previously detected intents
            previously_detected = db.get_previously_detected_intents(session_id)
            
            # Step 2: Use Model A to classify/filter for new intents
            model_a_response = await model_a.classify_intents(
                session_state=session_state,
                previously_detected=previously_detected
            )
            
            # Step 3: Check if we should bypass Model B
            if model_a_response.bypass_model_b:
                # No new intents and no form submissions - return empty events
                return WorkflowOutput(events=[]).dict()
            
            # Step 4: Pass to Model B for processing
            model_b_response = await model_b.process_model_a_response(model_a_response)
            
            # Step 5: Update database with new detected events
            new_events = []
            for event in model_b_response.events:
                event_dict = {
                    "type": event.type.value,
                    "data": event.data
                }
                new_events.append(event_dict)
            
            # Add new events to database
            if new_events:
                db.add_detected_events(session_id, new_events)
            
            # Mark form submission as processed if applicable
            if model_a_response.form_submission:
                db.update_form_submission_status(
                    session_id,
                    model_a_response.form_submission.function_name,
                    "processed"
                )
            
            # Step 6: Return structured response
            return WorkflowOutput(events=new_events).dict()
            
        except Exception:
            # Error handling - return empty events with error logging
            # In production, you might want to log this error
            return WorkflowOutput(events=[]).dict()
    
    # Return the function info for the workflow system
    yield FunctionInfo.from_fn(
        _execute_workflow,
        description=config.description
    )


# Additional utility functions for the workflow

async def create_sample_session_data(session_id: str = "test_session"):
    """Helper function to create sample session data for testing"""
    # This would typically be handled by a separate data ingestion process
    # For testing purposes, you can manually insert data into the database
    
    return {
        "session_id": session_id,
        "message": "Sample session data would be created in production system"
    }


async def get_workflow_status(session_id: str = "test_session") -> dict[str, Any]:
    """Helper function to get the current status of a session"""
    db = SessionStateDB()
    session_state = db.get_session_state(session_id)
    previously_detected = db.get_previously_detected_intents(session_id)
    
    return {
        "session_id": session_id,
        "transcript_length": len(session_state.transcript),
        "detected_events_count": len(session_state.detected_events),
        "has_pending_form_submission": session_state.latest_form_submission is not None,
        "previously_detected_function_intents": previously_detected.get("function_intents", []),
        "previously_detected_data_intents": previously_detected.get("data_intents", [])
    }
