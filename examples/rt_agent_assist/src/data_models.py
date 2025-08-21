from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class DataIntentType(str, Enum):
    """Enumeration of data intent types"""
    ACCOUNT_ID = "account_id"
    TRANSACTION_DATE = "transaction_date"
    CUSTOMER_NAME = "customer_name"
    PRODUCT_TYPE = "product_type"


class DataIntent(BaseModel):
    """Represents a data intent that needs to be collected"""
    name: DataIntentType
    description: str
    value: Optional[str] = None
    collected: bool = False


class FunctionIntent(BaseModel):
    """Represents a function intent that can be triggered"""
    name: str
    description: str
    required_data: list[DataIntentType]
    mcp_server: str
    triggered: bool = False


class EventType(str, Enum):
    """Types of events that can update the UI"""
    FUNCTION_INTENT = "function_intent"
    DATA_INTENT = "data_intent"
    FORM_SUBMISSION = "form_submission"


class UIEvent(BaseModel):
    """Represents an event to update the UI"""
    type: EventType
    data: dict[str, Any]
    timestamp: Optional[str] = None


class FormSubmission(BaseModel):
    """Represents a form submission from the UI"""
    function_name: str
    data: dict[str, str]
    status: str = "ready"  # ready, processed, etc.
    timestamp: Optional[str] = None


class SessionState(BaseModel):
    """Represents the current session state"""
    transcript: list[str]
    detected_events: list[dict[str, Any]]
    latest_form_submission: Optional[FormSubmission] = None


class DetectedDataIntent(BaseModel):
    """Represents a detected data intent with its extracted value"""
    name: str
    value: Optional[str] = None
    detected: bool = True
    metadata: Optional[dict[str, Any]] = None


class DetectedFunctionIntent(BaseModel):
    """Represents a detected function intent"""
    name: str
    detected: bool = True


class ModelAResponse(BaseModel):
    """Response from Model A (Classifier/Filter)"""
    new_function_intents: list[DetectedFunctionIntent] = Field(default_factory=list)
    new_data_intents: list[DetectedDataIntent] = Field(default_factory=list)
    form_submission: Optional[FormSubmission] = None
    bypass_model_b: bool = True


class ModelBResponse(BaseModel):
    """Response from Model B (Passthrough/MCP)"""
    events: list[UIEvent] = Field(default_factory=list)
    mcp_results: Optional[dict[str, Any]] = None


# Predefined function intents with their MCP server mappings
FUNCTION_INTENTS = {
    "Fee Inquiry": FunctionIntent(
        name="Fee Inquiry",
        description="Customer asks about, disputes, or shows curiosity about fees charged to their account, or future fees.",
        required_data=[DataIntentType.ACCOUNT_ID, DataIntentType.TRANSACTION_DATE],
        mcp_server="fee_inquiry_server"
    ),
    "Sales Opportunity": FunctionIntent(
        name="Sales Opportunity", 
        description="Customer asks about Credit Cards or Personal Loans, "
                   "or shows interest in offers available.",
        required_data=[DataIntentType.CUSTOMER_NAME, DataIntentType.ACCOUNT_ID, DataIntentType.PRODUCT_TYPE],
        mcp_server="sales_opportunity_server"
    ),
    "Knowledge Base Inquiry": FunctionIntent(
        name="Knowledge Base Inquiry",
        description="Customer has a general or specialized question about the "
                   "Royal Bank of Canada, not related to other intents.",
        required_data=[DataIntentType.CUSTOMER_NAME, DataIntentType.ACCOUNT_ID],
        mcp_server="knowledge_base_server"
    )
}

# Predefined data intents
DATA_INTENTS = {
    DataIntentType.ACCOUNT_ID: DataIntent(
        name=DataIntentType.ACCOUNT_ID,
        description="The customer's unique bank account ID or number."
    ),
    DataIntentType.TRANSACTION_DATE: DataIntent(
        name=DataIntentType.TRANSACTION_DATE,
        description="Date of the transaction in question."
    ),
    DataIntentType.CUSTOMER_NAME: DataIntent(
        name=DataIntentType.CUSTOMER_NAME,
        description="The customer's full legal name."
    ),
    DataIntentType.PRODUCT_TYPE: DataIntent(
        name=DataIntentType.PRODUCT_TYPE,
        description="The type of lending product (Credit Card, Personal Loan, etc.)."
    )
}
