import json
import logging
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import field_validator

from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


# LangChain message models for validation
class LangChainMessage(BaseModel):
    content: str
    additional_kwargs: dict[str, Any] = {}
    response_metadata: dict[str, Any] = {}
    type: str
    name: str | None = None
    id: str | None = None
    example: bool = False


def deserialize_input_value(input_value: dict[str, Any] | list[Any] | str) -> dict[str, Any] | list[Any]:
    """Deserialize a string input value to a dictionary, list, or None.

    Args:
        input_value (str): The input value to deserialize

    Returns:
        dict | list: The deserialized input value

    Raises:
        ValueError: If parsing fails
    """
    try:
        if isinstance(input_value, (dict, list)):
            return input_value
        deserialized_attribute = json.loads(input_value)
        return deserialized_attribute
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse input_value: {input_value}, error: {e}") from e


class BaseValidator(BaseModel):
    pass


class OpenAIValidator(BaseValidator):
    framework: Literal[LLMFrameworkEnum.LANGCHAIN] = LLMFrameworkEnum.LANGCHAIN
    provider: Literal["openai"] = "openai"
    input_value: list[LangChainMessage]
    tools_schema: list[dict[str, Any]] | None = None
    chat_responses: list[dict[str, Any]] | None = None

    @field_validator("input_value", mode="before")
    @classmethod
    def validate_input_value(cls, v: Any) -> list[LangChainMessage]:
        if v is None:
            raise ValueError("Input value is required")

        # Handle string input (JSON string)
        if isinstance(v, str):
            v = deserialize_input_value(v)

        # Handle dict input (single message)
        if isinstance(v, dict):
            v = [v]

        # Validate list of messages
        if isinstance(v, list):
            validated_messages = []
            for msg in v:
                if isinstance(msg, dict):
                    validated_messages.append(LangChainMessage(**msg))
                elif isinstance(msg, LangChainMessage):
                    validated_messages.append(msg)
                else:
                    raise ValueError(f"Invalid message format: {msg}")
            return validated_messages

        raise ValueError(f"Invalid input_value format: {v}")


class NIMValidator(BaseValidator):
    framework: Literal[LLMFrameworkEnum.LANGCHAIN] = LLMFrameworkEnum.LANGCHAIN
    provider: Literal["nim"] = "nim"
    input_value: list[LangChainMessage]
    tools_schema: list[dict[str, Any]]
    chat_responses: list[dict[str, Any]] | None = None

    @field_validator("input_value", mode="before")
    @classmethod
    def validate_input_value(cls, v: Any) -> list[LangChainMessage]:
        if v is None:
            raise ValueError("Input value is required")

        # Handle string input (JSON string)
        if isinstance(v, str):
            v = deserialize_input_value(v)

        # Handle dict input (single message)
        if isinstance(v, dict):
            v = [v]

        # Validate list of messages
        if isinstance(v, list):
            validated_messages = []
            for msg in v:
                if isinstance(msg, dict):
                    validated_messages.append(LangChainMessage(**msg))
                elif isinstance(msg, LangChainMessage):
                    validated_messages.append(msg)
                else:
                    raise ValueError(f"Invalid message format: {msg}")
            return validated_messages

        raise ValueError(f"Invalid input_value format: {v}")


TraceSourceUnion = NIMValidator | OpenAIValidator


class ValidatedTraceSource(BaseModel):
    source: TraceSourceUnion


def main():
    # Test with your actual data format
    input_dict = {
        "framework": "langchain",
        "input_value": [{
            "content": "what is 1 * 1?",
            "additional_kwargs": {},
            "response_metadata": {},
            "type": "human",
            "name": None,
            "id": None,
            "example": False
        }],
        "tools_schema": [{
            "a": "b"
        }],
    }

    # Use NIMValidator since it requires tools_schema
    validated_trace_source = ValidatedTraceSource(source=NIMValidator(**input_dict))
    print("NIM validator:", validated_trace_source)

    # Test with JSON string input
    json_input_dict = {
        "framework": "langchain",
        "input_value": '[{"content": "what is 1 * 1?", "type": "human"}]',
        "tools_schema": [{
            "a": "b"
        }],
    }
    validated_trace_source2 = ValidatedTraceSource(source=NIMValidator(**json_input_dict))
    print("\nJSON string input:", validated_trace_source2)

    # Test OpenAI validator (no tools_schema required)
    openai_input = {
        "framework": "langchain",
        "input_value": [{
            "content": "what is 1 * 1?", "type": "human"
        }],
        "tools_schema": None,
    }
    validated_trace_source3 = ValidatedTraceSource(source=OpenAIValidator(**openai_input))
    print("\nOpenAI validator:", validated_trace_source3)


if __name__ == "__main__":
    main()
