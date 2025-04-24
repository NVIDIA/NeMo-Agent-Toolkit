import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

from .a2a_client import A2AToolClient

logger = logging.getLogger(__name__)


class A2AFunctionConfig(FunctionBaseConfig, name="a2a_function_wrapper"):
    """Configuration for A2A functions."""
    url: HttpUrl = Field(description="URL of the A2A server")
    a2a_tool_name: str = Field(description="Name of the A2A tool to use")
    description: str | None = Field(default=None,
                                    description="""
        Description for the tool that will override the description provided by the A2A server. Should only be used if
        the description provided by the server is poor or nonexistent
        """)
    parameters: dict[str, Any] = Field(default_factory=dict, description="Default parameters for the function")


@register_function(config_type=A2AFunctionConfig)
async def a2a_function(config: A2AFunctionConfig, builder: Builder):
    """Generate an AgentIQ Function that wraps a tool provided by the A2A server."""

    # Setup a client for each tool
    tool = A2AToolClient(str(config.url), tool_name=config.a2a_tool_name)

    # Get the AgentCard for the tool. This can be deferred to the first tool call but
    # we need the description if it is not configured in the aiq config file
    await tool.get_card()
    # Set the description if configured otherwise use the AgentCard description
    tool.set_description(config.description)

    def _convert_from_str(input_str: str) -> tool.input_schema:
        return tool.input_schema.model_validate_json(input_str)

    async def _response_fn(tool_input: BaseModel | None = None, **kwargs) -> str:
        logger.info("Tool input: %s", tool_input)
        if tool_input:
            args = tool_input.model_dump()
            return await tool.acall(args)

        _ = tool.input_schema.model_validate(kwargs)
        return await tool.acall(kwargs)

    yield FunctionInfo.create(single_fn=_response_fn,
                              description=tool.description,
                              input_schema=tool.input_schema,
                              converters=[_convert_from_str])
