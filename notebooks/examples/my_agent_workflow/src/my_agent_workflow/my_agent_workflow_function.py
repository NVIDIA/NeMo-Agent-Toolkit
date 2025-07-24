import logging

from aiq.builder.framework_enum import LLMFrameworkEnum
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import FunctionRef, LLMRef
from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class MyAgentWorkflowFunctionConfig(FunctionBaseConfig, name="my_agent_workflow"):
    """
    AIQ Toolkit function template. Please update the description.
    """
    tool_names: list[FunctionRef] = Field(default=[], description="List of tool names to use")
    llm_name: LLMRef = Field(description="LLM name to use")
    max_history: int = Field(default=10, description="Maximum number of historical messages to provide to the agent")
    max_iterations: int = Field(default=15, description="Maximum number of iterations to run the agent")
    handle_parsing_errors: bool = Field(default=True, description="Whether to handle parsing errors")
    verbose: bool = Field(default=True, description="Whether to print verbose output")
    description: str = Field(default="", description="Description of the agent")


@register_function(config_type=MyAgentWorkflowFunctionConfig, framework_wrappers=LLMFrameworkEnum.LANGCHAIN)
async def my_agent_workflow_function(
    config: MyAgentWorkflowFunctionConfig, builder: Builder
):
    from langchain import hub
    from langchain.agents import AgentExecutor
    from langchain.agents import create_react_agent

    # Create a list of tools for the agent
    tools = builder.get_tools(config.tool_names, framework_wrappers=LLMFrameworkEnum.LANGCHAIN)


    llm = builder.get_llm(config.llm_name, framework_wrappers=LLMFrameworkEnum.LANGCHAIN)

    # Use an open source prompt
    prompt = hub.pull("hwchase17/react-chat")

    # Initialize a ReAct agent
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=["\nObservation"])

    # Initialize an agent executor to iterate through reasoning steps
    agent_executor = AgentExecutor(agent=react_agent,
                                    tools=tools,
                                    max_iterations=config.max_iterations,
                                    handle_parsing_errors=config.handle_parsing_errors,
                                    verbose=config.verbose)   

    async def _response_fn(input_message: str) -> str:
        response = agent_executor.invoke({"input": input_message, "chat_history": []})

        return response["output"]

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up my_agent_workflow workflow.")