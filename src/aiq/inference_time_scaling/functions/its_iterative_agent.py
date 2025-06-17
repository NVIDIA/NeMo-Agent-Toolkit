import json
import logging

from pydantic import BaseModel
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.inference_time_scaling.models.tool_use_config import ToolUseInputSchema
from aiq.inference_time_scaling.models.tool_use_config import ToolUseList
from aiq.utils.io.think_tags import remove_r1_think_tags

logger = logging.getLogger(__name__)


class IterativeAgentOutput(BaseModel):
    """
    Schema the LLM must produce at each iteration.

    If 'status' == 'ACTION', the LLM must provide a list of tools to call:
        {
          "status": "ACTION",
          "tools": [
            {
              "tool_name": "some_tool",
              "task_description": "some request for the tool",
              "motivation": "some optional reasoning or motivation"
            },
            ...
          ]
        }

    If 'status' == 'FINAL',
        {
          "status": "FINAL",
          "answer": "final answer string"
        }
    """
    status: str
    answer: str | None = None
    tools: list[ToolUseInputSchema] | None = None


class IterativeToolOrchestrationAgentConfig(FunctionBaseConfig, name="iterative_tool_orchestration_agent"):
    """
    Configuration for the Iterative Tool-Oriented Agent. At each iteration:
      1. The agent (LLM) either returns a final answer or a list of tools to call.
      2. If it wants to call tools, we call the `its_tool_orchestration` function with that list, get results,
         and feed them back to the agent.
    """
    llm_name: LLMRef = Field(description="The LLM to use for iterative reasoning.")
    orchestrator_fn: FunctionRef = Field(
        default="its_tool_orchestration",
        description="The name of the function that is the 'its_tool_orchestration' function.")
    max_iterations: int = Field(
        default=8, description="Maximum number of iterations allowed before giving up and returning partial output.")
    system_prompt: str = Field(default=(
        "You are an iterative tool-calling agent. On each step, RESPOND ONLY with a single valid JSON object "
        "and NOTHING else. Do not include any explanations or commentary.\n"
        "Prefix your response with the exact text `OUTPUT: ` followed immediately by the JSON.\n"
        "The JSON MUST follow this schema exactly:\n"
        "```\n"
        "{\n"
        "  \"status\": \"ACTION\" or \"FINAL\",\n"
        "  \"answer\": \"<string>\"         // Required if status == \"FINAL\"\n"
        "  \"tools\": [                    // Required if status == \"ACTION\"\n"
        "    {\n"
        "      \"tool_name\": \"<string>\",\n"
        "      \"task_description\": \"<string>\",\n"
        "      \"motivation\": \"<string>\"  // Optional\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "```\n"
        "- Use double quotes for all keys and string values.\n"
        "- If status==\"ACTION\", `tools` must be a non-empty array with at least one tool object.\n"
        "- If status==\"FINAL\", include only the `answer` field with a concise final answer.\n"
        "- Incorporate ALL relevant previous tool results into subsequent tool calls in `task_description`.\n"
        "- Ensure the entire JSON object is on a single line (no line breaks within the JSON).\n"
        "- Always use tools available to you to collect relevant information before producing a final answer.\n"
        "- Never produce any text after the JSON object. No explanations, no commentary.\n"
    ), description="A system prompt that instructs the LLM on the strict JSON-based output format.")
    max_retries: int = Field(default=3,
                             description="Maximum number of retries for the LLM to produce valid JSON output. ")
    max_history: int | None = Field(default=None,
                                    description="Maximum number of messages to keep in the conversation history. "
                                    "If None, no limit is applied. This can help control memory usage.")
    min_iterations: int = Field(
        default=1,
        description="Minimum number of iterations to run before stopping. "
        "If the agent returns 'FINAL' before this, we will keep iterating until we reach this number."
        "This is useful for agents that need to run at least a few iterations to produce a final answer.")

    continuation_prompt: str = Field(
        description="A prompt to use when continuing the conversation iterations until "
        "the minimum number of iterations is reached. ",
        default="Are you sure you have all the information you need to answer the question? Try to think of "
        "any additional tools you may need to call to get the answer. You can continue to use tool calls or,"
        " if you have your final answer, set status=\"FINAL\" and supply the 'answer'.\n")


@register_function(
    config_type=IterativeToolOrchestrationAgentConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def iterative_tool_orchestration_agent(config: IterativeToolOrchestrationAgentConfig, builder: Builder):
    """
    Registers an iterative agent that uses an LLM and a single user prompt. On each iteration:
      - The LLM must respond with either an Action (tools to call) or a Final answer.
      - If Action, we call the `its_tool_orchestration` function with the requested tools, gather results,
        then pass them back to the LLM. The LLM can keep iterating up to `max_iterations`.
      - Once the agent returns status=FINAL, we stop and return the final answer.
    """

    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.messages.system import SystemMessage
        from langchain_core.messages.ai import AIMessage
    except ImportError:
        raise ImportError("langchain-core is not installed. Please install it to use MultiLLMPlanner.\n"
                          "This error can be resolve by installing agentiq-langchain.")

    # 1) Get references to the orchestrator function (the function that actually calls the requested tools).
    orchestrator_fn: Function = builder.get_function(config.orchestrator_fn)

    # 2) Get the LLM
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # We'll direct-invoke it with a system + user prompt each time.

    system_msg = config.system_prompt + "\n\n" + ("You have access to the following tools should you wish to"
                                                  " call them:\n" + orchestrator_fn.description + "\n")

    async def single_invoke(message: str) -> str:
        """
        The main agent logic. We keep the conversation, repeatedly call the LLM, parse JSON,
        call the orchestrator if needed, and stop upon final answer or after max_iterations.
        """
        # We'll store the conversation as a single text block (or a set of chat messages).
        # For simplicity, we keep them as chat messages:

        input_message = HumanMessage(content=message)

        conversation = [SystemMessage(content="detailed thinking off"), HumanMessage(content=system_msg), input_message]

        iteration_count = 0
        final_answer = "No final answer produced."

        while iteration_count < config.max_iterations:
            iteration_count += 1
            # 3) Call LLM with the entire conversation. For concision, you may want to prune older messages.

            parse_iteration = 0
            iterative_out = None  # Reset the output for each iteration\

            if config.max_history:
                # Limit the conversation history to the last `max_history` messages
                conversation = conversation[-config.max_history:]

            while parse_iteration < config.max_retries:
                parse_iteration += 1
                try:
                    # Call the LLM with the conversation so far
                    logger.debug(f"Calling LLM at iteration {iteration_count} with messages:\n{conversation}")
                    # We use ainvoke to allow async execution
                    response = await llm.ainvoke(conversation)
                    text_output = response.content if hasattr(response, "content") else str(response)
                    text_output = remove_r1_think_tags(text_output).strip()

                    # downselect text output to the part after "OUTPUT:"
                    if "OUTPUT:" in text_output:
                        text_output = text_output.split("OUTPUT:", 1)[1].strip()
                    else:
                        continue

                    logger.info(f"LLM response at iteration {iteration_count}:\n{text_output}")
                    parsed = json.loads(text_output)
                    # Validate with our pydantic schema
                    iterative_out = IterativeAgentOutput(**parsed)
                    conversation.append(response)

                    break  # If successful, exit the retry loop
                except Exception as ex:
                    logger.info(f"Error parsing LLM output at iteration {iteration_count}: {ex}")
                    if parse_iteration >= config.max_retries:
                        raise

            # 5) Check the agent's "status":
            if iterative_out.status.upper() == "FINAL":
                # We have a final answer
                if iteration_count < config.min_iterations:
                    # If we hit min_iterations, we keep iterating until we reach it.
                    logger.debug(f"Iteration {iteration_count} reached min_iterations. Continuing.")
                    conversation.append(HumanMessage(content=config.continuation_prompt))
                    continue

                if not iterative_out.answer:
                    # If the LLM gave "status":"FINAL" but forgot "answer"?
                    final_answer = "Agent gave FINAL but no 'answer' was included."
                else:
                    final_answer = iterative_out.answer
                break

            elif iterative_out.status.upper() == "ACTION":
                if not iterative_out.tools or len(iterative_out.tools) == 0:
                    # The LLM said "ACTION" but gave no tools?
                    logger.warning(
                        f"Iteration {iteration_count}: Agent indicated ACTION but provided no tools. Stopping.")
                    final_answer = "Agent gave ACTION but no tools. Halting."
                    break

                # Construct a ToolUseList from the agent's requested tools
                tool_calls = ToolUseList(tools=iterative_out.tools)

                # 6) Call the orchestrator function to run them
                try:
                    orch_result = await orchestrator_fn.acall_invoke(tool_calls)
                    # orch_result is also a ToolUseList
                except Exception as ex:
                    logger.error(f"Error calling orchestrator function: {ex}")
                    final_answer = "Error calling orchestrator function: " + str(ex)
                    break

                # 7) We'll now add the "result" of that tool call to the conversation so the LLM can see it.
                # We'll create a message summarizing the tool results
                if len(orch_result.tools) > 0:
                    results_msg = ("Tool call results:\n" + "\n\n".join(f"Tool '{tool.tool_name}' "
                                                                        f"returned: {tool.output}"
                                                                    for tool in orch_result.tools))
                else:
                    results_msg = "The tools did not produce any relevant output. Try another query/ACTION."

                # We'll feed it to the LLM as if it was a new "assistant" message or "system" message
                # so that the LLM can read that result in the next iteration.
                conversation.append(AIMessage(content=results_msg))

            else:
                # The LLM returned an unknown status
                final_answer = f"Agent returned unknown status: {iterative_out.status}"
                break

        else:
            # If we exit the while loop normally, that means we hit max_iterations
            final_answer = (f"Max iterations {config.max_iterations} exceeded. No final answer found.\n"
                            f"Last known partial answer: {final_answer}")

        return final_answer

    # 8) Return the function info. We only produce a single_fn (non-streaming).
    yield FunctionInfo.from_fn(
        fn=single_invoke,
        description=("An iterative agent that calls `its_tool_orchestration` function for tool actions. "
                     "LLM must at each step produce JSON with either a final answer or a list of tools to run."),
    )
