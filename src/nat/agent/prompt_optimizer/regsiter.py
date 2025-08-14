from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.profiler.parameter_optimization.prompt_optimizer import PromptOptimizerInputSchema


class PromptOptimizerConfig(FunctionBaseConfig, name="prompt_init"):

    optimizer_llm: LLMRef = Field(description="LLM to use for prompt optimization")
    optimizer_prompt: str = Field(
        description="Prompt template for the optimizer",
        default=(
            "You are an expert at optimizing prompts for LLMs. "
            "Your task is to take a given prompt and suggest an optimized version of it. "
            "Note that the prompt might be a template with variables and curly braces. Remember to always keep the "
            "variables and curly braces in the prompt the same. Only modify the instructions in the prompt that are"
            "not variables. The system is meant to achieve the following objective\n"
            "{system_objective}\n Of which, the prompt is one part. The details of the prompt and context as below.\n"))
    system_objective: str = Field(description="Objective of the workflow")


@register_function(config_type=PromptOptimizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def prompt_optimizer_function(config: PromptOptimizerConfig, builder: Builder):
    """
    Function to optimize prompts for LLMs.
    """

    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError as exc:
        raise ImportError("langchain-core is not installed. Please install it to use MultiLLMPlanner.\n"
                          "This error can be resolve by installing nat-langchain.") from exc

    llm = await builder.get_llm(config.optimizer_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    template = PromptTemplate(template=config.optimizer_prompt,
                              input_variables=["system_objective"],
                              validate_template=True)

    base_prompt: str = (await template.ainvoke(input={"system_objective": config.system_objective})).to_string()

    async def _inner(input_message: PromptOptimizerInputSchema) -> str:
        """
        Optimize the prompt using the provided LLM.
        """

        original_prompt = input_message.original_prompt
        prompt_objective = input_message.objective
        feedback = input_message.oracle_feedback

        prompt = f"{base_prompt}\n\nOriginal Prompt: {original_prompt}\n\n with objective {prompt_objective}\n\n"

        if feedback:
            prompt += (f"The prompt evaluation mechanism also generated feedback on a few inputs "
                       f"which are provided below. Please use them to guide your edits to instructions."
                       f"\nFeedback: {feedback}\n\n")

        prompt += (
            "Please suggest an optimized version of the prompt that produces high accuracy at the correct times. "
            "Be creative in your modifications to suggest considerations for edge cases, or other helpful information. "
            "Only respond with the optimized prompt and no other text. Do not introduce new variables or change "
            "the existing variables in the prompt. Only modify the instructions in the prompt that are "
            "not variables. ")

        optimized_prompt = await llm.ainvoke(prompt)
        return optimized_prompt.content

    yield FunctionInfo.from_fn(
        fn=_inner,
        description="Optimize prompts for LLMs using a feedback LLM.",
    )


class PromptRecombinerConfig(FunctionBaseConfig, name="prompt_recombiner"):

    optimizer_llm: LLMRef = Field(description="LLM to use for prompt recombination")
    optimizer_prompt: str = Field(
        description="Prompt template for the recombiner",
        default=(
            "You are an expert at combining prompt instructions for LLMs. "
            "Your task is to merge two prompts for the same objective into a single, stronger prompt. "
            "Do not introduce new variables or modify existing placeholders."
        ),
    )
    system_objective: str = Field(description="Objective of the workflow")


@register_function(config_type=PromptRecombinerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def prompt_recombiner_function(config: PromptRecombinerConfig, builder: Builder):
    """
    Function to recombine two parent prompts into a child prompt using the optimizer LLM.
    Uses the same base template and objective instructions.
    """

    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError as exc:
        raise ImportError(
            "langchain-core is not installed. Please install it to use MultiLLMPlanner.\n"
            "This error can be resolve by installing agentiq-langchain."
        ) from exc

    llm = await builder.get_llm(config.optimizer_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    template = PromptTemplate(template=config.optimizer_prompt,
                              input_variables=["system_objective"],
                              validate_template=True)

    base_prompt: str = (await template.ainvoke(input={"system_objective": config.system_objective})).to_string()

    class RecombineSchema(PromptOptimizerInputSchema):
        parent_b: str | None = None

    async def _inner(input_message: RecombineSchema) -> str:
        parent_a = input_message.original_prompt
        parent_b = input_message.parent_b or ""
        prompt_objective = input_message.objective

        prompt = (
            f"{base_prompt}\n\n"
            "We are performing genetic recombination between two prompts that satisfy the same objective.\n"
            f"Objective: {prompt_objective}\n\n"
            f"Parent A:\n{parent_a}\n\n"
            f"Parent B:\n{parent_b}\n\n"
            "Combine the strongest instructions and phrasing from both parents to produce a single, coherent child "
            "prompt.\n"
            "Maintain variables and placeholders unchanged.\n"
            "Return only the child prompt text, with no additional commentary."
        )

        child_prompt = await llm.ainvoke(prompt)
        return child_prompt.content

    yield FunctionInfo.from_fn(
        fn=_inner,
        description="Recombine two prompts into a stronger child prompt.",
    )
