# flake8: noqa: E501, BLE001
# pylint: disable=C0301
import os
import logging
from typing import Dict, Any, List

from pydantic import Field
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class NDDClientConnectionConfig(BaseModel):
    """Configuration for NeMo Data Designer client connection."""
    ndd_client_url: str = Field(default="http://localhost:8000", description="URL of the NeMo Data Designer container")
    ndd_client_timeout: int = Field(default=600, description="Timeout for the NeMo Data Designer client")
    ndd_datastore_url: str = Field(default="http://localhost:3000", description="URL of the NMS Datastore container")


class NDDWorkflowConfig(FunctionBaseConfig, name="ndd_workflow"):
    """Configuration for the NeMo Data Designer workflow."""

    ndd_client_cfg: NDDClientConnectionConfig = Field(
        default_factory=NDDClientConnectionConfig,
        description=("Configuration dict for the NeMo Data Designer client.")
    )

    ndd_model_cfg: List[Dict[str, str]] = Field(
        default_factory=list,
        description=("List of model assignments. Each item in the list is a dict"
                     "each dict mapping model_alias -> llm_name"
                     "e.g., [{'candidate_llm': 'openai_llm'}]")
    )

    output_dir: str = Field(description="Directory to save the generated synthetic data.")


@register_function(config_type=NDDWorkflowConfig)
async def ndd_client_function(config: NDDWorkflowConfig, builder: Builder):
    """NeMo Data Designer client for synthetic data generation."""

    import uuid
    from datetime import datetime
    from datetime import timezone

    from nemo_microservices import NeMoMicroservices
    from nemo_microservices.beta.data_designer import (
        DataDesignerConfigBuilder,
        DataDesignerClient
    )

    from nemo_microservices.beta.data_designer.config import columns as C
    from nemo_microservices.beta.data_designer.config import params as P

    from sdg_workflow.data_models import AgentToolDetails
    from sdg_workflow.data_models import UserQuery
    from sdg_workflow.data_models import ScenarioDescription
    from sdg_workflow.data_models import AgentTrajectory

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Initialize NDD client and parse model configs
    ndd = DataDesignerClient(client=NeMoMicroservices(
        base_url=config.ndd_client_cfg.ndd_client_url,
        timeout=config.ndd_client_cfg.ndd_client_timeout
    ))

    async def parse_model_config(nat_llm_cfg: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Parse the NAT LLM config and create NDD model cfg"""
        ndd_model_configs = []

        for single_model_cfg in nat_llm_cfg:
            # Extract model alias and llm name from the dict
            model_alias = list(single_model_cfg.keys())[0]
            llm_name = single_model_cfg[model_alias]

            # Get the LLM config from NAT using the builder
            try:
                llm_config = builder.get_llm_config(llm_name)
                if not llm_config:
                    logger.warning("LLM '%s' not found in NAT config", llm_name)
                    continue

                # Get inference parameters
                inference_params = {
                    "temperature": getattr(llm_config, 'temperature', 0.7),
                    "max_tokens": getattr(llm_config, 'max_tokens', 1024),
                    "top_p": getattr(llm_config, 'top_p', 0.95),
                }

                # Get connection parameters
                model_id = getattr(llm_config, 'model_name', llm_name)
                base_url = getattr(llm_config, 'base_url', None)

                logger.info("Configuring NDD model: alias=%s, llm=%s, model_id=%s, base_url=%s, params=%s",
                           model_alias, llm_name, model_id, base_url, inference_params)

                # Create a NeMo Data Designer model config
                ndd_model_configs.append(
                    P.ModelConfig(
                        alias=model_alias,
                        inference_parameters=P.InferenceParameters(
                            **inference_params
                        ),
                        model=P.Model(
                            api_endpoint=P.ApiEndpoint(
                                model_id=model_id,
                                url=base_url,
                            ),
                            model_id=model_id
                        )
                    )
                )

            except Exception as e:
                logger.error("Failed to configure model '%s' with LLM '%s': %s", model_alias, llm_name, e)
                continue

        return ndd_model_configs

    async def get_tool_names(agent_tool_details: AgentToolDetails) -> List[str]:
        """Get the tool names from the agent tool details"""
        tool_names = [tool.name for tool in agent_tool_details.tools]

        tool_names.append("None")
        return tool_names

    async def generate_synthetic_data(agent_tool_details: AgentToolDetails) -> str:
        """Generate synthetic data using NeMo Data Designer patterns."""
        try:
            # Parse and configure models from NAT LLM configs
            ndd_model_configs = await parse_model_config(config.ndd_model_cfg)
            logger.info("Configured %d models for NDD workflow", len(ndd_model_configs))

            config_builder = DataDesignerConfigBuilder(model_configs=ndd_model_configs)

            # get the names of the tools that are available to the agent
            tool_names = await get_tool_names(agent_tool_details)

            #########################################################
            # COLUMN 1: TOOL CALLED; SAMPLED COLUMN
            #########################################################
            # add a column for the name of the tool that will be called in each scenario
            # we also add an emoty list so that we can generate examples where no tool is called
            config_builder.add_column(
                C.SamplerColumn(
                    name="tool_name",
                    type=P.SamplerType.CATEGORY,
                    params=P.CategorySamplerParams(values=tool_names),
                    description=("The tool that will be called in the given scenario."
                                 "Empty list means no tool is called."),
                ))

            #########################################################
            # COLUMN 2: SCENARIO DIFFICULTY; SAMPLED COLUMN
            #########################################################
            # add a column for the user query that will be called in each scenario
            config_builder.add_column(
                C.SamplerColumn(
                    name="scenario_difficulty",
                    type=P.SamplerType.CATEGORY,
                    params=P.CategorySamplerParams(values=["easy", "medium", "hard"]),
                    description=("The difficulty of the generated scenario."),
                ))

            #########################################################
            # COLUMN 3: SCENARIO DESCRIPTION; LLM GENERATED COLUMN
            #########################################################
            # add a column for the scenario description that will be generated by the LLM
            config_builder.add_column(
                C.LLMStructuredColumn(
                    name="scenario_description",
                    model_alias="candidate_llm",
                    system_prompt=(
                        "You are an expert at generating realistic evaluation scenarios for AI agents based on their specific capabilities and domain. "
                        "Your task is to create detailed scenario descriptions that test the agent's ability to use its available tools effectively. "
                        "Always consider the agent's workflow description, available tools, and their purpose when generating scenarios. "
                        "The scenarios must be relevant to the agent's domain and require the specified tool (or no tool if tool_name is 'None')."
                    ),
                    prompt=(
                        "Create a realistic scenario description for evaluating an agent with the following context:\n\n"
                        f"**Agent Workflow**: {agent_tool_details.workflow_description}\n"
                        f"**Available Tools**: {[tool.name + ' - ' + tool.description for tool in agent_tool_details.tools]}\n\n"
                        "**Scenario Requirements:**\n"
                        "- Target tool to be used: {{ tool_name }}\n"
                        "- Difficulty level: {{ scenario_difficulty }}\n\n"
                        "Generate a scenario that:\n"
                        "1. Is relevant to the agent's domain and capabilities\n"
                        "2. Requires the specified tool (if not 'None') to solve effectively\n"
                        "3. Matches the specified difficulty level\n"
                        "4. Represents a realistic use case for this type of agent\n\n"
                        "If tool_name is 'None', create a scenario where the agent can respond using only its base knowledge without calling any tools."
                    ),
                    output_format=ScenarioDescription,
                    ))

            #########################################################
            # COLUMN 4: USER QUERY; LLM GENERATED COLUMN
            #########################################################
            # add a column for the user query that will be generated by the LLM
            config_builder.add_column(
                C.LLMStructuredColumn(
                    name="user_query",
                    model_alias="candidate_llm",
                    system_prompt=(
                        "You are an expert at generating realistic user queries for AI agent evaluation. "
                        "Your task is to create natural, realistic user queries that match the given scenario and "
                        "require the agent to use its specific capabilities and tools. "
                        "The queries should sound like real user requests and be appropriate for the agent's domain expertise."
                    ),
                    prompt=(
                        "Generate a realistic user query based on the following context:\n\n"
                        f"**Agent Domain**: {agent_tool_details.workflow_description}\n"
                        f"**Available Tools**: {[tool.name + ' - ' + tool.description for tool in agent_tool_details.tools]}\n\n"
                        "**Scenario Context**: {{ scenario_description }}\n"
                        "**Target Tool**: {{ tool_name }}\n"
                        "**Difficulty**: {{ scenario_difficulty }}\n\n"
                        "Create a user query that:\n"
                        "1. Sounds natural and realistic for this domain\n"
                        "2. Matches the scenario requirements\n"
                        "3. Would logically require the target tool (if not 'None') to answer properly\n"
                        "4. Is appropriate for the specified difficulty level\n"
                        "5. Uses terminology and language typical for users of this type of agent\n\n"
                        "The query should be written as if a real user is asking the agent for help with their specific need."
                    ),
                    output_format=UserQuery,
                    )
                )

            #########################################################
            # COLUMN 5: AGENT TRAJECTORY; LLM GENERATED COLUMN
            #########################################################
            # add a column for the complete agent trajectory that will be generated by the LLM
            config_builder.add_column(
                C.LLMStructuredColumn(
                    name="agent_trajectory",
                    model_alias="candidate_llm",
                    system_prompt=(
                        "You are an expert at generating realistic agent trajectories for evaluation scenarios. "
                        "Your task is to create a complete sequence of steps that an agent would take to solve "
                        "a user query, from initial analysis through tool usage to final response. "
                        "Think step-by-step about what a real agent would do and generate the entire trajectory."
                    ),
                    prompt=(
                        "Generate a complete agent trajectory to solve this user query:\n\n"

                        "**USER QUERY**: {{ user_query }}\n"
                        "**SCENARIO**: {{ scenario_description }}\n"
                        "**DIFFICULTY**: {{ scenario_difficulty }}\n\n"

                        "**AVAILABLE TOOLS**:\n"
                        + "\n".join([
                            f"- {tool.name}: {tool.description}"
                            for tool in agent_tool_details.tools
                        ]) + "\n\n"

                        "**TRAJECTORY GENERATION RULES**:\n\n"

                        "1. **Think step-by-step**: What would a real agent do to solve this query?\n"
                        "2. **Choose tools naturally**: Select the most appropriate tools based on the query and scenario\n"
                        "3. **Be realistic**: Consider edge cases like missing information, tool failures, multi-step processes\n"
                        "4. **End appropriately**: Either complete the task OR ask for more information\n\n"

                        "**STEP TYPES & EXAMPLES**:\n\n"

                        "**Tool Call Step Example** (for wikipedia_search):\n"
                        "```json\n"
                        "{\n"
                        "  \"content\": null,\n"
                        "  \"tool_calls\": [\n"
                        "    {\n"
                        "      \"type\": \"function_call\",\n"
                        "      \"name\": \"wikipedia_search\",\n"
                        "      \"arguments\": [\n"
                        "        {\"name\": \"query\", \"value\": \"Great Fire of London history\"}\n"
                        "      ]\n"
                        "    }\n"
                        "  ]\n"
                        "}\n"
                        "```\n\n"

                        "**Tool Call Step Example** (for webpage_query):\n"
                        "```json\n"
                        "{\n"
                        "  \"content\": null,\n"
                        "  \"tool_calls\": [\n"
                        "    {\n"
                        "      \"type\": \"function_call\",\n"
                        "      \"name\": \"webpage_query\",\n"
                        "      \"arguments\": [\n"
                        "        {\"name\": \"query\", \"value\": \"LangSmith pricing features\"}\n"
                        "      ]\n"
                        "    }\n"
                        "  ]\n"
                        "}\n"
                        "```\n\n"

                        "**Text Response Step Example**:\n"
                        "```json\n"
                        "{\n"
                        "  \"content\": \"Based on my search results, the Great Fire of London started on September 2, 1666, and was caused by a fire in a bakery on Pudding Lane.\",\n"
                        "  \"tool_calls\": null\n"
                        "}\n"
                        "```\n\n"

                        "**CRITICAL**: Always populate arguments with realistic values. \n\n"

                        "**COMMON PATTERNS**:\n"
                        "- Simple query: [tool_call with arguments] → [text_response with final answer]\n"
                        "- Multi-step: [tool_call with arguments] → [tool_call with arguments] → [text_response with final answer]\n"
                        "- Missing info: [text_response asking for clarification]\n\n"

                        "Generate a realistic trajectory following these exact examples."
                    ),
                    output_format=AgentTrajectory,
                    )
                )

            #########################################################
            # LLM AS A JUDGE: SCORING RUBRICS
            #########################################################

            # Rubric 1: Scenario Realism and Relevance
            scenario_realism_rubric = P.Rubric(
                name="scenario_realism",
                description=(
                    f"Evaluate whether the generated scenario is realistic and represents a genuine use case "
                    f"for an agent specializing in: {agent_tool_details.workflow_description}. "
                    f"Consider if real users would encounter this type of situation."
                ),
                scoring={
                    "4": "Highly realistic - represents a common, authentic scenario that users would genuinely encounter",
                    "3": "Realistic - plausible scenario with minor artificial elements",
                    "2": "Somewhat realistic - generally plausible but may feel contrived in places",
                    "1": "Unrealistic - artificial scenario that users would rarely encounter",
                    "0": "Completely unrealistic - nonsensical or impossible scenario for this domain"
                })

                        # Rubric 2: Query-Scenario Alignment
            query_alignment_rubric = P.Rubric(
                name="query_alignment",
                description=(
                    f"Assess whether the generated user query naturally fits the scenario and represents "
                    f"a realistic request that users would make. Consider if the query sounds like something a real user "
                    f"would ask an agent specialized in: {agent_tool_details.workflow_description}."
                ),
                scoring={
                    "4": "Perfect alignment - query naturally fits scenario and represents a realistic user request",
                    "3": "Good alignment - query fits scenario well with minor gaps",
                    "2": "Adequate alignment - query generally matches scenario but connection could be stronger",
                    "1": "Poor alignment - query doesn't clearly connect to scenario requirements",
                    "0": "No alignment - query is unrelated to scenario or unrealistic for this domain"
                })

            # Rubric 3: Ground Truth Trajectory Quality
            trajectory_quality_rubric = P.Rubric(
                name="trajectory_quality",
                description=(
                    "Evaluate whether the generated agent trajectory represents a high-quality 'gold standard' sequence "
                    "that would be appropriate for evaluating real agents. Consider logical flow, realistic step progression, "
                    "proper tool usage patterns, and appropriate conclusion for the specified scenario."
                ),
                scoring={
                    "4": "Excellent ground truth - exemplary trajectory with logical flow, realistic steps, and proper tool usage",
                    "3": "Good ground truth - solid trajectory with mostly logical progression and appropriate tool usage",
                    "2": "Adequate ground truth - acceptable trajectory but some steps could be improved or more realistic",
                    "1": "Poor ground truth - trajectory has significant flaws in logic, tool usage, or step progression",
                    "0": "Invalid ground truth - trajectory is unrealistic, illogical, or unsuitable for evaluation"
                })

            # Rubric 4: Difficulty Calibration
            difficulty_calibration_rubric = P.Rubric(
                name="difficulty_calibration",
                description=(
                    "Assess whether the generated scenario, query, and expected response appropriately match "
                    "the specified difficulty level. Consider complexity, required domain knowledge, and tool usage sophistication."
                ),
                scoring={
                    "4": "Perfect calibration - difficulty level precisely matches scenario complexity and expected response sophistication",
                    "3": "Good calibration - difficulty level generally appropriate with minor misalignment",
                    "2": "Adequate calibration - difficulty level somewhat matches but could be better aligned",
                    "1": "Poor calibration - difficulty level doesn't match scenario complexity (too easy or too hard)",
                    "0": "No calibration - difficulty level is completely inappropriate for the generated content"
                })

            config_builder.add_column(
                C.LLMJudgeColumn(
                    name="quality_assessment",
                    model_alias="judge_llm",
                    prompt=(
                        f"You are an expert evaluator assessing the quality of SYNTHETIC GROUND TRUTH DATA generated "
                        f"for evaluating AI agents specialized in: {agent_tool_details.workflow_description}\n\n"

                        f"**Domain Context:**\n"
                        f"- Target Agent Domain: {agent_tool_details.workflow_description}\n"
                        f"- Available Agent Tools: {[tool.name + ' (' + tool.description + ')' for tool in agent_tool_details.tools]}\n\n"

                        f"**Evaluation Task:**\n"
                        f"Assess the quality of synthetically generated evaluation data to determine if it would be suitable "
                        f"for testing real agents. You are NOT evaluating an agent's performance - you are evaluating "
                        f"the quality of the generated training/evaluation examples.\n\n"

                        "**Generated Data to Evaluate:**\n"
                        "- **Scenario:** {{ scenario_description }}\n"
                        "- **User Query:** {{ user_query }}\n"
                        "- **Expected Agent Trajectory (Ground Truth):** {{ agent_trajectory }}\n"
                        "- **Claimed Difficulty:** {{ scenario_difficulty }}\n\n"

                        f"**Evaluation Criteria:**\n"
                        f"1. **Scenario Realism**: Is this a realistic scenario that users would actually encounter when using a {agent_tool_details.workflow_description.lower()} agent?\n"
                        f"2. **Query-Scenario Alignment**: Does the user query naturally fit the scenario and represent a realistic user request?\n"
                        f"3. **Trajectory Quality**: Is the generated agent trajectory a high-quality 'gold standard' that exemplifies realistic multi-step agent behavior?\n"
                        f"4. **Difficulty Calibration**: Does the complexity of the scenario, query, and trajectory match the claimed difficulty level?\n\n"

                        f"**Quality Assessment Focus:**\n"
                        f"- Would real users of {agent_tool_details.workflow_description.lower()} agents ask this type of question?\n"
                        f"- Is the scenario representative of genuine use cases in this domain?\n"
                        f"- Does the trajectory show realistic step-by-step agent behavior?\n"
                        f"- Are the tool calls properly formatted with realistic arguments?\n"
                        f"- Does the trajectory flow logically from step to step?\n"
                        f"- Is the final status (completed/needs_more_info) appropriate?\n"
                        f"- Is the difficulty classification accurate based on trajectory complexity?\n\n"

                        f"Rate each dimension and explain your reasoning with specific evidence from the generated content."
                    ),
                    rubrics=[scenario_realism_rubric, query_alignment_rubric, trajectory_quality_rubric, difficulty_calibration_rubric],
                ))

            # validate the various ops and columns that are to be generated
            config_builder.validate()

            # generate a preview of the synthetic data that will be generated
            preview = ndd.preview(config_builder, verbose_logging=True)

            preview.display_sample_record(0)
            preview.display_sample_record(1)

            # create uuid for the data
            data_id = str(uuid.uuid4())

            # get current time in UTC
            current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            output_dir = os.path.join(config.output_dir, current_date)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            save_path = os.path.join(output_dir, f"ndd_output_dataset_{data_id}.csv")

            preview.dataset.to_csv(save_path, index=True)

            return save_path

        except Exception as e:
            logger.error("Data generation failed: %s", str(e))
            return str(e)

    try:
        yield FunctionInfo.create(
            single_fn=generate_synthetic_data,
            description="Generate synthetic data using NeMo Data Designer patterns",
        )

    except GeneratorExit:
        logger.warning("NDD Client function exited early!")
    finally:
        logger.info("Cleaning up NDD Client workflow.")
