# flake8: noqa: E501, BLE001
# pylint: disable=C0301
import logging
from typing import Dict, Any, List

from pydantic import Field
from pydantic import BaseModel

from sdg_eval import data_models
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


@register_function(config_type=NDDWorkflowConfig)
async def ndd_client_function(config: NDDWorkflowConfig, builder: Builder):
    """NeMo Data Designer client for synthetic data generation."""

    from nemo_microservices import NeMoMicroservices
    from nemo_microservices.beta.data_designer import (
        DataDesignerConfigBuilder,
        DataDesignerClient
    )

    from nemo_microservices.beta.data_designer.config import columns as C
    from nemo_microservices.beta.data_designer.config import params as P

    # Initialize NDD client and parse model configs
    ndd = DataDesignerClient(client=NeMoMicroservices(
        base_url=config.ndd_client_cfg.ndd_client_url,
        timeout=config.ndd_client_cfg.ndd_client_timeout
    ))

    async def parse_model_config(ndd_model_cfg: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Parse the model cfg and create NDD model cfg using NAT LLM configs"""
        ndd_model_configs = []

        for single_model_cfg in ndd_model_cfg:
            # Extract model alias and llm name from the dict
            model_alias = list(single_model_cfg.keys())[0]
            llm_name = single_model_cfg[model_alias]

            # Get the LLM config from NAT using the builder
            try:
                llm_config = _builder.get_llm_config(llm_name)
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

    async def get_tool_names(agent_tool_details: data_models.AgentToolDetails) -> List[str]:
        """Get the tool names from the agent tool details"""
        tool_names = [tool.name for tool in agent_tool_details.tools]

        # we add a field so that we can also generate examples where no tool is called
        tool_names.append([])
        return tool_names

    async def generate_synthetic_data(agent_tool_details: data_models.AgentToolDetails) -> str:
        """Generate synthetic data using NeMo Data Designer patterns."""
        try:
            # Parse and configure models from NAT LLM configs
            ndd_model_configs = await parse_model_config(config.ndd_model_cfg)
            logger.info("Configured %d models for NDD workflow", len(ndd_model_configs))

            config_builder = DataDesignerConfigBuilder(model_configs=ndd_model_configs)

            # get the names of the tools that are available to the agent
            tool_names = await get_tool_names(agent_tool_details)

            # convert the agent tool details to a json schema to pass to the LLM
            agent_tool_details_json = agent_tool_details.model_json_schema() if agent_tool_details else {}

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
                    system_prompt=("You are a helpful assistant that helps users generate synthetic ground truth data"
                                   "to evaluate the performance of an agent a.k.a scenario. Your task is to generate a "
                                   "description of a scenario that the agent will be evaluated on. "
                                   "It must be consistent with the purpose and capabilities of the agent."),
                    user_prompt=("Generate a description of a scenario that uses the tool {{ tool_name }}. "
                                 "The difficulty of the scenario should be: {{ scenario_difficulty }}. "
                                 "The capabilities of the agent / tools available to it are provided here: \n"
                                 f"{agent_tool_details_json}"
                                 ),
                    output_schema=data_models.ScenarioDescription,
                    ))

            #########################################################
            # COLUMN 4: USER QUERY; LLM GENERATED COLUMN
            #########################################################
            # add a column for the user query that will be generated by the LLM
            config_builder.add_column(
                C.LLMStructuredColumn(
                    name="user_query",
                    model_alias="candidate_llm",
                    system_prompt=("You are a helpful assistant that helps users generate synthetic ground truth data"
                                   "to evaluate the performance of an agent a.k.a scenario. Your task is to generate a "
                                   "user query that the agent will be evaluated on based on the scenario description. "
                                   "It must be consistent with the purpose and capabilities of the agent."),
                    user_prompt=("Generate a user query that the agent will be evaluated on based on the"
                                 "scenario description. The scenario description is: {{ scenario_description }}"
                                 "The difficulty of the scenario is: {{ scenario_difficulty }}"
                                 "The capabilities of the agent / tools available to it are provided here: \n"
                                 f"{agent_tool_details_json}"),
                    output_schema=data_models.UserQuery,
                    )
                )

            #########################################################
            # COLUMN 5: AGENT RESPONSE; LLM GENERATED COLUMN
            #########################################################
            # add a column for the agent response that will be generated by the LLM
            config_builder.add_column(
                C.LLMStructuredColumn(
                    name="agent_response",
                    model_alias="candidate_llm",
                    system_prompt=("You are a helpful assistant that helps users generate synthetic ground truth data"
                                   "to evaluate the performance of an agent a.k.a scenario. Given the user query and "
                                   "scenario description, your task is to generate the response that the agent should "
                                   "respond with. It must be consistent with the purpose and capabilities "
                                   "of the agent."),
                    user_prompt=("Generate the correct response that the agent should respond with given the user query. "
                                 "The user query is: {{ user_query }}. "
                                 "The scenario description is: {{ scenario_description }}. "
                                 "The scenario difficulty is: {{ scenario_difficulty }}. "
                                 "The capabilities of the agent / tools available to it are provided here: \n"
                                 f"{agent_tool_details_json}."),
                    output_schema=data_models.UserQuery,
                    )
                )


            return "test"

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
