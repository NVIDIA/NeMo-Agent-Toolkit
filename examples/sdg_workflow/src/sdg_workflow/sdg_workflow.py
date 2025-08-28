# flake8: noqa: E501, BLE001
# pylint: disable=C0301
import json
import logging
from typing import Dict, Any

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.config import Config
from nat.runtime.loader import load_config

logger = logging.getLogger(__name__)


class SDGWorkflowConfig(FunctionBaseConfig, name="sdg_workflow"):
    """Configuration for the NeMo DD Synthetic Data Generation Workflow"""
    pass


@register_function(config_type=SDGWorkflowConfig)
async def sdg_workflow(config: SDGWorkflowConfig, builder: Builder):
    """NeMo DD Synthetic Data Generation Workflow"""

    from sdg_workflow.data_models import AgentToolDetails

    def extract_function_info(func_name: str, workflow_builder: Any) -> Dict[str, Any] | None:
        """Extract complete schema info for a function and return as dict."""
        try:
            # Get the function instance from the builder
            func = workflow_builder.get_function(func_name)

            # Extract schemas directly from NAT's function object
            func_input_schema_class = func.input_schema
            func_output_schema_class = getattr(func, '_single_output_schema', None)

            func_input_schema = func_input_schema_class.model_json_schema() if func_input_schema_class else {}
            func_output_schema = func_output_schema_class.model_json_schema() if func_output_schema_class else {}

            # Extract argument details more explicitly
            input_properties = func_input_schema.get('properties', {})
            output_properties = func_output_schema.get('properties', {})

            # Create explicit argument info
            input_args = []
            for arg_name, arg_info in input_properties.items():
                input_args.append({
                    'name': arg_name,
                    'type': arg_info.get('type', 'unknown'),
                    'title': arg_info.get('title', arg_name),
                    'description': arg_info.get('description', ''),
                    'required': arg_name in func_input_schema.get('required', [])
                })

            output_args = []
            for arg_name, arg_info in output_properties.items():
                output_args.append({
                    'name': arg_name,
                    'type': arg_info.get('type', 'unknown'),
                    'title': arg_info.get('title', arg_name),
                    'description': arg_info.get('description', '')
                })

            return {
                'name': func_name,
                'description': getattr(func, 'description', f"{func_name} function"),
                'input_args': input_args,
                'output_args': output_args
            }

        except Exception as e:
            logger.warning("Could not get function %s: %s", func_name, e)
            return None

    async def parse_nat_agent_cfg(nat_agent_cfg_path: str) -> Dict[str, Any]:
        """Parse the NAT agent config file and return complete result object."""
        try:
            # Load the NAT config
            nat_agent_cfg: Config = load_config(nat_agent_cfg_path)
            tools_data = []

            # Extract workflow description from config, or create one if not available
            workflow_description = getattr(nat_agent_cfg.workflow, 'description', None) if nat_agent_cfg.workflow else None

            # If no description found, create a fallback
            if not workflow_description:
                workflow_type = getattr(nat_agent_cfg.workflow, '_type', 'unknown') if nat_agent_cfg.workflow else 'unknown'
                workflow_llm = getattr(nat_agent_cfg.workflow, 'llm_name', 'unknown') if nat_agent_cfg.workflow else 'unknown'
                workflow_description = f"A {workflow_type} workflow using {workflow_llm} LLM"

            # Use WorkflowBuilder to instantiate functions
            async with WorkflowBuilder.from_config(config=nat_agent_cfg) as workflow_builder:
                # Get the list of tools used by the workflow
                # Handle both 'tool_names' and 'tools' namespace
                workflow_tools = []
                if hasattr(nat_agent_cfg, 'workflow') and nat_agent_cfg.workflow:
                    if hasattr(nat_agent_cfg.workflow, 'tool_names'):
                        workflow_tools = nat_agent_cfg.workflow.tool_names or []
                    elif hasattr(nat_agent_cfg.workflow, 'tools'):
                        workflow_tools = nat_agent_cfg.workflow.tools or []

                # Extract complete schema info for each tool
                for func_name in workflow_tools:
                    if func_name in nat_agent_cfg.functions:
                        tool_data = extract_function_info(func_name, workflow_builder)
                        if tool_data:
                            tools_data.append(tool_data)

            return {
                'config_source': nat_agent_cfg_path,
                'workflow_description': workflow_description,
                'tools_count': len(tools_data),
                'tools': tools_data
            }

        except Exception as e:
            logger.error("Failed to parse NAT agent config from %s: %s", nat_agent_cfg_path, e)
            raise

    async def _run_scenario_generator(nat_agent_cfg_path: str) -> str:
        """Generate synthetic data using NeMo Data Designer patterns"""
        try:
            logger.info("Extracting schema info for config: %s", nat_agent_cfg_path)

            # Parse function specifications and get complete result
            agent_tool_details = await parse_nat_agent_cfg(nat_agent_cfg_path)
            logger.info("Extracted schema details for %d tools", agent_tool_details['tools_count'])


            agent_tool_details_model = AgentToolDetails(**agent_tool_details)

            # Call NDD function to generate synthetic data
            ndd_function = builder.get_function("ndd_workflow")
            synthetic_data_result = await ndd_function.ainvoke(agent_tool_details_model)


            # Return both the extracted schema details and synthetic data result
            result = {
                'agent_tool_details': agent_tool_details,
                'synthetic_data_result': synthetic_data_result
            }
            return json.dumps(result)

        except Exception as e:
            logger.error("Schema extraction failed: %s", e)
            return f"Error: {str(e)}"

    try:
        yield FunctionInfo.create(
            single_fn=_run_scenario_generator,
            description="Generate synthetic data using NeMo Data Designer patterns",
            # single_output_schema=data_models.AgentToolDetails,
        )
    except GeneratorExit:
        logger.warning("Function exited early!")
    finally:
        logger.info("Cleaning up sdg_workflow.")
