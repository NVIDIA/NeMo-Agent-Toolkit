# flake8: noqa: E501, BLE001
import json
import logging
from typing import Dict, Any

from pydantic import Field
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.config import Config
from nat.runtime.loader import load_config

logger = logging.getLogger(__name__)


class SDGEvalConfig(FunctionBaseConfig, name="sdg_eval_workflow"):
    """Configuration for the schema extraction tool."""
    pass


@register_function(config_type=SDGEvalConfig)
async def sdg_eval_function(config: SDGEvalConfig, builder: Builder):
    """Extract input/output schemas for all functions in a NAT agent config."""

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

            return {
                'name': func_name,
                'input_schema': func_input_schema,
                'output_schema': func_output_schema,
                'input_fields': func_input_schema.get('properties', {}),
                'output_fields': func_output_schema.get('properties', {})
            }

        except Exception as e:
            logger.warning("Could not get function %s: %s", func_name, e)
            return None

    async def parse_nat_agent_cfg(nat_agent_cfg_path: str) -> Dict[str, Any]:
        """Parse the NAT agent config file and return complete result object."""
        try:
            # Load the NAT config
            config_obj: Config = load_config(nat_agent_cfg_path)
            tools_data = []

            # Use WorkflowBuilder to instantiate functions
            async with WorkflowBuilder.from_config(config=config_obj) as workflow_builder:
                # Get the list of tools used by the workflow
                workflow_tools = []
                if (hasattr(config_obj, 'workflow') and
                    config_obj.workflow and hasattr(config_obj.workflow, 'tools')):
                    workflow_tools = config_obj.workflow.tools or []

                # Extract complete schema info for each tool
                if hasattr(config_obj, 'functions') and config_obj.functions:
                    for func_name in workflow_tools:
                        if func_name in config_obj.functions:
                            tool_data = extract_function_info(func_name, workflow_builder)
                            if tool_data:
                                tools_data.append(tool_data)

            return {
                'config_source': nat_agent_cfg_path,
                'tools_count': len(tools_data),
                'tools': tools_data
            }

        except Exception as e:
            logger.error("Failed to parse NAT agent config from %s: %s", nat_agent_cfg_path, e)
            raise

    async def _generate_sdg_eval_data(nat_agent_cfg_path: str) -> str:
        """Extract input/output schemas for all functions in a NAT agent config."""
        try:
            logger.info("Extracting schema info for config: %s", nat_agent_cfg_path)

            # Parse function specifications and get complete result
            workflow_function_details = await parse_nat_agent_cfg(nat_agent_cfg_path)

            logger.info("Extracted schema details for %d tools", workflow_function_details['tools_count'])

            return json.dumps(workflow_function_details)

        except Exception as e:
            logger.error("Schema extraction failed: %s", e)
            return f"Error: {str(e)}"

    try:
        yield FunctionInfo.create(
            single_fn=_generate_sdg_eval_data,
            description="Extract input/output schemas for all functions in a NAT agent config.",
        )
    except GeneratorExit:
        logger.warning("Function exited early!")
    finally:
        logger.info("Cleaning up sdg_eval workflow.")
