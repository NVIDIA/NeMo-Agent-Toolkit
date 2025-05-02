import logging
from pydantic import BaseModel
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

class ParseOrderConfig(FunctionBaseConfig, name="agentic_workflow"):
    """
    AgentIQ function template.
    """
    data_path: str = "/home/varshashiveshwar/projects/AgentIQ/examples/agentic_workflow/src/agentic_workflow/data/customer_order.json"

@register_function(config_type=ParseOrderConfig)
async def parse_order_function(
    tool_config: ParseOrderConfig, builder: Builder
):
    import json

    with open(tool_config.data_path, 'r', encoding='utf-8') as f:
        order_data = json.load(f)

    async def _parse_order_fn(input_data: str) -> str:
        result = {
            "product_type": order_data.get("product_type", ""),
            "quantity": order_data.get("quantity", ""),
        }
        return str(result)

    try:
        yield FunctionInfo.create(single_fn=_parse_order_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up agentic_workflow workflow.")
