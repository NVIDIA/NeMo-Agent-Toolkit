import os
import pandas as pd

from pydantic import BaseModel
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

class DecisionAgentConfig(FunctionBaseConfig, name="decision_agent"):
    """
    AgentIQ decision agent function.
    """
    product_reference_file: str = "/home/varshashiveshwar/projects/AgentIQ/examples/agentic_workflow/src/agentic_workflow/data/product_reference.csv"

@register_function(config_type=DecisionAgentConfig)
async def decision_agent(tool_config: DecisionAgentConfig, builder: Builder):
    import pandas as pd

    def load_product_data(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Product reference file '{file_path}' not found.")
        return pd.read_csv(file_path)

    df = load_product_data(tool_config.product_reference_file)
    valid_products = df["product_type"].tolist()

    async def _decide(input_data: str) -> str:
        product_type = input_data

        if product_type in valid_products:
            result = {
                "is_valid_product": True,
                "reason": "Valid product type."
            }
        else:
            result = {
                "is_valid_product": False,
                "reason": f"Invalid product type '{product_type}'."
            }
        return str(result)

    try:
        yield FunctionInfo.create(single_fn=_decide)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up decision_agent workflow.")
