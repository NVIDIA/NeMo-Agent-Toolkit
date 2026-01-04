import importlib.util
import sys
from collections.abc import AsyncGenerator

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class LanggraphWorkflowConfig(FunctionBaseConfig, name="langgraph_workflow"):
    dependencies: list[str] = Field(default_factory=list)
    graph: str
    env: str | None = None


@register_function(config_type=LanggraphWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register(config: LanggraphWorkflowConfig, b: Builder):

    # Split the graph path into module and name
    module_path, name = config.graph.rsplit(":", 1)

    spec = importlib.util.spec_from_file_location("agent_code", module_path)

    if spec is None:
        raise ValueError(f"Spec not found for module: {module_path}")

    module = importlib.util.module_from_spec(spec)

    if module is None:
        raise ValueError(f"Module not found for module: {module_path}")

    sys.modules["agent_code"] = module

    if spec.loader is not None:
        spec.loader.exec_module(module)
    else:
        raise ValueError(f"Loader not found for module: {module_path}")

    graph: CompiledStateGraph = getattr(module, name)

    input_type = graph.InputType
    output_type = graph.OutputType or dict

    async def _inner_stream(input_data: input_type) -> output_type:

        try:
            output = await graph.ainvoke(input_data)
            return output
        except Exception as e:
            print(f"Error in graph: {e}")
            return f"Error in graph: {e}"

    yield _inner_stream
