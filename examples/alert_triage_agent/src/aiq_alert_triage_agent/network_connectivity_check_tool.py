import subprocess
import telnetlib
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import utils
from .prompts import ToolReasoningLayerPrompts

class NetworkConnectivityCheckToolConfig(FunctionBaseConfig, name="network_connectivity_check"):
    description: str = Field(
        default="This tool checks network connectivity of a host by running ping and telnet tests. Args: host_id: str",
        description="Description of the tool for the agent."
    )
    llm_name: LLMRef

@register_function(config_type=NetworkConnectivityCheckToolConfig)
async def network_connectivity_check_tool(config: NetworkConnectivityCheckToolConfig, builder: Builder):
    async def _arun(host_id: str) -> str:
        is_test_mode = utils.is_test_mode()
        utils.log_header("Network Connectivity Tester")
        
        try:
            if not is_test_mode:
                # NOTE: The ping and telnet commands below are example implementations of network connectivity checking.
                # Users should implement their own network connectivity check logic specific to their environment
                # and infrastructure setup.

                # Example ping command to test basic connectivity
                result = subprocess.run(
                    ["ping", "-c", "3", host_id],
                    capture_output=True, 
                    text=True,
                )

                if result.returncode == 0:
                    ping_data = result.stdout
                else:
                    ping_data = result.stderr

                # Example telnet command to test service availability
                telnet_port = 80  # example HTTP port
                telnet_timeout = 10
                with telnetlib.Telnet(host_id, telnet_port, telnet_timeout) as tn:
                    # Read until a prompt or timeout
                    output = tn.read_until(b"Escape character is '^]'.", 10)
                    telnet_data = output.decode("utf-8")

            else:
                # Load test data
                df = utils.load_test_data()

                # Get ping data from test data, falling back to static data if needed
                ping_data = utils.load_column_or_static(
                    df=df,
                    host_id=host_id,
                    column="network_connectivity_check_tool:ping_output"
                )

                # Get telnet data from test data, falling back to static data if needed 
                telnet_data = utils.load_column_or_static(
                    df=df,
                    host_id=host_id,
                    column="network_connectivity_check_tool:telnet_output"
                )

            # Additional LLM reasoning layer on playbook output to provide a summary of the results
            utils.log_header("LLM Reasoning", dash_length=50)

            prompt = ToolReasoningLayerPrompts.NETWORK_CONNECTIVITY_CHECK.format(
                ping_data=ping_data, telnet_data=telnet_data
            )
            conclusion = await utils.llm_ainvoke(config, builder, prompt)
            
            utils.logger.debug(conclusion)
            utils.log_footer()
            return conclusion
        except Exception as e:
            utils.logger.error(f"Error during connectivity check: {str(e)}")
            raise e

    yield FunctionInfo.from_fn(
        _arun,
        description=config.description,
    )
