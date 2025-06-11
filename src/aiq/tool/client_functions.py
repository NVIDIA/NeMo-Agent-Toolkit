from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.config import FunctionBaseConfig


async def register_client_functions(builder: WorkflowBuilder, client_configs: dict[str, FunctionBaseConfig]) -> None:
    """
    Register client functions from the config.
    This function is called by the workflow builder to register client functions.
    """
    from aiq.cli.type_registry import GlobalTypeRegistry

    type_registry = GlobalTypeRegistry.get()

    for name, client_config in client_configs.items():
        # try to get the handler from the global registry
        handler = type_registry.get_function(client_config.type)
        await handler.build_fn(client_config, builder)
