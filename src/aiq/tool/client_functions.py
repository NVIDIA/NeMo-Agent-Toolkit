import logging
from collections.abc import Awaitable
from collections.abc import Callable

from pydantic import ValidationError

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.client_functions import ClientFunctionConfig

logger = logging.getLogger(__name__)

# Registry for client function handlers
CLIENT_FUNCTION_HANDLERS: dict[type[ClientFunctionConfig],
                               Callable[[WorkflowBuilder, ClientFunctionConfig], Awaitable[None]]] = {}


def register_client_function_handler(config_cls: type[ClientFunctionConfig]):
    """
    Decorator to register a handler for a client function config type.
    Usage:
        @register_client_function_handler(MyClientConfig)
        async def handle_my_client(builder, config):
            ...
    """

    def decorator(fn: Callable[[WorkflowBuilder, ClientFunctionConfig], Awaitable[None]]):
        CLIENT_FUNCTION_HANDLERS[config_cls] = fn
        logger.debug(f"Registered client function handler for {config_cls.__name__}")
        return fn

    return decorator


async def register_client_functions(builder: WorkflowBuilder, client_configs: dict[str, dict]) -> None:
    """
    Register client functions from the config.
    This function is called by the workflow builder to register client functions.
    """
    for name, raw_config in client_configs.items():
        # Find the handler for this config type
        handler = None
        config_cls = None
        base_client_config = ClientFunctionConfig.model_validate(raw_config)
        for handler_cls, handler_fn in CLIENT_FUNCTION_HANDLERS.items():
            if handler_cls.static_type() == base_client_config.type:
                handler = handler_fn
                config_cls = handler_cls
                break

        if handler is None:
            raise ValueError(f"No handler found for client function type: {raw_config.get('type')}")

        # Validate the config against the correct class
        try:
            validated_config = config_cls.model_validate(raw_config)
        except ValidationError as e:
            raise ValueError(f"Invalid client config '{name}': {str(e)}")

        # Run the handler
        await handler(validated_config, builder)
