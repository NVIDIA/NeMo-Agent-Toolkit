"""Static classes for accessing Builder and Config instances stored in ContextVars."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.function import FunctionBaseConfig

_config_context: ContextVar[FunctionBaseConfig | None] = ContextVar("config", default=None)

T = TypeVar("T", bound=FunctionBaseConfig)


class SyncBuilder:

    def __init__(self, builder: Builder) -> None:
        self._builder = builder

        # Save the current loop
        self._loop = asyncio.get_running_loop()

    def get_llm(self, llm_name: str, wrapper_type: LLMFrameworkEnum):

        return self._loop.run_until_complete(self._builder.get_llm(llm_name, wrapper_type))


_builder_context: ContextVar[SyncBuilder | None] = ContextVar("builder", default=None)


class StaticBuilder:
    """Static class for accessing a Builder object from a ContextVar.

    This class provides a way to store and retrieve a Builder instance that
    is local to the current execution context (e.g., per async task).

    This class cannot be instantiated.
    """

    def __init__(self) -> None:
        """Prevent instantiation of StaticBuilder."""
        msg = "StaticBuilder cannot be instantiated"
        raise TypeError(msg)

    @staticmethod
    def get() -> SyncBuilder:
        """Get the Builder object from the current context.

        Returns:
            The Builder object stored in the ContextVar, or None if not set.
        """
        builder = _builder_context.get()
        if builder is None:
            raise ValueError("Builder not set in context")
        return builder

    @staticmethod
    def set(builder: Builder) -> None:
        """Set the Builder object in the current context.

        Args:
            builder: The Builder instance to store in the ContextVar.
        """
        _builder_context.set(SyncBuilder(builder))

    @staticmethod
    @contextmanager
    def use(builder: Builder) -> Generator[Builder, None, None]:
        """Context manager for temporarily setting the Builder object.

        Args:
            builder: The Builder instance to use within the context.

        Yields:
            The Builder instance that was set.

        Example:
            >>> with StaticBuilder.use(my_builder) as builder:
            >>>     # builder is active in this context
            >>>     result = StaticBuilder.get()
            >>> # Original builder is restored here
        """
        previous = _builder_context.get()
        _builder_context.set(SyncBuilder(builder))
        try:
            yield builder
        finally:
            _builder_context.set(previous)


class StaticConfig:
    """Static class for accessing a Config object from a ContextVar.

    This class provides a way to store and retrieve a Config instance that
    is local to the current execution context (e.g., per async task).

    This class cannot be instantiated.
    """

    def __init__(self) -> None:
        """Prevent instantiation of StaticConfig."""
        msg = "StaticConfig cannot be instantiated"
        raise TypeError(msg)

    @staticmethod
    def get(config_class: type[T]) -> T:
        """Get the Config object from the current context.

        Args:
            config_class: The expected class type of the config object.
                         Must be derived from FunctionBaseConfig.

        Returns:
            The Config object stored in the ContextVar, or None if not set.

        Raises:
            TypeError: If the stored config does not match the requested class type,
                      or if config_class is not derived from FunctionBaseConfig.
        """
        # Validate that config_class is derived from FunctionBaseConfig
        if not isinstance(config_class, type):
            msg = f"config_class must be a class type, got {type(config_class).__name__}"
            raise TypeError(msg)

        if not issubclass(config_class, FunctionBaseConfig):
            msg = (f"config_class must be derived from FunctionBaseConfig, "
                   f"but got {config_class.__name__}")
            raise TypeError(msg)

        config = _config_context.get()
        if config is None:
            raise ValueError(f"Config of type {config_class.__name__} not set in context")

        if not isinstance(config, config_class):
            msg = (f"Config type mismatch: expected {config_class.__name__}, "
                   f"but got {type(config).__name__}")
            raise TypeError(msg)

        return config

    @staticmethod
    def set(config: FunctionBaseConfig) -> None:
        """Set the Config object in the current context.

        Args:
            config: The Config instance to store in the ContextVar.
                   Must be derived from FunctionBaseConfig.
        """
        _config_context.set(config)

    @staticmethod
    @contextmanager
    def use(config: FunctionBaseConfig) -> Generator[FunctionBaseConfig, None, None]:
        """Context manager for temporarily setting the Config object.

        Args:
            config: The Config instance to use within the context.
                   Must be derived from FunctionBaseConfig.

        Yields:
            The Config instance that was set.

        Example:
            >>> with StaticConfig.use(my_config) as config:
            >>>     # config is active in this context
            >>>     result = StaticConfig.get(MyConfigClass)
            >>> # Original config is restored here
        """
        previous = _config_context.get()
        _config_context.set(config)
        try:
            yield config
        finally:
            _config_context.set(previous)
