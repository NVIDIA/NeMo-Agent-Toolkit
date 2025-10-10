"""Milvus client utilities for Vanna integration."""

import logging

logger = logging.getLogger(__name__)


def create_milvus_client(
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    is_async: bool = False,
):
    """Create Milvus client (sync or async).

    Args:
        host: Milvus host
        port: Milvus port
        user: Milvus username
        password: Milvus password
        db_name: Milvus database name
        is_async: Create async client

    Returns:
        Milvus client instance
    """
    from pymilvus import MilvusClient

    # Use defaults if not provided
    host = host or "localhost"
    port = port or 19530
    db_name = db_name or "default"

    # Build URI (default to http as standard for Milvus)
    uri = f"http://{host}:{port}"

    # Create client config
    client_config = {"uri": uri, "db_name": db_name, "timeout": 60.0}

    # Add authentication if provided
    if user and password:
        client_config["token"] = f"{user}:{password}"

    logger.info(f"Creating {'async' if is_async else 'sync'} Milvus client: {uri}")

    # Create client
    if is_async:
        # For async, pymilvus provides AsyncMilvusClient
        try:
            from pymilvus import AsyncMilvusClient

            return AsyncMilvusClient(**client_config)
        except ImportError:
            logger.warning(
                "AsyncMilvusClient not available, using sync client. "
                "Consider upgrading pymilvus."
            )
            return MilvusClient(**client_config)
    else:
        return MilvusClient(**client_config)
