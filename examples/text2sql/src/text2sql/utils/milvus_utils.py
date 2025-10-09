"""Common utilities for Milvus operations."""

import logging
import os

from dotenv import load_dotenv
from pymilvus import AsyncMilvusClient
from pymilvus import MilvusClient

from text2sql.utils.feature_flag import Flag
from text2sql.utils.feature_flag import get_flag_value

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


def create_milvus_client_remote(
    is_async: bool = False,
    milvus_host: str | None = None,
    milvus_port: str | None = None,
    milvus_user: str | None = None,
    milvus_db_name: str | None = None,
) -> MilvusClient | AsyncMilvusClient:
    """Create remote Milvus client.

    Args:
        is_async: Whether to create async client
        milvus_host: Milvus host URL
        milvus_port: Milvus port
        milvus_user: Milvus username
        milvus_db_name: Milvus database name

    Returns:
        MilvusClient or AsyncMilvusClient instance

    Raises:
        ValueError: If required credentials are missing
    """
    # Password always comes from environment for security
    milvus_password = os.getenv("MILVUS_PASSWORD")

    if not all([milvus_host, milvus_port, milvus_user, milvus_db_name, milvus_password]):
        error_msg = ("Missing required Milvus cloud credentials. "
                     "Please provide milvus_host, milvus_port, milvus_user, and "
                     "milvus_db_name in config, and set MILVUS_PASSWORD in your .env file")
        raise ValueError(error_msg)

    # Check if host already includes protocol
    if milvus_host.startswith(("http://", "https://")):
        uri = f"{milvus_host}:{milvus_port}"
    else:
        # Default to https for cloud instances
        uri = f"https://{milvus_host}:{milvus_port}"
    logger.info(f"Connecting to Milvus cloud at {uri}")

    client_class = AsyncMilvusClient if is_async else MilvusClient
    return client_class(
        uri=uri,
        user=milvus_user,
        password=milvus_password,
        db_name=milvus_db_name,
    )


def create_milvus_client_local(
    is_async: bool = False,
    vdb_path: str | None = None,
) -> MilvusClient | AsyncMilvusClient:
    """Create local Milvus client.

    Args:
        is_async: Whether to create async client
        vdb_path: Path to local Milvus database file

    Returns:
        MilvusClient or AsyncMilvusClient instance
    """
    if not vdb_path:
        from text2sql.utils.constant import VDB_PATH

        vdb_path = VDB_PATH

    logger.info(f"VDB_PATH: {vdb_path}")
    client_class = AsyncMilvusClient if is_async else MilvusClient

    try:
        # Try new API first
        return client_class(uri=vdb_path)
    except TypeError:
        # Fall back to legacy API if new API fails
        try:
            return client_class(vdb_path)
        except Exception as e:
            logger.error(f"Failed to create Milvus client with legacy API: {e}")
            raise


def create_milvus_client(
    is_async: bool = False,
    milvus_host: str | None = None,
    milvus_port: str | None = None,
    milvus_user: str | None = None,
    milvus_db_name: str | None = None,
    vanna_remote: bool | None = None,
    vdb_path: str | None = None,
) -> MilvusClient | AsyncMilvusClient:
    """Create Milvus client (local or remote).

    Args:
        is_async: Whether to create async client
        milvus_host: Milvus host URL (for remote)
        milvus_port: Milvus port (for remote)
        milvus_user: Milvus username (for remote)
        milvus_db_name: Milvus database name (for remote)
        vanna_remote: Whether to use remote Milvus (overrides env var)
        vdb_path: Path to local Milvus database file (for local)

    Returns:
        MilvusClient or AsyncMilvusClient instance
    """
    # Check if vanna_remote is explicitly provided (from config)
    # If not, fall back to environment variable
    if vanna_remote is None:
        vanna_remote = get_flag_value(Flag.VANNA_REMOTE)

    # Convert string to boolean if needed
    if isinstance(vanna_remote, str):
        vanna_remote = vanna_remote.lower() in ("true", "1", "yes", "on")

    if vanna_remote:
        logger.info("Using remote Milvus client")
        return create_milvus_client_remote(
            is_async=is_async,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            milvus_user=milvus_user,
            milvus_db_name=milvus_db_name,
        )
    else:
        logger.info("Using local Milvus client")
        return create_milvus_client_local(is_async=is_async, vdb_path=vdb_path)
