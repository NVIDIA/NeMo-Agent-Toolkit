"""NeMo Microservices API utilities.

This package provides Python clients and Pydantic models for interacting with
NeMo microservice APIs.

Subpackages:
    - customizer: NeMo Customizer API for model fine-tuning
    - deployment: NeMo Deployment Management API for model deployments
    - entity_store: NeMo Entity Store API for datasets, models, namespaces, projects

Example:
    >>> # Using the Customizer API
    >>> from nat.plugins.customizer.api_utils.customizer import (
    ...     CustomizerClient,
    ... )
    >>> client = CustomizerClient("http://localhost:8000")
    >>> targets = client.list_targets()

    >>> # Using the Deployment Management API
    >>> from nat.plugins.customizer.api_utils.deployment import (
    ...     DeploymentClient,
    ... )
    >>> client = DeploymentClient("http://localhost:8080")
    >>> configs = client.list_configs()

    >>> # Using the Entity Store API
    >>> from nat.plugins.customizer.api_utils.entity_store import (
    ...     EntityStoreClient,
    ... )
    >>> client = EntityStoreClient("http://localhost:8080")
    >>> datasets = client.list_datasets()

    >>> # Or import clients directly from api_utils
    >>> from nat.plugins.customizer.api_utils import (
    ...     CustomizerClient, DeploymentClient, EntityStoreClient,
    ... )
"""

# Re-export main clients for convenience
from .customizer import AsyncCustomizerClient
from .customizer import CustomizerAPIError
from .customizer import CustomizerClient
from .deployment import AsyncDeploymentClient
from .deployment import DeploymentAPIError
from .deployment import DeploymentClient
from .entity_store import AsyncEntityStoreClient
from .entity_store import EntityStoreAPIError
from .entity_store import EntityStoreClient

__all__ = [
    # Customizer API
    "CustomizerClient",
    "AsyncCustomizerClient",
    "CustomizerAPIError",  # Deployment Management API
    "DeploymentClient",
    "AsyncDeploymentClient",
    "DeploymentAPIError",  # Entity Store API
    "EntityStoreClient",
    "AsyncEntityStoreClient",
    "EntityStoreAPIError",
]
