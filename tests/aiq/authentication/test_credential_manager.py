import pytest

from aiq.cli.cli_utils.config_override import load_and_override_config
from aiq.utils.data_models.schema_validator import validate_schema
from aiq.data_models.config import AIQConfig
from aiq.authentication.credentials_manager import CredentialsManager


async def test_credential_manager_singleton():
    """Test that the credential manager is a singleton."""

    credentials1 = CredentialsManager()
    credentials2 = CredentialsManager()

    assert credentials1 is credentials2


async def test_credential_persistence():
    """Test that the credential manager can swap authorization configuration and persist credentials."""

    from pathlib import Path

    config_dict = load_and_override_config(Path("tests/aiq/authentication/config.yml"), overrides=())

    config = validate_schema(config_dict, AIQConfig)

    credentials = CredentialsManager()

    credentials._swap_authorization_providers(config.authentication)
    assert credentials._get_authentication_providers("jira") != config.authentication.get("jira")
