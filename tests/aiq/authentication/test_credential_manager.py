from aiq.authentication.credentials_manager import _CredentialsManager
from aiq.cli.cli_utils.config_override import load_and_override_config
from aiq.data_models.config import AIQConfig
from aiq.utils.data_models.schema_validator import validate_schema


async def test_credential_manager_singleton():
    """Test that the credential manager is a singleton."""

    credentials1 = _CredentialsManager()
    credentials2 = _CredentialsManager()

    assert credentials1 is credentials2


async def test_credential_persistence():
    """Test that the credential manager can swap authorization configuration and persist credentials."""

    from pathlib import Path

    config_dict = load_and_override_config(Path("tests/aiq/authentication/config.yml"), overrides=())

    config = validate_schema(config_dict, AIQConfig)

    # Swap credentials and ensure they are not the same.
    assert _CredentialsManager()._get_authentication_provider("jira") != config.authentication.get("jira")
    assert not config.authentication

    # Ensure credentials can only be swapped once.
    assert _CredentialsManager()._get_authentication_provider("jira") != config.authentication.get("jira")

    # Ensure None is returned if the provider does not exist.
    test = _CredentialsManager()._get_authentication_provider("invalid_provider")
    assert test is None
