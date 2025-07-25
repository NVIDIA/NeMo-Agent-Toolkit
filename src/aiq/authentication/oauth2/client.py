from pydantic import SecretStr

from aiq.authentication.interfaces import AuthenticationClientBase
from aiq.builder.context import AIQContext
from aiq.data_models.authentication import AuthFlowType
from aiq.data_models.authentication import AuthResult
from aiq.data_models.authentication import BearerTokenCred

from .authorization_code_flow_config import OAuth2AuthorizationCodeFlowConfig


class OAuth2Client(AuthenticationClientBase):

    def __init__(self, config: OAuth2AuthorizationCodeFlowConfig):
        super().__init__(config)
        self._authenticated_tokens: dict[str, AuthResult] = {}
        self._context = AIQContext.get()

    async def authenticate(self, user_id: str | None) -> AuthResult:
        if user_id and user_id in self._authenticated_tokens:
            auth_result = self._authenticated_tokens[user_id]
            if not auth_result.is_expired():
                return auth_result

        auth_callback = self._context.user_auth_callback
        if not auth_callback:
            raise RuntimeError("Authentication callback not set on AIQContext.")

        try:
            authenticated_context = await auth_callback(self.config, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)
        except Exception as e:
            raise RuntimeError(f"Authentication callback failed: {e}") from e

        auth_header = authenticated_context.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise RuntimeError("Invalid Authorization header")

        token = auth_header.split(" ")[1]

        auth_result = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr(token))],
            token_expires_at=authenticated_context.metadata.get("expires_at"),
            raw=authenticated_context.metadata.get("raw_token"),
        )

        if user_id:
            self._authenticated_tokens[user_id] = auth_result

        return auth_result
