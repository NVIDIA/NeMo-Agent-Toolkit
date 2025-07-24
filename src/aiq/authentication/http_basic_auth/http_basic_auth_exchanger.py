
from aiq.data_models.authentication import AuthenticationBaseConfig, AuthenticatedContext, AuthFlowType
from aiq.authentication.interfaces import AuthenticationClientBase


from aiq.builder.context import AIQContext

class HTTPBasicAuthExchanger(AuthenticationClientBase):
    """
    Abstract base class for HTTP Basic Authentication exchangers.
    """

    def __init__(self, config: AuthenticationBaseConfig):
        """
        Initialize the HTTP Basic Auth Exchanger with the given configuration.
        Args:
            config: Configuration for the authentication process.
        """
        super().__init__(config)
        self._authenticated_tokens: dict[str, AuthenticatedContext] = {}
        self._context = AIQContext.get()


    async def authenticate(self, user_id: str) -> AuthenticatedContext:
        """
        Performs simple HTTP Authentication using the provided user ID.
        Args:
            user_id: User identifier for whom to perform authentication.

        Returns:
            AuthenticatedContext: The context containing authentication headers.
        """
        if user_id in self._authenticated_tokens:
            return self._authenticated_tokens[user_id]

        auth_callback = self._context.user_auth_callback

        try:
            auth_context = await auth_callback(self.config, AuthFlowType.HTTP_BASIC)
        except RuntimeError as e:
            raise RuntimeError(f"Authentication callback failed: {str(e)}. Did you forget to set a "
                               f"callback handler for your frontend?") from e

        self._authenticated_tokens[user_id] = auth_context

        return auth_context
