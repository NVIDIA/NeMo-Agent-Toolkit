from aiq.authentication.interfaces import FlowHandlerBase
from aiq.data_models.authentication import AuthenticationBaseConfig, AuthenticatedContext, AuthFlowType
import click

class ConsoleAuthenticationFlowHandler(FlowHandlerBase):
    """
    Console-based authentication flow handler.
    """

    @staticmethod
    async def authenticate(config: AuthenticationBaseConfig, method: AuthFlowType) -> AuthenticatedContext:
        """
        Handles authentication by prompting the user for input via the console.

        Args:
            config: The authentication configuration.
            method: The authentication method to use.

        Returns:
            AuthenticatedContext: The context containing authentication details.
        """

        if method != AuthFlowType.HTTP_BASIC:
            raise ValueError(f"Unsupported authentication method: {method}. Only HTTP_BASIC is supported.")


        authenticated_context = ConsoleAuthenticationFlowHandler._authenticate_http()

        return authenticated_context

    @staticmethod
    def _authenticate_http() -> AuthenticatedContext:
        """
        Uses click to get username and password from user.

        Constructs HTTP Basic header from the credentials.

        Returns in AuthenticatedContext.
        """
        username = click.prompt("Enter your username", type=str)
        password = click.prompt("Enter your password", type=str, hide_input=True)

        import base64
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("ascii")

        headers = {
            "Authorization": f"Basic {encoded_credentials}"
        }
        query_params = {
            "username": username,
            "password": password
        }

        return AuthenticatedContext(headers=headers, query_params=query_params)

