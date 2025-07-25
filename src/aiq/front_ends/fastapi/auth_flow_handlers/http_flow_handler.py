from aiq.authentication.interfaces import FlowHandlerBase
from aiq.authentication.oauth2.authorization_code_flow_config import OAuth2AuthorizationCodeFlowConfig
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import AuthFlowType


class HTTPAuthenticationFlowHandler(FlowHandlerBase):

    @staticmethod
    async def authenticate(config: OAuth2AuthorizationCodeFlowConfig, method: AuthFlowType) -> AuthenticatedContext:

        raise NotImplementedError(f"Authentication method '{method}' is not supported by the HTTP frontend."
                                  f" Do you have Websockets enabled?")
