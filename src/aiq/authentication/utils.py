import httpx

from aiq.authentication.request_manager import RequestManager
from aiq.data_models.api_server import AuthenticatedRequest
from aiq.data_models.authentication import ExecutionMode


async def execute_api_request_default(request: AuthenticatedRequest) -> None:
    """
    Default callback handler for user requests. This no-op function raises a NotImplementedError error, to indicate that
    a valid request callback was not registered.
    """
    raise NotImplementedError("No request callback was registered. Unable to handle request.")


async def execute_api_request_console(request: AuthenticatedRequest) -> httpx.Response | None:
    """
    # TODO EE: Update

    Args:
        request (AuthenticatedRequest): _description_

    Returns:
        httpx.Response | None: _description_
    """

    request_manager: RequestManager = RequestManager()
    response: httpx.Response | None = None

    request_manager.authentication_manager._set_execution_mode(ExecutionMode.CONSOLE)

    response = await request_manager._send_request(url=request.url_path,
                                                   http_method=request.method,
                                                   authentication_provider=request.authentication_provider,
                                                   headers=request.headers,
                                                   query_params=request.query_params,
                                                   body_data=request.body_data)

    return response


async def execute_api_request_server(request: AuthenticatedRequest) -> httpx.Response | None:
    """
    # TODO EE: Update and Validate the CORS Settings to ensure proper domains can reach the API server.

    Args:
        request (AuthenticatedRequest): _description_

    Returns:
        httpx.Response | None: _description_
    """
    request_manager: RequestManager = RequestManager()
    response: httpx.Response | None = None

    request_manager.authentication_manager._set_execution_mode(ExecutionMode.SERVER)

    response = await request_manager._send_request(url=request.url_path,
                                                   http_method=request.method,
                                                   authentication_provider=request.authentication_provider,
                                                   headers=request.headers,
                                                   query_params=request.query_params,
                                                   body_data=request.body_data)

    return response
