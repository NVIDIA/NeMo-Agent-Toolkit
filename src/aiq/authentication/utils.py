import httpx

from aiq.authentication.request_manager import RequestManager
from aiq.data_models.api_server import AuthenticatedRequest
from aiq.data_models.authentication import ExecutionMode


async def execute_api_request_default(request: AuthenticatedRequest) -> None:
    """
    Default callback handler for user requests. This no-op function raises a NotImplementedError error, to indicate that
    a valid request callback was not registered.

    Args:
        request (AuthenticatedRequest): The authenticated request to be executed.
    """
    raise NotImplementedError("No request callback was registered. Unable to handle request.")


async def execute_api_request_console(request: AuthenticatedRequest) -> httpx.Response | None:
    """
    Callback function that executes an API request in console mode using the provided authenticated request.

    Args:
        request (AuthenticatedRequest): The authenticated request to be executed.

    Returns:
        httpx.Response | None: The response from the API request, or None if an error occurs.
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


async def execute_api_request_server_http(request: AuthenticatedRequest) -> httpx.Response | None:
    """
    Callback function that executes an API request in http server mode using the provided authenticated request.

    Args:
        request (AuthenticatedRequest): The authenticated request to be executed.

    Returns:
        httpx.Response | None: The response from the API request, or None if an error occurs.
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
