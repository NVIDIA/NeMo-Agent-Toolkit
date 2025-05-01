from aiq.authentication.interfaces import RequestManagerBase
from aiq.authentication.interfaces import ResponseManagerBase


async def execute_api_request(request: RequestManagerBase) -> ResponseManagerBase | None:
    """
    This function executes an API request and returns the appropriate response from the server.

    Args:
        request (RequestManagerBase): _description_
    """
    pass
