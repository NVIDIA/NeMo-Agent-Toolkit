import json
import logging
import re
import urllib.parse
from typing import Any
from typing import Dict
from typing import Optional

from aiq.authentication.exceptions.request_exceptions import BaseUrlValidationError
from aiq.authentication.exceptions.request_exceptions import BodyValidationError
from aiq.authentication.exceptions.request_exceptions import HTTPHeaderValidationError
from aiq.authentication.exceptions.request_exceptions import HTTPMethodValidationError
from aiq.authentication.exceptions.request_exceptions import QueryParameterValidationError
from aiq.data_models.authentication import HTTPMethod

logger = logging.getLogger(__name__)


class HTTPRequestValidator:
    """
    HTTP request validation utility class.

    Provides comprehensive validation for HTTP request components including URLs, methods,
    headers, query parameters, and body data. Supports flexible input parsing for both
    JSON strings and dictionaries.
    """

    @staticmethod
    def parse_flexible_input(input_data: str | dict | None, input_name: str) -> Optional[Dict[str, Any]]:
        """
        Parse input that can be either a JSON string or a dictionary.

        Args:
            input_data: Input data as string, dict, or None
            input_name: Name of the input for error messages

        Returns:
            Parsed dictionary or None

        Raises:
            ValueError: If input is invalid
        """
        if input_data is None:
            return None

        if isinstance(input_data, dict):
            return input_data

        if isinstance(input_data, str):
            try:
                return json.loads(input_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid {input_name} JSON: {str(e)}")

        raise ValueError(f"{input_name} must be a string or dictionary")

    @staticmethod
    def _validate_data(input_dict: dict) -> None:
        """
        Validates that the provided dictionary has valid keys and values.

        Args:
            input_dict (dict): The dictionary of key-value pairs to validate.

        Raises:
            ValueError: If any key is invalid or any value is empty or invalid.
        """
        # Check empty, whitespace-only, or non-string keys.
        invalid_keys = [
            repr(key) for key in input_dict.keys() if key is None or not isinstance(key, str) or key.strip() == ""
        ]

        if invalid_keys:
            raise ValueError(f"Invalid keys detected in input: {input_dict}. "
                             f"Invalid Keys: {', '.join(invalid_keys)}")

        # Check for None or empty-string values.
        invalid_values = [
            key for key, value in input_dict.items()
            if value is None or (isinstance(value, str) and value.strip() == "")
        ]
        if invalid_values:
            raise ValueError(f"Empty or invalid values detected in input: {input_dict}. "
                             f"Invalid Values: {', '.join(invalid_values)}")

    @staticmethod
    def validate_base_url(url: str) -> None:
        """Validates URL and raises BaseUrlValidationError if the URL is not a valid URL."""
        parsed_url: urllib.parse.ParseResult = urllib.parse.urlparse(url)

        # Ensure URL has both scheme and network location
        if not parsed_url.scheme or not parsed_url.netloc:
            error_message = "URL must have both scheme and network location"
            logger.error(error_message)
            raise BaseUrlValidationError('invalid_url_format', error_message)

        # Ensure URL scheme is (http or https)
        if parsed_url.scheme not in ['http', 'https']:
            error_message = f"Unsupported URL scheme: {parsed_url.scheme}. Must be http or https"
            logger.error(error_message)
            raise BaseUrlValidationError('unsupported_url_scheme', error_message)

    @staticmethod
    def validate_http_method(http_method: str | HTTPMethod) -> str:
        """
        Validates and processes the HTTP method.

        Args:
            http_method (str | HTTPMethod): The HTTP method to validate (e.g., 'GET', 'POST')
                                           as a string or HTTPMethod enum.

        Returns:
            str: The validated and normalized HTTP method string.
        """
        try:
            # Convert HTTPMethod enum to string if needed, then validate
            method_str = http_method.value if isinstance(http_method, HTTPMethod) else http_method

            # Validate the method and return the normalized uppercase version
            validated_method = HTTPMethod(method_str.upper())
            return validated_method.value
        except ValueError as e:
            valid_http_methods = ', '.join([method.value for method in HTTPMethod])
            error_message = f"Invalid HTTP method: '{http_method}'. Must be one of {valid_http_methods}"
            logger.error(error_message)
            raise HTTPMethodValidationError('invalid_http_method', error_message) from e

    @classmethod
    def validate_headers(cls, headers: str | dict | None) -> Optional[Dict[str, str]]:
        """
        Validates and processes headers for an HTTP request.

        Args:
            headers (str | dict | None): Headers as JSON string, dictionary, or None.

        Returns:
            Optional[Dict[str, str]]: Processed headers dictionary or None.
        """
        try:
            # Parse flexible input
            headers_dict = cls.parse_flexible_input(headers, "headers")

            if headers_dict is None:
                return None

            cls._validate_data(headers_dict)

            for key, value in headers_dict.items():
                # Checking for valid ASCII characters in the header name
                if not re.fullmatch(r"[A-Za-z0-9-]+", key):
                    error_message = f"Invalid header name: {key}"
                    logger.error(error_message)
                    raise HTTPHeaderValidationError('invalid_header_name', error_message)

                # Checking for disallowed control characters
                if any(ord(char) < 32 and char != '\t' or ord(char) == 127 for char in str(value)):
                    error_message = f"Invalid control character in header value: {key}: {value}"
                    logger.error(error_message)
                    raise HTTPHeaderValidationError('invalid_header_value', error_message)

            return headers_dict

        except ValueError as e:
            error_message = f"Invalid header data: {str(e)}"
            logger.error(error_message)
            raise HTTPHeaderValidationError('invalid_header_data', error_message) from e

    @classmethod
    def validate_query_parameters(cls, query_params: str | dict | None) -> Optional[Dict[str, Any]]:
        """
        Validates and processes query parameters for an HTTP request.

        Args:
            query_params (str | dict | None): Query parameters as JSON string, dictionary, or None.

        Returns:
            Optional[Dict[str, Any]]: Processed query parameters dictionary or None.
        """
        try:
            # Parse flexible input
            params_dict = cls.parse_flexible_input(query_params, "query_params")

            if params_dict is None:
                return None

            cls._validate_data(params_dict)

            for key, value in params_dict.items():
                # Catch keys with leading/trailing whitespace to prevent ambiguous parsing or bypassing
                if key.strip() != key:
                    error_message = f"Key has leading or trailing whitespace: '{key}'"
                    logger.error(error_message)
                    raise QueryParameterValidationError('invalid_query_param_key_whitespace', error_message)

                # Catch newlines in keys to prevent header injection and log splitting vulnerabilities
                if isinstance(key, str) and ('\n' in key or '\r' in key):
                    error_message = f"Key contains newline or control character: '{key}'"
                    logger.error(error_message)
                    raise QueryParameterValidationError('invalid_query_param_key_newline', error_message)

                # Catch newlines in values to avoid header injection and log splitting vulnerabilities
                if isinstance(value, str) and ('\n' in value or '\r' in value):
                    error_message = f"Value contains newline or control character for key '{key}': '{value}'"
                    logger.error(error_message)
                    raise QueryParameterValidationError('invalid_query_param_value_newline', error_message)

                # Try to URL-encode the key and value to ensure they are safe
                try:
                    urllib.parse.quote(str(key), safe='')
                    urllib.parse.quote(str(value), safe='')
                except Exception as e:
                    error_message = f"Unable to safely encode query parameter: ({key}: {value})"
                    logger.error(error_message)
                    raise QueryParameterValidationError('query_param_encoding_failed', error_message) from e

            return params_dict

        except ValueError as e:
            error_message = f"Invalid query parameter data: {str(e)}"
            logger.error(error_message)
            raise QueryParameterValidationError('invalid_query_param_data', error_message) from e

    @classmethod
    def validate_body_data(cls, body_data: str | dict | None) -> Optional[Dict[str, Any]]:
        """
        Validates and processes body data for an HTTP request.

        Args:
            body_data (str | dict | None): Body data as JSON string, dictionary, or None.

        Returns:
            Optional[Dict[str, Any]]: Processed body data dictionary or None.
        """
        # Parse flexible input
        data_dict = cls.parse_flexible_input(body_data, "body_data")

        if data_dict is None:
            return None

        try:
            json.dumps(data_dict)
            return data_dict
        except (TypeError, ValueError) as e:
            error_message = f"Request body is not JSON serializable: {str(e)}"
            logger.error(error_message)
            raise BodyValidationError('invalid_request_body', error_message) from e
