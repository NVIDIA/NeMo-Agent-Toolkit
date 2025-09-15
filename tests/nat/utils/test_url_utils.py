import pytest

from nat.utils.url_utils import url_join


def test_url_join_basic():
    result = url_join("http://example.com", "api", "v1")
    assert result == "http://example.com/api/v1"
