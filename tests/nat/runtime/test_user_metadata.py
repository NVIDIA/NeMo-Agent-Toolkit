from nat.runtime.user_metadata import RequestAttributes


def test_request_attributes_defaults():
    ra = RequestAttributes()
    assert ra.method is None
    assert ra.url_path is None
    assert ra.url_port is None
    assert ra.url_scheme is None
    assert ra.headers is None
    assert ra.query_params is None
    assert ra.path_params is None
    assert ra.client_host is None
    assert ra.client_port is None
    assert ra.cookies is None
