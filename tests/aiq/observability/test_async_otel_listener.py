from aiq.observability.async_otel_listener import merge_dicts


def test_merge_dicts_basic():
    """Test basic dictionary merging functionality."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2, "c": 4}


def test_merge_dicts_with_none_values():
    """Test merging dictionaries with None values."""
    dict1 = {"a": None, "b": 2, "c": None}
    dict2 = {"a": 1, "b": 3, "c": 4}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2, "c": 4}


def test_merge_dicts_empty_dicts():
    """Test merging empty dictionaries."""
    dict1 = {}
    dict2 = {}
    result = merge_dicts(dict1, dict2)
    assert result == {}


def test_merge_dicts_one_empty():
    """Test merging when one dictionary is empty."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2}

    dict1 = {}
    dict2 = {"a": 1, "b": 2}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2}


def test_merge_dicts_nested_values():
    """Test merging dictionaries with nested values."""
    dict1 = {"a": {"x": 1}, "b": None}
    dict2 = {"a": {"y": 2}, "b": {"z": 3}}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": {"x": 1}, "b": {"z": 3}}


def test_merge_dicts_complex_types():
    """Test merging dictionaries with complex types."""
    dict1 = {"a": [1, 2, 3], "b": None}
    dict2 = {"a": [4, 5, 6], "b": "test"}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": [1, 2, 3], "b": "test"}
