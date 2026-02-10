import json

import pytest

from nat.object_store.format_parsers import infer_format
from nat.object_store.format_parsers import parse_to_dataframe


class TestInferFormat:

    def test_infer_csv(self):
        assert infer_format("data.csv") == "csv"

    def test_infer_json(self):
        assert infer_format("data.json") == "json"

    def test_infer_jsonl(self):
        assert infer_format("data.jsonl") == "jsonl"

    def test_infer_parquet(self):
        assert infer_format("data.parquet") == "parquet"

    def test_infer_xls(self):
        assert infer_format("data.xlsx") == "xls"
        assert infer_format("data.xls") == "xls"

    def test_infer_nested_path(self):
        assert infer_format("path/to/data.csv") == "csv"

    def test_infer_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot infer format"):
            infer_format("data.txt")

    def test_infer_no_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot infer format"):
            infer_format("noextension")


class TestParseToDataframe:

    def test_parse_csv(self):
        data = b"name,age\nAlice,30\nBob,25"
        df = parse_to_dataframe(data, "csv")
        assert list(df.columns) == ["name", "age"]
        assert len(df) == 2

    def test_parse_json(self):
        data = json.dumps([{"a": 1}, {"a": 2}]).encode()
        df = parse_to_dataframe(data, "json")
        assert "a" in df.columns
        assert len(df) == 2

    def test_parse_jsonl(self):
        data = b'{"a": 1}\n{"a": 2}\n'
        df = parse_to_dataframe(data, "jsonl")
        assert "a" in df.columns
        assert len(df) == 2

    def test_parse_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            parse_to_dataframe(b"data", "unknown_format")
