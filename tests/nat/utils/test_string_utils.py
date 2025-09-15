import dataclasses

import pytest
from pydantic import BaseModel

from nat.utils.string_utils import convert_to_str


class _M(BaseModel):
    a: int
    b: str | None = None


def test_convert_to_str_primitives():
    assert convert_to_str("x") == "x"
    assert convert_to_str([1, 2, 3]) == "1, 2, 3"
    s = convert_to_str({"k": 1, "z": 2})
    assert (s.startswith("k: 1") or s.startswith("z: 2"))


def test_convert_to_str_object_with_str():
    @dataclasses.dataclass
    class C:
        x: int

        def __str__(self):
            return f"C({self.x})"

    assert convert_to_str(C(3)) == "C(3)"
