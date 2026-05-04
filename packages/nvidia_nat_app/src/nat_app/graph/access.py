# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-object, nested-path-aware read/write tracking.

``AccessSet`` tracks which fields of which state objects a node reads
or writes.  It supports:

- **Multiple state objects** (state, memory, config, etc.)
- **Nested path overlap** (``user.preferences`` overlaps ``user.preferences.theme``)
- **Fast path for flat access** (set intersection when no dotted paths)
- **Set-like operators** for ergonomic use in existing conflict-detection code

``ReducerSet`` is a ``dict[str, set[str]]`` mapping object names to
their reducer-protected field paths.
"""

from __future__ import annotations

from collections.abc import Iterator

# Object name -> set of reducer field paths
ReducerSet = dict[str, set[str]]

_DEFAULT_OBJ = "state"


def reducer_set(*fields: str, obj: str = _DEFAULT_OBJ) -> ReducerSet:
    """Convenience constructor for single-object reducer sets.

    Args:
        *fields: Reducer-protected field names.
        obj: The state object name (defaults to ``"state"``).

    Returns:
        A ``ReducerSet`` mapping the object to the given fields.
    """
    return {obj: set(fields)}


def _paths_overlap(a: str, b: str) -> bool:
    """Check if two dotted paths overlap (one is ancestor of the other).

    Args:
        a: First dotted path.
        b: Second dotted path.

    Returns:
        True if the paths are equal or one is a prefix of the other.
    """
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


class AccessSet:
    """
    Tracks read/write accesses across multiple state objects with nested path support.

    Internally stores ``{obj_name: set_of_paths}``.  When all paths are flat
    (no dots), overlap detection uses fast set intersection.  When nested
    paths are present, it falls back to prefix-aware comparison.

    Example (single object, flat -- LangGraph):

        reads = AccessSet.from_fields("query", "messages")
        writes = AccessSet.from_fields("response")
        reads.overlaps(writes)  # False -- no shared fields

    Example (multi-object, nested):

        reads = AccessSet()
        reads.add("state", "user.preferences.theme")
        reads.add("memory", "recent_history")

        writes = AccessSet()
        writes.add("state", "user.preferences")

        reads.overlaps(writes)  # True -- nested path overlap on state
    """

    __slots__ = ("_accesses", "_flat")

    def __init__(self) -> None:
        self._accesses: dict[str, set[str]] = {}
        self._flat: bool = True

    # -- Mutation ----------------------------------------------------------

    def add(self, obj: str, path: str) -> None:
        """Add an access entry for *obj* at *path* (e.g. ``add("state", "user.name")``).

        Args:
            obj: The state object name.
            path: Dotted path within the object.
        """
        self._accesses.setdefault(obj, set()).add(path)
        if "." in path:
            self._flat = False

    def add_flat(self, field: str) -> None:
        """Shortcut: ``add("state", field)`` for single-object frameworks.

        Args:
            field: The field name to add under the default object.
        """
        self._accesses.setdefault(_DEFAULT_OBJ, set()).add(field)

    # -- Overlap detection -------------------------------------------------

    def overlaps(self, other: AccessSet, exclude_reducers: ReducerSet | None = None) -> bool:
        """Check if any access in *self* overlaps with any in *other*.

        When both sides are flat (no dotted paths), this degenerates to
        set intersection -- identical performance to the old ``set[str]`` model.

        Args:
            other: The other access set to check against.
            exclude_reducers: Reducer fields to exclude from overlap checks.

        Returns:
            True if at least one non-reducer field overlaps.
        """
        if self._flat and other._flat:
            return self._flat_overlaps(other, exclude_reducers)
        return self._nested_overlaps(other, exclude_reducers)

    def _flat_overlaps(self, other: AccessSet, exclude_reducers: ReducerSet | None) -> bool:
        reducers = exclude_reducers or {}
        for obj, my_fields in self._accesses.items():
            other_fields = other._accesses.get(obj)
            if not other_fields:
                continue
            obj_reducers = reducers.get(obj, set())
            if (my_fields - obj_reducers) & (other_fields - obj_reducers):
                return True
        return False

    def _nested_overlaps(self, other: AccessSet, exclude_reducers: ReducerSet | None) -> bool:
        reducers = exclude_reducers or {}
        for obj, my_paths in self._accesses.items():
            other_paths = other._accesses.get(obj)
            if not other_paths:
                continue
            obj_reducers = reducers.get(obj, set())
            for my_path in my_paths:
                if my_path in obj_reducers:
                    continue
                for other_path in other_paths:
                    if other_path in obj_reducers:
                        continue
                    if _paths_overlap(my_path, other_path):
                        return True
        return False

    # -- Set-like operators ------------------------------------------------

    def __and__(self, other: AccessSet) -> AccessSet:
        """Intersection: entries present in both self and other.

        Args:
            other: The other access set to intersect with.

        Returns:
            A new ``AccessSet`` containing overlapping entries.
        """
        result = AccessSet()
        for obj, my_paths in self._accesses.items():
            other_paths = other._accesses.get(obj)
            if not other_paths:
                continue
            if self._flat and other._flat:
                common = my_paths & other_paths
            else:
                common = set()
                for mp in my_paths:
                    for op in other_paths:
                        if _paths_overlap(mp, op):
                            common.add(mp)
                            common.add(op)
            if common:
                result._accesses[obj] = common
                if any("." in p for p in common):
                    result._flat = False
        return result

    def __sub__(self, other: AccessSet | ReducerSet) -> AccessSet:
        """Difference: remove entries covered by *other*.

        Args:
            other: An ``AccessSet`` or ``ReducerSet`` whose entries to subtract.

        Returns:
            A new ``AccessSet`` with the matched entries removed.
        """
        result = AccessSet()
        if isinstance(other, dict):
            for obj, my_paths in self._accesses.items():
                exclude = other.get(obj, set())
                remaining = my_paths - exclude
                if remaining:
                    result._accesses[obj] = remaining
                    if any("." in p for p in remaining):
                        result._flat = False
        else:
            for obj, my_paths in self._accesses.items():
                other_paths = other._accesses.get(obj, set())
                remaining = my_paths - other_paths
                if remaining:
                    result._accesses[obj] = remaining
                    if any("." in p for p in remaining):
                        result._flat = False
        return result

    def __bool__(self) -> bool:
        return any(bool(paths) for paths in self._accesses.values())

    def __len__(self) -> int:
        return sum(len(paths) for paths in self._accesses.values())

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Yield ``(obj_name, path)`` pairs."""
        for obj, paths in self._accesses.items():
            for path in paths:
                yield obj, path

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AccessSet):
            return NotImplemented
        return self._accesses == other._accesses

    __hash__ = None  # type: ignore[assignment]  # Mutable; must not be used in sets or as dict keys

    def __repr__(self) -> str:
        if not self._accesses:
            return "AccessSet()"
        parts = []
        for obj, paths in sorted(self._accesses.items()):
            parts.append(f"{obj}={sorted(paths)}")
        return f"AccessSet({', '.join(parts)})"

    # -- Convenience constructors ------------------------------------------

    @classmethod
    def from_fields(cls, *fields: str, obj: str = _DEFAULT_OBJ) -> AccessSet:
        """Create from field names (single-object shortcut).

        Supports both flat (``"query"``) and nested (``"user.name"``) paths.

        Args:
            *fields: Field names (flat or dotted).
            obj: The state object name (defaults to ``"state"``).

        Returns:
            A new ``AccessSet`` initialized with the given fields.

        Example:

            reads = AccessSet.from_fields("query", "messages")
            nested = AccessSet.from_fields("user.name", "user.email")
        """
        instance = cls()
        if fields:
            instance._accesses[obj] = set(fields)
            if any("." in f for f in fields):
                instance._flat = False
        return instance

    @classmethod
    def from_set(cls, fields: set[str], obj: str = _DEFAULT_OBJ) -> AccessSet:
        """Create from an existing set of field names.

        Args:
            fields: Set of field names to include.
            obj: The state object name (defaults to ``"state"``).

        Returns:
            A new ``AccessSet`` initialized with the given fields.
        """
        instance = cls()
        if fields:
            instance._accesses[obj] = set(fields)
            if any("." in f for f in fields):
                instance._flat = False
        return instance

    # -- Query helpers -----------------------------------------------------

    @property
    def objects(self) -> set[str]:
        """All object names that have accesses."""
        return set(self._accesses.keys())

    def fields(self, obj: str = _DEFAULT_OBJ) -> set[str]:
        """Get field paths for a specific object.

        Args:
            obj: The state object name to query.

        Returns:
            Set of field paths for the given object.
        """
        return set(self._accesses.get(obj, set()))

    @property
    def all_fields_flat(self) -> set[str]:
        """Get all fields as a flat set (for backward compat with single-object code).

        Only meaningful when there is a single object.  Returns the union
        of all fields across all objects.
        """
        result: set[str] = set()
        for paths in self._accesses.values():
            result |= paths
        return result

    @property
    def is_flat(self) -> bool:
        """True if no nested (dotted) paths are present."""
        return self._flat
