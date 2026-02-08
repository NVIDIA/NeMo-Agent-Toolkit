# Latency Sensitivity Decorator Design

**Date:** 2026-02-08
**Status:** Approved
**Goal:** Add a `@latency_sensitive` decorator to NAT that tracks latency sensitivity levels across function calls with priority-based context nesting.

## Overview

Add support for marking functions with latency sensitivity levels (LOW, MEDIUM, HIGH) that propagate through the call stack with priority-based merging. This enables different parts of NAT (LLM routing, execution optimization, observability) to adapt behavior based on latency requirements.

**Key features:**
- Decorator works with sync/async functions and generators
- Context nesting with priority (HIGH > MEDIUM > LOW)
- Default sensitivity is MEDIUM
- Strict validation of sensitivity values
- Clean API: `Context.get().latency_sensitivity`

## Architecture

### Core Components

1. **`LatencySensitivity` enum** in `packages/nvidia_nat_core/src/nat/profiler/decorators/latency.py`:
   - Three values: LOW (priority=1), MEDIUM (priority=2), HIGH (priority=3)
   - Class method to parse from string (case-insensitive)
   - Validation logic

2. **Context integration** in `packages/nvidia_nat_core/src/nat/builder/context.py`:
   - Add `_latency_sensitivity_stack: ContextVar[list[LatencySensitivity]]` to `ContextState`
   - Default initialized with `[LatencySensitivity.MEDIUM]`
   - Add `latency_sensitivity` property to `Context` class that returns top of stack
   - Add `push_latency_sensitivity()` context manager for stack management

3. **`@latency_sensitive()` decorator** in `latency.py`:
   - Accepts `sensitivity: LatencySensitivity | str`
   - Validates and normalizes input at decoration time
   - Handles async/sync, generator/non-generator functions
   - On entry: compares priority, pushes to stack
   - On exit: always pops from stack

4. **Stack behavior:**
   - Stack always starts with `[MEDIUM]` and never empties
   - Push operation: append new sensitivity if higher priority, otherwise append current top
   - Pop operation: remove last item
   - Current value: always `stack[-1]`

## Data Structures

### LatencySensitivity Enum

```python
from enum import Enum

class LatencySensitivity(str, Enum):
    """Latency sensitivity levels for function execution."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def priority(self) -> int:
        """Return numeric priority (higher = more sensitive)."""
        return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[self.value]

    @classmethod
    def parse(cls, value: "LatencySensitivity | str") -> "LatencySensitivity":
        """Parse string or enum to LatencySensitivity.

        Args:
            value: Either a LatencySensitivity enum or string like "high", "MEDIUM"

        Returns:
            LatencySensitivity enum value

        Raises:
            ValueError: If value is not a valid sensitivity level
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            normalized = value.upper()
            if normalized in {"LOW", "MEDIUM", "HIGH"}:
                return cls(normalized)

        raise ValueError(
            f"Invalid latency sensitivity: {value!r}. "
            f"Must be 'LOW', 'MEDIUM', 'HIGH', or LatencySensitivity enum."
        )
```

**Design choices:**
- Inherits from `str, Enum` for JSON serialization and string comparison
- `priority` property centralizes comparison logic
- `parse()` method handles both enum and string inputs with strict validation

## Context Integration

### ContextState Changes

```python
class ContextState(metaclass=Singleton):
    def __init__(self):
        # ... existing fields ...
        self._latency_sensitivity_stack: ContextVar[list["LatencySensitivity"] | None] = ContextVar(
            "latency_sensitivity_stack",
            default=None
        )
```

### Context Class Changes

```python
@property
def latency_sensitivity(self) -> "LatencySensitivity":
    """Get current latency sensitivity (top of stack)."""
    from nat.profiler.decorators.latency import LatencySensitivity

    stack = ContextState()._latency_sensitivity_stack.get()
    if stack is None or len(stack) == 0:
        # Initialize with default MEDIUM
        stack = [LatencySensitivity.MEDIUM]
        ContextState()._latency_sensitivity_stack.set(stack)

    return stack[-1]

@contextmanager
def push_latency_sensitivity(self, sensitivity: "LatencySensitivity"):
    """Push latency sensitivity to stack, respecting priority.

    Args:
        sensitivity: The latency sensitivity to push

    Yields:
        None

    Notes:
        - If new sensitivity has higher priority than current, it becomes active
        - If new sensitivity has lower/equal priority, current remains active
        - Stack always pops on exit to maintain proper nesting
    """
    stack = ContextState()._latency_sensitivity_stack.get()
    if stack is None:
        from nat.profiler.decorators.latency import LatencySensitivity
        stack = [LatencySensitivity.MEDIUM]
        ContextState()._latency_sensitivity_stack.set(stack)

    current = stack[-1]
    # Push new value if higher priority, otherwise push current (maintain nesting)
    to_push = sensitivity if sensitivity.priority > current.priority else current
    stack.append(to_push)

    try:
        yield
    finally:
        stack.pop()
```

**Design choices:**
- Lazy initialization: stack created on first access with `[MEDIUM]` default
- Stack never empties (always has at least one element)
- Priority comparison happens at push time
- Context manager ensures proper cleanup even on exceptions

## Decorator Implementation

### Decorator Signature

```python
from typing import TypeVar, Callable, overload

F = TypeVar('F', bound=Callable[..., Any])

@overload
def latency_sensitive(func: F) -> F: ...

@overload
def latency_sensitive(sensitivity: LatencySensitivity | str) -> Callable[[F], F]: ...

def latency_sensitive(
    func_or_sensitivity: F | LatencySensitivity | str | None = None
) -> F | Callable[[F], F]:
    """
    Decorator to mark functions with latency sensitivity.

    Args:
        func_or_sensitivity: Either a function (direct decoration) or a
            sensitivity level (decorator factory)

    Returns:
        Decorated function or decorator factory

    Raises:
        ValueError: If sensitivity value is invalid

    Examples:
        >>> @latency_sensitive(LatencySensitivity.HIGH)
        ... async def critical_llm_call():
        ...     return await llm.generate()

        >>> @latency_sensitive("low")
        ... def background_task():
        ...     pass

        >>> def my_function():
        ...     sensitivity = Context.get().latency_sensitivity
        ...     if sensitivity == LatencySensitivity.HIGH:
        ...         # Use fast path
        ...         pass
    """
```

### Implementation Pattern

Following the same pattern as `track_function` and `track_unregistered_function`:

1. **Parse and validate at decoration time** (fail-fast)
2. **Detect function type** using inspect module
3. **Create appropriate wrapper** for each type:
   - Async functions: `async def async_wrapper(*args, **kwargs)`
   - Sync functions: `def sync_wrapper(*args, **kwargs)`
   - Async generators: `async def async_gen_wrapper(*args, **kwargs)`
   - Sync generators: `def sync_gen_wrapper(*args, **kwargs)`

4. **In each wrapper:**
   ```python
   with Context.get().push_latency_sensitivity(parsed_sensitivity):
       # Call original function
       result = await/yield from/... func(*args, **kwargs)
       return result
   ```

5. **Preserve function metadata** using `functools.wraps(func)`

## Testing Strategy

**Test file:** `packages/nvidia_nat_core/tests/nat/profiler/decorators/test_latency.py`

### Test Categories

1. **Enum tests:**
   - Parse valid strings (case-insensitive): "low", "MEDIUM", "High"
   - Parse enum values directly
   - Reject invalid strings with ValueError
   - Priority property returns correct values (LOW=1, MEDIUM=2, HIGH=3)

2. **Context integration tests:**
   - Default sensitivity is MEDIUM
   - Reading `Context.get().latency_sensitivity` works
   - Push/pop maintains stack correctly
   - Lazy initialization works

3. **Decorator tests (for each function type):**
   - Sync function: sensitivity changes inside, reverts after
   - Async function: same behavior
   - Sync generator: sensitivity set during iteration
   - Async generator: same behavior

4. **Priority nesting tests:**
   - HIGH → LOW stays HIGH
   - LOW → HIGH becomes HIGH
   - MEDIUM → HIGH → LOW = HIGH inside middle, reverts correctly
   - Deep nesting (5+ levels) works correctly
   - Mixed function types in call stack

5. **Edge cases:**
   - Invalid sensitivity raises ValueError at decoration time
   - Exception during function execution still pops stack
   - Multiple concurrent contexts (ContextVar isolation)

### Example Test

```python
@pytest.mark.asyncio
async def test_priority_nesting_high_to_low():
    """Test that HIGH sensitivity is maintained when calling LOW function."""

    @latency_sensitive("low")
    async def low_func():
        return Context.get().latency_sensitivity

    @latency_sensitive("high")
    async def high_func():
        inner_sensitivity = await low_func()
        return Context.get().latency_sensitivity, inner_sensitivity

    # Default is MEDIUM
    assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

    # Call high_func
    outer, inner = await high_func()

    # Both should be HIGH (priority wins)
    assert outer == LatencySensitivity.HIGH
    assert inner == LatencySensitivity.HIGH

    # Should revert to MEDIUM
    assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM
```

## Documentation

### Module Docstring

Add comprehensive module-level docstring to `latency.py` explaining:
- Purpose and use cases (LLM routing, optimization, observability)
- Priority behavior and nesting rules
- Usage examples for all decorator styles
- How to read current sensitivity from context

### API Documentation

1. **LatencySensitivity enum:**
   - Document each value (LOW, MEDIUM, HIGH)
   - Document priority property
   - Document parse() method

2. **@latency_sensitive decorator:**
   - Full docstring with args, returns, raises
   - Multiple usage examples
   - Explanation of priority nesting

3. **Context.latency_sensitivity property:**
   - Docstring explaining it returns current effective sensitivity
   - Note about default MEDIUM value

### Package Exports

Update `packages/nvidia_nat_core/src/nat/profiler/decorators/__init__.py`:

```python
from nat.profiler.decorators.latency import LatencySensitivity
from nat.profiler.decorators.latency import latency_sensitive

__all__ = [
    # ... existing exports ...
    "LatencySensitivity",
    "latency_sensitive",
]
```

## Use Cases

### LLM Routing

```python
@latency_sensitive("high")
async def user_facing_query(query: str):
    # Route to low-latency LLM backend
    sensitivity = Context.get().latency_sensitivity
    if sensitivity == LatencySensitivity.HIGH:
        llm = get_fast_llm()
    else:
        llm = get_standard_llm()
    return await llm.generate(query)
```

### Execution Optimization

```python
def get_timeout() -> float:
    sensitivity = Context.get().latency_sensitivity
    return {
        LatencySensitivity.LOW: 60.0,
        LatencySensitivity.MEDIUM: 30.0,
        LatencySensitivity.HIGH: 5.0,
    }[sensitivity]
```

### Observability

```python
def log_with_sensitivity(message: str):
    sensitivity = Context.get().latency_sensitivity
    logger.info(f"[{sensitivity.value}] {message}")
```

## Files Modified

1. **New file:** `packages/nvidia_nat_core/src/nat/profiler/decorators/latency.py`
   - LatencySensitivity enum
   - latency_sensitive decorator
   - ~200 lines

2. **Modified:** `packages/nvidia_nat_core/src/nat/builder/context.py`
   - Add `_latency_sensitivity_stack` to ContextState
   - Add `latency_sensitivity` property to Context
   - Add `push_latency_sensitivity()` context manager
   - ~40 lines added

3. **Modified:** `packages/nvidia_nat_core/src/nat/profiler/decorators/__init__.py`
   - Export new symbols
   - ~2 lines

4. **New file:** `packages/nvidia_nat_core/tests/nat/profiler/decorators/test_latency.py`
   - Comprehensive test suite
   - ~500 lines

## Implementation Checklist

- [ ] Create `latency.py` with LatencySensitivity enum
- [ ] Implement `parse()` and `priority` methods
- [ ] Add ContextVar to ContextState
- [ ] Add property and context manager to Context
- [ ] Implement decorator for all function types
- [ ] Write comprehensive tests
- [ ] Update `__init__.py` exports
- [ ] Add documentation and docstrings
- [ ] Run tests with coverage (target 80%+)
- [ ] Run linting and formatting
- [ ] Manual testing with example workflows

## Success Criteria

- [ ] All tests pass with 80%+ coverage
- [ ] Decorator works with sync/async, generator/non-generator
- [ ] Priority nesting behaves correctly
- [ ] Invalid inputs raise ValueError at decoration time
- [ ] Documentation is clear and comprehensive
- [ ] No performance regression in existing code
- [ ] Linting passes
