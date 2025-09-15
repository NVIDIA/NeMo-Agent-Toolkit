import pytest

from nat.eval.utils.tqdm_position_registry import TqdmPositionRegistry


def test_claim_and_release_positions():
    pos = TqdmPositionRegistry.claim()
    assert isinstance(pos, int)
    TqdmPositionRegistry.release(pos)
    # after release, we should be able to claim the same position again quickly
    reclaimed = TqdmPositionRegistry.claim()
    TqdmPositionRegistry.release(reclaimed)


def test_exhaust_positions_then_error(monkeypatch):
    # set small max to speed up
    monkeypatch.setattr(TqdmPositionRegistry, "_max_positions", 2)
    # reset positions
    # Reset positions (test-only)
    TqdmPositionRegistry._positions.clear()
    a = TqdmPositionRegistry.claim()
    b = TqdmPositionRegistry.claim()
    assert {a, b} == {0, 1}
    with pytest.raises(RuntimeError):
        TqdmPositionRegistry.claim()
    # cleanup
    TqdmPositionRegistry.release(a)
    TqdmPositionRegistry.release(b)
