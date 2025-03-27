class TqdmPositionRegistry:
    _positions = set()

    @classmethod
    def claim(cls) -> int:
        for i in range(100):
            if i not in cls._positions:
                cls._positions.add(i)
                return i
        raise RuntimeError("No available tqdm positions.")

    @classmethod
    def release(cls, pos: int):
        cls._positions.discard(pos)
