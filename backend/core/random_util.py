from typing import Any, List, Optional

import numpy as np

__all__ = ["mersenne_rng", "RandomGenerator"]


class mersenne_rng:
    def __init__(self, seed: int = 5489) -> None:
        self.state = [0] * 624
        self.f = 1812433253
        self.m = 397
        self.u = 11
        self.s = 7
        self.b = 0x9D2C5680
        self.t = 15
        self.c = 0xEFC60000
        self.l = 18
        self.index = 624
        self.lower_mask = (1 << 31) - 1
        self.upper_mask = 1 << 31

        # initialize state (ensure 32-bit ints)
        self.state[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            prev = self.state[i - 1]
            self.state[i] = self.int_32(self.f * (prev ^ (prev >> 30)) + i)

    def twist(self) -> None:
        for i in range(624):
            temp = self.int_32(
                (self.state[i] & self.upper_mask)
                + (self.state[(i + 1) % 624] & self.lower_mask)
            )
            temp_shift = temp >> 1
            if temp & 1:
                temp_shift ^= 0x9908B0DF
            self.state[i] = self.state[(i + self.m) % 624] ^ temp_shift
        self.index = 0

    def int_32(self, number: int) -> int:
        return int(0xFFFFFFFF & number)

    def randint(self) -> int:
        """Return a 32-bit unsigned integer (0 .. 2**32-1)."""
        if self.index >= 624:
            self.twist()
        y = self.state[self.index]
        y ^= y >> self.u
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= y >> self.l
        self.index += 1
        return self.int_32(y)

    def rand(self) -> float:
        """Return a float in [0.0, 1.0)."""
        return self.randint() * (1.0 / 4294967296.0)

    def randperm(self, n: int) -> List[int]:
        """Return a random permutation of range(n) as a list.

        Fixed an off-by-one bug: use modulo (i+1) so j is in [0, i].
        """
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.randint() % (i + 1)
            p[i], p[j] = p[j], p[i]
        return p


class RandomGenerator:
    def __init__(self, seed: int = 42, use_numpy: bool = True) -> None:
        self.use_numpy = bool(use_numpy)
        self.rng: Any
        if self.use_numpy:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = mersenne_rng(seed)

    def randint(self, low: int, high: Optional[int] = None) -> int:
        if self.use_numpy:
            if high is None:
                return int(self.rng.integers(low))
            return int(self.rng.integers(low, high))

        if high is None:
            high = low
            low = 0
        if high <= low:
            raise ValueError("high must be greater than low")
        return low + (self.rng.randint() % (high - low))

    def rand(self) -> float:
        if self.use_numpy:
            return float(self.rng.random())
        return self.rng.rand()

    def randperm(self, n: int) -> List[int]:
        if self.use_numpy:
            return self.rng.permutation(n).tolist()
        return self.rng.randperm(n)
