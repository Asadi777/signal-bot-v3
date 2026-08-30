"""Global token bucket.

One bucket is shared by every source in a run. The archive-first strategy keeps
REST traffic small, and this ceiling is what keeps an accidental loop from
turning into an IP ban that would cost days of research time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    rate_per_second: float
    capacity: float = 1.0
    clock: "callable" = time.monotonic
    sleeper: "callable" = time.sleep
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)
    total_wait_seconds: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self.capacity = max(self.capacity, 1.0)
        self._tokens = self.capacity
        self._last = self.clock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available. Returns the seconds waited."""
        waited = 0.0
        while True:
            now = self.clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_second)
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                self.total_wait_seconds += waited
                return waited
            need = (tokens - self._tokens) / self.rate_per_second
            self.sleeper(need)
            waited += need
