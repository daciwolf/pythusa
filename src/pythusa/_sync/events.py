from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional


__all__ = ["EventSpec", "WorkerEvent"]


@dataclass(frozen=True)
class EventSpec:
    """Describes a synchronization gate resolved by Manager."""

    name: str
    initial_state: bool = False

    def __repr__(self) -> str:
        state = "OPEN" if self.initial_state else "CLOSED"
        return f"EventSpec(name={self.name!r}, initial_state={state})"


class WorkerEvent:
    """
    Named process-shared event with counted activations.

    `signal()` increments the pending activation count and opens the wake
    event. `reset()` consumes one pending activation. The underlying event is
    cleared only when the pending count reaches zero.
    """

    __slots__ = ("name", "_count", "_event", "_lock", "_semaphore")

    def __init__(self, name: str, initial_state: bool = False):
        self.name = name
        initial_count = 1 if initial_state else 0
        self._count = mp.Value("q", initial_count, lock=False)
        self._event = mp.Event()
        self._lock = mp.Lock()
        self._semaphore = mp.Semaphore(initial_count)
        if initial_count:
            self._event.set()

    def signal(self) -> None:
        with self._lock:
            self._semaphore.release()
            self._count.value += 1
            self._event.set()

    def reset(self) -> None:
        with self._lock:
            if self._count.value <= 0:
                self._event.clear()
                return

            acquired = self._semaphore.acquire(block=False)
            if not acquired:
                self._count.value = 0
                self._event.clear()
                return

            self._count.value -= 1
            if self._count.value == 0:
                self._event.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def is_open(self) -> bool:
        return self._event.is_set()

    @property
    def event(self) -> mp.synchronize.Event:
        return self._event

    @property
    def pending(self) -> int:
        return int(self._count.value)

    def __repr__(self) -> str:
        state = "OPEN" if self.is_open() else "CLOSED"
        return f"<WorkerEvent '{self.name}' [{state}] pending={self.pending}>"

