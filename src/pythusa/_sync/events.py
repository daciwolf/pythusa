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
    """Human-readable wrapper around multiprocessing.Event."""

    __slots__ = ("name", "_event")

    def __init__(self, name: str, initial_state: bool = False):
        self.name = name
        self._event = mp.Event()
        if initial_state:
            self._event.set()

    def signal(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def is_open(self) -> bool:
        return self._event.is_set()

    @property
    def event(self) -> mp.synchronize.Event:
        return self._event

    def __repr__(self) -> str:
        state = "OPEN" if self.is_open() else "CLOSED"
        return f"<WorkerEvent '{self.name}' [{state}]>"

