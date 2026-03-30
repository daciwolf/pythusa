from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

from .._buffers.ring import SharedRingBuffer
from .._core import context
from .._sync.events import WorkerEvent
from .worker import Worker


@dataclass(slots=True)
class TaskBootstrap:
    """
    Picklable task bootstrap used as the multiprocessing spawn target.
    """

    name: str
    fn: Callable[..., Any]
    reading_ring_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    writing_ring_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: dict[str, WorkerEvent] = field(default_factory=dict)
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __call__(self) -> None:
        logging.debug("[%s] start", self.name)

        reading_rings = {name: SharedRingBuffer(**kw) for name, kw in self.reading_ring_kwargs.items()}
        writing_rings = {name: SharedRingBuffer(**kw) for name, kw in self.writing_ring_kwargs.items()}

        with ExitStack() as stack:
            for ring in {**reading_rings, **writing_rings}.values():
                stack.enter_context(ring)
            context._install(reading_rings, writing_rings, self.events)
            Worker(fn=self._run_task)()

        logging.debug("[%s] end", self.name)

    def _run_task(self) -> None:
        self.fn(*self.args, **self.kwargs)
