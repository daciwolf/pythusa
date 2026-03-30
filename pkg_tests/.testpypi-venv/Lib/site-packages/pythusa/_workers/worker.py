from __future__ import annotations

import signal
from dataclasses import dataclass, field
from typing import Any, Callable


__all__ = ["TaskSpec", "Worker"]


# ------------------------------------------------------------------ #
# Specs — pure frozen data, no logic, no live objects                 #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class TaskSpec:
    """
    One logical unit of work. First-class citizen — registered with Manager
    independently of any worker. The user decides later whether to group
    tasks into a shared process or let each run in its own process.

    - name:          unique identifier. Manager indexes tasks by this.
    - fn:            the callable to run as the process body. It owns its
                     own loop and should return only when the task should exit.
    - reading_rings: names of rings this task reads from. Manager opens
                     these with an assigned reader slot index.
    - writing_rings: names of rings this task writes to. Manager opens
                     these with reader=-1 (writer handle).
    - events:        event names this task interacts with. Direction is fn's concern.
    - args:          if provided, passed directly to fn instead of resolved rings.
    - kwargs:        passed to fn as keyword arguments.

    The read/write distinction is required — Manager uses it to open ring
    handles correctly in the child process. A reader handle opened as a
    writer (or vice versa) produces silent corruption or blocks forever.

    Error handling is fn's responsibility. The framework does not catch,
    retry, or route exceptions from inside a task.
    """
    name: str
    fn: Callable[..., Any]
    reading_rings: tuple[str, ...] = ()
    writing_rings: tuple[str, ...] = ()
    events:        tuple[str, ...] = ()
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = set(self.reading_rings) & set(self.writing_rings)
        if overlap:
            raise ValueError(
                f"TaskSpec '{self.name}': rings {overlap} appear in both "
                f"reading_rings and writing_rings. A ring is either read or written per task."
            )

    def __repr__(self) -> str:
        fn_name = getattr(self.fn, "__name__", type(self.fn).__name__)
        return (
            f"TaskSpec(name={self.name!r}, fn={fn_name!r}, "
            f"reading_rings={self.reading_rings}, writing_rings={self.writing_rings})"
        )

# ------------------------------------------------------------------ #
# Worker — dumb execution primitive                                   #
# ------------------------------------------------------------------ #

class Worker:
    """
    One process. One callable invocation.
    Manager composes everything before handing it here.
    Worker never sees tasks, rings, or events — only the composed callable.
    The callable itself owns any long-running loop.
    """

    # Registry for subclasses that need a genuinely different execution loop.
    # For all normal cases plain Worker is the right choice.
    #
    #   class AsyncWorker(Worker, worker_type="async"):
    #       def __call__(self): ...
    _registry: dict[str, type["Worker"]] = {}

    def __init_subclass__(cls, worker_type: str | None = None, **kw) -> None:
        super().__init_subclass__(**kw)
        if worker_type:
            Worker._registry[worker_type] = cls

    def __init__(self, fn: Callable[[], Any]):
        self._fn = fn  # composed by Manager — one opaque callable

    def __call__(self) -> None:
        """
        The process target. SIGTERM raises SystemExit which exits cleanly.
        The task callable is invoked exactly once and is responsible for
        managing its own run loop.

        NOTE: on Windows proc.terminate() is equivalent to SIGKILL — the
        signal handler never runs. Shared memory cleanup still happens via
        weakref.finalize on each SharedRingBuffer and the try/finally in
        Manager's _bootstrap.
        """
        def _handle_sigterm(*_):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _handle_sigterm)
        self._fn()

    def __repr__(self) -> str:
        fn_name = getattr(self._fn, "__name__", type(self._fn).__name__)
        return f"<Worker fn={fn_name}>"
