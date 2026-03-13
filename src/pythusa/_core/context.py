from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._buffers.ring import SharedRingBuffer
    from .._sync.events import WorkerEvent

# These are empty in the parent process.
# In the child process they are populated once by _bootstrap
# before the task loop starts. Module-level state is safe here
# because each worker is its own process; nothing is shared.
_reading_rings: dict[str, "SharedRingBuffer"] = {}
_writing_rings: dict[str, "SharedRingBuffer"] = {}
_events: dict[str, "WorkerEvent"] = {}


def get_reader(name: str) -> "SharedRingBuffer":
    try:
        return _reading_rings[name]
    except KeyError as exc:
        raise KeyError(
            f"No reading ring '{name}' in this process. "
            f"Did you add it to the task's reading_rings?"
        ) from exc


def get_writer(name: str) -> "SharedRingBuffer":
    try:
        return _writing_rings[name]
    except KeyError as exc:
        raise KeyError(
            f"No writing ring '{name}' in this process. "
            f"Did you add it to the task's writing_rings?"
        ) from exc


def get_event(name: str) -> "WorkerEvent":
    try:
        return _events[name]
    except KeyError as exc:
        raise KeyError(
            f"No event '{name}' in this process. "
            f"Did you add it to the task's events?"
        ) from exc


def _install(
    reading_rings: dict,
    writing_rings: dict,
    events: dict,
) -> None:
    """
    Called once by Manager._bootstrap before any task runs.
    Never called by user code.
    """
    _reading_rings.clear()
    _writing_rings.clear()
    _events.clear()
    _reading_rings.update(reading_rings)
    _writing_rings.update(writing_rings)
    _events.update(events)
