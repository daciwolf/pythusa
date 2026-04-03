from __future__ import annotations

from importlib.metadata import version as _version

from ._buffers.ring import RingSpec, SharedRingBuffer
from ._core.context import get_event, get_reader, get_writer
from ._pipeline import Pipeline
from ._sync.events import EventSpec, WorkerEvent
from ._workers.manager import Manager, ProcessMetrics
from ._workers.worker import TaskSpec, Worker

__version__ = _version("pythusa")

__all__ = [
    "__version__",
    "Manager",
    "Pipeline",
    "ProcessMetrics",
    "SharedRingBuffer",
    "RingSpec",
    "Worker",
    "TaskSpec",
    "EventSpec",
    "WorkerEvent",
    "get_reader",
    "get_writer",
    "get_event",
]

