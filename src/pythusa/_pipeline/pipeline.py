from __future__ import annotations

"""
Minimal public pipeline scaffold.

Approved target workflow:

```python
pipe = pythusa.Pipeline("radar")

pipe.add_stream("samples", shape=(4096,), dtype=np.float32)
pipe.add_stream("fft", shape=(2049,), dtype=np.complex64)

pipe.add_task(
    "acquire",
    fn=acquire,
    writes={"samples": "samples"},
)

pipe.add_task(
    "fft_worker_1",
    fn=fft_worker,
    reads={"samples": "samples"},
    writes={"fft": "fft"},
)

pipe.add_task(
    "db_writer",
    fn=store_fft,
    reads={"fft": "fft"},
)
```

Important invariants for the eventual implementation:

- the public API stays centered on `Pipeline`, `add_stream()`, and `add_task()`
- each stream compiles to one low-level ring
- each ring has exactly one writer and zero or more readers
- fan-out is native
- fan-in happens at the task level by reading from multiple streams
- events remain first-class because the low-level runtime already supports them

This file is intentionally a scaffold. It documents the shape we agreed on
without implementing behavior yet.
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .._sync.events import EventSpec
from .._workers.manager import Manager, ProcessMetrics
from ._helpers import (
    _invoke_task_with_bindings,
    build_stream_topology,
    build_task_graph,
    ring_spec_for_stream,
    task_spec_for_name,
    topological_task_order,
    warn_on_shared_event_fanout,
)
from ._task_wrappers import _TaskRegistrationAPI
from ._toml_io import (
    read_pipeline_toml,
    render_pipeline_toml,
    require_keys,
    resolve_callable,
)


class Pipeline:
    def __init__(self, name: str) -> None:
        self.name = name
        self._manager: Manager = Manager()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._events: dict[str, dict[str, Any]] = {}
        self._streams: dict[str, dict[str, Any]] = {}
        self._compiled = False
        self._started = False
        self._closed = False
        self._task_order: tuple[str, ...] = ()
        self._task_start_order: tuple[str, ...] = ()
        self.add_task = _TaskRegistrationAPI(self)

    def add_stream(
        self,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        cache_align: bool = True,
        description: str | None = None,
    ) -> "Pipeline":
        self._ensure_open()
        self._register_unique(
            store=self._streams,
            kind="Stream",
            name=name,
            declaration={
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "cache_align": cache_align,
                "description": description,
            },
        )
        return self

    def _add_task(
        self,
        name: str,
        *,
        fn: Callable[..., Any],
        reads: dict[str, str] | None = None,
        writes: dict[str, str] | None = None,
        events: dict[str, str] | None = None,
        description: str | None = None,
        control_mode: str | None = None,
        control_event: str | None = None,
    ) -> "Pipeline":
        self._ensure_open()
        task_events = dict(events or {})
        if control_mode is not None and control_event not in task_events:
            raise ValueError(
                f"Task '{name}' declares activate_on='{control_event}' but that "
                "name is not present in the task's event bindings."
            )
        self._register_unique(
            store=self._tasks,
            kind="Task",
            name=name,
            declaration={
                "name": name,
                "fn": fn,
                "reads": dict(reads or {}),
                "writes": dict(writes or {}),
                "events": task_events,
                "description": description,
                "control_mode": control_mode,
                "control_event": control_event,
            },
        )
        return self

    def add_event(
        self,
        name: str,
        *,
        initial_state: bool = False,
        description: str | None = None,
    ) -> "Pipeline":
        self._ensure_open()
        self._register_unique(
            store=self._events,
            kind="Event",
            name=name,
            declaration={
                "name": name,
                "initial_state": initial_state,
                "description": description,
            },
        )
        return self

    def compile(self) -> None:
        self._ensure_open()
        if self._compiled:
            raise RuntimeError("Pipeline has already been compiled.")

        warn_on_shared_event_fanout(self._tasks, self._events)
        stream_writers, stream_readers = build_stream_topology(
            self._tasks,
            self._streams,
            self._events,
        )
        task_graph = build_task_graph(
            self._tasks,
            self._streams,
            stream_writers,
            stream_readers,
        )

        self._task_order = topological_task_order(task_graph)
        self._task_start_order = tuple(reversed(self._task_order))

        self._register_events_with_manager()
        self._register_streams_with_manager(stream_readers)
        self._register_tasks_with_manager()
        self._compiled = True

    def start(self) -> None:
        self._ensure_open()
        if self._started:
            raise RuntimeError("Pipeline has already been started.")
        if not self._compiled:
            self.compile()

        for task_name in self._task_start_order:
            self._manager.start(task_name)

        self._started = True

    def run(self) -> None:
        self.start()
        self.join()

    def start_monitor(self, interval_s: float = 0.01) -> "Pipeline":
        self._ensure_open()
        self._manager.start_monitor(interval_s=interval_s)
        return self

    def metrics(
        self,
        task_name: str | None = None,
    ) -> ProcessMetrics | dict[str, ProcessMetrics | None] | None:
        if task_name is not None:
            if task_name not in self._tasks:
                raise KeyError(f"Task '{task_name}' is not registered.")
            return self._manager.get_metrics(task_name)

        return {
            name: self._manager.get_metrics(name)
            for name in self._tasks
        }

    def stop(self) -> None:
        if self._closed or not self._started:
            return
        self._manager.stop_all()

    def join(self, timeout: float | None = None) -> None:
        if self._closed or not self._started:
            return
        self._manager.join_all(timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self.stop()
        self.join()
        self._manager.close()
        self._started = False
        self._closed = True

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render_pipeline_toml(
                name=self.name,
                streams=self._streams,
                events=self._events,
                tasks=self._tasks,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def reconstruct(cls, path: str | Path) -> "Pipeline":
        data = read_pipeline_toml(Path(path))
        pipe = cls(data["name"])

        for stream in data.get("streams", []):
            require_keys(stream, "stream", "name", "shape", "dtype")
            pipe.add_stream(
                stream["name"],
                shape=tuple(stream["shape"]),
                dtype=np.dtype(stream["dtype"]),
                cache_align=stream.get("cache_align", True),
                description=stream.get("description"),
            )

        for event in data.get("events", []):
            require_keys(event, "event", "name")
            pipe.add_event(
                event["name"],
                initial_state=event.get("initial_state", False),
                description=event.get("description"),
            )

        for task in data.get("tasks", []):
            require_keys(task, "task", "name", "function_module", "function_qualname")
            task_fn = resolve_callable(task["function_module"], task["function_qualname"])
            control_mode = task.get("control_mode")
            control_event = task.get("control_event")

            if control_mode == "switchable":
                pipe.add_task.switchable(
                    task["name"],
                    activate_on=control_event,
                    fn=task_fn,
                    reads=dict(task.get("reads", {})),
                    writes=dict(task.get("writes", {})),
                    events=dict(task.get("events", {})),
                    description=task.get("description"),
                )
                continue

            if control_mode == "toggleable":
                pipe.add_task.toggleable(
                    task["name"],
                    activate_on=control_event,
                    fn=task_fn,
                    reads=dict(task.get("reads", {})),
                    writes=dict(task.get("writes", {})),
                    events=dict(task.get("events", {})),
                    description=task.get("description"),
                )
                continue

            pipe.add_task(
                task["name"],
                fn=task_fn,
                reads=dict(task.get("reads", {})),
                writes=dict(task.get("writes", {})),
                events=dict(task.get("events", {})),
                description=task.get("description"),
            )

        return pipe

    def __enter__(self) -> "Pipeline":
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Pipeline(name={self.name!r}, streams={len(self._streams)}, "
            f"tasks={len(self._tasks)}, events={len(self._events)}, "
            f"compiled={self._compiled}, started={self._started}, closed={self._closed})"
        )

    def _register_unique(
        self,
        *,
        store: dict[str, dict[str, Any]],
        kind: str,
        name: str,
        declaration: dict[str, Any],
    ) -> None:
        if name in store:
            raise ValueError(f"{kind} '{name}' is already registered.")
        store[name] = declaration

    def _register_events_with_manager(self) -> None:
        for event in self._events.values():
            self._manager.create_event(
                EventSpec(
                    name=event["name"],
                    initial_state=event["initial_state"],
                )
            )

    def _register_streams_with_manager(
        self,
        stream_readers: dict[str, list[str]],
    ) -> None:
        for stream in self._streams.values():
            self._manager.create_ring(
                ring_spec_for_stream(
                    stream,
                    reader_count=len(stream_readers[stream["name"]]),
                )
            )

    def _register_tasks_with_manager(self) -> None:
        for task_name in self._task_order:
            self._manager.create_task(
                task_spec_for_name(
                    task_name,
                    self._tasks[task_name],
                    self._streams,
                )
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Pipeline is closed.")



    


__all__ = ["Pipeline"]
