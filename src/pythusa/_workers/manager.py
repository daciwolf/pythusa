from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time

from dataclasses import dataclass

from .._buffers.ring import RingSpec, SharedRingBuffer
from .._sync.events import EventSpec, WorkerEvent
from .bootstrap import TaskBootstrap
from .worker import TaskSpec

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    class _MissingPsutil:
        @staticmethod
        def Process(*_args, **_kwargs):
            raise ModuleNotFoundError("psutil is required for process monitoring features")

    psutil = _MissingPsutil()


__all__ = [
    "Manager",
    "ProcessMetrics",
]


@dataclass
class ProcessMetrics:
    """
    Snapshot of a worker process's resource usage and ring pressure.
    Collected by the monitor thread and stored in Manager._metrics
    so the user can query it at any time.

    Fields:
        name          : task name this process is running
        pid           : OS process id
        cpu_percent   : CPU usage % since last sample (psutil)
        memory_rss_mb : resident set size in megabytes
        nice          : current nice value
        ring_pressure : dict of ring_name -> pressure % for every
                        ring this task reads or writes. Empty if the
                        task has no rings registered in _rings.
        sampled_at    : time.perf_counter() timestamp of this sample
    """
    name:          str
    pid:           int
    cpu_percent:   float
    memory_rss_mb: float
    nice:          int
    ring_pressure: dict[str, int]
    sampled_at:    float

class Manager:
    """
    Owns the extracted runtime's live rings, events, and worker processes.

    The extracted library currently starts one ``TaskSpec`` per process.
    The old app repository described grouped tasks, but that behavior was
    not implemented cleanly enough to expose as part of PYTHUSA yet.
    """

    def __init__(self, mp_context: str = "spawn"):
        self._ctx           = mp.get_context(mp_context)
        self._ring_specs:   dict[str, RingSpec]       = {}
        self._rings:        dict[str, SharedRingBuffer] = {}  # strong refs: Manager owns
        self._event_specs:  dict[str, EventSpec]    = {}
        self._events:       dict[str, WorkerEvent]  = {}
        self._task_specs:   dict[str, TaskSpec]     = {}
        self._processes:    dict[str, mp.Process]   = {}
        # Tracks how many reader slots have been assigned per ring so each
        # reader process gets a unique index matching the header layout.
        self._ring_reader_counters: dict[str, int]  = {}
        self._tasks_started: dict[str, bool] = {}
        self._metrics: dict[str, ProcessMetrics] = {}

    # ------------------------------------------------------------------ #
    # Registration                                                       #
    # ------------------------------------------------------------------ #

    def create_ring(self, spec: RingSpec) -> "Manager":
        self._ring_specs[spec.name] = spec
        self._rings[spec.name] = SharedRingBuffer(
            **spec.to_kwargs(create=True, reader=SharedRingBuffer._NO_READER)
        )
        self._ring_reader_counters[spec.name] = 0
        return self

    def create_event(self, spec: EventSpec) -> "Manager":
        """Register an event spec and create the live WorkerEvent."""
        self._event_specs[spec.name] = spec
        self._events[spec.name] = WorkerEvent(name=spec.name, initial_state=spec.initial_state)
        return self

    def create_task(self, spec: TaskSpec) -> "Manager":
        """Register a task that will run in its own worker process."""
        if spec.name in self._task_specs:
            raise ValueError(f"Task '{spec.name}' is already registered")
        self._task_specs[spec.name] = spec
        self._tasks_started[spec.name] = False
        return self

    def _create_ring_kwargs(self, tasks: list[TaskSpec]) -> tuple[dict[str, dict], dict[str, dict]]:
        reading_ring_kwargs: dict[str, dict] = {}
        writing_ring_kwargs: dict[str, dict] = {}
         
        for task in tasks:
            for rname in task.reading_rings:
                if rname not in self._ring_specs:
                    raise KeyError(
                        f"Task '{task.name}' reading_ring '{rname}' is not registered. "
                        f"Call create_ring() first."
                    )
                if rname not in reading_ring_kwargs:
                    # Assign the next available reader slot for this ring
                    slot = self._ring_reader_counters[rname]
                    spec = self._ring_specs[rname]
                    if slot >= spec.num_readers:
                        raise ValueError(
                            f"Ring '{rname}' has num_readers={spec.num_readers} but "
                            f"slot {slot} was requested. Increase num_readers in the RingSpec."
                        )
                    self._ring_reader_counters[rname] += 1
                    reading_ring_kwargs[rname] = spec.to_kwargs(create=False, reader=slot)

            for rname in task.writing_rings:
                if rname not in self._ring_specs:
                    raise KeyError(
                        f"Task '{task.name}' writing_ring '{rname}' is not registered. "
                        f"Call create_ring() first."
                    )
                if rname not in writing_ring_kwargs:
                    writing_ring_kwargs[rname] = self._ring_specs[rname].to_kwargs(
                        create=False, reader=SharedRingBuffer._NO_READER
                    )
        return (reading_ring_kwargs, writing_ring_kwargs)


    def _create_events(self, tasks: list[TaskSpec]) -> dict[str, WorkerEvent]:
        events: dict[str, WorkerEvent] = {}
        for task in tasks:
            for ename in task.events:
                if ename not in self._events:
                    raise KeyError(
                        f"Task '{task.name}' references event '{ename}' which is not registered. "
                        f"Call create_event() first."
                    )
                events[ename] = self._events[ename]
        return events
    

    def _validate_task(self, name: str) -> TaskSpec:
        # make sure task exists and if so return task
        task = self._task_specs.get(name, None)
        if task:
            return task
        else:
            raise ValueError(f"TaskSpec of {name} is not registered with manager")

    def _task_bootstrap(self, name: str) -> TaskBootstrap:
        task = self._validate_task(name=name)
        reading_ring_kwargs, writing_ring_kwargs = self._create_ring_kwargs([task])
        events = self._create_events([task])

        return TaskBootstrap(
            name=name,
            fn=task.fn,
            reading_ring_kwargs=reading_ring_kwargs,
            writing_ring_kwargs=writing_ring_kwargs,
            events=events,
            args=task.args,
            kwargs=dict(task.kwargs),
        )
    

    def start(self, name: str) -> mp.Process:
        if name not in self._task_specs:
            raise KeyError(
                f"'{name}' is not a registered task. "
                f"Registered tasks: {list(self._task_specs)}"
            )
        bootstrap = self._task_bootstrap(name)
        proc = self._ctx.Process(target=bootstrap, name=name, daemon=True)
        proc.start()
        self._processes[name] = proc
        self._tasks_started[name] = True
        return proc

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def stop(self, name: str) -> None:
        proc = self._processes.get(name)
        if proc and proc.is_alive():
            proc.terminate()

    def join(self, name: str, timeout: float = 5.0) -> None:
        proc = self._processes.get(name)
        if proc:
            proc.join(timeout=timeout)

    def stop_all(self) -> None:
        for name in list(self._processes):
            self.stop(name)

    def join_all(self, timeout: float = 5.0) -> None:
        for name in list(self._processes):
            self.join(name, timeout=timeout)

    def _collect_ring_pressures(
        self,
        rings: dict,
    ) -> dict[str, int]:
        pressures = {}
        for rname, ring in rings.items():
            try:
                pressures[rname] = ring.calculate_pressure()
            except Exception:
                pass
        return pressures

    def _check_sustained_pressure(
        self,
        ring_pressures: dict[str, int],
        high_pressure_since: dict[str, float],
        now: float,
    ) -> None:
        for rname, pressure in ring_pressures.items():
            if pressure > 90:
                if rname not in high_pressure_since:
                    high_pressure_since[rname] = now
                elif now - high_pressure_since[rname] > 1.0:
                    logging.warning(
                        "Ring '%s' has been at %d%% pressure for %.1fs  "
                        "reader may be stalled.",
                        rname,
                        pressure,
                        now - high_pressure_since[rname],
                    )
            else:
                high_pressure_since.pop(rname, None)

    def _sample_process(
        self,
        name: str,
        proc: mp.Process,
        ring_pressures: dict[str, int],
        task_specs: dict,
        metrics: dict,
        now: float,
    ) -> dict[str, int]:
        """
        Samples one process. Stores a ProcessMetrics snapshot.
        Returns the task_ring_pressure dict for use by _adjust_process_nice.
        """
        ps = psutil.Process(proc.pid)

        cpu  = ps.cpu_percent()
        mem  = ps.memory_info().rss / (1024 * 1024)
        nice = ps.nice()

        task = task_specs.get(name)
        task_ring_pressure: dict[str, int] = {}
        if task is not None:
            for rname in list(task.reading_rings) + list(task.writing_rings):
                if rname in ring_pressures:
                    task_ring_pressure[rname] = ring_pressures[rname]

        metrics[name] = ProcessMetrics(
            name=name,
            pid=proc.pid,
            cpu_percent=cpu,
            memory_rss_mb=mem,
            nice=nice,
            ring_pressure=task_ring_pressure,
            sampled_at=now,
        )
        return task_ring_pressure

    def _adjust_process_nice(
        self,
        proc: mp.Process,
        task_ring_pressure: dict[str, int],
    ) -> None:
        if not task_ring_pressure:
            return
        ps = psutil.Process(proc.pid)
        worst = max(task_ring_pressure.values())
        if worst > 80:
            ps.nice(-10)
        elif worst < 20:
            ps.nice(10)

    def start_monitor(self, interval_s: float = 0.01) -> None:
        """
        Starts the daemon monitor thread.
        Orchestrates _collect_ring_pressures, _check_sustained_pressure,
        _sample_process, and _adjust_process_nice once per interval.
        All psutil calls inside the delegated methods are wrapped in
        try/except so a dead process never crashes the monitor.
        Manager is NOT captured by the thread closure  only plain
        dicts extracted before the thread starts are referenced.
        """
        rings      = self._rings
        processes  = self._processes
        metrics    = self._metrics
        task_specs = self._task_specs

        _high_pressure_since: dict[str, float] = {}

        def _monitor():
            while True:
                now = time.perf_counter()
                ring_pressures = self._collect_ring_pressures(rings)
                self._check_sustained_pressure(
                    ring_pressures, _high_pressure_since, now
                )
                for name, proc in list(processes.items()):
                    try:
                        if not proc.is_alive():
                            continue
                        task_ring_pressure = self._sample_process(
                            name, proc, ring_pressures, task_specs, metrics, now
                        )
                        self._adjust_process_nice(proc, task_ring_pressure)
                    except Exception:
                        pass
                time.sleep(interval_s)

        t = threading.Thread(
            target=_monitor, daemon=True, name="pythusa_monitor"
        )
        t.start()

    def get_metrics(self, name: str) -> ProcessMetrics | None:
        """
        Return the most recent ProcessMetrics snapshot for a task,
        or None if no sample has been collected yet.
        """
        return self._metrics.get(name)

    def close(self) -> None:
        """Close and unlink all creator-side ring handles."""
        for ring in self._rings.values():
            ring.__exit__(None, None, None)
        self._rings.clear()

    def __enter__(self) -> "Manager":
        return self

    def __exit__(self, *_):
        self.stop_all()
        self.join_all()
        self.close()

    def __repr__(self):
        return (
            f"<Manager"
            f" tasks={list(self._task_specs)}"
            f" processes={list(self._processes)}"
            f" rings={list(self._ring_specs)}"
            f" events={list(self._event_specs)}>"
        )
