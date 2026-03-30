from __future__ import annotations

import inspect
import warnings
from graphlib import CycleError, TopologicalSorter
from typing import Any, Callable

from .._buffers.ring import RingSpec
from .._core.context import get_event, get_reader, get_writer
from .._processing.numpy import bytes_for_shape
from .._utils import align_size
from .._workers.worker import TaskSpec
from ._stream_io import make_reader_binding, make_writer_binding
from ._task_wrappers import run_controlled_task

_DEFAULT_RING_DEPTH = 32
_CACHE_ALIGNMENT_BYTES = 64


def build_stream_topology(
    tasks: dict[str, dict[str, Any]],
    streams: dict[str, dict[str, Any]],
    events: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    stream_writers: dict[str, str] = {}
    stream_readers = {name: [] for name in streams}

    for task_name, task in tasks.items():
        _validate_task_bindings(task_name, task)
        _collect_task_reads(task_name, task, stream_readers, streams)
        _collect_task_writes(task_name, task, stream_writers, streams)
        _validate_task_events(task_name, task, events)

    return stream_writers, stream_readers


def warn_on_shared_event_fanout(
    tasks: dict[str, dict[str, Any]],
    events: dict[str, dict[str, Any]],
) -> None:
    event_users = {name: [] for name in events}

    for task_name, task in tasks.items():
        for event_name in task["events"].values():
            if event_name in event_users:
                event_users[event_name].append(task_name)

    for event_name, task_names in event_users.items():
        if len(task_names) <= 2:
            continue

        warnings.warn(
            "Event "
            f"'{event_name}' is bound into {len(task_names)} tasks "
            f"({', '.join(task_names)}). "
            "PYTHUSA events are intended for one producer/one consumer style "
            "coordination. Prefer separate events per consumer instead of "
            "sharing one event across many tasks.",
            stacklevel=2,
        )


def build_task_graph(
    tasks: dict[str, dict[str, Any]],
    streams: dict[str, dict[str, Any]],
    stream_writers: dict[str, str],
    stream_readers: dict[str, list[str]],
) -> dict[str, set[str]]:
    task_graph = {name: set() for name in tasks}

    for stream_name in streams:
        writer = stream_writers.get(stream_name)
        readers = stream_readers[stream_name]

        if writer is None:
            raise ValueError(f"Stream '{stream_name}' does not have a writer.")
        if not readers:
            raise ValueError(f"Stream '{stream_name}' does not have any readers.")

        for reader in readers:
            task_graph[reader].add(writer)

    return task_graph


def topological_task_order(task_graph: dict[str, set[str]]) -> tuple[str, ...]:
    try:
        return tuple(TopologicalSorter(task_graph).static_order())
    except CycleError as exc:
        raise ValueError("Pipeline task graph contains a cycle.") from exc


def ring_spec_for_stream(stream: dict[str, Any], *, reader_count: int) -> RingSpec:
    ring_size = bytes_for_shape(stream["shape"], stream["dtype"]) * _DEFAULT_RING_DEPTH
    if stream["cache_align"]:
        ring_size = align_size(ring_size, _CACHE_ALIGNMENT_BYTES)

    return RingSpec(
        name=stream["name"],
        size=ring_size,
        num_readers=reader_count,
        cache_align=stream["cache_align"],
    )


def task_spec_for_name(
    task_name: str,
    task: dict[str, Any],
    streams: dict[str, dict[str, Any]],
) -> TaskSpec:
    return TaskSpec(
        name=task_name,
        fn=_invoke_task_with_bindings,
        reading_rings=tuple(task["reads"].values()),
        writing_rings=tuple(task["writes"].values()),
        events=tuple(task["events"].values()),
        args=(
            task["fn"],
            dict(task["reads"]),
            dict(task["writes"]),
            dict(task["events"]),
            task.get("control_mode"),
            task.get("control_event"),
            _binding_stream_specs(task["reads"], streams),
            _binding_stream_specs(task["writes"], streams),
        ),
    )


def _invoke_task_with_bindings(
    fn: Callable[..., Any],
    reads: dict[str, str],
    writes: dict[str, str],
    events: dict[str, str],
    control_mode: str | None = None,
    control_event: str | None = None,
    read_specs: dict[str, dict[str, Any]] | None = None,
    write_specs: dict[str, dict[str, Any]] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    read_specs = read_specs or {}
    write_specs = write_specs or {}

    for local_name, stream_name in reads.items():
        kwargs[local_name] = make_reader_binding(
            get_reader(stream_name),
            **read_specs[local_name],
        )

    for local_name, stream_name in writes.items():
        kwargs[local_name] = make_writer_binding(
            get_writer(stream_name),
            **write_specs[local_name],
        )

    for local_name, event_name in events.items():
        kwargs[local_name] = get_event(event_name)

    return run_controlled_task(
        fn,
        control_mode=control_mode,
        activate_on=control_event,
        kwargs=kwargs,
    )


def _binding_stream_specs(
    bindings: dict[str, str],
    streams: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for local_name, stream_name in bindings.items():
        stream = streams[stream_name]
        specs[local_name] = {
            "name": stream_name,
            "shape": tuple(stream["shape"]),
            "dtype": stream["dtype"],
        }
    return specs


def _collect_task_reads(
    task_name: str,
    task: dict[str, Any],
    stream_readers: dict[str, list[str]],
    streams: dict[str, dict[str, Any]],
) -> None:
    for stream_name in task["reads"].values():
        _require_stream(task_name, stream_name, action="reads", streams=streams)
        readers = stream_readers[stream_name]
        if task_name not in readers:
            readers.append(task_name)


def _collect_task_writes(
    task_name: str,
    task: dict[str, Any],
    stream_writers: dict[str, str],
    streams: dict[str, dict[str, Any]],
) -> None:
    for stream_name in task["writes"].values():
        _require_stream(task_name, stream_name, action="writes", streams=streams)

        existing_writer = stream_writers.get(stream_name)
        if existing_writer is not None:
            raise ValueError(
                f"Stream '{stream_name}' has multiple writers: "
                f"'{existing_writer}' and '{task_name}'."
            )

        stream_writers[stream_name] = task_name


def _validate_task_events(
    task_name: str,
    task: dict[str, Any],
    events: dict[str, dict[str, Any]],
) -> None:
    for event_name in task["events"].values():
        if event_name not in events:
            raise KeyError(
                f"Task '{task_name}' references unknown event '{event_name}'."
            )


def _require_stream(
    task_name: str,
    stream_name: str,
    *,
    action: str,
    streams: dict[str, dict[str, Any]],
) -> None:
    if stream_name not in streams:
        raise KeyError(f"Task '{task_name}' {action} unknown stream '{stream_name}'.")


def _validate_task_bindings(task_name: str, task: dict[str, Any]) -> None:
    all_binding_names = _binding_names(task)
    callable_binding_names = _callable_binding_names(task)
    _validate_unique_binding_names(task_name, all_binding_names)
    _validate_task_control(task_name, task)

    signature = inspect.signature(task["fn"])
    _validate_callable_accepts_bound_names(
        task_name,
        task["fn"],
        signature,
        callable_binding_names,
    )
    _validate_required_parameters_are_bound(
        task_name,
        signature,
        set(callable_binding_names),
    )


def _validate_task_control(task_name: str, task: dict[str, Any]) -> None:
    control_mode = task.get("control_mode")
    control_event = task.get("control_event")

    if control_mode is None and control_event is None:
        return

    if control_mode not in {"switchable", "toggleable"}:
        raise ValueError(
            f"Task '{task_name}' has unsupported control mode {control_mode!r}."
        )

    if not control_event:
        raise ValueError(
            f"Task '{task_name}' must declare an 'activate_on' event binding."
        )

    if control_event not in task["events"]:
        raise ValueError(
            f"Task '{task_name}' declares activate_on='{control_event}' but that "
            "name is not present in the task's event bindings."
        )


def _binding_names(task: dict[str, Any]) -> list[str]:
    return [
        *task["reads"].keys(),
        *task["writes"].keys(),
        *task["events"].keys(),
    ]


def _callable_binding_names(task: dict[str, Any]) -> list[str]:
    binding_names = _binding_names(task)
    control_event = task.get("control_event")
    control_mode = task.get("control_mode")
    if control_mode is None or control_event is None:
        return binding_names
    return [name for name in binding_names if name != control_event]


def _validate_unique_binding_names(task_name: str, binding_names: list[str]) -> None:
    duplicates = _duplicate_names(binding_names)
    if not duplicates:
        return

    raise ValueError(
        f"Task '{task_name}' reuses local binding names across reads/writes/events: "
        f"{duplicates}."
    )


def _duplicate_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for name in names:
        if name in seen:
            duplicates.add(name)
            continue
        seen.add(name)

    return sorted(duplicates)


def _validate_callable_accepts_bound_names(
    task_name: str,
    fn: Callable[..., Any],
    signature: inspect.Signature,
    binding_names: list[str],
) -> None:
    parameters = signature.parameters
    accepts_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )

    for binding_name in binding_names:
        parameter = parameters.get(binding_name)
        if parameter is None:
            if accepts_kwargs:
                continue
            raise ValueError(
                f"Task '{task_name}' binds local name '{binding_name}' but "
                f"callable '{getattr(fn, '__name__', type(fn).__name__)}' does not accept it."
            )

        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(
                f"Task '{task_name}' binding '{binding_name}' maps to a positional-only "
                "parameter, which cannot be supplied by keyword."
            )


def _validate_required_parameters_are_bound(
    task_name: str,
    signature: inspect.Signature,
    bound_names: set[str],
) -> None:
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if parameter.default is inspect._empty:
                raise ValueError(
                    f"Task '{task_name}' callable has required positional-only "
                    f"parameter '{parameter.name}', which cannot be bound by the pipeline."
                )
            continue

        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue

        if parameter.default is not inspect._empty:
            continue

        if parameter.name in bound_names:
            continue

        raise ValueError(
            f"Task '{task_name}' callable requires parameter "
            f"'{parameter.name}' but no read/write/event binding provides it."
        )
