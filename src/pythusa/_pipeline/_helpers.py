from __future__ import annotations

import inspect
from graphlib import CycleError, TopologicalSorter
from typing import Any, Callable

from .._buffers.ring import RingSpec
from .._core.context import get_event, get_reader, get_writer
from .._processing.numpy import bytes_for_shape
from .._utils import align_size
from .._workers.worker import TaskSpec

_DEFAULT_RING_DEPTH = 32
_DEFAULT_CACHE_SIZE = 64


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
        ring_size = align_size(ring_size, _DEFAULT_CACHE_SIZE)

    return RingSpec(
        name=stream["name"],
        size=ring_size,
        num_readers=reader_count,
        cache_align=stream["cache_align"],
    )


def task_spec_for_name(task_name: str, task: dict[str, Any]) -> TaskSpec:
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
        ),
    )


def _invoke_task_with_bindings(
    fn: Callable[..., Any],
    reads: dict[str, str],
    writes: dict[str, str],
    events: dict[str, str],
) -> Any:
    kwargs: dict[str, Any] = {}

    for local_name, stream_name in reads.items():
        kwargs[local_name] = get_reader(stream_name)

    for local_name, stream_name in writes.items():
        kwargs[local_name] = get_writer(stream_name)

    for local_name, event_name in events.items():
        kwargs[local_name] = get_event(event_name)

    return fn(**kwargs)


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
    binding_names = _binding_names(task)
    _validate_unique_binding_names(task_name, binding_names)

    signature = inspect.signature(task["fn"])
    _validate_callable_accepts_bound_names(task_name, task["fn"], signature, binding_names)
    _validate_required_parameters_are_bound(task_name, signature, set(binding_names))


def _binding_names(task: dict[str, Any]) -> list[str]:
    return [
        *task["reads"].keys(),
        *task["writes"].keys(),
        *task["events"].keys(),
    ]


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
