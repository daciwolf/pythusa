from __future__ import annotations

import importlib
from pathlib import Path
import tomllib
from typing import Any, Callable

import numpy as np

_PIPELINE_TOML_VERSION = 1
_IMPORTABLE_CALLABLE_ERROR = (
    "Pipeline.save() requires task functions to be importable top-level callables."
)


def render_pipeline_toml(
    *,
    name: str,
    streams: dict[str, dict[str, Any]],
    events: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
) -> str:
    lines = [
        f"format_version = {_PIPELINE_TOML_VERSION}",
        f'name = {_toml_string(name)}',
        "",
    ]

    for stream in _sorted_declarations(streams):
        _append_stream_table(lines, stream)
    for event in _sorted_declarations(events):
        _append_event_table(lines, event)
    for task in _sorted_declarations(tasks):
        _append_task_table(lines, task)

    return "\n".join(lines).rstrip() + "\n"


def read_pipeline_toml(source: Path) -> dict[str, Any]:
    data = tomllib.loads(source.read_text(encoding="utf-8"))

    format_version = data.get("format_version", 1)
    if format_version != _PIPELINE_TOML_VERSION:
        raise ValueError(
            f"Unsupported pipeline TOML format_version={format_version}. "
            f"Expected {_PIPELINE_TOML_VERSION}."
        )

    if "name" not in data:
        raise ValueError("Pipeline TOML is missing required top-level 'name'.")

    return data


def resolve_callable(module_name: str, qualname: str) -> Callable[..., Any]:
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def require_keys(section: dict[str, Any], section_name: str, *keys: str) -> None:
    missing = [key for key in keys if key not in section]
    if not missing:
        return

    joined = ", ".join(f"'{key}'" for key in keys)
    raise ValueError(f"Each saved {section_name} must include {joined}.")


def callable_reference(fn: Callable[..., Any]) -> tuple[str, str]:
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)

    if not module_name or not qualname or "<locals>" in qualname:
        raise ValueError(_IMPORTABLE_CALLABLE_ERROR)

    try:
        resolved = resolve_callable(module_name, qualname)
    except (ImportError, AttributeError) as exc:
        raise ValueError(_IMPORTABLE_CALLABLE_ERROR) from exc

    if resolved is not fn:
        raise ValueError(_IMPORTABLE_CALLABLE_ERROR)

    return module_name, qualname


def _sorted_declarations(
    declarations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [declarations[name] for name in sorted(declarations)]


def _append_stream_table(lines: list[str], stream: dict[str, Any]) -> None:
    lines.append("[[streams]]")
    lines.append(f'name = {_toml_string(stream["name"])}')
    lines.append(f"shape = {_toml_int_array(stream['shape'])}")
    lines.append(f'dtype = {_toml_string(np.dtype(stream["dtype"]).str)}')
    lines.append(f"cache_align = {_toml_bool(stream['cache_align'])}")
    if stream["description"] is not None:
        lines.append(f'description = {_toml_string(stream["description"])}')
    lines.append("")


def _append_event_table(lines: list[str], event: dict[str, Any]) -> None:
    lines.append("[[events]]")
    lines.append(f'name = {_toml_string(event["name"])}')
    lines.append(f"initial_state = {_toml_bool(event['initial_state'])}")
    if event["description"] is not None:
        lines.append(f'description = {_toml_string(event["description"])}')
    lines.append("")


def _append_task_table(lines: list[str], task: dict[str, Any]) -> None:
    module_name, qualname = callable_reference(task["fn"])

    lines.append("[[tasks]]")
    lines.append(f'name = {_toml_string(task["name"])}')
    lines.append(f'function_module = {_toml_string(module_name)}')
    lines.append(f'function_qualname = {_toml_string(qualname)}')
    if task["description"] is not None:
        lines.append(f'description = {_toml_string(task["description"])}')
    _append_toml_mapping(lines, "tasks.reads", task["reads"])
    _append_toml_mapping(lines, "tasks.writes", task["writes"])
    _append_toml_mapping(lines, "tasks.events", task["events"])
    lines.append("")


def _toml_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_int_array(values: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _append_toml_mapping(
    lines: list[str],
    header: str,
    mapping: dict[str, str],
) -> None:
    if not mapping:
        return

    lines.append(f"[{header}]")
    for key in sorted(mapping):
        lines.append(f"{key} = {_toml_string(mapping[key])}")
