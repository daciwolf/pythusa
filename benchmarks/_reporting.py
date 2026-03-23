from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - benchmark environments include numpy.
    np = None


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def build_payload(
    *,
    benchmark: str,
    config: dict[str, Any],
    results: Any,
    notes: list[str] | tuple[str, ...] = (),
    label: str | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _to_jsonable(config),
        "results": _to_jsonable(results),
        "notes": list(notes),
    }
    if label:
        payload["label"] = label
    if summary is not None:
        payload["summary"] = _to_jsonable(summary)
    return payload


def emit_payload(
    payload: dict[str, Any],
    *,
    json_stdout: bool,
    json_out: str | None,
) -> None:
    text = json.dumps(_to_jsonable(payload), indent=2, sort_keys=True)
    if json_out is not None:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    if json_stdout:
        print(text)
