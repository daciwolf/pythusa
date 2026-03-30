from __future__ import annotations

import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np

import pythusa


FRAME = np.arange(8, dtype=np.float32)
EXPECTED = FRAME * 2.0


def source(samples) -> None:
    samples.write(FRAME)


def scale(samples, doubled) -> None:
    while True:
        frame = samples.read()
        if frame is None:
            time.sleep(0.001)
            continue
        doubled.write((frame * 2.0).astype(np.float32, copy=False))
        return


def sink(doubled, *, results) -> None:
    while True:
        frame = doubled.read()
        if frame is None:
            time.sleep(0.001)
            continue
        results.put(frame.tolist())
        return


def main() -> None:
    module_path = Path(pythusa.__file__).resolve()
    if "site-packages" not in str(module_path).lower():
        raise AssertionError(
            f"Smoke test imported pythusa from {module_path}, not an installed wheel."
        )

    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    sink_task = partial(sink, results=results)

    with pythusa.Pipeline("pkg-smoke") as pipe:
        pipe.add_stream("samples", shape=(8,), dtype=np.float32)
        pipe.add_stream("doubled", shape=(8,), dtype=np.float32)

        pipe.add_task("source", fn=source, writes={"samples": "samples"})
        pipe.add_task(
            "scale",
            fn=scale,
            reads={"samples": "samples"},
            writes={"doubled": "doubled"},
        )
        pipe.add_task("sink", fn=sink_task, reads={"doubled": "doubled"})

        pipe.run()

    result = np.asarray(results.get(timeout=5.0), dtype=np.float32)
    if not np.array_equal(result, EXPECTED):
        raise AssertionError(
            f"Installed-wheel smoke test failed: got {result}, expected {EXPECTED}."
        )

    print("package smoke test passed")


if __name__ == "__main__":
    main()
