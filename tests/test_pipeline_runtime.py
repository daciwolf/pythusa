"""Run with: pytest tests/test_pipeline_runtime.py -q"""

from __future__ import annotations

import time
import unittest

try:
    import numpy as np
    from pythusa import Pipeline
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    np = None
    Pipeline = None


_FRAME = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) if np is not None else None
_EXPECTED = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32) if np is not None else None


def _runtime_source(samples) -> None:
    samples.write(_FRAME)


def _runtime_scale(samples, scaled) -> None:
    while True:
        frame = samples.read()
        if frame is None:
            time.sleep(0.001)
            continue
        scaled.write((frame * 2.0).astype(np.float32, copy=False))
        return


def _runtime_sink(scaled, done) -> None:
    while True:
        frame = scaled.read()
        if frame is None:
            time.sleep(0.001)
            continue
        if np.array_equal(frame, _EXPECTED):
            done.signal()
        return


def _toggle_runtime_source(samples) -> None:
    next_value = getattr(_toggle_runtime_source, "_next_value", 0) + 1
    _toggle_runtime_source._next_value = next_value
    samples.write(np.array([next_value], dtype=np.int32))


def _toggle_runtime_sink(samples, done) -> None:
    seen = []
    while True:
        frame = samples.read()
        if frame is None:
            time.sleep(0.001)
            continue
        seen.append(int(frame[0]))
        if seen == [1, 2]:
            done.signal()
            return


def _look_runtime_sink(samples, done) -> None:
    while True:
        view = samples.look()
        if view is None:
            time.sleep(0.001)
            continue
        first = bytes(view)
        second = bytes(samples.look())
        samples.increment()
        self_check = samples.look()
        if first == _FRAME.tobytes() and second == _FRAME.tobytes() and self_check is None:
            done.signal()
        return


@unittest.skipIf(Pipeline is None or np is None, "pipeline runtime dependencies are required")
class PipelineRuntimeTests(unittest.TestCase):
    def test_pipeline_runtime_moves_data_end_to_end(self) -> None:
        pipe = Pipeline("runtime")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_stream("scaled", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_event("done")
            pipe.add_task("source", fn=_runtime_source, writes={"samples": "samples"})
            pipe.add_task(
                "scale",
                fn=_runtime_scale,
                reads={"samples": "samples"},
                writes={"scaled": "scaled"},
            )
            pipe.add_task(
                "sink",
                fn=_runtime_sink,
                reads={"scaled": "scaled"},
                events={"done": "done"},
            )

            pipe.start()

            self.assertTrue(pipe._manager._events["done"].wait(timeout=3.0))

            pipe.join(timeout=3.0)
            self.assertFalse(pipe._manager._processes["source"].is_alive())
            self.assertFalse(pipe._manager._processes["scale"].is_alive())
            self.assertFalse(pipe._manager._processes["sink"].is_alive())
        finally:
            pipe.close()

    def test_toggleable_task_processes_multiple_activations(self) -> None:
        pipe = Pipeline("toggle-runtime")

        try:
            pipe.add_stream("samples", shape=(1,), dtype=np.int32, cache_align=False)
            pipe.add_event("go")
            pipe.add_event("done")
            pipe.add_task.toggleable(
                "source",
                activate_on="go",
                fn=_toggle_runtime_source,
                writes={"samples": "samples"},
                events={"go": "go"},
            )
            pipe.add_task(
                "sink",
                fn=_toggle_runtime_sink,
                reads={"samples": "samples"},
                events={"done": "done"},
            )

            pipe.start()

            trigger = pipe._manager._events["go"]
            trigger.signal()
            trigger.signal()

            self.assertTrue(pipe._manager._events["done"].wait(timeout=3.0))
        finally:
            pipe.close()

    def test_stream_look_requires_manual_increment(self) -> None:
        pipe = Pipeline("look-runtime")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_event("done")
            pipe.add_task("source", fn=_runtime_source, writes={"samples": "samples"})
            pipe.add_task(
                "sink",
                fn=_look_runtime_sink,
                reads={"samples": "samples"},
                events={"done": "done"},
            )

            pipe.start()

            self.assertTrue(pipe._manager._events["done"].wait(timeout=3.0))
        finally:
            pipe.close()


if __name__ == "__main__":
    unittest.main()
