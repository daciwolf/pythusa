"""Run with: pytest tests/test_bootstrap_basic.py -q"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pythusa._workers.bootstrap import TaskBootstrap


class TaskBootstrapBasicTests(unittest.TestCase):
    def test_bootstrap_installs_context_and_runs_task_once(self) -> None:
        calls: list[object] = []

        def make_ring(**kwargs):
            ring_name = kwargs["name"]

            class _Ring:
                def __enter__(self_nonlocal):
                    calls.append(f"enter:{ring_name}")
                    return self_nonlocal

                def __exit__(self_nonlocal, *_exc_info):
                    calls.append(f"exit:{ring_name}")
                    return False

            return _Ring()

        def install_context(reading_rings, writing_rings, events):
            calls.append(
                (
                    "install",
                    tuple(sorted(reading_rings)),
                    tuple(sorted(writing_rings)),
                    tuple(sorted(events)),
                )
            )

        class FakeWorker:
            def __init__(self, fn):
                self._fn = fn
                calls.append("worker_init")

            def __call__(self):
                calls.append("worker_call")
                self._fn()

        def task(value, *, scale):
            calls.append(("task", value, scale))

        bootstrap = TaskBootstrap(
            name="task",
            fn=task,
            reading_ring_kwargs={"input": {"name": "input"}},
            writing_ring_kwargs={"output": {"name": "output"}},
            events={"ready": object()},
            args=(3,),
            kwargs={"scale": 4},
        )

        with patch("pythusa._workers.bootstrap.SharedRingBuffer", side_effect=make_ring), patch(
            "pythusa._workers.bootstrap.context._install",
            side_effect=install_context,
        ), patch("pythusa._workers.bootstrap.Worker", FakeWorker):
            bootstrap()

        install_index = calls.index(("install", ("input",), ("output",), ("ready",)))
        worker_index = calls.index("worker_call")
        task_index = calls.index(("task", 3, 4))
        self.assertLess(calls.index("enter:input"), install_index)
        self.assertLess(calls.index("enter:output"), install_index)
        self.assertLess(install_index, worker_index)
        self.assertLess(worker_index, task_index)
        self.assertGreater(calls.index("exit:input"), task_index)
        self.assertGreater(calls.index("exit:output"), task_index)

    def test_bootstrap_closes_rings_when_task_raises(self) -> None:
        calls: list[str] = []

        def make_ring(**kwargs):
            ring_name = kwargs["name"]

            class _Ring:
                def __enter__(self_nonlocal):
                    calls.append(f"enter:{ring_name}")
                    return self_nonlocal

                def __exit__(self_nonlocal, *_exc_info):
                    calls.append(f"exit:{ring_name}")
                    return False

            return _Ring()

        class FakeWorker:
            def __init__(self, fn):
                self._fn = fn

            def __call__(self):
                self._fn()

        def task() -> None:
            raise RuntimeError("boom")

        bootstrap = TaskBootstrap(
            name="task",
            fn=task,
            reading_ring_kwargs={"input": {"name": "input"}},
            writing_ring_kwargs={"output": {"name": "output"}},
        )

        with patch("pythusa._workers.bootstrap.SharedRingBuffer", side_effect=make_ring), patch(
            "pythusa._workers.bootstrap.context._install"
        ), patch("pythusa._workers.bootstrap.Worker", FakeWorker):
            with self.assertRaises(RuntimeError):
                bootstrap()

        self.assertIn("exit:input", calls)
        self.assertIn("exit:output", calls)


if __name__ == "__main__":
    unittest.main()
