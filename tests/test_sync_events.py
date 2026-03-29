"""Run with: pytest tests/test_sync_events.py -q"""

from __future__ import annotations

import multiprocessing as mp
import unittest

try:
    from pythusa._sync.events import WorkerEvent
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    WorkerEvent = None


def _signal_event_n_times(event: WorkerEvent, count: int) -> None:
    for _ in range(count):
        event.signal()


def _reset_event_n_times(event: WorkerEvent, count: int) -> None:
    for _ in range(count):
        event.reset()


@unittest.skipIf(WorkerEvent is None, "sync dependencies are required")
class WorkerEventTests(unittest.TestCase):
    def test_signal_opens_event_and_increments_pending(self) -> None:
        event = WorkerEvent("ready")

        event.signal()
        event.signal()

        self.assertTrue(event.is_open())
        self.assertEqual(event.pending, 2)

    def test_reset_consumes_one_pending_activation(self) -> None:
        event = WorkerEvent("ready")
        event.signal()
        event.signal()

        event.reset()

        self.assertTrue(event.is_open())
        self.assertEqual(event.pending, 1)

    def test_reset_closes_event_when_last_pending_activation_is_consumed(self) -> None:
        event = WorkerEvent("ready")
        event.signal()

        event.reset()

        self.assertFalse(event.is_open())
        self.assertEqual(event.pending, 0)

    def test_initial_state_starts_open_with_one_pending_activation(self) -> None:
        event = WorkerEvent("ready", initial_state=True)

        self.assertTrue(event.is_open())
        self.assertEqual(event.pending, 1)

        event.reset()

        self.assertFalse(event.is_open())
        self.assertEqual(event.pending, 0)

    def test_wait_times_out_while_closed(self) -> None:
        event = WorkerEvent("ready")

        self.assertFalse(event.wait(timeout=0.01))

    def test_wait_returns_immediately_when_open(self) -> None:
        event = WorkerEvent("ready")
        event.signal()

        self.assertTrue(event.wait(timeout=0.0))

    def test_reset_on_empty_event_is_a_noop(self) -> None:
        event = WorkerEvent("ready")

        event.reset()

        self.assertFalse(event.is_open())
        self.assertEqual(event.pending, 0)

    def test_repr_includes_open_closed_state_and_pending_count(self) -> None:
        event = WorkerEvent("ready")
        self.assertIn("[CLOSED]", repr(event))
        self.assertIn("pending=0", repr(event))

        event.signal()

        self.assertIn("[OPEN]", repr(event))
        self.assertIn("pending=1", repr(event))

    def test_signal_in_child_process_increments_pending_in_parent(self) -> None:
        event = WorkerEvent("ready")
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_signal_event_n_times, args=(event, 2))

        proc.start()
        proc.join(timeout=5.0)

        self.assertEqual(proc.exitcode, 0)
        self.assertTrue(event.is_open())
        self.assertEqual(event.pending, 2)

    def test_reset_in_child_process_consumes_pending_in_parent(self) -> None:
        event = WorkerEvent("ready")
        event.signal()
        event.signal()
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_reset_event_n_times, args=(event, 1))

        proc.start()
        proc.join(timeout=5.0)

        self.assertEqual(proc.exitcode, 0)
        self.assertTrue(event.is_open())
        self.assertEqual(event.pending, 1)


if __name__ == "__main__":
    unittest.main()
