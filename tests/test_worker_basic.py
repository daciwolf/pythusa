"""Run with: pytest tests/test_worker_basic.py -q"""

import signal
import unittest
from unittest.mock import patch

try:
    from pythusa import Worker
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    Worker = None


@unittest.skipIf(Worker is None, "worker dependencies are required")
class WorkerBasicTests(unittest.TestCase):
    def test_worker_invokes_fn_once_and_returns(self):
        calls: list[str] = []
        worker = Worker(lambda: calls.append("ran"))

        with patch("pythusa._workers.worker.signal.signal") as install_signal:
            worker()

        self.assertEqual(calls, ["ran"])
        self.assertEqual(install_signal.call_args.args[0], signal.SIGTERM)
        self.assertTrue(callable(install_signal.call_args.args[1]))


if __name__ == "__main__":
    unittest.main()
