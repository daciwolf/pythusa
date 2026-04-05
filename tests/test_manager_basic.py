"""Run with: pytest tests/test_manager_basic.py -q"""

import unittest
from unittest.mock import MagicMock, patch

try:
    import pythusa
    from pythusa import Manager, ProcessMetrics, RingSpec, TaskSpec
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    pythusa = None
    Manager = None
    ProcessMetrics = None
    RingSpec = None
    TaskSpec = None


@unittest.skipIf(Manager is None, "manager dependencies are required")
class ManagerBasicTests(unittest.TestCase):
    def test_manager_module_exports_expected_symbols(self):
        self.assertTrue(hasattr(pythusa, "__all__"))
        self.assertIn("Manager", pythusa.__all__)
        self.assertIn("SharedRingBuffer", pythusa.__all__)
        self.assertIn("Worker", pythusa.__all__)
        self.assertIn("TaskSpec", pythusa.__all__)
        self.assertIn("EventSpec", pythusa.__all__)
        self.assertIn("ProcessMetrics", pythusa.__all__)

    def test_create_ring_registers_spec_live_ring_and_counter(self):
        mgr = Manager()

        try:
            spec = RingSpec(
                name="rb",
                size=32,
                num_readers=2,
                min_reader_pos_refresh_interval=17,
                min_reader_pos_refresh_s=0.125,
            )
            returned = mgr.create_ring(spec)

            self.assertIs(returned, mgr)
            self.assertIs(mgr._ring_specs["rb"], spec)
            self.assertIn("rb", mgr._rings)
            self.assertEqual(mgr._ring_reader_counters["rb"], 0)
            self.assertEqual(mgr._rings["rb"]._min_reader_pos_refresh_interval, 17)
            self.assertEqual(mgr._rings["rb"]._min_reader_pos_refresh_s, 0.125)
        finally:
            mgr.close()

    def test_task_bootstrap_propagates_ring_refresh_config(self):
        mgr = Manager()

        try:
            mgr.create_ring(
                RingSpec(
                    name="input",
                    size=32,
                    num_readers=1,
                    min_reader_pos_refresh_interval=7,
                    min_reader_pos_refresh_s=0.015,
                )
            )
            mgr.create_ring(
                RingSpec(
                    name="output",
                    size=32,
                    num_readers=1,
                    min_reader_pos_refresh_interval=13,
                    min_reader_pos_refresh_s=0.2,
                )
            )
            mgr.create_task(
                TaskSpec(
                    name="task",
                    fn=lambda: None,
                    reading_rings=("input",),
                    writing_rings=("output",),
                )
            )

            bootstrap = mgr._task_bootstrap("task")

            self.assertEqual(
                bootstrap.reading_ring_kwargs["input"]["min_reader_pos_refresh_interval"],
                7,
            )
            self.assertEqual(
                bootstrap.reading_ring_kwargs["input"]["min_reader_pos_refresh_s"],
                0.015,
            )
            self.assertEqual(
                bootstrap.writing_ring_kwargs["output"]["min_reader_pos_refresh_interval"],
                13,
            )
            self.assertEqual(
                bootstrap.writing_ring_kwargs["output"]["min_reader_pos_refresh_s"],
                0.2,
            )
        finally:
            mgr.close()

    def test_collect_ring_pressures_ignores_failures(self):
        mgr = Manager()
        good_ring = MagicMock()
        good_ring.calculate_pressure.return_value = 73
        bad_ring = MagicMock()
        bad_ring.calculate_pressure.side_effect = RuntimeError("boom")

        pressures = mgr._collect_ring_pressures({"good": good_ring, "bad": bad_ring})

        self.assertEqual(pressures, {"good": 73})

    def test_sample_process_stores_latest_metrics_snapshot(self):
        mgr = Manager()
        mgr.create_task(
            TaskSpec(
                name="task",
                fn=lambda: None,
                reading_rings=("input",),
                writing_rings=("output",),
            )
        )
        proc = MagicMock()
        proc.pid = 4321
        ring_pressures = {"input": 15, "output": 88, "other": 99}
        metrics = {}

        ps = MagicMock()
        ps.cpu_percent.return_value = 12.5
        ps.memory_info.return_value = MagicMock(rss=9 * 1024 * 1024)
        ps.nice.return_value = 5

        with patch("pythusa._workers.manager.psutil.Process", return_value=ps):
            task_pressures = mgr._sample_process(
                "task",
                proc,
                ring_pressures,
                mgr._task_specs,
                metrics,
                123.456,
            )

        self.assertEqual(task_pressures, {"input": 15, "output": 88})
        self.assertIn("task", metrics)
        self.assertEqual(
            metrics["task"],
            ProcessMetrics(
                name="task",
                pid=4321,
                cpu_percent=12.5,
                memory_rss_mb=9.0,
                nice=5,
                ring_pressure={"input": 15, "output": 88},
                sampled_at=123.456,
            ),
        )

    def test_adjust_process_nice_uses_worst_ring_pressure(self):
        mgr = Manager()
        ps = MagicMock()

        with patch("pythusa._workers.manager.psutil.Process", return_value=ps):
            mgr._adjust_process_nice(MagicMock(pid=1), {"a": 81, "b": 40})
            mgr._adjust_process_nice(MagicMock(pid=2), {"a": 10})
            mgr._adjust_process_nice(MagicMock(pid=3), {})

        self.assertEqual(ps.nice.call_args_list[0].args, (-10,))
        self.assertEqual(ps.nice.call_args_list[1].args, (10,))
        self.assertEqual(len(ps.nice.call_args_list), 2)

    def test_start_monitor_starts_named_daemon_thread(self):
        mgr = Manager()
        fake_thread = MagicMock()

        with patch("pythusa._workers.manager.threading.Thread", return_value=fake_thread) as thread_cls:
            mgr.start_monitor(interval_s=0.25)

        kwargs = thread_cls.call_args.kwargs
        self.assertTrue(callable(kwargs["target"]))
        self.assertTrue(kwargs["daemon"])
        self.assertEqual(kwargs["name"], "pythusa_monitor")
        fake_thread.start.assert_called_once_with()

    def test_get_metrics_returns_latest_snapshot_or_none(self):
        mgr = Manager()
        self.assertIsNone(mgr.get_metrics("missing"))

        snap = ProcessMetrics(
            name="task",
            pid=1,
            cpu_percent=1.0,
            memory_rss_mb=2.0,
            nice=0,
            ring_pressure={"rb": 50},
            sampled_at=3.0,
        )
        mgr._metrics["task"] = snap

        self.assertIs(mgr.get_metrics("task"), snap)

    def test_task_bootstrap_preserves_args_and_kwargs(self):
        mgr = Manager()

        def task(value, *, scale=1):
            return value * scale

        mgr.create_task(
            TaskSpec(
                name="task",
                fn=task,
                args=(3,),
                kwargs={"scale": 4},
            )
        )

        bootstrap = mgr._task_bootstrap("task")

        self.assertEqual(bootstrap.args, (3,))
        self.assertEqual(bootstrap.kwargs, {"scale": 4})


if __name__ == "__main__":
    unittest.main()
