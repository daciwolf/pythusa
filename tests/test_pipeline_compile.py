import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    from pythusa import Pipeline, ProcessMetrics
    from pythusa._pipeline._helpers import _invoke_task_with_bindings
    from pythusa._pipeline._stream_io import make_reader_binding, make_writer_binding
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    np = None
    Pipeline = None
    ProcessMetrics = None
    _invoke_task_with_bindings = None
    make_reader_binding = None


@unittest.skipIf(Pipeline is None, "pipeline dependencies are required")
class PipelineCompileTests(unittest.TestCase):
    def test_compile_registers_specs_and_task_orders(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_event("shutdown")
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task(
                "consumer",
                fn=_consumer_with_shutdown,
                reads={"samples": "samples"},
                events={"shutdown": "shutdown"},
            )

            pipe.compile()

            self.assertTrue(pipe._compiled)
            self.assertEqual(pipe._task_order, ("acquire", "consumer"))
            self.assertEqual(pipe._task_start_order, ("consumer", "acquire"))
            self.assertIn("shutdown", pipe._manager._event_specs)
            self.assertIn("samples", pipe._manager._ring_specs)
            self.assertEqual(pipe._manager._ring_specs["samples"].size, 512)
            self.assertEqual(pipe._manager._ring_specs["samples"].num_readers, 1)
            self.assertEqual(
                pipe._manager._task_specs["acquire"].writing_rings,
                ("samples",),
            )
            self.assertEqual(
                pipe._manager._task_specs["consumer"].reading_rings,
                ("samples",),
            )
            self.assertEqual(
                pipe._manager._task_specs["consumer"].events,
                ("shutdown",),
            )
        finally:
            pipe._manager.close()

    def test_compile_counts_fanout_readers_on_one_stream(self):
        pipe = Pipeline("fanout")

        try:
            pipe.add_stream("samples", shape=(8,), dtype=np.float32, cache_align=False)
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task("worker_1", fn=_sample_reader, reads={"samples": "samples"})
            pipe.add_task("worker_2", fn=_sample_reader, reads={"samples": "samples"})

            pipe.compile()

            self.assertEqual(pipe._manager._ring_specs["samples"].num_readers, 2)
            self.assertEqual(pipe._task_order[0], "acquire")
            self.assertCountEqual(pipe._task_order[1:], ("worker_1", "worker_2"))
        finally:
            pipe._manager.close()

    def test_compile_rejects_task_cycles(self):
        pipe = Pipeline("cycle")

        pipe.add_stream("a", shape=(1,), dtype=np.float32, cache_align=False)
        pipe.add_stream("b", shape=(1,), dtype=np.float32, cache_align=False)
        pipe.add_task(
            "task_a",
            fn=_cycle_task_a,
            reads={"b_in": "b"},
            writes={"a_out": "a"},
        )
        pipe.add_task(
            "task_b",
            fn=_cycle_task_b,
            reads={"a_in": "a"},
            writes={"b_out": "b"},
        )

        with self.assertRaisesRegex(ValueError, "cycle"):
            pipe.compile()

    def test_compile_rejects_unknown_read_stream(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
        pipe.add_task("sink", fn=_sample_reader, reads={"samples": "missing"})

        with self.assertRaisesRegex(KeyError, "reads unknown stream"):
            pipe.compile()

    def test_compile_rejects_unknown_write_stream(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task("acquire", fn=_source_task, writes={"samples": "missing"})
        pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})

        with self.assertRaisesRegex(KeyError, "writes unknown stream"):
            pipe.compile()

    def test_compile_rejects_unknown_event(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task(
            "acquire",
            fn=_source_with_shutdown,
            writes={"samples": "samples"},
            events={"shutdown": "missing"},
        )
        pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})

        with self.assertRaisesRegex(KeyError, "unknown event"):
            pipe.compile()

    def test_compile_warns_when_event_is_shared_across_more_than_two_tasks(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32)
            pipe.add_event("shutdown")
            pipe.add_task(
                "acquire",
                fn=_source_with_shutdown,
                writes={"samples": "samples"},
                events={"shutdown": "shutdown"},
            )
            pipe.add_task(
                "worker_a",
                fn=_sample_reader_with_shutdown,
                reads={"samples": "samples"},
                events={"shutdown": "shutdown"},
            )
            pipe.add_task(
                "worker_b",
                fn=_sample_reader_with_shutdown,
                reads={"samples": "samples"},
                events={"shutdown": "shutdown"},
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipe.compile()

            self.assertTrue(
                any("one producer/one consumer" in str(w.message) for w in caught)
            )
        finally:
            pipe._manager.close()

    def test_compile_rejects_stream_without_writer(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})

        with self.assertRaisesRegex(ValueError, "does not have a writer"):
            pipe.compile()

    def test_compile_rejects_stream_without_reader(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})

        with self.assertRaisesRegex(ValueError, "does not have any readers"):
            pipe.compile()

    def test_compile_rejects_multiple_writers_for_stream(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task("acquire_a", fn=_source_task, writes={"samples": "samples"})
        pipe.add_task("acquire_b", fn=_source_task, writes={"samples": "samples"})
        pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})

        with self.assertRaisesRegex(ValueError, "multiple writers"):
            pipe.compile()

    def test_compile_registers_task_binding_adapter(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("raw_adc", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_stream("spectra", shape=(4,), dtype=np.complex64, cache_align=False)
            pipe.add_event("halt")
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "raw_adc"})
            pipe.add_task(
                "worker",
                fn=_worker_task,
                reads={"samples": "raw_adc"},
                writes={"fft": "spectra"},
                events={"shutdown": "halt"},
            )
            pipe.add_task("sink", fn=_sink_task, reads={"fft": "spectra"})

            pipe.compile()

            task_spec = pipe._manager._task_specs["worker"]
            self.assertIs(task_spec.fn, _invoke_task_with_bindings)
            self.assertIs(task_spec.args[0], _worker_task)
            self.assertEqual(task_spec.args[1], {"samples": "raw_adc"})
            self.assertEqual(task_spec.args[2], {"fft": "spectra"})
            self.assertEqual(task_spec.args[3], {"shutdown": "halt"})
            self.assertIsNone(task_spec.args[4])
            self.assertIsNone(task_spec.args[5])
            self.assertEqual(
                task_spec.args[6],
                {"samples": {"name": "raw_adc", "shape": (4,), "dtype": np.float32}},
            )
            self.assertEqual(
                task_spec.args[7],
                {"fft": {"name": "spectra", "shape": (4,), "dtype": np.complex64}},
            )
            self.assertEqual(task_spec.reading_rings, ("raw_adc",))
            self.assertEqual(task_spec.writing_rings, ("spectra",))
            self.assertEqual(task_spec.events, ("halt",))
        finally:
            pipe._manager.close()

    def test_add_stream_rejects_duplicate_names(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "already registered"):
            pipe.add_stream("samples", shape=(4,), dtype=np.float32)

    def test_add_stream_stores_configured_frame_count(self):
        pipe = Pipeline("radar")

        pipe.add_stream("samples", shape=(4,), dtype=np.float32, frames=48)

        self.assertEqual(pipe._streams["samples"]["frames"], 48)

    def test_add_stream_rejects_non_integer_frame_count(self):
        pipe = Pipeline("radar")

        with self.assertRaisesRegex(TypeError, "frames must be an integer"):
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, frames=3.5)

    def test_add_stream_rejects_non_positive_frame_count(self):
        pipe = Pipeline("radar")

        with self.assertRaisesRegex(ValueError, "frames must be >= 1"):
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, frames=0)

    def test_add_task_rejects_duplicate_names(self):
        pipe = Pipeline("radar")
        pipe.add_task("acquire", fn=_task_fn)

        with self.assertRaisesRegex(ValueError, "already registered"):
            pipe.add_task("acquire", fn=_task_fn)

    def test_add_task_supports_decorator_registration(self):
        pipe = Pipeline("radar")

        @pipe.add_task("acquire", writes={"samples": "samples"})
        def acquire(samples) -> None:
            return None

        self.assertIs(pipe._tasks["acquire"]["fn"], acquire)
        self.assertEqual(pipe._tasks["acquire"]["writes"], {"samples": "samples"})
        self.assertIsNone(pipe._tasks["acquire"]["control_mode"])
        self.assertIsNone(pipe._tasks["acquire"]["control_event"])

    def test_add_task_toggleable_registers_control_metadata(self):
        pipe = Pipeline("radar")

        returned = pipe.add_task.toggleable(
            "acquire",
            activate_on="shutdown",
            fn=_source_task,
            writes={"samples": "samples"},
            events={"shutdown": "shutdown"},
        )

        self.assertIs(returned, pipe)
        self.assertEqual(pipe._tasks["acquire"]["control_mode"], "toggleable")
        self.assertEqual(pipe._tasks["acquire"]["control_event"], "shutdown")

    def test_add_task_switchable_requires_activate_on_event_binding(self):
        pipe = Pipeline("radar")

        with self.assertRaisesRegex(ValueError, "activate_on='shutdown'"):
            pipe.add_task.switchable(
                "worker",
                activate_on="shutdown",
                fn=_sample_reader,
                reads={"samples": "samples"},
                events={"halt": "shutdown"},
            )

    def test_add_event_rejects_duplicate_names(self):
        pipe = Pipeline("radar")
        pipe.add_event("shutdown")

        with self.assertRaisesRegex(ValueError, "already registered"):
            pipe.add_event("shutdown")

    def test_compile_rejects_second_compile(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})
            pipe.compile()

            with self.assertRaisesRegex(RuntimeError, "already been compiled"):
                pipe.compile()
        finally:
            pipe._manager.close()

    def test_start_uses_reverse_topological_order(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task("consumer", fn=_sample_reader, reads={"samples": "samples"})
            pipe.compile()
            pipe._manager.start = MagicMock()

            pipe.start()

            self.assertTrue(pipe._started)
            self.assertEqual(
                [call.args[0] for call in pipe._manager.start.call_args_list],
                ["consumer", "acquire"],
            )
        finally:
            pipe._manager.close()

    def test_start_auto_compiles_when_needed(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})
            pipe._manager.start = MagicMock()

            pipe.start()

            self.assertTrue(pipe._compiled)
            self.assertTrue(pipe._started)
            self.assertEqual(
                [call.args[0] for call in pipe._manager.start.call_args_list],
                ["sink", "acquire"],
            )
        finally:
            pipe._manager.close()

    def test_start_rejects_second_start(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, cache_align=False)
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})
            pipe._manager.start = MagicMock()

            pipe.start()

            with self.assertRaisesRegex(RuntimeError, "already been started"):
                pipe.start()
        finally:
            pipe._manager.close()

    def test_stop_and_join_delegate_to_manager(self):
        pipe = Pipeline("radar")
        pipe._started = True
        pipe._manager.stop_all = MagicMock()
        pipe._manager.join_all = MagicMock()

        pipe.stop()
        pipe.join(timeout=1.25)

        pipe._manager.stop_all.assert_called_once_with()
        pipe._manager.join_all.assert_called_once_with(timeout=1.25)

    def test_stop_and_join_are_noops_before_start(self):
        pipe = Pipeline("radar")
        pipe._manager.stop_all = MagicMock()
        pipe._manager.join_all = MagicMock()

        pipe.stop()
        pipe.join(timeout=2.0)

        pipe._manager.stop_all.assert_not_called()
        pipe._manager.join_all.assert_not_called()

    def test_close_stops_joins_and_closes_manager(self):
        pipe = Pipeline("radar")
        pipe._started = True
        pipe._manager.stop_all = MagicMock()
        pipe._manager.join_all = MagicMock()
        pipe._manager.close = MagicMock()

        pipe.close()

        self.assertTrue(pipe._closed)
        self.assertFalse(pipe._started)
        pipe._manager.stop_all.assert_called_once_with()
        pipe._manager.join_all.assert_called_once_with(timeout=None)
        pipe._manager.close.assert_called_once_with()

    def test_close_is_idempotent(self):
        pipe = Pipeline("radar")
        pipe._manager.close = MagicMock()

        pipe.close()
        pipe.close()

        pipe._manager.close.assert_called_once_with()

    def test_pipeline_context_manager_closes_on_exit(self):
        pipe = Pipeline("radar")
        pipe._manager.close = MagicMock()

        with pipe as active:
            self.assertIs(active, pipe)

        self.assertTrue(pipe._closed)
        pipe._manager.close.assert_called_once_with()

    def test_add_stream_rejects_closed_pipeline(self):
        pipe = Pipeline("radar")
        pipe.close()

        with self.assertRaisesRegex(RuntimeError, "Pipeline is closed"):
            pipe.add_stream("samples", shape=(4,), dtype=np.float32)

    def test_run_calls_start_then_join(self):
        pipe = Pipeline("radar")
        pipe.start = MagicMock()
        pipe.join = MagicMock()

        pipe.run()

        pipe.start.assert_called_once_with()
        pipe.join.assert_called_once_with()

    def test_start_monitor_delegates_to_manager_and_returns_pipeline(self):
        pipe = Pipeline("radar")
        pipe._manager.start_monitor = MagicMock()

        returned = pipe.start_monitor(interval_s=0.25)

        self.assertIs(returned, pipe)
        pipe._manager.start_monitor.assert_called_once_with(interval_s=0.25)

    def test_metrics_returns_one_snapshot_for_named_task(self):
        pipe = Pipeline("radar")
        pipe.add_task("worker", fn=_task_fn)
        snap = ProcessMetrics(
            name="worker",
            pid=1,
            cpu_percent=2.0,
            memory_rss_mb=3.0,
            nice=0,
            ring_pressure={"samples": 40},
            sampled_at=4.0,
        )
        pipe._manager._metrics["worker"] = snap

        self.assertIs(pipe.metrics("worker"), snap)

    def test_metrics_returns_snapshot_mapping_for_all_tasks(self):
        pipe = Pipeline("radar")
        pipe.add_task("worker_a", fn=_task_fn)
        pipe.add_task("worker_b", fn=_task_fn)
        snap = ProcessMetrics(
            name="worker_a",
            pid=1,
            cpu_percent=2.0,
            memory_rss_mb=3.0,
            nice=0,
            ring_pressure={},
            sampled_at=4.0,
        )
        pipe._manager._metrics["worker_a"] = snap

        self.assertEqual(
            pipe.metrics(),
            {"worker_a": snap, "worker_b": None},
        )

    def test_metrics_rejects_unknown_task_name(self):
        pipe = Pipeline("radar")

        with self.assertRaisesRegex(KeyError, "not registered"):
            pipe.metrics("missing")

    def test_save_and_reconstruct_round_trip_pipeline_declaration(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32, frames=12, cache_align=False)
        pipe.add_event("shutdown", initial_state=True)
        pipe.add_task(
            "acquire",
            fn=_task_fn,
            writes={"samples": "samples"},
            events={"shutdown": "shutdown"},
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            saved = pipe.save(path)
            text = saved.read_text(encoding="utf-8")
            restored = Pipeline.reconstruct(saved)

        self.assertIn("format_version = 1", text)
        self.assertIn('name = "radar"', text)
        self.assertIn("frames = 12", text)
        self.assertIn(f'function_module = "{_task_fn.__module__}"', text)
        self.assertIn('function_qualname = "_task_fn"', text)
        self.assertEqual(restored.name, "radar")
        self.assertEqual(restored._streams["samples"]["shape"], (4,))
        self.assertEqual(restored._streams["samples"]["dtype"], np.dtype(np.float32))
        self.assertEqual(restored._streams["samples"]["frames"], 12)
        self.assertFalse(restored._streams["samples"]["cache_align"])
        self.assertTrue(restored._events["shutdown"]["initial_state"])
        self.assertIs(restored._tasks["acquire"]["fn"], _task_fn)
        self.assertEqual(
            restored._tasks["acquire"]["writes"],
            {"samples": "samples"},
        )
        self.assertEqual(
            restored._tasks["acquire"]["events"],
            {"shutdown": "shutdown"},
        )

    def test_stream_reader_look_returns_memoryview_without_advancing(self):
        reader = _FakeRawReader(np.array([1, 2, 3, 4], dtype=np.int16))
        stream = make_reader_binding(
            reader,
            name="samples",
            shape=(4,),
            dtype=np.int16,
        )

        view = stream.look()

        self.assertIsInstance(view, memoryview)
        self.assertEqual(view.tobytes(), b"\x01\x00\x02\x00\x03\x00\x04\x00")
        self.assertEqual(reader.calls, ["expose_reader_mem_view:8"])

        stream.increment()

        self.assertEqual(reader.calls, ["expose_reader_mem_view:8", "inc_reader_pos:8"])

    def test_stream_writer_look_returns_memoryview_without_advancing(self):
        writer = _FakeRawWriter()
        stream = make_writer_binding(
            writer,
            name="samples",
            shape=(4,),
            dtype=np.int16,
        )

        view = stream.look()

        self.assertIsInstance(view, memoryview)
        view.cast("h")[:] = np.array([1, 2, 3, 4], dtype=np.int16)
        self.assertEqual(writer.calls, ["expose_writer_mem_view:8"])

        stream.increment()

        self.assertEqual(writer.calls, ["expose_writer_mem_view:8", "inc_writer_pos:8"])

    def test_save_creates_parent_directories(self):
        pipe = Pipeline("radar")
        pipe.add_task("idle", fn=_task_fn)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "configs" / "pipeline.toml"
            saved = pipe.save(path)

            self.assertTrue(saved.exists())
            self.assertEqual(saved, path)

    def test_save_rejects_non_importable_callable(self):
        pipe = Pipeline("radar")
        pipe.add_task("idle", fn=lambda: None)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            with self.assertRaisesRegex(ValueError, "importable top-level callables"):
                pipe.save(path)

    def test_save_and_reconstruct_preserve_task_control_metadata(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_event("shutdown")
        pipe.add_task.toggleable(
            "acquire",
            activate_on="shutdown",
            fn=_source_task,
            writes={"samples": "samples"},
            events={"shutdown": "shutdown"},
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            text = pipe.save(path).read_text(encoding="utf-8")
            restored = Pipeline.reconstruct(path)

        self.assertIn('control_mode = "toggleable"', text)
        self.assertIn('control_event = "shutdown"', text)
        self.assertEqual(restored._tasks["acquire"]["control_mode"], "toggleable")
        self.assertEqual(restored._tasks["acquire"]["control_event"], "shutdown")

    def test_save_orders_sections_deterministically(self):
        pipe = Pipeline("radar")
        pipe.add_stream("z_stream", shape=(1,), dtype=np.uint8)
        pipe.add_stream("a_stream", shape=(1,), dtype=np.uint8)
        pipe.add_event("z_event")
        pipe.add_event("a_event")
        pipe.add_task("z_task", fn=_task_fn)
        pipe.add_task("a_task", fn=_task_fn)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            text = pipe.save(path).read_text(encoding="utf-8")

        self.assertLess(text.index('name = "a_stream"'), text.index('name = "z_stream"'))
        self.assertLess(text.index('name = "a_event"'), text.index('name = "z_event"'))
        self.assertLess(text.index('name = "a_task"'), text.index('name = "z_task"'))

    def test_repr_reports_counts_and_state(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task("acquire", fn=_task_fn, writes={"samples": "samples"})
        pipe.add_event("shutdown")

        text = repr(pipe)

        self.assertIn("Pipeline(", text)
        self.assertIn("streams=1", text)
        self.assertIn("tasks=1", text)
        self.assertIn("events=1", text)
        self.assertIn("compiled=False", text)
        self.assertIn("started=False", text)

    def test_reconstruct_rejects_unknown_format_version(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            path.write_text(
                'format_version = 999\nname = "radar"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "format_version=999"):
                Pipeline.reconstruct(path)

    def test_reconstruct_rejects_missing_top_level_name(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            path.write_text("format_version = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "top-level 'name'"):
                Pipeline.reconstruct(path)

    def test_reconstruct_rejects_stream_missing_required_fields(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            path.write_text(
                'format_version = 1\nname = "radar"\n[[streams]]\nname = "samples"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "saved stream must include"):
                Pipeline.reconstruct(path)

    def test_reconstruct_defaults_missing_stream_frame_count_to_32(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            path.write_text(
                (
                    'format_version = 1\n'
                    'name = "radar"\n'
                    '[[streams]]\n'
                    'name = "samples"\n'
                    'shape = [4]\n'
                    'dtype = "<f4"\n'
                    "cache_align = false\n"
                ),
                encoding="utf-8",
            )

            restored = Pipeline.reconstruct(path)

        self.assertEqual(restored._streams["samples"]["frames"], 32)

    def test_reconstruct_rejects_event_missing_name(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            path.write_text(
                'format_version = 1\nname = "radar"\n[[events]]\ninitial_state = true\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "saved event must include 'name'"):
                Pipeline.reconstruct(path)

    def test_reconstruct_rejects_task_missing_required_fields(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.toml"
            path.write_text(
                'format_version = 1\nname = "radar"\n[[tasks]]\nname = "worker"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "saved task must include"):
                Pipeline.reconstruct(path)

    def test_binding_adapter_invokes_user_fn_with_stream_binding_objects(self):
        reader = _FakeRawReader(np.arange(4, dtype=np.float32))
        writer = _FakeRawWriter()
        event = object()
        seen: dict[str, object] = {}

        def task_fn(**kwargs):
            seen.update(kwargs)

        with patch("pythusa._pipeline._helpers.get_reader", return_value=reader) as get_reader_mock:
            with patch("pythusa._pipeline._helpers.get_writer", return_value=writer) as get_writer_mock:
                with patch("pythusa._pipeline._helpers.get_event", return_value=event) as get_event_mock:
                    _invoke_task_with_bindings(
                        task_fn,
                        {"samples": "raw_adc"},
                        {"fft": "spectra"},
                        {"shutdown": "halt"},
                        None,
                        None,
                        {"samples": {"name": "raw_adc", "shape": (4,), "dtype": np.float32}},
                        {"fft": {"name": "spectra", "shape": (4,), "dtype": np.complex64}},
                    )

        self.assertIs(seen["samples"].raw, reader)
        self.assertIs(seen["fft"].raw, writer)
        self.assertIs(seen["shutdown"], event)
        get_reader_mock.assert_called_once_with("raw_adc")
        get_writer_mock.assert_called_once_with("spectra")
        get_event_mock.assert_called_once_with("halt")

    def test_stream_reader_read_returns_shaped_array_and_writer_write_validates_frame(self):
        reader = _FakeRawReader(np.arange(4, dtype=np.float32))
        writer = _FakeRawWriter()
        captured: dict[str, object] = {}

        def task_fn(*, samples, fft) -> None:
            frame = samples.read()
            captured["frame"] = frame
            captured["write_ok"] = fft.write(np.ones(4, dtype=np.complex64))
            captured["reader_raw"] = samples.raw
            captured["writer_raw"] = fft.raw

        with patch("pythusa._pipeline._helpers.get_reader", return_value=reader):
            with patch("pythusa._pipeline._helpers.get_writer", return_value=writer):
                _invoke_task_with_bindings(
                    task_fn,
                    {"samples": "raw_adc"},
                    {"fft": "spectra"},
                    {},
                    None,
                    None,
                    {"samples": {"name": "raw_adc", "shape": (2, 2), "dtype": np.float32}},
                    {"fft": {"name": "spectra", "shape": (4,), "dtype": np.complex64}},
                )

        self.assertTrue(np.array_equal(captured["frame"], np.arange(4, dtype=np.float32).reshape(2, 2)))
        self.assertTrue(captured["write_ok"])
        self.assertIs(captured["reader_raw"], reader)
        self.assertIs(captured["writer_raw"], writer)
        self.assertEqual(writer.last_written.shape, (4,))
        self.assertEqual(writer.last_written.dtype, np.complex64)

    def test_stream_reader_set_blocking_toggles_reader_participation(self):
        reader = _FakeRawReader(np.arange(4, dtype=np.float32))
        captured: dict[str, object] = {}

        def task_fn(*, samples) -> None:
            samples.set_blocking(False)
            captured["inactive"] = samples.is_blocking()
            samples.set_blocking(True)
            captured["active"] = samples.is_blocking()

        with patch("pythusa._pipeline._helpers.get_reader", return_value=reader):
            _invoke_task_with_bindings(
                task_fn,
                {"samples": "raw_adc"},
                {},
                {},
                None,
                None,
                {"samples": {"name": "raw_adc", "shape": (4,), "dtype": np.float32}},
                {},
            )

        self.assertFalse(captured["inactive"])
        self.assertTrue(captured["active"])
        self.assertEqual(reader.calls, ["set_reader_active:False", "jump_to_writer", "set_reader_active:True"])

    def test_stream_reader_read_returns_owned_array(self):
        raw = _StickyRawReader(np.arange(4, dtype=np.float32))
        reader = make_reader_binding(
            raw,
            name="samples",
            shape=(2, 2),
            dtype=np.float32,
        )

        frame = reader.read()
        frame[0, 0] = 999.0

        self.assertEqual(raw._array[0], 0.0)

    def test_binding_adapter_runs_toggleable_tasks_with_wait_reset_then_fn(self):
        event = _FakeToggleEvent()
        calls: list[str] = []

        def task_fn() -> None:
            calls.append("fn")
            if len(calls) == 2:
                raise _StopLoop

        with patch("pythusa._pipeline._helpers.get_event", return_value=event):
            with self.assertRaises(_StopLoop):
                _invoke_task_with_bindings(
                    task_fn,
                    {},
                    {},
                    {"shutdown": "halt"},
                    "toggleable",
                    "shutdown",
                )

        self.assertEqual(calls, ["fn", "fn"])
        self.assertEqual(event.calls, ["wait", "reset", "wait", "reset"])

    def test_binding_adapter_runs_switchable_tasks_with_wait_then_fn(self):
        event = _FakeSwitchEvent()
        calls: list[str] = []

        def task_fn() -> None:
            calls.append("fn")
            if len(calls) == 2:
                raise _StopLoop

        with patch("pythusa._pipeline._helpers.get_event", return_value=event):
            with self.assertRaises(_StopLoop):
                _invoke_task_with_bindings(
                    task_fn,
                    {},
                    {},
                    {"shutdown": "halt"},
                    "switchable",
                    "shutdown",
                )

        self.assertEqual(calls, ["fn", "fn"])
        self.assertEqual(event.calls, ["wait", "wait"])

    def test_compile_allows_control_event_not_accepted_by_callable(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(1,), dtype=np.int32)
            pipe.add_event("go")
            pipe.add_task.toggleable(
                "source",
                activate_on="go",
                fn=_source_task,
                writes={"samples": "samples"},
                events={"go": "go"},
            )
            pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})

            pipe.compile()

            self.assertIn("source", pipe._manager._task_specs)
        finally:
            pipe._manager.close()

    def test_compile_rejects_duplicate_local_binding_names(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_event("shutdown")
        pipe.add_task(
            "acquire",
            fn=_source_task,
            writes={"samples": "samples"},
            events={"samples": "shutdown"},
        )
        pipe.add_task("sink", fn=_sink_task, reads={"fft": "samples"})

        with self.assertRaisesRegex(ValueError, "reuses local binding names"):
            pipe.compile()

    def test_compile_rejects_missing_required_callable_binding(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
        pipe.add_task("sink", fn=_sink_requires_shutdown, reads={"samples": "samples"})

        with self.assertRaisesRegex(ValueError, "requires parameter 'shutdown'"):
            pipe.compile()

    def test_compile_allows_defaulted_unbound_callable_parameter(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32)
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task("sink", fn=_sink_with_default_shutdown, reads={"samples": "samples"})

            pipe.compile()

            self.assertIn("sink", pipe._manager._task_specs)
        finally:
            pipe._manager.close()

    def test_compile_allows_callable_with_kwargs_catchall(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32)
            pipe.add_event("shutdown")
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task(
                "sink",
                fn=_task_with_kwargs,
                reads={"input_samples": "samples"},
                events={"halt": "shutdown"},
            )

            pipe.compile()

            self.assertIn("sink", pipe._manager._task_specs)
        finally:
            pipe._manager.close()

    def test_compile_rejects_positional_only_bound_parameter(self):
        pipe = Pipeline("radar")
        pipe.add_stream("samples", shape=(4,), dtype=np.float32)
        pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
        pipe.add_task("sink", fn=_positional_only_sink, reads={"samples": "samples"})

        with self.assertRaisesRegex(ValueError, "positional-only"):
            pipe.compile()

    def test_compile_aligns_ring_size_when_cache_align_enabled(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("bytes", shape=(3,), dtype=np.uint8, cache_align=True)
            pipe.add_task("acquire", fn=_bytes_source, writes={"payload": "bytes"})
            pipe.add_task("sink", fn=_bytes_sink, reads={"payload": "bytes"})

            pipe.compile()

            self.assertEqual(pipe._manager._ring_specs["bytes"].size, 128)
        finally:
            pipe._manager.close()

    def test_compile_uses_configured_frame_count_for_ring_size(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("samples", shape=(4,), dtype=np.float32, frames=5, cache_align=False)
            pipe.add_task("acquire", fn=_source_task, writes={"samples": "samples"})
            pipe.add_task("sink", fn=_sample_reader, reads={"samples": "samples"})

            pipe.compile()

            self.assertEqual(pipe._manager._ring_specs["samples"].size, 80)
        finally:
            pipe._manager.close()

    def test_compile_keeps_ring_size_unaligned_when_cache_align_disabled(self):
        pipe = Pipeline("radar")

        try:
            pipe.add_stream("bytes", shape=(3,), dtype=np.uint8, cache_align=False)
            pipe.add_task("acquire", fn=_bytes_source, writes={"payload": "bytes"})
            pipe.add_task("sink", fn=_bytes_sink, reads={"payload": "bytes"})

            pipe.compile()

            self.assertEqual(pipe._manager._ring_specs["bytes"].size, 96)
        finally:
            pipe._manager.close()


def _task_fn() -> None:
    return None


def _source_task(samples) -> None:
    return None


def _source_with_shutdown(samples, shutdown) -> None:
    return None


def _worker_task(samples, fft, shutdown) -> None:
    return None


def _sink_task(fft) -> None:
    return None


def _sink_requires_shutdown(samples, shutdown) -> None:
    return None


def _consumer_with_shutdown(samples, shutdown) -> None:
    return None


def _sample_reader(samples) -> None:
    return None


def _sample_reader_with_shutdown(samples, shutdown) -> None:
    return None


def _cycle_task_a(b_in, a_out) -> None:
    return None


def _cycle_task_b(a_in, b_out) -> None:
    return None


def _sink_with_default_shutdown(samples, shutdown=None) -> None:
    return None


def _task_with_kwargs(**kwargs) -> None:
    return None


def _positional_only_sink(samples, /) -> None:
    return None


def _bytes_source(payload) -> None:
    return None


def _bytes_sink(payload) -> None:
    return None


class _StopLoop(Exception):
    pass


class _FakeToggleEvent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def wait(self, timeout=None) -> bool:
        _ = timeout
        self.calls.append("wait")
        return True

    def reset(self) -> None:
        self.calls.append("reset")


class _FakeSwitchEvent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def wait(self, timeout=None) -> bool:
        _ = timeout
        self.calls.append("wait")
        return True


class _FakeRawReader:
    def __init__(self, array):
        self._array = np.asarray(array)
        self.calls: list[str] = []
        self._active = True

    def read_array(self, nbytes, dtype):
        _ = nbytes
        _ = dtype
        return self._array.copy()

    def expose_reader_mem_view(self, nbytes):
        self.calls.append(f"expose_reader_mem_view:{nbytes}")
        if nbytes > self._array.nbytes:
            return memoryview(self._array), None, self._array.nbytes, False
        return memoryview(self._array)[:nbytes], None, nbytes, False

    def simple_read(self, *_args, **_kwargs):
        raise AssertionError("read_into path is not exercised in this test")

    def inc_reader_pos(self, nbytes, *_args, **_kwargs):
        self.calls.append(f"inc_reader_pos:{nbytes}")

    def set_reader_active(self, active: bool) -> None:
        self.calls.append(f"set_reader_active:{active}")
        self._active = active

    def is_reader_active(self) -> bool:
        return self._active

    def jump_to_writer(self) -> None:
        self.calls.append("jump_to_writer")


class _FakeRawWriter:
    def __init__(self) -> None:
        self.last_written = None
        self.calls: list[str] = []
        self._view_buffer = np.zeros((0,), dtype=np.uint8)

    def write_array(self, array) -> int:
        self.last_written = np.asarray(array).copy()
        return int(self.last_written.nbytes)

    def expose_writer_mem_view(self, nbytes):
        self.calls.append(f"expose_writer_mem_view:{nbytes}")
        if self._view_buffer.nbytes < nbytes:
            self._view_buffer = np.zeros((nbytes,), dtype=np.uint8)
        return memoryview(self._view_buffer)[:nbytes], None, nbytes, False

    def inc_writer_pos(self, nbytes, *_args, **_kwargs):
        self.calls.append(f"inc_writer_pos:{nbytes}")


class _StickyRawReader:
    def __init__(self, array):
        self._array = np.asarray(array)

    def read_array(self, nbytes, dtype):
        _ = nbytes
        _ = dtype
        return self._array


if __name__ == "__main__":
    unittest.main()
