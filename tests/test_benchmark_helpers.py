"""Run with: pytest tests/test_benchmark_helpers.py -q"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))


def _load_benchmark_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark module {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


DSP_BENCH = _load_benchmark_module("test_dsp_benchmark_suite_module", "benchmarks/dsp_benchmark_suite.py")
REPORTING = _load_benchmark_module("test_benchmark_reporting_module", "benchmarks/_reporting.py")
ROCKETDATA = _load_benchmark_module("test_rocketdata_module", "benchmarks/rocketdata_test.py")
FFT_COMPARE = _load_benchmark_module("test_compare_fft_module", "benchmarks/compare_fft_benchmarks.py")


class DSPBenchmarkHelperTests(unittest.TestCase):
    def test_frame_from_ring_view_returns_none_for_partial_batch(self) -> None:
        ring_view = (memoryview(bytearray(DSP_BENCH.BATCH_NBYTES)), None, DSP_BENCH.BATCH_NBYTES - 1, False)
        self.assertIsNone(DSP_BENCH._frame_from_ring_view(ring_view))
        DSP_BENCH._release_ring_view(ring_view)

    def test_frame_from_ring_view_returns_zero_copy_array(self) -> None:
        raw = bytearray(DSP_BENCH.BATCH_NBYTES)
        ring_view = (memoryview(raw), None, DSP_BENCH.BATCH_NBYTES, False)
        batch = DSP_BENCH._frame_from_ring_view(ring_view)
        self.assertIsNotNone(batch)
        batch_mv, frame = batch
        try:
            frame[0, 0] = np.asarray(1.25, dtype=DSP_BENCH.DTYPE)
            payload = np.frombuffer(
                raw,
                dtype=DSP_BENCH.DTYPE,
                offset=DSP_BENCH.HEADER_NBYTES,
            ).reshape(DSP_BENCH.PAYLOAD_SHAPE)
            self.assertEqual(payload[0, 0], frame[0, 0])
            self.assertIs(batch_mv, ring_view[0])
        finally:
            del frame
            DSP_BENCH._release_ring_view(ring_view)

    def test_frame_from_ring_view_rejects_wrapped_full_batch(self) -> None:
        ring_view = (memoryview(bytearray(DSP_BENCH.BATCH_NBYTES)), None, DSP_BENCH.BATCH_NBYTES, True)
        with self.assertRaises(AssertionError):
            DSP_BENCH._frame_from_ring_view(ring_view)
        DSP_BENCH._release_ring_view(ring_view)

    def test_release_ring_view_releases_all_views(self) -> None:
        first = memoryview(bytearray(8))
        second = memoryview(bytearray(4))
        DSP_BENCH._release_ring_view((first, second, 0, True))
        with self.assertRaises(ValueError):
            len(first)
        with self.assertRaises(ValueError):
            len(second)

    def test_wait_for_inflight_budget_succeeds_when_backlog_is_below_cap(self) -> None:
        writer = MagicMock()
        writer.ring_buffer_size = DSP_BENCH.BATCH_NBYTES * 4
        writer.compute_max_amount_writable.return_value = writer.ring_buffer_size - (DSP_BENCH.BATCH_NBYTES - 1)

        with patch.object(DSP_BENCH, "MAX_IN_FLIGHT_BATCHES", 1):
            self.assertTrue(DSP_BENCH._wait_for_inflight_budget(writer, time.perf_counter_ns() + 1_000_000))

        writer.compute_max_amount_writable.assert_called_once_with(force_rescan=True)

    def test_wait_for_inflight_budget_returns_false_after_deadline(self) -> None:
        writer = MagicMock()
        writer.ring_buffer_size = DSP_BENCH.BATCH_NBYTES * 4
        with patch.object(DSP_BENCH, "MAX_IN_FLIGHT_BATCHES", 1):
            self.assertFalse(DSP_BENCH._wait_for_inflight_budget(writer, time.perf_counter_ns() - 1))
        writer.compute_max_amount_writable.assert_not_called()

    def test_parse_args_accepts_graph_options(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "dsp_benchmark_suite.py",
                "--graph",
                "--graph-out",
                "/tmp/dsp-heatmaps.png",
                "--no-show",
            ],
        ):
            args = DSP_BENCH._parse_args()

        self.assertTrue(args.graph)
        self.assertEqual(args.graph_out, Path("/tmp/dsp-heatmaps.png"))
        self.assertTrue(args.no_show)

    def test_graph_requested_accepts_flag_or_output_path(self) -> None:
        self.assertTrue(DSP_BENCH._graph_requested(argparse.Namespace(graph=True, graph_out=None)))
        self.assertTrue(DSP_BENCH._graph_requested(argparse.Namespace(graph=False, graph_out=Path("/tmp/out.png"))))
        self.assertFalse(DSP_BENCH._graph_requested(argparse.Namespace(graph=False, graph_out=None)))

    def test_metric_matrix_preserves_result_order(self) -> None:
        results = [
            DSP_BENCH.BenchmarkResult(
                kernel="gain",
                throughput_mb_s=12.5,
                batches=10,
                payload_bytes=100,
                latency_mean_ms=1.0,
                latency_min_ms=0.8,
                latency_p50_ms=0.9,
                latency_p95_ms=1.4,
                latency_p99_ms=1.8,
                latency_max_ms=2.0,
                ring_reserved_mb=3.0,
                peak_task_rss_mb=4.0,
                peak_parent_rss_mb=5.0,
            ),
            DSP_BENCH.BenchmarkResult(
                kernel="rfft",
                throughput_mb_s=33.25,
                batches=20,
                payload_bytes=200,
                latency_mean_ms=0.4,
                latency_min_ms=0.2,
                latency_p50_ms=0.3,
                latency_p95_ms=0.7,
                latency_p99_ms=0.9,
                latency_max_ms=1.1,
                ring_reserved_mb=3.0,
                peak_task_rss_mb=4.0,
                peak_parent_rss_mb=5.0,
            ),
        ]

        matrix = DSP_BENCH._metric_matrix(results, "throughput_mb_s")

        np.testing.assert_array_equal(matrix, np.asarray([[12.5, 33.25]], dtype=np.float64))

    def test_load_pyplot_reports_missing_matplotlib_clearly(self) -> None:
        with patch.object(DSP_BENCH.importlib, "import_module", side_effect=ModuleNotFoundError("matplotlib")):
            with self.assertRaises(SystemExit) as exc_info:
                DSP_BENCH._load_pyplot()

        self.assertIn("Install benchmark extras", str(exc_info.exception))


class BenchmarkReportingTests(unittest.TestCase):
    def test_build_payload_normalizes_dataclasses_and_numpy_scalars(self) -> None:
        @dataclass(frozen=True)
        class SampleResult:
            value: np.int64

        payload = REPORTING.build_payload(
            benchmark="sample",
            config={"dtype": np.str_("float32")},
            results=[SampleResult(value=np.int64(7))],
            summary={"count": np.int64(3)},
            label="demo",
        )

        self.assertEqual(payload["benchmark"], "sample")
        self.assertEqual(payload["label"], "demo")
        self.assertEqual(payload["results"][0]["value"], 7)
        self.assertEqual(payload["summary"]["count"], 3)

    def test_emit_payload_writes_json_file(self) -> None:
        payload = REPORTING.build_payload(
            benchmark="sample",
            config={"rows": 128},
            results=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "result.json"
            REPORTING.emit_payload(payload, json_stdout=False, json_out=str(out_path))
            self.assertTrue(out_path.exists())
            text = out_path.read_text(encoding="utf-8")
            self.assertIn('"benchmark": "sample"', text)
            self.assertIn('"rows": 128', text)


class RocketBenchmarkHelperTests(unittest.TestCase):
    def test_rocket_results_add_window_fill_to_processing_latency(self) -> None:
        suite_payload = {
            "results": [
                {
                    "kernel": "rfft",
                    "latency_mean_ms": 0.12,
                    "latency_p95_ms": 0.20,
                    "latency_p99_ms": 0.30,
                    "latency_max_ms": 0.40,
                    "throughput_mb_s": 123.4,
                    "batches": 10,
                    "ring_reserved_mb": 1.5,
                }
            ]
        }

        results = ROCKETDATA._rocket_results_for_window(
            sample_rates_hz=(100_000,),
            window_samples=512,
            suite_payload=suite_payload,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertAlmostEqual(result.window_fill_ms, 5.12, places=6)
        self.assertAlmostEqual(result.total_mean_detection_ms, 5.24, places=6)
        self.assertAlmostEqual(result.total_p95_detection_ms, 5.32, places=6)
        self.assertAlmostEqual(result.total_p99_detection_ms, 5.42, places=6)
        self.assertAlmostEqual(result.total_max_detection_ms, 5.52, places=6)


class FFTComparisonHelperTests(unittest.TestCase):
    def test_trial_record_keeps_comparison_fields_only(self) -> None:
        result = FFT_COMPARE.TrialResult(
            implementation="pythusa",
            name="candidate",
            throughput_mb_s=12.3,
            total_fft_bytes=456,
            stdout="noisy logs",
        )

        record = FFT_COMPARE._trial_record(result)

        self.assertEqual(
            record,
            {
                "implementation": "pythusa",
                "candidate": "candidate",
                "throughput_mb_s": 12.3,
                "total_fft_bytes": 456,
            },
        )


if __name__ == "__main__":
    unittest.main()
