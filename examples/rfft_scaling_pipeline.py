from __future__ import annotations

"""Run with: python examples/rfft_scaling_pipeline.py"""

import argparse
import multiprocessing as mp
import uuid
from functools import partial
from pathlib import Path

import numpy as np

import pythusa
from fir128_scaling_pipeline import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_MAX_RATE_HZ,
    DEFAULT_MIN_BENCHMARK_FRAMES,
    DEFAULT_MIN_RATE_HZ,
    DEFAULT_SEGMENT_DURATION_S,
    META_FIELDS,
    RunResult,
    _best_by_rate,
    _collect_result,
    _default_frame_sizes_csv,
    _default_max_workers,
    _engine_signal_frames,
    _expected_worker_frames,
    _first_failed_rate,
    _highest_sustained_rate,
    _maybe_sleep,
    _parse_csv_ints,
    _plot_sweeps,
    _power_of_two_counts,
    _print_run_result,
    _rate_ladder,
    _raw_try_read_into,
    _raw_write_exact,
    _wait_for_active_readers,
    aggregate_results,
    source_signal,
    split_round_robin,
)


def rfft_worker(
    worker_frames,
    worker_meta,
    result_frames,
    result_meta,
    *,
    frame_length: int,
    result_frame_length: int,
    expected_frames: int,
) -> None:
    worker_frame_ring = worker_frames.raw
    worker_meta_ring = worker_meta.raw
    result_frame_ring = result_frames.raw
    result_meta_ring = result_meta.raw
    _wait_for_active_readers(result_frame_ring, result_meta_ring)

    frame = np.empty((frame_length,), dtype=np.float32)
    frame_view = memoryview(frame).cast("B")
    output = np.empty((result_frame_length,), dtype=np.float32)
    output_view = memoryview(output).cast("B")
    meta = np.empty((META_FIELDS,), dtype=np.float64)
    meta_view = memoryview(meta).cast("B")
    pending_meta: np.ndarray | None = None
    processed = 0

    while processed < expected_frames:
        if pending_meta is None:
            if _raw_try_read_into(worker_meta_ring, meta_view, meta.nbytes):
                pending_meta = np.array(meta, copy=True)
            else:
                _maybe_sleep()
                continue

        if not _raw_try_read_into(worker_frame_ring, frame_view, frame.nbytes):
            _maybe_sleep()
            continue

        spectrum_mag = np.abs(np.fft.rfft(frame))
        output[:] = spectrum_mag.astype(np.float32, copy=False)
        _raw_write_exact(result_meta_ring, memoryview(pending_meta).cast("B"), pending_meta.nbytes)
        _raw_write_exact(result_frame_ring, output_view, output.nbytes)
        pending_meta = None
        processed += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a split RFFT PYTHUSA pipeline across sample-rate, frame-size, "
            "and worker-count sweeps using engine-data-derived synthetic signals."
        )
    )
    parser.add_argument(
        "--segment-duration-s",
        type=float,
        default=DEFAULT_SEGMENT_DURATION_S,
        help="Interpolated high-interest segment duration in seconds before repetition.",
    )
    parser.add_argument(
        "--min-rate-hz",
        type=int,
        default=DEFAULT_MIN_RATE_HZ,
        help="Starting interpolated sample rate for the rate sweep.",
    )
    parser.add_argument(
        "--max-rate-hz",
        type=int,
        default=DEFAULT_MAX_RATE_HZ,
        help="Maximum interpolated sample rate for the rate sweep.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=_default_max_workers(),
        help="Maximum number of RFFT workers to try. Rounded down to powers of two.",
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=DEFAULT_FRAME_LENGTH,
        help="Frame length used during the sample-rate sweep.",
    )
    parser.add_argument(
        "--frame-sizes",
        type=str,
        default=_default_frame_sizes_csv(),
        help="Comma-separated frame sizes for the frame-size sweep.",
    )
    parser.add_argument(
        "--frame-sweep-rate-hz",
        type=int,
        default=None,
        help="Sample rate for the frame-size sweep. Defaults to the highest sustained rate.",
    )
    parser.add_argument(
        "--min-benchmark-frames",
        type=int,
        default=DEFAULT_MIN_BENCHMARK_FRAMES,
        help="Minimum number of frames to benchmark by repeating the pre-generated signal.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional path to save the heatmap figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plots without opening a matplotlib window.",
    )
    return parser.parse_args()


def _run_configuration(
    *,
    sample_rate_hz: int,
    frame_length: int,
    worker_count: int,
    segment_duration_s: float,
    min_benchmark_frames: int,
    pace_source: bool,
) -> RunResult:
    frames = _engine_signal_frames(
        sample_rate_hz,
        segment_duration_s,
        frame_length=frame_length,
        min_benchmark_frames=min_benchmark_frames,
    )
    total_frames = int(frames.shape[0])
    worker_count = min(worker_count, total_frames)
    result_frame_length = frame_length // 2 + 1
    frame_interval_s = (frame_length / float(sample_rate_hz)) if pace_source else None
    startup_delay_s = 0.0 if pace_source else 0.1

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    source_task = partial(
        source_signal,
        frames=frames,
        frame_interval_s=frame_interval_s,
        startup_delay_s=startup_delay_s,
    )
    aggregate_task = partial(
        aggregate_results,
        frame_length=frame_length,
        result_frame_length=result_frame_length,
        total_frames=total_frames,
        worker_count=worker_count,
        result_queue=result_queue,
    )

    with pythusa.Pipeline(f"rfft-{sample_rate_hz}-{frame_length}-{worker_count}") as pipe:
        run_prefix = f"rfft_{uuid.uuid4().hex[:10]}"
        source_frames_name = f"{run_prefix}_source_frames"
        source_meta_name = f"{run_prefix}_source_meta"
        pipe.add_stream(source_frames_name, shape=(frame_length,), dtype=np.float32, cache_align=False)
        pipe.add_stream(source_meta_name, shape=(META_FIELDS,), dtype=np.float64, cache_align=False)

        splitter_writes: dict[str, str] = {}
        aggregator_reads: dict[str, str] = {}

        for worker_index in range(worker_count):
            worker_frame_name = f"{run_prefix}_worker_{worker_index}_frames"
            worker_meta_name = f"{run_prefix}_worker_{worker_index}_meta"
            result_frame_name = f"{run_prefix}_result_{worker_index}_frames"
            result_meta_name = f"{run_prefix}_result_{worker_index}_meta"

            pipe.add_stream(worker_frame_name, shape=(frame_length,), dtype=np.float32, cache_align=False)
            pipe.add_stream(worker_meta_name, shape=(META_FIELDS,), dtype=np.float64, cache_align=False)
            pipe.add_stream(result_frame_name, shape=(result_frame_length,), dtype=np.float32, cache_align=False)
            pipe.add_stream(result_meta_name, shape=(META_FIELDS,), dtype=np.float64, cache_align=False)

            splitter_writes[f"worker_{worker_index}_frames"] = worker_frame_name
            splitter_writes[f"worker_{worker_index}_meta"] = worker_meta_name
            aggregator_reads[f"result_{worker_index}_frames"] = result_frame_name
            aggregator_reads[f"result_{worker_index}_meta"] = result_meta_name

        pipe.add_task(
            "source",
            fn=source_task,
            writes={
                "source_frames": source_frames_name,
                "source_meta": source_meta_name,
            },
        )
        pipe.add_task(
            "splitter",
            fn=partial(
                split_round_robin,
                frame_length=frame_length,
                total_frames=total_frames,
                worker_count=worker_count,
            ),
            reads={
                "source_frames": source_frames_name,
                "source_meta": source_meta_name,
            },
            writes=splitter_writes,
        )

        for worker_index in range(worker_count):
            worker_frame_name = f"{run_prefix}_worker_{worker_index}_frames"
            worker_meta_name = f"{run_prefix}_worker_{worker_index}_meta"
            result_frame_name = f"{run_prefix}_result_{worker_index}_frames"
            result_meta_name = f"{run_prefix}_result_{worker_index}_meta"
            pipe.add_task(
                f"rfft_{worker_index}",
                fn=partial(
                    rfft_worker,
                    frame_length=frame_length,
                    result_frame_length=result_frame_length,
                    expected_frames=_expected_worker_frames(
                        total_frames,
                        worker_index,
                        worker_count,
                    ),
                ),
                reads={
                    "worker_frames": worker_frame_name,
                    "worker_meta": worker_meta_name,
                },
                writes={
                    "result_frames": result_frame_name,
                    "result_meta": result_meta_name,
                },
            )

        pipe.add_task(
            "aggregate",
            fn=aggregate_task,
            reads=aggregator_reads,
        )

        pipe.start()
        try:
            summary = _collect_result(result_queue)
            pipe.join(timeout=120.0)
        finally:
            result_queue.close()
            result_queue.join_thread()

    target_ksps = sample_rate_hz / 1_000.0
    required_throughput_mb_s = sample_rate_hz * np.dtype(np.float32).itemsize / 1_000_000.0
    achieved_ksps = float(summary["ksps"])
    realtime_ratio = achieved_ksps / max(target_ksps, 1e-9)
    margin_pct = (realtime_ratio - 1.0) * 100.0

    return RunResult(
        sample_rate_hz=sample_rate_hz,
        frame_length=frame_length,
        worker_count=worker_count,
        paced_source=pace_source,
        total_frames=int(summary["total_frames"]),
        total_samples=int(summary["total_samples"]),
        throughput_mb_s=float(summary["input_equivalent_mb_s"]),
        result_payload_mb_s=float(summary["result_payload_mb_s"]),
        required_throughput_mb_s=required_throughput_mb_s,
        ksps=achieved_ksps,
        target_ksps=target_ksps,
        realtime_ratio=realtime_ratio,
        margin_pct=margin_pct,
        keeps_up=realtime_ratio >= 1.0,
        latency_mean_ms=float(summary["latency_mean_ms"]),
        latency_p95_ms=float(summary["latency_p95_ms"]),
        elapsed_s=float(summary["elapsed_s"]),
    )


def _run_rate_sweep(
    *,
    rates: list[int],
    worker_counts: list[int],
    frame_length: int,
    segment_duration_s: float,
    min_benchmark_frames: int,
) -> list[RunResult]:
    results: list[RunResult] = []
    for sample_rate_hz in rates:
        for worker_count in worker_counts:
            result = _run_configuration(
                sample_rate_hz=sample_rate_hz,
                frame_length=frame_length,
                worker_count=worker_count,
                segment_duration_s=segment_duration_s,
                min_benchmark_frames=min_benchmark_frames,
                pace_source=True,
            )
            results.append(result)
            _print_run_result(result)
    return results


def _run_frame_sweep(
    *,
    sample_rate_hz: int,
    frame_sizes: list[int],
    worker_counts: list[int],
    segment_duration_s: float,
    min_benchmark_frames: int,
) -> list[RunResult]:
    results: list[RunResult] = []
    for frame_length in frame_sizes:
        for worker_count in worker_counts:
            result = _run_configuration(
                sample_rate_hz=sample_rate_hz,
                frame_length=frame_length,
                worker_count=worker_count,
                segment_duration_s=segment_duration_s,
                min_benchmark_frames=min_benchmark_frames,
                pace_source=False,
            )
            results.append(result)
            _print_run_result(result)
    return results


def main() -> None:
    args = _parse_args()
    rates = _rate_ladder(args.min_rate_hz, args.max_rate_hz)
    worker_counts = _power_of_two_counts(max(1, args.max_workers))
    frame_sizes = sorted(set(_parse_csv_ints(args.frame_sizes)))

    print("starting RFFT scaling example")
    print(
        f"segment_duration={args.segment_duration_s:.1f}s "
        f"rate_sweep_frame_length={args.frame_length:,} "
        f"rates={rates} "
        f"worker_counts={worker_counts} "
        f"frame_sizes={frame_sizes}"
    )
    print("signal generation is done before the pipeline starts; the source task only writes pre-generated frames")
    print("low-rate signals are repeated as needed so each run has a stable benchmark window")
    print()
    print("rate sweep")

    rate_results = _run_rate_sweep(
        rates=rates,
        worker_counts=worker_counts,
        frame_length=args.frame_length,
        segment_duration_s=args.segment_duration_s,
        min_benchmark_frames=args.min_benchmark_frames,
    )

    best_rate_results = _best_by_rate(rate_results)
    highest_sustained = _highest_sustained_rate(best_rate_results)
    first_failed = _first_failed_rate(best_rate_results)

    frame_sweep_rate_hz = (
        args.frame_sweep_rate_hz
        if args.frame_sweep_rate_hz is not None
        else (highest_sustained.sample_rate_hz if highest_sustained is not None else rates[-1])
    )

    print()
    print("frame-size sweep")
    print("source mode is UNCAPPED for this sweep")
    frame_results = _run_frame_sweep(
        sample_rate_hz=frame_sweep_rate_hz,
        frame_sizes=frame_sizes,
        worker_counts=worker_counts,
        segment_duration_s=args.segment_duration_s,
        min_benchmark_frames=args.min_benchmark_frames,
    )

    all_results = rate_results + frame_results
    best_throughput = max(all_results, key=lambda item: item.throughput_mb_s)
    lowest_latency = min(all_results, key=lambda item: item.latency_mean_ms)

    print()
    if highest_sustained is not None:
        print(
            f"max sample rate handled={highest_sustained.sample_rate_hz:,} Hz "
            f"at frame_length={highest_sustained.frame_length:,} "
            f"with workers={highest_sustained.worker_count} "
            f"({highest_sustained.ksps:,.2f} ksps, margin {highest_sustained.margin_pct:+.2f}%)"
        )
    else:
        print("max sample rate handled=none in the tested range")

    if first_failed is not None:
        print(
            f"first cap-out rate={first_failed.sample_rate_hz:,} Hz "
            f"at frame_length={first_failed.frame_length:,} "
            f"with workers={first_failed.worker_count} "
            f"({first_failed.ksps:,.2f} ksps, margin {first_failed.margin_pct:+.2f}%)"
        )
    else:
        print(f"cap-out not reached through {rates[-1]:,} Hz")

    print(
        f"max processed throughput={best_throughput.throughput_mb_s:.2f} MB/s "
        f"mode={'PACED' if best_throughput.paced_source else 'UNCAPPED'} "
        f"at rate={best_throughput.sample_rate_hz:,} Hz "
        f"frame_length={best_throughput.frame_length:,} "
        f"workers={best_throughput.worker_count}"
    )
    print(
        f"lowest latency={lowest_latency.latency_mean_ms:.3f} ms "
        f"mode={'PACED' if lowest_latency.paced_source else 'UNCAPPED'} "
        f"at rate={lowest_latency.sample_rate_hz:,} Hz "
        f"frame_length={lowest_latency.frame_length:,} "
        f"workers={lowest_latency.worker_count}"
    )

    _plot_sweeps(
        title_label="RFFT",
        rate_results=rate_results,
        frame_results=frame_results,
        rate_values=rates,
        rate_frame_length=args.frame_length,
        frame_values=frame_sizes,
        worker_values=worker_counts,
        frame_sweep_rate_hz=frame_sweep_rate_hz,
        highest_sustained=highest_sustained,
        best_throughput=best_throughput,
        lowest_latency=lowest_latency,
        plot_out=args.plot_out,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()
