from __future__ import annotations

"""Run with: python examples/rfft_benchmark_like.py"""

import argparse
import multiprocessing as mp
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Empty

import matplotlib.pyplot as plt
import numpy as np

import pythusa
from fir128_scaling_pipeline import (
    DEFAULT_MIN_BENCHMARK_FRAMES,
    DEFAULT_SEGMENT_DURATION_S,
    HEATMAP_CMAP,
    META_FIELDS,
    _default_frame_sizes_csv,
    _engine_signal_frames,
    _maybe_sleep,
    _parse_csv_ints,
    _raw_try_read_into,
    source_signal,
)


DEFAULT_SIGNAL_RATE_HZ = 256_000
CHANNELS = 2
ITEMSIZE = np.dtype(np.float32).itemsize


@dataclass(frozen=True)
class BenchmarkLikeResult:
    signal_rate_hz: int
    frame_length: int
    channels: int
    total_frames: int
    total_samples: int
    throughput_mb_s: float
    max_channel_rate_hz: float
    latency_mean_ms: float
    latency_p95_ms: float
    elapsed_s: float


def _collect_benchmark_result(result_queue: mp.queues.Queue) -> dict[str, float]:
    try:
        return result_queue.get(timeout=120.0)
    except Empty as exc:
        raise RuntimeError("Timed out waiting for RFFT benchmark-like result.") from exc


def _engine_signal_frames_2ch(
    signal_rate_hz: int,
    segment_duration_s: float,
    *,
    frame_length: int,
    min_benchmark_frames: int,
) -> np.ndarray:
    primary = _engine_signal_frames(
        signal_rate_hz,
        segment_duration_s,
        frame_length=frame_length,
        min_benchmark_frames=min_benchmark_frames,
        seed=5,
    )
    secondary = _engine_signal_frames(
        signal_rate_hz,
        segment_duration_s,
        frame_length=frame_length,
        min_benchmark_frames=min_benchmark_frames,
        seed=17,
    )
    frames = np.empty((primary.shape[0], frame_length, CHANNELS), dtype=np.float32)
    frames[:, :, 0] = primary
    frames[:, :, 1] = 0.82 * secondary + 0.18 * primary
    return np.ascontiguousarray(frames)


def rfft_benchmark_worker(
    source_frames,
    source_meta,
    *,
    frame_length: int,
    channels: int,
    expected_frames: int,
    result_queue,
) -> None:
    source_frame_ring = source_frames.raw
    source_meta_ring = source_meta.raw

    frame = np.empty((frame_length, channels), dtype=np.float32)
    frame_view = memoryview(frame).cast("B")
    meta = np.empty((META_FIELDS,), dtype=np.float64)
    meta_view = memoryview(meta).cast("B")

    pending_meta: np.ndarray | None = None
    latencies_ms: list[float] = []
    bytes_processed = 0
    processed = 0
    first_emitted_at: float | None = None
    finished_at = 0.0
    last_token = 0.0

    while processed < expected_frames:
        if pending_meta is None:
            if _raw_try_read_into(source_meta_ring, meta_view, meta.nbytes):
                pending_meta = np.array(meta, copy=True)
            else:
                _maybe_sleep()
                continue

        if not _raw_try_read_into(source_frame_ring, frame_view, frame.nbytes):
            _maybe_sleep()
            continue

        last_token = float(np.abs(np.fft.rfft(frame, axis=0)).max())
        arrived_at = time.perf_counter()
        emitted_at = float(pending_meta[1])
        if first_emitted_at is None:
            first_emitted_at = emitted_at

        latencies_ms.append((arrived_at - emitted_at) * 1_000.0)
        bytes_processed += frame.nbytes
        pending_meta = None
        processed += 1
        finished_at = arrived_at

    if np.isnan(last_token):  # pragma: no cover - defensive for benchmark-only path.
        raise RuntimeError("RFFT benchmark worker produced NaN token.")
    if first_emitted_at is None:
        raise RuntimeError("RFFT benchmark worker completed without receiving any frames.")

    elapsed_s = max(finished_at - first_emitted_at, 1e-9)
    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    total_samples = expected_frames * frame_length * channels
    result_queue.put(
        {
            "elapsed_s": elapsed_s,
            "throughput_mb_s": bytes_processed / elapsed_s / 1_000_000.0,
            "latency_mean_ms": float(np.mean(latency_array)),
            "latency_p95_ms": float(np.percentile(latency_array, 95.0)),
            "total_frames": expected_frames,
            "total_samples": total_samples,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-like single-worker RFFT sweep for PYTHUSA. "
            "This example matches the benchmark shape more closely than the split/join example."
        )
    )
    parser.add_argument(
        "--signal-rate-hz",
        type=int,
        default=DEFAULT_SIGNAL_RATE_HZ,
        help="Sample rate used only to generate the synthetic 2-channel signal before uncapped playback.",
    )
    parser.add_argument(
        "--segment-duration-s",
        type=float,
        default=DEFAULT_SEGMENT_DURATION_S,
        help="Interpolated high-interest segment duration in seconds before repetition.",
    )
    parser.add_argument(
        "--frame-sizes",
        type=str,
        default=_default_frame_sizes_csv(),
        help="Comma-separated frame sizes to benchmark.",
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
        help="Optional path to save the plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plots without opening a matplotlib window.",
    )
    return parser.parse_args()


def _run_configuration(
    *,
    signal_rate_hz: int,
    frame_length: int,
    segment_duration_s: float,
    min_benchmark_frames: int,
) -> BenchmarkLikeResult:
    frames = _engine_signal_frames_2ch(
        signal_rate_hz,
        segment_duration_s,
        frame_length=frame_length,
        min_benchmark_frames=min_benchmark_frames,
    )
    total_frames = int(frames.shape[0])

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    source_task = partial(
        source_signal,
        frames=frames,
        frame_interval_s=None,
        startup_delay_s=0.0,
    )
    worker_task = partial(
        rfft_benchmark_worker,
        frame_length=frame_length,
        channels=CHANNELS,
        expected_frames=total_frames,
        result_queue=result_queue,
    )

    run_prefix = f"rb{uuid.uuid4().hex[:6]}"
    source_frames_name = f"{run_prefix}_sf"
    source_meta_name = f"{run_prefix}_sm"

    with pythusa.Pipeline(f"rfft-bench-{frame_length}") as pipe:
        pipe.add_stream(source_frames_name, shape=(frame_length, CHANNELS), dtype=np.float32, cache_align=False)
        pipe.add_stream(source_meta_name, shape=(META_FIELDS,), dtype=np.float64, cache_align=False)
        pipe.add_task(
            "source",
            fn=source_task,
            writes={
                "source_frames": source_frames_name,
                "source_meta": source_meta_name,
            },
        )
        pipe.add_task(
            "rfft",
            fn=worker_task,
            reads={
                "source_frames": source_frames_name,
                "source_meta": source_meta_name,
            },
        )

        pipe.start()
        try:
            summary = _collect_benchmark_result(result_queue)
            pipe.join(timeout=120.0)
        finally:
            result_queue.close()
            result_queue.join_thread()

    throughput_mb_s = float(summary["throughput_mb_s"])
    max_channel_rate_hz = throughput_mb_s * 1_000_000.0 / (CHANNELS * ITEMSIZE)
    return BenchmarkLikeResult(
        signal_rate_hz=signal_rate_hz,
        frame_length=frame_length,
        channels=CHANNELS,
        total_frames=int(summary["total_frames"]),
        total_samples=int(summary["total_samples"]),
        throughput_mb_s=throughput_mb_s,
        max_channel_rate_hz=max_channel_rate_hz,
        latency_mean_ms=float(summary["latency_mean_ms"]),
        latency_p95_ms=float(summary["latency_p95_ms"]),
        elapsed_s=float(summary["elapsed_s"]),
    )


def _print_result(result: BenchmarkLikeResult) -> None:
    print(
        f"frame={result.frame_length:>6,d} "
        f"channels={result.channels} "
        f"throughput={result.throughput_mb_s:>8.2f} MB/s "
        f"equiv_rate={result.max_channel_rate_hz:>10,.0f} Hz/channel "
        f"latency_mean={result.latency_mean_ms:>8.3f} ms "
        f"latency_p95={result.latency_p95_ms:>8.3f} ms"
    )


def _plot_results(
    results: list[BenchmarkLikeResult],
    *,
    plot_out: Path | None,
    no_show: bool,
) -> None:
    frame_sizes = [result.frame_length for result in results]
    throughputs = [result.throughput_mb_s for result in results]
    mean_latencies = [result.latency_mean_ms for result in results]
    p95_latencies = [result.latency_p95_ms for result in results]

    cmap = plt.get_cmap(HEATMAP_CMAP)
    throughput_color = cmap(0.78)
    mean_color = cmap(0.20)
    p95_color = cmap(0.52)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    speed_ax = axes[0]
    speed_ax.plot(frame_sizes, throughputs, linewidth=2.2, color=throughput_color)
    speed_ax.set_xscale("log", base=2)
    speed_ax.set_xlabel("Frame Length")
    speed_ax.set_ylabel("Input-Equivalent Throughput (MB/s)")
    speed_ax.set_title("Max RFFT Processed Speed vs Frame Size")
    speed_ax.grid(alpha=0.25, which="both")

    rate_axis = speed_ax.secondary_yaxis(
        "right",
        functions=(
            lambda mb_s: mb_s * 1_000_000.0 / (CHANNELS * ITEMSIZE) / 1_000.0,
            lambda khz: khz * 1_000.0 * CHANNELS * ITEMSIZE / 1_000_000.0,
        ),
    )
    rate_axis.set_ylabel("Equivalent Max Rate (kHz/channel)")

    for frame_length, throughput in zip(frame_sizes, throughputs, strict=True):
        speed_ax.annotate(
            f"{throughput:.0f}",
            (frame_length, throughput),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    latency_ax = axes[1]
    latency_ax.plot(frame_sizes, mean_latencies, linewidth=2.2, color=mean_color, label="Mean")
    latency_ax.plot(frame_sizes, p95_latencies, linewidth=2.2, color=p95_color, label="P95")
    latency_ax.set_xscale("log", base=2)
    latency_ax.set_xlabel("Frame Length")
    latency_ax.set_ylabel("Latency (ms)")
    latency_ax.set_title("RFFT Latency vs Frame Size")
    latency_ax.grid(alpha=0.25, which="both")
    latency_ax.legend()

    figure.suptitle(
        "PYTHUSA benchmark-like single-worker RFFT sweep\n"
        "One source, one worker, no splitter, no result-ring writeback",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if plot_out is not None:
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_out, dpi=180)
        print(f"saved plots to {plot_out}")

    if no_show:
        plt.close(figure)
    else:
        plt.show()


def main() -> None:
    args = _parse_args()
    frame_sizes = sorted(set(_parse_csv_ints(args.frame_sizes)))

    print("starting benchmark-like RFFT example")
    print(
        f"signal_rate={args.signal_rate_hz:,} Hz "
        f"segment_duration={args.segment_duration_s:.1f}s "
        f"channels={CHANNELS} "
        f"frame_sizes={frame_sizes}"
    )
    print("signal generation is done before the pipeline starts; the source task only writes pre-generated 2-channel frames")
    print("this example matches the benchmark shape more closely than the split/join scaling example")
    print()

    results: list[BenchmarkLikeResult] = []
    for frame_length in frame_sizes:
        result = _run_configuration(
            signal_rate_hz=args.signal_rate_hz,
            frame_length=frame_length,
            segment_duration_s=args.segment_duration_s,
            min_benchmark_frames=args.min_benchmark_frames,
        )
        results.append(result)
        _print_result(result)

    best_throughput = max(results, key=lambda item: item.throughput_mb_s)
    lowest_latency = min(results, key=lambda item: item.latency_mean_ms)

    print()
    print(
        f"max processed throughput={best_throughput.throughput_mb_s:.2f} MB/s "
        f"at frame_length={best_throughput.frame_length:,} "
        f"({best_throughput.max_channel_rate_hz:,.0f} Hz/channel equivalent)"
    )
    print(
        f"lowest mean latency={lowest_latency.latency_mean_ms:.3f} ms "
        f"at frame_length={lowest_latency.frame_length:,} "
        f"(p95={lowest_latency.latency_p95_ms:.3f} ms)"
    )

    _plot_results(
        results,
        plot_out=args.plot_out,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()
