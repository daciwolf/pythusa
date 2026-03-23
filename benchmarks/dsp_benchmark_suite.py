from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import struct
import time

import numpy as np
import psutil

import pythusa
from _reporting import build_payload, emit_payload
from pythusa._processing.dsp import DSP_KERNEL_NAMES, make_benchmark_processor


MEGABYTE = 1_000_000.0
HEADER_STRUCT = struct.Struct("Q")
HEADER_NBYTES = HEADER_STRUCT.size
VALID_MODES = {"throughput", "balanced", "latency"}
MODE_DEFAULT_RING_DEPTH = {
    "throughput": 256,
    "balanced": 64,
    "latency": 16,
}
MODE_DEFAULT_MAX_IN_FLIGHT = {
    "throughput": 0,
    "balanced": 4,
    "latency": 1,
}

ROWS = int(os.environ.get("DSP_BENCH_ROWS", "16384"))
CHANNELS = int(os.environ.get("DSP_BENCH_CHANNELS", "2"))
PIPELINES = int(os.environ.get("DSP_BENCH_PIPELINES", "4"))
DURATION_S = float(os.environ.get("DSP_BENCH_DURATION_S", "2.0"))
BENCHMARK_MODE = os.environ.get("DSP_BENCH_MODE", "balanced").lower()
if BENCHMARK_MODE not in VALID_MODES:
    raise SystemExit(f"DSP_BENCH_MODE must be one of {sorted(VALID_MODES)}.")
RING_DEPTH = int(os.environ.get("DSP_BENCH_RING_DEPTH", str(MODE_DEFAULT_RING_DEPTH[BENCHMARK_MODE])))
IDLE_SLEEP_S = float(os.environ.get("DSP_BENCH_IDLE_SLEEP_S", "0.000001"))
MONITOR_INTERVAL_S = float(os.environ.get("DSP_BENCH_MONITOR_INTERVAL_S", "0.05"))
LATENCY_SAMPLE_STRIDE = int(os.environ.get("DSP_BENCH_LATENCY_SAMPLE_STRIDE", "16"))
MAX_LATENCY_SAMPLES = int(os.environ.get("DSP_BENCH_MAX_LATENCY_SAMPLES", "16384"))
MAX_IN_FLIGHT_BATCHES = int(
    os.environ.get("DSP_BENCH_MAX_IN_FLIGHT_BATCHES", str(MODE_DEFAULT_MAX_IN_FLIGHT[BENCHMARK_MODE]))
)
DTYPE = np.dtype(np.float32)
SCALAR_DTYPE = DTYPE.type
ITEMSIZE = np.dtype(DTYPE).itemsize
PAYLOAD_SHAPE = (ROWS, CHANNELS)
PAYLOAD_NBYTES = math.prod(PAYLOAD_SHAPE) * ITEMSIZE
BATCH_NBYTES = HEADER_NBYTES + PAYLOAD_NBYTES

KERNEL_NAMES = DSP_KERNEL_NAMES
SELECTED_KERNELS = tuple(
    name.strip()
    for name in os.environ.get("DSP_BENCH_KERNELS", ",".join(KERNEL_NAMES)).split(",")
    if name.strip()
)
UNKNOWN_KERNELS = sorted(set(SELECTED_KERNELS) - set(KERNEL_NAMES))
if UNKNOWN_KERNELS:
    raise SystemExit(f"Unknown DSP_BENCH_KERNELS entries: {', '.join(UNKNOWN_KERNELS)}")


@dataclass(frozen=True)
class BenchmarkResult:
    kernel: str
    throughput_mb_s: float
    batches: int
    payload_bytes: int
    latency_mean_ms: float
    latency_min_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    ring_reserved_mb: float
    peak_task_rss_mb: float
    peak_parent_rss_mb: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PYTHUSA DSP benchmark suite.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--throughput-max",
        action="store_true",
        help="Use the throughput preset: deep rings and no in-flight cap.",
    )
    mode_group.add_argument(
        "--latency-min",
        action="store_true",
        help="Use the latency preset: shallow rings and single-batch backlog.",
    )
    mode_group.add_argument(
        "--balanced",
        action="store_true",
        help="Use the balanced preset: moderate ring depth and bounded backlog.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        help="Explicit benchmark mode. Equivalent to the preset flags above.",
    )
    parser.add_argument(
        "--ring-depth",
        type=int,
        help="Override the ring depth after applying any mode preset.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        help="Override the maximum number of queued batches after applying any mode preset.",
    )
    parser.add_argument(
        "--kernels",
        help="Comma-separated kernel subset to run, for example 'rfft,stft'.",
    )
    parser.add_argument(
        "--label",
        help="Optional label stored in structured output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary to stdout instead of the human-readable table.",
    )
    parser.add_argument(
        "--json-out",
        help="Write the JSON summary to this file path.",
    )
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    global BENCHMARK_MODE, RING_DEPTH, MAX_IN_FLIGHT_BATCHES, SELECTED_KERNELS

    selected_mode = BENCHMARK_MODE
    if args.mode is not None:
        selected_mode = args.mode
    elif args.throughput_max:
        selected_mode = "throughput"
    elif args.latency_min:
        selected_mode = "latency"
    elif args.balanced:
        selected_mode = "balanced"

    BENCHMARK_MODE = selected_mode
    if args.mode is not None or args.throughput_max or args.latency_min or args.balanced:
        RING_DEPTH = MODE_DEFAULT_RING_DEPTH[BENCHMARK_MODE]
        MAX_IN_FLIGHT_BATCHES = MODE_DEFAULT_MAX_IN_FLIGHT[BENCHMARK_MODE]

    if args.ring_depth is not None:
        RING_DEPTH = args.ring_depth
    if args.max_in_flight is not None:
        MAX_IN_FLIGHT_BATCHES = args.max_in_flight
    if args.kernels is not None:
        SELECTED_KERNELS = tuple(name.strip() for name in args.kernels.split(",") if name.strip())


def _validate_selected_kernels() -> None:
    unknown_kernels = sorted(set(SELECTED_KERNELS) - set(KERNEL_NAMES))
    if unknown_kernels:
        raise SystemExit(f"Unknown kernels: {', '.join(unknown_kernels)}")


def _maybe_sleep() -> None:
    if IDLE_SLEEP_S > 0.0:
        time.sleep(IDLE_SLEEP_S)


def _ring_name(run_prefix: str, kernel_name: str, pipeline_index: int) -> str:
    return f"{run_prefix}_{kernel_name}_{pipeline_index}"


def _frame_from_ring_view(
    ring_view: tuple[memoryview, memoryview | None, int, bool],
) -> tuple[memoryview, np.ndarray] | None:
    mv1, _, size_ready, wrap_around = ring_view
    if size_ready < BATCH_NBYTES:
        return None
    # Ring size is an integer multiple of BATCH_NBYTES, so full batches
    # should stay contiguous.
    assert not wrap_around, "expected contiguous benchmark batch"
    payload_mv = mv1[HEADER_NBYTES:]
    frame = np.ndarray(PAYLOAD_SHAPE, dtype=DTYPE, buffer=payload_mv)
    return mv1, frame


def _release_ring_view(ring_view: tuple[memoryview, memoryview | None, int, bool]) -> None:
    mv1, mv2, _, _ = ring_view
    mv1.release()
    if mv2 is not None:
        mv2.release()


def _channel_frequencies() -> np.ndarray:
    return np.arange(1, CHANNELS + 1, dtype=DTYPE)


def _wait_for_inflight_budget(writer: pythusa.SharedRingBuffer, stop_at_ns: int) -> bool:
    if MAX_IN_FLIGHT_BATCHES <= 0:
        return True

    max_inflight_bytes = BATCH_NBYTES * MAX_IN_FLIGHT_BATCHES
    while time.perf_counter_ns() < stop_at_ns:
        writable = writer.compute_max_amount_writable(force_rescan=True)
        inflight_bytes = writer.ring_buffer_size - writable
        if inflight_bytes < max_inflight_bytes:
            return True
        _maybe_sleep()
    return False


def signal_producer(pipeline_index: int, run_prefix: str, kernel_name: str, stop_at_ns: int) -> None:
    writer = pythusa.get_writer(_ring_name(run_prefix, kernel_name, pipeline_index))
    phase = SCALAR_DTYPE(0.0)
    phase_step = SCALAR_DTYPE(0.2)
    base_t = np.linspace(0.0, SCALAR_DTYPE(2.0 * np.pi), ROWS, endpoint=False, dtype=DTYPE)
    channel_freqs = _channel_frequencies()
    channel_bases = np.multiply.outer(base_t, channel_freqs)
    angle = np.empty(ROWS, dtype=DTYPE)

    while time.perf_counter_ns() < stop_at_ns:
        if not _wait_for_inflight_budget(writer, stop_at_ns):
            break
        ring_view = writer.expose_writer_mem_view(BATCH_NBYTES)
        batch = None
        try:
            batch = _frame_from_ring_view(ring_view)
            if batch is None:
                _maybe_sleep()
                continue
            batch_mv, frame = batch
            for channel_index in range(CHANNELS):
                np.add(channel_bases[:, channel_index], phase, out=angle)
                if channel_index % 2 == 0:
                    np.sin(angle, out=frame[:, channel_index])
                else:
                    np.cos(angle, out=frame[:, channel_index])
            HEADER_STRUCT.pack_into(batch_mv, 0, time.perf_counter_ns())
            writer.inc_writer_pos(BATCH_NBYTES)
            phase += phase_step
        finally:
            if batch is not None:
                _, frame = batch
                del frame
            _release_ring_view(ring_view)


def dsp_consumer(
    pipeline_index: int,
    run_prefix: str,
    kernel_name: str,
    stop_at_ns: int,
    processed_bytes: "mp.sharedctypes.SynchronizedArray",
    batches_seen: "mp.sharedctypes.SynchronizedArray",
    latency_sum_ns: "mp.sharedctypes.SynchronizedArray",
    latency_min_ns: "mp.sharedctypes.SynchronizedArray",
    latency_max_ns: "mp.sharedctypes.SynchronizedArray",
    latency_samples_ns: "mp.sharedctypes.SynchronizedArray",
    latency_sample_count: "mp.sharedctypes.Synchronized",
) -> None:
    reader = pythusa.get_reader(_ring_name(run_prefix, kernel_name, pipeline_index))
    processor = make_benchmark_processor(
        kernel_name,
        rows=ROWS,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    reader.jump_to_writer()

    local_processed = 0
    local_batches = 0
    local_latency_sum = 0
    local_latency_min = np.uint64(np.iinfo(np.uint64).max)
    local_latency_max = np.uint64(0)
    last_token = 0.0

    while True:
        ring_view = None
        batch = None
        try:
            ring_view = reader.expose_reader_mem_view(BATCH_NBYTES)
            batch = _frame_from_ring_view(ring_view)
            if batch is None:
                if time.perf_counter_ns() >= stop_at_ns:
                    break
                _maybe_sleep()
                continue

            batch_mv, frame = batch
            sent_at_ns = HEADER_STRUCT.unpack_from(batch_mv, 0)[0]
            last_token = processor(frame)
            finished_at_ns = time.perf_counter_ns()
            latency_ns = finished_at_ns - sent_at_ns
            reader.inc_reader_pos(BATCH_NBYTES)

            local_processed += PAYLOAD_NBYTES
            local_batches += 1
            local_latency_sum += latency_ns
            local_latency_min = min(local_latency_min, np.uint64(latency_ns))
            local_latency_max = max(local_latency_max, np.uint64(latency_ns))

            if LATENCY_SAMPLE_STRIDE > 0 and (local_batches % LATENCY_SAMPLE_STRIDE) == 0:
                with latency_sample_count.get_lock():
                    sample_index = latency_sample_count.value
                    if sample_index < len(latency_samples_ns):
                        latency_samples_ns[sample_index] = latency_ns
                        latency_sample_count.value = sample_index + 1
        finally:
            if batch is not None:
                _, frame = batch
                del frame
            if ring_view is not None:
                _release_ring_view(ring_view)

    if math.isnan(last_token):  # pragma: no cover - benchmark data never produces NaN by design.
        raise RuntimeError("kernel produced NaN token")

    processed_bytes[pipeline_index] = local_processed
    batches_seen[pipeline_index] = local_batches
    latency_sum_ns[pipeline_index] = local_latency_sum
    latency_min_ns[pipeline_index] = 0 if local_batches == 0 else int(local_latency_min)
    latency_max_ns[pipeline_index] = int(local_latency_max)


def _sample_peak_task_rss_mb(manager: pythusa.Manager, task_names: tuple[str, ...]) -> float:
    total_rss_mb = 0.0
    for task_name in task_names:
        metrics = manager.get_metrics(task_name)
        if metrics is not None:
            total_rss_mb += metrics.memory_rss_mb
    return total_rss_mb


def _percentile_ms(samples_ns: np.ndarray, percentile: float) -> float:
    if samples_ns.size == 0:
        return 0.0
    return float(np.percentile(samples_ns / 1_000_000.0, percentile))


def run_kernel_benchmark(kernel_name: str) -> BenchmarkResult:
    ctx = mp.get_context("spawn")
    processed_bytes = ctx.Array("Q", PIPELINES)
    batches_seen = ctx.Array("Q", PIPELINES)
    latency_sum_ns = ctx.Array("Q", PIPELINES)
    latency_min_ns = ctx.Array("Q", [0] * PIPELINES)
    latency_max_ns = ctx.Array("Q", PIPELINES)
    latency_samples_ns = ctx.Array("Q", MAX_LATENCY_SAMPLES)
    latency_sample_count = ctx.Value("Q", 0)
    task_names: list[str] = []

    for pipeline_index in range(PIPELINES):
        latency_min_ns[pipeline_index] = np.iinfo(np.uint64).max

    ring_reserved_mb = (PIPELINES * BATCH_NBYTES * RING_DEPTH) / MEGABYTE
    peak_task_rss_mb = 0.0
    peak_parent_rss_mb = 0.0
    parent_process = psutil.Process()
    stop_at_ns = time.perf_counter_ns() + int(DURATION_S * 1_000_000_000)
    run_prefix = os.environ.get("DSP_BENCH_RUN_PREFIX", f"dsp{time.time_ns() & 0xfffffff:x}")

    with pythusa.Manager() as manager:
        for pipeline_index in range(PIPELINES):
            ring_name = _ring_name(run_prefix, kernel_name, pipeline_index)
            manager.create_ring(
                pythusa.RingSpec(
                    name=ring_name,
                    size=BATCH_NBYTES * RING_DEPTH,
                    num_readers=1,
                    cache_align=True,
                    cache_size=64,
                )
            )
            producer_name = f"{kernel_name}_producer_{pipeline_index}"
            consumer_name = f"{kernel_name}_consumer_{pipeline_index}"
            task_names.extend((producer_name, consumer_name))
            manager.create_task(
                pythusa.TaskSpec(
                    name=producer_name,
                    fn=signal_producer,
                    writing_rings=(ring_name,),
                    args=(pipeline_index, run_prefix, kernel_name, stop_at_ns),
                )
            )
            manager.create_task(
                pythusa.TaskSpec(
                    name=consumer_name,
                    fn=dsp_consumer,
                    reading_rings=(ring_name,),
                    args=(
                        pipeline_index,
                        run_prefix,
                        kernel_name,
                        stop_at_ns,
                        processed_bytes,
                        batches_seen,
                        latency_sum_ns,
                        latency_min_ns,
                        latency_max_ns,
                        latency_samples_ns,
                        latency_sample_count,
                    ),
                )
            )

        for pipeline_index in range(PIPELINES):
            manager.start(f"{kernel_name}_consumer_{pipeline_index}")
        time.sleep(0.1)
        for pipeline_index in range(PIPELINES):
            manager.start(f"{kernel_name}_producer_{pipeline_index}")
        manager.start_monitor(interval_s=MONITOR_INTERVAL_S)

        monitor_end = time.perf_counter() + DURATION_S + 1.0
        while time.perf_counter() < monitor_end:
            peak_task_rss_mb = max(peak_task_rss_mb, _sample_peak_task_rss_mb(manager, tuple(task_names)))
            peak_parent_rss_mb = max(
                peak_parent_rss_mb,
                parent_process.memory_info().rss / (1024 * 1024),
            )
            time.sleep(MONITOR_INTERVAL_S)

        for task_name in task_names:
            manager.join(task_name, timeout=0.5)

    total_processed = int(sum(processed_bytes))
    total_batches = int(sum(batches_seen))
    total_latency_sum = int(sum(latency_sum_ns))
    latency_min_values = [int(value) for value in latency_min_ns if int(value) > 0]
    latency_max_values = [int(value) for value in latency_max_ns]
    sample_count = min(int(latency_sample_count.value), MAX_LATENCY_SAMPLES)
    latency_samples = np.frombuffer(latency_samples_ns.get_obj(), dtype=np.uint64, count=sample_count)

    elapsed_s = max(DURATION_S, 1e-12)
    mean_latency_ms = (total_latency_sum / total_batches / 1_000_000.0) if total_batches else 0.0
    min_latency_ms = (min(latency_min_values) / 1_000_000.0) if latency_min_values else 0.0
    max_latency_ms = (max(latency_max_values) / 1_000_000.0) if latency_max_values else 0.0

    return BenchmarkResult(
        kernel=kernel_name,
        throughput_mb_s=total_processed / elapsed_s / MEGABYTE,
        batches=total_batches,
        payload_bytes=total_processed,
        latency_mean_ms=mean_latency_ms,
        latency_min_ms=min_latency_ms,
        latency_p50_ms=_percentile_ms(latency_samples, 50),
        latency_p95_ms=_percentile_ms(latency_samples, 95),
        latency_p99_ms=_percentile_ms(latency_samples, 99),
        latency_max_ms=max_latency_ms,
        ring_reserved_mb=ring_reserved_mb,
        peak_task_rss_mb=peak_task_rss_mb,
        peak_parent_rss_mb=peak_parent_rss_mb,
    )


def _print_header() -> None:
    print("PYTHUSA DSP Benchmark Suite")
    print(
        "config "
        f"rows={ROWS} channels={CHANNELS} pipelines={PIPELINES} "
        f"dtype={DTYPE.name} duration={DURATION_S:.1f}s ring_depth={RING_DEPTH} "
        f"mode={BENCHMARK_MODE} max_in_flight={MAX_IN_FLIGHT_BATCHES}"
    )
    print(
        "kernel            throughput   batches    mean_ms    p50_ms    p95_ms    p99_ms    max_ms   ring_mb  task_rss_mb parent_rss_mb"
    )


def _print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.kernel:16s}"
        f"{result.throughput_mb_s:11.2f}"
        f"{result.batches:10d}"
        f"{result.latency_mean_ms:11.3f}"
        f"{result.latency_p50_ms:10.3f}"
        f"{result.latency_p95_ms:10.3f}"
        f"{result.latency_p99_ms:10.3f}"
        f"{result.latency_max_ms:10.3f}"
        f"{result.ring_reserved_mb:10.1f}"
        f"{result.peak_task_rss_mb:13.1f}"
        f"{result.peak_parent_rss_mb:14.1f}"
    )


def _build_payload(results: list[BenchmarkResult], label: str | None) -> dict[str, object]:
    return build_payload(
        benchmark="dsp_benchmark_suite",
        label=label,
        config={
            "rows": ROWS,
            "channels": CHANNELS,
            "pipelines": PIPELINES,
            "dtype": DTYPE.name,
            "duration_s": DURATION_S,
            "ring_depth": RING_DEPTH,
            "mode": BENCHMARK_MODE,
            "max_in_flight_batches": MAX_IN_FLIGHT_BATCHES,
            "kernels": list(SELECTED_KERNELS),
        },
        results=results,
        notes=(
            "task_rss_mb is summed worker RSS and can overcount shared-memory mappings.",
        ),
    )


def main() -> None:
    args = _parse_args()
    _apply_cli_overrides(args)
    _validate_selected_kernels()
    if CHANNELS < 1:
        raise SystemExit("DSP_BENCH_CHANNELS must be at least 1.")
    if RING_DEPTH < 2:
        raise SystemExit("DSP_BENCH_RING_DEPTH must be at least 2.")
    if MAX_IN_FLIGHT_BATCHES < 0:
        raise SystemExit("DSP_BENCH_MAX_IN_FLIGHT_BATCHES must be >= 0.")
    if MAX_IN_FLIGHT_BATCHES >= RING_DEPTH:
        raise SystemExit("DSP_BENCH_MAX_IN_FLIGHT_BATCHES must be smaller than DSP_BENCH_RING_DEPTH.")
    results: list[BenchmarkResult] = []
    if not args.json:
        _print_header()
    for kernel_name in SELECTED_KERNELS:
        result = run_kernel_benchmark(kernel_name)
        results.append(result)
        if not args.json:
            _print_result(result)
    payload = _build_payload(results, args.label)
    emit_payload(payload, json_stdout=args.json, json_out=args.json_out)
    if not args.json:
        print("memory note: task_rss_mb is summed worker RSS and can overcount shared-memory mappings.")
        if args.json_out is not None:
            print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
