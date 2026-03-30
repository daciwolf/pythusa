from __future__ import annotations

"""Run with: python examples/fir128_scaling_pipeline.py"""

import argparse
import csv
import multiprocessing as mp
import os
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Empty

import matplotlib.pyplot as plt
import numpy as np

import pythusa


ROOT = Path(__file__).resolve().parents[1]
ENGINE_DATA_DIR = ROOT / "engine_data"
GSE_PATH = ENGINE_DATA_DIR / "gse-vtf-5.csv"
ECU_PATH = ENGINE_DATA_DIR / "ecu_vtf5.csv"

META_FIELDS = 2
FIR_TAP_COUNT = 128
DEFAULT_FRAME_LENGTH = 256
DEFAULT_FRAME_SIZES = (128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384)
DEFAULT_SEGMENT_DURATION_S = 4.0
DEFAULT_MIN_RATE_HZ = 1_000
DEFAULT_MAX_RATE_HZ = 1_024_000
DEFAULT_MIN_BENCHMARK_FRAMES = 4
IDLE_SLEEP_S = 0
HEATMAP_CMAP = "viridis"

FIR128_TAPS = np.hanning(FIR_TAP_COUNT).astype(np.float32)
FIR128_TAPS /= FIR128_TAPS.sum()

_TELEMETRY_CACHE: dict[str, object] | None = None


def _maybe_sleep() -> None:
    if IDLE_SLEEP_S > 0.0:
        time.sleep(IDLE_SLEEP_S)


def _write_exact(writer, array: np.ndarray) -> None:
    while not writer.write(array):
        _maybe_sleep()


def _release_ring_view(ring_view) -> None:
    mv1, mv2, _, _ = ring_view
    mv1.release()
    if mv2 is not None:
        mv2.release()


def _raw_try_read_into(ring, dst_view: memoryview, nbytes: int) -> bool:
    ring_view = ring.expose_reader_mem_view(nbytes)
    try:
        if ring_view[2] < nbytes:
            return False
        ring.simple_read(ring_view, dst_view)
        ring.inc_reader_pos(nbytes)
        return True
    finally:
        _release_ring_view(ring_view)


def _raw_write_exact(ring, src_view: memoryview, nbytes: int) -> None:
    while True:
        ring_view = ring.expose_writer_mem_view(nbytes)
        try:
            if ring_view[2] < nbytes:
                _maybe_sleep()
                continue
            ring.simple_write(ring_view, src_view)
            ring.inc_writer_pos(nbytes)
            return
        finally:
            _release_ring_view(ring_view)


def _active_reader_count(ring) -> int:
    active = 0
    for slot in range(6, len(ring.header), 3):
        active += int(ring.header[slot + 1] != 0)
    return active


def _wait_for_active_readers(*rings, timeout_s: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while True:
        if all(_active_reader_count(ring) >= 1 for ring in rings):
            return
        if time.perf_counter() >= deadline:
            raise RuntimeError("Timed out waiting for downstream readers to become active.")
        time.sleep(0.001)


def source_signal(
    source_frames,
    source_meta,
    *,
    frames: np.ndarray,
    frame_interval_s: float | None,
    startup_delay_s: float,
) -> None:
    if startup_delay_s > 0.0:
        time.sleep(startup_delay_s)

    source_frame_ring = source_frames.raw
    source_meta_ring = source_meta.raw
    _wait_for_active_readers(source_frame_ring, source_meta_ring)
    meta = np.empty((META_FIELDS,), dtype=np.float64)
    meta_view = memoryview(meta).cast("B")
    next_emit_at = time.perf_counter() if frame_interval_s is not None else None

    for frame_index, frame in enumerate(frames):
        if next_emit_at is not None and frame_interval_s is not None:
            while True:
                remaining_s = next_emit_at - time.perf_counter()
                if remaining_s <= 0.0:
                    break
                time.sleep(min(remaining_s, 0.001))

        meta[0] = float(frame_index)
        meta[1] = time.perf_counter()
        _raw_write_exact(source_meta_ring, meta_view, meta.nbytes)
        _raw_write_exact(source_frame_ring, memoryview(frame).cast("B"), frame.nbytes)
        if next_emit_at is not None and frame_interval_s is not None:
            next_emit_at += frame_interval_s


def split_round_robin(
    source_frames,
    source_meta,
    *,
    frame_length: int,
    total_frames: int,
    worker_count: int,
    **kwargs,
) -> None:
    source_frame_ring = source_frames.raw
    source_meta_ring = source_meta.raw
    worker_frame_rings = [kwargs[f"worker_{index}_frames"].raw for index in range(worker_count)]
    worker_meta_rings = [kwargs[f"worker_{index}_meta"].raw for index in range(worker_count)]
    _wait_for_active_readers(*worker_frame_rings, *worker_meta_rings)

    frame = np.empty((frame_length,), dtype=np.float32)
    frame_view = memoryview(frame).cast("B")
    meta = np.empty((META_FIELDS,), dtype=np.float64)
    meta_view = memoryview(meta).cast("B")
    pending_meta: np.ndarray | None = None
    routed = 0

    while routed < total_frames:
        if pending_meta is None:
            if _raw_try_read_into(source_meta_ring, meta_view, meta.nbytes):
                pending_meta = np.array(meta, copy=True)
            else:
                _maybe_sleep()
                continue

        if not _raw_try_read_into(source_frame_ring, frame_view, frame.nbytes):
            _maybe_sleep()
            continue

        target = int(round(float(pending_meta[0]))) % worker_count
        _raw_write_exact(
            worker_meta_rings[target],
            memoryview(pending_meta).cast("B"),
            pending_meta.nbytes,
        )
        _raw_write_exact(worker_frame_rings[target], frame_view, frame.nbytes)
        pending_meta = None
        routed += 1


def fir128_worker(
    worker_frames,
    worker_meta,
    result_frames,
    result_meta,
    *,
    frame_length: int,
    expected_frames: int,
    fir_fft_size: int,
    fir_same_start: int,
    tap_spectrum: np.ndarray,
) -> None:
    worker_frame_ring = worker_frames.raw
    worker_meta_ring = worker_meta.raw
    result_frame_ring = result_frames.raw
    result_meta_ring = result_meta.raw
    _wait_for_active_readers(result_frame_ring, result_meta_ring)
    frame = np.empty((frame_length,), dtype=np.float32)
    frame_view = memoryview(frame).cast("B")
    filtered = np.empty((frame_length,), dtype=np.float32)
    filtered_view = memoryview(filtered).cast("B")
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

        spectrum = np.fft.rfft(frame, n=fir_fft_size)
        spectrum *= tap_spectrum
        filtered_full = np.fft.irfft(spectrum, n=fir_fft_size)
        filtered[:] = filtered_full[fir_same_start:fir_same_start + frame_length]
        _raw_write_exact(result_meta_ring, memoryview(pending_meta).cast("B"), pending_meta.nbytes)
        _raw_write_exact(result_frame_ring, filtered_view, filtered.nbytes)
        pending_meta = None
        processed += 1


def aggregate_results(
    *,
    frame_length: int,
    result_frame_length: int | None = None,
    total_frames: int,
    worker_count: int,
    result_queue,
    **kwargs,
) -> None:
    frame_rings = [kwargs[f"result_{index}_frames"].raw for index in range(worker_count)]
    meta_rings = [kwargs[f"result_{index}_meta"].raw for index in range(worker_count)]
    payload_frame_length = frame_length if result_frame_length is None else result_frame_length

    frame_buffers = [
        np.empty((payload_frame_length,), dtype=np.float32)
        for _ in range(worker_count)
    ]
    meta_buffers = [
        np.empty((META_FIELDS,), dtype=np.float64)
        for _ in range(worker_count)
    ]
    pending_meta: list[np.ndarray | None] = [None] * worker_count
    frame_views = [memoryview(buffer).cast("B") for buffer in frame_buffers]
    meta_views = [memoryview(buffer).cast("B") for buffer in meta_buffers]

    latencies_ms: list[float] = []
    bytes_processed = 0
    completed = 0
    first_emitted_at: float | None = None
    finished_at = 0.0

    while completed < total_frames:
        made_progress = False

        for worker_index in range(worker_count):
            if pending_meta[worker_index] is None:
                if _raw_try_read_into(meta_rings[worker_index], meta_views[worker_index], meta_buffers[worker_index].nbytes):
                    pending_meta[worker_index] = np.array(meta_buffers[worker_index], copy=True)
                    made_progress = True

            worker_meta = pending_meta[worker_index]
            if worker_meta is None:
                continue

            if not _raw_try_read_into(frame_rings[worker_index], frame_views[worker_index], frame_buffers[worker_index].nbytes):
                continue

            arrived_at = time.perf_counter()
            emitted_at = float(worker_meta[1])
            if first_emitted_at is None:
                first_emitted_at = emitted_at

            latencies_ms.append((arrived_at - emitted_at) * 1_000.0)
            bytes_processed += frame_buffers[worker_index].nbytes
            pending_meta[worker_index] = None
            completed += 1
            finished_at = arrived_at
            made_progress = True

        if not made_progress:
            _maybe_sleep()

    if first_emitted_at is None:
        raise RuntimeError("Aggregator completed without receiving any frames.")

    elapsed_s = max(finished_at - first_emitted_at, 1e-9)
    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    total_samples = total_frames * frame_length

    result_queue.put(
        {
            "elapsed_s": elapsed_s,
            "result_payload_mb_s": bytes_processed / elapsed_s / 1_000_000.0,
            "input_equivalent_mb_s": total_samples * np.dtype(np.float32).itemsize / elapsed_s / 1_000_000.0,
            "ksps": total_samples / elapsed_s / 1_000.0,
            "latency_mean_ms": float(np.mean(latency_array)),
            "latency_p95_ms": float(np.percentile(latency_array, 95.0)),
            "total_frames": total_frames,
            "total_samples": total_samples,
        }
    )


@dataclass(frozen=True)
class FirPlan:
    frame_length: int
    fft_size: int
    same_start: int
    tap_spectrum: np.ndarray


@dataclass(frozen=True)
class RunResult:
    sample_rate_hz: int
    frame_length: int
    worker_count: int
    paced_source: bool
    total_frames: int
    total_samples: int
    throughput_mb_s: float
    result_payload_mb_s: float
    required_throughput_mb_s: float
    ksps: float
    target_ksps: float
    realtime_ratio: float
    margin_pct: float
    keeps_up: bool
    latency_mean_ms: float
    latency_p95_ms: float
    elapsed_s: float


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    max_power = 1
    while max_power * 2 <= cpu_count and max_power * 2 <= 8:
        max_power *= 2
    return max_power


def _default_frame_sizes_csv() -> str:
    return ",".join(str(size) for size in DEFAULT_FRAME_SIZES)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a split FIR128 PYTHUSA pipeline across sample-rate, frame-size, "
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
        help="Maximum number of FIR workers to try. Rounded down to powers of two.",
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


def _read_numeric_csv(
    path: Path,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows: list[tuple[float, list[float]]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            timestamp = float(row["time_recv"])
            values = [float(row[column]) for column in columns]
            rows.append((timestamp, values))

    rows.sort(key=lambda item: item[0])
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    value_matrix = np.asarray([row[1] for row in rows], dtype=np.float64)

    unique_times, unique_index = np.unique(times, return_index=True)
    series = {
        column: value_matrix[unique_index, column_index]
        for column_index, column in enumerate(columns)
    }
    return unique_times, series


def _normalize(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    centered = series - np.mean(series)
    scale = np.std(centered)
    if scale <= 0.0:
        return np.zeros_like(centered)
    return centered / scale


def _telemetry_bundle() -> dict[str, object]:
    global _TELEMETRY_CACHE
    if _TELEMETRY_CACHE is not None:
        return _TELEMETRY_CACHE

    gse_columns = ("pressuregn2", "pressurevent", "pressureloxinjtee", "pressureloxmvas")
    ecu_columns = ("pressurelox", "pressurelng", "pressureinjectorlox", "pressureinjectorlng")

    gse_time, gse = _read_numeric_csv(GSE_PATH, gse_columns)
    ecu_time, ecu = _read_numeric_csv(ECU_PATH, ecu_columns)

    overlap_start = max(gse_time[0], ecu_time[0])
    overlap_end = min(gse_time[-1], ecu_time[-1])
    overlap_mask = (ecu_time >= overlap_start) & (ecu_time <= overlap_end)

    injector_activity = (
        np.asarray(ecu["pressureinjectorlox"])[overlap_mask]
        + np.asarray(ecu["pressureinjectorlng"])[overlap_mask]
    )
    active_index = int(np.argmax(injector_activity))
    active_time = ecu_time[overlap_mask][active_index]

    _TELEMETRY_CACHE = {
        "gse_time": gse_time,
        "gse": gse,
        "ecu_time": ecu_time,
        "ecu": ecu,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "active_time": active_time,
    }
    return _TELEMETRY_CACHE


def _segment_bounds(segment_duration_s: float) -> tuple[float, float]:
    bundle = _telemetry_bundle()
    overlap_start = float(bundle["overlap_start"])
    overlap_end = float(bundle["overlap_end"])
    active_time = float(bundle["active_time"])

    half_window = segment_duration_s * 0.5
    segment_start = max(overlap_start, active_time - half_window)
    segment_end = segment_start + segment_duration_s
    if segment_end > overlap_end:
        segment_end = overlap_end
        segment_start = segment_end - segment_duration_s
    return segment_start, segment_end


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    if any(value <= 0 for value in values):
        raise ValueError("all values must be positive integers")
    return values


def _power_of_two_counts(max_workers: int) -> list[int]:
    counts: list[int] = []
    current = 1
    while current <= max_workers:
        counts.append(current)
        current *= 2
    return counts


def _rate_ladder(min_rate_hz: int, max_rate_hz: int) -> list[int]:
    rates: list[int] = []
    current = min_rate_hz
    while current <= max_rate_hz:
        rates.append(current)
        current *= 2
    return rates


def _make_fir_plan(frame_length: int) -> FirPlan:
    fft_size = 1 << (frame_length + FIR128_TAPS.size - 2).bit_length()
    same_start = (FIR128_TAPS.size - 1) // 2
    tap_spectrum = np.fft.rfft(FIR128_TAPS, n=fft_size).astype(np.complex64, copy=False)
    return FirPlan(
        frame_length=frame_length,
        fft_size=fft_size,
        same_start=same_start,
        tap_spectrum=tap_spectrum,
    )


def _engine_signal_frames(
    sample_rate_hz: int,
    segment_duration_s: float,
    *,
    frame_length: int,
    min_benchmark_frames: int,
    seed: int = 5,
) -> np.ndarray:
    bundle = _telemetry_bundle()
    gse_time = bundle["gse_time"]
    ecu_time = bundle["ecu_time"]
    gse = bundle["gse"]
    ecu = bundle["ecu"]

    segment_start, _segment_end = _segment_bounds(segment_duration_s)
    total_samples = int(sample_rate_hz * segment_duration_s)
    target_times = segment_start + (
        np.arange(total_samples, dtype=np.float64) / float(sample_rate_hz)
    )

    pressuregn2 = np.interp(target_times, gse_time, gse["pressuregn2"])
    pressurevent = np.interp(target_times, gse_time, gse["pressurevent"])
    pressureloxinjtee = np.interp(target_times, gse_time, gse["pressureloxinjtee"])
    pressureloxmvas = np.interp(target_times, gse_time, gse["pressureloxmvas"])
    pressurelox = np.interp(target_times, ecu_time, ecu["pressurelox"])
    pressurelng = np.interp(target_times, ecu_time, ecu["pressurelng"])
    pressureinjectorlox = np.interp(target_times, ecu_time, ecu["pressureinjectorlox"])
    pressureinjectorlng = np.interp(target_times, ecu_time, ecu["pressureinjectorlng"])

    rng = np.random.default_rng(seed + sample_rate_hz + frame_length)
    injector_delta = pressureinjectorlng - pressureinjectorlox
    tank_delta = pressurelng - pressurelox
    gse_drive = _normalize(pressuregn2 + pressureloxmvas - pressurevent)
    injector_drive = np.abs(_normalize(injector_delta))
    dynamic_drive = _normalize(np.gradient(injector_delta + 0.2 * tank_delta))

    carrier_hz = 1_800.0 + 220.0 * gse_drive + 90.0 * dynamic_drive
    phase = 2.0 * np.pi * np.cumsum(carrier_hz) / float(sample_rate_hz)

    signal = (
        injector_delta
        + 0.35 * tank_delta
        + 0.18 * pressureloxinjtee
        + (0.30 + 0.45 * injector_drive) * np.sin(phase)
        + 0.22 * np.sin(1.9 * phase + 0.25)
        + 0.08 * rng.standard_normal(total_samples)
    ).astype(np.float32, copy=False)

    target_benchmark_samples = max(total_samples, frame_length * min_benchmark_frames)
    if signal.shape[0] < target_benchmark_samples:
        repeats = (target_benchmark_samples + signal.shape[0] - 1) // signal.shape[0]
        signal = np.tile(signal, repeats)

    usable = target_benchmark_samples - (target_benchmark_samples % frame_length)
    return signal[:usable].reshape(-1, frame_length)


def _expected_worker_frames(total_frames: int, worker_index: int, worker_count: int) -> int:
    if worker_index >= total_frames:
        return 0
    return ((total_frames - 1 - worker_index) // worker_count) + 1


def _collect_result(result_queue: mp.queues.Queue) -> dict[str, float]:
    try:
        return result_queue.get(timeout=120.0)
    except Empty as exc:
        raise RuntimeError("Timed out waiting for FIR128 benchmark result.") from exc


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
    fir_plan = _make_fir_plan(frame_length)
    frame_interval_s = (frame_length / float(sample_rate_hz)) if pace_source else None
    startup_delay_s = 0.0 if pace_source else 0.1
    run_prefix = f"{'fir128'}_{uuid.uuid4().hex[:10]}"

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
        total_frames=total_frames,
        worker_count=worker_count,
        result_queue=result_queue,
    )

    with pythusa.Pipeline(f"fir128-{sample_rate_hz}-{frame_length}-{worker_count}") as pipe:
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
            pipe.add_stream(result_frame_name, shape=(frame_length,), dtype=np.float32, cache_align=False)
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
                f"fir128_{worker_index}",
                fn=partial(
                    fir128_worker,
                    frame_length=frame_length,
                    expected_frames=_expected_worker_frames(
                        total_frames,
                        worker_index,
                        worker_count,
                    ),
                    fir_fft_size=fir_plan.fft_size,
                    fir_same_start=fir_plan.same_start,
                    tap_spectrum=fir_plan.tap_spectrum,
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


def _format_hz(rate_hz: int) -> str:
    if rate_hz >= 1_000_000:
        return f"{rate_hz / 1_000_000:.3g}M"
    if rate_hz >= 1_000:
        return f"{rate_hz / 1_000:.3g}k"
    return str(rate_hz)


def _print_run_result(result: RunResult) -> None:
    returned_suffix = ""
    if not np.isclose(result.result_payload_mb_s, result.throughput_mb_s):
        returned_suffix = f" returned={result.result_payload_mb_s:>8.2f} MB/s"

    if result.paced_source:
        status = "KEEPS_UP" if result.keeps_up else "CAPS_OUT"
        print(
            f"rate={result.sample_rate_hz:>9,d} Hz "
            f"frame={result.frame_length:>6,d} "
            f"workers={result.worker_count:>2d} "
            f"ksps={result.ksps:>10,.2f} "
            f"margin={result.margin_pct:+7.2f}% "
            f"throughput={result.throughput_mb_s:>8.2f} MB/s "
            f"latency_mean={result.latency_mean_ms:>8.3f} ms "
            f"latency_p95={result.latency_p95_ms:>8.3f} ms "
            f"status={status}"
            f"{returned_suffix}"
        )
        return

    print(
        f"frame={result.frame_length:>6,d} "
        f"workers={result.worker_count:>2d} "
        f"mode=UNCAPPED "
        f"ksps={result.ksps:>10,.2f} "
        f"throughput={result.throughput_mb_s:>8.2f} MB/s "
        f"latency_mean={result.latency_mean_ms:>8.3f} ms "
        f"latency_p95={result.latency_p95_ms:>8.3f} ms"
        f"{returned_suffix}"
    )


def _best_by_rate(results: list[RunResult]) -> dict[int, RunResult]:
    best: dict[int, RunResult] = {}
    for result in results:
        current = best.get(result.sample_rate_hz)
        if current is None or result.throughput_mb_s > current.throughput_mb_s:
            best[result.sample_rate_hz] = result
    return best


def _highest_sustained_rate(best_rate_results: dict[int, RunResult]) -> RunResult | None:
    sustained = [result for result in best_rate_results.values() if result.keeps_up]
    if not sustained:
        return None
    return max(sustained, key=lambda item: item.sample_rate_hz)


def _first_failed_rate(best_rate_results: dict[int, RunResult]) -> RunResult | None:
    failed = [result for result in best_rate_results.values() if not result.keeps_up]
    if not failed:
        return None
    return min(failed, key=lambda item: item.sample_rate_hz)


def _matrix_from_results(
    results: list[RunResult],
    row_values: list[int],
    col_values: list[int],
    *,
    row_attr: str,
    col_attr: str,
    value_attr: str,
) -> np.ndarray:
    matrix = np.full((len(row_values), len(col_values)), np.nan, dtype=np.float64)
    row_index = {value: index for index, value in enumerate(row_values)}
    col_index = {value: index for index, value in enumerate(col_values)}

    for result in results:
        matrix[row_index[getattr(result, row_attr)], col_index[getattr(result, col_attr)]] = getattr(
            result,
            value_attr,
        )
    return matrix


def _result_cell(
    result: RunResult,
    row_values: list[int],
    col_values: list[int],
    *,
    row_attr: str,
    col_attr: str,
) -> tuple[int, int]:
    return (
        row_values.index(getattr(result, row_attr)),
        col_values.index(getattr(result, col_attr)),
    )


def _annotate_heatmap(ax, matrix: np.ndarray, *, fmt: str) -> None:
    finite = matrix[np.isfinite(matrix)]
    threshold = np.nanmean(finite) if finite.size else 0.0

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not np.isfinite(value):
                continue
            color = "black" if value >= threshold else "white"
            ax.text(
                col_index,
                row_index,
                format(value, fmt),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )


def _draw_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xticklabels: list[str],
    yticklabels: list[str],
    cmap: str,
    colorbar_label: str,
    annotation_fmt: str,
) -> None:
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    _annotate_heatmap(ax, matrix, fmt=annotation_fmt)
    plt.colorbar(image, ax=ax, label=colorbar_label)


def _plot_sweeps(
    *,
    title_label: str,
    rate_results: list[RunResult],
    frame_results: list[RunResult],
    rate_values: list[int],
    rate_frame_length: int,
    frame_values: list[int],
    worker_values: list[int],
    frame_sweep_rate_hz: int,
    highest_sustained: RunResult | None,
    best_throughput: RunResult,
    lowest_latency: RunResult,
    plot_out: Path | None,
    no_show: bool,
) -> None:
    throughput_by_rate = _matrix_from_results(
        rate_results,
        rate_values,
        worker_values,
        row_attr="sample_rate_hz",
        col_attr="worker_count",
        value_attr="throughput_mb_s",
    )
    ratio_by_rate = _matrix_from_results(
        rate_results,
        rate_values,
        worker_values,
        row_attr="sample_rate_hz",
        col_attr="worker_count",
        value_attr="realtime_ratio",
    )
    throughput_by_frame = _matrix_from_results(
        frame_results,
        frame_values,
        worker_values,
        row_attr="frame_length",
        col_attr="worker_count",
        value_attr="throughput_mb_s",
    )
    latency_by_frame = _matrix_from_results(
        frame_results,
        frame_values,
        worker_values,
        row_attr="frame_length",
        col_attr="worker_count",
        value_attr="latency_mean_ms",
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    rate_labels = [_format_hz(rate) for rate in rate_values]
    frame_labels = [f"{frame:,}" for frame in frame_values]
    worker_labels = [str(worker) for worker in worker_values]

    _draw_heatmap(
        axes[0, 0],
        throughput_by_rate,
        title=(
            "Input-Equivalent Throughput vs Sample Rate and Worker Count\n"
            f"Frame Size {rate_frame_length:,}"
        ),
        xlabel="Workers",
        ylabel="Sample Rate (Hz)",
        xticklabels=worker_labels,
        yticklabels=rate_labels,
        cmap=HEATMAP_CMAP,
        colorbar_label="Input MB/s",
        annotation_fmt=".1f",
    )
    _draw_heatmap(
        axes[0, 1],
        ratio_by_rate,
        title=(
            "Real-Time Headroom vs Sample Rate and Worker Count\n"
            f"Frame Size {rate_frame_length:,}"
        ),
        xlabel="Workers",
        ylabel="Sample Rate (Hz)",
        xticklabels=worker_labels,
        yticklabels=rate_labels,
        cmap=HEATMAP_CMAP,
        colorbar_label="Achieved / Target",
        annotation_fmt=".2f",
    )
    _draw_heatmap(
        axes[1, 0],
        throughput_by_frame,
        title="Uncapped Input-Equivalent Throughput vs Frame Size and Worker Count",
        xlabel="Workers",
        ylabel="Frame Length",
        xticklabels=worker_labels,
        yticklabels=frame_labels,
        cmap=HEATMAP_CMAP,
        colorbar_label="Input MB/s",
        annotation_fmt=".1f",
    )
    _draw_heatmap(
        axes[1, 1],
        latency_by_frame,
        title="Uncapped Mean Latency vs Frame Size and Worker Count",
        xlabel="Workers",
        ylabel="Frame Length",
        xticklabels=worker_labels,
        yticklabels=frame_labels,
        cmap=HEATMAP_CMAP,
        colorbar_label="ms",
        annotation_fmt=".2f",
    )

    if rate_results:
        rate_best_throughput = max(rate_results, key=lambda item: item.throughput_mb_s)
        rate_best_cell = _result_cell(
            rate_best_throughput,
            rate_values,
            worker_values,
            row_attr="sample_rate_hz",
            col_attr="worker_count",
        )
        axes[0, 0].scatter(rate_best_cell[1], rate_best_cell[0], marker="*", s=220, c="red", edgecolors="white")

    frame_best_throughput = max(frame_results, key=lambda item: item.throughput_mb_s)
    frame_best_cell = _result_cell(
        frame_best_throughput,
        frame_values,
        worker_values,
        row_attr="frame_length",
        col_attr="worker_count",
    )
    axes[1, 0].scatter(frame_best_cell[1], frame_best_cell[0], marker="*", s=220, c="red", edgecolors="white")

    latency_best_cell = _result_cell(
        lowest_latency,
        frame_values,
        worker_values,
        row_attr="frame_length",
        col_attr="worker_count",
    )
    axes[1, 1].scatter(latency_best_cell[1], latency_best_cell[0], marker="*", s=220, c="cyan", edgecolors="black")

    if highest_sustained is not None:
        sustained_cell = _result_cell(
            highest_sustained,
            rate_values,
            worker_values,
            row_attr="sample_rate_hz",
            col_attr="worker_count",
        )
        axes[0, 1].scatter(
            sustained_cell[1],
            sustained_cell[0],
            marker="o",
            s=180,
            facecolors="none",
            edgecolors="white",
            linewidths=2.0,
        )

    if highest_sustained is not None:
        summary_title = (
            f"PYTHUSA {title_label} scaling sweep\n"
            f"Max processed throughput {best_throughput.throughput_mb_s:.2f} MB/s | "
            f"Lowest mean latency {lowest_latency.latency_mean_ms:.3f} ms | "
            f"Max sustained rate {highest_sustained.sample_rate_hz:,} Hz"
        )
    else:
        summary_title = (
            f"PYTHUSA {title_label} scaling sweep\n"
            f"Max processed throughput {best_throughput.throughput_mb_s:.2f} MB/s | "
            f"Lowest mean latency {lowest_latency.latency_mean_ms:.3f} ms"
        )
    figure.suptitle(summary_title, fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if plot_out is not None:
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_out, dpi=180)
        print(f"saved heatmaps to {plot_out}")

    if no_show:
        plt.close(figure)
    else:
        plt.show()


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

    print("starting FIR128 scaling example")
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
        title_label="FIR128",
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
