from __future__ import annotations

from dataclasses import dataclass, field
import os
import queue
import threading
import time

import numpy as np
from joblib import Parallel, delayed


ROWS = int(os.environ.get("FFT_BENCH_ROWS", "8192"))
CONSUMER_COUNT = int(os.environ.get("FFT_BENCH_CONSUMERS", "4"))
COLS_PER_CONSUMER = int(os.environ.get("FFT_BENCH_COLS_PER_CONSUMER", "2"))
DTYPE = np.float64
ITEMSIZE = np.dtype(DTYPE).itemsize
PAIR_NBYTES = ROWS * COLS_PER_CONSUMER * ITEMSIZE
PHASE_STEP = float(os.environ.get("FFT_BENCH_PHASE_STEP", "0.2"))
REPORT_INTERVAL_S = float(os.environ.get("FFT_BENCH_REPORT_INTERVAL_S", "0.5"))
MEGABYTE = 1_000_000.0
JOBLIB_PREFER = os.environ.get("FFT_BENCH_JOBLIB_PREFER", "threads")
JOBLIB_QUEUE_DEPTH = int(os.environ.get("FFT_BENCH_JOBLIB_QUEUE_DEPTH", "256"))
JOBLIB_QUEUE_TIMEOUT_S = float(os.environ.get("FFT_BENCH_JOBLIB_QUEUE_TIMEOUT_S", "0.01"))
JOBLIB_N_JOBS = int(os.environ.get("FFT_BENCH_JOBLIB_N_JOBS", str(CONSUMER_COUNT * 2)))
BENCHMARK_DURATION_S = float(os.environ.get("FFT_BENCH_DURATION_S", "0"))
ENABLE_PERIODIC_REPORTS = os.environ.get("FFT_BENCH_PERIODIC_REPORTS", "1") != "0"
PRINT_LOCK = threading.Lock()

BASE_T = np.linspace(0.0, 2.0 * np.pi, ROWS, endpoint=False, dtype=DTYPE)
FIRST_BASES = tuple((consumer_index + 1.0) * BASE_T for consumer_index in range(CONSUMER_COUNT))
SECOND_BASES = tuple((consumer_index + 1.5) * BASE_T for consumer_index in range(CONSUMER_COUNT))


@dataclass(slots=True)
class PipelineState:
    consumer_index: int
    start_col: int
    stop_col: int
    phase: float = 0.0
    batches_sent: int = 0
    batches_seen: int = 0
    producer_report_started_at: float = 0.0
    producer_report_bytes: int = 0
    consumer_report_started_at: float = 0.0
    consumer_report_bytes: int = 0


@dataclass(slots=True)
class SharedCounters:
    processed_bytes: list[int]
    lock: threading.Lock = field(default_factory=threading.Lock)


def _throughput_mb_s(state: PipelineState, role: str, nbytes: int) -> float | None:
    now = time.perf_counter()
    started_attr = f"{role}_report_started_at"
    bytes_attr = f"{role}_report_bytes"
    started_at = getattr(state, started_attr)
    if started_at == 0.0:
        setattr(state, started_attr, now)
        setattr(state, bytes_attr, nbytes)
        return None

    bytes_seen = getattr(state, bytes_attr) + nbytes
    elapsed = now - started_at
    setattr(state, bytes_attr, bytes_seen)
    if elapsed < REPORT_INTERVAL_S:
        return None

    setattr(state, started_attr, now)
    setattr(state, bytes_attr, 0)
    return bytes_seen / elapsed / MEGABYTE


def _make_pipeline_state(consumer_index: int) -> PipelineState:
    start_col = consumer_index * COLS_PER_CONSUMER
    stop_col = start_col + COLS_PER_CONSUMER
    return PipelineState(
        consumer_index=consumer_index,
        start_col=start_col,
        stop_col=stop_col,
    )


def _emit(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def _make_buffer_pool() -> tuple["queue.Queue[np.ndarray]", "queue.Queue[np.ndarray]"]:
    free_buffers: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=JOBLIB_QUEUE_DEPTH)
    filled_buffers: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=JOBLIB_QUEUE_DEPTH)
    for _ in range(JOBLIB_QUEUE_DEPTH):
        free_buffers.put(np.empty((ROWS, COLS_PER_CONSUMER), dtype=DTYPE))
    return free_buffers, filled_buffers


def signal_producer(
    state: PipelineState,
    free_buffers: "queue.Queue[np.ndarray]",
    filled_buffers: "queue.Queue[np.ndarray]",
    stop_event: threading.Event,
) -> None:
    first_base = FIRST_BASES[state.consumer_index]
    second_base = SECOND_BASES[state.consumer_index]
    first_angles = np.empty(ROWS, dtype=DTYPE)
    second_angles = np.empty(ROWS, dtype=DTYPE)

    while not stop_event.is_set():
        try:
            frame = free_buffers.get(timeout=JOBLIB_QUEUE_TIMEOUT_S)
        except queue.Empty:
            continue

        np.add(first_base, state.phase, out=first_angles)
        np.sin(first_angles, out=frame[:, 0])
        np.subtract(second_base, state.phase, out=second_angles)
        np.cos(second_angles, out=frame[:, 1])

        while True:
            try:
                filled_buffers.put(frame, timeout=JOBLIB_QUEUE_TIMEOUT_S)
                break
            except queue.Full:
                if stop_event.is_set():
                    free_buffers.put(frame)
                    return

        state.batches_sent += 1
        throughput_mb_s = _throughput_mb_s(state, "producer", PAIR_NBYTES)
        if ENABLE_PERIODIC_REPORTS and throughput_mb_s is not None:
            _emit(
                f"producer={state.consumer_index} cols={state.start_col}:{state.stop_col} "
                f"batch={state.batches_sent} phase={state.phase:.2f} "
                f"throughput={throughput_mb_s:.2f} MB/s"
            )
        state.phase += PHASE_STEP


def fft_consumer(
    state: PipelineState,
    free_buffers: "queue.Queue[np.ndarray]",
    filled_buffers: "queue.Queue[np.ndarray]",
    counters: SharedCounters,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set() or not filled_buffers.empty():
        try:
            frame = filled_buffers.get(timeout=JOBLIB_QUEUE_TIMEOUT_S)
        except queue.Empty:
            continue

        spectrum = np.fft.rfft(frame, axis=0)
        peak = float(np.abs(spectrum).max())
        with counters.lock:
            counters.processed_bytes[state.consumer_index] += PAIR_NBYTES

        state.batches_seen += 1
        throughput_mb_s = _throughput_mb_s(state, "consumer", PAIR_NBYTES)
        if ENABLE_PERIODIC_REPORTS and throughput_mb_s is not None:
            _emit(
                f"consumer={state.consumer_index} cols={state.start_col}:{state.stop_col} "
                f"batch={state.batches_seen} peak_magnitude={peak:.4f} "
                f"throughput={throughput_mb_s:.2f} MB/s"
            )
        free_buffers.put(frame)


def _report_total_fft_throughput(
    counters: SharedCounters,
    total_bytes_seen: int,
    started_at: float,
) -> tuple[int, float]:
    now = time.perf_counter()
    elapsed = now - started_at
    if elapsed < REPORT_INTERVAL_S:
        return total_bytes_seen, started_at

    with counters.lock:
        current_total = int(sum(counters.processed_bytes))
    delta_bytes = current_total - total_bytes_seen
    throughput_mb_s = delta_bytes / elapsed / MEGABYTE
    if ENABLE_PERIODIC_REPORTS:
        _emit(
            f"total_fft_bytes={current_total} "
            f"total_fft_throughput={throughput_mb_s:.2f} MB/s"
        )
    return current_total, now


def main() -> None:
    if JOBLIB_PREFER != "threads":
        raise SystemExit("This staged joblib benchmark currently supports FFT_BENCH_JOBLIB_PREFER=threads only.")
    if JOBLIB_QUEUE_DEPTH <= 0:
        raise SystemExit("FFT_BENCH_JOBLIB_QUEUE_DEPTH must be greater than 0.")
    if JOBLIB_N_JOBS < CONSUMER_COUNT * 2:
        raise SystemExit(f"FFT_BENCH_JOBLIB_N_JOBS must be at least {CONSUMER_COUNT * 2}.")

    states = [_make_pipeline_state(consumer_index) for consumer_index in range(CONSUMER_COUNT)]
    buffer_pools = [_make_buffer_pool() for _ in range(CONSUMER_COUNT)]
    counters = SharedCounters(processed_bytes=[0] * CONSUMER_COUNT)
    stop_event = threading.Event()
    parallel = Parallel(n_jobs=JOBLIB_N_JOBS, prefer=JOBLIB_PREFER)
    worker_error: list[BaseException] = []

    tasks = []
    for state, (free_buffers, filled_buffers) in zip(states, buffer_pools, strict=True):
        tasks.append(delayed(signal_producer)(state, free_buffers, filled_buffers, stop_event))
        tasks.append(delayed(fft_consumer)(state, free_buffers, filled_buffers, counters, stop_event))

    def run_parallel() -> None:
        try:
            parallel(tasks)
        except BaseException as exc:  # pragma: no cover - surfaced to main thread.
            worker_error.append(exc)
            stop_event.set()

    runner = threading.Thread(target=run_parallel, name="joblib-runner", daemon=True)
    benchmark_started_at = time.perf_counter()
    runner.start()
    total_bytes_seen = 0
    started_at = benchmark_started_at

    try:
        while runner.is_alive():
            time.sleep(REPORT_INTERVAL_S)
            total_bytes_seen, started_at = _report_total_fft_throughput(
                counters,
                total_bytes_seen,
                started_at,
            )
            if BENCHMARK_DURATION_S > 0.0 and (time.perf_counter() - benchmark_started_at) >= BENCHMARK_DURATION_S:
                stop_event.set()
                break
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        runner.join()
        if worker_error:
            raise worker_error[0]

        with counters.lock:
            final_total = int(sum(counters.processed_bytes))
        elapsed = max(time.perf_counter() - benchmark_started_at, 1e-12)
        _emit(
            f"final_total_fft_bytes={final_total} "
            f"final_total_fft_throughput={final_total / elapsed / MEGABYTE:.2f} MB/s "
            f"implementation=joblib prefer={JOBLIB_PREFER} "
            f"queue_depth={JOBLIB_QUEUE_DEPTH} n_jobs={JOBLIB_N_JOBS}"
        )


if __name__ == "__main__":
    main()
