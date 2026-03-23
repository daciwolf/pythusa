from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

import pythusa


RING_PREFIX = os.environ.get("FFT_BENCH_RING_PREFIX", "fft_batches")
ROWS = int(os.environ.get("FFT_BENCH_ROWS", "8192"))
CONSUMER_COUNT = int(os.environ.get("FFT_BENCH_CONSUMERS", "4"))
COLS_PER_CONSUMER = int(os.environ.get("FFT_BENCH_COLS_PER_CONSUMER", "2"))
DTYPE = np.float64
ITEMSIZE = np.dtype(DTYPE).itemsize
PAIR_NBYTES = ROWS * COLS_PER_CONSUMER * ITEMSIZE
RING_DEPTH = int(os.environ.get("FFT_BENCH_RING_DEPTH", "256"))
IDLE_SLEEP_S = float(os.environ.get("FFT_BENCH_IDLE_SLEEP_S", "0.0000001"))
REPORT_INTERVAL_S = float(os.environ.get("FFT_BENCH_REPORT_INTERVAL_S", "0.5"))
MEGABYTE = 1_000_000.0
MAIN_IDLE_SLEEP_S = REPORT_INTERVAL_S
CONSUMER_STARTUP_S = float(os.environ.get("FFT_BENCH_CONSUMER_STARTUP_S", "0.1"))
BENCHMARK_DURATION_S = float(os.environ.get("FFT_BENCH_DURATION_S", "0"))
ENABLE_PERIODIC_REPORTS = os.environ.get("FFT_BENCH_PERIODIC_REPORTS", "1") != "0"


def _maybe_sleep() -> None:
    if IDLE_SLEEP_S > 0.0:
        time.sleep(IDLE_SLEEP_S)


def _ring_name(consumer_index: int) -> str:
    return f"{RING_PREFIX}_{consumer_index}"


def _ring_frame_from_view(ring_view: tuple[memoryview, memoryview | None, int, bool]) -> np.ndarray | None:
    mv1, _, size_ready, wrap_around = ring_view
    if size_ready < PAIR_NBYTES:
        return None
    # Each batch is exactly PAIR_NBYTES and the ring size is an integer
    # multiple of that batch size, so a single batch should always be contiguous.
    assert not wrap_around, "expected contiguous batch-sized ring view"
    return np.ndarray((ROWS, COLS_PER_CONSUMER), dtype=DTYPE, buffer=mv1)


def _release_ring_view(ring_view: tuple[memoryview, memoryview | None, int, bool]) -> None:
    mv1, mv2, _, _ = ring_view
    mv1.release()
    if mv2 is not None:
        mv2.release()


def _throughput_mb_s(owner: object, nbytes: int) -> float | None:
    now = time.perf_counter()
    if not hasattr(owner, "_report_started_at"):
        setattr(owner, "_report_started_at", now)
        setattr(owner, "_report_bytes", nbytes)
        return None

    bytes_seen = getattr(owner, "_report_bytes", 0) + nbytes
    started_at = getattr(owner, "_report_started_at")
    elapsed = now - started_at
    setattr(owner, "_report_bytes", bytes_seen)
    if elapsed < REPORT_INTERVAL_S:
        return None

    setattr(owner, "_report_started_at", now)
    setattr(owner, "_report_bytes", 0)
    return bytes_seen / elapsed / MEGABYTE


def signal_producer(consumer_index: int) -> None:
    writer = pythusa.get_writer(_ring_name(consumer_index))
    phase = 0.0
    freq = float(consumer_index + 1)
    start_col = consumer_index * COLS_PER_CONSUMER
    stop_col = start_col + COLS_PER_CONSUMER
    base_t = np.linspace(0.0, 2.0 * np.pi, ROWS, endpoint=False, dtype=DTYPE)
    first_base = freq * base_t
    second_base = (freq + 0.5) * base_t
    first_angles = np.empty(ROWS, dtype=DTYPE)
    second_angles = np.empty(ROWS, dtype=DTYPE)

    while True:
        writer_mem_view = writer.expose_writer_mem_view(PAIR_NBYTES)
        frame = None
        try:
            frame = _ring_frame_from_view(writer_mem_view)
            if frame is None:
                _maybe_sleep()
                continue
            np.add(first_base, phase, out=first_angles)
            np.sin(first_angles, out=frame[:, 0])
            np.subtract(second_base, phase, out=second_angles)
            np.cos(second_angles, out=frame[:, 1])
            writer.inc_writer_pos(PAIR_NBYTES)
        finally:
            if frame is not None:
                del frame
            _release_ring_view(writer_mem_view)

        batches_sent = getattr(signal_producer, "_batches_sent", 0) + 1
        signal_producer._batches_sent = batches_sent
        throughput_mb_s = _throughput_mb_s(signal_producer, PAIR_NBYTES)
        if ENABLE_PERIODIC_REPORTS and throughput_mb_s is not None:
            print(
                f"producer={consumer_index} cols={start_col}:{stop_col} batch={batches_sent} "
                f"phase={phase:.2f} throughput={throughput_mb_s:.2f} MB/s"
            )
        phase += 0.2


def fft_consumer(
    consumer_index: int,
    processed_bytes: "mp.sharedctypes.SynchronizedArray",
) -> None:
    reader = pythusa.get_reader(_ring_name(consumer_index))
    start_col = consumer_index * COLS_PER_CONSUMER
    stop_col = start_col + COLS_PER_CONSUMER
    # Drop any backlog written before this reader finished starting up.
    reader.jump_to_writer()

    while True:
        try:
            reader_mem_view = reader.expose_reader_mem_view(PAIR_NBYTES)
        except AssertionError as exc:
            if "max_amount_readable > ring_buffer_size" not in str(exc):
                raise
            reader.jump_to_writer()
            _maybe_sleep()
            continue
        frame = None
        try:
            frame = _ring_frame_from_view(reader_mem_view)
            if frame is None:
                _maybe_sleep()
                continue

            spectrum = np.fft.rfft(frame, axis=0)
            peak = float(np.abs(spectrum).max())
            reader.inc_reader_pos(PAIR_NBYTES)
        finally:
            if frame is not None:
                del frame
            _release_ring_view(reader_mem_view)
        with processed_bytes.get_lock():
            processed_bytes[consumer_index] += PAIR_NBYTES

        batches_seen = getattr(fft_consumer, "_batches_seen", 0) + 1
        fft_consumer._batches_seen = batches_seen
        throughput_mb_s = _throughput_mb_s(fft_consumer, PAIR_NBYTES)
        if ENABLE_PERIODIC_REPORTS and throughput_mb_s is not None:
            print(
                f"consumer={consumer_index} cols={start_col}:{stop_col} batch={batches_seen} "
                f"peak_magnitude={peak:.4f} throughput={throughput_mb_s:.2f} MB/s"
            )


def _report_total_fft_throughput(
    processed_bytes: "mp.sharedctypes.SynchronizedArray",
    total_bytes_seen: int,
    started_at: float,
) -> tuple[int, float]:
    now = time.perf_counter()
    elapsed = now - started_at
    if elapsed < REPORT_INTERVAL_S:
        return total_bytes_seen, started_at

    with processed_bytes.get_lock():
        current_total = int(sum(processed_bytes))
    delta_bytes = current_total - total_bytes_seen
    throughput_mb_s = delta_bytes / elapsed / MEGABYTE
    if ENABLE_PERIODIC_REPORTS:
        print(
            f"total_fft_bytes={current_total} "
            f"total_fft_throughput={throughput_mb_s:.2f} MB/s"
        )
    return current_total, now


def main() -> None:
    ctx = mp.get_context("spawn")
    processed_bytes = ctx.Array("Q", CONSUMER_COUNT)

    with pythusa.Manager() as manager:
        for consumer_index in range(CONSUMER_COUNT):
            ring_name = _ring_name(consumer_index)
            manager.create_ring(
                pythusa.RingSpec(
                    name=ring_name,
                    size=PAIR_NBYTES * RING_DEPTH,
                    num_readers=1,
                    cache_align=True,
                    cache_size=64,
                )
            )
            manager.create_task(
                pythusa.TaskSpec(
                    name=f"fft_consumer_{consumer_index}",
                    fn=fft_consumer,
                    reading_rings=(ring_name,),
                    args=(consumer_index, processed_bytes),
                )
            )
            manager.create_task(
                pythusa.TaskSpec(
                    name=f"signal_producer_{consumer_index}",
                    fn=signal_producer,
                    writing_rings=(ring_name,),
                    args=(consumer_index,),
                )
            )

        for consumer_index in range(CONSUMER_COUNT):
            manager.start(f"fft_consumer_{consumer_index}")
        time.sleep(CONSUMER_STARTUP_S)
        for consumer_index in range(CONSUMER_COUNT):
            manager.start(f"signal_producer_{consumer_index}")

        total_bytes_seen = 0
        benchmark_started_at = time.perf_counter()
        started_at = benchmark_started_at
        try:
            while True:
                if BENCHMARK_DURATION_S > 0.0 and (time.perf_counter() - benchmark_started_at) >= BENCHMARK_DURATION_S:
                    break
                time.sleep(MAIN_IDLE_SLEEP_S)
                total_bytes_seen, started_at = _report_total_fft_throughput(
                    processed_bytes,
                    total_bytes_seen,
                    started_at,
                )
        except KeyboardInterrupt:
            pass
        finally:
            with processed_bytes.get_lock():
                final_total = int(sum(processed_bytes))
            elapsed = max(time.perf_counter() - benchmark_started_at, 1e-12)
            print(
                f"final_total_fft_bytes={final_total} "
                f"final_total_fft_throughput={final_total / elapsed / MEGABYTE:.2f} MB/s "
                f"implementation=pythusa"
            )


if __name__ == "__main__":
    main()
