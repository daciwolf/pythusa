from __future__ import annotations

import time

import numpy as np

import pythusa


RING_NAME = "counter_stream"
DTYPE = np.float64
BATCH = 16
NBYTES = BATCH * np.dtype(DTYPE).itemsize
IDLE_SLEEP_S = 0.001
REPORT_INTERVAL_S = 0.5
MEGABYTE = 1_000_000.0
MAIN_IDLE_SLEEP_S = 1.0


def _consume_exact(reader: pythusa.SharedRingBuffer, dst: memoryview, nbytes: int) -> bool:
    reader_mem_view = reader.expose_reader_mem_view(nbytes)
    if reader_mem_view[2] < nbytes:
        return False
    reader.simple_read(reader_mem_view, dst)
    reader.inc_reader_pos(nbytes)
    return True


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


def producer() -> None:
    writer = pythusa.get_writer(RING_NAME)
    offset = 0
    while True:
        payload = np.arange(offset, offset + BATCH, dtype=DTYPE)
        written = writer.write_array(payload)
        if written == 0:
            time.sleep(IDLE_SLEEP_S)
            continue

        batches_sent = getattr(producer, "_batches_sent", 0) + 1
        producer._batches_sent = batches_sent
        throughput_mb_s = _throughput_mb_s(producer, written)
        if throughput_mb_s is not None:
            print(
                f"producer_batch={batches_sent} first={payload[0]:.0f} "
                f"last={payload[-1]:.0f} throughput={throughput_mb_s:.2f} MB/s"
            )
        offset += BATCH


def consumer() -> None:
    reader = pythusa.get_reader(RING_NAME)
    batch = np.empty(BATCH, dtype=DTYPE)
    batch_bytes = memoryview(batch).cast("B")

    while True:
        if not _consume_exact(reader, batch_bytes, NBYTES):
            time.sleep(IDLE_SLEEP_S)
            continue

        batches_seen = getattr(consumer, "_batches_seen", 0) + 1
        consumer._batches_seen = batches_seen
        throughput_mb_s = _throughput_mb_s(consumer, NBYTES)
        if throughput_mb_s is not None:
            print(
                f"consumer_batch={batches_seen} first={batch[0]:.0f} "
                f"last={batch[-1]:.0f} mean={batch.mean():.1f} "
                f"throughput={throughput_mb_s:.2f} MB/s"
            )


def main() -> None:
    with pythusa.Manager() as manager:
        manager.create_ring(pythusa.RingSpec(name=RING_NAME, size=NBYTES * 32, num_readers=1))
        manager.create_task(pythusa.TaskSpec(name="consumer", fn=consumer, reading_rings=(RING_NAME,)))
        manager.create_task(pythusa.TaskSpec(name="producer", fn=producer, writing_rings=(RING_NAME,)))
        manager.start("consumer")
        manager.start("producer")
        try:
            # Keep the manager context open while the task processes run.
            while True:
                time.sleep(MAIN_IDLE_SLEEP_S)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
