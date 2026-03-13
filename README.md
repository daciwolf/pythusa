# PYTHUSA

PYTHUSA is a standalone Python library for high-throughput IPC, shared memory, worker orchestration, and parallel NumPy-based data processing.

The package layout is intentionally modeled after the organizational style of scientific Python libraries such as NumPy: a small top-level public API, logically separated internal subpackages, and clear boundaries between low-level primitives and higher-level orchestration.

## Design Goals

- Keep the shared-memory and worker runtime reusable outside any single application.
- Preserve the performance-sensitive ring-buffer implementation from the original VIVIIAN codebase.
- Expose stable entry points for user code while keeping internal implementation details in underscored packages.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PYTHUSA currently targets Python 3.12 only.

## Quick Example

```python
import time
import numpy as np
import pythusa

RING_NAME = "samples"
BATCH = 16
DTYPE = np.float64
NBYTES = BATCH * np.dtype(DTYPE).itemsize


def producer() -> None:
    writer = pythusa.get_writer(RING_NAME)
    while True:
        payload = np.linspace(0.0, 1.0, BATCH, dtype=DTYPE)
        if writer.write_array(payload) == 0:
            time.sleep(0.001)


def consumer() -> None:
    reader = pythusa.get_reader(RING_NAME)
    while True:
        batch = reader.read_array(NBYTES, dtype=DTYPE)
        if batch.size == BATCH:
            print(batch.mean())
        else:
            time.sleep(0.001)


with pythusa.Manager() as manager:
    manager.create_ring(pythusa.RingSpec(name=RING_NAME, size=NBYTES * 8, num_readers=1))
    manager.create_task(pythusa.TaskSpec(name="producer", fn=producer, writing_rings=(RING_NAME,)))
    manager.create_task(pythusa.TaskSpec(name="consumer", fn=consumer, reading_rings=(RING_NAME,)))
    manager.start("consumer")
    manager.start("producer")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
```

Task functions are the process body in PYTHUSA. If a task should run continuously, put the loop inside the task function.

## Module Overview

- `pythusa.Manager`: process lifecycle, ring registration, monitoring, and task startup.
- `pythusa.SharedRingBuffer`: shared-memory ring primitive for zero-copy byte and NumPy array transport.
- `pythusa.RingSpec`: immutable ring configuration.
- `pythusa.TaskSpec`: task registration metadata for worker processes.
- `pythusa.EventSpec` and `pythusa.WorkerEvent`: synchronization gates.
- `pythusa.get_reader`, `pythusa.get_writer`, `pythusa.get_event`: process-local accessors installed inside worker processes.

## Notes

- The current manager intentionally keeps one `TaskSpec` per process. Grouped multi-task workers from the old app repo were not carried over because they were not implemented cleanly in the extracted runtime.
- NumPy-oriented helpers currently remain internal while the stable public surface settles.
