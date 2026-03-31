# Runtime

The runtime is the lower-level execution layer underneath `Pipeline`.
If `Pipeline` is the factory plan, the runtime is the actual machinery: the rings, worker processes, bootstrap logic, and synchronization objects that make the line move.

Most users should start with `Pipeline`.
Drop to the runtime only when you need direct control over ring sizing, task startup order, raw shared-memory access, or custom orchestration.

## What Runtime Means

At the runtime level, PYTHUSA is built from a small set of explicit parts:

- `Manager`: owns live rings, events, and worker processes
- `RingSpec`: describes one shared-memory ring
- `TaskSpec`: describes one worker task
- `EventSpec`: describes one process-shared event
- `SharedRingBuffer`: the actual shared-memory transport object
- `WorkerEvent`: the actual process-shared event object

`Pipeline.compile()` eventually creates these lower-level objects for you.
Using the runtime directly means building them yourself.

The tradeoff is simple:

- `Pipeline` is smaller, safer, and easier to reason about
- the runtime is more explicit and more flexible, but easier to misuse

## Runtime Object Model

### `Manager`

`Manager` is the runtime owner.
It keeps strong references to:

- live creator-side ring handles
- live event objects
- registered task specs
- spawned worker processes
- sampled process metrics

At a high level, `Manager` does four things:

1. register rings, events, and tasks
2. assign reader slots to task processes
3. build a task bootstrap for each process
4. start, stop, join, and monitor those processes

Unlike `Pipeline`, `Manager` is imperative.
You register objects and start named tasks yourself.

### `RingSpec`

`RingSpec` is pure configuration for one ring buffer:

- `name`
- `size` in bytes
- `num_readers`
- `cache_align`
- `cache_size`

Important: at the runtime level, ring size is always in bytes.
The raw ring does not know about NumPy frame shape or dtype beyond whatever the caller chooses to interpret.

### `TaskSpec`

`TaskSpec` declares one worker process:

- `name`
- `fn`
- `reading_rings`
- `writing_rings`
- `events`
- `args`
- `kwargs`

The runtime starts one `TaskSpec` per process.
The task callable is invoked exactly once.
If a task should run continuously, the loop belongs inside the task function.

### `EventSpec`

`EventSpec` is pure configuration:

- `name`
- `initial_state`

It tells the manager to create a named `WorkerEvent`.

### `SharedRingBuffer`

`SharedRingBuffer` is the actual shared-memory transport.
It is a fixed-size byte ring with one writer position and one reader position per registered reader slot.

The ring is concerned only with bytes.
Shape and dtype interpretation are the caller's responsibility.

### `WorkerEvent`

`WorkerEvent` is the runtime control primitive used across processes.
It combines:

- a process-shared `Event` for blocking wakeup
- a process-shared `Semaphore` for counted activations
- a shared counter and lock to keep state consistent

That makes it more than a plain binary flag.

## How Ring Buffers Work Under The Hood

`SharedRingBuffer` stores two things in one shared-memory block:

- a header
- the ring payload bytes

The header tracks:

- logical ring size
- pressure
- dropped size
- global write position
- computed writable capacity
- number of readers
- per-reader position, active flag, and last-seen slot data

Conceptually:

- the writer only advances the write position
- each reader advances its own reader position
- writable capacity is determined by the slowest active reader

That last point is the backpressure rule.
If one active reader falls behind, it limits how far the writer may advance.

### Writer Path

The writer:

1. asks how many bytes are writable
2. obtains one or two memoryviews into the ring payload
3. copies bytes into those views
4. advances the write position

If there is not enough room, the write fails and returns `0`.
The caller decides whether to retry, wait, or drop.

### Reader Path

The reader:

1. checks how many bytes are readable
2. obtains one or two memoryviews into the ring payload
3. copies or views the readable bytes
4. advances its own reader position

Each reader has its own slot.
Readers do not contend by rewriting a shared reader cursor.

### Wrap-Around

If a read or write spans the end of the payload, the ring exposes two views:

- one to the tail of the buffer
- one to the beginning of the buffer

The helper methods handle this for you.

### Pressure

Ring pressure is the complement of writable space, expressed as a percentage.
If little writable space remains, pressure rises.
The monitor thread uses this to report stalled or overloaded stages.

## How Tasks Work Under The Hood

The task runtime path is:

1. you register a `TaskSpec`
2. `Manager.start(name)` validates the task
3. the manager assigns reader slots for the task's input rings
4. it builds a `TaskBootstrap`
5. a child process is spawned with that bootstrap as the target
6. the child opens its reader and writer ring handles
7. the child enters those ring handles as context managers
8. worker-local context is installed
9. the worker callable is invoked exactly once

The callable itself owns its loop.
The runtime does not inject a scheduler around normal tasks.

This is an important design point:

- the runtime provides transport and process orchestration
- task behavior still belongs to the task function

## Worker Execution Model

The worker process sees the runtime through module-level context.
Inside a worker, helpers like:

- `pythusa.get_reader(...)`
- `pythusa.get_writer(...)`
- `pythusa.get_event(...)`

resolve objects that were installed by bootstrap.

That installation happens once per child process.
It is safe because each worker is its own process with its own module state.

The bootstrap also enters every ring under an `ExitStack`, so reader activation and ring cleanup follow the ring object's context-manager lifecycle.

One platform note:

- on POSIX, `SIGTERM` can exit a worker cleanly through the installed handler
- on Windows, process termination is effectively hard termination, so you should prefer normal task completion and explicit lifecycle management where possible

## Minimal Raw Runtime Example

This is the simplest end-to-end example using `Manager`, `RingSpec`, and `TaskSpec` directly.

```python
from __future__ import annotations

import time
import numpy as np
import pythusa


FRAME = np.arange(8, dtype=np.float32)
FRAME_NBYTES = FRAME.nbytes


def source() -> None:
    writer = pythusa.get_writer("samples")
    while writer.write_array(FRAME) == 0:
        time.sleep(0.001)


def sink() -> None:
    reader = pythusa.get_reader("samples")
    while True:
        frame = reader.read_array(FRAME_NBYTES, dtype=np.float32)
        if frame.size == 0:
            time.sleep(0.001)
            continue
        print(frame.reshape(8))
        return


def main() -> None:
    with pythusa.Manager() as manager:
        manager.create_ring(
            pythusa.RingSpec(
                name="samples",
                size=FRAME_NBYTES * 32,
                num_readers=1,
            )
        )

        manager.create_task(
            pythusa.TaskSpec(
                name="source",
                fn=source,
                writing_rings=("samples",),
            )
        )
        manager.create_task(
            pythusa.TaskSpec(
                name="sink",
                fn=sink,
                reading_rings=("samples",),
            )
        )

        manager.start("sink")
        manager.start("source")
        manager.join_all()


if __name__ == "__main__":
    main()
```

Notes:

- the ring size is declared in bytes
- the sink is started before the source so the reader is alive first
- `read_array(...)` and `write_array(...)` are raw byte-oriented helpers
- on Windows and other `spawn`-based environments, the `main()` guard is required for the same reason as the pipeline examples: child processes re-import the script module during startup

## Raw Ring API

The main raw ring methods are:

- `write_array(arr) -> int`
- `read_array(nbytes, dtype) -> np.ndarray`
- `expose_writer_mem_view(size)`
- `expose_reader_mem_view(size)`
- `simple_write(...)`
- `simple_read(...)`

The first two are the most common low-level entry points.

### `write_array`

`write_array(arr)`:

- treats the NumPy array as raw bytes
- writes it if enough space exists
- returns the number of bytes written
- returns `0` if insufficient space exists

It does not validate application-level semantics such as expected shape.

### `read_array`

`read_array(nbytes, dtype)`:

- reads exactly `nbytes` if available
- returns an empty array if insufficient data exists
- interprets the bytes as `dtype`
- leaves shape reconstruction to the caller

Example:

```python
frame = reader.read_array(FRAME_NBYTES, dtype=np.float32)
if frame.size:
    frame = frame.reshape(8)
```

### View And Copy Semantics

One important caveat:

- in the non-wrap case, `read_array(...)` may return an array backed by the shared-memory view
- in the wrap case, it must copy into a temporary contiguous buffer first

This means the raw path is powerful but less forgiving.
If you keep arrays backed by shared memory alive for too long, shutdown and cleanup become easier to get wrong.

That is why the higher-level pipeline binding `read()` returns an owned copy by default.

## Raw Ring Access From Pipeline

If you like `Pipeline` but still want the raw path in a specific task, use `.raw` or `.ring` from a stream binding:

```python
def worker(samples, fft) -> None:
    raw_reader = samples.raw
    raw_writer = fft.raw

    frame_nbytes = 4096 * np.dtype(np.float32).itemsize

    while True:
        frame = raw_reader.read_array(frame_nbytes, dtype=np.float32)
        if frame.size == 0:
            time.sleep(0.001)
            continue

        frame = frame.reshape(4096)
        spectrum = np.fft.rfft(frame).astype(np.complex64, copy=False)

        if raw_writer.write_array(spectrum) == spectrum.nbytes:
            return
```

This is the escape hatch for users who want `Pipeline` orchestration with lower-level ring semantics inside selected tasks.

## Convenience Stream Bindings

The pipeline stream bindings are wrappers over raw rings.

Reader bindings provide:

- `read()`
- `read_into(out)`
- `look()`
- `increment()`
- `set_blocking(bool)`
- `is_blocking()`
- `.raw`
- `.ring`

Writer bindings provide:

- `write(array)`
- `look()`
- `increment()`
- `.raw`
- `.ring`

For performance-sensitive tasks, prefer `read_into(...)` over `read()` to avoid one allocation per frame:

```python
def worker(samples, fft) -> None:
    frame = np.empty((4096,), dtype=np.float32)
    while True:
        if not samples.read_into(frame):
            time.sleep(0.001)
            continue
        spectrum = np.fft.rfft(frame).astype(np.complex64, copy=False)
        if fft.write(spectrum):
            return
```

If you need a zero-copy borrow instead of filling a caller-owned array, `look()` returns a memoryview for the next contiguous frame and leaves the reader position unchanged. Call `increment()` after you are done with the view. If the next frame is wrapped across the ring boundary, `look()` returns `None` rather than copying.

Writers have the same pattern: `look()` returns a writable memoryview for the next contiguous frame, and `increment()` commits that frame once you are done filling it. If the next slot would wrap, `look()` returns `None` rather than copying.

Example:

```python
def worker(samples, fft) -> None:
    while True:
        frame_view = samples.look()
        if frame_view is None:
            time.sleep(0.001)
            continue
        frame = np.frombuffer(frame_view, dtype=np.float32).reshape((4096,))

        fft_view = fft.look()
        if fft_view is None:
            time.sleep(0.001)
            continue

        spectrum = np.frombuffer(fft_view, dtype=np.complex64).reshape((2049,))
        spectrum[:] = np.fft.rfft(frame).astype(np.complex64, copy=False)

        samples.increment()
        fft.increment()
        return
```

## Blocking And Backpressure

By default, each reader participates in backpressure.
The writer's available space is limited by the slowest active reader.

That is the runtime's default safety model:

- if a reader is alive and active, the writer must respect it

You can opt a reader out temporarily:

```python
samples.set_blocking(False)
```

This marks the reader inactive.
Inactive readers do not constrain the writer's writable space.

When you re-enable it:

```python
samples.set_blocking(True)
```

the reader jumps to the current writer position.
Unread backlog is discarded.

This is useful for "latest frame wins" consumers, but it is not a lossless mode.

## Events And Controlled Tasks

At the runtime level, `WorkerEvent` exposes:

- `signal()`
- `wait(timeout=None)`
- `reset()`
- `is_open()`
- `pending`

Semantics:

- `signal()` increments pending activation count and opens the event
- `wait()` blocks on the wake event
- `reset()` consumes one pending activation
- the wake event is only cleared when pending count reaches zero

Example:

```python
def controller() -> None:
    event = pythusa.get_event("flush")
    event.signal()


def consumer() -> None:
    event = pythusa.get_event("flush")
    event.wait()
    event.reset()
    print("flush requested")
```

Important limitation:

- a `WorkerEvent` is not a general-purpose multi-consumer counted queue

The wake primitive is still event-like.
If many consumers all wait on the same event, the ownership and reset semantics become ambiguous.

Recommended pattern:

- many signalers is fine
- one reset-owning consumer is the intended model
- if many downstream tasks need separate activations, use many events

### Controlled Tasks

The pipeline control helpers sit above this runtime event model:

- `switchable`: wait, then rerun without resetting
- `toggleable`: wait, reset, then run once per activation

These wrappers do not change the worker model.
They are still just runtime loops around the task callable.

## Monitoring And Metrics

`Manager.start_monitor(...)` launches a daemon monitor thread that samples:

- ring pressure
- worker CPU usage
- worker RSS
- worker nice level

Per-task snapshots are exposed as `ProcessMetrics`.

This is operational visibility, not deterministic timing analysis.
Sampling can tell you that a stage is falling behind or starving, but it is not a substitute for application-level latency measurement.

## Errors And Current Limits

The runtime layer is intentionally narrow and explicit.

Important constraints:

- one `TaskSpec` maps to one process
- one ring has one writer handle and one reader slot per registered reader
- raw rings move bytes, not semantic frames
- the runtime does not provide automatic fan-in synchronization
- grouped multi-task workers are not currently a public feature

Some important error cases:

- `RingSpec(size <= 0)` raises `ValueError`
- `RingSpec(num_readers < 1)` raises `ValueError`
- `TaskSpec` raises `ValueError` if the same ring appears in both `reading_rings` and `writing_rings`
- `Manager.create_task(...)` raises on duplicate task names
- starting a task that references an unregistered ring or event fails during bootstrap construction

Be especially careful on the raw path with:

- dtype mismatches
- shape mismatches
- assuming `read_array(...)` returns owned memory
- forgetting that ring size is declared in bytes

## When To Bypass Pipeline

Use the runtime directly when you need:

- manual ring sizing in bytes
- direct ring header and pressure behavior
- custom startup order
- custom worker composition experiments
- benchmark or debugging scenarios that should avoid the higher-level binding layer

Stay with `Pipeline` when:

- your graph fits the machine-and-conveyor-belt model
- typed frame bindings are enough
- you do not need to manage raw ring semantics directly

The runtime is the right place for power users.
`Pipeline` is still the right default for almost everyone else.
