# Pipeline API

The pipeline API is the main public entry point for building DSP graphs in PYTHUSA.
If the low-level runtime is the set of motors, belts, and bearings, `Pipeline` is the factory plan that tells those parts what to build.

## Overview

A pipeline represents the flow of data and execution used by PYTHUSA.
The simplest mental model is a factory line:

- tasks are the machines
- streams are the conveyor belts between machines
- events are control switches that tell certain machines when they may run

Each task runs in its own worker process.
Each stream is a fixed-shape, fixed-dtype shared-memory channel.
When you compile the pipeline, PYTHUSA turns that declaration into live rings, events, and worker processes.

Use `Pipeline` when you want:

- a declarative graph of your processing stages
- typed NumPy frame transport between processes
- automatic ring and worker setup
- a smaller public API than the raw `Manager` path

Use the lower-level runtime directly only when you need custom orchestration or direct ring control.

## Basic Workflow

Most pipelines follow the same sequence:

1. Create a `Pipeline`.
2. Declare one or more streams with `add_stream(...)`.
3. Declare any control events with `add_event(...)`.
4. Register tasks with `add_task(...)`.
5. Start the pipeline with `start()` or `run()`.
6. Stop, join, or close it explicitly, or use it as a context manager.

In factory-line terms:

- you name the machines
- you define the conveyor belts between them
- you wire any start/stop switches
- then you turn the line on

## Quick Example

```python
from __future__ import annotations

import time
import numpy as np
import pythusa


FRAME = np.arange(8, dtype=np.float32)


def source(samples) -> None:
    samples.write(FRAME)


def scale(samples, doubled) -> None:
    while True:
        frame = samples.read()
        if frame is None:
            time.sleep(0.001)
            continue
        doubled.write((frame * 2.0).astype(np.float32, copy=False))
        return


def sink(doubled) -> None:
    while True:
        frame = doubled.read()
        if frame is None:
            time.sleep(0.001)
            continue
        print(frame)
        return


def main() -> None:
    with pythusa.Pipeline("demo") as pipe:
        pipe.add_stream("samples", shape=(8,), dtype=np.float32)
        pipe.add_stream("doubled", shape=(8,), dtype=np.float32)

        pipe.add_task("source", fn=source, writes={"samples": "samples"})
        pipe.add_task(
            "scale",
            fn=scale,
            reads={"samples": "samples"},
            writes={"doubled": "doubled"},
        )
        pipe.add_task("sink", fn=sink, reads={"doubled": "doubled"})

        pipe.run()


if __name__ == "__main__":
    main()
```

This is a three-machine line:

- `source` publishes frames
- `scale` transforms them
- `sink` consumes the result

On Windows and other `spawn`-based multiprocessing environments, `pipe.start()` and `pipe.run()` must live behind an `if __name__ == "__main__":` guard or the child process will re-import the module and try to create the same shared-memory rings again. This does not change pipeline semantics on Linux or macOS; it is simply the correct portable `multiprocessing` pattern for standalone scripts.

## Creating A Pipeline

Create a pipeline by naming it:

```python
pipe = pythusa.Pipeline("radar")
```

The pipeline owns a runtime `Manager` internally and is the lifecycle owner for the graph you declare.
It can be used directly or as a context manager:

```python
with pythusa.Pipeline("radar") as pipe:
    ...
```

The context-manager form is the recommended default because it gives deterministic cleanup.

## Streams

Declare a stream with:

```python
pipe.add_stream(
    "samples",
    shape=(4096,),
    dtype=np.float32,
    frames=64,
    cache_align=True,
)
```

Parameters:

- `name`: unique stream name within the pipeline
- `shape`: frame shape passed to task bindings
- `dtype`: NumPy dtype for one frame
- `frames`: number of frames to allocate in the backing ring buffer
- `cache_align`: whether compile-time ring sizing should apply cache-line alignment

Conceptually, a stream is the conveyor belt between two machines.
At compile time, that declaration becomes a shared-memory ring sized for 32 frames by default, or the explicit `frames=` value you provide.

Current rules:

- one stream has exactly one writer task
- one stream may have zero or more reader tasks at declaration time
- compile requires each stream to have at least one reader

If you register the same stream name twice, `add_stream(...)` raises `ValueError`.

## Events

Declare an event with:

```python
pipe.add_event("shutdown")
pipe.add_event("armed", initial_state=True)
```

Events are process-shared control primitives.
In the factory-line analogy, they are the switches and gates that tell machines whether they may run.

Current guidance:

- many signalers is acceptable
- one consumer-side owner is the intended pattern
- many consumers on one event are discouraged

If an event is bound into more than two tasks, compile emits a warning because that usually indicates a design that should be split into separate events.

## Tasks

Register a task with:

```python
pipe.add_task(
    "fft_worker",
    fn=fft_worker,
    reads={"samples": "raw_adc"},
    writes={"fft": "spectra"},
    events={"shutdown": "shutdown"},
)
```

Parameters:

- `name`: unique task name within the pipeline
- `fn`: callable that runs in the worker process
- `reads`: mapping from local function argument names to stream names
- `writes`: mapping from local function argument names to stream names
- `events`: mapping from local function argument names to event names

The mapping direction matters:

- keys are the local argument names seen by the function
- values are the real stream or event names registered on the pipeline

So this:

```python
reads={"samples": "raw_adc"}
```

means:

- the pipeline stream is named `raw_adc`
- the task function receives it as `samples=...`

That lets reusable task functions stay stable even when pipeline names change.

### Decorator Form

`add_task(...)` also supports decorator registration for plain tasks:

```python
@pipe.add_task(
    "sink",
    reads={"samples": "samples"},
)
def sink(samples) -> None:
    ...
```

This decorator form is only for plain `add_task(...)`.
Controlled-task registration uses explicit `fn=...`.

## Controlled Tasks

Two controlled task forms are available:

- `pipe.add_task.switchable(...)`
- `pipe.add_task.toggleable(...)`

Example:

```python
def flush_buffer(output) -> None:
    output.write(np.zeros(8, dtype=np.float32))


pipe.add_event("flush")
pipe.add_stream("output", shape=(8,), dtype=np.float32)

pipe.add_task.toggleable(
    "flush_once",
    activate_on="flush",
    fn=flush_buffer,
    writes={"output": "output"},
    events={"flush": "flush"},
)
```

Behavior:

- `switchable`: waits for the event, then keeps rerunning without resetting it
- `toggleable`: waits for the event, resets it, then runs once per activation

Rules:

- `activate_on` must be one of the task's bound event names
- controlled tasks are event-driven wrappers around the task function
- the control event is not passed through to the task function itself

If `activate_on` is missing from the task's event bindings, registration raises `ValueError`.

## Stream Bindings Inside Tasks

At runtime, tasks do not receive raw arrays directly.
They receive stream binding objects.

Reader bindings support:

- `read() -> np.ndarray | None`
- `read_into(out) -> bool`
- `look() -> memoryview | None`
- `increment()`
- `set_blocking(bool)`
- `is_blocking()`
- `.raw`
- `.ring`

Writer bindings support:

- `write(array) -> bool`
- `look() -> memoryview | None`
- `increment()`
- `.raw`
- `.ring`

Example:

```python
def worker(samples, fft) -> None:
    while True:
        frame = samples.read()
        if frame is None:
            time.sleep(0.001)
            continue
        spectrum = np.fft.rfft(frame).astype(np.complex64, copy=False)
        if fft.write(spectrum):
            return
```

For lower-allocation paths, prefer `read_into(...)`:

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

Notes:

- `read()` returns an owned NumPy array copy
- `read_into(...)` avoids that allocation by filling a provided array
- `look()` returns a zero-copy memoryview for the next contiguous frame and does not advance the reader
- call `increment()` after you finish using the view from `look()`
- `look()` returns `None` when the next frame is not available or is wrapped across the ring boundary
- `writer.look()` returns a zero-copy writable memoryview for the next contiguous frame and does not advance the writer
- call `writer.increment()` after you fill the view from `writer.look()`
- `writer.look()` returns `None` when the next frame would wrap across the ring boundary
- `write(...)` validates shape and dtype before publishing
- `.raw` and `.ring` expose the underlying shared-memory ring for direct low-level access

## Blocking And Backpressure

Readers participate in writer backpressure by default.
That is usually what you want.

If you need a reader to stop holding writers back:

```python
samples.set_blocking(False)
```

When a reader is made non-blocking:

- it is marked inactive
- writers stop treating it as a backpressure participant

When it is re-enabled:

```python
samples.set_blocking(True)
```

the reader jumps to the current writer position.
Unread backlog is discarded rather than replayed.

This is an advanced control.
It is useful for "latest frame only" style consumers, but it is not lossless.

## Lifecycle Methods

The main lifecycle methods are:

- `compile()`
- `start()`
- `run()`
- `stop()`
- `join(timeout=None)`
- `close()`

Typical usage:

```python
with pythusa.Pipeline("demo") as pipe:
    ...
    pipe.start()
    pipe.join()
```

or, for short-running pipelines:

```python
with pythusa.Pipeline("demo") as pipe:
    ...
    pipe.run()
```

Method behavior:

- `compile()`: validates the graph and registers runtime objects
- `start()`: compiles if needed, then starts tasks in reverse topological order
- `run()`: convenience method for `start()` followed by `join()`
- `stop()`: requests shutdown of running worker processes
- `join()`: waits for worker processes to exit
- `close()`: stops, joins, and closes the owned runtime manager

Current lifecycle model:

- a pipeline instance is compile-once
- a pipeline instance is start-once
- `compile()` twice raises `RuntimeError`
- `start()` twice raises `RuntimeError`
- `close()` is idempotent

## Metrics And Monitoring

Start monitoring with:

```python
pipe.start_monitor(interval_s=0.05)
```

Read metrics with:

```python
all_metrics = pipe.metrics()
worker_metrics = pipe.metrics("worker")
```

Metrics expose snapshots of:

- PID
- CPU percent
- RSS memory
- nice level
- ring pressure

Use this for operational visibility, not as a hard real-time timing guarantee.

If you ask for an unknown task name, `metrics(task_name)` raises `KeyError`.

## Saving And Reconstructing Pipelines

Pipelines can be serialized to TOML:

```python
pipe.save("radar.toml")
restored = pythusa.Pipeline.reconstruct("radar.toml")
```

This persists the declaration, not a live runtime.

Saved content includes:

- pipeline name
- stream declarations
- event declarations
- task bindings
- controlled-task metadata
- callable module and qualified name

Important limitation:

- saved task callables must be importable top-level functions
- lambdas, nested functions, and other non-importable callables are not supported

## Errors And Current Limits

PYTHUSA v0 is intentionally narrow.
The constraints are there to keep the execution model explicit.

Current unsupported or constrained patterns:

- cyclic task graphs are not supported
- multiple writers to one stream are not supported
- arbitrary Python objects as stream payloads are not supported
- variable-shape frames on one stream are not supported
- automatic fan-in coordination is not provided
- many independent consumers on one event are not the intended event model

Some important consequences:

- if a task reads from multiple input streams, that task is responsible for deciding how to synchronize them
- if two upstream producers run at different rates, the downstream task must define the join logic
- if you need counted per-activation semantics, design the event ownership carefully instead of treating one event like a broadcast queue

Common failure modes:

- duplicate stream, task, or event names raise `ValueError`
- invalid task bindings raise `ValueError`
- compile-time topology problems raise `ValueError`
- trying to compile or start the same pipeline twice raises `RuntimeError`

## When To Drop Lower

`Pipeline` should be the default.
Drop to the low-level runtime only when you specifically need:

- direct `Manager` control
- manual ring construction
- custom bootstrap behavior
- experiments that depend on raw ring semantics

If your application still fits the factory-line model of machines plus conveyor belts, stay with `Pipeline`.
If you need to rewire the motors themselves, use the runtime layer.
