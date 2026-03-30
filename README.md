# PYTHUSA

PYTHUSA makes it easy to build high-throughput multiprocess Data Processing pipelines in Python. Through the use of shared memory, select operating system privatives and seperate python proccesses PYTHUSA allows for the effective bypassing of the GIL and cross platform compatabiliy in order to maximize latency, throughput and portability.

You write the processing code; PYTHUSA handles zero-copy transport, process orchestration, and the throughput/latency behavior around it.

PYTHUSA is built around typed streams of fixed-shape NumPy frames. You declare a pipeline DAG, map task parameters to streams or events, and let the runtime compile that declaration into one process per task with shared-memory transport underneath.

PYTHUSA started as backend infrastructure for DSP processing work on the UCI Rocket Project Liquids team and was later extracted into its own library once the shared-memory runtime proved useful more broadly.

![PYTHUSA simple dataflow](docs/assets/simple-dataflow.svg)

## How It Works

PYTHUSA has three runtime building blocks: streams, tasks, and optional events.

Streams are fixed-shape, typed channels for moving frames between processes. Under the hood, each declared stream becomes a shared-memory ring buffer sized for 32 frames by default. That gives the runtime a reusable transport layer with low per-transfer overhead, stable backpressure behavior, and no need to allocate a new IPC object for every frame that moves through the graph.

Tasks are normal Python callables. Today, one registered task maps to one worker process. The `reads`, `writes`, and `events` mappings on `add_task()` tell the pipeline which local parameter names should be bound to which stream or event names when that task starts inside its own process.

Compilation turns the declaration into a runnable DAG. `Pipeline.compile()` validates that every stream has exactly one writer, at least one reader, and no cycles in the task graph. It then creates the live rings and events, wraps each task with typed stream bindings, and records the startup order for the worker processes.

Startup is explicit. `Pipeline.start()` launches tasks in reverse topological order so downstream readers are alive before upstream writers begin publishing. At runtime, read bindings are `StreamReader` objects and write bindings are `StreamWriter` objects. `read()` returns one shaped NumPy frame or `None` when no frame is ready, and `write()` validates shape and dtype before publishing the frame into shared memory.

Events are optional coordination primitives for gating work between tasks. `add_task.toggleable(...)` consumes one event activation per run, while `add_task.switchable(...)` waits for an event and reruns without resetting it. For observability, `start_monitor()` can sample per-task CPU, RSS, nice level, and ring pressure while the pipeline is running.

If a task should run continuously, the loop belongs inside the task function. If you need raw ring access instead of framed stream helpers, the lower-level `Manager`, `RingSpec`, and `TaskSpec` APIs are available.

## Install

PYTHUSA currently targets Python 3.12 only.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[examples]"
python -m pip install -e ".[benchmarks]"
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Optional extras:

```powershell
python -m pip install -e ".[test]"
python -m pip install -e ".[examples]"
python -m pip install -e ".[benchmarks]"
```

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
```

In `reads`, `writes`, and `events`, the keys are the task's local parameter names and the values are the registered stream or event names.

## Public API

- `pythusa.Pipeline`: high-level DAG builder and lifecycle owner for shared-memory multiprocess pipelines.
- `pipe.add_stream(name, shape, dtype, cache_align=True)`: declare a framed stream.
- `pipe.add_task(...)`: bind task parameters to readers, writers, and events. Use `pipe.add_task.toggleable(...)` or `pipe.add_task.switchable(...)` for event-controlled tasks.
- `pipe.start_monitor()` and `pipe.metrics()`: collect CPU, RSS, nice, and ring-pressure snapshots for running tasks.
- `pipe.save(path)` and `pythusa.Pipeline.reconstruct(path)`: persist or restore pipeline declarations as TOML. Saved task callables must be importable top-level functions.
- `pythusa.Manager`, `pythusa.RingSpec`, `pythusa.TaskSpec`, `pythusa.get_reader`, `pythusa.get_writer`, and `pythusa.get_event`: lower-level primitives for users who want direct ring and worker control.

## Examples

- `python examples/basic_workers.py` shows the raw `Manager` plus `SharedRingBuffer` path.
- `python examples/engine_dsp_pipeline.py` shows a larger `Pipeline` graph with branching streams, monitoring, and plotting. Install `.[examples]` first.

## Benchmarks

The current benchmark suite is DSP-heavy because that is the workload family the runtime has been exercised against most. The runtime itself is not limited to DSP as long as your stages exchange fixed-shape NumPy frames.

Run the representative suite with:

```bash
python benchmarks/dsp_benchmark_suite.py
```

Useful knobs and modes:

```bash
DSP_BENCH_ROWS=16384 DSP_BENCH_PIPELINES=4 DSP_BENCH_DURATION_S=2.0 python benchmarks/dsp_benchmark_suite.py
python benchmarks/dsp_benchmark_suite.py --throughput-max
python benchmarks/dsp_benchmark_suite.py --latency-min --kernels fir32,fir128,rfft
```

The suite reports per-kernel throughput, latency, and memory for passthrough, windowing, FIR filters, FFT, power spectrum, and STFT workloads. `balanced` is the default mode. `task_rss_mb` is summed worker RSS and can overcount shared-memory mappings; `ring_mb` is reserved shared-memory ring capacity.

Structured output is available with `--json` and `--json-out`:

```bash
python benchmarks/dsp_benchmark_suite.py --balanced --json-out benchmarks/results/dsp-balanced.json
```

Additional benchmark entry points:

```bash
python benchmarks/rocketdata_test.py --json-out benchmarks/results/rocket-latency.json
python benchmarks/compare_fft_benchmarks.py --json-out benchmarks/results/fft-compare.json
python benchmarks/numba_candidate_benchmark.py
```

The benchmark command set and output conventions are documented in [benchmarks/README.md](./benchmarks/README.md). Install `.[benchmarks]` for the full benchmark and comparison set.
