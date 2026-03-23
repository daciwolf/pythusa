# PYTHUSA

PYTHUSA makes it easy to build high-throughput DSP pipelines in Python without giving up shared-memory performance.

You write the processing code; PYTHUSA handles zero-copy transport, process orchestration, and the throughput/latency behavior around it.

The project is deliberately narrow. The goal is not to grow into a general-purpose DSP kitchen sink; it is to make a small set of core pipeline patterns production-ready for scientific and sensor-style workloads with minimal, high-leverage changes.

## Design Goals

- Keep the package centered on Python-first DSP pipelines with shared-memory transport.
- Keep the shared-memory and worker runtime reusable outside any single application.
- Preserve the performance-sensitive ring-buffer implementation from the original VIVIIAN codebase.
- Expose stable entry points for user code while keeping internal implementation details in underscored packages.
- Prefer minimal, high-value changes that move the package toward production-ready scientific use.

## Project Status

The core runtime is already strong: zero-copy ring access works, benchmark suites exist, and the package can sustain multi-GB/s DSP-style pipelines on the tested machine. The remaining work is not a rewrite. It is productization: a smaller public API, observability, documentation, and release quality.

The next milestone is a small public pipeline API aimed at research and sensor users who want explicit frame shape, dtype, sample-rate metadata, and predictable throughput or latency behavior without wiring rings by hand.

The detailed roadmap lives in [PLAN.md](./PLAN.md).

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

## DSP Benchmark Suite

Run the representative DSP suite with:

```bash
python benchmarks/dsp_benchmark_suite.py
```

Key environment knobs:

```bash
DSP_BENCH_ROWS=16384 DSP_BENCH_PIPELINES=4 DSP_BENCH_DURATION_S=2.0 python benchmarks/dsp_benchmark_suite.py
```

Useful benchmark modes:

```bash
python benchmarks/dsp_benchmark_suite.py --throughput-max
python benchmarks/dsp_benchmark_suite.py --latency-min --kernels fir32,fir128,rfft
```

The suite reports per-kernel throughput, latency, and memory for several NumPy-based DSP workloads, including passthrough, windowing, FIR filters, FFT, power spectrum, and STFT. `balanced` mode is the default; `--throughput-max` and `--latency-min` apply the high-throughput and low-latency presets directly. `task_rss_mb` is the summed worker RSS and can overcount shared-memory mappings; `ring_mb` is the reserved shared-memory ring capacity.

Structured output is available with `--json` and `--json-out`:

```bash
python benchmarks/dsp_benchmark_suite.py --balanced --json-out benchmarks/results/dsp-balanced.json
```

For sensor-style latency budgeting, run:

```bash
python benchmarks/rocketdata_test.py --json-out benchmarks/results/rocket-latency.json
```

For the staged FFT comparison, run:

```bash
python benchmarks/compare_fft_benchmarks.py --json-out benchmarks/results/fft-compare.json
```

The benchmark command set and output conventions are documented in [benchmarks/README.md](./benchmarks/README.md).

For backend experiments, there is also a focused Numba benchmark that compares `numpy`, `numba_serial`, and `numba_parallel` on kernels where JIT compilation has a real chance to help:

```bash
python -m pip install -e .[benchmarks]
python benchmarks/numba_candidate_benchmark.py
```

That script targets signal generation, windowing, and `fir32`. It intentionally does not benchmark FFT with Numba because the FFT-heavy path already relies on NumPy's native FFT implementation.

PYTHUSA does not plan to depend on Numba in its core runtime. The intended use is optional and user-side: if a pipeline stage contains explicit Python loops or small custom DSP kernels, users may JIT those loops with Numba inside their own stage function and let PYTHUSA handle the process orchestration, shared-memory transport, and pipeline lifecycle around it. In the tested cases, that pattern was promising for some kernels such as `fir32`, but not a universal win.

## Roadmap

The next major steps are:

- define a small public pipeline API around `Pipeline`, `StreamSpec`, `Profile`, and explicit stage roles
- make the common research path obvious: acquire -> process -> detect -> log
- promote throughput/latency profiles and metrics into the runtime API
- keep the low-level runtime intact for power users who need direct control
- finish packaging, CI, and release validation

See [PLAN.md](./PLAN.md) for the full phased plan.

## Notes

- The current manager intentionally keeps one `TaskSpec` per process. Grouped multi-task workers from the old app repo were not carried over because they were not implemented cleanly in the extracted runtime.
- NumPy-oriented helpers currently remain internal while the stable public surface settles.
- Optional user-side accelerators such as Numba are compatible with the package direction, but they are treated as stage-level implementation choices rather than core runtime dependencies.
