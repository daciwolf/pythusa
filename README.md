# PYTHUSA

PYTHUSA is a Python-first shared-memory runtime for fixed-shape NumPy data pipelines.
It is built for workloads where you want multiple Python processes moving numeric frames between stages with low overhead and explicit control over latency, throughput, and backpressure.

You write the processing code; PYTHUSA handles zero-copy transport, process orchestration, and the throughput/latency behavior around it.

PYTHUSA is built around typed streams of fixed-shape NumPy frames. You declare a pipeline DAG, map task parameters to streams or events, and let the runtime compile that declaration into one process per task with shared-memory transport underneath.

PYTHUSA started as backend infrastructure for DSP processing work on the UCI Rocket Project Liquids team and was later extracted into its own library once the shared-memory runtime proved useful more broadly.

## Documentation

- [Docs Home](https://daciwolf.github.io/pythusa/)
- [Showcase Demos](https://daciwolf.github.io/pythusa/demos/)
- [Under the Hood](https://daciwolf.github.io/pythusa/internals/)
- [Pipeline API](https://daciwolf.github.io/pythusa/pipeline/)
- [Runtime](https://daciwolf.github.io/pythusa/runtime/)
- [Benchmarks](https://daciwolf.github.io/pythusa/benchmarks/)

![PYTHUSA simple dataflow](https://daciwolf.github.io/pythusa/assets/simple-dataflow.svg)

**[Showcase Demos](https://daciwolf.github.io/pythusa/demos/)** -- FFT pipeline hitting **~73 Gbit/s** across 49 signals and a market microstructure replay desk pushing **~50 Gbit/s** across 8 symbols with live quant analytics. No C extensions. Performance numbers, architecture diagrams, and run commands.

## Core Ideas

### Streams

Streams move fixed-shape, fixed-dtype NumPy frames between tasks.
At compile time, a stream becomes a shared-memory ring buffer sized in bytes.

### Tasks

Tasks are normal Python callables.
Today, one registered task maps to one worker process.

### Events

Events are process-shared control primitives used to gate or trigger work.
Use them when a task should react to a signal instead of running unconditionally.

## Install

PYTHUSA currently targets Python 3.12 only.

### Install From PyPI

```bash
python -m pip install pythusa
```

Optional extras from PyPI:

```bash
python -m pip install "pythusa[test]"
python -m pip install "pythusa[examples]"
python -m pip install "pythusa[benchmarks]"
```

### Install From Source

#### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Optional extras:

```bash
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

In `reads`, `writes`, and `events`, the keys are the task's local parameter names and the values are the registered stream or event names.

On Windows and other `spawn`-based multiprocessing environments, keep `pipe.start()` and `pipe.run()` inside a `main()` guarded by `if __name__ == "__main__":`.

## Public API

- `pythusa.Pipeline`: high-level DAG builder and lifecycle owner for shared-memory multiprocess pipelines.
- `pipe.add_stream(name, shape, dtype, frames=32, cache_align=True, min_reader_pos_refresh_interval=64, min_reader_pos_refresh_s=0.005)`: declare a framed stream, optionally set ring capacity in frames, and tune how often writers rescan slow-reader state.
- `pipe.add_task(...)`: bind task parameters to readers, writers, and events. Use `pipe.add_task.toggleable(...)`, `pipe.add_task.switchable(...)`, or `pipe.add_task.terminator(...)` for special task forms.
- `pipe.start_monitor()` and `pipe.metrics()`: collect CPU, RSS, nice, and ring-pressure snapshots for running tasks.
- `pipe.save(path)` and `pythusa.Pipeline.reconstruct(path)`: persist or restore pipeline declarations as TOML. Saved task callables must be importable top-level functions.
- `pythusa.Manager`, `pythusa.RingSpec`, `pythusa.TaskSpec`, `pythusa.get_reader`, `pythusa.get_writer`, and `pythusa.get_event`: lower-level primitives for users who want direct ring and worker control, including direct access to the same min-reader refresh controls.

## What To Read Next

- Read [Under the Hood](https://daciwolf.github.io/pythusa/internals/) for a guided walkthrough of the hot path -- the code behind 73 Gbit/s.
- Read [Pipeline API](https://daciwolf.github.io/pythusa/pipeline/) for the high-level programming model.
- Read [Runtime](https://daciwolf.github.io/pythusa/runtime/) if you need to understand ring buffers, task bootstrap, or raw ring access.
- Read [Benchmarks](https://daciwolf.github.io/pythusa/benchmarks/) if you want to compare throughput and latency modes.

## Showcase Demos

All benchmark numbers below were recorded on a **MacBook Air M2**.

### FFT Pipeline Demo

A multi-channel FFT pipeline that streams synthetic sensor data through shared-memory ring buffers into parallel FFT workers. Scales from ~21 Gbit/s with 2 generators to **~73 Gbit/s sustained** and **~140k FFT/s** across 49 signals with 7 generators.

### Stock Quant Demo

A simulated L3 market microstructure replay desk pushing **~50 Gbit/s** aggregate market data throughput across 8 symbols with 9 live quant metrics per symbol, end-to-end latency tracking, and speedup against a serial baseline.

See the full **[Showcase Demos](https://daciwolf.github.io/pythusa/demos/)** page for architecture diagrams, performance tables, flags, and run commands.

## More Examples

- `python examples/basic_workers.py` -- raw `Manager` plus `SharedRingBuffer` usage.
- `python examples/engine_dsp_pipeline.py` -- larger `Pipeline` example with plotting, monitoring, and real DSP-style stages. Install `.[examples]` first.
- `python examples/fir128_scaling_pipeline.py` -- round-robin FIR128 fan-out/fan-in scaling example over engine-data-derived signals.

## Benchmarks

Run the representative DSP benchmark suite with:

```bash
python benchmarks/dsp_benchmark_suite.py
```

The suite reports per-kernel throughput, latency, and memory for passthrough, windowing, FIR filters, FFT, power spectrum, and STFT workloads. Structured JSON output, DSP heatmaps, and additional benchmark entry points are documented in the full [Benchmarks](https://daciwolf.github.io/pythusa/benchmarks/) guide and [benchmarks/README.md](./benchmarks/README.md). Install `.[benchmarks]` for the full benchmark and comparison set.

## License

PYTHUSA is licensed under the GNU General Public License, version 2 only (`GPL-2.0-only`).
See [LICENSE](./LICENSE) for the full license text.
