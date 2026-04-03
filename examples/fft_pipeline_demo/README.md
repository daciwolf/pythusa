# FFT Pipeline Demo

A high-rate, multi-channel FFT pipeline built on Pythusa. Synthetic sensor data
flows through shared-memory ring buffers into parallel FFT workers, with an
optional ImGui operator desk for live signal monitoring.

The shape of this demo comes from a real instrumentation need: move
rocket-sensor-class data off a DAQ, fan it out through shared memory, and keep
both human monitoring and online FFT analysis alive without turning the GUI into
the bottleneck.

## What it demonstrates

This demo exercises the core of Pythusa in a single coherent application:

- **Shared-memory streams** replace Python object serialization between stages.
  Data never gets pickled. Producers write frames into ring buffers; consumers
  read them through zero-copy memoryviews.
- **Pipeline DAG compilation** validates the topology at build time, catches
  missing bindings and cycles, and topologically sorts tasks so consumers start
  before producers.
- **Event-driven task gating** lets FFT workers sleep until an operator arms
  them from the dashboard, then run continuously once signaled.
- **Concurrent fanout** delivers the same generator stream to both the display
  path and the analysis path without duplicating data.
- **Headless benchmarking** strips the GUI stack entirely and measures sustained
  FFT throughput in isolation.

## Signal shape

| Parameter | Value |
| --- | --- |
| Generators | 2 |
| Signals per generator | 7 |
| Total signals | 14 |
| Sample rate | 61.44 kS/s per signal |
| Signal composition | 16 randomized sinusoids + Gaussian noise per channel |

Each signal is a sum of 16 tones at random frequencies (60--1000 Hz), random
amplitudes, and random phases, plus additive noise. This produces dense spectra
representative of vibration or acoustic sensor data.

## Pipeline topology

### GUI path

```
generator(1,2) --> raw stream --> display sampler --> display stream --> ImGui desk
                       |
                       +--> FFT worker (event-gated) --> filtered stream --> display
                       |                                      |
                       |                                      +--> stats stream --> ImGui desk
```

1. Two generators emit timestamped multi-channel frames into shared memory.
2. Display samplers downsample the raw stream (every 1000th sample) for the
   render path.
3. Per-channel FFT workers wait for an operator event, then compute the FFT,
   extract the dominant frequency component, and reconstruct it as a filtered
   overlay.
4. The ImGui desk reads display streams, filtered overlays, and per-channel
   throughput statistics. Operators arm FFT lanes individually from the sidebar.

### Headless path

```
benchmark generator(1,2) --> benchmark stream --> batched FFT worker --> stats stream --> console
```

1. Benchmark generators emit FFT-friendly batched frames (signals-first layout).
2. One batched FFT worker per generator runs continuous `rfft` on all 7 signals.
3. A console reporter aggregates per-generator statistics and prints throughput.

## Observed results

| Mode | Generators | Signals | FFT window | Throughput | FFT rate |
| --- | --- | --- | --- | --- | --- |
| `throughput` | 2 (default) | 14 | 8192 samples | ~21 Gbit/s | ~40k FFT/s |
| `throughput` | 7 | 49 | 8192 samples | ~73 Gbit/s | ~140k FFT/s |
| `latency` | 2 (default) | 14 | 1024 samples | ~5.8 Gbit/s | ~88k FFT/s |

Throughput is measured as **FFT input signal payload** in gigabits per
second. This is the data consumed by the analysis path, not total DRAM bandwidth
or temporary array traffic. Scaling from 2 to 7 generators (`--generators 7`)
yields a 3.4x throughput increase by filling CPU headroom that the default
configuration leaves unused.

For reference, at 16-bit / 250 kS/s (one NI USB-6423-class channel), the
7-generator throughput result is enough payload for roughly 17,000 channels.

## Run

### GUI mode

```bash
python examples/fft_pipeline_demo/main.py
```

Opens the ImGui dashboard. Click channel buttons in the sidebar to arm FFT
extraction per signal. Plots show downsampled raw traces with dominant-frequency
reconstruction overlays. Metric cards report aggregate and per-channel throughput.

### Headless throughput benchmark

```bash
python examples/fft_pipeline_demo/main.py --headless --mode throughput --duration 10 --report-interval 1
```

### Headless latency benchmark

```bash
python examples/fft_pipeline_demo/main.py --headless --mode latency --duration 10 --report-interval 1
```

### Scaled-up benchmark (more generators)

The default run uses 2 generator/FFT-worker pairs and typically reaches ~21 Gbit/s
at around 30% CPU utilization. Use `--generators` to add more pairs and push
throughput higher:

```bash
python examples/fft_pipeline_demo/main.py --headless --mode throughput --generators 4 --duration 10 --report-interval 1
```

Each `--generators N` value creates N data producers and N FFT consumers
(plus one console reporter), so `--generators 4` spawns 9 worker processes.

### Custom FFT window sweep

```bash
python examples/fft_pipeline_demo/main.py --headless --frame-rows 4096 --duration 10 --report-interval 1
```

## Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--headless` / `--no-gui` | off | Disable ImGui, print benchmark stats to stdout |
| `--mode throughput` | `throughput` | Large FFT window (8192 rows) for maximum payload |
| `--mode latency` | | Small FFT window (1024 rows) for shorter acquisition |
| `--generators N` | 2 | Number of generator/FFT-worker pairs in headless mode |
| `--frame-rows N` | (from mode) | Override the FFT frame length for custom tradeoff sweeps |
| `--duration SEC` | unlimited | Stop after a fixed interval |
| `--report-interval SEC` | 1.0 | Control the print cadence in headless mode |

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Pipeline definition, task functions, CLI, and ImGui dashboard |
| `graphing.py` | ImGui draw-list graphing helper for signal traces |

## How to interpret the benchmark modes

- **Throughput** answers: what is the maximum sustained FFT signal payload this
  pipeline can digest? Use this for peak headline numbers.
- **Latency** answers: how short can the FFT window be while still keeping very
  high FFT turnover? Use this when the acquisition window matters more than peak
  payload.
- **`--frame-rows`** lets you sweep the tradeoff between window length, FFT
  count, and payload throughput at any point between the two presets.
