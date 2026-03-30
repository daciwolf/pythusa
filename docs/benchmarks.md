# Benchmark Guide

This directory contains reproducible benchmark runners. Each script does one job:

- `dsp_benchmark_suite.py`: general DSP throughput, latency, and memory.
- `compare_fft_benchmarks.py`: staged FFT comparison between PYTHUSA and joblib.
- `rocketdata_test.py`: sample-rate and window-size driven detection-latency budgeting.
- `numba_candidate_benchmark.py`: backend experiment for user-side loop acceleration.

## Canonical Commands

Use these commands for repeatable release and README measurements:

```bash
python benchmarks/dsp_benchmark_suite.py --balanced --json-out benchmarks/results/dsp-balanced.json
python benchmarks/dsp_benchmark_suite.py --latency-min --kernels rfft,power_spectrum,stft --json-out benchmarks/results/dsp-latency-fft.json
python benchmarks/dsp_benchmark_suite.py --balanced --graph --graph-out benchmarks/results/dsp-balanced-heatmaps.png --no-show
python benchmarks/compare_fft_benchmarks.py --json-out benchmarks/results/fft-compare.json
python benchmarks/rocketdata_test.py --json-out benchmarks/results/rocket-latency.json
```

## Structured Output

The main benchmark runners support:

- `--json`: print structured JSON to stdout instead of the text table.
- `--json-out PATH`: write structured JSON to a file.
- `--label NAME`: attach a run label to the structured output.
- `--graph`: render DSP suite heatmaps after the benchmark completes.
- `--graph-out PATH`: save those heatmaps to an image file.
- `--no-show`: build graph output without opening a matplotlib window.

The recommended result filename pattern is:

```text
benchmarks/results/<benchmark>-<profile-or-focus>.json
```

Examples:

- `benchmarks/results/dsp-balanced.json`
- `benchmarks/results/dsp-latency-fft.json`
- `benchmarks/results/rocket-latency.json`

## Measurement Notes

- DSP suite latency fields are processing-side latencies from "frame ready" to "consumer finished".
- Rocket benchmark `total_*_detection_ms` fields add window fill time to that processing latency.
- `task_rss_mb` is summed worker RSS and can overcount shared-memory mappings.
- Benchmark outputs are machine-dependent; compare like-for-like configurations.
