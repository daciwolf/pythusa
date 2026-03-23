from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from _reporting import build_payload, emit_payload


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
DSP_SUITE = BENCHMARKS / "dsp_benchmark_suite.py"
DEFAULT_SAMPLE_RATES = (100_000, 250_000, 1_000_000)
DEFAULT_WINDOWS = (512, 1024, 2048, 4096)
DEFAULT_KERNELS = ("rfft", "power_spectrum", "stft")
VALID_MODES = ("throughput", "balanced", "latency")


@dataclass(frozen=True)
class RocketResult:
    kernel: str
    sample_rate_hz: int
    window_samples: int
    window_fill_ms: float
    processing_mean_ms: float
    processing_p95_ms: float
    processing_p99_ms: float
    processing_max_ms: float
    total_mean_detection_ms: float
    total_p95_detection_ms: float
    total_p99_detection_ms: float
    total_max_detection_ms: float
    throughput_mb_s: float
    batches: int
    ring_reserved_mb: float


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise SystemExit("Expected at least one integer value.")
    return values


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise SystemExit("Expected at least one non-empty string value.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rocket-style latency benchmarks derived from the DSP suite."
    )
    parser.add_argument(
        "--sample-rates",
        default=",".join(str(value) for value in DEFAULT_SAMPLE_RATES),
        help="Comma-separated sample rates in Hz. Default: 100000,250000,1000000.",
    )
    parser.add_argument(
        "--windows",
        default=",".join(str(value) for value in DEFAULT_WINDOWS),
        help="Comma-separated FFT window sizes in samples. Default: 512,1024,2048,4096.",
    )
    parser.add_argument(
        "--kernels",
        default=",".join(DEFAULT_KERNELS),
        help="Comma-separated kernel subset to run. Default: rfft,power_spectrum,stft.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="latency",
        help="DSP suite mode to use for the underlying runs. Default: latency.",
    )
    parser.add_argument(
        "--pipelines",
        type=int,
        default=4,
        help="Number of parallel pipelines passed through to the DSP suite. Default: 4.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Channel count passed through to the DSP suite. Default: 2.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=1.0,
        help="Per-run benchmark duration in seconds. Default: 1.0.",
    )
    parser.add_argument(
        "--ring-depth",
        type=int,
        help="Optional ring depth override for the underlying DSP suite.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        help="Optional max in-flight override for the underlying DSP suite.",
    )
    parser.add_argument(
        "--label",
        help="Optional label stored in structured output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary to stdout instead of the human-readable table.",
    )
    parser.add_argument(
        "--json-out",
        help="Write the JSON summary to this file path.",
    )
    return parser.parse_args()


def _run_dsp_suite(
    *,
    window_samples: int,
    kernels: tuple[str, ...],
    mode: str,
    pipelines: int,
    channels: int,
    duration_s: float,
    ring_depth: int | None,
    max_in_flight: int | None,
) -> dict[str, object]:
    env = os.environ.copy()
    env["DSP_BENCH_ROWS"] = str(window_samples)
    env["DSP_BENCH_CHANNELS"] = str(channels)
    env["DSP_BENCH_PIPELINES"] = str(pipelines)
    env["DSP_BENCH_DURATION_S"] = str(duration_s)
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        str(DSP_SUITE),
        "--mode",
        mode,
        "--kernels",
        ",".join(kernels),
        "--json",
    ]
    if ring_depth is not None:
        command.extend(["--ring-depth", str(ring_depth)])
    if max_in_flight is not None:
        command.extend(["--max-in-flight", str(max_in_flight)])

    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"dsp_benchmark_suite.py failed for window={window_samples} with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "dsp_benchmark_suite.py did not emit valid JSON.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        ) from exc


def _rocket_results_for_window(
    *,
    sample_rates_hz: tuple[int, ...],
    window_samples: int,
    suite_payload: dict[str, object],
) -> list[RocketResult]:
    results: list[RocketResult] = []
    suite_results = suite_payload["results"]
    assert isinstance(suite_results, list)
    for sample_rate_hz in sample_rates_hz:
        window_fill_ms = (window_samples / sample_rate_hz) * 1000.0
        for suite_result in suite_results:
            assert isinstance(suite_result, dict)
            results.append(
                RocketResult(
                    kernel=str(suite_result["kernel"]),
                    sample_rate_hz=sample_rate_hz,
                    window_samples=window_samples,
                    window_fill_ms=window_fill_ms,
                    processing_mean_ms=float(suite_result["latency_mean_ms"]),
                    processing_p95_ms=float(suite_result["latency_p95_ms"]),
                    processing_p99_ms=float(suite_result["latency_p99_ms"]),
                    processing_max_ms=float(suite_result["latency_max_ms"]),
                    total_mean_detection_ms=window_fill_ms + float(suite_result["latency_mean_ms"]),
                    total_p95_detection_ms=window_fill_ms + float(suite_result["latency_p95_ms"]),
                    total_p99_detection_ms=window_fill_ms + float(suite_result["latency_p99_ms"]),
                    total_max_detection_ms=window_fill_ms + float(suite_result["latency_max_ms"]),
                    throughput_mb_s=float(suite_result["throughput_mb_s"]),
                    batches=int(suite_result["batches"]),
                    ring_reserved_mb=float(suite_result["ring_reserved_mb"]),
                )
            )
    return results


def _print_header(args: argparse.Namespace) -> None:
    print("PYTHUSA Rocket Data Benchmark")
    print(
        "config "
        f"sample_rates_hz={','.join(str(v) for v in _parse_csv_ints(args.sample_rates))} "
        f"windows={','.join(str(v) for v in _parse_csv_ints(args.windows))} "
        f"kernels={','.join(_parse_csv_strings(args.kernels))} "
        f"mode={args.mode} pipelines={args.pipelines} channels={args.channels} "
        f"duration={args.duration_s:.1f}s"
    )
    print(
        "kernel         sample_hz  window  fill_ms  proc_mean  total_mean   proc_p95   total_p95 throughput   ring_mb"
    )


def _print_result(result: RocketResult) -> None:
    print(
        f"{result.kernel:14s}"
        f"{result.sample_rate_hz:11d}"
        f"{result.window_samples:8d}"
        f"{result.window_fill_ms:9.3f}"
        f"{result.processing_mean_ms:11.3f}"
        f"{result.total_mean_detection_ms:12.3f}"
        f"{result.processing_p95_ms:11.3f}"
        f"{result.total_p95_detection_ms:12.3f}"
        f"{result.throughput_mb_s:11.2f}"
        f"{result.ring_reserved_mb:10.1f}"
    )


def main() -> None:
    args = _parse_args()
    sample_rates_hz = _parse_csv_ints(args.sample_rates)
    window_sizes = _parse_csv_ints(args.windows)
    kernels = _parse_csv_strings(args.kernels)

    all_results: list[RocketResult] = []
    runs: list[dict[str, object]] = []
    if not args.json:
        _print_header(args)

    for window_samples in window_sizes:
        suite_payload = _run_dsp_suite(
            window_samples=window_samples,
            kernels=kernels,
            mode=args.mode,
            pipelines=args.pipelines,
            channels=args.channels,
            duration_s=args.duration_s,
            ring_depth=args.ring_depth,
            max_in_flight=args.max_in_flight,
        )
        run_results = _rocket_results_for_window(
            sample_rates_hz=sample_rates_hz,
            window_samples=window_samples,
            suite_payload=suite_payload,
        )
        all_results.extend(run_results)
        runs.append(
            {
                "window_samples": window_samples,
                "suite_config": suite_payload["config"],
                "suite_results": suite_payload["results"],
            }
        )
        if not args.json:
            for result in run_results:
                _print_result(result)

    payload = build_payload(
        benchmark="rocketdata_test",
        label=args.label,
        config={
            "sample_rates_hz": list(sample_rates_hz),
            "windows": list(window_sizes),
            "kernels": list(kernels),
            "mode": args.mode,
            "pipelines": args.pipelines,
            "channels": args.channels,
            "duration_s": args.duration_s,
            "ring_depth_override": args.ring_depth,
            "max_in_flight_override": args.max_in_flight,
        },
        results=all_results,
        notes=(
            "total_*_detection_ms is window fill time plus processing-side latency from dsp_benchmark_suite.",
            "This benchmark is aimed at sensor-style detection latency budgeting, not hard real-time guarantees.",
        ),
        summary={"runs": runs},
    )
    emit_payload(payload, json_stdout=args.json, json_out=args.json_out)
    if not args.json and args.json_out is not None:
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
