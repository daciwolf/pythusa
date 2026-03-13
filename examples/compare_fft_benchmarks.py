from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
FINAL_RE = re.compile(
    r"final_total_fft_bytes=(?P<bytes>\d+)\s+"
    r"final_total_fft_throughput=(?P<throughput>[0-9.]+)\s+MB/s\s+"
    r"implementation=(?P<implementation>\w+)"
)

COMMON_ENV = {
    "FFT_BENCH_ROWS": "8192",
    "FFT_BENCH_CONSUMERS": "4",
    "FFT_BENCH_COLS_PER_CONSUMER": "2",
    "FFT_BENCH_REPORT_INTERVAL_S": "0.5",
    "FFT_BENCH_DURATION_S": "4.0",
    "FFT_BENCH_PERIODIC_REPORTS": "0",
    "PYTHONUNBUFFERED": "1",
}

PYTHUSA_CANDIDATES = [
    {"name": "spin_ring64", "FFT_BENCH_RING_DEPTH": "64", "FFT_BENCH_IDLE_SLEEP_S": "0"},
    {"name": "spin_ring256", "FFT_BENCH_RING_DEPTH": "256", "FFT_BENCH_IDLE_SLEEP_S": "0"},
    {"name": "spin_ring1024", "FFT_BENCH_RING_DEPTH": "1024", "FFT_BENCH_IDLE_SLEEP_S": "0"},
    {"name": "sleep1e-7_ring256", "FFT_BENCH_RING_DEPTH": "256", "FFT_BENCH_IDLE_SLEEP_S": "0.0000001"},
    {"name": "sleep1e-6_ring256", "FFT_BENCH_RING_DEPTH": "256", "FFT_BENCH_IDLE_SLEEP_S": "0.000001"},
]

JOBLIB_CANDIDATES = [
    {"name": "threads_q64", "FFT_BENCH_JOBLIB_PREFER": "threads", "FFT_BENCH_JOBLIB_QUEUE_DEPTH": "64"},
    {"name": "threads_q256", "FFT_BENCH_JOBLIB_PREFER": "threads", "FFT_BENCH_JOBLIB_QUEUE_DEPTH": "256"},
    {"name": "threads_q1024", "FFT_BENCH_JOBLIB_PREFER": "threads", "FFT_BENCH_JOBLIB_QUEUE_DEPTH": "1024"},
]


@dataclass
class TrialResult:
    implementation: str
    name: str
    throughput_mb_s: float
    total_fft_bytes: int
    stdout: str


def _run_trial(script_name: str, candidate: dict[str, str]) -> TrialResult:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update({key: value for key, value in candidate.items() if key != "name"})
    if script_name == "numpy_fft_pipeline.py":
        env["FFT_BENCH_RING_PREFIX"] = f"fft_bench_{uuid.uuid4().hex[:8]}"

    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script_name)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script_name} candidate {candidate['name']} failed with code {proc.returncode}\n{output}"
        )

    match = FINAL_RE.search(output)
    if match is None:
        raise RuntimeError(
            f"{script_name} candidate {candidate['name']} did not emit a final summary line\n{output}"
        )

    return TrialResult(
        implementation=match.group("implementation"),
        name=candidate["name"],
        throughput_mb_s=float(match.group("throughput")),
        total_fft_bytes=int(match.group("bytes")),
        stdout=output,
    )


def _run_suite(script_name: str, candidates: list[dict[str, str]]) -> list[TrialResult]:
    results: list[TrialResult] = []
    for candidate in candidates:
        result = _run_trial(script_name, candidate)
        results.append(result)
        print(
            f"{result.implementation:7s} {result.name:18s} "
            f"{result.throughput_mb_s:10.2f} MB/s total_fft_bytes={result.total_fft_bytes}"
        )
    return results


def main() -> None:
    print("Running staged FFT pipeline benchmark comparison")
    print(
        "Workload: "
        f"rows={COMMON_ENV['FFT_BENCH_ROWS']} "
        f"consumers={COMMON_ENV['FFT_BENCH_CONSUMERS']} "
        f"cols_per_consumer={COMMON_ENV['FFT_BENCH_COLS_PER_CONSUMER']} "
        f"duration={COMMON_ENV['FFT_BENCH_DURATION_S']}s"
    )
    print("Metric: bytes that actually went through np.fft.rfft on the consumer side")
    print()

    pythusa_results = _run_suite("numpy_fft_pipeline.py", PYTHUSA_CANDIDATES)
    print()
    joblib_results = _run_suite("joblib_fft_pipeline.py", JOBLIB_CANDIDATES)
    print()

    best_pythusa = max(pythusa_results, key=lambda result: result.throughput_mb_s)
    best_joblib = max(joblib_results, key=lambda result: result.throughput_mb_s)
    winner = max((best_pythusa, best_joblib), key=lambda result: result.throughput_mb_s)
    loser = best_joblib if winner is best_pythusa else best_pythusa
    speedup = winner.throughput_mb_s / loser.throughput_mb_s if loser.throughput_mb_s else float("inf")

    print("Best per implementation:")
    print(
        f"pythusa {best_pythusa.name:18s} {best_pythusa.throughput_mb_s:10.2f} MB/s "
        f"total_fft_bytes={best_pythusa.total_fft_bytes}"
    )
    print(
        f"joblib  {best_joblib.name:18s} {best_joblib.throughput_mb_s:10.2f} MB/s "
        f"total_fft_bytes={best_joblib.total_fft_bytes}"
    )
    print()
    print(
        f"winner={winner.implementation} candidate={winner.name} "
        f"throughput={winner.throughput_mb_s:.2f} MB/s "
        f"speedup_vs_other_best={speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
