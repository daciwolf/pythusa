from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import time

import numpy as np

try:
    import numba as nb
except ImportError as exc:  # pragma: no cover - exercised manually.
    raise SystemExit(
        "Numba is not installed. Run `python -m pip install -e .[benchmarks]` "
        "or `python -m pip install numba` and try again."
    ) from exc


MEGABYTE = 1_000_000.0
DTYPE = np.dtype(np.float32)
ROWS = int(os.environ.get("NUMBA_BENCH_ROWS", "16384"))
CHANNELS = int(os.environ.get("NUMBA_BENCH_CHANNELS", "2"))
DURATION_S = float(os.environ.get("NUMBA_BENCH_DURATION_S", "1.5"))
VALID_KERNELS = ("signal_fill", "window", "fir32")
DEFAULT_KERNELS = tuple(
    name.strip()
    for name in os.environ.get("NUMBA_BENCH_KERNELS", ",".join(VALID_KERNELS)).split(",")
    if name.strip()
)


@dataclass(frozen=True)
class BenchmarkResult:
    kernel: str
    backend: str
    compile_ms: float
    throughput_mb_s: float
    mean_ms: float
    p95_ms: float
    max_ms: float
    speedup_vs_numpy: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Numba against NumPy on kernels where Numba might help."
    )
    parser.add_argument(
        "--kernels",
        help="Comma-separated subset, for example 'signal_fill,fir32'.",
    )
    return parser.parse_args()


def _validate_kernels(kernel_names: tuple[str, ...]) -> None:
    unknown = sorted(set(kernel_names) - set(VALID_KERNELS))
    if unknown:
        raise SystemExit(f"Unknown kernels: {', '.join(unknown)}")


@nb.njit(cache=True, fastmath=True)
def _signal_fill_numba_serial(out: np.ndarray, channel_bases: np.ndarray, phase: np.float32) -> None:
    rows, channels = out.shape
    for row_index in range(rows):
        for channel_index in range(channels):
            angle = channel_bases[row_index, channel_index] + phase
            if (channel_index & 1) == 0:
                out[row_index, channel_index] = math.sin(angle)
            else:
                out[row_index, channel_index] = math.cos(angle)


@nb.njit(cache=True, fastmath=True, parallel=True)
def _signal_fill_numba_parallel(out: np.ndarray, channel_bases: np.ndarray, phase: np.float32) -> None:
    rows, channels = out.shape
    for row_index in nb.prange(rows):
        for channel_index in range(channels):
            angle = channel_bases[row_index, channel_index] + phase
            if (channel_index & 1) == 0:
                out[row_index, channel_index] = math.sin(angle)
            else:
                out[row_index, channel_index] = math.cos(angle)


@nb.njit(cache=True, fastmath=True)
def _window_numba_serial(out: np.ndarray, frame: np.ndarray, window: np.ndarray) -> None:
    rows, channels = frame.shape
    for row_index in range(rows):
        weight = window[row_index]
        for channel_index in range(channels):
            out[row_index, channel_index] = frame[row_index, channel_index] * weight


@nb.njit(cache=True, fastmath=True, parallel=True)
def _window_numba_parallel(out: np.ndarray, frame: np.ndarray, window: np.ndarray) -> None:
    rows, channels = frame.shape
    for row_index in nb.prange(rows):
        weight = window[row_index]
        for channel_index in range(channels):
            out[row_index, channel_index] = frame[row_index, channel_index] * weight


@nb.njit(cache=True, fastmath=True)
def _fir32_numba_serial(out: np.ndarray, frame: np.ndarray, taps: np.ndarray) -> None:
    rows, channels = frame.shape
    tap_count = taps.shape[0]
    same_start = (tap_count - 1) // 2
    for row_index in range(rows):
        full_index = row_index + same_start
        for channel_index in range(channels):
            acc = 0.0
            for tap_index in range(tap_count):
                source_index = full_index - tap_index
                if 0 <= source_index < rows:
                    acc += frame[source_index, channel_index] * taps[tap_index]
            out[row_index, channel_index] = acc


@nb.njit(cache=True, fastmath=True, parallel=True)
def _fir32_numba_parallel(out: np.ndarray, frame: np.ndarray, taps: np.ndarray) -> None:
    rows, channels = frame.shape
    tap_count = taps.shape[0]
    same_start = (tap_count - 1) // 2
    for row_index in nb.prange(rows):
        full_index = row_index + same_start
        for channel_index in range(channels):
            acc = 0.0
            for tap_index in range(tap_count):
                source_index = full_index - tap_index
                if 0 <= source_index < rows:
                    acc += frame[source_index, channel_index] * taps[tap_index]
            out[row_index, channel_index] = acc


def _make_signal_fill_numpy(channel_bases: np.ndarray):
    angle = np.empty(ROWS, dtype=DTYPE)

    def _run(out: np.ndarray, phase: np.float32) -> None:
        for channel_index in range(CHANNELS):
            np.add(channel_bases[:, channel_index], phase, out=angle)
            if (channel_index & 1) == 0:
                np.sin(angle, out=out[:, channel_index])
            else:
                np.cos(angle, out=out[:, channel_index])

    return _run


def _make_window_numpy(window_2d: np.ndarray):
    def _run(out: np.ndarray, frame: np.ndarray) -> None:
        np.multiply(frame, window_2d, out=out)

    return _run


def _make_fir32_numpy(taps: np.ndarray):
    def _run(out: np.ndarray, frame: np.ndarray) -> None:
        for channel_index in range(CHANNELS):
            out[:, channel_index] = np.convolve(frame[:, channel_index], taps, mode="same")

    return _run


def _quantiles_ms(latencies_ns: list[int]) -> tuple[float, float, float]:
    samples_ms = np.asarray(latencies_ns, dtype=np.float64) / 1_000_000.0
    return (
        float(samples_ms.mean()),
        float(np.percentile(samples_ms, 95)),
        float(samples_ms.max()),
    )


def _benchmark_signal_fill(
    backend: str,
    output: np.ndarray,
    channel_bases: np.ndarray,
    numpy_impl,
    compile_costs_ms: dict[tuple[str, str], float],
) -> tuple[float, float, float, float]:
    phase = np.float32(0.0)
    phase_step = np.float32(0.2)
    compile_ms = compile_costs_ms.get(("signal_fill", backend), 0.0)

    if backend == "numpy":
        runner = lambda: numpy_impl(output, phase)  # noqa: E731
    elif backend == "numba_serial":
        runner = lambda: _signal_fill_numba_serial(output, channel_bases, phase)  # noqa: E731
    elif backend == "numba_parallel":
        runner = lambda: _signal_fill_numba_parallel(output, channel_bases, phase)  # noqa: E731
    else:  # pragma: no cover - kept defensive for backend selection.
        raise ValueError(backend)

    latencies_ns: list[int] = []
    started_at_ns = time.perf_counter_ns()
    iterations = 0
    while (time.perf_counter_ns() - started_at_ns) < int(DURATION_S * 1_000_000_000):
        op_started_at = time.perf_counter_ns()
        runner()
        latencies_ns.append(time.perf_counter_ns() - op_started_at)
        iterations += 1
        phase = np.float32(phase + phase_step)
    elapsed_s = (time.perf_counter_ns() - started_at_ns) / 1_000_000_000.0

    throughput_mb_s = (iterations * output.nbytes) / elapsed_s / MEGABYTE
    mean_ms, p95_ms, max_ms = _quantiles_ms(latencies_ns)
    return compile_ms, throughput_mb_s, mean_ms, p95_ms, max_ms


def _benchmark_frame_kernel(
    backend: str,
    kernel_name: str,
    output: np.ndarray,
    frame: np.ndarray,
    window: np.ndarray,
    taps: np.ndarray,
    numpy_window_impl,
    numpy_fir_impl,
    compile_costs_ms: dict[tuple[str, str], float],
) -> tuple[float, float, float, float]:
    compile_ms = compile_costs_ms.get((kernel_name, backend), 0.0)

    if kernel_name == "window":
        if backend == "numpy":
            runner = lambda: numpy_window_impl(output, frame)  # noqa: E731
        elif backend == "numba_serial":
            runner = lambda: _window_numba_serial(output, frame, window)  # noqa: E731
        elif backend == "numba_parallel":
            runner = lambda: _window_numba_parallel(output, frame, window)  # noqa: E731
        else:  # pragma: no cover - kept defensive for backend selection.
            raise ValueError(backend)
    elif kernel_name == "fir32":
        if backend == "numpy":
            runner = lambda: numpy_fir_impl(output, frame)  # noqa: E731
        elif backend == "numba_serial":
            runner = lambda: _fir32_numba_serial(output, frame, taps)  # noqa: E731
        elif backend == "numba_parallel":
            runner = lambda: _fir32_numba_parallel(output, frame, taps)  # noqa: E731
        else:  # pragma: no cover - kept defensive for backend selection.
            raise ValueError(backend)
    else:  # pragma: no cover - kept defensive for kernel selection.
        raise ValueError(kernel_name)

    latencies_ns: list[int] = []
    started_at_ns = time.perf_counter_ns()
    iterations = 0
    while (time.perf_counter_ns() - started_at_ns) < int(DURATION_S * 1_000_000_000):
        op_started_at = time.perf_counter_ns()
        runner()
        latencies_ns.append(time.perf_counter_ns() - op_started_at)
        iterations += 1
    elapsed_s = (time.perf_counter_ns() - started_at_ns) / 1_000_000_000.0

    throughput_mb_s = (iterations * frame.nbytes) / elapsed_s / MEGABYTE
    mean_ms, p95_ms, max_ms = _quantiles_ms(latencies_ns)
    return compile_ms, throughput_mb_s, mean_ms, p95_ms, max_ms


def _validate_correctness(
    channel_bases: np.ndarray,
    frame: np.ndarray,
    window: np.ndarray,
    taps: np.ndarray,
) -> None:
    signal_reference = np.empty((ROWS, CHANNELS), dtype=DTYPE)
    signal_fill_numpy = _make_signal_fill_numpy(channel_bases)
    signal_fill_numpy(signal_reference, np.float32(0.5))

    signal_serial = np.empty_like(signal_reference)
    signal_parallel = np.empty_like(signal_reference)
    _signal_fill_numba_serial(signal_serial, channel_bases, np.float32(0.5))
    _signal_fill_numba_parallel(signal_parallel, channel_bases, np.float32(0.5))
    if not np.allclose(signal_reference, signal_serial, rtol=1e-5, atol=1e-6):
        raise RuntimeError("signal_fill numba_serial diverged from numpy")
    if not np.allclose(signal_reference, signal_parallel, rtol=1e-5, atol=1e-6):
        raise RuntimeError("signal_fill numba_parallel diverged from numpy")

    window_reference = np.empty((ROWS, CHANNELS), dtype=DTYPE)
    window_serial = np.empty_like(window_reference)
    window_parallel = np.empty_like(window_reference)
    _make_window_numpy(window[:, None])(window_reference, frame)
    _window_numba_serial(window_serial, frame, window)
    _window_numba_parallel(window_parallel, frame, window)
    if not np.allclose(window_reference, window_serial, rtol=1e-5, atol=1e-6):
        raise RuntimeError("window numba_serial diverged from numpy")
    if not np.allclose(window_reference, window_parallel, rtol=1e-5, atol=1e-6):
        raise RuntimeError("window numba_parallel diverged from numpy")

    fir_reference = np.empty((ROWS, CHANNELS), dtype=DTYPE)
    fir_serial = np.empty_like(fir_reference)
    fir_parallel = np.empty_like(fir_reference)
    _make_fir32_numpy(taps)(fir_reference, frame)
    _fir32_numba_serial(fir_serial, frame, taps)
    _fir32_numba_parallel(fir_parallel, frame, taps)
    if not np.allclose(fir_reference, fir_serial, rtol=1e-4, atol=1e-5):
        raise RuntimeError("fir32 numba_serial diverged from numpy")
    if not np.allclose(fir_reference, fir_parallel, rtol=1e-4, atol=1e-5):
        raise RuntimeError("fir32 numba_parallel diverged from numpy")


def _measure_numba_compile_costs(
    channel_bases: np.ndarray,
    frame: np.ndarray,
    window: np.ndarray,
    taps: np.ndarray,
) -> dict[tuple[str, str], float]:
    compile_costs_ms: dict[tuple[str, str], float] = {}
    output = np.empty((ROWS, CHANNELS), dtype=DTYPE)
    phase = np.float32(0.0)
    compile_targets = (
        ("signal_fill", "numba_serial", lambda: _signal_fill_numba_serial(output, channel_bases, phase)),
        ("signal_fill", "numba_parallel", lambda: _signal_fill_numba_parallel(output, channel_bases, phase)),
        ("window", "numba_serial", lambda: _window_numba_serial(output, frame, window)),
        ("window", "numba_parallel", lambda: _window_numba_parallel(output, frame, window)),
        ("fir32", "numba_serial", lambda: _fir32_numba_serial(output, frame, taps)),
        ("fir32", "numba_parallel", lambda: _fir32_numba_parallel(output, frame, taps)),
    )
    for kernel_name, backend, runner in compile_targets:
        started_at = time.perf_counter_ns()
        runner()
        compile_costs_ms[(kernel_name, backend)] = (time.perf_counter_ns() - started_at) / 1_000_000.0
    return compile_costs_ms


def main() -> None:
    args = _parse_args()
    selected_kernels = DEFAULT_KERNELS
    if args.kernels is not None:
        selected_kernels = tuple(name.strip() for name in args.kernels.split(",") if name.strip())
    _validate_kernels(selected_kernels)

    rng = np.random.default_rng(0)
    base_t = np.linspace(0.0, np.float32(2.0 * np.pi), ROWS, endpoint=False, dtype=DTYPE)
    channel_freqs = np.arange(1, CHANNELS + 1, dtype=DTYPE)
    channel_bases = np.multiply.outer(base_t, channel_freqs).astype(DTYPE, copy=False)
    frame = rng.standard_normal((ROWS, CHANNELS), dtype=DTYPE)
    window = np.hanning(ROWS).astype(DTYPE)
    taps = np.hanning(32).astype(DTYPE)
    taps /= taps.sum()

    compile_costs_ms = _measure_numba_compile_costs(channel_bases, frame, window, taps)
    _validate_correctness(channel_bases, frame, window, taps)

    signal_fill_numpy = _make_signal_fill_numpy(channel_bases)
    window_numpy = _make_window_numpy(window[:, None])
    fir_numpy = _make_fir32_numpy(taps)
    backends = ("numpy", "numba_serial", "numba_parallel")
    results: list[BenchmarkResult] = []

    print("PYTHUSA Numba Candidate Benchmark")
    print(
        f"config rows={ROWS} channels={CHANNELS} dtype={DTYPE.name} "
        f"duration={DURATION_S:.1f}s numba_threads={nb.get_num_threads()}"
    )
    print(
        "kernel         backend            compile_ms throughput   mean_ms    p95_ms    max_ms  speedup"
    )

    for kernel_name in selected_kernels:
        kernel_results: list[BenchmarkResult] = []
        for backend in backends:
            output = np.empty((ROWS, CHANNELS), dtype=DTYPE)
            if kernel_name == "signal_fill":
                compile_ms, throughput_mb_s, mean_ms, p95_ms, max_ms = _benchmark_signal_fill(
                    backend=backend,
                    output=output,
                    channel_bases=channel_bases,
                    numpy_impl=signal_fill_numpy,
                    compile_costs_ms=compile_costs_ms,
                )
            else:
                compile_ms, throughput_mb_s, mean_ms, p95_ms, max_ms = _benchmark_frame_kernel(
                    backend=backend,
                    kernel_name=kernel_name,
                    output=output,
                    frame=frame,
                    window=window,
                    taps=taps,
                    numpy_window_impl=window_numpy,
                    numpy_fir_impl=fir_numpy,
                    compile_costs_ms=compile_costs_ms,
                )
            kernel_results.append(
                BenchmarkResult(
                    kernel=kernel_name,
                    backend=backend,
                    compile_ms=compile_ms,
                    throughput_mb_s=throughput_mb_s,
                    mean_ms=mean_ms,
                    p95_ms=p95_ms,
                    max_ms=max_ms,
                    speedup_vs_numpy=0.0,
                )
            )

        numpy_throughput = next(
            result.throughput_mb_s for result in kernel_results if result.backend == "numpy"
        )
        for result in kernel_results:
            speedup = result.throughput_mb_s / numpy_throughput if numpy_throughput > 0.0 else 0.0
            updated = BenchmarkResult(
                kernel=result.kernel,
                backend=result.backend,
                compile_ms=result.compile_ms,
                throughput_mb_s=result.throughput_mb_s,
                mean_ms=result.mean_ms,
                p95_ms=result.p95_ms,
                max_ms=result.max_ms,
                speedup_vs_numpy=speedup,
            )
            results.append(updated)
            print(
                f"{updated.kernel:<14} {updated.backend:<16} "
                f"{updated.compile_ms:10.2f} {updated.throughput_mb_s:10.2f} "
                f"{updated.mean_ms:9.3f} {updated.p95_ms:9.3f} {updated.max_ms:9.3f} "
                f"{updated.speedup_vs_numpy:8.2f}x"
            )

    print("note: compile_ms is the one-time JIT cost for Numba and is excluded from throughput timing.")


if __name__ == "__main__":
    main()
