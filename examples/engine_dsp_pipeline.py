from __future__ import annotations

"""Run with: python examples/engine_dsp_pipeline.py"""

import csv
import multiprocessing as mp
import time
from collections import deque
from functools import partial
from pathlib import Path
from queue import Empty

import matplotlib.pyplot as plt
import numpy as np

import pythusa


ROOT = Path(__file__).resolve().parents[1]
ENGINE_DATA_DIR = ROOT / "engine_data"
GSE_PATH = ENGINE_DATA_DIR / "gse-vtf-5.csv"
ECU_PATH = ENGINE_DATA_DIR / "ecu_vtf5.csv"

TARGET_SAMPLE_RATE_HZ = 250_000.0
SEGMENT_DURATION_S = 10.0
FRAME_LENGTH = 256
CHANNEL_COUNT = 4
SPECTRUM_BINS = FRAME_LENGTH // 2 + 1
META_FIELDS = 2
IDLE_SLEEP_S = 0.0
REPORT_EVERY_FRAMES = 96
THROUGHPUT_WINDOW_FRAMES = 128
FRAME_PERIOD_S = FRAME_LENGTH / TARGET_SAMPLE_RATE_HZ

_PRESSURE_TAPS = np.hanning(31).astype(np.float32)
_PRESSURE_TAPS /= _PRESSURE_TAPS.sum()
_VIBRATION_BASELINE_TAPS = np.hanning(63).astype(np.float32)
_VIBRATION_BASELINE_TAPS /= _VIBRATION_BASELINE_TAPS.sum()
_FFT_WINDOW = np.hanning(FRAME_LENGTH).astype(np.float32)
_FREQ_AXIS_HZ = np.fft.rfftfreq(FRAME_LENGTH, d=1.0 / TARGET_SAMPLE_RATE_HZ).astype(np.float32)

_ENGINE_FRAMES_CACHE: np.ndarray | None = None
_TELEMETRY_CACHE: dict[str, object] | None = None


def _read_numeric_csv(path: Path, columns: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows: list[tuple[float, list[float]]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            timestamp = float(row["time_recv"])
            values = [float(row[column]) for column in columns]
            rows.append((timestamp, values))

    rows.sort(key=lambda item: item[0])
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    value_matrix = np.asarray([row[1] for row in rows], dtype=np.float64)

    unique_times, unique_index = np.unique(times, return_index=True)
    deduped = {
        column: value_matrix[unique_index, column_index]
        for column_index, column in enumerate(columns)
    }
    return unique_times, deduped


def _normalize(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    centered = series - np.mean(series)
    scale = np.std(centered)
    if scale <= 0.0:
        return np.zeros_like(centered)
    return centered / scale


def _segment_bounds(
    overlap_start: float,
    overlap_end: float,
    active_time: float,
) -> tuple[float, float]:
    half_window = SEGMENT_DURATION_S * 0.5
    segment_start = max(overlap_start, active_time - half_window)
    segment_end = segment_start + SEGMENT_DURATION_S
    if segment_end > overlap_end:
        segment_end = overlap_end
        segment_start = segment_end - SEGMENT_DURATION_S
    return segment_start, segment_end


def _telemetry_bundle() -> dict[str, object]:
    global _TELEMETRY_CACHE
    if _TELEMETRY_CACHE is not None:
        return _TELEMETRY_CACHE

    gse_columns = ("pressuregn2", "pressurevent", "pressureloxinjtee", "pressureloxmvas")
    ecu_columns = ("pressurelox", "pressurelng", "pressureinjectorlox", "pressureinjectorlng")

    gse_time, gse = _read_numeric_csv(GSE_PATH, gse_columns)
    ecu_time, ecu = _read_numeric_csv(ECU_PATH, ecu_columns)

    overlap_start = max(gse_time[0], ecu_time[0])
    overlap_end = min(gse_time[-1], ecu_time[-1])

    overlap_mask = (ecu_time >= overlap_start) & (ecu_time <= overlap_end)
    injector_activity = (
        np.asarray(ecu["pressureinjectorlox"])[overlap_mask]
        + np.asarray(ecu["pressureinjectorlng"])[overlap_mask]
    )
    active_index = int(np.argmax(injector_activity))
    active_time = ecu_time[overlap_mask][active_index]

    _TELEMETRY_CACHE = {
        "gse_time": gse_time,
        "gse": gse,
        "ecu_time": ecu_time,
        "ecu": ecu,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "active_time": active_time,
    }
    return _TELEMETRY_CACHE


def _resampled_engine_inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bundle = _telemetry_bundle()
    gse_time = bundle["gse_time"]
    ecu_time = bundle["ecu_time"]
    gse = bundle["gse"]
    ecu = bundle["ecu"]
    overlap_start = float(bundle["overlap_start"])
    overlap_end = float(bundle["overlap_end"])
    active_time = float(bundle["active_time"])

    segment_start, _segment_end = _segment_bounds(overlap_start, overlap_end, active_time)
    target_times = segment_start + (
        np.arange(int(TARGET_SAMPLE_RATE_HZ * SEGMENT_DURATION_S), dtype=np.float64)
        / TARGET_SAMPLE_RATE_HZ
    )

    signals = {
        "pressuregn2": np.interp(target_times, gse_time, gse["pressuregn2"]),
        "pressurevent": np.interp(target_times, gse_time, gse["pressurevent"]),
        "pressureloxinjtee": np.interp(target_times, gse_time, gse["pressureloxinjtee"]),
        "pressureloxmvas": np.interp(target_times, gse_time, gse["pressureloxmvas"]),
        "pressurelox": np.interp(target_times, ecu_time, ecu["pressurelox"]),
        "pressurelng": np.interp(target_times, ecu_time, ecu["pressurelng"]),
        "pressureinjectorlox": np.interp(target_times, ecu_time, ecu["pressureinjectorlox"]),
        "pressureinjectorlng": np.interp(target_times, ecu_time, ecu["pressureinjectorlng"]),
    }
    return target_times, signals


def _build_simulated_engine_samples() -> np.ndarray:
    target_times, telemetry = _resampled_engine_inputs()
    t = target_times - target_times[0]
    rng = np.random.default_rng(5)

    pressuregn2 = _normalize(telemetry["pressuregn2"])
    pressurevent = _normalize(telemetry["pressurevent"])
    pressureloxinjtee = _normalize(telemetry["pressureloxinjtee"])
    pressureloxmvas = _normalize(telemetry["pressureloxmvas"])
    pressurelox = np.asarray(telemetry["pressurelox"], dtype=np.float64)
    pressurelng = np.asarray(telemetry["pressurelng"], dtype=np.float64)
    pressureinjectorlox = np.asarray(telemetry["pressureinjectorlox"], dtype=np.float64)
    pressureinjectorlng = np.asarray(telemetry["pressureinjectorlng"], dtype=np.float64)

    injector_drive = np.abs(
        _normalize((pressureinjectorlox + pressureinjectorlng) - (pressurelox + pressurelng))
    )
    tank_drive = _normalize(np.gradient(pressureloxmvas + pressuregn2))

    tank_lox = (
        pressurelox
        + 0.25 * np.sin(2.0 * np.pi * 42.0 * t)
        + 0.15 * pressureloxinjtee
    )
    tank_lng = (
        pressurelng
        + 0.22 * np.sin(2.0 * np.pi * 51.0 * t + 0.2)
        + 0.10 * pressurevent
    )

    injector_base_hz = 1_650.0 + 190.0 * pressuregn2 + 140.0 * tank_drive
    injector_phase = 2.0 * np.pi * np.cumsum(injector_base_hz) / TARGET_SAMPLE_RATE_HZ

    injector_lox = (
        pressureinjectorlox
        + (0.45 + 0.65 * injector_drive) * np.sin(injector_phase)
        + 0.25 * np.sin(2.15 * injector_phase)
        + 0.04 * rng.standard_normal(t.size)
    )
    injector_lng = (
        pressureinjectorlng
        + (0.40 + 0.60 * injector_drive) * np.sin(injector_phase + 0.3)
        + 0.21 * np.sin(2.35 * injector_phase + 0.15)
        + 0.10 * pressureloxmvas
        + 0.04 * rng.standard_normal(t.size)
    )

    return np.column_stack(
        (
            tank_lox,
            tank_lng,
            injector_lox,
            injector_lng,
        )
    ).astype(np.float32)


def _engine_frames() -> np.ndarray:
    global _ENGINE_FRAMES_CACHE
    if _ENGINE_FRAMES_CACHE is None:
        samples = _build_simulated_engine_samples()
        usable = samples.shape[0] - (samples.shape[0] % FRAME_LENGTH)
        _ENGINE_FRAMES_CACHE = samples[:usable].reshape(-1, FRAME_LENGTH, CHANNEL_COUNT)
    return _ENGINE_FRAMES_CACHE


def _expected_frame_count() -> int:
    return int(TARGET_SAMPLE_RATE_HZ * SEGMENT_DURATION_S) // FRAME_LENGTH


def _engine_frame_nbytes() -> int:
    return FRAME_LENGTH * CHANNEL_COUNT * np.dtype(np.float32).itemsize


def _fir_same_into(frame: np.ndarray, taps: np.ndarray, out: np.ndarray) -> None:
    for channel_index in range(frame.shape[1]):
        out[:, channel_index] = np.convolve(frame[:, channel_index], taps, mode="same")


def _condition_frame_into(frame: np.ndarray, out: np.ndarray) -> None:
    _fir_same_into(frame[:, :2], _PRESSURE_TAPS, out[:, :2])
    for channel_index in range(2, frame.shape[1]):
        out[:, channel_index] = frame[:, channel_index] - np.convolve(
            frame[:, channel_index],
            _VIBRATION_BASELINE_TAPS,
            mode="same",
        )


def _condition_frame(frame: np.ndarray) -> np.ndarray:
    conditioned = np.empty_like(frame)
    _condition_frame_into(frame, conditioned)
    return conditioned


def _vibration_power_spectrum(frame: np.ndarray) -> np.ndarray:
    injector_delta = frame[:, 3] - frame[:, 2]
    injector_delta -= np.convolve(injector_delta, _VIBRATION_BASELINE_TAPS, mode="same")
    windowed = injector_delta * _FFT_WINDOW
    return np.square(np.abs(np.fft.rfft(windowed))).astype(np.float32, copy=False)

def _write_frame(writer, frame: np.ndarray) -> None:
    while not writer.write(frame):
        _idle_pause()


def _read_exact_into(reader, out: np.ndarray) -> np.ndarray:
    while not reader.read_into(out):
        _idle_pause()
    return out


def _idle_pause() -> None:
    if IDLE_SLEEP_S > 0.0:
        time.sleep(IDLE_SLEEP_S)


def _sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > 0.002:
            time.sleep(remaining * 0.5)
            continue
        if remaining > 0.0005:
            time.sleep(0)
            continue


def source_engine_frames(engine_frames, frame_meta) -> None:
    next_emit_at = time.perf_counter()
    meta = np.empty((META_FIELDS,), dtype=np.float64)

    for frame_index, frame in enumerate(_engine_frames()):
        _sleep_until(next_emit_at)
        emitted_at = time.perf_counter()
        meta[0] = float(frame_index)
        meta[1] = emitted_at
        _write_frame(engine_frames, frame)
        _write_frame(frame_meta, meta)
        next_emit_at += FRAME_PERIOD_S


def condition_engine_frames(engine_frames, frame_meta, conditioned_frames, conditioned_meta) -> None:
    frame = np.empty((FRAME_LENGTH, CHANNEL_COUNT), dtype=np.float32)
    meta = np.empty((META_FIELDS,), dtype=np.float64)
    conditioned = np.empty((FRAME_LENGTH, CHANNEL_COUNT), dtype=np.float32)

    for _ in range(_expected_frame_count()):
        _read_exact_into(engine_frames, frame)
        _read_exact_into(frame_meta, meta)
        _condition_frame_into(frame, conditioned)
        _write_frame(conditioned_frames, conditioned)
        _write_frame(conditioned_meta, meta)


def report_engine_state(
    conditioned_frames,
    conditioned_meta,
    *,
    metrics_queue,
) -> None:
    frame_index = 0
    last_tank_lox_mean = 0.0
    last_tank_lng_mean = 0.0
    last_injector_spread = 0.0
    last_peak_hz = 0.0
    arrival_window: deque[float] = deque(maxlen=THROUGHPUT_WINDOW_FRAMES)
    frame_nbytes = _engine_frame_nbytes()
    conditioned = np.empty((FRAME_LENGTH, CHANNEL_COUNT), dtype=np.float32)
    conditioned_meta_frame = np.empty((META_FIELDS,), dtype=np.float64)

    for _ in range(_expected_frame_count()):
        _read_exact_into(conditioned_frames, conditioned)
        _read_exact_into(conditioned_meta, conditioned_meta_frame)
        arrived_at = time.perf_counter()

        emitted_at = float(conditioned_meta_frame[1])
        frame_index = int(round(float(conditioned_meta_frame[0]))) + 1
        spectrum = _vibration_power_spectrum(conditioned)

        last_tank_lox_mean = float(np.mean(conditioned[:, 0]))
        last_tank_lng_mean = float(np.mean(conditioned[:, 1]))
        last_injector_spread = float(np.std(conditioned[:, 3] - conditioned[:, 2]))
        peak_bin = int(np.argmax(spectrum[1:])) + 1
        last_peak_hz = float(_FREQ_AXIS_HZ[peak_bin])
        latency_ms = (arrived_at - emitted_at) * 1_000.0

        arrival_window.append(arrived_at)
        if len(arrival_window) > 1:
            window_elapsed = arrival_window[-1] - arrival_window[0]
            throughput_mb_s = (
                (frame_nbytes * len(arrival_window)) / window_elapsed / 1_000_000.0
                if window_elapsed > 0.0
                else 0.0
            )
        else:
            throughput_mb_s = 0.0

        metrics_queue.put(
            {
                "frame_index": frame_index - 1,
                "segment_time_s": (frame_index - 1) * FRAME_LENGTH / TARGET_SAMPLE_RATE_HZ,
                "latency_ms": latency_ms,
                "throughput_mb_s": throughput_mb_s,
                "dominant_hz": last_peak_hz,
            }
        )

        if frame_index % REPORT_EVERY_FRAMES == 0:
            print(
                f"frame={frame_index:03d} "
                f"tank_lox_mean={last_tank_lox_mean:7.2f} psi "
                f"tank_lng_mean={last_tank_lng_mean:7.2f} psi "
                f"injector_spread={last_injector_spread:6.3f} psi "
                f"dominant_injector_oscillation={last_peak_hz:7.1f} Hz "
                f"latency={latency_ms:6.2f} ms "
                f"throughput={throughput_mb_s:6.2f} MB/s"
            )

    print(
        "final_summary "
        f"frames={frame_index} "
        f"tank_lox_mean={last_tank_lox_mean:.2f} psi "
        f"tank_lng_mean={last_tank_lng_mean:.2f} psi "
        f"injector_spread={last_injector_spread:.3f} psi "
        f"dominant_injector_oscillation={last_peak_hz:.1f} Hz"
    )


def _print_metrics(pipe: pythusa.Pipeline) -> None:
    for task_name, metrics in pipe.metrics().items():
        if metrics is None:
            continue
        print(
            f"metrics task={task_name} pid={metrics.pid} "
            f"cpu={metrics.cpu_percent:.1f}% rss={metrics.memory_rss_mb:.1f}MB "
            f"pressure={metrics.ring_pressure}"
        )


def _collect_pipeline_records(metrics_queue: mp.queues.Queue, frame_count: int) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for _ in range(frame_count):
        try:
            record = metrics_queue.get(timeout=10.0)
        except Empty as exc:
            raise RuntimeError(
                f"Expected {frame_count} pipeline metric records, got {len(records)}."
            ) from exc
        records.append(record)
    records.sort(key=lambda item: int(item["frame_index"]))
    return records


def _plot_engine_results(records: list[dict[str, float]]) -> None:
    bundle = _telemetry_bundle()
    active_time = float(bundle["active_time"])
    segment_start, segment_end = _segment_bounds(
        float(bundle["overlap_start"]),
        float(bundle["overlap_end"]),
        active_time,
    )

    target_times, telemetry = _resampled_engine_inputs()

    frames = _engine_frames()
    samples = frames.reshape(-1, CHANNEL_COUNT)
    conditioned_frames = np.stack([_condition_frame(frame) for frame in frames], axis=0)
    conditioned_samples = conditioned_frames.reshape(-1, CHANNEL_COUNT)

    telemetry_time_s = target_times - target_times[0]
    pipeline_time_s = telemetry_time_s[:samples.shape[0]]
    decimation = max(1, samples.shape[0] // 6_000)
    decimated_time_s = pipeline_time_s[::decimation]

    vibration_channel = conditioned_samples[:, 2]
    peak_sample = int(np.argmax(np.abs(vibration_channel)))
    window_radius = int(0.012 * TARGET_SAMPLE_RATE_HZ)
    window_start = max(0, peak_sample - window_radius)
    window_stop = min(samples.shape[0], peak_sample + window_radius)
    vibration_time_ms = pipeline_time_s[window_start:window_stop] * 1_000.0

    frame_energy = np.mean(np.square(conditioned_frames[:, :, 3] - conditioned_frames[:, :, 2]), axis=1)
    spectrum_frame_index = int(np.argmax(frame_energy))
    spectrum = _vibration_power_spectrum(conditioned_frames[spectrum_frame_index])
    dominant_bin = int(np.argmax(spectrum[1:])) + 1
    dominant_hz = float(_FREQ_AXIS_HZ[dominant_bin])
    segment_label = (
        f"{segment_start:.3f} to {segment_end:.3f} unix time "
        f"({segment_end - segment_start:.2f} s)"
    )

    throughput_time_s = np.asarray([record["segment_time_s"] for record in records], dtype=np.float64)
    throughput_mb_s = np.asarray([record["throughput_mb_s"] for record in records], dtype=np.float64)
    latency_ms = np.asarray([record["latency_ms"] for record in records], dtype=np.float64)

    figure, axes = plt.subplots(6, 1, figsize=(13, 18), constrained_layout=True)
    figure.suptitle("PYTHUSA Tanks and Injectors DSP Example", fontsize=16)

    axes[0].plot(
        telemetry_time_s,
        telemetry["pressurelox"],
        label="Tank LOX",
        linewidth=1.0,
    )
    axes[0].plot(
        telemetry_time_s,
        telemetry["pressurelng"],
        label="Tank LNG",
        linewidth=1.0,
    )
    axes[0].plot(
        telemetry_time_s,
        telemetry["pressureinjectorlox"],
        label="ECU Injector LOX",
        linewidth=1.0,
    )
    axes[0].plot(
        telemetry_time_s,
        telemetry["pressureinjectorlng"],
        label="ECU Injector LNG",
        linewidth=1.0,
    )
    axes[0].set_title(f"Tanks and Injectors ({segment_label})")
    axes[0].set_ylabel("Pressure (psi)")
    axes[0].set_xlabel("Time Within High-Interest Section (s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        telemetry_time_s,
        telemetry["pressuregn2"],
        label="GSE GN2 Pressure",
        linewidth=1.0,
    )
    axes[1].plot(
        telemetry_time_s,
        telemetry["pressurevent"],
        label="GSE Vent Pressure",
        linewidth=1.0,
    )
    axes[1].plot(
        telemetry_time_s,
        telemetry["pressureloxinjtee"],
        label="GSE LOX Injector Tee Pressure",
        linewidth=1.0,
    )
    axes[1].plot(
        telemetry_time_s,
        telemetry["pressureloxmvas"],
        label="GSE LOX MVAS Pressure",
        linewidth=1.0,
    )
    axes[1].set_title("Tank Pressures")
    axes[1].set_ylabel("Pressure (psi)")
    axes[1].set_xlabel("Time Within High-Interest Section (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(
        decimated_time_s,
        samples[::decimation, 0],
        label="Tank LOX Pressure",
        linewidth=1.0,
    )
    axes[2].plot(
        decimated_time_s,
        samples[::decimation, 1],
        label="Tank LNG Pressure",
        linewidth=1.0,
    )
    axes[2].plot(
        decimated_time_s,
        samples[::decimation, 2],
        label="Injector LOX Pressure",
        linewidth=1.0,
    )
    axes[2].plot(
        decimated_time_s,
        samples[::decimation, 3],
        label="Injector LNG Pressure",
        linewidth=1.0,
    )
    axes[2].set_title("Tanks and Injectors High-Rate Segment")
    axes[2].set_xlabel("Segment Time (s)")
    axes[2].set_ylabel("Pressure (psi)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    injector_delta = samples[:, 3] - samples[:, 2]
    conditioned_delta = conditioned_samples[:, 3] - conditioned_samples[:, 2]
    axes[3].plot(
        vibration_time_ms,
        injector_delta[window_start:window_stop],
        label="Raw Injector Differential",
        linewidth=1.0,
        alpha=0.75,
    )
    axes[3].plot(
        vibration_time_ms,
        conditioned_delta[window_start:window_stop],
        label="Conditioned Injector Differential",
        linewidth=1.2,
    )
    axes[3].set_title("Injector Differential")
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_ylabel("Pressure Delta (psi)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    axes[4].plot(_FREQ_AXIS_HZ / 1_000.0, spectrum, linewidth=1.2)
    axes[4].axvline(dominant_hz / 1_000.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[4].set_title(
        f"Injector Differential Spectrum (dominant tone {dominant_hz:,.0f} Hz)"
    )
    axes[4].set_xlabel("Frequency (kHz)")
    axes[4].set_ylabel("Power")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(
        throughput_time_s,
        throughput_mb_s,
        color="tab:orange",
        linewidth=1.2,
        alpha=0.85,
        label="Pipeline Throughput",
    )
    metrics_ax = axes[5].twinx()
    metrics_ax.plot(
        throughput_time_s,
        latency_ms,
        color="tab:green",
        linewidth=1.2,
        alpha=0.85,
        label="Pipeline Latency",
    )
    axes[5].set_title("Pipeline Throughput and Latency")
    axes[5].set_xlabel("Segment Time (s)")
    axes[5].set_ylabel("Throughput (MB/s)")
    axes[5].grid(True, alpha=0.3)
    metrics_ax.set_ylabel("Latency (ms)")

    handles_left, labels_left = axes[5].get_legend_handles_labels()
    handles_right, labels_right = metrics_ax.get_legend_handles_labels()
    axes[5].legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")

    plt.show()


def main() -> None:
    frame_count = _expected_frame_count()
    metrics_queue = mp.get_context("spawn").Queue()
    report_task = partial(report_engine_state, metrics_queue=metrics_queue)

    with pythusa.Pipeline("engine-dsp") as pipe:
        pipe.add_stream(
            "engine_frames",
            shape=(FRAME_LENGTH, CHANNEL_COUNT),
            dtype=np.float32,
            cache_align=False,
        )
        pipe.add_stream(
            "frame_meta",
            shape=(META_FIELDS,),
            dtype=np.float64,
            cache_align=False,
        )
        pipe.add_stream(
            "conditioned_frames",
            shape=(FRAME_LENGTH, CHANNEL_COUNT),
            dtype=np.float32,
            cache_align=False,
        )
        pipe.add_stream(
            "conditioned_meta",
            shape=(META_FIELDS,),
            dtype=np.float64,
            cache_align=False,
        )
        pipe.add_task(
            "source",
            fn=source_engine_frames,
            writes={
                "engine_frames": "engine_frames",
                "frame_meta": "frame_meta",
            },
        )
        pipe.add_task(
            "condition",
            fn=condition_engine_frames,
            reads={
                "engine_frames": "engine_frames",
                "frame_meta": "frame_meta",
            },
            writes={
                "conditioned_frames": "conditioned_frames",
                "conditioned_meta": "conditioned_meta",
            },
        )
        pipe.add_task(
            "report",
            fn=report_task,
            reads={
                "conditioned_frames": "conditioned_frames",
                "conditioned_meta": "conditioned_meta",
            },
        )

        print(
            "starting engine dsp pipeline "
            f"duration={SEGMENT_DURATION_S:.1f}s "
            f"sample_rate={TARGET_SAMPLE_RATE_HZ:,.0f} Hz "
            f"frames={frame_count}"
        )

        pipe.start_monitor(interval_s=0.05)
        started_at = time.perf_counter()
        pipe.start()
        try:
            records = _collect_pipeline_records(metrics_queue, frame_count)
            pipe.join(timeout=10.0)
        finally:
            metrics_queue.close()
            metrics_queue.join_thread()

        elapsed = time.perf_counter() - started_at
        print(f"pipeline completed in {elapsed:.2f}s")
        _print_metrics(pipe)
        _plot_engine_results(records)


if __name__ == "__main__":
    main()
