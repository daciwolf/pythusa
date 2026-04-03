from __future__ import annotations

"""
FFT pipeline demo built on Pythusa.

Streams synthetic multi-channel sensor data through shared-memory ring buffers
and runs per-channel FFT analysis in parallel worker processes. The GUI path
provides an operator desk with live signal monitoring and on-demand FFT
extraction. The headless path strips the display stack and benchmarks sustained
FFT throughput.
"""

import argparse
from collections import deque
from dataclasses import dataclass
from functools import partial
import time

import numpy as np
import pythusa

FRAME_ROWS = 1024 * 8
SIGNALS_PER_GENERATOR = 7
FRAME_COLS = SIGNALS_PER_GENERATOR + 1
GENERATOR_IDS = (1, 2)
SIGNAL_SPECS = tuple(
    (generator, channel)
    for generator in GENERATOR_IDS
    for channel in range(1, SIGNALS_PER_GENERATOR + 1)
)
CHANNEL_IDS = tuple(range(1, len(SIGNAL_SPECS) + 1))
TOTAL_SIGNALS = len(SIGNAL_SPECS)
SAMPLE_RATE_HZ = 61440.0
DT = 1.0 / SAMPLE_RATE_HZ
PREGENERATED_FRAMES = 64
DISPLAY_SAMPLE_STRIDE = 1000
DISPLAY_ROWS = 32
DISPLAY_HISTORY_ROWS = 32
DISPLAY_DTYPE = np.float32
SLEEP = 1e-4
FLOAT64_BITS = np.dtype(np.float64).itemsize * 8
MAX_DISPLAY_SAMPLES_PER_FRAME = (FRAME_ROWS - 1) // DISPLAY_SAMPLE_STRIDE + 1
DEFAULT_REPORT_INTERVAL_S = 1.0
BENCHMARK_STATS_SHAPE = (2,)
BITS_PER_GIGABIT = 1_000_000_000.0
HOTPATH_EVENT_POLL_INTERVAL = 1024 * 16
HOTPATH_EVENT_POLL_MASK = HOTPATH_EVENT_POLL_INTERVAL - 1


@dataclass(frozen=True)
class HeadlessMode:
    name: str
    frame_rows: int
    description: str


HEADLESS_MODES = {
    "throughput": HeadlessMode(
        name="throughput",
        frame_rows=FRAME_ROWS,
        description="Large FFT frames maximize aggregate signal payload per second.",
    ),
    "latency": HeadlessMode(
        name="latency",
        frame_rows=1024,
        description="Smaller FFT frames reduce acquisition window at the cost of peak throughput.",
    ),
}


def _signal_bits_per_fft(frame_rows: int) -> int:
    return frame_rows * FLOAT64_BITS


def _benchmark_signal_bits_per_batch(frame_rows: int) -> int:
    return _signal_bits_per_fft(frame_rows) * SIGNALS_PER_GENERATOR


def _frame_window_ms(frame_rows: int) -> float:
    return frame_rows * DT * 1_000.0


def _headless_console_shutdown_event_name() -> str:
    return "headless_shutdown_console"


def _headless_generator_shutdown_event_name(generator_id: int) -> str:
    return f"headless_shutdown_generator{generator_id}"


def _headless_fft_shutdown_event_name(generator_id: int) -> str:
    return f"headless_shutdown_fft{generator_id}"


def _pregenerated_signal_bank(
    seed: int,
    *,
    frame_rows: int = FRAME_ROWS,
    channels_last: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bank = np.empty((PREGENERATED_FRAMES, frame_rows, SIGNALS_PER_GENERATOR), dtype=np.float64)
    base_t = np.arange(frame_rows, dtype=np.float64) * DT
    noise = np.empty(frame_rows, dtype=np.float64)
    twopi = 2.0 * np.pi

    for frame in bank:
        amps = rng.uniform(0.05, 1.0, size=(SIGNALS_PER_GENERATOR, 16))
        freqs = rng.uniform(60.0, 1000.0, size=(SIGNALS_PER_GENERATOR, 16))
        phases = rng.uniform(0.0, twopi, size=(SIGNALS_PER_GENERATOR, 16))
        for channel, signal in enumerate(frame.T):
            signal.fill(0.0)
            for amp, freq, phase in zip(amps[channel], freqs[channel], phases[channel]):
                signal += amp * np.sin((twopi * freq * base_t) + phase)
            rng.standard_normal(out=noise)
            noise *= 0.05
            signal += noise
    if channels_last:
        return bank
    return np.ascontiguousarray(bank.transpose(0, 2, 1))


def _generator(output, *, seed: int) -> None:
    bank = _pregenerated_signal_bank(seed, frame_rows=FRAME_ROWS)
    row_offsets = np.arange(FRAME_ROWS, dtype=np.float64) * DT
    frame_shape = (FRAME_ROWS, FRAME_COLS)
    frame_index = 0
    timestamp0 = 0.0
    sleep = time.sleep

    while True:
        frame_view = output.look()
        if frame_view is None:
            sleep(SLEEP)
            continue
        frame = np.frombuffer(frame_view, dtype=np.float64).reshape(frame_shape)
        frame[:, 0] = timestamp0 + row_offsets
        frame[:, 1:] = bank[frame_index]
        timestamp0 += FRAME_ROWS * DT
        frame_index = (frame_index + 1) % PREGENERATED_FRAMES
        del frame, frame_view
        output.increment()


def _benchmark_generator(output, shutdown, *, seed: int, frame_rows: int) -> None:
    bank = _pregenerated_signal_bank(seed, frame_rows=frame_rows, channels_last=False)
    frame_shape = (SIGNALS_PER_GENERATOR, frame_rows)
    frame_index = 0
    poll_counter = 0
    poll_mask = HOTPATH_EVENT_POLL_MASK
    sleep = time.sleep
    shutdown_is_open = shutdown.is_open

    while True:
        if (poll_counter & poll_mask) == 0 and shutdown_is_open():
            break
        poll_counter += 1
        frame_view = output.look()
        if frame_view is None:
            sleep(SLEEP)
            continue
        frame = np.frombuffer(frame_view, dtype=np.float64).reshape(frame_shape)
        frame[:] = bank[frame_index]
        frame_index = (frame_index + 1) % PREGENERATED_FRAMES
        del frame, frame_view
        output.increment()


def _sample_task(samples, output, cols: int) -> None:
    samples.set_blocking(False)
    frame_shape = (FRAME_ROWS, cols)
    out_shape = (DISPLAY_ROWS, cols)
    pending = np.empty((DISPLAY_ROWS + MAX_DISPLAY_SAMPLES_PER_FRAME, cols), dtype=DISPLAY_DTYPE)
    count = 0
    cursor = 0
    sleep = time.sleep

    while True:
        if count >= DISPLAY_ROWS:
            output_view = output.look()
            if output_view is None:
                sleep(SLEEP)
                continue
            out = np.frombuffer(output_view, dtype=DISPLAY_DTYPE).reshape(out_shape)
            out[:] = pending[:DISPLAY_ROWS]
            del out, output_view
            output.increment()
            count -= DISPLAY_ROWS
            if count:
                pending[:count] = pending[DISPLAY_ROWS : DISPLAY_ROWS + count]
            continue

        frame_view = samples.look()
        if frame_view is None:
            sleep(SLEEP)
            continue
        frame = np.frombuffer(frame_view, dtype=np.float64).reshape(frame_shape)
        start = (-cursor) % DISPLAY_SAMPLE_STRIDE
        if start < FRAME_ROWS:
            rows = frame[start::DISPLAY_SAMPLE_STRIDE]
            take = rows.shape[0]
            pending[count : count + take] = rows
            count += take
        cursor = (cursor + FRAME_ROWS) % DISPLAY_SAMPLE_STRIDE
        del frame, frame_view
        samples.increment()


def display_raw_task(samples, output) -> None:
    _sample_task(samples, output, FRAME_COLS)


def _display_filtered_task(samples, output) -> None:
    _sample_task(samples, output, 2)


def _removed_component_chunk(data: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(data[:, 1])
    removed = np.zeros_like(spectrum)
    if spectrum.size > 1:
        peak = 1 + int(np.abs(spectrum[1:]).argmax())
        removed[peak] = spectrum[peak]
    return np.column_stack((data[:, 0], np.fft.irfft(removed, n=data.shape[0])))


def _overlay_arrays(
    chunks: deque[np.ndarray],
    filtered: dict[float, np.ndarray],
) -> list[np.ndarray]:
    overlays: list[np.ndarray] = []
    for chunk in chunks:
        key = float(chunk[0, 0])
        overlay = filtered.get(key)
        overlays.append(overlay if overlay is not None else _removed_component_chunk(chunk))
    return overlays


def _signal_title(signal_index: int, rate: float) -> str:
    generator, channel = SIGNAL_SPECS[signal_index]
    return f"G{generator} S{channel} | FFT {rate:.3f} Gbit/s"


def _channel_label(signal_index: int) -> str:
    generator, channel = SIGNAL_SPECS[signal_index]
    return f"G{generator} S{channel}"


def _apply_imgui_theme(imgui) -> None:
    style = imgui.get_style()
    style.window_rounding = 10.0
    style.child_rounding = 8.0
    style.frame_rounding = 6.0
    style.grab_rounding = 6.0
    style.scrollbar_rounding = 8.0
    style.popup_rounding = 8.0
    style.tab_rounding = 6.0
    style.window_border_size = 1.0
    style.child_border_size = 1.0
    style.frame_border_size = 1.0
    style.window_padding = (16.0, 14.0)
    style.frame_padding = (10.0, 6.0)
    style.item_spacing = (10.0, 8.0)
    style.item_inner_spacing = (8.0, 6.0)
    style.cell_padding = (10.0, 6.0)
    style.window_title_align = (0.02, 0.5)
    style.button_text_align = (0.5, 0.5)
    style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.035, 0.055, 0.090, 1.0)
    style.colors[imgui.COLOR_CHILD_BACKGROUND] = (0.055, 0.075, 0.115, 0.98)
    style.colors[imgui.COLOR_POPUP_BACKGROUND] = (0.055, 0.075, 0.115, 0.98)
    style.colors[imgui.COLOR_BORDER] = (0.180, 0.255, 0.345, 0.92)
    style.colors[imgui.COLOR_BORDER_SHADOW] = (0.0, 0.0, 0.0, 0.0)
    style.colors[imgui.COLOR_FRAME_BACKGROUND] = (0.075, 0.105, 0.155, 1.0)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.095, 0.135, 0.205, 1.0)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.115, 0.165, 0.235, 1.0)
    style.colors[imgui.COLOR_TITLE_BACKGROUND] = (0.025, 0.040, 0.065, 1.0)
    style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.035, 0.055, 0.090, 1.0)
    style.colors[imgui.COLOR_BUTTON] = (0.085, 0.175, 0.255, 1.0)
    style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.125, 0.235, 0.340, 1.0)
    style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.165, 0.300, 0.415, 1.0)
    style.colors[imgui.COLOR_HEADER] = (0.095, 0.145, 0.215, 1.0)
    style.colors[imgui.COLOR_HEADER_HOVERED] = (0.125, 0.190, 0.280, 1.0)
    style.colors[imgui.COLOR_HEADER_ACTIVE] = (0.155, 0.225, 0.330, 1.0)
    style.colors[imgui.COLOR_SCROLLBAR_BACKGROUND] = (0.040, 0.055, 0.085, 1.0)
    style.colors[imgui.COLOR_SCROLLBAR_GRAB] = (0.135, 0.205, 0.290, 1.0)
    style.colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = (0.165, 0.250, 0.355, 1.0)
    style.colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = (0.195, 0.295, 0.420, 1.0)
    style.colors[imgui.COLOR_TEXT] = (0.910, 0.940, 0.975, 1.0)
    style.colors[imgui.COLOR_TEXT_DISABLED] = (0.540, 0.620, 0.710, 1.0)
    style.colors[imgui.COLOR_CHECK_MARK] = (0.300, 0.780, 0.960, 1.0)
    style.colors[imgui.COLOR_SEPARATOR] = (0.170, 0.245, 0.340, 1.0)
    style.colors[imgui.COLOR_SEPARATOR_HOVERED] = (0.220, 0.320, 0.445, 1.0)
    style.colors[imgui.COLOR_SEPARATOR_ACTIVE] = (0.260, 0.385, 0.530, 1.0)
    style.colors[imgui.COLOR_RESIZE_GRIP] = (0.095, 0.145, 0.215, 1.0)
    style.colors[imgui.COLOR_RESIZE_GRIP_HOVERED] = (0.135, 0.205, 0.290, 1.0)
    style.colors[imgui.COLOR_RESIZE_GRIP_ACTIVE] = (0.165, 0.250, 0.355, 1.0)


def _render_metric_card(
    imgui,
    *,
    title: str,
    value: str,
    accent: tuple[float, float, float, float],
    width: float,
) -> None:
    imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.050, 0.075, 0.115, 1.0)
    imgui.push_style_color(imgui.COLOR_BORDER, accent[0], accent[1], accent[2], 0.65)
    imgui.begin_child(f"metric_{title}", width, 62.0, border=True)
    imgui.text_colored(title, *accent)
    imgui.spacing()
    imgui.text_unformatted(value)
    imgui.end_child()
    imgui.pop_style_color(2)


def _render_fft_button(
    imgui,
    *,
    signal_index: int,
    event,
    enabled: bool,
    rate: float,
    width: float,
) -> bool:
    accent = (0.220, 0.770, 0.945, 1.0) if enabled else (0.345, 0.440, 0.555, 1.0)
    button = (0.090, 0.225, 0.320, 1.0) if enabled else (0.080, 0.115, 0.165, 1.0)
    hover = (0.120, 0.275, 0.390, 1.0) if enabled else (0.100, 0.150, 0.215, 1.0)
    active = (0.150, 0.330, 0.465, 1.0) if enabled else (0.120, 0.180, 0.255, 1.0)
    imgui.push_style_color(imgui.COLOR_BUTTON, *button)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *hover)
    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *active)
    pressed = imgui.button(_channel_label(signal_index), width=width, height=32.0)
    imgui.pop_style_color(3)
    if pressed:
        event.signal()
        enabled = True
    imgui.text_colored("ACTIVE" if enabled else "STANDBY", *accent)
    imgui.same_line()
    imgui.text_disabled(f"{rate:.3f} Gbit/s")
    return enabled


def _render_signal_card(
    imgui,
    *,
    card_id: str,
    signal_index: int,
    history: deque[np.ndarray],
    filtered: dict[float, np.ndarray],
    fft_enabled: bool,
    rate: float,
) -> None:
    from graphing import graph_data_arrays

    imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.048, 0.070, 0.108, 1.0)
    imgui.push_style_color(imgui.COLOR_BORDER, 0.170, 0.255, 0.360, 0.9)
    imgui.begin_child(card_id, 0, 154.0, border=True)
    imgui.text_unformatted(_channel_label(signal_index))
    imgui.same_line()
    imgui.text_disabled("FFT enabled" if fft_enabled else "FFT idle")
    imgui.same_line()
    status_color = (0.220, 0.770, 0.945, 1.0) if fft_enabled else (0.530, 0.610, 0.710, 1.0)
    imgui.text_colored(f"{rate:.3f} Gbit/s", *status_color)
    imgui.separator()
    avail = imgui.get_content_region_available()
    plot_width = float(avail[0] if isinstance(avail, tuple) else avail.x)
    graph_data_arrays(
        history,
        overlay_arrays=_overlay_arrays(history, filtered) if fft_enabled else None,
        size=(max(32.0, plot_width), 102.0),
        color=(0.260, 0.790, 0.980, 1.0),
        overlay_color=(1.000, 0.470, 0.260, 1.0),
        background_color=(0.028, 0.040, 0.065, 1.0),
        border_color=(0.150, 0.235, 0.330, 1.0),
        thickness=1.7,
    )
    imgui.end_child()
    imgui.pop_style_color(2)


def _render_dashboard(
    imgui,
    *,
    fft_events,
    fft_enabled: list[bool],
    fft_gbit_s: list[float],
    chunks: list[deque[np.ndarray]],
    filtered_chunks: list[dict[float, np.ndarray]],
) -> None:
    total_gbit_s = sum(fft_gbit_s)
    active_count = sum(fft_enabled)
    active_rates = [rate for enabled, rate in zip(fft_enabled, fft_gbit_s) if enabled]
    avg_active_gbit_s = (sum(active_rates) / len(active_rates)) if active_rates else 0.0
    peak_signal = max(range(TOTAL_SIGNALS), key=fft_gbit_s.__getitem__, default=0)

    expanded, _ = imgui.begin("Pythusa Signal Lab")
    if not expanded:
        imgui.end()
        return

    imgui.text_colored("Realtime FFT Operations Dashboard", 0.300, 0.780, 0.960, 1.0)
    imgui.text_disabled(
        "Desk view: raw signal monitoring, on-demand FFT extraction, and live throughput telemetry."
    )
    imgui.spacing()

    avail = imgui.get_content_region_available()
    avail_width = float(avail[0] if isinstance(avail, tuple) else avail.x)
    sidebar_width = min(340.0, max(280.0, avail_width * 0.24))
    content_width = max(320.0, avail_width - sidebar_width - 12.0)
    metric_width = max(120.0, (sidebar_width - 10.0) / 2.0)

    imgui.begin_child("control_rail", sidebar_width, 0, border=True)
    _render_metric_card(
        imgui,
        title="Desk Throughput",
        value=f"{total_gbit_s:.3f} Gbit/s",
        accent=(0.220, 0.770, 0.945, 1.0),
        width=metric_width,
    )
    imgui.same_line()
    _render_metric_card(
        imgui,
        title="FFT Coverage",
        value=f"{active_count}/{TOTAL_SIGNALS}",
        accent=(0.985, 0.640, 0.250, 1.0),
        width=metric_width,
    )
    _render_metric_card(
        imgui,
        title="Avg Active",
        value=f"{avg_active_gbit_s:.3f} Gbit/s",
        accent=(0.420, 0.860, 0.620, 1.0),
        width=metric_width,
    )
    imgui.same_line()
    _render_metric_card(
        imgui,
        title="Peak Line",
        value=_channel_label(peak_signal),
        accent=(0.930, 0.420, 0.390, 1.0),
        width=metric_width,
    )
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    button_width = max(96.0, sidebar_width - 32.0)
    for generator in GENERATOR_IDS:
        imgui.text_colored(f"Generator {generator}", 0.640, 0.790, 0.920, 1.0)
        imgui.text_disabled("Arm FFT lanes individually. Plots update with extracted dominant frequency.")
        imgui.spacing()
        base = (generator - 1) * SIGNALS_PER_GENERATOR
        for local_index in range(SIGNALS_PER_GENERATOR):
            signal_index = base + local_index
            fft_enabled[signal_index] = _render_fft_button(
                imgui,
                signal_index=signal_index,
                event=fft_events[signal_index],
                enabled=fft_enabled[signal_index],
                rate=fft_gbit_s[signal_index],
                width=button_width,
            )
        if generator != GENERATOR_IDS[-1]:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
    imgui.end_child()

    imgui.same_line()

    imgui.begin_child("plot_deck", content_width, 0, border=False)
    generator_panel_width = max(220.0, (content_width - 10.0) / 2.0)
    for generator_index, generator in enumerate(GENERATOR_IDS):
        imgui.begin_child(f"generator_panel_{generator}", generator_panel_width, 0, border=True)
        imgui.text_unformatted(f"Generator {generator}")
        imgui.same_line()
        panel_rates = fft_gbit_s[
            generator_index * SIGNALS_PER_GENERATOR : (generator_index + 1) * SIGNALS_PER_GENERATOR
        ]
        imgui.text_disabled(f"{sum(panel_rates):.3f} Gbit/s aggregated")
        imgui.separator()
        base = generator_index * SIGNALS_PER_GENERATOR
        for local_index in range(SIGNALS_PER_GENERATOR):
            signal_index = base + local_index
            history = chunks[signal_index]
            if not history:
                continue
            valid_keys = {float(chunk[0, 0]) for chunk in history}
            filtered = filtered_chunks[signal_index]
            if len(filtered) > len(valid_keys):
                filtered_chunks[signal_index] = filtered = {
                    key: value for key, value in filtered.items() if key in valid_keys
                }
            _render_signal_card(
                imgui,
                card_id=f"signal_card_{signal_index}",
                signal_index=signal_index,
                history=history,
                filtered=filtered,
                fft_enabled=fft_enabled[signal_index],
                rate=fft_gbit_s[signal_index],
            )
        imgui.end_child()
        if generator_index < len(GENERATOR_IDS) - 1:
            imgui.same_line()
    imgui.end_child()
    imgui.end()


def imgui_task(**bindings) -> None:
    import glfw
    import imgui
    from OpenGL import GL as gl
    from imgui.integrations.glfw import GlfwRenderer
    from graphing import graph_data_arrays

    if not glfw.init():
        raise SystemExit("failed to initialize glfw")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(1600, 1000, "pythusa", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("failed to create window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    imgui.create_context()
    _apply_imgui_theme(imgui)
    impl = GlfwRenderer(window)

    sample_readers = [bindings[f"samples{generator}"] for generator in GENERATOR_IDS]
    filtered_readers = [bindings[f"filtered{channel}"] for channel in CHANNEL_IDS]
    stats_readers = [bindings[f"stats{channel}"] for channel in CHANNEL_IDS]
    fft_events = [bindings[f"enable_fft{channel}"] for channel in CHANNEL_IDS]
    chunks = [deque(maxlen=DISPLAY_HISTORY_ROWS) for _ in CHANNEL_IDS]
    filtered_chunks = [{} for _ in CHANNEL_IDS]
    fft_enabled = [False] * TOTAL_SIGNALS
    fft_gbit_s = [0.0] * TOTAL_SIGNALS

    for reader in (*sample_readers, *filtered_readers, *stats_readers):
        reader.set_blocking(False)

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()
            imgui.new_frame()

            for generator_index, reader in enumerate(sample_readers):
                frame_view = reader.look()
                if frame_view is None:
                    continue
                frame = np.frombuffer(frame_view, dtype=DISPLAY_DTYPE).reshape((DISPLAY_ROWS, FRAME_COLS))
                base = generator_index * SIGNALS_PER_GENERATOR
                for local_channel in range(1, SIGNALS_PER_GENERATOR + 1):
                    segment = np.empty((DISPLAY_ROWS, 2), dtype=DISPLAY_DTYPE)
                    segment[:, 0] = frame[:, 0]
                    segment[:, 1] = frame[:, local_channel]
                    chunks[base + local_channel - 1].append(segment)
                del frame, frame_view
                reader.increment()

            for channel_index, reader in enumerate(filtered_readers):
                filtered_view = reader.look()
                if filtered_view is None:
                    continue
                filtered_frame = np.frombuffer(filtered_view, dtype=DISPLAY_DTYPE).reshape((DISPLAY_ROWS, 2))
                candidate = filtered_frame.copy()
                filtered_chunks[channel_index][float(candidate[0, 0])] = candidate
                del filtered_frame, filtered_view
                reader.increment()

            for channel_index, reader in enumerate(stats_readers):
                stats_view = reader.look()
                if stats_view is None:
                    continue
                fft_gbit_s[channel_index] = float(np.frombuffer(stats_view, dtype=np.float64, count=1)[0])
                del stats_view
                reader.increment()

            _render_dashboard(
                imgui,
                fft_events=fft_events,
                fft_enabled=fft_enabled,
                fft_gbit_s=fft_gbit_s,
                chunks=chunks,
                filtered_chunks=filtered_chunks,
            )

            imgui.render()
            gl.glClearColor(0.020, 0.030, 0.050, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
    finally:
        impl.shutdown()
        glfw.terminate()


def _fft_task(samples, output, stats, enable_fft, *, channel: int) -> None:
    samples.set_blocking(False)
    frame_shape = (FRAME_ROWS, FRAME_COLS)
    out_shape = (FRAME_ROWS, 2)
    removed = np.zeros((FRAME_ROWS // 2) + 1, dtype=np.complex128)
    processed_frames = 0
    last_t = time.perf_counter()
    sleep = time.sleep
    enabled = False

    while True:
        if not enabled:
            enable_fft.wait()
            processed_frames = 0
            last_t = time.perf_counter()
            enabled = True

        frame_view = samples.look()
        if frame_view is None:
            sleep(SLEEP)
            continue

        frame = np.frombuffer(frame_view, dtype=np.float64).reshape(frame_shape)
        spectrum = np.fft.rfft(frame[:, channel])
        removed.fill(0)
        if spectrum.size > 1:
            peak = 1 + int(np.abs(spectrum[1:]).argmax())
            removed[peak] = spectrum[peak]

        output_view = output.look()
        if output_view is not None:
            out = np.frombuffer(output_view, dtype=np.float64).reshape(out_shape)
            out[:, 0] = frame[:, 0]
            out[:, 1] = np.fft.irfft(removed, n=FRAME_ROWS)
            output.increment()

        samples.increment()
        processed_frames += 1

        now = time.perf_counter()
        elapsed = now - last_t
        if elapsed < 0.25:
            continue

        stats_view = stats.look()
        if stats_view is not None:
            np.frombuffer(stats_view, dtype=np.float64, count=1)[0] = (
                processed_frames * _signal_bits_per_fft(FRAME_ROWS)
            ) / elapsed / BITS_PER_GIGABIT
            stats.increment()
        processed_frames = 0
        last_t = now


def _fft_benchmark_task(samples, stats, shutdown, *, frame_rows: int) -> None:
    samples.set_blocking(False)
    frame_shape = (SIGNALS_PER_GENERATOR, frame_rows)
    processed_batches = 0
    last_t = time.perf_counter()
    poll_counter = 0
    poll_mask = HOTPATH_EVENT_POLL_MASK
    sleep = time.sleep
    shutdown_is_open = shutdown.is_open

    while True:
        if (poll_counter & poll_mask) == 0 and shutdown_is_open():
            break
        poll_counter += 1
        frame_view = samples.look()
        if frame_view is None:
            sleep(SLEEP)
            continue

        frame = np.frombuffer(frame_view, dtype=np.float64).reshape(frame_shape)
        np.fft.rfft(frame, axis=1)
        samples.increment()
        processed_batches += 1

        now = time.perf_counter()
        elapsed = now - last_t
        if elapsed < 0.25:
            continue

        stats_view = stats.look()
        if stats_view is not None:
            stats_array = np.frombuffer(stats_view, dtype=np.float64, count=2)
            stats_array[0] = (
                processed_batches * _benchmark_signal_bits_per_batch(frame_rows)
            ) / elapsed / BITS_PER_GIGABIT
            stats_array[1] = (processed_batches * SIGNALS_PER_GENERATOR) / elapsed
            stats.increment()
        processed_batches = 0
        last_t = now


def _headless_console_stats_task(
    shutdown,
    *,
    report_interval_s: float = DEFAULT_REPORT_INTERVAL_S,
    mode_name: str,
    frame_rows: int,
    generator_ids: tuple[int, ...],
    total_signals: int,
    **bindings,
) -> None:
    stats_readers = [bindings[f"stats{gid}"] for gid in generator_ids]
    generator_stats = np.zeros((len(generator_ids), BENCHMARK_STATS_SHAPE[0]), dtype=np.float64)
    sleep = time.sleep
    last_report = time.perf_counter()
    frame_window_ms = _frame_window_ms(frame_rows)
    n_generators = len(generator_ids)

    for reader in stats_readers:
        reader.set_blocking(False)

    while not shutdown.is_open():
        had_update = False
        for generator_index, reader in enumerate(stats_readers):
            stats_view = reader.look()
            if stats_view is None:
                continue
            generator_stats[generator_index] = np.frombuffer(stats_view, dtype=np.float64, count=2)
            del stats_view
            reader.increment()
            had_update = True

        now = time.perf_counter()
        if now - last_report >= report_interval_s:
            total_gbit_s = float(generator_stats[:, 0].sum())
            total_ffts_s = float(generator_stats[:, 1].sum())
            print(
                (
                    f"[{mode_name} | {n_generators} generators | "
                    f"{frame_rows:,} rows | {frame_window_ms:.1f} ms window] "
                    "FFT throughput: "
                    f"{total_gbit_s:.3f} Gbit/s signal payload, "
                    f"{total_ffts_s:,.0f} FFT/s across {total_signals} signals"
                ),
                flush=True,
            )
            last_report = now
            continue

        if not had_update:
            sleep(SLEEP)


GENERATOR_TASKS = {
    1: partial(_generator, seed=0),
    2: partial(_generator, seed=1),
}

BENCHMARK_GENERATOR_SEEDS = {
    1: 0,
    2: 1,
}

FFT_TASKS = {
    index: partial(_fft_task, channel=channel)
    for index, (_, channel) in zip(CHANNEL_IDS, SIGNAL_SPECS)
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pythusa FFT demo.")
    parser.add_argument(
        "--no-gui",
        "--headless",
        action="store_true",
        help="Run a headless FFT throughput benchmark and print stats to stdout.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(HEADLESS_MODES),
        default="throughput",
        help="Headless benchmark mode. throughput maximizes payload; latency reduces frame window.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional number of seconds to run in headless mode before stopping.",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=DEFAULT_REPORT_INTERVAL_S,
        help="Seconds between throughput reports in headless mode.",
    )
    parser.add_argument(
        "--frame-rows",
        type=int,
        default=None,
        help="Optional headless-only FFT frame length override in samples.",
    )
    parser.add_argument(
        "--generators",
        type=int,
        default=len(GENERATOR_IDS),
        help=(
            "Number of generator/FFT-worker pairs in headless mode. "
            f"Default {len(GENERATOR_IDS)}. Each pair adds one data producer "
            "and one FFT consumer process."
        ),
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be greater than 0.")
    if args.report_interval <= 0.0:
        parser.error("--report-interval must be greater than 0.")
    if args.frame_rows is not None and args.frame_rows <= 0:
        parser.error("--frame-rows must be greater than 0.")
    if args.generators <= 0:
        parser.error("--generators must be greater than 0.")
    return args


def _resolve_headless_frame_rows(args: argparse.Namespace) -> int:
    if args.frame_rows is not None:
        return args.frame_rows
    return HEADLESS_MODES[args.mode].frame_rows


def _configure_gui_generators(pipe: pythusa.Pipeline) -> None:
    for generator in GENERATOR_IDS:
        pipe.add_stream(
            f"generator{generator}",
            shape=(FRAME_ROWS, FRAME_COLS),
            dtype=np.float64,
            cache_align=True,
        )
        pipe.add_task(
            f"generator{generator}task",
            fn=GENERATOR_TASKS[generator],
            writes={"output": f"generator{generator}"},
        )


def _configure_gui(pipe: pythusa.Pipeline) -> None:
    _configure_gui_generators(pipe)

    for channel in CHANNEL_IDS:
        pipe.add_stream(f"fft_stats{channel}", shape=(1,), dtype=np.float64, cache_align=True)

    for channel in CHANNEL_IDS:
        pipe.add_event(f"enable_fft{channel}", initial_state=False)
        pipe.add_stream(f"filtered{channel}", shape=(FRAME_ROWS, 2), dtype=np.float64, cache_align=True)
        pipe.add_stream(
            f"display_filtered{channel}",
            shape=(DISPLAY_ROWS, 2),
            dtype=DISPLAY_DTYPE,
            cache_align=True,
        )

    for generator in GENERATOR_IDS:
        pipe.add_stream(
            f"display_raw{generator}",
            shape=(DISPLAY_ROWS, FRAME_COLS),
            dtype=DISPLAY_DTYPE,
            cache_align=True,
        )
        pipe.add_task(
            f"display_raw_task{generator}",
            fn=display_raw_task,
            reads={"samples": f"generator{generator}"},
            writes={"output": f"display_raw{generator}"},
        )

    pipe.add_task(
        "imgui",
        fn=imgui_task,
        reads={
            **{f"samples{generator}": f"display_raw{generator}" for generator in GENERATOR_IDS},
            **{f"filtered{channel}": f"display_filtered{channel}" for channel in CHANNEL_IDS},
            **{f"stats{channel}": f"fft_stats{channel}" for channel in CHANNEL_IDS},
        },
        events={f"enable_fft{channel}": f"enable_fft{channel}" for channel in CHANNEL_IDS},
    )

    for channel_index, (generator, _) in zip(CHANNEL_IDS, SIGNAL_SPECS):
        pipe.add_task(
            f"fftsample{channel_index}",
            fn=FFT_TASKS[channel_index],
            reads={"samples": f"generator{generator}"},
            writes={"output": f"filtered{channel_index}", "stats": f"fft_stats{channel_index}"},
            events={"enable_fft": f"enable_fft{channel_index}"},
        )
        pipe.add_task(
            f"display_filtered_task{channel_index}",
            fn=_display_filtered_task,
            reads={"samples": f"filtered{channel_index}"},
            writes={"output": f"display_filtered{channel_index}"},
        )


def _configure_headless(
    pipe: pythusa.Pipeline,
    *,
    report_interval_s: float,
    mode_name: str,
    frame_rows: int,
    generator_ids: tuple[int, ...],
) -> None:
    total_signals = len(generator_ids) * SIGNALS_PER_GENERATOR
    pipe.add_event(_headless_console_shutdown_event_name(), initial_state=False)
    for gid in generator_ids:
        pipe.add_event(_headless_generator_shutdown_event_name(gid), initial_state=False)
        pipe.add_event(_headless_fft_shutdown_event_name(gid), initial_state=False)

    generator_streams = {
        gid: f"benchmark_generator{gid}"
        for gid in generator_ids
    }
    stat_streams = {
        gid: f"benchmark_stats{gid}"
        for gid in generator_ids
    }

    for gid in generator_ids:
        pipe.add_stream(
            generator_streams[gid],
            shape=(SIGNALS_PER_GENERATOR, frame_rows),
            dtype=np.float64,
            cache_align=True,
        )
        pipe.add_stream(
            stat_streams[gid],
            shape=BENCHMARK_STATS_SHAPE,
            dtype=np.float64,
            cache_align=True,
        )
        pipe.add_task(
            f"benchmark_generator_task{gid}",
            fn=partial(_benchmark_generator, seed=gid - 1, frame_rows=frame_rows),
            writes={"output": generator_streams[gid]},
            events={"shutdown": _headless_generator_shutdown_event_name(gid)},
        )

    pipe.add_task(
        "console_stats",
        fn=partial(
            _headless_console_stats_task,
            report_interval_s=report_interval_s,
            mode_name=mode_name,
            frame_rows=frame_rows,
            generator_ids=generator_ids,
            total_signals=total_signals,
        ),
        reads={
            f"stats{gid}": stat_streams[gid]
            for gid in generator_ids
        },
        events={"shutdown": _headless_console_shutdown_event_name()},
    )

    for gid in generator_ids:
        pipe.add_task(
            f"fft_benchmark{gid}",
            fn=partial(_fft_benchmark_task, frame_rows=frame_rows),
            reads={"samples": generator_streams[gid]},
            writes={"stats": stat_streams[gid]},
            events={"shutdown": _headless_fft_shutdown_event_name(gid)},
        )


def main() -> None:
    args = _parse_args()

    with pythusa.Pipeline("Demo") as pipe:
        if args.no_gui:
            frame_rows = _resolve_headless_frame_rows(args)
            generator_ids = tuple(range(1, args.generators + 1))
            _configure_headless(
                pipe,
                report_interval_s=args.report_interval,
                mode_name=args.mode,
                frame_rows=frame_rows,
                generator_ids=generator_ids,
            )
            pipe.start()
            try:
                if args.duration is None:
                    while True:
                        time.sleep(1.0)
                else:
                    time.sleep(args.duration)
            except KeyboardInterrupt:
                pass
            finally:
                pipe._manager._events[_headless_console_shutdown_event_name()].signal()
                for gid in generator_ids:
                    pipe._manager._events[_headless_generator_shutdown_event_name(gid)].signal()
                    pipe._manager._events[_headless_fft_shutdown_event_name(gid)].signal()
                pipe.join(timeout=max(5.0, args.report_interval + 1.0))
        else:
            _configure_gui(pipe)
            pipe.run()


if __name__ == "__main__":
    main()
