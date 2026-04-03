from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from functools import partial
import time

import numpy as np
import pythusa

from data import (
    BOOK_COLS,
    COL_TS,
    DEFAULT_DISPLAY_ROWS,
    DEFAULT_REPORT_INTERVAL_S,
    DEFAULT_TARGET_BANK_GB,
    METRIC_COLS,
    METRIC_DRAWDOWN,
    METRIC_FLOW_Z,
    METRIC_IMBALANCE,
    METRIC_MICRO_EDGE_BPS,
    METRIC_MOMENTUM,
    METRIC_REALIZED_VOL,
    METRIC_SESSION_RETURN,
    METRIC_SPREAD_BPS,
    METRIC_VWAP_BPS,
    PERF_COLS,
    PERF_GBPS,
    PERF_LATENCY_MEAN_US,
    PERF_LATENCY_P50_US,
    PERF_LATENCY_P95_US,
    PERF_LATENCY_P99_US,
    PERF_TICKS_PER_SECOND,
    PERF_TOTAL_FRAMES,
    PERF_TOTAL_GB,
    PERF_TOTAL_TICKS,
    SYMBOLS,
    BaselineMetrics,
    SimulationConfig,
    SymbolScenario,
    bank_frames_for_target_gb,
    baseline_benchmark,
    build_scenarios,
    build_symbol_bank,
    config_from_env,
    describe_demo,
    evaluate_tick_frame,
    reset_quant_state,
)

DISPLAY_DTYPE = np.float32
DISPLAY_FRAME_COLS = 3
DISPLAY_TS_COL = 0
DISPLAY_MID_COL = 1
DISPLAY_EMA_COL = 2
DEFAULT_DISPLAY_HISTORY = 96
SLEEP = 1e-4
STAMP_COLS = 2
STAMP_SEQUENCE = 0
STAMP_PUBLISHED_NS = 1
GRAPH_PADDING_RATIO = 0.08
GRAPH_PADDING_FLOOR_RATIO = 2.5e-5
GRAPH_MIN_SPAN_RATIO = 1.0e-4
GRAPH_RANGE_SHRINK_ALPHA = 0.18
PLOT_PANEL_WIDTH_FILL = 0.88
PLOT_PANEL_HEIGHT_FILL = 0.84
PLOT_PANEL_MIN_WIDTH = 220.0
PLOT_PANEL_MIN_HEIGHT = 145.0
HOTPATH_EVENT_POLL_INTERVAL = 1024 * 8
HOTPATH_EVENT_POLL_MASK = HOTPATH_EVENT_POLL_INTERVAL - 1


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    frame_ticks: int
    raw_ring_frames: int
    derived_ring_frames: int
    perf_report_interval_s: float
    idle_sleep_s: float
    description: str


RUNTIME_PROFILES = {
    "latency": RuntimeProfile(
        name="latency",
        frame_ticks=256,
        raw_ring_frames=4,
        derived_ring_frames=8,
        perf_report_interval_s=0.10,
        idle_sleep_s=1e-5,
        description="Small frames and shallow raw rings minimize batching and queue buildup.",
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        frame_ticks=2048,
        raw_ring_frames=16,
        derived_ring_frames=16,
        perf_report_interval_s=0.25,
        idle_sleep_s=1e-4,
        description="Balanced frame size keeps latency reasonable while sustaining strong throughput.",
    ),
    "throughput": RuntimeProfile(
        name="throughput",
        frame_ticks=8192,
        raw_ring_frames=32,
        derived_ring_frames=32,
        perf_report_interval_s=0.50,
        idle_sleep_s=2e-4,
        description="Large frames and deeper rings maximize payload moved per unit overhead.",
    ),
}


def _empty_display_series() -> np.ndarray:
    return np.empty((0, DISPLAY_FRAME_COLS), dtype=DISPLAY_DTYPE)


@dataclass
class DisplaySeriesState:
    max_frames: int
    frames: deque[np.ndarray] = field(init=False)
    series: np.ndarray = field(default_factory=_empty_display_series)
    y_limits: tuple[float, float] | None = None
    last_timestamp: float | None = None

    def __post_init__(self) -> None:
        self.frames = deque(maxlen=self.max_frames)

    def append_batch(self, frames: list[np.ndarray]) -> bool:
        had_update = False
        for frame in frames:
            if self._needs_reset(frame):
                self.reset()
            self.frames.append(frame)
            if frame.size:
                self.last_timestamp = float(frame[-1, DISPLAY_TS_COL])
            had_update = True
        if had_update:
            self._refresh()
        return had_update

    def reset(self) -> None:
        self.frames.clear()
        self.series = _empty_display_series()
        self.y_limits = None
        self.last_timestamp = None

    def _needs_reset(self, frame: np.ndarray) -> bool:
        if self.last_timestamp is None or frame.size == 0:
            return False
        return float(frame[0, DISPLAY_TS_COL]) <= self.last_timestamp

    def _refresh(self) -> None:
        self.series = _concat_display_frames(self.frames)
        target = _target_display_limits(self.series)
        self.y_limits = _stabilize_graph_limits(self.y_limits, target)


def generator_task(output, stamp, stop_event, *, symbol: str, config: SimulationConfig, idle_sleep_s: float) -> None:
    bank, _ = build_symbol_bank(symbol, config=config)
    frame_shape = (config.frame_ticks, BOOK_COLS)
    frame_index = 0
    frame_sequence = 0
    poll_counter = 0
    poll_mask = HOTPATH_EVENT_POLL_MASK
    sleep = time.sleep
    stop_is_open = stop_event.is_open

    while True:
        if (poll_counter & poll_mask) == 0 and stop_is_open():
            break
        poll_counter += 1
        output_view = output.look()
        stamp_view = stamp.look()
        if output_view is None or stamp_view is None:
            if output_view is not None:
                del output_view
            if stamp_view is not None:
                del stamp_view
            sleep(idle_sleep_s)
            continue
        out = np.frombuffer(output_view, dtype=np.float64).reshape(frame_shape)
        stamp_out = np.frombuffer(stamp_view, dtype=np.int64, count=STAMP_COLS)
        out[:] = bank[frame_index]
        stamp_out[STAMP_SEQUENCE] = frame_sequence
        stamp_out[STAMP_PUBLISHED_NS] = time.perf_counter_ns()
        output.increment()
        stamp.increment()
        frame_index = (frame_index + 1) % config.bank_frames
        frame_sequence += 1


def analyzer_task(
    raw,
    stamp,
    metrics,
    perf,
    stop_event,
    display=None,
    *,
    config: SimulationConfig,
    display_rows: int,
    perf_report_interval_s: float,
    idle_sleep_s: float,
) -> None:
    state = reset_quant_state()
    display_enabled = display is not None
    processed_ticks_window = 0
    processed_bytes_window = 0
    total_ticks = 0
    total_bytes = 0
    total_frames = 0
    latency_samples_us: list[float] = []
    last_report = time.perf_counter()
    frame_shape = (config.frame_ticks, BOOK_COLS)
    poll_counter = 0
    poll_mask = HOTPATH_EVENT_POLL_MASK
    sleep = time.sleep
    stop_is_open = stop_event.is_open

    while True:
        if (poll_counter & poll_mask) == 0 and stop_is_open():
            break
        poll_counter += 1
        raw_view = raw.look()
        stamp_view = stamp.look()
        if raw_view is None or stamp_view is None:
            if raw_view is not None:
                del raw_view
            if stamp_view is not None:
                del stamp_view
            sleep(idle_sleep_s)
            continue

        frame = np.frombuffer(raw_view, dtype=np.float64).reshape(frame_shape)
        stamp_values = np.frombuffer(stamp_view, dtype=np.int64, count=STAMP_COLS)
        mid_trace, ema_trace, metric_values, state = evaluate_tick_frame(
            frame,
            state,
            annualization_factor=config.annualization_factor,
            compute_traces=display_enabled,
        )
        if display_enabled:
            display_view = display.look()
            if display_view is not None:
                assert mid_trace is not None and ema_trace is not None
                _write_display_view(
                    display_view,
                    frame[:, COL_TS],
                    mid_trace,
                    ema_trace,
                    rows=display_rows,
                )
                display.increment()
        _try_write_vector(metrics, metric_values, length=METRIC_COLS)

        finished_ns = time.perf_counter_ns()
        latency_samples_us.append(
            max(0.0, (finished_ns - int(stamp_values[STAMP_PUBLISHED_NS])) / 1_000.0)
        )
        raw.increment()
        stamp.increment()
        total_frames += 1
        total_ticks += config.frame_ticks
        total_bytes += frame.nbytes
        processed_ticks_window += config.frame_ticks
        processed_bytes_window += frame.nbytes

        now = time.perf_counter()
        elapsed = now - last_report
        if elapsed < perf_report_interval_s:
            continue

        latency_values = np.asarray(latency_samples_us, dtype=np.float64)
        if latency_values.size:
            latency_mean_us = float(latency_values.mean())
            latency_p50_us = float(np.percentile(latency_values, 50.0))
            latency_p95_us = float(np.percentile(latency_values, 95.0))
            latency_p99_us = float(np.percentile(latency_values, 99.0))
        else:
            latency_mean_us = 0.0
            latency_p50_us = 0.0
            latency_p95_us = 0.0
            latency_p99_us = 0.0

        perf_values = np.array(
            (
                processed_ticks_window / elapsed,
                (processed_bytes_window * 8.0) / elapsed / 1_000_000_000.0,
                float(total_ticks),
                total_bytes / 1_000_000_000.0,
                float(total_frames),
                latency_mean_us,
                latency_p50_us,
                latency_p95_us,
                latency_p99_us,
            ),
            dtype=np.float64,
        )
        _try_write_vector(perf, perf_values, length=PERF_COLS)
        processed_ticks_window = 0
        processed_bytes_window = 0
        latency_samples_us.clear()
        last_report = now


def ui_task(
    *,
    config: SimulationConfig,
    profile: RuntimeProfile,
    scenarios: dict[str, SymbolScenario],
    baseline: BaselineMetrics,
    insights: tuple[str, ...],
    display_rows: int,
    display_history: int,
    **bindings,
) -> None:
    import glfw
    import imgui
    from OpenGL import GL as gl
    from imgui.integrations.glfw import GlfwRenderer

    from graphing import graph_display_series

    if not glfw.init():
        raise SystemExit("failed to initialize glfw")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(960, 600, "pythusa stock quant lab", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("failed to create window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    imgui.create_context()
    _apply_imgui_theme(imgui)
    impl = GlfwRenderer(window)

    display_readers = {symbol: bindings[f"display_{symbol}"] for symbol in SYMBOLS}
    metric_readers = {symbol: bindings[f"metric_{symbol}"] for symbol in SYMBOLS}
    perf_readers = {symbol: bindings[f"perf_{symbol}"] for symbol in SYMBOLS}
    stop_events = [
        bindings[f"stop_gen_{symbol}"]
        for symbol in SYMBOLS
    ] + [
        bindings[f"stop_ana_{symbol}"]
        for symbol in SYMBOLS
    ]

    for reader in (*display_readers.values(), *metric_readers.values(), *perf_readers.values()):
        reader.set_blocking(False)

    display_states = {symbol: DisplaySeriesState(max_frames=display_history) for symbol in SYMBOLS}
    metrics = {symbol: np.zeros(METRIC_COLS, dtype=np.float64) for symbol in SYMBOLS}
    perf = {symbol: np.zeros(PERF_COLS, dtype=np.float64) for symbol in SYMBOLS}

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()
            imgui.new_frame()

            _drain_display_states(display_readers, display_states)
            _drain_reader_vectors(metric_readers, metrics, length=METRIC_COLS)
            _drain_reader_vectors(perf_readers, perf, length=PERF_COLS)

            summary = _live_summary(metrics, perf, baseline)
            live_lines = _live_lines(summary=summary, metrics=metrics, perf=perf, baseline=baseline)

            imgui.begin("Pythusa L3 Quant Dashboard")
            imgui.text_colored("Simulated L3 Microstructure Desk", 0.300, 0.780, 0.960, 1.0)
            imgui.text_disabled(
                "Brownian-anchor order books feeding 8 parallel quant workers through zero-copy shared-memory streams."
            )
            imgui.text_unformatted(
                (
                    f"Mode {profile.name} | seed {config.seed} | {config.frame_ticks:,} ticks/frame | "
                    f"raw ring {profile.raw_ring_frames} | derived ring {profile.derived_ring_frames} | "
                    f"{config.bank_frames} frames/symbol | {config.tick_ms:.1f} ms/tick | "
                    f"display {display_history} frames"
                )
            )
            imgui.text_disabled(profile.description)
            imgui.spacing()

            avail = imgui.get_content_region_available()
            total_width = float(avail[0] if isinstance(avail, tuple) else avail.x)
            sidebar_width = min(360.0, max(300.0, total_width * 0.24))
            deck_width = max(320.0, total_width - sidebar_width - 12.0)
            card_width = max(105.0, (total_width - 48.0) / 7.0)

            _render_metric_card(
                imgui,
                title="Live Ticks/s",
                value=_format_rate(summary["live_ticks_s"], unit="ticks/s"),
                accent=(0.220, 0.770, 0.945, 1.0),
                width=card_width,
            )
            imgui.same_line()
            _render_metric_card(
                imgui,
                title="Live Gbit/s",
                value=f"{summary['live_gbps']:.2f}",
                accent=(0.420, 0.860, 0.620, 1.0),
                width=card_width,
            )
            imgui.same_line()
            _render_metric_card(
                imgui,
                title="Mean Lat",
                value=_format_latency_ms(summary["latency_mean_us"]),
                accent=(0.985, 0.640, 0.250, 1.0),
                width=card_width,
            )
            imgui.same_line()
            _render_metric_card(
                imgui,
                title="P95 Lat",
                value=_format_latency_ms(summary["latency_p95_us"]),
                accent=(0.930, 0.420, 0.390, 1.0),
                width=card_width,
            )
            imgui.same_line()
            _render_metric_card(
                imgui,
                title="P99 Lat",
                value=_format_latency_ms(summary["latency_p99_us"]),
                accent=(0.730, 0.620, 0.980, 1.0),
                width=card_width,
            )
            imgui.same_line()
            _render_metric_card(
                imgui,
                title="Speedup",
                value=f"{summary['speedup']:.2f}x",
                accent=(0.985, 0.640, 0.250, 1.0),
                width=card_width,
            )
            imgui.same_line()
            _render_metric_card(
                imgui,
                title="Processed",
                value=_format_count(summary["total_ticks"]),
                accent=(0.930, 0.420, 0.390, 1.0),
                width=card_width,
            )

            imgui.spacing()
            imgui.begin_child("sidebar", sidebar_width, 0, border=True)
            imgui.text_unformatted("Replay Design")
            for line in insights:
                imgui.text_wrapped(f"- {line}")
            imgui.text_wrapped(
                f"- Payload processed so far: {summary['total_gb']:.2f} GB | "
                f"display {display_rows} rows x {display_history} frames."
            )

            imgui.spacing()
            imgui.separator()
            imgui.text_unformatted("Live Cross-Section")
            for line in live_lines:
                imgui.text_wrapped(f"- {line}")

            imgui.spacing()
            imgui.separator()
            imgui.text_unformatted("Universe Setup")
            for symbol in SYMBOLS:
                scenario = scenarios[symbol]
                color = (0.420, 0.860, 0.620, 1.0) if scenario.trend_sign > 0 else (0.930, 0.420, 0.390, 1.0)
                imgui.text_colored(
                    f"{symbol} {scenario.trend_label}",
                    *color,
                )
                imgui.text_disabled(
                    (
                        f"{scenario.anchor_count} anchors | "
                        f"path {scenario.return_pct:+.2%} | "
                        f"spread {scenario.avg_spread_bps:.2f} bps"
                    )
                )
            imgui.end_child()

            imgui.same_line()
            imgui.begin_child("plot_deck", deck_width, 0, border=False)
            deck_avail = imgui.get_content_region_available()
            deck_w = float(deck_avail[0] if isinstance(deck_avail, tuple) else deck_avail.x)
            deck_h = float(deck_avail[1] if isinstance(deck_avail, tuple) else deck_avail.y)
            plot_width = max(PLOT_PANEL_MIN_WIDTH, ((deck_w - 10.0) / 2.0) * PLOT_PANEL_WIDTH_FILL)
            plot_height = max(PLOT_PANEL_MIN_HEIGHT, ((deck_h - 24.0) / 4.0) * PLOT_PANEL_HEIGHT_FILL)

            for index, symbol in enumerate(SYMBOLS):
                if index % 2 == 1:
                    imgui.same_line()
                _render_symbol_panel(
                    imgui,
                    graph_display_series=graph_display_series,
                    symbol=symbol,
                    scenario=scenarios[symbol],
                    display_state=display_states[symbol],
                    metrics=metrics[symbol],
                    perf=perf[symbol],
                    width=plot_width,
                    height=plot_height,
                )
            imgui.end_child()
            imgui.end()

            imgui.render()
            gl.glClearColor(0.020, 0.030, 0.050, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
    finally:
        for event in stop_events:
            event.signal()
        impl.shutdown()
        glfw.terminate()


def console_task(
    stop_console,
    *,
    baseline: BaselineMetrics,
    profile: RuntimeProfile,
    report_interval_s: float,
    **bindings,
) -> None:
    metric_readers = {symbol: bindings[f"metric_{symbol}"] for symbol in SYMBOLS}
    perf_readers = {symbol: bindings[f"perf_{symbol}"] for symbol in SYMBOLS}
    for reader in (*metric_readers.values(), *perf_readers.values()):
        reader.set_blocking(False)

    metrics = {symbol: np.zeros(METRIC_COLS, dtype=np.float64) for symbol in SYMBOLS}
    perf = {symbol: np.zeros(PERF_COLS, dtype=np.float64) for symbol in SYMBOLS}
    last_report = time.perf_counter()
    sleep = time.sleep

    while not stop_console.is_open():
        had_update = False
        had_update |= _drain_reader_vectors(metric_readers, metrics, length=METRIC_COLS)
        had_update |= _drain_reader_vectors(perf_readers, perf, length=PERF_COLS)

        now = time.perf_counter()
        if now - last_report >= report_interval_s:
            summary = _live_summary(metrics, perf, baseline)
            leader = summary["leader"]
            laggard = summary["laggard"]
            flow = summary["flow_symbol"]
            latency_symbol = summary["latency_symbol"]
            print(
                (
                    f"mode={profile.name} "
                    f"live={summary['live_gbps']:.2f} Gbit/s "
                    f"| ticks={_format_count(summary['live_ticks_s'])}/s "
                    f"| mean={_format_latency_ms(summary['latency_mean_us'])} "
                    f"| p95={_format_latency_ms(summary['latency_p95_us'])} "
                    f"| p99={_format_latency_ms(summary['latency_p99_us'])} "
                    f"| speedup={summary['speedup']:.2f}x "
                    f"| processed={_format_count(summary['total_ticks'])} ticks "
                    f"| leader={leader} {metrics[leader][METRIC_SESSION_RETURN]:+.2%} "
                    f"| laggard={laggard} {metrics[laggard][METRIC_SESSION_RETURN]:+.2%} "
                    f"| flow={flow} z {metrics[flow][METRIC_FLOW_Z]:+.2f} "
                    f"| worst-p95={latency_symbol} {_format_latency_ms(perf[latency_symbol][PERF_LATENCY_P95_US])}"
                ),
                flush=True,
            )
            last_report = now
            continue

        if not had_update:
            sleep(SLEEP)


def _configure_pipeline(
    pipe: pythusa.Pipeline,
    *,
    config: SimulationConfig,
    profile: RuntimeProfile,
    scenarios: dict[str, SymbolScenario],
    baseline: BaselineMetrics,
    insights: tuple[str, ...],
    headless: bool,
    report_interval_s: float,
    display_rows: int,
    display_history: int,
) -> None:
    for symbol in SYMBOLS:
        pipe.add_event(f"stop_gen_{symbol}", initial_state=False)
        pipe.add_event(f"stop_ana_{symbol}", initial_state=False)
        pipe.add_stream(
            f"raw_{symbol}",
            shape=(config.frame_ticks, BOOK_COLS),
            dtype=np.float64,
            frames=profile.raw_ring_frames,
            cache_align=True,
        )
        pipe.add_stream(
            f"stamp_{symbol}",
            shape=(STAMP_COLS,),
            dtype=np.int64,
            frames=profile.raw_ring_frames,
            cache_align=True,
        )
        if not headless:
            pipe.add_stream(
                f"display_{symbol}",
                shape=(display_rows, DISPLAY_FRAME_COLS),
                dtype=DISPLAY_DTYPE,
                frames=profile.derived_ring_frames,
                cache_align=True,
            )
        pipe.add_stream(
            f"metric_{symbol}",
            shape=(METRIC_COLS,),
            dtype=np.float64,
            frames=profile.derived_ring_frames,
            cache_align=True,
        )
        pipe.add_stream(
            f"perf_{symbol}",
            shape=(PERF_COLS,),
            dtype=np.float64,
            frames=profile.derived_ring_frames,
            cache_align=True,
        )
        pipe.add_task(
            f"generator_{symbol}",
            fn=partial(generator_task, symbol=symbol, config=config, idle_sleep_s=profile.idle_sleep_s),
            writes={"output": f"raw_{symbol}", "stamp": f"stamp_{symbol}"},
            events={"stop_event": f"stop_gen_{symbol}"},
        )
        pipe.add_task(
            f"analyzer_{symbol}",
            fn=partial(
                analyzer_task,
                config=config,
                display_rows=display_rows,
                perf_report_interval_s=profile.perf_report_interval_s,
                idle_sleep_s=profile.idle_sleep_s,
            ),
            reads={"raw": f"raw_{symbol}", "stamp": f"stamp_{symbol}"},
            writes=(
                {
                    "display": f"display_{symbol}",
                    "metrics": f"metric_{symbol}",
                    "perf": f"perf_{symbol}",
                }
                if not headless
                else {
                    "metrics": f"metric_{symbol}",
                    "perf": f"perf_{symbol}",
                }
            ),
            events={"stop_event": f"stop_ana_{symbol}"},
        )

    if headless:
        pipe.add_event("stop_console", initial_state=False)
        pipe.add_task(
            "console",
            fn=partial(
                console_task,
                baseline=baseline,
                profile=profile,
                report_interval_s=report_interval_s,
            ),
            reads={
                **{f"metric_{symbol}": f"metric_{symbol}" for symbol in SYMBOLS},
                **{f"perf_{symbol}": f"perf_{symbol}" for symbol in SYMBOLS},
            },
            events={"stop_console": "stop_console"},
        )
        return

    pipe.add_task(
        "ui",
        fn=partial(
            ui_task,
            config=config,
            profile=profile,
            scenarios=scenarios,
            baseline=baseline,
            insights=insights,
            display_rows=display_rows,
            display_history=display_history,
        ),
        reads={
            **{f"display_{symbol}": f"display_{symbol}" for symbol in SYMBOLS},
            **{f"metric_{symbol}": f"metric_{symbol}" for symbol in SYMBOLS},
            **{f"perf_{symbol}": f"perf_{symbol}" for symbol in SYMBOLS},
        },
        events={
            **{f"stop_gen_{symbol}": f"stop_gen_{symbol}" for symbol in SYMBOLS},
            **{f"stop_ana_{symbol}": f"stop_ana_{symbol}" for symbol in SYMBOLS},
        },
    )


def _write_display_view(
    view,
    timestamps: np.ndarray,
    mid_values: np.ndarray,
    ema_values: np.ndarray,
    *,
    rows: int,
) -> None:
    out = np.frombuffer(view, dtype=DISPLAY_DTYPE).reshape((rows, DISPLAY_FRAME_COLS))
    if rows == timestamps.size:
        out[:, DISPLAY_TS_COL] = timestamps
        out[:, DISPLAY_MID_COL] = mid_values
        out[:, DISPLAY_EMA_COL] = ema_values
        return

    indices = np.linspace(0, timestamps.size - 1, rows, dtype=np.int64)
    out[:, DISPLAY_TS_COL] = timestamps[indices]
    out[:, DISPLAY_MID_COL] = mid_values[indices]
    out[:, DISPLAY_EMA_COL] = ema_values[indices]


def _try_write_vector(stream, values: np.ndarray, *, length: int) -> None:
    view = stream.look()
    if view is None:
        return
    out = np.frombuffer(view, dtype=np.float64, count=length)
    out[:] = values
    stream.increment()


def _drain_display_states(
    readers: dict[str, object],
    states: dict[str, DisplaySeriesState],
) -> bool:
    had_update = False
    for key, reader in readers.items():
        batch: list[np.ndarray] = []
        while True:
            frame = reader.read()
            if frame is None:
                break
            batch.append(frame)
        if batch and states[key].append_batch(batch):
            had_update = True
    return had_update


def _drain_reader_vectors(
    readers: dict[str, object],
    values: dict[str, np.ndarray],
    *,
    length: int,
) -> bool:
    had_update = False
    for key, reader in readers.items():
        view = reader.look()
        if view is None:
            continue
        values[key][:] = np.frombuffer(view, dtype=np.float64, count=length)
        reader.increment()
        had_update = True
    return had_update


def _concat_display_frames(frames: deque[np.ndarray]) -> np.ndarray:
    if not frames:
        return _empty_display_series()
    if len(frames) == 1:
        return frames[0]
    return np.concatenate(tuple(frames), axis=0)


def _stabilize_graph_limits(
    previous: tuple[float, float] | None,
    target: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if target is None:
        return previous
    if previous is None:
        return target

    prev_min, prev_max = previous
    target_min, target_max = target
    next_min = target_min if target_min <= prev_min else prev_min + (target_min - prev_min) * GRAPH_RANGE_SHRINK_ALPHA
    next_max = target_max if target_max >= prev_max else prev_max + (target_max - prev_max) * GRAPH_RANGE_SHRINK_ALPHA
    if next_max <= next_min:
        return target
    return (next_min, next_max)


def _target_display_limits(
    series: np.ndarray,
) -> tuple[float, float] | None:
    if series.size == 0:
        return None

    values = series[:, DISPLAY_MID_COL:DISPLAY_EMA_COL + 1]
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None

    y_min = float(finite_values.min())
    y_max = float(finite_values.max())

    span = y_max - y_min
    scale = max(abs(y_min), abs(y_max), 1.0)
    min_span = max(scale * GRAPH_MIN_SPAN_RATIO, 1e-6)
    if span < min_span:
        center = 0.5 * (y_min + y_max)
        half_span = 0.5 * min_span
        y_min = center - half_span
        y_max = center + half_span
        span = min_span

    padding = max(span * GRAPH_PADDING_RATIO, scale * GRAPH_PADDING_FLOOR_RATIO, 1e-6)
    return (y_min - padding, y_max + padding)


def _live_summary(
    metrics: dict[str, np.ndarray],
    perf: dict[str, np.ndarray],
    baseline: BaselineMetrics,
) -> dict[str, object]:
    live_ticks_s = float(sum(values[PERF_TICKS_PER_SECOND] for values in perf.values()))
    live_gbps = float(sum(values[PERF_GBPS] for values in perf.values()))
    total_ticks = int(sum(values[PERF_TOTAL_TICKS] for values in perf.values()))
    total_gb = float(sum(values[PERF_TOTAL_GB] for values in perf.values()))
    speedup = 0.0 if baseline.ticks_per_second <= 0.0 else live_ticks_s / baseline.ticks_per_second

    leader = max(SYMBOLS, key=lambda symbol: metrics[symbol][METRIC_SESSION_RETURN])
    laggard = min(SYMBOLS, key=lambda symbol: metrics[symbol][METRIC_SESSION_RETURN])
    flow_symbol = max(SYMBOLS, key=lambda symbol: abs(metrics[symbol][METRIC_FLOW_Z]))
    micro_symbol = max(SYMBOLS, key=lambda symbol: abs(metrics[symbol][METRIC_MICRO_EDGE_BPS]))
    tightest = min(SYMBOLS, key=lambda symbol: metrics[symbol][METRIC_SPREAD_BPS] if metrics[symbol][METRIC_SPREAD_BPS] > 0.0 else np.inf)
    widest = max(SYMBOLS, key=lambda symbol: metrics[symbol][METRIC_SPREAD_BPS])
    active_symbols = [symbol for symbol in SYMBOLS if perf[symbol][PERF_TOTAL_FRAMES] > 0.0]
    if active_symbols:
        weights = np.fromiter(
            (max(perf[symbol][PERF_TICKS_PER_SECOND], 1.0) for symbol in active_symbols),
            dtype=np.float64,
        )
        latency_mean_us = float(
            np.average([perf[symbol][PERF_LATENCY_MEAN_US] for symbol in active_symbols], weights=weights)
        )
        latency_p50_us = float(
            np.average([perf[symbol][PERF_LATENCY_P50_US] for symbol in active_symbols], weights=weights)
        )
        latency_p95_us = float(max(perf[symbol][PERF_LATENCY_P95_US] for symbol in active_symbols))
        latency_p99_us = float(max(perf[symbol][PERF_LATENCY_P99_US] for symbol in active_symbols))
        latency_symbol = max(active_symbols, key=lambda symbol: perf[symbol][PERF_LATENCY_P95_US])
    else:
        latency_mean_us = 0.0
        latency_p50_us = 0.0
        latency_p95_us = 0.0
        latency_p99_us = 0.0
        latency_symbol = SYMBOLS[0]

    return {
        "live_ticks_s": live_ticks_s,
        "live_gbps": live_gbps,
        "total_ticks": total_ticks,
        "total_gb": total_gb,
        "speedup": speedup,
        "latency_mean_us": latency_mean_us,
        "latency_p50_us": latency_p50_us,
        "latency_p95_us": latency_p95_us,
        "latency_p99_us": latency_p99_us,
        "latency_symbol": latency_symbol,
        "leader": leader,
        "laggard": laggard,
        "flow_symbol": flow_symbol,
        "micro_symbol": micro_symbol,
        "tightest": tightest,
        "widest": widest,
    }


def _live_lines(
    *,
    summary: dict[str, object],
    metrics: dict[str, np.ndarray],
    perf: dict[str, np.ndarray],
    baseline: BaselineMetrics,
) -> tuple[str, ...]:
    leader = summary["leader"]
    laggard = summary["laggard"]
    flow_symbol = summary["flow_symbol"]
    micro_symbol = summary["micro_symbol"]
    tightest = summary["tightest"]
    widest = summary["widest"]
    latency_symbol = summary["latency_symbol"]

    total_frames = int(sum(values[PERF_TOTAL_FRAMES] for values in perf.values()))

    return (
        (
            f"Live {summary['live_gbps']:.2f} Gbit/s and "
            f"{_format_rate(summary['live_ticks_s'], unit='ticks/s')} against a "
            f"{_format_rate(baseline.ticks_per_second, unit='ticks/s')} serial baseline."
        ),
        (
            f"Latency mean {_format_latency_ms(summary['latency_mean_us'])} | "
            f"desk p95 {_format_latency_ms(summary['latency_p95_us'])} | "
            f"desk p99 {_format_latency_ms(summary['latency_p99_us'])}."
        ),
        (
            f"Leader {leader} {metrics[leader][METRIC_SESSION_RETURN]:+.2%}; "
            f"laggard {laggard} {metrics[laggard][METRIC_SESSION_RETURN]:+.2%}; "
            f"frames processed {total_frames:,}."
        ),
        (
            f"Flow stress {flow_symbol} z {metrics[flow_symbol][METRIC_FLOW_Z]:+.2f}; "
            f"microprice edge {micro_symbol} {metrics[micro_symbol][METRIC_MICRO_EDGE_BPS]:+.2f} bps."
        ),
        (
            f"Worst latency lane: {latency_symbol} p95 {_format_latency_ms(perf[latency_symbol][PERF_LATENCY_P95_US])}."
        ),
        (
            f"Spread regime: tightest {tightest} {metrics[tightest][METRIC_SPREAD_BPS]:.2f} bps; "
            f"widest {widest} {metrics[widest][METRIC_SPREAD_BPS]:.2f} bps."
        ),
    )


def _render_symbol_panel(
    imgui,
    *,
    graph_display_series,
    symbol: str,
    scenario: SymbolScenario,
    display_state: DisplaySeriesState,
    metrics: np.ndarray,
    perf: np.ndarray,
    width: float,
    height: float,
) -> None:
    color = (0.420, 0.860, 0.620, 1.0) if metrics[METRIC_SESSION_RETURN] >= 0.0 else (0.930, 0.420, 0.390, 1.0)
    imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.048, 0.070, 0.108, 1.0)
    imgui.push_style_color(imgui.COLOR_BORDER, color[0], color[1], color[2], 0.70)
    imgui.begin_child(f"panel_{symbol}", width, height, border=True)
    imgui.text_colored(symbol, *color)
    imgui.same_line()
    imgui.text_disabled(scenario.trend_label)
    imgui.same_line()
    imgui.text_disabled(
        f"{_format_rate(perf[PERF_TICKS_PER_SECOND], unit='ticks/s')} | {perf[PERF_GBPS]:.2f} Gbit/s"
    )
    imgui.same_line()
    imgui.text_disabled(
        (
            f"| mean {_format_latency_ms(perf[PERF_LATENCY_MEAN_US])} "
            f"p95 {_format_latency_ms(perf[PERF_LATENCY_P95_US])} "
            f"p99 {_format_latency_ms(perf[PERF_LATENCY_P99_US])}"
        )
    )
    imgui.text_disabled(
        (
            f"ret {metrics[METRIC_SESSION_RETURN]:+.2%} | "
            f"mom {metrics[METRIC_MOMENTUM]:+.2%} | "
            f"vol {metrics[METRIC_REALIZED_VOL]:.2%} | "
            f"dd {metrics[METRIC_DRAWDOWN]:+.2%}"
        )
    )
    imgui.text_disabled(
        (
            f"imb {metrics[METRIC_IMBALANCE]:+.2f} | "
            f"flow z {metrics[METRIC_FLOW_Z]:+.2f} | "
            f"micro {metrics[METRIC_MICRO_EDGE_BPS]:+.2f} bps | "
            f"vwap {metrics[METRIC_VWAP_BPS]:+.2f} bps | "
            f"spr {metrics[METRIC_SPREAD_BPS]:.2f} bps"
        )
    )
    imgui.separator()

    avail = imgui.get_content_region_available()
    plot_width = float(avail[0] if isinstance(avail, tuple) else avail.x)
    plot_height = max(90.0, float(avail[1] if isinstance(avail, tuple) else avail.y) - 8.0)
    if display_state.series.size:
        graph_display_series(
            list(display_state.frames),
            size=(max(30.0, plot_width), plot_height),
            y_limits=display_state.y_limits,
            color=color,
            overlay_color=(0.985, 0.640, 0.250, 0.78),
            background_color=(0.028, 0.040, 0.065, 1.0),
            border_color=(0.150, 0.235, 0.330, 1.0),
            thickness=1.7,
            overlay_thickness=1.1,
        )
    else:
        imgui.text_disabled("Waiting for market data...")
    imgui.end_child()
    imgui.pop_style_color(2)


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


def _format_count(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000.0:
        return f"{value / 1_000_000_000.0:.2f}B"
    if magnitude >= 1_000_000.0:
        return f"{value / 1_000_000.0:.2f}M"
    if magnitude >= 1_000.0:
        return f"{value / 1_000.0:.1f}K"
    return f"{value:.0f}"


def _format_rate(value: float, *, unit: str) -> str:
    return f"{_format_count(value)} {unit}"


def _format_latency_ms(value_us: float) -> str:
    return f"{value_us / 1_000.0:.3f} ms"


def _parse_args() -> argparse.Namespace:
    defaults = config_from_env()
    parser = argparse.ArgumentParser(description="Run the simulated Pythusa stock quant L3 demo.")
    parser.add_argument(
        "--mode",
        choices=tuple(RUNTIME_PROFILES),
        default="balanced",
        help="Runtime profile: latency for lower frame delay, throughput for higher aggregate payload, balanced by default.",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed, help=f"Simulation seed (default: {defaults.seed})")
    parser.add_argument(
        "--frame-ticks",
        type=int,
        default=None,
        help="Override the mode's ticks per shared-memory frame.",
    )
    parser.add_argument(
        "--bank-frames",
        type=int,
        default=None,
        help="Override the computed frames per symbol replay bank.",
    )
    parser.add_argument(
        "--bank-gb",
        type=float,
        default=DEFAULT_TARGET_BANK_GB,
        help=f"Target total precomputed replay-bank size across all symbols in GB (default: {DEFAULT_TARGET_BANK_GB:.2f})",
    )
    parser.add_argument(
        "--stream-frames",
        type=int,
        default=None,
        help="Override the mode's shared-memory ring depth in frames.",
    )
    parser.add_argument(
        "--tick-ms",
        type=float,
        default=defaults.tick_ms,
        help=f"Tick spacing in milliseconds inside the simulated replay (default: {defaults.tick_ms})",
    )
    parser.add_argument(
        "--baseline-loops",
        type=int,
        default=defaults.baseline_loops,
        help=f"Serial benchmark passes used to compute the speedup card (default: {defaults.baseline_loops})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without ImGui and print live throughput to stdout.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional run time in seconds for headless mode before signaling shutdown.",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=None,
        help=f"Seconds between headless console reports (default: {DEFAULT_REPORT_INTERVAL_S})",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable the Pythusa runtime monitor and ring-pressure diagnostics in headless mode.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.10,
        help="Seconds between monitor samples when --monitor is enabled.",
    )
    parser.add_argument(
        "--display-history",
        type=int,
        default=DEFAULT_DISPLAY_HISTORY,
        help=f"Number of sampled frames to retain per symbol in the GUI history window (default: {DEFAULT_DISPLAY_HISTORY})",
    )
    args = parser.parse_args()
    if args.bank_gb <= 0.0:
        parser.error("--bank-gb must be greater than 0")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be greater than 0")
    if args.report_interval is not None and args.report_interval <= 0.0:
        parser.error("--report-interval must be greater than 0")
    if args.monitor_interval <= 0.0:
        parser.error("--monitor-interval must be greater than 0")
    if args.display_history <= 0:
        parser.error("--display-history must be greater than 0")
    return args


def _signal_shutdown(pipe: pythusa.Pipeline) -> None:
    for event in pipe._manager._events.values():
        if not event.is_open():
            event.signal()


def _print_monitor_snapshot(pipe: pythusa.Pipeline) -> None:
    snapshots = [snapshot for snapshot in pipe.metrics().values() if snapshot is not None]
    if not snapshots:
        return
    total_cpu = sum(snapshot.cpu_percent for snapshot in snapshots)
    total_rss = sum(snapshot.memory_rss_mb for snapshot in snapshots)
    print(
        f"monitor: cpu={total_cpu:.1f}% rss={total_rss:.1f} MB across {len(snapshots)} tasks",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    profile = RUNTIME_PROFILES[args.mode]
    frame_ticks = profile.frame_ticks if args.frame_ticks is None else args.frame_ticks
    bank_frames = (
        bank_frames_for_target_gb(frame_ticks=frame_ticks, target_bank_gb=args.bank_gb)
        if args.bank_frames is None
        else args.bank_frames
    )
    stream_frames = profile.raw_ring_frames if args.stream_frames is None else args.stream_frames
    if stream_frames <= 0:
        raise SystemExit("--stream-frames must be greater than 0")
    if stream_frames != profile.raw_ring_frames:
        profile = RuntimeProfile(
            name=profile.name,
            frame_ticks=frame_ticks,
            raw_ring_frames=stream_frames,
            derived_ring_frames=stream_frames,
            perf_report_interval_s=profile.perf_report_interval_s,
            idle_sleep_s=profile.idle_sleep_s,
            description=profile.description,
        )
    else:
        profile = RuntimeProfile(
            name=profile.name,
            frame_ticks=frame_ticks,
            raw_ring_frames=profile.raw_ring_frames,
            derived_ring_frames=profile.derived_ring_frames,
            perf_report_interval_s=profile.perf_report_interval_s,
            idle_sleep_s=profile.idle_sleep_s,
            description=profile.description,
    )
    config = SimulationConfig(
        seed=args.seed,
        frame_ticks=frame_ticks,
        bank_frames=bank_frames,
        tick_ms=args.tick_ms,
        baseline_loops=args.baseline_loops,
    )
    report_interval_s = DEFAULT_REPORT_INTERVAL_S if args.report_interval is None else args.report_interval
    display_rows = min(DEFAULT_DISPLAY_ROWS, config.frame_ticks)
    display_history = args.display_history
    scenarios = build_scenarios(config)
    baseline = baseline_benchmark(config, display_rows=display_rows, display_enabled=not args.headless)
    insights = describe_demo(config=config, scenarios=scenarios, baseline=baseline)

    with pythusa.Pipeline("StockQuantL3Demo") as pipe:
        _configure_pipeline(
            pipe,
            config=config,
            profile=profile,
            scenarios=scenarios,
            baseline=baseline,
            insights=insights,
            headless=args.headless,
            report_interval_s=report_interval_s,
            display_rows=display_rows,
            display_history=display_history,
        )

        if args.headless:
            if args.monitor:
                pipe.start_monitor(interval_s=args.monitor_interval)
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
                _signal_shutdown(pipe)
                pipe.join(timeout=5.0)
                if args.monitor:
                    _print_monitor_snapshot(pipe)
        else:
            pipe.run()


if __name__ == "__main__":
    main()
