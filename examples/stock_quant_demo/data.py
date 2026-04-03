from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import os
from time import perf_counter

import numpy as np

SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOG",
    "TSLA",
    "JPM",
)

DEFAULT_SEED = 7
DEFAULT_FRAME_TICKS = 2048
DEFAULT_TICK_MS = 10.0
DEFAULT_BASELINE_LOOPS = 4
DEFAULT_DISPLAY_ROWS = 128
DEFAULT_REPORT_INTERVAL_S = 1.0
DEFAULT_TARGET_BANK_BYTES = 4_000_000_000
DEFAULT_TARGET_BANK_GB = DEFAULT_TARGET_BANK_BYTES / 1_000_000_000.0
DEFAULT_TARGET_BANK_GIB = DEFAULT_TARGET_BANK_BYTES / float(1024 * 1024 * 1024)
DEFAULT_BASELINE_SAMPLE_FRAMES = 64

MOMENTUM_WINDOW = 256
VOL_WINDOW = 512
FLOW_WINDOW = 256
EMA_ALPHA = 2.0 / 65.0

BOOK_COLS = 15
METRIC_COLS = 9
PERF_COLS = 9

COL_TS = 0
COL_BID_1 = 1
COL_ASK_1 = 2
COL_BID_1_SZ = 3
COL_ASK_1_SZ = 4
COL_BID_2 = 5
COL_ASK_2 = 6
COL_BID_2_SZ = 7
COL_ASK_2_SZ = 8
COL_BID_3 = 9
COL_ASK_3 = 10
COL_BID_3_SZ = 11
COL_ASK_3_SZ = 12
COL_TRADE_PX = 13
COL_TRADE_SZ = 14

METRIC_SESSION_RETURN = 0
METRIC_MOMENTUM = 1
METRIC_REALIZED_VOL = 2
METRIC_DRAWDOWN = 3
METRIC_IMBALANCE = 4
METRIC_FLOW_Z = 5
METRIC_MICRO_EDGE_BPS = 6
METRIC_VWAP_BPS = 7
METRIC_SPREAD_BPS = 8

PERF_TICKS_PER_SECOND = 0
PERF_GBPS = 1
PERF_TOTAL_TICKS = 2
PERF_TOTAL_GB = 3
PERF_TOTAL_FRAMES = 4
PERF_LATENCY_MEAN_US = 5
PERF_LATENCY_P50_US = 6
PERF_LATENCY_P95_US = 7
PERF_LATENCY_P99_US = 8

_ITEMSIZE = np.dtype(np.float64).itemsize
_SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(SYMBOLS)}
DEFAULT_BANK_FRAMES = DEFAULT_TARGET_BANK_BYTES // (
    len(SYMBOLS) * DEFAULT_FRAME_TICKS * BOOK_COLS * _ITEMSIZE
)


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = DEFAULT_SEED
    frame_ticks: int = DEFAULT_FRAME_TICKS
    bank_frames: int = DEFAULT_BANK_FRAMES
    tick_ms: float = DEFAULT_TICK_MS
    baseline_loops: int = DEFAULT_BASELINE_LOOPS

    def __post_init__(self) -> None:
        if self.frame_ticks <= 0:
            raise ValueError("frame_ticks must be greater than 0")
        if self.bank_frames <= 0:
            raise ValueError("bank_frames must be greater than 0")
        if self.tick_ms <= 0.0:
            raise ValueError("tick_ms must be greater than 0")
        if self.baseline_loops <= 0:
            raise ValueError("baseline_loops must be greater than 0")

    @property
    def tick_dt_s(self) -> float:
        return self.tick_ms / 1_000.0

    @property
    def total_ticks(self) -> int:
        return self.frame_ticks * self.bank_frames

    @property
    def raw_frame_bytes(self) -> int:
        return self.frame_ticks * BOOK_COLS * _ITEMSIZE

    @property
    def raw_frame_bits(self) -> int:
        return self.raw_frame_bytes * 8

    @property
    def annualization_factor(self) -> float:
        ticks_per_session = max(1.0, (6.5 * 3_600.0) / self.tick_dt_s)
        return math.sqrt(252.0 * ticks_per_session)

    @property
    def total_bank_bytes(self) -> int:
        return self.raw_frame_bytes * self.bank_frames * len(SYMBOLS)

    @property
    def total_bank_gb(self) -> float:
        return self.total_bank_bytes / 1_000_000_000.0

    @property
    def total_bank_gib(self) -> float:
        return self.total_bank_bytes / float(1024 * 1024 * 1024)


@dataclass(frozen=True)
class SymbolSpec:
    base_price: float
    tick_size: float
    base_spread_bps: float
    vol_per_sqrt_tick: float
    base_depth: float
    base_trade: float


@dataclass(frozen=True)
class SymbolScenario:
    symbol: str
    trend_sign: int
    anchor_count: int
    start_price: float
    end_price: float
    low_price: float
    high_price: float
    avg_spread_bps: float

    @property
    def trend_label(self) -> str:
        return "uptrend" if self.trend_sign > 0 else "downtrend"

    @property
    def return_pct(self) -> float:
        return (self.end_price / self.start_price) - 1.0


@dataclass(frozen=True)
class BaselineMetrics:
    ticks_per_second: float
    gbps: float
    processed_ticks: int
    processed_gb: float
    loops: int


@dataclass
class QuantState:
    last_timestamp: float | None = None
    anchor_mid: float | None = None
    ema_mid: float | None = None
    prev_mid: float | None = None
    peak_mid: float = 0.0
    cum_notional: float = 0.0
    cum_volume: float = 0.0
    mid_window: deque[float] = field(default_factory=lambda: deque(maxlen=MOMENTUM_WINDOW))
    return_window: deque[float] = field(default_factory=lambda: deque(maxlen=VOL_WINDOW))
    flow_window: deque[float] = field(default_factory=lambda: deque(maxlen=FLOW_WINDOW))


SYMBOL_SPECS: dict[str, SymbolSpec] = {
    "AAPL": SymbolSpec(192.0, 0.01, 1.10, 4.7e-4, 1_500.0, 380.0),
    "MSFT": SymbolSpec(418.0, 0.01, 0.95, 4.2e-4, 1_350.0, 340.0),
    "NVDA": SymbolSpec(875.0, 0.01, 1.55, 7.8e-4, 1_850.0, 520.0),
    "AMZN": SymbolSpec(181.0, 0.01, 1.20, 5.1e-4, 1_420.0, 360.0),
    "META": SymbolSpec(505.0, 0.01, 1.05, 5.4e-4, 1_280.0, 320.0),
    "GOOG": SymbolSpec(154.0, 0.01, 0.95, 4.4e-4, 1_220.0, 300.0),
    "TSLA": SymbolSpec(172.0, 0.01, 1.80, 8.1e-4, 1_720.0, 480.0),
    "JPM": SymbolSpec(198.0, 0.01, 1.00, 3.9e-4, 1_180.0, 300.0),
}


def config_from_env() -> SimulationConfig:
    return SimulationConfig(
        seed=int(os.environ.get("SIM_SEED", str(DEFAULT_SEED))),
        frame_ticks=int(os.environ.get("SIM_FRAME_TICKS", str(DEFAULT_FRAME_TICKS))),
        bank_frames=int(os.environ.get("SIM_BANK_FRAMES", str(DEFAULT_BANK_FRAMES))),
        tick_ms=float(os.environ.get("SIM_TICK_MS", str(DEFAULT_TICK_MS))),
        baseline_loops=int(os.environ.get("SIM_BASELINE_LOOPS", str(DEFAULT_BASELINE_LOOPS))),
    )


def reset_quant_state() -> QuantState:
    return QuantState()


def bank_frames_for_target_gb(
    *,
    frame_ticks: int,
    target_bank_gb: float,
) -> int:
    if frame_ticks <= 0:
        raise ValueError("frame_ticks must be greater than 0")
    if target_bank_gb <= 0.0:
        raise ValueError("target_bank_gb must be greater than 0")

    target_bytes = int(target_bank_gb * 1_000_000_000.0)
    per_symbol_frame_bytes = frame_ticks * BOOK_COLS * _ITEMSIZE
    return max(1, target_bytes // (len(SYMBOLS) * per_symbol_frame_bytes))


def bank_frames_for_target_gib(
    *,
    frame_ticks: int,
    target_bank_gib: float,
) -> int:
    if target_bank_gib <= 0.0:
        raise ValueError("target_bank_gib must be greater than 0")
    target_bank_bytes = int(target_bank_gib * 1024 * 1024 * 1024)
    return bank_frames_for_target_gb(
        frame_ticks=frame_ticks,
        target_bank_gb=target_bank_bytes / 1_000_000_000.0,
    )


def build_symbol_bank(
    symbol: str,
    *,
    config: SimulationConfig,
) -> tuple[np.ndarray, SymbolScenario]:
    spec = SYMBOL_SPECS[symbol]
    rng = np.random.default_rng(_symbol_seed(symbol, config=config))
    total_ticks = config.total_ticks
    mid, log_returns, spread_bps, scenario = _simulate_symbol_path(
        symbol=symbol,
        spec=spec,
        total_ticks=total_ticks,
        rng=rng,
    )
    return_scale = max(float(np.std(log_returns)), 1e-12)
    spread = np.maximum(mid * spread_bps * 1e-4, spec.tick_size)
    level_step = np.maximum(spread * 0.75, spec.tick_size)

    bid_1 = mid - (spread * 0.5)
    ask_1 = mid + (spread * 0.5)
    bid_2 = bid_1 - level_step
    ask_2 = ask_1 + level_step
    bid_3 = bid_2 - level_step
    ask_3 = ask_2 + level_step

    depth_regime = 1.0 + 1.4 * np.abs(log_returns) / return_scale
    bid_1_sz = spec.base_depth * depth_regime * rng.lognormal(mean=0.0, sigma=0.28, size=total_ticks)
    ask_1_sz = spec.base_depth * depth_regime * rng.lognormal(mean=0.0, sigma=0.28, size=total_ticks)
    bid_2_sz = (
        spec.base_depth
        * 1.35
        * depth_regime
        * rng.lognormal(mean=0.05, sigma=0.30, size=total_ticks)
    )
    ask_2_sz = (
        spec.base_depth
        * 1.35
        * depth_regime
        * rng.lognormal(mean=0.05, sigma=0.30, size=total_ticks)
    )
    bid_3_sz = (
        spec.base_depth
        * 1.70
        * depth_regime
        * rng.lognormal(mean=0.10, sigma=0.33, size=total_ticks)
    )
    ask_3_sz = (
        spec.base_depth
        * 1.70
        * depth_regime
        * rng.lognormal(mean=0.10, sigma=0.33, size=total_ticks)
    )

    front_imbalance = (bid_1_sz - ask_1_sz) / np.maximum(bid_1_sz + ask_1_sz, 1.0)
    buy_prob = np.clip(
        0.50
        + 0.23 * np.tanh(log_returns / return_scale * 2.0)
        + 0.18 * front_imbalance,
        0.05,
        0.95,
    )
    is_buy = rng.random(total_ticks) < buy_prob
    trade_px = np.where(is_buy, ask_1, bid_1)
    trade_size = (
        spec.base_trade
        * depth_regime
        * rng.lognormal(mean=-0.05, sigma=0.42, size=total_ticks)
    )

    flat = np.empty((total_ticks, BOOK_COLS), dtype=np.float64)
    flat[:, COL_TS] = np.arange(total_ticks, dtype=np.float64) * config.tick_dt_s
    flat[:, COL_BID_1] = bid_1
    flat[:, COL_ASK_1] = ask_1
    flat[:, COL_BID_1_SZ] = bid_1_sz
    flat[:, COL_ASK_1_SZ] = ask_1_sz
    flat[:, COL_BID_2] = bid_2
    flat[:, COL_ASK_2] = ask_2
    flat[:, COL_BID_2_SZ] = bid_2_sz
    flat[:, COL_ASK_2_SZ] = ask_2_sz
    flat[:, COL_BID_3] = bid_3
    flat[:, COL_ASK_3] = ask_3
    flat[:, COL_BID_3_SZ] = bid_3_sz
    flat[:, COL_ASK_3_SZ] = ask_3_sz
    flat[:, COL_TRADE_PX] = trade_px
    flat[:, COL_TRADE_SZ] = trade_size

    bank = np.ascontiguousarray(flat.reshape((config.bank_frames, config.frame_ticks, BOOK_COLS)))
    return bank, scenario


def build_symbol_scenario(
    symbol: str,
    *,
    config: SimulationConfig,
) -> SymbolScenario:
    spec = SYMBOL_SPECS[symbol]
    rng = np.random.default_rng(_symbol_seed(symbol, config=config))
    _, _, _, scenario = _simulate_symbol_path(
        symbol=symbol,
        spec=spec,
        total_ticks=config.total_ticks,
        rng=rng,
    )
    return scenario


def build_scenarios(config: SimulationConfig) -> dict[str, SymbolScenario]:
    return {
        symbol: build_symbol_scenario(symbol, config=config)
        for symbol in SYMBOLS
    }


def describe_demo(
    *,
    config: SimulationConfig,
    scenarios: dict[str, SymbolScenario],
    baseline: BaselineMetrics,
) -> tuple[str, ...]:
    bullish = [symbol for symbol in SYMBOLS if scenarios[symbol].trend_sign > 0]
    bearish = [symbol for symbol in SYMBOLS if scenarios[symbol].trend_sign < 0]
    leader = max(scenarios.values(), key=lambda item: item.return_pct)
    widest = max(scenarios.values(), key=lambda item: item.avg_spread_bps)

    return (
        (
            f"{len(SYMBOLS)} symbols | {config.frame_ticks:,} ticks/frame | "
            f"{config.bank_frames} frames/symbol | {config.total_ticks * len(SYMBOLS):,} ticks/replay | "
            f"bank {config.total_bank_gb:.2f} GB."
        ),
        (
            f"Anchor regimes: bulls {', '.join(bullish)} | "
            f"bears {', '.join(bearish)}."
        ),
        (
            f"Strongest preset path: {leader.symbol} {leader.return_pct:+.2%} "
            f"with {leader.anchor_count} anchors; widest average spread: "
            f"{widest.symbol} {widest.avg_spread_bps:.2f} bps."
        ),
        (
            f"Serial benchmark: {baseline.ticks_per_second:,.0f} ticks/s | "
            f"{baseline.gbps:.2f} Gbit/s over {baseline.processed_ticks:,} benchmarked ticks."
        ),
    )


def evaluate_tick_frame(
    frame: np.ndarray,
    state: QuantState,
    *,
    annualization_factor: float,
    compute_traces: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, QuantState]:
    frame = np.asarray(frame, dtype=np.float64)
    if frame.ndim != 2 or frame.shape[1] != BOOK_COLS:
        raise ValueError(f"expected frame shape (n, {BOOK_COLS}), got {frame.shape}")

    timestamps = frame[:, COL_TS]
    if state.last_timestamp is not None and float(timestamps[0]) <= state.last_timestamp:
        state = reset_quant_state()

    bid_1 = frame[:, COL_BID_1]
    ask_1 = frame[:, COL_ASK_1]
    bid_1_sz = frame[:, COL_BID_1_SZ]
    ask_1_sz = frame[:, COL_ASK_1_SZ]
    trade_px = frame[:, COL_TRADE_PX]
    trade_sz = frame[:, COL_TRADE_SZ]

    mid = 0.5 * (bid_1 + ask_1)
    depth_bid = bid_1_sz + frame[:, COL_BID_2_SZ] + frame[:, COL_BID_3_SZ]
    depth_ask = ask_1_sz + frame[:, COL_ASK_2_SZ] + frame[:, COL_ASK_3_SZ]
    top_depth = np.maximum(bid_1_sz + ask_1_sz, 1e-12)
    total_depth = np.maximum(depth_bid + depth_ask, 1e-12)
    microprice = ((ask_1 * bid_1_sz) + (bid_1 * ask_1_sz)) / top_depth
    spread_bps = 10_000.0 * (ask_1 - bid_1) / mid
    imbalance = (depth_bid - depth_ask) / total_depth
    trade_sign = np.where(trade_px >= mid, 1.0, -1.0)
    signed_notional = trade_sign * trade_px * trade_sz
    micro_edge_bps = 10_000.0 * (microprice - mid) / mid

    if state.anchor_mid is None:
        state.anchor_mid = float(mid[0])
        state.peak_mid = float(mid[0])

    log_returns = np.empty_like(mid)
    if state.prev_mid is None:
        log_returns[0] = 0.0
        if mid.size > 1:
            log_returns[1:] = np.log(mid[1:] / mid[:-1])
    else:
        log_returns[0] = math.log(mid[0] / state.prev_mid)
        if mid.size > 1:
            log_returns[1:] = np.log(mid[1:] / mid[:-1])

    state.prev_mid = float(mid[-1])
    state.last_timestamp = float(timestamps[-1])
    state.peak_mid = max(state.peak_mid, float(np.max(mid)))
    state.cum_notional += float(np.dot(trade_px, trade_sz))
    state.cum_volume += float(np.sum(trade_sz))
    state.mid_window.extend(mid)
    state.return_window.extend(log_returns)
    state.flow_window.extend(signed_notional)

    mid_trace = None
    ema_trace = None
    if compute_traces:
        mid_trace = (mid / state.anchor_mid) - 1.0
        ema_trace = np.empty(frame.shape[0], dtype=np.float64)
        ema_mid = state.ema_mid
        for index, mid_value in enumerate(mid):
            if ema_mid is None:
                ema_mid = float(mid_value)
            else:
                ema_mid = ema_mid + EMA_ALPHA * (float(mid_value) - ema_mid)
            ema_trace[index] = (ema_mid / state.anchor_mid) - 1.0
        state.ema_mid = ema_mid

    latest_mid = float(mid[-1])
    session_return = 0.0 if state.anchor_mid is None else (latest_mid / state.anchor_mid) - 1.0
    momentum = 0.0
    if len(state.mid_window) >= MOMENTUM_WINDOW:
        momentum = (latest_mid / state.mid_window[0]) - 1.0

    realized_vol = 0.0
    if len(state.return_window) >= VOL_WINDOW:
        realized_vol = float(np.std(state.return_window) * annualization_factor)

    drawdown = 0.0 if state.peak_mid <= 0.0 else (latest_mid / state.peak_mid) - 1.0
    vwap = state.cum_notional / max(state.cum_volume, 1e-12)
    vwap_bps = 0.0 if vwap <= 0.0 else 10_000.0 * ((latest_mid / vwap) - 1.0)

    flow_z = 0.0
    if len(state.flow_window) >= 32:
        flow_mean = float(np.mean(state.flow_window))
        flow_std = float(np.std(state.flow_window))
        if flow_std > 0.0:
            flow_z = (float(signed_notional[-1]) - flow_mean) / flow_std

    metrics = np.array(
        (
            session_return,
            momentum,
            realized_vol,
            drawdown,
            float(imbalance[-1]),
            flow_z,
            float(micro_edge_bps[-1]),
            vwap_bps,
            float(spread_bps[-1]),
        ),
        dtype=np.float64,
    )
    return mid_trace, ema_trace, metrics, state


def downsample_frame(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    rows: int,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if timestamps.shape != values.shape:
        raise ValueError("timestamps and values must have identical shapes")
    if rows <= 0:
        raise ValueError("rows must be greater than 0")

    if timestamps.size == 0:
        return np.empty((0, 2), dtype=dtype)

    take = min(rows, timestamps.size)
    if take == timestamps.size:
        sampled_ts = timestamps
        sampled_values = values
    else:
        indices = np.linspace(0, timestamps.size - 1, take, dtype=np.int64)
        sampled_ts = timestamps[indices]
        sampled_values = values[indices]

    out = np.empty((take, 2), dtype=dtype)
    out[:, 0] = sampled_ts
    out[:, 1] = sampled_values
    return out


def baseline_benchmark(
    config: SimulationConfig,
    *,
    display_rows: int = DEFAULT_DISPLAY_ROWS,
    display_enabled: bool = True,
) -> BaselineMetrics:
    benchmark_frames = min(config.bank_frames, DEFAULT_BASELINE_SAMPLE_FRAMES)
    processed_ticks = 0
    processed_bytes = 0
    started = perf_counter()

    for symbol in SYMBOLS:
        bank, _ = build_symbol_bank(symbol, config=config)
        state = reset_quant_state()
        for _ in range(config.baseline_loops):
            for frame in bank[:benchmark_frames]:
                mid_trace, ema_trace, _metrics, state = evaluate_tick_frame(
                    frame,
                    state,
                    annualization_factor=config.annualization_factor,
                    compute_traces=display_enabled,
                )
                if display_enabled:
                    assert mid_trace is not None and ema_trace is not None
                    downsample_frame(frame[:, COL_TS], mid_trace, rows=display_rows)
                    downsample_frame(frame[:, COL_TS], ema_trace, rows=display_rows)
                processed_ticks += frame.shape[0]
                processed_bytes += frame.nbytes

    elapsed = max(perf_counter() - started, 1e-9)
    return BaselineMetrics(
        ticks_per_second=processed_ticks / elapsed,
        gbps=(processed_bytes * 8.0) / elapsed / 1_000_000_000.0,
        processed_ticks=processed_ticks,
        processed_gb=processed_bytes / 1_000_000_000.0,
        loops=config.baseline_loops,
    )


def _symbol_seed(symbol: str, *, config: SimulationConfig) -> int:
    return config.seed + (1_009 * (_SYMBOL_INDEX[symbol] + 1))


def _simulate_symbol_path(
    *,
    symbol: str,
    spec: SymbolSpec,
    total_ticks: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SymbolScenario]:
    trend_sign = -1 if rng.random() < 0.5 else 1
    anchor_count = int(rng.integers(5, 10))
    anchor_positions = _anchor_positions(total_ticks, anchor_count, rng)
    anchor_returns = _anchor_returns(anchor_count, trend_sign, rng)
    mid = _mid_path(
        spec=spec,
        total_ticks=total_ticks,
        anchor_positions=anchor_positions,
        anchor_returns=anchor_returns,
        rng=rng,
    )
    log_mid = np.log(mid)
    log_returns = np.diff(log_mid, prepend=log_mid[0])
    return_scale = max(float(np.std(log_returns)), 1e-12)
    spread_bps = spec.base_spread_bps * (
        1.0 + 2.8 * np.abs(log_returns) / return_scale + 0.18 * rng.random(total_ticks)
    )
    scenario = SymbolScenario(
        symbol=symbol,
        trend_sign=trend_sign,
        anchor_count=anchor_count,
        start_price=float(mid[0]),
        end_price=float(mid[-1]),
        low_price=float(mid.min()),
        high_price=float(mid.max()),
        avg_spread_bps=float(np.mean(spread_bps)),
    )
    return mid, log_returns, spread_bps, scenario


def _anchor_positions(total_ticks: int, anchor_count: int, rng: np.random.Generator) -> np.ndarray:
    base = np.linspace(0, total_ticks - 1, anchor_count, dtype=np.float64)
    jitter_width = max(1, total_ticks // (anchor_count * 8))
    jitter = rng.integers(-jitter_width, jitter_width + 1, size=anchor_count)
    positions = np.rint(base + jitter).astype(np.int64)
    positions[0] = 0
    positions[-1] = total_ticks - 1

    for index in range(1, anchor_count - 1):
        minimum = positions[index - 1] + 1
        maximum = (total_ticks - 1) - (anchor_count - index - 1)
        positions[index] = min(max(positions[index], minimum), maximum)

    return positions


def _anchor_returns(
    anchor_count: int,
    trend_sign: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_drift = trend_sign * rng.uniform(0.015, 0.085)
    progress = np.linspace(0.0, 1.0, anchor_count, dtype=np.float64)
    sway = rng.normal(0.0, 0.0045, size=anchor_count).cumsum()
    sway -= progress * sway[-1]
    anchor_returns = (total_drift * progress) + sway
    anchor_returns[0] = 0.0
    anchor_returns[-1] = total_drift
    return anchor_returns


def _mid_path(
    *,
    spec: SymbolSpec,
    total_ticks: int,
    anchor_positions: np.ndarray,
    anchor_returns: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    start_log = math.log(spec.base_price * math.exp(rng.normal(0.0, 0.04)))
    anchor_logs = start_log + anchor_returns
    path = np.empty(total_ticks, dtype=np.float64)

    for segment_index, (start, stop) in enumerate(zip(anchor_positions[:-1], anchor_positions[1:])):
        steps = stop - start
        t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
        bridge_noise = np.empty(steps + 1, dtype=np.float64)
        bridge_noise[0] = 0.0

        if steps:
            segment_scale = 1.0 + 0.35 * abs((segment_index / max(anchor_positions.size - 2, 1)) - 0.5) * 2.0
            draws = rng.normal(0.0, spec.vol_per_sqrt_tick * segment_scale, size=steps)
            bridge_noise[1:] = np.cumsum(draws)
        bridge = bridge_noise - (t * bridge_noise[-1])
        segment = anchor_logs[segment_index] + ((anchor_logs[segment_index + 1] - anchor_logs[segment_index]) * t)
        segment += bridge

        if segment_index == 0:
            path[start : stop + 1] = segment
        else:
            path[start + 1 : stop + 1] = segment[1:]

    return np.exp(path)
