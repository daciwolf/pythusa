from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_data_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "stock_quant_demo" / "data.py"
    spec = importlib.util.spec_from_file_location("stock_quant_demo_data", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_main_module():
    root = Path(__file__).resolve().parents[1]
    stock_demo_root = root / "examples" / "stock_quant_demo"
    pythusa_src = root / "src"
    for path in (stock_demo_root, pythusa_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    module_path = stock_demo_root / "main.py"
    spec = importlib.util.spec_from_file_location("stock_quant_demo_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


data = _load_data_module()
main = _load_main_module()


def test_default_bank_target_is_about_four_gb() -> None:
    config = data.SimulationConfig()

    assert 3.99 <= config.total_bank_gb <= 4.01


def test_bank_frames_for_target_gb_scales_with_frame_size() -> None:
    small = data.bank_frames_for_target_gb(frame_ticks=256, target_bank_gb=4.0)
    large = data.bank_frames_for_target_gb(frame_ticks=8192, target_bank_gb=4.0)

    assert small > large
    assert small > 0
    assert large > 0


def test_build_symbol_bank_is_deterministic_and_well_formed() -> None:
    config = data.SimulationConfig(seed=7, frame_ticks=128, bank_frames=6, tick_ms=25.0, baseline_loops=1)

    bank_a, scenario_a = data.build_symbol_bank("AAPL", config=config)
    bank_b, scenario_b = data.build_symbol_bank("AAPL", config=config)

    np.testing.assert_allclose(bank_a, bank_b)
    assert scenario_a == scenario_b
    assert bank_a.shape == (config.bank_frames, config.frame_ticks, data.BOOK_COLS)

    flat = bank_a.reshape((-1, data.BOOK_COLS))
    np.testing.assert_allclose(np.diff(flat[:, data.COL_TS]), np.full(flat.shape[0] - 1, config.tick_dt_s))
    assert np.all(flat[:, data.COL_BID_3] < flat[:, data.COL_BID_2])
    assert np.all(flat[:, data.COL_BID_2] < flat[:, data.COL_BID_1])
    assert np.all(flat[:, data.COL_BID_1] < flat[:, data.COL_ASK_1])
    assert np.all(flat[:, data.COL_ASK_1] < flat[:, data.COL_ASK_2])
    assert np.all(flat[:, data.COL_ASK_2] < flat[:, data.COL_ASK_3])
    assert np.all(flat[:, data.COL_BID_1_SZ : data.COL_TRADE_SZ + 1] > 0.0)
    assert np.all(
        np.logical_or(
            np.isclose(flat[:, data.COL_TRADE_PX], flat[:, data.COL_BID_1]),
            np.isclose(flat[:, data.COL_TRADE_PX], flat[:, data.COL_ASK_1]),
        )
    )
    assert np.sign(scenario_a.return_pct) == scenario_a.trend_sign


def test_evaluate_tick_frame_produces_finite_quant_metrics() -> None:
    config = data.SimulationConfig(seed=11, frame_ticks=256, bank_frames=4, tick_ms=10.0, baseline_loops=1)
    bank, _ = data.build_symbol_bank("NVDA", config=config)
    state = data.reset_quant_state()

    metrics = None
    for frame in bank:
        mid_trace, ema_trace, metrics, state = data.evaluate_tick_frame(
            frame,
            state,
            annualization_factor=config.annualization_factor,
        )
        assert mid_trace.shape == (config.frame_ticks,)
        assert ema_trace.shape == (config.frame_ticks,)

    assert metrics is not None
    assert metrics.shape == (data.METRIC_COLS,)
    assert np.all(np.isfinite(metrics))
    assert metrics[data.METRIC_REALIZED_VOL] >= 0.0
    assert metrics[data.METRIC_DRAWDOWN] <= 1e-12
    assert abs(metrics[data.METRIC_IMBALANCE]) <= 1.0
    assert metrics[data.METRIC_SPREAD_BPS] > 0.0
    assert state.cum_volume > 0.0
    assert state.anchor_mid is not None


def test_evaluate_tick_frame_can_skip_display_traces() -> None:
    config = data.SimulationConfig(seed=13, frame_ticks=128, bank_frames=2, tick_ms=10.0, baseline_loops=1)
    bank, _ = data.build_symbol_bank("MSFT", config=config)
    state = data.reset_quant_state()

    mid_trace, ema_trace, metrics, state = data.evaluate_tick_frame(
        bank[0],
        state,
        annualization_factor=config.annualization_factor,
        compute_traces=False,
    )

    assert mid_trace is None
    assert ema_trace is None
    assert metrics.shape == (data.METRIC_COLS,)
    assert np.all(np.isfinite(metrics))
    assert state.cum_volume > 0.0


def test_write_display_view_downsamples_triplet_frame() -> None:
    timestamps = np.arange(8, dtype=np.float64)
    mid_values = np.linspace(-0.01, 0.02, 8, dtype=np.float64)
    ema_values = np.linspace(-0.008, 0.015, 8, dtype=np.float64)
    buffer = bytearray(4 * main.DISPLAY_FRAME_COLS * np.dtype(main.DISPLAY_DTYPE).itemsize)

    main._write_display_view(memoryview(buffer), timestamps, mid_values, ema_values, rows=4)

    out = np.frombuffer(buffer, dtype=main.DISPLAY_DTYPE).reshape((4, main.DISPLAY_FRAME_COLS))
    expected_indices = np.linspace(0, timestamps.size - 1, 4, dtype=np.int64)
    np.testing.assert_allclose(out[:, main.DISPLAY_TS_COL], timestamps[expected_indices])
    np.testing.assert_allclose(out[:, main.DISPLAY_MID_COL], mid_values[expected_indices])
    np.testing.assert_allclose(out[:, main.DISPLAY_EMA_COL], ema_values[expected_indices])


def test_display_series_state_concatenates_and_resets_on_replay_boundary() -> None:
    state = main.DisplaySeriesState(max_frames=3)

    frame_a = np.array(
        (
            (0.0, 0.00, 0.00),
            (1.0, 0.01, 0.005),
        ),
        dtype=main.DISPLAY_DTYPE,
    )
    frame_b = np.array(
        (
            (2.0, 0.02, 0.010),
            (3.0, 0.03, 0.015),
        ),
        dtype=main.DISPLAY_DTYPE,
    )
    frame_reset = np.array(
        (
            (0.0, -0.01, -0.008),
            (1.0, -0.02, -0.010),
        ),
        dtype=main.DISPLAY_DTYPE,
    )

    assert state.append_batch([frame_a, frame_b])
    assert state.series.shape == (4, main.DISPLAY_FRAME_COLS)
    np.testing.assert_allclose(state.series[:2], frame_a)
    np.testing.assert_allclose(state.series[2:], frame_b)
    assert state.y_limits is not None

    assert state.append_batch([frame_reset])
    assert state.series.shape == (2, main.DISPLAY_FRAME_COLS)
    np.testing.assert_allclose(state.series, frame_reset)
    np.testing.assert_allclose(state.series[:, main.DISPLAY_TS_COL], frame_reset[:, main.DISPLAY_TS_COL])


def test_baseline_benchmark_and_demo_description_expose_throughput() -> None:
    config = data.SimulationConfig(seed=5, frame_ticks=96, bank_frames=3, tick_ms=20.0, baseline_loops=1)
    scenarios = data.build_scenarios(config)
    baseline = data.baseline_benchmark(config, display_rows=64, display_enabled=False)
    lines = data.describe_demo(config=config, scenarios=scenarios, baseline=baseline)

    assert set(scenarios) == set(data.SYMBOLS)
    assert baseline.processed_ticks > 0
    assert baseline.processed_gb > 0.0
    assert baseline.ticks_per_second > 0.0
    assert baseline.gbps > 0.0
    assert len(lines) == 4
    assert any("ticks/replay" in line for line in lines)
    assert any("Serial benchmark" in line for line in lines)
