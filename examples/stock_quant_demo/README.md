# Stock Quant Demo

A simulated L3 market microstructure replay desk built on Pythusa. Eight
parallel generators stream synthetic order-book data through shared-memory ring
buffers into per-symbol quant analytics workers, with an optional ImGui
dashboard for live monitoring and a headless mode for throughput benchmarking.

## What it demonstrates

This demo uses Pythusa to build a realistic market data pipeline that would
normally require a C++ or Java infrastructure:

- **One generator process per symbol** writes 3-level order-book snapshots and
  trade prints into shared-memory streams. No pickling, no serialization.
- **One analytics process per symbol** reads raw book data through zero-copy
  memoryviews and computes a full suite of microstructure metrics per frame.
- **End-to-end latency tracking** stamps each frame at publish time and measures
  the time until the analyzer finishes quant math, reporting mean, p50, p95, and
  p99 frame latency.
- **Speedup against a serial baseline** runs the same quant math single-threaded
  at startup and reports the live parallel speedup factor on the dashboard.
- **Runtime profiles** let you tune the pipeline for latency, throughput, or a
  balanced default by adjusting frame size, ring depth, and report cadence.

## Universe

| Symbol | Sector | Description |
| --- | --- | --- |
| AAPL | Tech | Large-cap consumer electronics |
| MSFT | Tech | Large-cap enterprise software |
| NVDA | Semis | Large-cap GPUs and accelerators |
| AMZN | Tech | Large-cap e-commerce and cloud |
| META | Tech | Large-cap social media |
| GOOG | Tech | Large-cap search and cloud |
| TSLA | Auto | Large-cap electric vehicles |
| JPM | Finance | Large-cap investment bank |

## Market simulation

The simulation builds a precomputed replay bank (~4 GB by default across all 8
symbols) using a multi-stage process:

1. **Trend assignment.** Each symbol is randomly assigned an uptrend or
   downtrend for the session.
2. **Anchor placement.** 5--9 anchor points are distributed across the replay
   with jitter. Each anchor has a target cumulative return consistent with the
   assigned trend.
3. **Brownian bridge interpolation.** The price path between anchors is filled
   with a Brownian bridge in log-space, producing paths that hit the targets
   while exhibiting realistic stochastic behavior between them.
4. **Order book construction.** A 3-level book is built around the midprice at
   each tick: spreads widen during volatile periods, depth sizes follow
   lognormal distributions that thicken at deeper levels, and trade aggression
   correlates with momentum and book imbalance.
5. **Trade generation.** Each tick has a trade print at the inside bid or ask.
   Buy probability depends on recent returns and front-of-book imbalance.

Each tick has 15 columns: timestamp, 3 bid/ask price-size pairs, and a
trade price-size pair.

## Quant metrics

Each per-symbol analytics worker computes live microstructure metrics:

| Metric | What it measures |
| --- | --- |
| Session return | Cumulative return from the session anchor price |
| Momentum | Short-horizon return over a 256-tick window |
| Realized volatility | Annualized standard deviation of log returns (512-tick window) |
| Drawdown | Decline from the session peak midprice |
| Depth imbalance | (total bid depth - total ask depth) / total depth across 3 levels |
| Signed-flow z-score | How extreme the latest signed trade notional is vs. the rolling window |
| Microprice edge | Size-weighted midprice deviation from the simple midprice, in basis points |
| VWAP deviation | Current midprice vs. session VWAP, in basis points |
| Spread | Inside spread in basis points |

## Performance telemetry

The pipeline reports live performance per symbol and in aggregate:

- **Ticks/s** -- aggregate tick processing rate across all 8 workers
- **Gbit/s** -- raw payload throughput (frame bytes * 8 / elapsed)
- **Frame latency** -- end-to-end from generator publish to analyzer completion
  (mean, p50, p95, p99)
- **Speedup** -- live parallel throughput divided by the serial baseline measured
  at startup
- **Total processed** -- cumulative ticks and gigabytes since start

## Runtime profiles

| Profile | Ticks/frame | Raw ring depth | Idle sleep | Best for |
| --- | --- | --- | --- | --- |
| `latency` | 256 | 4 frames | 10 us | Minimizing frame delay |
| `balanced` | 2048 | 16 frames | 100 us | General interactive use |
| `throughput` | 8192 | 32 frames | 200 us | Maximum aggregate payload |

## Run

### GUI mode

```bash
python examples/stock_quant_demo/main.py
```

Opens the ImGui dashboard at 1820x1180. The left sidebar shows replay design
notes, live cross-section commentary, and per-symbol scenario details. The right
panel shows 8 signal cards (2x4 grid) with live midprice and EMA traces, quant
metrics, and per-symbol throughput and latency.

### Headless throughput benchmark

```bash
python examples/stock_quant_demo/main.py --headless --mode throughput --bank-gb 1 --duration 20 --report-interval 1
```

### Headless latency benchmark

```bash
python examples/stock_quant_demo/main.py --headless --mode latency --bank-gb 1 --duration 20 --report-interval 1
```

Prints a live summary line per interval with aggregate Gbit/s, ticks/s, latency
percentiles, speedup, and cross-section highlights (leader, laggard, flow
stress, worst latency lane). `--bank-gb` controls the precomputed replay bank
size; smaller values (e.g. 1) reduce startup time while still saturating the
pipeline.

### Custom configuration

```bash
python examples/stock_quant_demo/main.py \
  --mode latency \
  --seed 11 \
  --bank-gb 2 \
  --tick-ms 10 \
  --baseline-loops 4
```

## Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--mode` | `balanced` | Runtime profile: `latency`, `balanced`, or `throughput` |
| `--seed` | 7 | Simulation RNG seed for reproducibility |
| `--frame-ticks` | (from mode) | Override ticks per shared-memory frame |
| `--bank-frames` | (computed) | Override frames per symbol replay bank |
| `--bank-gb` | 4.0 | Target total precomputed replay bank size in GB |
| `--stream-frames` | (from mode) | Override shared-memory ring depth in frames |
| `--tick-ms` | 10.0 | Tick spacing in milliseconds inside the simulated replay |
| `--baseline-loops` | 4 | Serial benchmark passes for speedup calculation |
| `--headless` | off | Disable ImGui, print stats to stdout |
| `--duration` | unlimited | Stop after N seconds in headless mode |
| `--report-interval` | 1.0 | Seconds between headless console reports |
| `--monitor` | off | Enable Pythusa runtime monitor and ring-pressure diagnostics |
| `--monitor-interval` | 0.10 | Seconds between monitor samples |
| `--display-history` | 96 | GUI history window depth in frames per symbol |

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Pipeline definition, task functions, CLI, and ImGui dashboard |
| `data.py` | Market simulation, order-book generation, quant math, and baseline benchmark |
| `graphing.py` | ImGui draw-list graphing helper for price traces |

## Pipeline topology

```
generator(symbol) --> raw stream -----> analyzer(symbol) --> metrics stream --> UI / console
                  \-> stamp stream -/                    \-> perf stream ---/
                                                          \-> display stream -/  (GUI only)
```

Each symbol runs as an independent generator-analyzer pair. The UI or console
task fans in all metric and performance streams for aggregate reporting. Stop
events provide graceful shutdown from the dashboard or after a headless duration.
