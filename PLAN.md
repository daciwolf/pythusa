# PYTHUSA Plan

## Mission

PYTHUSA makes it easy to build high-throughput DSP pipelines in Python without giving up shared-memory performance.

You write the processing code; PYTHUSA handles zero-copy transport, process orchestration, and the throughput/latency behavior around it.

This pitch should drive the roadmap. The goal is not to broaden the package into a general-purpose DSP framework; the goal is to make this narrow promise production-ready for scientific research and sensor-style workloads with the fewest necessary changes.

## Current State

The core is already promising:

- zero-copy shared-memory rings are working in the hot path
- process orchestration is functional and benchmarkable
- the FFT pipeline is already reaching multi-GB/s throughput in the right configuration
- the DSP benchmark suite reports throughput, latency, and memory
- benchmarks now live in `benchmarks/`, separate from examples

What is still missing is product quality: stable APIs, correctness guarantees, documentation, built-in observability, and a coherent release story. The path forward should stay conservative: preserve the working runtime, reduce surface-area complexity, and harden the package around the use case it already serves well.

## Guiding Constraint

We will optimize for minimal change to production-ready scientific use.

That means:

- keep the existing zero-copy shared-memory core unless there is a clear correctness or portability problem
- avoid adding broad new abstractions unless they make common DSP pipelines materially easier to build
- prefer documentation, tests, metrics, and packaging work over architectural churn
- treat optional accelerators and backend experiments as user-level choices unless they clearly improve the core promise
- judge every roadmap item against the package pitch above

## Research-Ready Target

For this project, "research-ready" does not mean feature-complete or enterprise-hardened. It means a scientist or engineer can install PYTHUSA, build a pipeline, trust the results, reproduce the reported performance, and understand the runtime behavior without reading internal code.

Research-ready should include:

- a small public API for the common pipeline cases
- correctness coverage for the marketed DSP kernels and core transport behavior
- reproducible benchmark commands with throughput, latency, and memory reporting
- runtime metrics and profiles that are available outside benchmark scripts
- explicit supported-platform documentation and verification notes
- examples and docs that match the actual intended usage

Research-ready does not require:

- hard real-time guarantees
- a full DSP algorithm catalog
- GPU support
- a plugin ecosystem
- replacing the existing runtime core

## What "Stellar" Means

PYTHUSA should feel stellar when it meets all of these conditions:

1. A new user can build a working DSP pipeline in minutes without learning internal ring mechanics.
2. Core DSP operations have correctness coverage against known-good NumPy outputs.
3. Throughput and latency behavior are measurable, documented, and reproducible.
4. The library works on macOS, Linux, and Windows with consistent semantics.
5. Users can choose a simple policy: maximize throughput, minimize latency, or balance both.
6. Runtime metrics are first-class, not benchmark-only.
7. The public API is small, stable, and documented.

## Workstreams

### 1. Public API and UX

Goal: make the fast path easy.

We will design a minimal public API around a few concepts:

- `Pipeline`: high-level composition of stages
- `Stream` or `Channel`: named data flow between stages
- `Processor`: user-defined DSP stage
- `Profile`: `throughput_max`, `latency_min`, `balanced`

Deliverables:

- a first draft public pipeline builder
- typed stage interfaces for array-in/array-out processing
- defaults that hide ring sizing and backlog policy unless the user opts in
- migration-safe separation between public and internal APIs

Constraint:

- wrap the current runtime rather than replacing it

Exit criteria:

- at least three examples using only the public API
- no example in the README requires direct access to internal modules

### 2. Correctness and Reliability

Goal: make results trustworthy.

We will add focused correctness tests for:

- passthrough
- gain
- windowing
- FIR filters
- FFT and power spectrum
- STFT framing assumptions
- dtype handling (`float32`, `float64`)
- shape and channel behavior

We will also harden runtime behavior around:

- worker shutdown
- ring cleanup
- dropped-frame policies
- backlog handling
- startup sequencing

Exit criteria:

- golden-output tests for every benchmarked DSP kernel
- regression tests for zero-copy paths and cleanup behavior
- no benchmark-only code path without at least one targeted test

### 3. Latency, Throughput, and Runtime Policies

Goal: make performance intentional, not accidental.

We now have benchmark modes. The next step is to expose similar policies in the runtime itself:

- `throughput_max`
- `latency_min`
- `balanced`

These policies should affect:

- ring depth
- maximum in-flight batches
- drop-vs-block behavior
- whether stale frames are discarded under pressure

We should also make room for engine- or sensor-style workloads:

- small windows for low-latency detection
- large windows for spectral accuracy and throughput
- explicit reporting of window fill time versus processing time

Exit criteria:

- public runtime profiles, not just benchmark presets
- clear docs on when to use each profile
- benchmark results that show the tradeoff directly

### 4. Built-In Observability

Goal: users should not need ad hoc print statements to understand the system.

We will promote metrics into the package:

- throughput
- processing latency
- queue/ring pressure
- dropped frames
- worker CPU
- worker RSS
- backlog depth

This should support:

- programmatic access
- periodic logging
- optional benchmark export formats such as JSON or CSV

Exit criteria:

- one public metrics API
- optional structured output for benchmarks
- docs that explain how to monitor a live pipeline

### 5. Benchmarking and Competitive Positioning

Goal: keep the performance story real and reproducible.

We already have a good start. Next we will expand it into a benchmark matrix:

- generic DSP suite
- FFT comparison suite
- rocket-style sensor benchmark
- latency-focused benchmark profiles
- throughput-focused benchmark profiles

Comparisons should include:

- NumPy-only local execution
- PYTHUSA process pipelines
- joblib threads where relevant
- optionally multiprocessing or SciPy reference cases

Benchmarks should report:

- throughput
- processing latency
- total detection latency where sample rate matters
- memory reservation and RSS
- configuration details

Exit criteria:

- reproducible benchmark scripts with documented commands
- benchmark outputs suitable for README snippets and release notes
- a clear story for when PYTHUSA wins and why

### 6. Documentation and Examples

Goal: make the package teach itself.

We need three levels of documentation:

1. Quickstart
- simple producer/consumer
- public API first

2. Concepts
- rings
- stages
- profiles
- zero-copy behavior
- latency vs throughput tradeoffs

3. Recipes
- FFT monitoring
- FIR filtering
- streaming sensor analysis
- multi-stage DSP chains

Examples should stay simple. Benchmarks and tuning tools belong in `benchmarks/`.

Exit criteria:

- README focuses on what PYTHUSA is and how to start
- dedicated docs or markdown guides for concepts and recipes
- examples use the stable public API only

### 7. Packaging, CI, and Release

Goal: make the package installable and trustworthy.

We need:

- CI on macOS, Linux, and Windows
- benchmark smoke runs where feasible
- versioning policy
- changelog
- release checklist
- metadata cleanup and PyPI readiness

Exit criteria:

- green cross-platform CI for tests
- documented supported platforms and Python version

## Immediate Focus

The next work should stay tightly scoped:

1. define the first small public pipeline API on top of the current runtime
2. add correctness tests for the benchmarked DSP kernels
3. expose runtime profiles and metrics in a user-facing way
4. tighten packaging, docs, and CI until the package is credible for scientific research use

If a proposed change does not strengthen the package pitch or reduce the gap to production-ready scientific use, it should be deprioritized.

## Research-Ready Timeline

If we stay disciplined and avoid architectural churn, the realistic target for a research-ready baseline is about 2 to 4 weeks of focused work. A stronger release candidate with cleaner docs and broader validation is about 4 to 6 weeks.

### Milestone A: Benchmark Credibility

Target: 2 to 4 days

Status:

- complete on the current branch

Purpose:

- make the performance story reproducible and defensible

Tasks:

- finish the rocket-style benchmark and make total detection latency explicit
- add structured benchmark output for machine-readable comparison
- standardize benchmark commands, presets, and output naming
- capture a small canonical result set for README and release notes

Exit criteria:

- one-command benchmark runs for generic DSP, FFT comparison, and rocket-style workloads
- benchmark outputs are stable enough to compare between branches
- docs explain what each benchmark is actually measuring

### Milestone B: Correctness Coverage

Target: 4 to 6 days

Status:

- complete on the current branch
- benchmarked DSP kernels, benchmark helper paths, and benchmark-adjacent runtime lifecycle assumptions now have targeted correctness coverage

Purpose:

- make the marketed kernels and zero-copy transport trustworthy

Tasks:

- add golden-output tests for passthrough, gain, window, FIR, FFT, power spectrum, and STFT
- add dtype and shape coverage for representative `float32` and `float64` cases
- add tests for startup sequencing, zero-copy contracts, and cleanup behavior
- ensure benchmark-only kernel paths are covered by targeted tests

Exit criteria:

- every benchmarked DSP kernel has at least one direct correctness test
- the ring fast path and cleanup behavior have regression coverage
- the test suite is strong enough to refactor the public API without fear

### Milestone C: Public API v0

Target: 4 to 7 days

Status:

- planning started on the current branch

Purpose:

- make the package usable without teaching users the ring internals first
- make the common research and sensor pipeline path obvious while keeping the current runtime intact

Design principles:

- keep the API small, explicit, and composable
- wrap the existing runtime instead of replacing it
- make data contracts visible: shape, dtype, and optional sample rate should be first-class
- prefer explicit policies over hidden heuristics for backlog, dropping, and latency behavior
- preserve a clear path from high-level API down to the low-level runtime for debugging and power use
- avoid building a mini framework or graph DSL when a few plain Python objects will do

User focus:

- research users who want to stand up a streaming DSP pipeline quickly and reproduce its behavior
- rocket and sensor users who care about frame size, sample rate, detection latency, and operational visibility
- power users who still want direct access to `Manager`, `RingSpec`, and raw rings when the high-level API is too limiting

Non-goals for v0:

- no visual graph builder
- no large catalog of built-in DSP algorithms
- no hidden auto-parallelization or speculative stage fusion
- no attempt to abstract away the fact that this is a framed streaming pipeline system

Proposed v0 public concepts:

- `Pipeline`: owns orchestration, lifecycle, and metrics access
- `StreamSpec`: declares a stream's shape, dtype, and optional `sample_rate_hz`
- `Profile`: `throughput_max`, `latency_min`, `balanced`
- `SourceStage`: produces frames into a stream
- `ProcessStage`: transforms one stream into another
- `SinkStage`: consumes frames or derived results

Public API constraints:

- no decorators are required to define a stage
- no hidden graph rewriting or automatic stage fusion
- no hidden dtype or shape inference across stage boundaries
- no mandatory callback object model; plain callables remain the default extension point
- no loss of access to `Manager`, `RingSpec`, or task-level control for advanced users

Why this shape fits the Linux mentality:

- each stage does one job
- stream contracts are explicit rather than inferred
- stages compose through well-defined byte and array boundaries
- users can reason locally about one stage without understanding the whole runtime
- advanced users can still drop down a layer without fighting the abstraction

Research and rocket-oriented requirements:

- frame shape and dtype must be explicit
- sample rate metadata must be attachable to streams even if not every stage uses it
- latency-sensitive users must be able to choose behavior intentionally rather than inherit hidden defaults
- the API must support the common research loop: acquire -> process -> detect -> log
- the API must not hide where buffering happens
- configuration must be serializable and easy to record in experiment notes or test-stand logs
- stage boundaries must stay obvious enough that a user can reason about total detection latency

First-cut API shape:

```python
samples = pythusa.StreamSpec(
    name="chamber_pressure",
    shape=(2048, 4),
    dtype=np.float32,
    sample_rate_hz=250_000,
)

pipeline = pythusa.Pipeline(name="engine-monitor", profile="latency_min")
pipeline.add_source("acquire", output=samples, fn=acquire_frame)
pipeline.add_process("spectrum", input=samples, output="spectrum", fn=compute_spectrum)
pipeline.add_sink("detect", input="spectrum", fn=detect_instability)
pipeline.run()
```

The target here is not method-chaining flair. It is explicit construction with a small number of nouns and no hidden runtime model.

Tasks:

- draft `Pipeline`, `Stage`, and `Profile` around the current runtime
- design a stage contract for array-in/array-out processing
- support at least the common cases: source -> processor -> sink and multi-stage chains
- rebuild the README quickstart and a few examples on the new API

Detailed implementation sequence:

1. Define the stream contract.
- Introduce `StreamSpec` with `name`, `shape`, `dtype`, and optional `sample_rate_hz`.
- Keep it as pure data with no live runtime behavior.
- Validate only what the runtime truly needs: fixed frame shape, concrete dtype, optional sample rate, optional description text.
- Do not add channel units, labels, or rich metadata objects in v0 unless a concrete use case forces it.

2. Define the stage contract.
- Start with single-input, single-output stages.
- Make stage functions ordinary Python callables that receive arrays and return arrays or write into provided outputs.
- Keep stage roles explicit: source, process, sink.
- Require stage names to be explicit and stable so metrics and logs can refer to them.
- Keep stage objects as thin data wrappers over the user callable and stream references.

3. Build the pipeline wrapper.
- `Pipeline` should compile down to the current `Manager`, `TaskSpec`, and `RingSpec`.
- Keep ring sizing and reader counts internal by default, but allow explicit override for advanced users.
- Limit v0 to linear and branch-free pipelines unless a real use case requires fan-out.
- Make the compile step inspectable so a power user can see the generated rings and tasks.

4. Expose operational policy at construction time.
- Let the user choose a `Profile` and optional explicit overrides.
- Keep the default path simple while preserving deterministic behavior.
- Keep profile names identical to the benchmark story so docs and runtime do not drift.

5. Add metrics access at the pipeline level.
- Users should be able to ask the pipeline for worker and stream metrics without touching manager internals.
- Do not solve every observability problem in Milestone C; expose the path cleanly and keep the rest for Milestone D.
- Stage names and stream names should appear in the metric keys.

6. Rework the first examples and quickstart.
- Replace the low-level README example with one public API example.
- Keep one low-level example for power users, but move it out of the first impression path.

Implementation slices:

Slice C1: stream and profile data objects
- add `StreamSpec`
- add `Profile`
- add focused tests for validation and repr behavior

Slice C2: stage descriptors
- add `SourceStage`, `ProcessStage`, `SinkStage`
- keep them as pure data plus callable validation
- add tests that stage wiring errors fail early and clearly

Slice C3: pipeline compilation
- add `Pipeline`
- compile to `Manager`, `RingSpec`, and `TaskSpec`
- support `source -> process -> sink`
- add tests that the compiled low-level specs match expectations

Slice C4: pipeline runtime surface
- add `run()`, `start()`, `stop()`, `join()`, and context-manager behavior
- expose manager-backed metrics through the pipeline object
- add tests for lifecycle and worker startup ordering through the new API

Slice C5: first user-facing docs
- rewrite the README quickstart on the new API
- add one research-style example and one sensor/rocket-style example
- keep one advanced low-level example documented separately

Open design decisions to settle before coding:

- whether `Profile` is an enum-like object or a frozen dataclass with explicit ring/backlog settings
- whether process stages should support both return-by-value and output-buffer-writing in v0, or only one fast path first
- whether pipeline compilation should happen eagerly on `add_*` calls or lazily at `run()`
- whether `Pipeline.metrics()` should return a single snapshot dict first or a richer typed object

Default decisions unless implementation proves otherwise:

- use plain dataclasses, not a class hierarchy with behavior-heavy subclasses
- compile lazily at `run()` but make the compiled plan inspectable
- start with return-by-value stage functions plus explicit source/sink roles
- keep branching and fan-out out of v0
- make `sample_rate_hz` first-class and leave richer scientific metadata in user space for now

Acceptance examples for Milestone C:

- a user can build `source -> rfft -> sink` in under 20 lines of code
- a user can attach `sample_rate_hz` to the input stream and retrieve it from the pipeline config
- a user can switch from `balanced` to `latency_min` without rewriting stage code
- a user can still recreate the same pipeline manually with `Manager` and `TaskSpec` if needed

Exit criteria:

- a new user can build a simple DSP pipeline without touching `Manager` or `RingSpec`
- the low-level runtime still exists unchanged for power users
- the public API surface is small enough to document clearly in one page
- the first public API examples are credible for both research-style streaming analysis and sensor/rocket-style detection pipelines

### Milestone D: Runtime Profiles and Metrics

Target: 3 to 5 days

Purpose:

- move the best operational features out of benchmark scripts and into the library

Tasks:

- expose `throughput_max`, `latency_min`, and `balanced` as runtime profiles
- surface backlog, pressure, dropped-frame, throughput, and latency metrics
- provide both programmatic access and simple periodic logging hooks
- align benchmark presets with runtime profiles so the behavior is consistent

Exit criteria:

- users can choose pipeline behavior intentionally without editing benchmark code
- users can inspect a live pipeline with built-in metrics instead of ad hoc prints
- docs explain how to interpret the metrics and when to use each profile

### Milestone E: Research Packaging and Validation

Target: 3 to 5 days

Purpose:

- make the package credible to an external scientific user

Tasks:

- add CI for supported platforms and supported Python version
- document platform behavior explicitly, especially process termination and cleanup semantics
- tighten package metadata, install docs, versioning, and release checklist
- produce a short reproducibility guide with exact benchmark commands

Exit criteria:

- tests run automatically on the declared supported platforms
- supported behavior is documented rather than implied
- install, benchmark, and example commands work from a clean environment
- the package can ship a first tagged research-ready release

## Phased Roadmap

## Phase 1: Stabilize the Core

Focus:

- lock down zero-copy fast paths
- finish benchmark refactor
- add correctness tests for current DSP kernels
- clean up benchmark output and naming

Immediate tasks:

- add tests for `dsp_benchmark_suite` kernel math against NumPy references
- create a rocket-style benchmark focused on sample rate and total detection latency
- add structured benchmark output (`--json` later if useful)

Definition of done:

- core benchmarks are stable
- current examples and benchmarks do not rely on internal hacks
- cleanup and shutdown are reliable

## Phase 2: Public API Draft

Focus:

- create a first public pipeline abstraction
- reduce direct user exposure to `Manager`, `RingSpec`, and low-level ring decisions for common cases
- keep the abstraction honest enough for scientific users who need to reason about buffering, window size, and latency

Immediate tasks:

- add `StreamSpec` and `Profile` as pure public data objects
- add explicit `SourceStage`, `ProcessStage`, and `SinkStage`
- build a minimal `Pipeline` that compiles down to the current runtime
- prove the API on one research-style example and one rocket/sensor-style example
- keep the low-level runtime untouched and documented as the escape hatch

Definition of done:

- README quickstart uses the public API
- low-level API remains available for power users
- stream shape, dtype, and sample rate are visible in the public API
- the first public API is small enough to explain on one page without hand-waving

## Phase 3: Runtime Profiles and Observability

Focus:

- move benchmark policies into real runtime configuration
- make metrics first-class

Immediate tasks:

- add public runtime profiles
- add dropped-frame counters and backlog metrics
- expose worker and ring telemetry cleanly

Definition of done:

- users can configure throughput/latency behavior without benchmark scripts
- users can inspect live performance without custom instrumentation

## Phase 4: Cross-Platform Validation and Reliability

Focus:

- validate semantics across macOS, Linux, and Windows
- tighten lifecycle behavior

Immediate tasks:

- CI matrix
- platform-specific cleanup tests
- Windows behavior documentation

Definition of done:

- platform support is explicit, tested, and documented

## Phase 5: Product and Release Quality

Focus:

- docs
- packaging
- versioning
- first release candidate

Immediate tasks:

- write concept guides
- finalize benchmark docs
- publish performance narrative with reproducible commands

Definition of done:

- release candidate feels coherent to an external user

## What We Should Do Next

The next sequence should be:

1. Build the public API in small slices
- `StreamSpec` and `Profile` first
- then stage descriptors
- then the minimal `Pipeline` compiler

2. Prove the public API on real use cases
- one research pipeline example
- one rocket/sensor detection example
- keep both examples honest about frame size and latency

3. Promote profiles and metrics into runtime APIs
- do not keep the best ergonomics trapped in benchmark scripts

4. Rewrite the README quickstart around the stable API
- the low-level path should become an advanced section, not the first impression

5. Finish research packaging and validation
- CI, install polish, release checklist, and reproducibility docs

## Collaboration Model

To move quickly without thrashing, we should work in small, benchmarked increments:

- define one milestone
- implement it end to end
- benchmark it
- add tests
- document it
- then move to the next milestone

That means we should avoid broad rewrites without measurement. Every major performance or API change should ship with:

- validation commands
- benchmark impact
- test impact
- docs impact

## Success Checkpoints

We should consider the package on track when:

- latency-mode and throughput-mode behavior are both reproducible and understandable
- users can process streaming arrays without touching low-level rings
- the benchmark suite shows clear wins for the intended niche
- the package has a credible story for monitoring, high-throughput pipelines, and sensor-style DSP

We should consider it release-ready when:

- the public API is intentionally small
- correctness tests cover the marketed functionality
- docs match actual usage
- benchmark commands are stable and repeatable
- CI passes across supported platforms
