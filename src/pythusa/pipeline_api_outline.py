from __future__ import annotations

"""
Outline for the first small public pipeline API.

This file is intentionally design-oriented rather than executable runtime code.
It exists as a code-shaped implementation guide for Milestone C so the future
runtime can be built incrementally without inventing the API from scratch again.

The goal is to keep the public surface small:

- `StreamSpec` describes one framed stream.
- `Profile` describes operational policy.
- `SourceStage`, `ProcessStage`, and `SinkStage` describe explicit stage roles.
- `Pipeline` owns compilation to the existing low-level runtime.

The intended data flow for v0 is:

1. A source stage produces one frame for its output stream.
2. The pipeline runtime writes that frame into the ring assigned to the stream.
3. A process stage reads exactly one frame from its input stream, transforms it,
   and emits exactly one frame to its output stream.
4. A sink stage reads one frame and performs a side effect such as logging,
   threshold detection, storage, or external publication.
5. Metrics are reported by stage name and stream name so users can reason about
   throughput, latency, backlog, and drops without studying ring internals.

The intended compile path is:

- public `Pipeline`
-> internal stream/ring sizing decisions
-> `RingSpec`
-> `TaskSpec`
-> `Manager`

Internal design note:

- v0 should still store stages and streams as a small graph using plain Python
  data structures so future fan-out and fan-in can be added without rewriting
  the planner
- v0 does not need an external graph library if adjacency maps, cycle checks,
  and topological ordering are enough
- public branching semantics remain a follow-on step after the linear API is
  stable

The public API should not hide that buffering exists, but it should remove the
need for a new user to wire rings and tasks by hand for common cases.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


ProfileName = Literal["throughput_max", "latency_min", "balanced"]


@dataclass(frozen=True)
class StreamSpec:
    """
    Pure data describing one framed stream.

    Expected use:
    - research users declare the frame shape and dtype explicitly
    - the pipeline compiler uses this object to derive ring size and view shape

    Important design rule:
    - this object carries metadata only
    - it should not know anything about processes, rings, or worker lifecycle

    Data-flow note:
    - one `StreamSpec` normally maps to one shared-memory ring in v0
    - the stream name becomes the stable user-facing identifier for metrics,
      compile plans, benchmark output, and reproducibility notes
    - later fan-out may map one logical stream either to one shared ring with
      multiple readers or to per-branch rings depending on the public semantics
      chosen for DAG support
    """

    name: str
    shape: tuple[int, ...]
    dtype: Any
    description: str | None = None


@dataclass(frozen=True)
class Profile:
    """
    Declarative runtime policy for throughput/latency behavior.

    Expected use:
    - the user chooses a profile by intent
    - the pipeline compiler resolves that intent into concrete ring depth,
      backlog limits, and polling/sleep behavior using the existing runtime

    Policy note:
    - profile is configuration, not control flow
    - a user should be able to switch profiles without rewriting stage code

    Implementation note:
    - v0 can start with the benchmark-aligned presets and only a few explicit
      overrides
    - later work can grow this carefully if real workloads justify it
    """

    name: ProfileName
    ring_depth: int | None = None
    max_in_flight_batches: int | None = None
    idle_sleep_s: float | None = None


@dataclass(frozen=True)
class SourceStage:
    """
    Produces frames for exactly one output stream.

    Callable contract under consideration:
    - simplest form: `fn() -> array`
    - lower-copy form: `fn(output: array) -> None`

    The final choice should be explicit and documented rather than inferred.

    Data-flow note:
    - this stage owns acquisition or generation only
    - it should not know about the downstream ring implementation
    - the runtime adapter will move the returned or filled frame into the
      stream's ring using the fast path that matches the configured profile
    """

    name: str
    output: str
    fn: Callable[..., Any]
    description: str | None = None


@dataclass(frozen=True)
class ProcessStage:
    """
    Reads one framed stream and emits one framed stream.

    Callable contract under consideration:
    - simplest form: `fn(frame: array) -> array`
    - lower-copy form: `fn(frame: array, output: array) -> None`

    Linux-style constraint:
    - one process stage does one transform
    - chaining many small transforms is preferred over a monolithic opaque stage
      unless the user explicitly chooses to fuse them

    Data-flow note:
    - process stages should not allocate transport objects
    - they operate on arrays, not on rings or manager state
    - the pipeline adapter owns reading from the input stream and publishing the
      result to the output stream
    """

    name: str
    input: str
    output: str
    fn: Callable[..., Any]
    description: str | None = None


@dataclass(frozen=True)
class SinkStage:
    """
    Consumes frames from one input stream and performs a side effect.

    Typical sink roles:
    - logging or file output
    - threshold detection
    - metric emission
    - forwarding to hardware or network code owned by the user

    Data-flow note:
    - sinks terminate a public pipeline branch in v0
    - the sink should not need to know where the frame came from or how many
      rings/tasks were required to deliver it
    """

    name: str
    input: str
    fn: Callable[..., Any]
    description: str | None = None


@dataclass(frozen=True)
class CompiledPipelinePlan:
    """
    Inspectable low-level plan produced from a `Pipeline`.

    Why this exists:
    - advanced users should be able to see exactly what the high-level API
      generated
    - research users may want to save the compiled plan alongside experiments
    - this gives a clear escape hatch back to the low-level runtime

    Intended contents:
    - stream specs keyed by name
    - resolved profile values
    - generated ring names and their sizes
    - generated task names and stage-to-task mapping
    """

    streams: dict[str, StreamSpec] = field(default_factory=dict)
    profile: Profile | None = None
    ring_specs: dict[str, Any] = field(default_factory=dict)
    task_specs: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


class Pipeline:
    """
    Public pipeline builder and lifecycle owner.

    Intended responsibilities:
    - hold the user-declared streams and stages
    - validate the pipeline graph for the restricted v0 model
    - compile to the existing `Manager`, `RingSpec`, and `TaskSpec`
    - expose lifecycle methods and metrics without leaking the whole manager API

    v0 scope:
    - linear or simple chained pipelines only
    - no hidden fan-out
    - no graph optimizer
    - no automatic fusion

    Internal representation note:
    - the planner should still keep topology in a graph-shaped form
    - plain adjacency data and topological ordering are preferred first
    - external graph libraries are unnecessary unless those simple structures
      stop being sufficient

    Data-flow summary:
    - `add_*` methods collect declarative objects only
    - `compile()` decides how many rings and tasks are needed
    - `start()` materializes the manager and launches processes
    - `run()` is a convenience wrapper for start + wait + shutdown behavior
    """

    def __init__(self, name: str, profile: ProfileName | Profile = "balanced") -> None:
        """
        Hold the user-facing pipeline name and desired operational profile.

        Planned behavior:
        - accept a preset profile name for convenience
        - also accept an explicit `Profile` object for advanced override
        - keep construction side-effect free so pipelines are easy to inspect,
          serialize, and test before they are started
        """
        ...

    def add_stream(self, stream: StreamSpec) -> "Pipeline":
        """
        Register a named stream explicitly.

        Planned use:
        - useful when the same stream should be referenced by multiple stage
          declarations or when sample-rate metadata should be declared up front

        Validation expectations:
        - stream names must be unique
        - shape and dtype must be concrete and stable for the life of the
          pipeline in v0
        """
        ...

    def add_source(
        self,
        name: str,
        *,
        output: str | StreamSpec,
        fn: Callable[..., Any],
        description: str | None = None,
    ) -> "Pipeline":
        """
        Register a source stage.

        Flow:
        - if `output` is a `StreamSpec`, the pipeline should register it
        - if `output` is a name, the stream must already exist
        - the stage is stored as pure configuration until compile time
        """
        ...

    def add_process(
        self,
        name: str,
        *,
        input: str,
        output: str | StreamSpec,
        fn: Callable[..., Any],
        description: str | None = None,
    ) -> "Pipeline":
        """
        Register a transform stage.

        Flow:
        - the declared input stream supplies exactly one frame per invocation
        - the declared output stream receives exactly one frame per invocation
        - a future compiled task wrapper will translate this stage into array
          reads/writes against the correct rings
        """
        ...

    def add_sink(
        self,
        name: str,
        *,
        input: str,
        fn: Callable[..., Any],
        description: str | None = None,
    ) -> "Pipeline":
        """
        Register a terminal sink stage.

        Flow:
        - the sink is the last consumer for a stream in v0
        - the sink callable owns side effects, not the pipeline framework
        """
        ...

    def compile(self) -> CompiledPipelinePlan:
        """
        Resolve the declarative pipeline into low-level runtime objects.

        Planned compile steps:
        1. validate that stage ordering and stream references are legal
        2. resolve the chosen profile into concrete runtime settings
        3. map streams to ring specs
        4. wrap stage callables in task functions that perform the correct read,
           transform, write, and metrics bookkeeping
        5. return an inspectable plan object before any process is started

        Important design rule:
        - compile should be deterministic and side-effect free
        - users should be able to inspect or snapshot the plan before running it
        """
        ...

    def start(self) -> None:
        """
        Materialize the compiled plan into a live `Manager` and start workers.

        Planned flow:
        - create rings first
        - create tasks second
        - start consumer-side tasks before producer-side tasks where needed to
          reduce startup overruns
        - store enough internal state that `stop()`, `join()`, and `metrics()`
          can operate without exposing the raw manager by default
        """
        ...

    def run(self) -> None:
        """
        Convenience entry point for the common long-running case.

        Planned behavior:
        - compile if needed
        - start the runtime
        - block until interrupted or until a configured stop condition occurs
        - stop and join workers cleanly on exit

        This method is for ergonomics only.
        The lower-level lifecycle methods should remain available.
        """
        ...

    def stop(self) -> None:
        """
        Request worker shutdown through the compiled manager surface.

        Planned scope:
        - stop all tasks created for this pipeline
        - preserve a clean path for later support of targeted stage shutdown if
          a real use case requires it
        """
        ...

    def join(self, timeout: float | None = None) -> None:
        """
        Wait for the pipeline's worker processes to exit.

        Planned behavior:
        - forward to the manager with a consistent default timeout policy
        - keep shutdown semantics explicit rather than magical
        """
        ...

    def metrics(self) -> dict[str, Any]:
        """
        Return a pipeline-scoped metrics snapshot.

        Intended contents for v0:
        - process metrics keyed by stage name
        - ring or stream pressure keyed by stream name
        - resolved profile and stream metadata for context

        Important design rule:
        - metric names should be understandable without reading manager internals
        - stage and stream names should remain stable across runs
        """
        ...

    def compiled_plan(self) -> CompiledPipelinePlan | None:
        """
        Return the last compiled plan if one exists.

        Why this matters:
        - debugging
        - reproducibility
        - a clean escape hatch for users who outgrow the high-level API
        """
        ...
