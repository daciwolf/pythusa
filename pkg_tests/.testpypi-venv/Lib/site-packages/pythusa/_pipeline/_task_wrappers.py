from __future__ import annotations

"""Task registration helpers and controlled-task run loops."""

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .pipeline import Pipeline


_TaskFn = Callable[..., Any]
_TaskDecorator = Callable[[_TaskFn], _TaskFn]


class _TaskRegistrationAPI:
    """Callable task-registration helper exposed as ``pipe.add_task``."""

    def __init__(self, pipeline: "Pipeline") -> None:
        self._pipeline = pipeline

    def __call__(
        self,
        name: str,
        *,
        fn: _TaskFn | None = None,
        reads: dict[str, str] | None = None,
        writes: dict[str, str] | None = None,
        events: dict[str, str] | None = None,
        description: str | None = None,
    ) -> "Pipeline" | _TaskDecorator:
        return self._decorate_or_register(
            name=name,
            fn=fn,
            reads=reads,
            writes=writes,
            events=events,
            description=description,
            control_mode=None,
            control_event=None,
        )

    def switchable(
        self,
        name: str,
        *,
        activate_on: str,
        fn: _TaskFn,
        reads: dict[str, str] | None = None,
        writes: dict[str, str] | None = None,
        events: dict[str, str] | None = None,
        description: str | None = None,
    ) -> "Pipeline":
        return self._register_controlled_task(
            name=name,
            fn=fn,
            reads=reads,
            writes=writes,
            events=events,
            description=description,
            control_mode="switchable",
            control_event=activate_on,
        )

    def toggleable(
        self,
        name: str,
        *,
        activate_on: str,
        fn: _TaskFn,
        reads: dict[str, str] | None = None,
        writes: dict[str, str] | None = None,
        events: dict[str, str] | None = None,
        description: str | None = None,
    ) -> "Pipeline":
        return self._register_controlled_task(
            name=name,
            fn=fn,
            reads=reads,
            writes=writes,
            events=events,
            description=description,
            control_mode="toggleable",
            control_event=activate_on,
        )

    def _register_controlled_task(
        self,
        *,
        name: str,
        fn: _TaskFn,
        reads: dict[str, str] | None,
        writes: dict[str, str] | None,
        events: dict[str, str] | None,
        description: str | None,
        control_mode: str,
        control_event: str,
    ) -> "Pipeline":
        return self._pipeline._add_task(
            name=name,
            fn=fn,
            reads=reads,
            writes=writes,
            events=events,
            description=description,
            control_mode=control_mode,
            control_event=control_event,
        )

    def _decorate_or_register(
        self,
        *,
        name: str,
        fn: _TaskFn | None,
        reads: dict[str, str] | None,
        writes: dict[str, str] | None,
        events: dict[str, str] | None,
        description: str | None,
        control_mode: str | None,
        control_event: str | None,
    ) -> "Pipeline" | _TaskDecorator:
        if fn is not None:
            return self._pipeline._add_task(
                name=name,
                fn=fn,
                reads=reads,
                writes=writes,
                events=events,
                description=description,
                control_mode=control_mode,
                control_event=control_event,
            )

        def decorator(task_fn: _TaskFn) -> _TaskFn:
            self._pipeline._add_task(
                name=name,
                fn=task_fn,
                reads=reads,
                writes=writes,
                events=events,
                description=description,
                control_mode=control_mode,
                control_event=control_event,
            )
            return task_fn

        return decorator


def run_controlled_task(
    fn: _TaskFn,
    *,
    control_mode: str | None,
    activate_on: str | None,
    kwargs: dict[str, Any],
) -> Any:
    """Run a task directly or under a switch/toggle control loop."""
    if control_mode is None:
        return fn(**kwargs)

    event = _activation_event(kwargs, activate_on)
    fn_kwargs = _controlled_call_kwargs(kwargs, activate_on)

    if control_mode == "toggleable":
        while True:
            event.wait()
            event.reset()
            fn(**fn_kwargs)

    if control_mode == "switchable":
        while True:
            event.wait()
            fn(**fn_kwargs)

    raise ValueError(f"Unsupported task control mode: {control_mode!r}")


def _activation_event(
    kwargs: dict[str, Any],
    activate_on: str | None,
) -> Any:
    if not activate_on:
        raise ValueError("Controlled tasks require an 'activate_on' event binding.")

    try:
        return kwargs[activate_on]
    except KeyError as exc:
        raise KeyError(
            f"Controlled task expected bound event '{activate_on}' in kwargs."
        ) from exc


def _controlled_call_kwargs(
    kwargs: dict[str, Any],
    activate_on: str | None,
) -> dict[str, Any]:
    task_kwargs = dict(kwargs)
    if activate_on is not None:
        task_kwargs.pop(activate_on, None)
    return task_kwargs
