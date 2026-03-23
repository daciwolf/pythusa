from __future__ import annotations

"""
Design-only example for the ideal public workflow.

This file is intentionally not executable product code.
It exists to show the shape of the API we are building and the kind of user
code it should make easy to write.

Key ideas shown here:

- the user owns source and compute logic
- the pipeline owns topology declaration and later orchestration
- streams are declared explicitly
- tasks declare what they read and what they write with stable binding names
- tasks may also bind named events for coordination
- one stream compiles to one ring
- one ring has one writer and zero or more readers
"""

import numpy as np

import pythusa


def acquire(samples) -> None:
    """
    Example user-owned source task.

    Real user code might:
    - connect to hardware
    - read from a socket
    - receive async data from another system
    - batch that data into frames

    The eventual runtime will decide exactly how named outputs are handed to
    this function.
    """
    frame_size = 4096
    phase = 0.0
    phase_step = 0.05
    base = np.linspace(0.0, 2.0 * np.pi, frame_size, endpoint=False, dtype=np.float32)

    while True:
        # Produce one frame of synthetic sample data. Real code would read from
        # a device, socket, SDK callback, or async source.
        frame = np.sin(base + phase).astype(np.float32, copy=False)
        samples.write(frame)
        phase += phase_step


def fft_worker(samples, fft) -> None:
    """
    Example user-owned compute task.

    Real user code might:
    - read frames from the named `samples` input
    - run FFT or other DSP on those frames
    - publish results to the named `fft` output

    The user is responsible for making this efficient and correct.
    """
    while True:
        frame = samples.read()
        spectrum = np.fft.rfft(frame).astype(np.complex64, copy=False)
        fft.write(spectrum)


def store_fft(fft) -> None:
    """
    Example user-owned sink task.

    Real user code might:
    - read FFT frames
    - serialize or summarize them
    - write them to a database or file
    """
    records_written = 0

    while True:
        spectrum = fft.read()
        peak_bin = int(np.argmax(np.abs(spectrum)))
        peak_power = float(np.abs(spectrum[peak_bin]))

        # Real code would write to a database, file, or external service.
        record = {
            "peak_bin": peak_bin,
            "peak_power": peak_power,
            "records_written": records_written,
        }
        _write_record(record)
        records_written += 1


def _write_record(record: dict[str, float | int]) -> None:
    """
    Small helper used by `store_fft`.

    This stands in for whatever user-owned persistence layer exists in the real
    application.
    """
    _ = record


def build_pipeline() -> pythusa.Pipeline:
    """
    Example of the ideal topology declaration.

    This is the workflow we are currently designing toward.
    """
    pipe = pythusa.Pipeline("radar")

    pipe.add_stream("samples", shape=(4096,), dtype=np.float32)
    pipe.add_stream("fft", shape=(2049,), dtype=np.complex64)
    pipe.add_event("shutdown")

    pipe.add_task(
        "acquire",
        fn=acquire,
        writes={"samples": "samples"},
        events={"shutdown": "shutdown"},
    )

    pipe.add_task(
        "fft_worker_1",
        fn=fft_worker,
        reads={"samples": "samples"},
        writes={"fft": "fft"},
        events={"shutdown": "shutdown"},
    )

    pipe.add_task(
        "db_writer",
        fn=store_fft,
        reads={"fft": "fft"},
        events={"shutdown": "shutdown"},
    )

    return pipe
