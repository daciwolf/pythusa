from __future__ import annotations

"""Thin framed-array helpers layered on top of raw shared-memory rings."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._buffers.ring import SharedRingBuffer
from .._processing.numpy import bytes_for_shape


@dataclass(frozen=True)
class _StreamBindingSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


class StreamReader:
    """
    Convenience reader for framed array streams.

    `raw` exposes the underlying SharedRingBuffer for users who want direct
    ring access. `read()` and `read_into()` are convenience helpers on top.
    `look()` returns a zero-copy memoryview for the next contiguous frame, and
    `increment()` advances the reader after you are done with that view.

    `set_blocking(False)` marks this reader inactive so writers stop treating
    it as a backpressure participant. Re-enabling jumps the reader to the
    current writer position to avoid stale unread backlog.
    """

    def __init__(self, raw: SharedRingBuffer, spec: _StreamBindingSpec) -> None:
        self.raw = raw
        self.ring = raw
        self.name = spec.name
        self.shape = tuple(spec.shape)
        self.dtype = np.dtype(spec.dtype)
        self.frame_nbytes = bytes_for_shape(self.shape, self.dtype)
        self.frame_size = int(np.prod(self.shape, dtype=np.int64))

    def read(self) -> np.ndarray | None:
        array = self.raw.read_array(self.frame_nbytes, dtype=self.dtype)
        if array.size != self.frame_size:
            return None
        return np.array(array, copy=True).reshape(self.shape)

    def read_into(self, out: np.ndarray) -> bool:
        array = _require_frame_array(out, shape=self.shape, dtype=self.dtype)
        reader_mem_view = self.raw.expose_reader_mem_view(self.frame_nbytes)
        if reader_mem_view[2] < self.frame_nbytes:
            return False
        self.raw.simple_read(reader_mem_view, memoryview(array).cast("B"))
        self.raw.inc_reader_pos(self.frame_nbytes)
        return True

    def look(self) -> memoryview | None:
        mv1, mv2, size_readable, wrap_around = self.raw.expose_reader_mem_view(self.frame_nbytes)
        if size_readable < self.frame_nbytes or wrap_around or mv2 is not None:
            return None
        return mv1

    def increment(self) -> None:
        self.raw.inc_reader_pos(self.frame_nbytes)

    def set_blocking(self, blocking: bool) -> None:
        if blocking:
            self.raw.jump_to_writer()
            self.raw.set_reader_active(True)
            return
        self.raw.set_reader_active(False)

    def is_blocking(self) -> bool:
        return self.raw.is_reader_active()


class StreamWriter:
    """
    Convenience writer for framed array streams.

    `raw` exposes the underlying SharedRingBuffer for direct low-level access.
    `write()` validates one frame and publishes it using the raw ring helper.
    `look()` returns a zero-copy writable memoryview for the next contiguous
    frame, and `increment()` advances the writer after you fill that view.
    """

    def __init__(self, raw: SharedRingBuffer, spec: _StreamBindingSpec) -> None:
        self.raw = raw
        self.ring = raw
        self.name = spec.name
        self.shape = tuple(spec.shape)
        self.dtype = np.dtype(spec.dtype)
        self.frame_nbytes = bytes_for_shape(self.shape, self.dtype)

    def write(self, array: np.ndarray) -> bool:
        frame = _require_frame_array(array, shape=self.shape, dtype=self.dtype)
        return self.raw.write_array(frame) == self.frame_nbytes

    def look(self) -> memoryview | None:
        mv1, mv2, size_writeable, wrap_around = self.raw.expose_writer_mem_view(self.frame_nbytes)
        if size_writeable < self.frame_nbytes or wrap_around or mv2 is not None:
            return None
        return mv1

    def increment(self) -> None:
        self.raw.inc_writer_pos(self.frame_nbytes)


def make_reader_binding(
    raw: SharedRingBuffer,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> StreamReader:
    return StreamReader(raw, _binding_spec(name=name, shape=shape, dtype=dtype))


def make_writer_binding(
    raw: SharedRingBuffer,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> StreamWriter:
    return StreamWriter(raw, _binding_spec(name=name, shape=shape, dtype=dtype))


def _binding_spec(
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> _StreamBindingSpec:
    """Normalize binding metadata into one immutable stream spec."""
    return _StreamBindingSpec(
        name=name,
        shape=tuple(shape),
        dtype=np.dtype(dtype),
    )


def _require_frame_array(
    array: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Validate one frame for shape, dtype, and contiguous layout."""
    frame = np.asarray(array)
    if tuple(frame.shape) != tuple(shape):
        raise ValueError(
            f"Expected frame with shape {shape}, got {tuple(frame.shape)}."
        )
    if frame.dtype != dtype:
        raise ValueError(
            f"Expected frame with dtype {dtype}, got {frame.dtype}."
        )
    if not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)
    return frame
