from __future__ import annotations

import math

import numpy as np

from ..typing import ShapeLike


def buffer_to_array(
    buffer: memoryview | bytes | bytearray,
    *,
    dtype: np.dtype,
    shape: ShapeLike | None = None,
) -> np.ndarray:
    """Interpret a raw buffer as a NumPy array, optionally reshaped."""

    array = np.frombuffer(buffer, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    return array


def bytes_for_shape(shape: ShapeLike, dtype: np.dtype) -> int:
    """Return the contiguous byte size for an ndarray with the given shape and dtype."""

    return math.prod(shape) * np.dtype(dtype).itemsize

