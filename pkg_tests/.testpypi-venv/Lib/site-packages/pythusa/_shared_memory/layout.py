from __future__ import annotations

from .._utils.alignment import align_size, is_power_of_two


HEADER_STATIC_FIELDS = 6
READER_FIELDS = 3
UINT64_BYTES = 8


def header_u64_length(num_readers: int) -> int:
    return HEADER_STATIC_FIELDS + (num_readers * READER_FIELDS)


def compute_header_size(
    num_readers: int,
    *,
    cache_align: bool = False,
    cache_size: int = 64,
) -> int:
    header_size = UINT64_BYTES * header_u64_length(num_readers)
    if not cache_align:
        return header_size
    if not is_power_of_two(cache_size):
        raise ValueError("cache_size must be a positive power of two when cache_align is True")
    return align_size(header_size, cache_size)


def reader_slot(reader: int) -> int:
    return HEADER_STATIC_FIELDS + (reader * READER_FIELDS)

