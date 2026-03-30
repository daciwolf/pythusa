from __future__ import annotations


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def align_size(size: int, alignment: int) -> int:
    if not is_power_of_two(alignment):
        raise ValueError("alignment must be a positive power of two")
    return (size + alignment - 1) & ~(alignment - 1)

