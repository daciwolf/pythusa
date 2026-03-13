from __future__ import annotations

from typing import TypeAlias


RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]
BytesLike: TypeAlias = bytes | bytearray | memoryview
ShapeLike: TypeAlias = tuple[int, ...]

