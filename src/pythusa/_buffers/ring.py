from __future__ import annotations

from multiprocessing import shared_memory
import time
import weakref
from dataclasses import dataclass

import numpy as np

from .._shared_memory.layout import compute_header_size, header_u64_length, reader_slot
from ..typing import RingView


__all__ = ["RingSpec", "SharedRingBuffer"]

_DEFAULT_MIN_READER_POS_REFRESH_INTERVAL = 64
_DEFAULT_MIN_READER_POS_REFRESH_S = 0.005

# Pressure is the complement of available writer space expressed as a percentage.
# For example, if the writer has 33 free bytes left in a 100-byte ring, the ring
# is considered to be under 67% pressure.


@dataclass(frozen=True)
class RingSpec:
    name: str
    size: int
    num_readers: int
    reader: int = -1
    cache_align: bool = True
    cache_size: int = 64
    min_reader_pos_refresh_interval: int = _DEFAULT_MIN_READER_POS_REFRESH_INTERVAL
    min_reader_pos_refresh_s: float = _DEFAULT_MIN_READER_POS_REFRESH_S

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"RingSpec '{self.name}': size must be > 0")
        if self.num_readers < 1:
            raise ValueError(f"RingSpec '{self.name}': need at least 1 reader")
        if self.cache_align and (self.cache_size & (self.cache_size - 1)):
            raise ValueError("cache_size must be a power of two")
        if (
            isinstance(self.min_reader_pos_refresh_interval, bool)
            or not isinstance(self.min_reader_pos_refresh_interval, int)
        ):
            raise TypeError("min_reader_pos_refresh_interval must be an integer")
        if self.min_reader_pos_refresh_interval < 1:
            raise ValueError("min_reader_pos_refresh_interval must be >= 1")
        if isinstance(self.min_reader_pos_refresh_s, bool) or not isinstance(
            self.min_reader_pos_refresh_s, (int, float)
        ):
            raise TypeError("min_reader_pos_refresh_s must be a real number")
        if self.min_reader_pos_refresh_s < 0:
            raise ValueError("min_reader_pos_refresh_s must be >= 0")

    def to_kwargs(self, *, create: bool, reader: int) -> dict:
        return dict(
            name=self.name, create=create, size=self.size,
            num_readers=self.num_readers, reader=reader,
            cache_align=self.cache_align, cache_size=self.cache_size,
            min_reader_pos_refresh_interval=self.min_reader_pos_refresh_interval,
            min_reader_pos_refresh_s=float(self.min_reader_pos_refresh_s),
        )

    def __repr__(self) -> str:
        return f"RingSpec(name={self.name!r}, size={self.size}, readers={self.num_readers})"


class SharedRingBuffer(shared_memory.SharedMemory):
    # Header layout, all uint64:
    # 0  size
    # 1  pressure
    # 2  dropped_size
    # 3  write_position
    # 4  max_amount_writable
    # 5  num_readers
    # Repeated per reader:
    #    reader_position
    #    reader_alive
    #    reader_last_seen_ns

    # Sentinel: instances constructed with reader=_NO_READER are writer-only.
    # reader_pos_index is set to None for these instances so any accidental
    # call to a reader-path method fails loudly rather than silently
    # corrupting header slots (reader=-1 maps to index 3 = write_pos_index).
    _NO_READER: int = -1

    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        num_readers: int,
        reader: int,
        cache_align: bool = False,
        cache_size: int = 64,
        min_reader_pos_refresh_interval: int = _DEFAULT_MIN_READER_POS_REFRESH_INTERVAL,
        min_reader_pos_refresh_s: float = _DEFAULT_MIN_READER_POS_REFRESH_S,
    ):
        self.cache_align = cache_align
        if self.cache_align:
            if cache_size <= 0:
                raise ValueError("cache_size must be > 0 when cache_align is True")
            if cache_size & (cache_size - 1):
                raise ValueError("cache_size must be a power of two when cache_align is True")
        if (
            isinstance(min_reader_pos_refresh_interval, bool)
            or not isinstance(min_reader_pos_refresh_interval, int)
        ):
            raise TypeError("min_reader_pos_refresh_interval must be an integer")
        if min_reader_pos_refresh_interval < 1:
            raise ValueError("min_reader_pos_refresh_interval must be >= 1")
        if isinstance(min_reader_pos_refresh_s, bool) or not isinstance(
            min_reader_pos_refresh_s, (int, float)
        ):
            raise TypeError("min_reader_pos_refresh_s must be a real number")
        if min_reader_pos_refresh_s < 0:
            raise ValueError("min_reader_pos_refresh_s must be >= 0")
        if reader != self._NO_READER and not 0 <= reader < num_readers:
            raise ValueError(f"reader index {reader} is out of range for num_readers={num_readers}")

        self.header_size = compute_header_size(
            num_readers,
            cache_align=self.cache_align,
            cache_size=cache_size,
        )

        self.shared_mem_size = size + self.header_size
        self.max_amount_writable_index = 4
        self.write_pos_index = 3
        self.size_index = 0
        self.pressure_index = 1
        self.dropped_size_index = 2
        self.num_readers_index = 5
        self.ring_buffer_size = size
        self.num_readers = num_readers
        super().__init__(name=name, create=create, size=self.shared_mem_size)
        self.header = np.ndarray(
            (header_u64_length(num_readers),),
            np.uint64,
            memoryview(self.buf[0:self.header_size]),
            0,
        )
        if create:
            self.header[:] = 0
            self.header[0] = self.ring_buffer_size
            self.header[self.num_readers_index] = self.num_readers
        self.reader = reader

        # reader=-1 (writer-only instance) would map to index 3, which is
        # write_pos_index. Set to None instead so reader-path methods fail
        # loudly rather than silently corrupting the write position.
        if self.reader == self._NO_READER:
            self.reader_pos_index = None
        else:
            self.reader_pos_index = reader_slot(reader)

        self.reader_pos = 0
        self.write_pos = 0
        self.max_amount_writable = self.ring_buffer_size
        # Restrict payload view to logical ring size; shared memory may be page-rounded.
        self.ring_buffer = memoryview(self.buf[self.header_size:self.header_size + self.ring_buffer_size])
        # min-reader cache: avoids O(num_readers) scans on every write-path query.
        # Stale cache is conservative (smaller writable), never optimistic.
        self._min_reader_pos_refresh_interval = min_reader_pos_refresh_interval
        # Also refresh on wall-clock cadence so external reader progress is seen
        # even when writer is stalled (no local writes to trigger periodic scan).
        self._min_reader_pos_refresh_s = float(min_reader_pos_refresh_s)
        self._writes_since_min_scan = 0
        self._reader_positions_dirty = False
        self._min_reader_pos_cache = self._scan_min_reader_pos()
        self._last_min_scan_t = time.perf_counter()
        self._is_creator = create
        weakref.finalize(
            self,
            SharedRingBuffer._finalizer_cleanup,
            self.name,
            create,         # only the creator unlinks
        )

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _finalizer_cleanup(name: str, is_creator: bool) -> None:
        """Called by GC or interpreter shutdown. Must be a static/free function."""
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            if is_creator:  # only unlinks the memory if it was allocated by this process
                shm.unlink()
        except FileNotFoundError:
            pass

    def _release_local_views(self) -> None:
        ring_buffer = getattr(self, "ring_buffer", None)
        if ring_buffer is not None:
            try:
                ring_buffer.release()
            except Exception:
                pass
            self.ring_buffer = None

        header = getattr(self, "header", None)
        if header is not None:
            base = getattr(header, "base", None)
            if isinstance(base, memoryview):
                try:
                    base.release()
                except Exception:
                    pass
            self.header = None

    def close(self) -> None:
        self._release_local_views()
        super().close()

    def __enter__(self) -> "SharedRingBuffer":
        if self.reader != self._NO_READER:
            self.set_reader_active(True)
        return self

    def __exit__(self, *_):
        if self.reader != self._NO_READER:
            self.set_reader_active(False)
        self.close()
        if self._is_creator:
            try:
                self.unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------ #
    # Pressure                                                             #
    # ------------------------------------------------------------------ #

    def calculate_pressure(self) -> int:
        writable = self.compute_max_amount_writable(force_rescan=True)
        used = self.ring_buffer_size - writable
        pressure = int((used / self.ring_buffer_size) * 100)
        self.header[self.pressure_index] = pressure
        return pressure

    # ------------------------------------------------------------------ #
    # Position helpers                                                     #
    # ------------------------------------------------------------------ #

    def int_to_pos(self, value: int) -> int:
        return value % self.ring_buffer_size

    def _assert_is_reader(self, method: str) -> None:
        """Raises clearly if a reader-only method is called on a writer instance."""
        if self.reader_pos_index is None:
            raise RuntimeError(
                f"{method} called on a writer-only SharedRingBuffer instance "
                f"(reader={self.reader}). Construct with a valid reader index to use reader-path methods."
            )

    def update_reader_pos(self, new_reader_pos: int) -> None:
        self._assert_is_reader("update_reader_pos")
        self.header[self.reader_pos_index] = new_reader_pos
        self.reader_pos = new_reader_pos
        self._reader_positions_dirty = True

    def set_reader_active(self, active: bool) -> None:
        self._assert_is_reader("set_reader_active")
        self.header[self.reader_pos_index + 1] = 1 if active else 0
        self._reader_positions_dirty = True

    def is_reader_active(self) -> bool:
        self._assert_is_reader("is_reader_active")
        return bool(self.header[self.reader_pos_index + 1])

    def update_write_pos(self, new_writer_pos: int) -> None:
        self.header[self.write_pos_index] = new_writer_pos
        self.write_pos = new_writer_pos
        self._writes_since_min_scan += 1

    def inc_writer_pos(self, inc_amount: int) -> None:
        self.header[self.write_pos_index] = self.header[self.write_pos_index] + inc_amount
        self.write_pos = self.header[self.write_pos_index]
        self._writes_since_min_scan += 1

    def inc_reader_pos(self, inc_amount: int) -> None:
        self._assert_is_reader("inc_reader_pos")
        self.header[self.reader_pos_index] = self.header[self.reader_pos_index] + inc_amount
        self.reader_pos = self.header[self.reader_pos_index]
        self._reader_positions_dirty = True

    def get_write_pos(self) -> np.uint64:
        return self.header[self.write_pos_index]

    # ------------------------------------------------------------------ #
    # Min-reader scan + writable computation                              #
    # ------------------------------------------------------------------ #

    def _scan_min_reader_pos(self) -> int:
        min_reader_pos = int(self.header[self.write_pos_index])
        for i in range(6, len(self.header), 3):
            reader_pos = int(self.header[i])
            reader_alive = int(self.header[i + 1])
            if reader_pos < min_reader_pos and reader_alive:
                min_reader_pos = reader_pos
        return min_reader_pos

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        """Returns max amount writable in bytes."""
        write_pos = int(self.header[self.write_pos_index])
        if (
            force_rescan
            or self._min_reader_pos_cache is None
            or self._reader_positions_dirty
            or self._writes_since_min_scan >= self._min_reader_pos_refresh_interval
            or (time.perf_counter() - self._last_min_scan_t) >= self._min_reader_pos_refresh_s
        ):
            min_reader_pos = self._scan_min_reader_pos()
            self._min_reader_pos_cache = min_reader_pos
            self._writes_since_min_scan = 0
            self._reader_positions_dirty = False
            self._last_min_scan_t = time.perf_counter()
        else:
            min_reader_pos = self._min_reader_pos_cache

        used = write_pos - min_reader_pos
        # If cached min is too old, used can exceed ring size conservatively.
        # Refresh once before treating it as a hard invariant failure.
        if used > self.ring_buffer_size:
            min_reader_pos = self._scan_min_reader_pos()
            self._min_reader_pos_cache = min_reader_pos
            self._writes_since_min_scan = 0
            self._reader_positions_dirty = False
            self._last_min_scan_t = time.perf_counter()
            used = write_pos - min_reader_pos

        assert used >= 0, (
            f"invariant violation: used < 0 (write_pos={write_pos}, "
            f"min_reader_pos={min_reader_pos}, used={used})"
        )
        assert used <= self.ring_buffer_size, (
            f"invariant violation: used > ring_buffer_size "
            f"(write_pos={write_pos}, min_reader_pos={min_reader_pos}, "
            f"used={used}, ring_buffer_size={self.ring_buffer_size})"
        )
        self.max_amount_writable = self.ring_buffer_size - used
        self.header[self.max_amount_writable_index] = self.max_amount_writable
        return self.max_amount_writable

    def jump_to_writer(self) -> None:
        """Advance this reader's position to the current write position, discarding unread data."""
        self._assert_is_reader("jump_to_writer")
        self.update_reader_pos(int(self.get_write_pos()))

    # ------------------------------------------------------------------ #
    # Memory view exposure                                                 #
    # ------------------------------------------------------------------ #

    def expose_writer_mem_view(self, size: int) -> RingView:
        """
        Return a memoryview (or pair) for a direct write into the ring buffer.
        size_writeable is the real size available (may be less than requested).
        wrap_around is True when the write spans the end of the buffer and mv2 is needed.
        """
        # Recompute every call; cached writable becomes stale after position updates.
        self.compute_max_amount_writable()
        if self.max_amount_writable >= size:
            size_writeable = size
        else:
            size_writeable = self.max_amount_writable
        write_pos = self.int_to_pos(int(self.header[self.write_pos_index]))
        # <= not <: when write_pos + size_writeable == ring_buffer_size the
        # write fits exactly at the end with no wrap needed.
        if write_pos + size_writeable <= self.ring_buffer_size:
            mv1 = memoryview(self.ring_buffer[write_pos: write_pos + size_writeable])
            mv2 = None
            wrap_around = False
        else:
            mv1 = memoryview(self.ring_buffer[write_pos:])
            mv2 = memoryview(self.ring_buffer[0:(size_writeable - (self.ring_buffer_size - write_pos))])
            wrap_around = True

        return (mv1, mv2, size_writeable, wrap_around)

    def expose_reader_mem_view(self, size: int) -> RingView:
        """
        Return a memoryview (or pair) for a direct read from the ring buffer.
        size_readable is the real size available (may be less than requested).
        wrap_around is True when the read spans the end of the buffer and mv2 is needed.
        """
        self._assert_is_reader("expose_reader_mem_view")
        write_pos = int(self.header[self.write_pos_index])
        read_pos = int(self.header[self.reader_pos_index])
        max_amount_readable = write_pos - read_pos
        assert max_amount_readable >= 0, (
            f"invariant violation: max_amount_readable < 0 "
            f"(write_pos={write_pos}, read_pos={read_pos}, "
            f"max_amount_readable={max_amount_readable})"
        )
        if max_amount_readable > self.ring_buffer_size:
            # This reader has fallen behind beyond one full ring, which can
            # happen for inactive/non-blocking or not-yet-ready readers while
            # writers continue to advance. Resync to the current writer and
            # treat the unread history as dropped.
            self.jump_to_writer()
            read_pos = int(self.header[self.reader_pos_index])
            max_amount_readable = 0
        if max_amount_readable >= size:
            size_readable = size
        else:
            size_readable = max_amount_readable
        reader_pos = self.int_to_pos(read_pos)
        if reader_pos + size_readable <= self.ring_buffer_size:
            mv1 = memoryview(self.ring_buffer[reader_pos: reader_pos + size_readable])
            mv2 = None
            wrap_around = False
        else:
            mv1 = memoryview(self.ring_buffer[reader_pos:])
            remaining = size_readable - (self.ring_buffer_size - reader_pos)
            mv2 = memoryview(self.ring_buffer[0:remaining])
            wrap_around = True

        return (mv1, mv2, size_readable, wrap_around)

    # ------------------------------------------------------------------ #
    # Simple copy helpers                                                  #
    # ------------------------------------------------------------------ #

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        """
        Copy src into the memoryview pair returned by expose_writer_mem_view.
        If src is larger than the allocated area the excess is discarded.
        """
        mv1, mv2, _, _ = writer_mem_view
        src_mv = memoryview(src).cast("B")
        total_dst = mv1.nbytes + (mv2.nbytes if mv2 is not None else 0)
        bytes_to_copy = min(src_mv.nbytes, total_dst)
        first_copy = min(bytes_to_copy, mv1.nbytes)
        if first_copy:
            mv1[:first_copy] = src_mv[:first_copy]
        if mv2 is not None and bytes_to_copy > first_copy:
            second_copy = bytes_to_copy - first_copy
            mv2[:second_copy] = src_mv[first_copy:first_copy + second_copy]

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """
        Copy from the memoryview pair returned by expose_reader_mem_view into dst.
        If dst is smaller than the available data the excess is discarded.
        """
        mv1, mv2, _, _ = reader_mem_view
        dst_mv = memoryview(dst).cast("B")
        total_src = mv1.nbytes + (mv2.nbytes if mv2 is not None else 0)
        bytes_to_copy = min(dst_mv.nbytes, total_src)
        first_copy = min(bytes_to_copy, mv1.nbytes)
        if first_copy:
            dst_mv[:first_copy] = mv1[:first_copy]
        if mv2 is not None and bytes_to_copy > first_copy:
            second_copy = bytes_to_copy - first_copy
            dst_mv[first_copy:first_copy + second_copy] = mv2[:second_copy]

    def write_array(self, arr: np.ndarray) -> int:
        """
        Zero-copy write of a numpy array into the ring as raw bytes.
        Returns bytes written, or 0 if insufficient space  caller decides.
        arr is any numpy array. dtype and shape are the caller's concern.
        The ring only moves bytes.
        """
        src = memoryview(arr).cast("B")
        nbytes = src.nbytes
        mv1, mv2, size_writeable, wrap_around = self.expose_writer_mem_view(nbytes)
        if size_writeable < nbytes:
            return 0
        self.simple_write((mv1, mv2, size_writeable, wrap_around), src)
        self.inc_writer_pos(nbytes)
        return nbytes

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read nbytes from the ring and interpret them as dtype.
        Returns an empty array of the correct dtype if insufficient data.
        Wrap-around case copies into a contiguous buffer first because
        np.frombuffer requires contiguous memory.
        dtype and shape are the caller's concern  the ring only moves bytes.
        """
        mv1, mv2, size_readable, wrap_around = self.expose_reader_mem_view(nbytes)
        if size_readable < nbytes:
            return np.empty(0, dtype=dtype)
        if not wrap_around:
            arr = np.frombuffer(mv1, dtype=dtype)
        else:
            buf = bytearray(size_readable)
            self.simple_read((mv1, mv2, size_readable, wrap_around), memoryview(buf))
            arr = np.frombuffer(buf, dtype=dtype)
        self.inc_reader_pos(size_readable)
        return arr
