"""Run with: pytest tests/test_ring_buffer_edge_cases.py -q"""

import gc
import unittest
import uuid

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    np = None

try:
    from pythusa import SharedRingBuffer
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    SharedRingBuffer = None


@unittest.skipIf(np is None or SharedRingBuffer is None, "numpy/shared_ring_buffer dependencies are required")
class SharedRingBufferEdgeCaseTests(unittest.TestCase):
    def _make_name(self) -> str:
        return f"re{uuid.uuid4().hex[:10]}"

    def _make_ring(self, *, size=16, num_readers=1, reader=0):
        name = self._make_name()
        ring = SharedRingBuffer(name=name, create=True, size=size, num_readers=num_readers, reader=reader)
        self.addCleanup(self._cleanup_ring, ring)
        return ring

    @staticmethod
    def _mark_reader_alive(ring: SharedRingBuffer, reader_index: int | None = None):
        if reader_index is None:
            reader_index = ring.reader
        slot = 6 + (reader_index * 3)
        ring.header[slot + 1] = 1

    @staticmethod
    def _cleanup_ring(ring: SharedRingBuffer):
        try:
            ring.ring_buffer.release()
        except Exception:
            pass
        try:
            del ring.ring_buffer
        except Exception:
            pass
        try:
            ring.header = None
        except Exception:
            pass
        try:
            del ring.header
        except Exception:
            pass
        gc.collect()
        try:
            ring.close()
        finally:
            try:
                ring.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _release_mvs(*mvs):
        for mv in mvs:
            if mv is None:
                continue
            try:
                mv.release()
            except Exception:
                pass

    def _set_reader_slot_pos(self, ring: SharedRingBuffer, reader_index: int, value: int):
        slot = 6 + (reader_index * 3)
        ring.header[slot] = value
        ring.header[slot + 1] = 1

    def test_int_to_pos_wraps_large_values(self):
        ring = self._make_ring(size=16)
        self.assertEqual(ring.int_to_pos(0), 0)
        self.assertEqual(ring.int_to_pos(16), 0)
        self.assertEqual(ring.int_to_pos(17), 1)
        self.assertEqual(ring.int_to_pos((16 * 100) + 7), 7)

    def test_expose_writer_zero_request_returns_empty(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(5)
        ring.update_reader_pos(0)
        mv1, mv2, n, wrap = ring.expose_writer_mem_view(0)
        self.assertEqual(n, 0)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 0)
        self.assertIsNone(mv2)
        self._release_mvs(mv1, mv2)

    def test_expose_reader_zero_request_returns_empty(self):
        ring = self._make_ring(size=16)
        ring.update_reader_pos(3)
        ring.update_write_pos(8)
        mv1, mv2, n, wrap = ring.expose_reader_mem_view(0)
        self.assertEqual(n, 0)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 0)
        self.assertIsNone(mv2)
        self._release_mvs(mv1, mv2)

    def test_expose_writer_clamps_when_request_exceeds_writable(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(12)
        ring.update_reader_pos(0)  # used=12, writable=4
        self._mark_reader_alive(ring)
        mv1, mv2, n, wrap = ring.expose_writer_mem_view(100)
        self.assertEqual(n, 4)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 4)
        self.assertEqual((len(mv2) if mv2 is not None else 0), 0)
        self._release_mvs(mv1, mv2)

    def test_expose_reader_clamps_when_request_exceeds_readable_wrap(self):
        ring = self._make_ring(size=16)
        ring.update_reader_pos(14)
        ring.update_write_pos(20)  # readable=6
        mv1, mv2, n, wrap = ring.expose_reader_mem_view(100)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)
        self.assertEqual((len(mv2) if mv2 is not None else 0), 4)
        self._release_mvs(mv1, mv2)

    def test_expose_writer_many_wraps_positions(self):
        ring = self._make_ring(size=16)
        write_pos = (16 * 100) + 14
        read_pos = write_pos - 8  # writable=8
        ring.update_write_pos(write_pos)
        ring.update_reader_pos(read_pos)
        mv1, mv2, n, wrap = ring.expose_writer_mem_view(6)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)
        self.assertEqual((len(mv2) if mv2 is not None else 0), 4)
        self._release_mvs(mv1, mv2)

    def test_expose_reader_many_wraps_positions(self):
        ring = self._make_ring(size=16)
        read_pos = (16 * 100) + 14
        write_pos = read_pos + 6
        ring.update_reader_pos(read_pos)
        ring.update_write_pos(write_pos)
        mv1, mv2, n, wrap = ring.expose_reader_mem_view(6)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)
        self.assertEqual((len(mv2) if mv2 is not None else 0), 4)
        self._release_mvs(mv1, mv2)

    def test_simple_write_accepts_bytes_like_inputs(self):
        ring = self._make_ring(size=16)
        ring.ring_buffer[:] = bytes([0] * 16)
        ring.update_write_pos(2)
        ring.update_reader_pos(0)
        writer_mv = ring.expose_writer_mem_view(4)

        ring.simple_write(writer_mv, b"\x01\x02\x03\x04")
        self.assertEqual(bytes(ring.ring_buffer[2:6]), b"\x01\x02\x03\x04")

        ring.simple_write(writer_mv, memoryview(b"\x0a\x0b\x0c\x0d"))
        self.assertEqual(bytes(ring.ring_buffer[2:6]), b"\x0a\x0b\x0c\x0d")
        self._release_mvs(writer_mv[0], writer_mv[1])

    def test_simple_read_accepts_bytearray_destination(self):
        ring = self._make_ring(size=16)
        ring.ring_buffer[:] = bytes(range(16))
        ring.update_reader_pos(5)
        ring.update_write_pos(11)
        reader_mv = ring.expose_reader_mem_view(4)
        dst = bytearray(4)
        ring.simple_read(reader_mv, dst)
        self.assertEqual(bytes(dst), b"\x05\x06\x07\x08")
        self._release_mvs(reader_mv[0], reader_mv[1])

    def test_simple_read_leaves_destination_tail_unchanged_when_larger(self):
        ring = self._make_ring(size=16)
        ring.ring_buffer[:] = bytes(range(16))
        ring.update_reader_pos(14)
        ring.update_write_pos(20)  # readable=6
        reader_mv = ring.expose_reader_mem_view(6)

        dst = bytearray([0xAA] * 10)
        ring.simple_read(reader_mv, dst)
        self.assertEqual(bytes(dst[:6]), b"\x0e\x0f\x00\x01\x02\x03")
        self.assertEqual(bytes(dst[6:]), b"\xaa\xaa\xaa\xaa")
        self._release_mvs(reader_mv[0], reader_mv[1])

    def test_simple_write_leaves_ring_tail_unchanged_when_src_shorter_wrap(self):
        ring = self._make_ring(size=16)
        ring.ring_buffer[:] = bytes([0xEE] * 16)
        ring.update_write_pos(14)
        ring.update_reader_pos(6)  # writable=8
        writer_mv = ring.expose_writer_mem_view(6)  # 2 + 4

        ring.simple_write(writer_mv, b"\x01\x02\x03")  # shorter than 6
        self.assertEqual(bytes(ring.ring_buffer[14:16]), b"\x01\x02")
        self.assertEqual(bytes(ring.ring_buffer[0:1]), b"\x03")
        self.assertEqual(bytes(ring.ring_buffer[1:4]), b"\xee\xee\xee")
        self._release_mvs(writer_mv[0], writer_mv[1])

    def test_compute_max_force_rescan_overrides_stale_cache(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(40)
        ring.update_reader_pos(30)  # true used=10, writable=6
        self._mark_reader_alive(ring)
        ring._min_reader_pos_cache = 0  # stale/wrong on purpose
        ring._reader_positions_dirty = False
        ring._writes_since_min_scan = 0
        writable = ring.compute_max_amount_writable(force_rescan=True)
        self.assertEqual(writable, 6)

    def test_compute_max_recovers_when_cached_used_exceeds_size(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(100)
        ring.update_reader_pos(95)  # true used=5, writable=11
        self._mark_reader_alive(ring)
        ring._min_reader_pos_cache = 80  # stale cache makes used=20 (> size)
        ring._reader_positions_dirty = False
        ring._writes_since_min_scan = 0
        writable = ring.compute_max_amount_writable()
        self.assertEqual(writable, 11)

    def test_compute_max_periodic_rescan_picks_external_reader_update(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(20)
        ring.update_reader_pos(10)  # used=10, writable=6
        self._mark_reader_alive(ring)
        self.assertEqual(ring.compute_max_amount_writable(), 6)

        # Simulate a different process advancing this reader directly in shared header.
        ring.header[ring.reader_pos_index] = 19
        ring._reader_positions_dirty = False
        ring._writes_since_min_scan = ring._min_reader_pos_refresh_interval

        self.assertEqual(ring.compute_max_amount_writable(), 15)

    def test_compute_max_multi_reader_uses_global_min(self):
        ring = self._make_ring(size=16, num_readers=3, reader=0)
        ring.update_write_pos(105)
        self._set_reader_slot_pos(ring, 0, 100)
        self._set_reader_slot_pos(ring, 1, 90)
        self._set_reader_slot_pos(ring, 2, 95)
        ring._reader_positions_dirty = True
        writable = ring.compute_max_amount_writable()
        self.assertEqual(writable, 1)  # used=15

    def test_compute_max_negative_used_asserts(self):
        ring = self._make_ring(size=16)
        ring.update_write_pos(5)
        ring._min_reader_pos_cache = 10
        ring._reader_positions_dirty = False
        ring._writes_since_min_scan = 0
        with self.assertRaises(AssertionError):
            ring.compute_max_amount_writable()


if __name__ == "__main__":
    unittest.main()
