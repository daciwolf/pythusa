"""Run with: pytest tests/test_ring_buffer_basic.py -q"""

import unittest
import uuid
import gc

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    np = None

try:
    from pythusa import SharedRingBuffer
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    SharedRingBuffer = None


@unittest.skipIf(np is None or SharedRingBuffer is None, "numpy/shared_ring_buffer dependencies are required")
class SharedRingBufferBasicTests(unittest.TestCase):
    def _release_mvs(self, *mvs):
        for mv in mvs:
            if mv is None:
                continue
            try:
                mv.release()
            except AttributeError:
                pass

    def _make_name(self) -> str:
        return f"rb{uuid.uuid4().hex[:10]}"

    def _make_ring(self, *, size=32, num_readers=1, reader=0):
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
        # SharedMemory.close() fails if exported pointers still exist.
        # This class keeps both a numpy header view and a payload memoryview alive.
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
    def _snapshot_case(label, size, write_pos, read_pos, req_size, got):
        mv1, mv2, n, wrap = got
        return (
            f"{label}: size={size}, write_pos={write_pos}, read_pos={read_pos}, request={req_size} -> "
            f"size_returned={n}, wrap={wrap}, len(mv1)={len(mv1)}, len(mv2)={(len(mv2) if mv2 is not None else 0)}"
        )

    def test_initial_header_and_sizes(self):
        ring = self._make_ring(size=64, num_readers=2, reader=0)

        self.assertEqual(ring.header_size, 8 * (6 + 2 * 3))
        self.assertEqual(ring.shared_mem_size, ring.header_size + 64)
        self.assertEqual(int(ring.header[0]), 64)
        self.assertEqual(int(ring.header[ring.num_readers_index]), 2)
        self.assertEqual(len(ring.ring_buffer), 64)

    def test_header_size_cache_aligned_rounds_up(self):
        name = self._make_name()
        ring = SharedRingBuffer(
            name=name,
            create=True,
            size=64,
            num_readers=2,
            reader=0,
            cache_align=True,
            cache_size=64,
        )
        self.addCleanup(self._cleanup_ring, ring)

        raw_header_size = 8 * (6 + 2 * 3)  # 96
        self.assertEqual(ring.header_size, 128)
        self.assertEqual(ring.header_size % 64, 0)
        self.assertGreaterEqual(ring.header_size, raw_header_size)
        self.assertEqual(len(ring.ring_buffer), 64)

    def test_header_size_cache_aligned_already_aligned(self):
        name = self._make_name()
        ring = SharedRingBuffer(
            name=name,
            create=True,
            size=64,
            num_readers=2,
            reader=0,
            cache_align=True,
            cache_size=32,
        )
        self.addCleanup(self._cleanup_ring, ring)

        self.assertEqual(ring.header_size, 96)
        self.assertEqual(ring.header_size % 32, 0)
        self.assertEqual(len(ring.ring_buffer), 64)

    def test_cache_aligned_invalid_cache_size_raises(self):
        with self.assertRaises(ValueError):
            SharedRingBuffer(
                name=self._make_name(),
                create=True,
                size=64,
                num_readers=1,
                reader=0,
                cache_align=True,
                cache_size=48,
            )
        with self.assertRaises(ValueError):
            SharedRingBuffer(
                name=self._make_name(),
                create=True,
                size=64,
                num_readers=1,
                reader=0,
                cache_align=True,
                cache_size=0,
            )

    def test_refresh_settings_default_to_existing_values(self):
        ring = self._make_ring(size=64, num_readers=1, reader=0)

        self.assertEqual(ring._min_reader_pos_refresh_interval, 64)
        self.assertEqual(ring._min_reader_pos_refresh_s, 0.005)

    def test_refresh_settings_accept_explicit_values(self):
        name = self._make_name()
        ring = SharedRingBuffer(
            name=name,
            create=True,
            size=64,
            num_readers=1,
            reader=0,
            min_reader_pos_refresh_interval=11,
            min_reader_pos_refresh_s=0.125,
        )
        self.addCleanup(self._cleanup_ring, ring)

        self.assertEqual(ring._min_reader_pos_refresh_interval, 11)
        self.assertEqual(ring._min_reader_pos_refresh_s, 0.125)

    def test_refresh_settings_validation_rejects_invalid_values(self):
        with self.assertRaisesRegex(TypeError, "min_reader_pos_refresh_interval"):
            SharedRingBuffer(
                name=self._make_name(),
                create=True,
                size=64,
                num_readers=1,
                reader=0,
                min_reader_pos_refresh_interval=1.5,
            )
        with self.assertRaisesRegex(ValueError, "min_reader_pos_refresh_interval"):
            SharedRingBuffer(
                name=self._make_name(),
                create=True,
                size=64,
                num_readers=1,
                reader=0,
                min_reader_pos_refresh_interval=0,
            )
        with self.assertRaisesRegex(TypeError, "min_reader_pos_refresh_s"):
            SharedRingBuffer(
                name=self._make_name(),
                create=True,
                size=64,
                num_readers=1,
                reader=0,
                min_reader_pos_refresh_s="fast",
            )
        with self.assertRaisesRegex(ValueError, "min_reader_pos_refresh_s"):
            SharedRingBuffer(
                name=self._make_name(),
                create=True,
                size=64,
                num_readers=1,
                reader=0,
                min_reader_pos_refresh_s=-0.1,
            )

    def test_update_and_increment_positions(self):
        ring = self._make_ring(size=64, num_readers=1, reader=0)

        ring.update_write_pos(10)
        ring.update_reader_pos(4)
        self.assertEqual(int(ring.get_write_pos()), 10)
        self.assertEqual(int(ring.header[ring.reader_pos_index]), 4)

        ring.inc_writer_pos(7)
        ring.inc_reader_pos(3)
        self.assertEqual(int(ring.header[ring.write_pos_index]), 17)
        self.assertEqual(int(ring.header[ring.reader_pos_index]), 7)

    def test_compute_max_amount_writable_single_reader(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)

        ring.update_write_pos(20)
        ring.update_reader_pos(5)
        self._mark_reader_alive(ring)
        writable = ring.compute_max_amount_writable()
        self.assertEqual(writable, 17)  # used = 20-5 = 15, free = 32-15 = 17
        self.assertEqual(int(ring.header[ring.max_amount_writable_index]), 17)

    def test_set_reader_active_controls_liveness_slot(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)

        ring.set_reader_active(True)
        self.assertTrue(ring.is_reader_active())
        self.assertEqual(int(ring.header[ring.reader_pos_index + 1]), 1)

        ring.set_reader_active(False)
        self.assertFalse(ring.is_reader_active())
        self.assertEqual(int(ring.header[ring.reader_pos_index + 1]), 0)

    def test_expose_writer_mem_view_contiguous(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)

        ring.update_write_pos(3)
        ring.update_reader_pos(0)
        ring.compute_max_amount_writable()

        mv1, mv2, n, wrap = ring.expose_writer_mem_view(8)
        self.assertEqual(n, 8)
        self.assertFalse(wrap)
        self.assertIsNotNone(mv1)
        self.assertIsNone(mv2)
        self.assertEqual(len(mv1), 8)
        self._release_mvs(mv1, mv2)

    def test_expose_writer_mem_view_wraparound(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)

        ring.update_write_pos(14)
        ring.update_reader_pos(6)  # used=8, writable=8
        ring.compute_max_amount_writable()

        mv1, mv2, n, wrap = ring.expose_writer_mem_view(6)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)  # [14..15]
        self.assertIsNotNone(mv2)
        self.assertEqual(len(mv2), 4)  # [0..3]
        self._release_mvs(mv1, mv2)

    def test_expose_reader_mem_view_contiguous(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)

        ring.update_reader_pos(4)
        ring.update_write_pos(15)

        mv1, mv2, n, wrap = ring.expose_reader_mem_view(8)
        self.assertEqual(n, 8)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 8)
        self.assertIsNone(mv2)
        self._release_mvs(mv1, mv2)

    def test_expose_reader_mem_view_wraparound_shape(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)

        # Logical positions: readable = 20 - 14 = 6; wrapped read starts at pos 14
        ring.update_reader_pos(14)
        ring.update_write_pos(20)

        mv1, mv2, n, wrap = ring.expose_reader_mem_view(6)
        self.assertEqual(n, 6)
        self.assertTrue(wrap)
        self.assertEqual(len(mv1), 2)  # bytes at [14..15]
        self.assertIsNotNone(mv2)
        self.assertEqual(len(mv2), 4)  # bytes at [0..3]
        self._release_mvs(mv1, mv2)

    def test_simple_write_contiguous(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes([0] * 16)
        ring.update_write_pos(3)
        ring.update_reader_pos(0)
        writer_mv = ring.expose_writer_mem_view(5)

        src = np.array([10, 11, 12, 13, 14], dtype=np.uint8)
        ring.simple_write(writer_mv, src)

        self.assertEqual(bytes(ring.ring_buffer[3:8]), b"\x0a\x0b\x0c\x0d\x0e")
        self._release_mvs(writer_mv[0], writer_mv[1])

    def test_simple_write_wrap(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes([0] * 16)
        ring.update_write_pos(14)
        ring.update_reader_pos(6)  # writable=8
        writer_mv = ring.expose_writer_mem_view(6)

        src = np.array([100, 101, 102, 103, 104, 105], dtype=np.uint8)
        ring.simple_write(writer_mv, src)

        self.assertEqual(bytes(ring.ring_buffer[14:16]), b"\x64\x65")
        self.assertEqual(bytes(ring.ring_buffer[0:4]), b"\x66\x67\x68\x69")
        self._release_mvs(writer_mv[0], writer_mv[1])

    def test_simple_write_truncates_when_src_shorter(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes([0] * 16)
        ring.update_write_pos(3)
        ring.update_reader_pos(0)
        writer_mv = ring.expose_writer_mem_view(5)

        src = np.array([7, 8, 9], dtype=np.uint8)
        ring.simple_write(writer_mv, src)

        self.assertEqual(bytes(ring.ring_buffer[3:6]), b"\x07\x08\x09")
        self.assertEqual(bytes(ring.ring_buffer[6:8]), b"\x00\x00")
        self._release_mvs(writer_mv[0], writer_mv[1])

    def test_simple_read_contiguous(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes(range(16))
        ring.update_reader_pos(4)
        ring.update_write_pos(12)
        reader_mv = ring.expose_reader_mem_view(5)

        dst = np.zeros(5, dtype=np.uint8)
        ring.simple_read(reader_mv, dst)

        self.assertEqual(dst.tolist(), [4, 5, 6, 7, 8])
        self._release_mvs(reader_mv[0], reader_mv[1])

    def test_simple_read_wrap(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes(range(16))
        ring.update_reader_pos(14)
        ring.update_write_pos(20)  # readable=6
        reader_mv = ring.expose_reader_mem_view(6)

        dst = np.zeros(6, dtype=np.uint8)
        ring.simple_read(reader_mv, dst)

        self.assertEqual(dst.tolist(), [14, 15, 0, 1, 2, 3])
        self._release_mvs(reader_mv[0], reader_mv[1])

    def test_simple_read_truncates_when_dst_shorter(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes(range(16))
        ring.update_reader_pos(14)
        ring.update_write_pos(20)  # readable=6, wrapped
        reader_mv = ring.expose_reader_mem_view(6)

        dst = np.zeros(3, dtype=np.uint8)
        ring.simple_read(reader_mv, dst)

        self.assertEqual(dst.tolist(), [14, 15, 0])
        self._release_mvs(reader_mv[0], reader_mv[1])

    def test_write_array_contiguous_advances_writer(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)
        ring.update_reader_pos(0)
        ring.update_write_pos(0)
        arr = np.array([1, 2, 3, 4], dtype=np.int16)

        written = ring.write_array(arr)

        self.assertEqual(written, arr.nbytes)
        self.assertEqual(int(ring.get_write_pos()), arr.nbytes)
        self.assertEqual(bytes(ring.ring_buffer[:arr.nbytes]), arr.tobytes())

    def test_write_array_wraparound_writes_both_segments(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes([0] * 16)
        ring.update_reader_pos(6)
        ring.update_write_pos(14)
        arr = np.array([100, 101, 102, 103, 104, 105], dtype=np.uint8)

        written = ring.write_array(arr)

        self.assertEqual(written, arr.nbytes)
        self.assertEqual(bytes(ring.ring_buffer[14:16]), b"\x64\x65")
        self.assertEqual(bytes(ring.ring_buffer[0:4]), b"\x66\x67\x68\x69")

    def test_write_array_returns_zero_when_space_is_insufficient(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.ring_buffer[:] = bytes([0xEE] * 16)
        ring.update_reader_pos(0)
        ring.update_write_pos(12)
        self._mark_reader_alive(ring)
        arr = np.arange(8, dtype=np.uint8)

        written = ring.write_array(arr)

        self.assertEqual(written, 0)
        self.assertEqual(int(ring.get_write_pos()), 12)
        self.assertEqual(bytes(ring.ring_buffer), bytes([0xEE] * 16))

    def test_read_array_contiguous_returns_expected_dtype(self):
        ring = self._make_ring(size=32, num_readers=1, reader=0)
        arr = np.array([10, 20, 30, 40], dtype=np.int16)
        ring.ring_buffer[:arr.nbytes] = arr.tobytes()
        ring.update_reader_pos(0)
        ring.update_write_pos(arr.nbytes)

        got = ring.read_array(arr.nbytes, dtype=np.int16)

        self.assertTrue(np.array_equal(got, arr))
        self.assertEqual(got.dtype, np.int16)
        self.assertEqual(int(ring.header[ring.reader_pos_index]), arr.nbytes)

    def test_read_array_wraparound_returns_contiguous_array(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        arr = np.array([14, 15, 16, 17, 18, 19], dtype=np.uint8)
        ring.ring_buffer[14:16] = arr[:2].tobytes()
        ring.ring_buffer[0:4] = arr[2:].tobytes()
        ring.update_reader_pos(14)
        ring.update_write_pos(20)

        got = ring.read_array(arr.nbytes, dtype=np.uint8)

        self.assertTrue(np.array_equal(got, arr))
        self.assertEqual(int(ring.header[ring.reader_pos_index]), 20)

    def test_read_array_returns_empty_when_data_is_insufficient(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.update_reader_pos(0)
        ring.update_write_pos(2)

        got = ring.read_array(4, dtype=np.uint16)

        self.assertEqual(got.size, 0)
        self.assertEqual(got.dtype, np.uint16)
        self.assertEqual(int(ring.header[ring.reader_pos_index]), 0)

    def test_expose_writer_mem_view_recomputes_writable_each_call(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)
        ring.update_reader_pos(0)

        first = ring.expose_writer_mem_view(10)
        self.assertEqual(first[2], 10)
        ring.inc_writer_pos(10)

        second = ring.expose_writer_mem_view(10)
        # After writing 10 bytes into a 16-byte ring with reader at 0, only 6 bytes remain writable.
        self.assertEqual(second[2], 6)

        self._release_mvs(first[0], first[1], second[0], second[1])

    def test_reference_case_table_with_expected_outputs(self):
        # This test prints each canonical case as text so output is easy to inspect.
        cases = [
            {
                "label": "writer-contiguous",
                "size": 32,
                "write_pos": 3,
                "read_pos": 0,
                "request": 8,
                "fn": "writer",
                "expect": (8, False, 8, 0),
            },
            {
                "label": "writer-wrap",
                "size": 16,
                "write_pos": 14,
                "read_pos": 6,
                "request": 6,
                "fn": "writer",
                "expect": (6, True, 2, 4),
            },
            {
                "label": "reader-contiguous",
                "size": 32,
                "write_pos": 15,
                "read_pos": 4,
                "request": 8,
                "fn": "reader",
                "expect": (8, False, 8, 0),
            },
            {
                "label": "reader-wrap",
                "size": 16,
                "write_pos": 20,
                "read_pos": 14,
                "request": 6,
                "fn": "reader",
                "expect": (6, True, 2, 4),
            },
        ]

        for case in cases:
            with self.subTest(case=case["label"]):
                ring = self._make_ring(size=case["size"], num_readers=1, reader=0)
                ring.update_write_pos(case["write_pos"])
                ring.update_reader_pos(case["read_pos"])

                if case["fn"] == "writer":
                    ring.compute_max_amount_writable()
                    got = ring.expose_writer_mem_view(case["request"])
                else:
                    got = ring.expose_reader_mem_view(case["request"])

                msg = self._snapshot_case(
                    case["label"],
                    case["size"],
                    case["write_pos"],
                    case["read_pos"],
                    case["request"],
                    got,
                )
                print(msg)

                mv1, mv2, n, wrap = got
                exp_n, exp_wrap, exp_mv1_len, exp_mv2_len = case["expect"]
                self.assertEqual(n, exp_n, msg)
                self.assertEqual(wrap, exp_wrap, msg)
                self.assertEqual(len(mv1), exp_mv1_len, msg)
                self.assertEqual((len(mv2) if mv2 is not None else 0), exp_mv2_len, msg)
                self._release_mvs(mv1, mv2)

    def test_compute_max_amount_writable_reader_ahead_is_treated_as_empty_used(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)

        ring.update_write_pos(5)
        ring.update_reader_pos(9)  # impossible state for monotonic counters
        writable = ring.compute_max_amount_writable()
        # Current implementation uses min reader <= writer when computing used.
        self.assertEqual(writable, 16)

    def test_reader_expose_stale_reader_jumps_to_writer_when_unread_exceeds_ring_size(self):
        ring = self._make_ring(size=16, num_readers=1, reader=0)

        ring.update_reader_pos(0)
        ring.update_write_pos(32)  # impossible if writer obeys max writable
        mv1, mv2, n, wrap = ring.expose_reader_mem_view(1)

        self.assertEqual(int(ring.header[ring.reader_pos_index]), 32)
        self.assertEqual(n, 0)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 0)
        self.assertIsNone(mv2)
        self._release_mvs(mv1, mv2)


if __name__ == "__main__":
    unittest.main()
