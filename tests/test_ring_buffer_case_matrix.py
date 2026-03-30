"""Run with: pytest tests/test_ring_buffer_case_matrix.py -q"""

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


def _u64_list(arr) -> list[int]:
    return [int(x) for x in arr.tolist()]


def _hex_bytes(data: bytes, group=2) -> str:
    if not data:
        return ""
    hx = data.hex()
    return " ".join(hx[i:i + (group * 2)] for i in range(0, len(hx), group * 2))


@unittest.skipIf(np is None or SharedRingBuffer is None, "numpy/shared_ring_buffer dependencies are required")
class SharedRingBufferCaseMatrixTests(unittest.TestCase):
    def _make_name(self) -> str:
        return f"rc{uuid.uuid4().hex[:10]}"

    def _make_ring(self, *, size=16, num_readers=1, reader=0):
        name = self._make_name()
        ring = SharedRingBuffer(name=name, create=True, size=size, num_readers=num_readers, reader=reader)
        self.addCleanup(self._cleanup_ring, ring)
        # Seed payload with deterministic bytes so snapshots are interpretable.
        ring.ring_buffer[:] = bytes(i % 256 for i in range(size))
        return ring

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

    def _set_reader_pos(self, ring: SharedRingBuffer, reader_index: int, value: int):
        slot = 6 + (reader_index * 3)
        ring.header[slot] = value
        ring.header[slot + 1] = 1

    def _snapshot(self, label: str, ring: SharedRingBuffer, req: int, got, mode: str, expected: tuple[int, bool, int, int]) -> str:
        mv1, mv2, size_ret, wrap = got
        header_u64 = _u64_list(ring.header)
        header_raw = bytes(ring.buf[:ring.header_size])
        payload_raw = bytes(ring.buf[ring.header_size:ring.header_size + ring.ring_buffer_size])
        mv1_bytes = bytes(mv1)
        mv2_bytes = bytes(mv2) if mv2 is not None else b""
        exp_size, exp_wrap, exp_mv1_len, exp_mv2_len = expected
        write_pos = int(ring.header[ring.write_pos_index])
        read_pos = int(ring.header[ring.reader_pos_index])
        return (
            f"\nCASE: {label}\n"
            f"mode={mode} request={req}\n"
            f"ring(size={ring.ring_buffer_size}, header_size={ring.header_size}, shared_mem_size={ring.shared_mem_size})\n"
            f"header_u64={header_u64}\n"
            f"header_raw_hex={_hex_bytes(header_raw)}\n"
            f"payload_raw_hex={_hex_bytes(payload_raw)}\n"
            f"logical_positions(write={write_pos}, read={read_pos}, write_idx={ring.int_to_pos(write_pos)}, read_idx={ring.int_to_pos(read_pos)})\n"
            f"actual(size={size_ret}, wrap={wrap}, len(mv1)={len(mv1)}, len(mv2)={(len(mv2) if mv2 is not None else 0)})\n"
            f"actual_mv1_hex={_hex_bytes(mv1_bytes)}\n"
            f"actual_mv2_hex={_hex_bytes(mv2_bytes)}\n"
            f"expected(size={exp_size}, wrap={exp_wrap}, len(mv1)={exp_mv1_len}, len(mv2)={exp_mv2_len})"
        )

    @staticmethod
    def _expected_writer(ring_size: int, write_pos: int, min_reader_pos: int, request: int) -> tuple[int, bool, int, int]:
        used = write_pos - min_reader_pos
        writable = ring_size - used
        n = request if request <= writable else writable
        idx = write_pos % ring_size
        if idx + n <= ring_size:
            return (n, False, n, 0)
        mv1 = ring_size - idx
        mv2 = n - mv1
        return (n, True, mv1, mv2)

    @staticmethod
    def _expected_reader(ring_size: int, write_pos: int, read_pos: int, request: int) -> tuple[int, bool, int, int]:
        readable = write_pos - read_pos
        n = request if request <= readable else readable
        idx = read_pos % ring_size
        if idx + n <= ring_size:
            return (n, False, n, 0)
        mv1 = ring_size - idx
        mv2 = n - mv1
        return (n, True, mv1, mv2)

    def test_case_matrix_writer_and_reader_with_memory_snapshots(self):
        # Writer side matrix
        writer_cases = [
            # label, size, write_pos, read_pos, request
            ("writer-zero", 16, 3, 0, 0),
            ("writer-contiguous-small", 16, 3, 0, 5),
            ("writer-exact-end-boundary", 16, 12, 0, 4),
            ("writer-wrap", 16, 14, 6, 6),
            ("writer-clamp-to-writable", 16, 12, 0, 8),   # writable=4, request=8 -> clamp to 4
            ("writer-clamp-small-free", 16, 14, 13, 8),   # writable=15, request=8 -> no clamp
            ("writer-full", 16, 16, 0, 5),                # used=16 -> writable=0
        ]

        for label, size, write_pos, read_pos, request in writer_cases:
            with self.subTest(case=label):
                ring = self._make_ring(size=size, num_readers=1, reader=0)
                ring.update_write_pos(write_pos)
                self._set_reader_pos(ring, 0, read_pos)
                # This matrix mutates reader slots directly to build exact
                # header states. Force a rescan so expectations are based on
                # the current shared header rather than the writer's cache.
                ring.compute_max_amount_writable(force_rescan=True)

                got = ring.expose_writer_mem_view(request)
                expected = self._expected_writer(size, write_pos, read_pos, request)
                msg = self._snapshot(label, ring, request, got, "writer", expected)
                print(msg)

                mv1, mv2, n, wrap = got
                exp_n, exp_wrap, exp_mv1_len, exp_mv2_len = expected
                self.assertEqual(n, exp_n, msg)
                self.assertEqual(wrap, exp_wrap, msg)
                self.assertEqual(len(mv1), exp_mv1_len, msg)
                self.assertEqual((len(mv2) if mv2 is not None else 0), exp_mv2_len, msg)
                self._release_mvs(mv1, mv2)

        # Reader side matrix
        reader_cases = [
            # label, size, write_pos, read_pos, request
            ("reader-zero", 16, 8, 4, 0),
            ("reader-empty", 16, 8, 8, 5),
            ("reader-contiguous-small", 16, 12, 4, 5),
            ("reader-clamp-to-readable", 16, 12, 4, 20),
            ("reader-exact-end-boundary", 16, 20, 12, 4),
            ("reader-wrap", 16, 20, 14, 6),
            ("reader-wrap-small-request", 16, 21, 14, 3),  # catches second-slice sizing logic
        ]

        for label, size, write_pos, read_pos, request in reader_cases:
            with self.subTest(case=label):
                ring = self._make_ring(size=size, num_readers=1, reader=0)
                ring.update_write_pos(write_pos)
                ring.update_reader_pos(read_pos)

                got = ring.expose_reader_mem_view(request)
                expected = self._expected_reader(size, write_pos, read_pos, request)
                msg = self._snapshot(label, ring, request, got, "reader", expected)
                print(msg)

                mv1, mv2, n, wrap = got
                exp_n, exp_wrap, exp_mv1_len, exp_mv2_len = expected
                self.assertEqual(n, exp_n, msg)
                self.assertEqual(wrap, exp_wrap, msg)
                self.assertEqual(len(mv1), exp_mv1_len, msg)
                self.assertEqual((len(mv2) if mv2 is not None else 0), exp_mv2_len, msg)
                self._release_mvs(mv1, mv2)

    def test_compute_max_amount_writable_multi_reader_min_reader_selected(self):
        ring = self._make_ring(size=32, num_readers=3, reader=0)
        ring.update_write_pos(40)
        self._set_reader_pos(ring, 0, 35)
        self._set_reader_pos(ring, 1, 22)
        self._set_reader_pos(ring, 2, 30)

        writable = ring.compute_max_amount_writable(force_rescan=True)
        # min_reader_pos=22 -> used=18 -> writable=14
        self.assertEqual(writable, 14)
        self.assertEqual(int(ring.header[ring.max_amount_writable_index]), 14)

        print(
            "\nCASE: multi-reader-min-selection\n"
            f"header_u64={_u64_list(ring.header)}\n"
            f"expected(min_reader_pos=22, used=18, writable=14)\n"
            f"actual(writable={writable})"
        )

    def test_reader_invariant_cases(self):
        # unread < 0
        ring1 = self._make_ring(size=16, num_readers=1, reader=0)
        ring1.update_write_pos(5)
        ring1.update_reader_pos(9)
        with self.assertRaises(AssertionError):
            ring1.expose_reader_mem_view(1)

        # unread > size -> stale reader resyncs to writer
        ring2 = self._make_ring(size=16, num_readers=1, reader=0)
        ring2.update_write_pos(33)
        ring2.update_reader_pos(0)
        mv1, mv2, n, wrap = ring2.expose_reader_mem_view(1)

        self.assertEqual(int(ring2.header[ring2.reader_pos_index]), 33)
        self.assertEqual(n, 0)
        self.assertFalse(wrap)
        self.assertEqual(len(mv1), 0)
        self.assertIsNone(mv2)
        self._release_mvs(mv1, mv2)


if __name__ == "__main__":
    unittest.main()
