"""Run with: pytest tests/test_ring_buffer_contract.py -q"""

from __future__ import annotations

import gc
import time
import unittest
import uuid
import weakref
from dataclasses import dataclass
from multiprocessing import shared_memory
from unittest.mock import patch

try:
    import pythusa._buffers.ring as shared_ring_buffer_module
except ModuleNotFoundError:  # pragma: no cover - environment dependency
    shared_ring_buffer_module = None

SharedRingBuffer = None if shared_ring_buffer_module is None else getattr(shared_ring_buffer_module, "SharedRingBuffer", None)
RingSpec = None if shared_ring_buffer_module is None else getattr(
    shared_ring_buffer_module,
    "RingSpec",
    getattr(shared_ring_buffer_module, "RingSpec", None),
)

NO_READER = -1 if SharedRingBuffer is None else SharedRingBuffer._NO_READER


@dataclass
class RingPair:
    creator: SharedRingBuffer
    peer: SharedRingBuffer
    probe: shared_memory.SharedMemory


@unittest.skipIf(SharedRingBuffer is None or RingSpec is None, "shared_ring_buffer dependencies are required")
class SharedRingBufferContractTests(unittest.TestCase):
    def _make_name(self, prefix: str = "rb") -> str:
        return f"{prefix}{uuid.uuid4().hex[:20]}"

    @staticmethod
    def _reader_slot(reader: int) -> int:
        return 6 + (reader * 3)

    @staticmethod
    def _release_mem_views(*views: memoryview | None) -> None:
        for view in views:
            if view is None:
                continue
            try:
                view.release()
            except Exception:
                pass

    @staticmethod
    def _drop_local_views(ring: SharedRingBuffer | None) -> None:
        if ring is None:
            return
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

    def _cleanup_ring(self, ring: SharedRingBuffer | None, *, unlink: bool) -> None:
        if ring is None:
            return
        self._drop_local_views(ring)
        gc.collect()
        try:
            ring.close()
        except Exception:
            pass
        if unlink:
            try:
                ring.unlink()
            except FileNotFoundError:
                pass

    def _set_reader_state(
        self,
        ring: SharedRingBuffer,
        reader: int,
        *,
        pos: int | None = None,
        alive: int | None = None,
    ) -> None:
        slot = self._reader_slot(reader)
        if pos is not None:
            ring.header[slot] = pos
        if alive is not None:
            ring.header[slot + 1] = alive
        if hasattr(ring, "_reader_positions_dirty"):
            ring._reader_positions_dirty = True

    def _make_pair(
        self,
        *,
        size: int = 16,
        num_readers: int = 1,
        creator_reader: int = NO_READER,
        peer_reader: int = 0,
        cache_align: bool = False,
        cache_size: int = 64,
    ) -> RingPair:
        name = self._make_name()
        creator = SharedRingBuffer(
            name=name,
            create=True,
            size=size,
            num_readers=num_readers,
            reader=creator_reader,
            cache_align=cache_align,
            cache_size=cache_size,
        )
        peer = SharedRingBuffer(
            name=name,
            create=False,
            size=size,
            num_readers=num_readers,
            reader=peer_reader,
            cache_align=cache_align,
            cache_size=cache_size,
        )
        probe = shared_memory.SharedMemory(name=name, create=False)

        self.addCleanup(self._cleanup_ring, creator, unlink=True)
        self.addCleanup(self._cleanup_ring, peer, unlink=False)
        self.addCleanup(probe.close)
        return RingPair(creator=creator, peer=peer, probe=probe)

    def _collect_until_gone(self, ref: weakref.ReferenceType[SharedRingBuffer], attempts: int = 8) -> None:
        for _ in range(attempts):
            if ref() is None:
                return
            gc.collect()
            time.sleep(0.01)
        self.assertIsNone(ref())

    def test_construction_reader_and_writer_slots(self) -> None:
        pair = self._make_pair(num_readers=3, peer_reader=2)
        self.assertIsNone(pair.creator.reader_pos_index)
        self.assertEqual(pair.peer.reader_pos_index, self._reader_slot(2))
        self.assertEqual(pair.probe.name, pair.creator.name)

    def test_cache_alignment_accepts_valid_sizes(self) -> None:
        cases = [
            (32, 96),
            (64, 128),
        ]
        for cache_size, expected_header_size in cases:
            with self.subTest(cache_size=cache_size):
                pair = self._make_pair(
                    size=64,
                    num_readers=2,
                    cache_align=True,
                    cache_size=cache_size,
                )
                self.assertEqual(pair.creator.header_size, expected_header_size)
                self.assertEqual(pair.creator.header_size % cache_size, 0)

    def test_cache_alignment_rejects_invalid_sizes(self) -> None:
        for cache_size in (0, 48):
            with self.subTest(cache_size=cache_size):
                with self.assertRaises(ValueError):
                    SharedRingBuffer(
                        name=self._make_name(),
                        create=True,
                        size=32,
                        num_readers=1,
                        reader=NO_READER,
                        cache_align=True,
                        cache_size=cache_size,
                    )

    def test_shared_memory_spec_rejects_invalid_size_and_reader_count(self) -> None:
        cases = [
            (0, 1),
            (-1, 1),
            (16, 0),
            (16, -1),
        ]
        for size, num_readers in cases:
            with self.subTest(size=size, num_readers=num_readers):
                with self.assertRaises(ValueError):
                    RingSpec(name="spec", size=size, num_readers=num_readers)

    def test_reader_only_methods_raise_on_writer_instances(self) -> None:
        pair = self._make_pair()
        cases = [
            ("update_reader_pos", (1,)),
            ("inc_reader_pos", (1,)),
            ("expose_reader_mem_view", (1,)),
            ("jump_to_writer", ()),
        ]
        for method_name, args in cases:
            with self.subTest(method=method_name):
                with self.assertRaises(RuntimeError):
                    getattr(pair.creator, method_name)(*args)

    def test_reader_only_methods_work_on_reader_instances(self) -> None:
        pair = self._make_pair()
        pair.creator.update_write_pos(5)

        pair.peer.update_reader_pos(1)
        pair.peer.inc_reader_pos(1)
        self.assertEqual(int(pair.peer.header[pair.peer.reader_pos_index]), 2)

        mv1, mv2, size_readable, wrap_around = pair.peer.expose_reader_mem_view(2)
        self.assertEqual(size_readable, 2)
        self.assertFalse(wrap_around)
        self._release_mem_views(mv1, mv2)

        pair.peer.jump_to_writer()
        self.assertEqual(int(pair.peer.header[pair.peer.reader_pos_index]), 5)

    def test_reader_context_manager_updates_alive_flag(self) -> None:
        pair = self._make_pair()
        alive_index = pair.peer.reader_pos_index + 1

        self.assertEqual(int(pair.creator.header[alive_index]), 0)
        pair.peer.__enter__()
        self.assertEqual(int(pair.creator.header[alive_index]), 1)
        pair.peer.__exit__(None, None, None)
        self.assertEqual(int(pair.creator.header[alive_index]), 0)

    def test_writer_context_manager_does_not_touch_alive_slots(self) -> None:
        pair = self._make_pair()
        alive_index = self._reader_slot(0) + 1
        pair.creator.header[alive_index] = 7

        pair.creator.__enter__()
        pair.creator.__exit__(None, None, None)

        self.assertEqual(int(pair.peer.header[alive_index]), 7)

    def test_writer_position_arithmetic_updates_header(self) -> None:
        pair = self._make_pair()
        pair.creator.update_write_pos(10)
        pair.creator.inc_writer_pos(7)

        self.assertEqual(int(pair.creator.header[pair.creator.write_pos_index]), 17)
        self.assertEqual(int(pair.creator.get_write_pos()), 17)

    def test_reader_position_arithmetic_updates_header(self) -> None:
        pair = self._make_pair()
        pair.peer.update_reader_pos(4)
        pair.peer.inc_reader_pos(3)

        self.assertEqual(int(pair.peer.header[pair.peer.reader_pos_index]), 7)
        self.assertEqual(int(pair.peer.reader_pos), 7)

    def test_int_to_pos_wraps_at_ring_boundaries(self) -> None:
        pair = self._make_pair(size=16)
        cases = [
            (0, 0),
            (16, 0),
            (17, 1),
            (39, 7),
            ((16 * 100) + 5, 5),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(pair.creator.int_to_pos(value), expected)

    def test_compute_max_amount_writable_fresh_buffer_all_writable(self) -> None:
        pair = self._make_pair(size=16)
        self.assertEqual(pair.creator.compute_max_amount_writable(), 16)
        self.assertEqual(int(pair.creator.header[pair.creator.max_amount_writable_index]), 16)

    def test_compute_max_amount_writable_respects_alive_reader(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.update_write_pos(5)
        self._set_reader_state(pair.creator, 0, pos=0, alive=1)

        self.assertEqual(pair.creator.compute_max_amount_writable(), 11)
        self.assertEqual(int(pair.creator.header[pair.creator.max_amount_writable_index]), 11)

    def test_compute_max_amount_writable_force_rescan_ignores_warm_cache(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.update_write_pos(10)
        self._set_reader_state(pair.creator, 0, pos=4, alive=1)
        pair.creator._min_reader_pos_cache = 4
        pair.creator._reader_positions_dirty = False
        pair.creator._writes_since_min_scan = 0
        pair.creator._last_min_scan_t = time.perf_counter()

        with patch.object(pair.creator, "_scan_min_reader_pos", wraps=pair.creator._scan_min_reader_pos) as scan:
            writable = pair.creator.compute_max_amount_writable(force_rescan=True)

        self.assertEqual(writable, 10)
        self.assertEqual(scan.call_count, 1)

    def test_compute_max_amount_writable_rescans_when_cached_used_exceeds_ring_size(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.update_write_pos(100)
        self._set_reader_state(pair.creator, 0, pos=95, alive=1)
        pair.creator._min_reader_pos_cache = 80
        pair.creator._reader_positions_dirty = False
        pair.creator._writes_since_min_scan = 0
        pair.creator._last_min_scan_t = time.perf_counter()

        with patch.object(pair.creator, "_scan_min_reader_pos", wraps=pair.creator._scan_min_reader_pos) as scan:
            writable = pair.creator.compute_max_amount_writable()

        self.assertEqual(writable, 11)
        self.assertGreaterEqual(scan.call_count, 1)

    def test_compute_max_amount_writable_asserts_on_impossible_state(self) -> None:
        negative_used = self._make_pair(size=16)
        negative_used.creator.update_write_pos(5)
        negative_used.creator._min_reader_pos_cache = 10
        negative_used.creator._reader_positions_dirty = False
        negative_used.creator._writes_since_min_scan = 0
        negative_used.creator._last_min_scan_t = time.perf_counter()
        with self.assertRaisesRegex(AssertionError, "used < 0"):
            negative_used.creator.compute_max_amount_writable()

        too_much_used = self._make_pair(size=16)
        too_much_used.creator.update_write_pos(40)
        self._set_reader_state(too_much_used.creator, 0, pos=0, alive=1)
        too_much_used.creator._min_reader_pos_cache = 0
        too_much_used.creator._reader_positions_dirty = False
        too_much_used.creator._writes_since_min_scan = 0
        too_much_used.creator._last_min_scan_t = time.perf_counter()
        with self.assertRaisesRegex(AssertionError, "used > ring_buffer_size"):
            too_much_used.creator.compute_max_amount_writable()

    def test_expose_writer_mem_view_boundary_cases(self) -> None:
        cases = [
            ("non_wrapping", 3, 0, 5, 5, False, 5, 0),
            ("exact_fit", 12, 0, 4, 4, False, 4, 0),
            ("wrapping", 14, 8, 6, 6, True, 2, 4),
        ]
        for label, write_pos, reader_pos, request, expected_size, expected_wrap, mv1_len, mv2_len in cases:
            with self.subTest(case=label):
                pair = self._make_pair(size=16)
                pair.creator.update_write_pos(write_pos)
                self._set_reader_state(pair.creator, 0, pos=reader_pos, alive=1)

                mv1, mv2, size_writable, wrap_around = pair.creator.expose_writer_mem_view(request)
                self.assertEqual(size_writable, expected_size)
                self.assertEqual(wrap_around, expected_wrap)
                self.assertEqual(len(mv1), mv1_len)
                self.assertEqual(len(mv2) if mv2 is not None else 0, mv2_len)
                self._release_mem_views(mv1, mv2)

    def test_expose_writer_mem_view_clamps_to_available_space(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.update_write_pos(10)
        self._set_reader_state(pair.creator, 0, pos=0, alive=1)

        mv1, mv2, size_writable, wrap_around = pair.creator.expose_writer_mem_view(20)
        self.assertEqual(size_writable, 6)
        self.assertFalse(wrap_around)
        self.assertEqual(len(mv1), 6)
        self.assertIsNone(mv2)
        self._release_mem_views(mv1, mv2)

    def test_expose_reader_mem_view_boundary_cases(self) -> None:
        cases = [
            ("non_wrapping", 4, 9, 5, 5, False, 5, 0),
            ("exact_fit", 12, 16, 4, 4, False, 4, 0),
            ("wrapping", 14, 20, 6, 6, True, 2, 4),
        ]
        for label, read_pos, write_pos, request, expected_size, expected_wrap, mv1_len, mv2_len in cases:
            with self.subTest(case=label):
                pair = self._make_pair(size=16)
                pair.peer.update_reader_pos(read_pos)
                pair.creator.update_write_pos(write_pos)

                mv1, mv2, size_readable, wrap_around = pair.peer.expose_reader_mem_view(request)
                self.assertEqual(size_readable, expected_size)
                self.assertEqual(wrap_around, expected_wrap)
                self.assertEqual(len(mv1), mv1_len)
                self.assertEqual(len(mv2) if mv2 is not None else 0, mv2_len)
                self._release_mem_views(mv1, mv2)

    def test_expose_reader_mem_view_clamps_to_available_data(self) -> None:
        pair = self._make_pair(size=16)
        pair.peer.update_reader_pos(14)
        pair.creator.update_write_pos(20)

        mv1, mv2, size_readable, wrap_around = pair.peer.expose_reader_mem_view(20)
        self.assertEqual(size_readable, 6)
        self.assertTrue(wrap_around)
        self.assertEqual(len(mv1), 2)
        self.assertEqual(len(mv2), 4)
        self._release_mem_views(mv1, mv2)

    def test_expose_reader_mem_view_asserts_on_impossible_state(self) -> None:
        cases = [
            (5, 9, "max_amount_readable < 0"),
            (32, 0, "max_amount_readable > ring_buffer_size"),
        ]
        for write_pos, read_pos, message in cases:
            with self.subTest(write_pos=write_pos, read_pos=read_pos):
                pair = self._make_pair(size=16)
                pair.creator.update_write_pos(write_pos)
                pair.peer.update_reader_pos(read_pos)
                with self.assertRaisesRegex(AssertionError, message):
                    pair.peer.expose_reader_mem_view(1)

    def test_simple_write_and_read_roundtrip(self) -> None:
        cases = [
            ("contiguous", 0, b"hello"),
            ("wrapping", 14, b"ABCDEF"),
        ]
        for label, start_pos, payload in cases:
            with self.subTest(case=label):
                pair = self._make_pair(size=16)
                pair.creator.ring_buffer[:] = b"\x00" * pair.creator.ring_buffer_size
                pair.creator.update_write_pos(start_pos)
                pair.peer.update_reader_pos(start_pos)
                self._set_reader_state(pair.creator, 0, pos=start_pos, alive=1)

                writer_mem_view = pair.creator.expose_writer_mem_view(len(payload))
                pair.creator.simple_write(writer_mem_view, payload)
                pair.creator.inc_writer_pos(writer_mem_view[2])
                self._release_mem_views(writer_mem_view[0], writer_mem_view[1])

                reader_mem_view = pair.peer.expose_reader_mem_view(len(payload))
                dst = bytearray(len(payload))
                pair.peer.simple_read(reader_mem_view, dst)
                self.assertEqual(bytes(dst), payload)
                self._release_mem_views(reader_mem_view[0], reader_mem_view[1])

    def test_simple_write_discards_bytes_beyond_allocated_space(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.ring_buffer[:] = b"\xEE" * pair.creator.ring_buffer_size
        pair.creator.update_write_pos(12)
        self._set_reader_state(pair.creator, 0, pos=0, alive=1)

        writer_mem_view = pair.creator.expose_writer_mem_view(8)
        pair.creator.simple_write(writer_mem_view, b"ABCDEFGH")

        self.assertEqual(writer_mem_view[2], 4)
        self.assertEqual(bytes(pair.creator.ring_buffer[12:16]), b"ABCD")
        self.assertEqual(bytes(pair.creator.ring_buffer[0:4]), b"\xEE\xEE\xEE\xEE")
        self._release_mem_views(writer_mem_view[0], writer_mem_view[1])

    def test_simple_read_discards_bytes_when_destination_is_smaller(self) -> None:
        pair = self._make_pair(size=16)
        payload = b"ABCDEF"
        pair.creator.ring_buffer[:] = b"\x00" * pair.creator.ring_buffer_size
        pair.creator.update_write_pos(0)
        pair.peer.update_reader_pos(0)
        self._set_reader_state(pair.creator, 0, pos=0, alive=1)

        writer_mem_view = pair.creator.expose_writer_mem_view(len(payload))
        pair.creator.simple_write(writer_mem_view, payload)
        pair.creator.inc_writer_pos(writer_mem_view[2])
        self._release_mem_views(writer_mem_view[0], writer_mem_view[1])

        reader_mem_view = pair.peer.expose_reader_mem_view(len(payload))
        dst = bytearray(3)
        pair.peer.simple_read(reader_mem_view, dst)

        self.assertEqual(bytes(dst), payload[:3])
        self.assertEqual(bytes(pair.creator.ring_buffer[0:6]), payload)
        self._release_mem_views(reader_mem_view[0], reader_mem_view[1])

    def test_calculate_pressure_updates_header(self) -> None:
        empty_pair = self._make_pair(size=16)
        self.assertEqual(empty_pair.creator.calculate_pressure(), 0)
        self.assertEqual(int(empty_pair.creator.header[empty_pair.creator.pressure_index]), 0)

        full_pair = self._make_pair(size=16)
        full_pair.creator.update_write_pos(full_pair.creator.ring_buffer_size)
        self._set_reader_state(full_pair.creator, 0, pos=0, alive=1)
        self.assertEqual(full_pair.creator.calculate_pressure(), 100)
        self.assertEqual(int(full_pair.creator.header[full_pair.creator.pressure_index]), 100)

    def test_jump_to_writer_discards_unread_data(self) -> None:
        pair = self._make_pair(size=16)
        pair.creator.update_write_pos(9)
        pair.peer.update_reader_pos(2)

        pair.peer.jump_to_writer()
        self.assertEqual(int(pair.peer.header[pair.peer.reader_pos_index]), 9)

        mv1, mv2, size_readable, wrap_around = pair.peer.expose_reader_mem_view(4)
        self.assertEqual(size_readable, 0)
        self.assertFalse(wrap_around)
        self.assertEqual(len(mv1), 0)
        self.assertIsNone(mv2)
        self._release_mem_views(mv1, mv2)

    def test_finalizer_cleanup_unlinks_only_for_creator(self) -> None:
        creator_calls = []
        creator_name = self._make_name("finalizer")

        def creator_recorder(name: str, is_creator: bool) -> None:
            creator_calls.append((name, is_creator))
            creator_cleanup(name, is_creator)

        creator_cleanup = SharedRingBuffer._finalizer_cleanup
        with patch.object(SharedRingBuffer, "_finalizer_cleanup", new=staticmethod(creator_recorder)):
            creator = SharedRingBuffer(
                name=creator_name,
                create=True,
                size=16,
                num_readers=1,
                reader=NO_READER,
            )
            creator_ref = weakref.ref(creator)
            self._drop_local_views(creator)
            del creator
            self._collect_until_gone(creator_ref)

        self.assertEqual(creator_calls, [(creator_name, True)])
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=creator_name, create=False)

        peer_calls = []
        peer_name = self._make_name("finalizer")

        def peer_recorder(name: str, is_creator: bool) -> None:
            peer_calls.append((name, is_creator))
            peer_cleanup(name, is_creator)

        peer_cleanup = SharedRingBuffer._finalizer_cleanup
        with patch.object(SharedRingBuffer, "_finalizer_cleanup", new=staticmethod(peer_recorder)):
            creator = SharedRingBuffer(
                name=peer_name,
                create=True,
                size=16,
                num_readers=1,
                reader=NO_READER,
            )
            peer = SharedRingBuffer(
                name=peer_name,
                create=False,
                size=16,
                num_readers=1,
                reader=0,
            )
            peer_ref = weakref.ref(peer)
            self._drop_local_views(peer)
            del peer
            self._collect_until_gone(peer_ref)

        self.assertEqual(peer_calls, [(peer_name, False)])
        probe = shared_memory.SharedMemory(name=peer_name, create=False)
        probe.close()
        self._cleanup_ring(creator, unlink=True)


if __name__ == "__main__":
    unittest.main()
