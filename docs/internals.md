# Under the Hood

How does pure Python hit 68 Gbit/s? No C extensions, no Cython, no ctypes FFI.
The answer is `multiprocessing.shared_memory`, `memoryview`, and a ring buffer
that never touches the data more than it has to.

This page walks through the hot path -- the code that runs on every frame in a
PYTHUSA pipeline.

---

## Ring buffer memory layout

Everything starts with `SharedRingBuffer`, a subclass of
`multiprocessing.shared_memory.SharedMemory`. A single POSIX shared-memory
segment holds two regions: a **uint64 header** for coordination metadata and a
**byte payload** for the actual ring data.

```python
# pythusa/src/pythusa/_buffers/ring.py

class SharedRingBuffer(SharedMemory):
    def __init__(self, name, create, size, num_readers, reader, ...):
        ...
        super().__init__(name=name, create=create, size=self.shared_mem_size)

        # Header: a numpy uint64 array backed by the first bytes of
        # the shared-memory buffer. Fields: ring size, pressure counter,
        # dropped-byte counter, write position, max writable, num_readers,
        # and per-reader triples (position, alive flag, last_seen timestamp).
        self.header = np.ndarray(
            (header_u64_length(num_readers),),
            np.uint64,
            memoryview(self.buf[0:self.header_size]),
            0,
        )

        ...

        # Payload: a memoryview slice over the rest of the segment.
        # This IS the ring -- readers and writers operate directly on
        # this view. No intermediate buffers, no copies to get here.
        self.ring_buffer = memoryview(
            self.buf[self.header_size:self.header_size + self.ring_buffer_size]
        )
```

The header layout is computed at compile time and optionally **cache-line
aligned** to avoid false sharing between the header and payload regions:

```python
# pythusa/src/pythusa/_shared_memory/layout.py

HEADER_STATIC_FIELDS = 6   # size, pressure, dropped, write_pos, max_writable, num_readers
READER_FIELDS = 3           # position, alive, last_seen -- per reader slot
UINT64_BYTES = 8

def header_u64_length(num_readers: int) -> int:
    return HEADER_STATIC_FIELDS + (num_readers * READER_FIELDS)

def compute_header_size(num_readers, *, cache_align=False, cache_size=64):
    header_size = UINT64_BYTES * header_u64_length(num_readers)
    if not cache_align:
        return header_size
    return align_size(header_size, cache_size)

def reader_slot(reader: int) -> int:
    return HEADER_STATIC_FIELDS + (reader * READER_FIELDS)
```

One shared-memory segment. One `memoryview`. No serialization, no pickling.

---

## Zero-copy writer path

When a producer wants to write a frame, it asks the ring for a **direct
`memoryview` into the payload region**. The ring computes how much space is
available and returns either one contiguous view or a split pair for
wrap-around:

```python
# pythusa/src/pythusa/_buffers/ring.py

def expose_writer_mem_view(self, size: int) -> RingView:
    self.compute_max_amount_writable()
    if self.max_amount_writable >= size:
        size_writeable = size
    else:
        size_writeable = self.max_amount_writable

    write_pos = self.int_to_pos(int(self.header[self.write_pos_index]))

    if write_pos + size_writeable <= self.ring_buffer_size:
        # Contiguous -- single memoryview slice, no copy needed.
        mv1 = memoryview(self.ring_buffer[write_pos: write_pos + size_writeable])
        mv2 = None
        wrap_around = False
    else:
        # Wrap-around -- two views: tail of ring + head of ring.
        mv1 = memoryview(self.ring_buffer[write_pos:])
        mv2 = memoryview(
            self.ring_buffer[0:(size_writeable - (self.ring_buffer_size - write_pos))]
        )
        wrap_around = True

    return (mv1, mv2, size_writeable, wrap_around)
```

The `write_array` convenience method writes a numpy array in one call -- still
zero-copy on the contiguous path:

```python
def write_array(self, arr: np.ndarray) -> int:
    src = memoryview(arr).cast("B")
    nbytes = src.nbytes
    mv1, mv2, size_writeable, wrap_around = self.expose_writer_mem_view(nbytes)
    if size_writeable < nbytes:
        return 0
    self.simple_write((mv1, mv2, size_writeable, wrap_around), src)
    self.inc_writer_pos(nbytes)
    return nbytes
```

The user-facing API is even simpler. `StreamWriter.look()` returns a writable
`memoryview` directly into shared memory; `increment()` publishes it:

```python
# pythusa/src/pythusa/_pipeline/_stream_io.py

class StreamWriter:
    def look(self) -> memoryview | None:
        mv1, mv2, size_writeable, wrap_around = self.raw.expose_writer_mem_view(
            self.frame_nbytes
        )
        if size_writeable < self.frame_nbytes or wrap_around or mv2 is not None:
            return None
        return mv1

    def increment(self) -> None:
        self.raw.inc_writer_pos(self.frame_nbytes)
```

Fill the view, call increment. That's the entire writer hot path.

---

## Zero-copy reader path

The reader side mirrors the writer. `expose_reader_mem_view` returns a view
into the payload at the reader's current position:

```python
# pythusa/src/pythusa/_buffers/ring.py

def expose_reader_mem_view(self, size: int) -> RingView:
    write_pos = int(self.header[self.write_pos_index])
    read_pos = int(self.header[self.reader_pos_index])
    max_amount_readable = write_pos - read_pos

    if max_amount_readable > self.ring_buffer_size:
        # Reader fell behind beyond one full ring. Resync to writer.
        self.jump_to_writer()
        read_pos = int(self.header[self.reader_pos_index])
        max_amount_readable = 0

    ...

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
```

On the non-wrapped path, `read_array` uses **`np.frombuffer`** -- a view, not a
copy -- directly into the shared-memory ring:

```python
def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
    mv1, mv2, size_readable, wrap_around = self.expose_reader_mem_view(nbytes)
    if size_readable < nbytes:
        return np.empty(0, dtype=dtype)
    if not wrap_around:
        arr = np.frombuffer(mv1, dtype=dtype)   # <-- view into ring memory
    else:
        buf = bytearray(size_readable)
        self.simple_read((mv1, mv2, size_readable, wrap_around), memoryview(buf))
        arr = np.frombuffer(buf, dtype=dtype)
    self.inc_reader_pos(size_readable)
    return arr
```

At the stream level, `StreamReader.look()` gives you the same direct view:

```python
# pythusa/src/pythusa/_pipeline/_stream_io.py

class StreamReader:
    def look(self) -> memoryview | None:
        mv1, mv2, size_readable, wrap_around = self.raw.expose_reader_mem_view(
            self.frame_nbytes
        )
        if size_readable < self.frame_nbytes or wrap_around or mv2 is not None:
            return None
        return mv1

    def increment(self) -> None:
        self.raw.inc_reader_pos(self.frame_nbytes)
```

Inspect the view, call increment. Data never leaves shared memory unless you
explicitly ask for a copy.

---

## Backpressure without locks on the data path

The write path needs to know how much ring space is available. That requires
knowing the position of the **slowest reader**. Scanning all reader slots on
every write would be O(readers) per frame -- too expensive at 140,000 frames/s.

PYTHUSA uses a **cached min-reader position** that amortizes the scan:

```python
# pythusa/src/pythusa/_buffers/ring.py

def _scan_min_reader_pos(self) -> int:
    min_reader_pos = int(self.header[self.write_pos_index])
    for i in range(6, len(self.header), 3):
        reader_pos = int(self.header[i])
        reader_alive = int(self.header[i + 1])
        if reader_pos < min_reader_pos and reader_alive:
            min_reader_pos = reader_pos
    return min_reader_pos

def compute_max_amount_writable(self, force_rescan=False) -> int:
    write_pos = int(self.header[self.write_pos_index])

    if (
        force_rescan
        or self._min_reader_pos_cache is None
        or self._reader_positions_dirty
        or self._writes_since_min_scan >= self._min_reader_pos_refresh_interval
        or (time.perf_counter() - self._last_min_scan_t)
            >= self._min_reader_pos_refresh_s
    ):
        min_reader_pos = self._scan_min_reader_pos()
        self._min_reader_pos_cache = min_reader_pos
        self._writes_since_min_scan = 0
        self._reader_positions_dirty = False
        self._last_min_scan_t = time.perf_counter()
    else:
        min_reader_pos = self._min_reader_pos_cache

    used = write_pos - min_reader_pos
    self.max_amount_writable = self.ring_buffer_size - used
    return self.max_amount_writable
```

The cache refreshes every 64 writes **or** every 5 ms, whichever comes first.
Between refreshes the cached value is **conservative** -- it can only
under-report writable space, never over-report. No locks, no atomics, no
syscalls on the fast path.

---

## Worker bootstrap

Each PYTHUSA worker is a separate OS process spawned via `multiprocessing`.
The bootstrap overhead is minimal: re-attach shared-memory rings by name,
install the per-process context, run the user function.

```python
# pythusa/src/pythusa/_workers/bootstrap.py

@dataclass(slots=True)
class TaskBootstrap:
    name: str
    fn: Callable[..., Any]
    reading_ring_kwargs: dict[str, dict[str, Any]]
    writing_ring_kwargs: dict[str, dict[str, Any]]
    events: dict[str, WorkerEvent]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __call__(self) -> None:
        reading_rings = {
            name: SharedRingBuffer(**kw)
            for name, kw in self.reading_ring_kwargs.items()
        }
        writing_rings = {
            name: SharedRingBuffer(**kw)
            for name, kw in self.writing_ring_kwargs.items()
        }

        with ExitStack() as stack:
            for ring in {**reading_rings, **writing_rings}.values():
                stack.enter_context(ring)
            context._install(reading_rings, writing_rings, self.events)
            Worker(fn=self._run_task)()
```

`context._install` populates the process-local registry that `get_reader`,
`get_writer`, and `get_event` look up at runtime:

```python
# pythusa/src/pythusa/_core/context.py

_reading_rings: dict[str, SharedRingBuffer] = {}
_writing_rings: dict[str, SharedRingBuffer] = {}
_events: dict[str, WorkerEvent] = {}

def get_reader(name: str) -> SharedRingBuffer:
    return _reading_rings[name]

def get_writer(name: str) -> SharedRingBuffer:
    return _writing_rings[name]

def _install(reading_rings, writing_rings, events) -> None:
    _reading_rings.update(reading_rings)
    _writing_rings.update(writing_rings)
    _events.update(events)
```

Module-level state is safe here because each worker is its own process.
No shared mutable state, no locks, no coordination beyond the ring header.

---

## Putting it together

A user-facing PYTHUSA pipeline compiles down to:

1. **Shared-memory segments** -- one per stream, sized at `frame_bytes * ring_depth`.
2. **Header arrays** -- uint64 metadata living at the start of each segment, optionally cache-aligned.
3. **`memoryview` slices** -- writers and readers get direct pointers into the payload region.
4. **Cached backpressure** -- writers amortize the min-reader scan, staying lock-free on the data path.
5. **Process-local registries** -- each child re-attaches rings by name and runs with zero per-frame coordination overhead.

The result: **68 Gbit/s of FFT signal payload across 49 signals on a MacBook Air M2**, with Python doing the orchestration and NumPy doing the math.

See the [Showcase Demos](demos.md) for end-to-end benchmark results, or the
[Pipeline API](pipeline.md) to start building your own pipeline.
