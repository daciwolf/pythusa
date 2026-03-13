# Repository Guidelines

## Project Structure & Module Organization
The installable package lives in `src/pythusa/`. Keep the public API small in `src/pythusa/__init__.py`; place implementation details in the existing underscored subpackages: `_buffers/` for the shared ring buffer, `_workers/` for process orchestration, `_sync/` for events, `_core/` for worker-local accessors, `_processing/` for NumPy helpers, `_shared_memory/` for layout logic, and `_utils/` for small support code. Tests live in `tests/`, and runnable examples live in `examples/`.

## Build, Test, and Development Commands
Use the editable install while developing:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the full test suite with `python -m pytest -q`. Run a focused file during iteration, for example `python -m pytest tests/test_ring_buffer_basic.py -q`. Smoke-test examples directly, such as `python examples/basic_workers.py`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on public functions, `from __future__ import annotations` in modules, and concise docstrings where behavior is non-obvious. Use `snake_case` for modules, functions, and variables; use `PascalCase` for classes and dataclasses such as `RingSpec` and `ProcessMetrics`. Keep internal-only modules under underscored packages and preserve the current import grouping: standard library, third-party, then local imports.

## Testing Guidelines
Pytest is the test runner, but most tests use `unittest.TestCase`. Add new coverage in `tests/test_*.py`, and keep test names descriptive, for example `test_create_ring_registers_spec_live_ring_and_counter`. Favor focused regression tests around ring-buffer wraparound, shared-memory cleanup, and manager/task lifecycle behavior. No coverage gate is configured, so every bug fix or public API change should include a targeted test.

## Commit & Pull Request Guidelines
This repository currently has no commit history, so no house style is established yet. Use short imperative commit subjects under 72 characters, for example `manager: guard duplicate task registration`. Pull requests should explain the behavioral change, list the validation commands you ran, and link any related issue. Include screenshots only when changing documentation or example output that benefits from visual comparison.
