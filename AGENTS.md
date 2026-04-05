# model-switcher — Coding Agent Instructions

## Role

You are the **implementer**. Claude Code owns design documents and plans.
Do not modify `CLAUDE.md` or any `PLAN_*.md` files — only write code.

If you see a design problem, add a comment (`# TODO: question here`) and continue.

## Session Start — Do This First

1. Read `CLAUDE.md` — it is the authoritative architecture reference
2. Read any `PLAN_*.md` file relevant to your task
3. Do not redo work already in `model_switcher.py`

## Coding Guidelines

- Python 3.13, single-file app (`model_switcher.py`)
- All UI callbacks run on the **main thread** (rumps/AppKit requirement)
- Background work goes in daemon threads; set `self._rebuild_pending = True` to trigger a main-thread menu rebuild
- Never call `_build_menu()` from a background thread
- Never reuse `rumps.MenuItem` instances across menu rebuilds
- After any edit, verify syntax: `python3 -c "import ast; ast.parse(open('model_switcher.py').read())"`

## PyObjC — Known Mistakes to Avoid

- **`super().init()` crashes** — Python's `super()` does not work in NSObject subclasses. Avoid custom `init` entirely (preferred), or use `objc.super(ClassName, self).init()`.
- **No arbitrary attributes on ObjC objects** — `nsbutton._field = x` raises `AttributeError`. Store metadata on the Python handler, keyed by the object's `tag()` integer.
- **`NSWindow` not `NSPanel` for modals** — `NSPanel` may not become key window in menu bar apps. Use `NSWindow` + `window.makeKeyAndOrderFront_(None)` before `NSApp.runModalForWindow_(window)`.
- **rumps MenuItem callbacks** — always `s.add(rumps.MenuItem("Title", callback=fn))`. The dict-style `s["Title"] = fn` assigns the function as the item's value, not its callback.

## General — Known Mistakes to Avoid

- **Never guess external tool JSON fields — verify first** — llama-bench `--output json` fields are `n_prompt`, `n_gen`, `avg_ts` (tok/sec), `avg_ns` (nanoseconds). Before writing any parser for a CLI tool's output, run the tool with a minimal test and inspect the actual output.
- **All blocking work goes in background threads** — HTTP requests, subprocess calls, and file I/O on large files must run in `threading.Thread(target=..., daemon=True)`. Post results back to the main thread via `self._rebuild_pending = True` (polled by the 1s idle timer).
- **Honour the config the user set in the panel** — if a `BenchmarkConfig` has a `mode` field, the callback must read that field. Never infer mode from model type.
- **No duplicate code paths** — read the file before writing; if logic already exists, extend it rather than duplicating it.
- **`rumps.Timer` from a background thread silently does nothing** — set `self._rebuild_pending = True` and let `_on_idle_tick` handle UI work on the main thread.
- **Always try to parse llama-bench stdout even on non-zero exit** — recover incomplete JSON by finding the last `}` and appending `]` before parsing.
- **Use `omlx_stop()` + `pkill -9 omlx` + poll `pgrep` before llama-bench** — `kill_port()` alone does not free unified memory held by the oMLX launchd service.
- **`path.read_bytes()[:N]` loads the entire file** — use `open(path, "rb"); f.read(N)` to read only N bytes from large GGUF files.
- **`UNUserNotificationCenter` completion handlers must be non-nil** — passing `None` crashes the run loop. Always pass a real callable.
- **Notification permission on main thread** — request in the first idle timer tick, not in `__init__` or a background thread.

## File Writing Limit

50 lines max per write tool call — use multiple calls for larger files.
