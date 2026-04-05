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
- Background work goes in daemon threads; use `_schedule_rebuild()` to get back to main thread
- Never call `_build_menu()` from a background thread
- Never reuse `rumps.MenuItem` instances across menu rebuilds
- After any edit, verify syntax: `python3 -c "import ast; ast.parse(open('model_switcher.py').read())"`

## PyObjC — Known Mistakes to Avoid

- **`super().init()` crashes** — Python's `super()` does not work in NSObject subclasses. Avoid custom `init` entirely (preferred), or use `objc.super(ClassName, self).init()`.
- **No arbitrary attributes on ObjC objects** — `nsbutton._field = x` raises `AttributeError`. Store metadata on the Python handler, keyed by the object's `tag()` integer.
- **`NSWindow` not `NSPanel` for modals** — `NSPanel` may not become key window in menu bar apps. Use `NSWindow` + `window.makeKeyAndOrderFront_(None)` before `NSApp.runModalForWindow_(window)`.
- **rumps MenuItem callbacks** — always `s.add(rumps.MenuItem("Title", callback=fn))`. The dict-style `s["Title"] = fn` assigns the function as the item's value, not its callback.

## General — Known Mistakes to Avoid

- **Never guess external tool JSON fields — verify first** — llama-bench `--output json` fields are `n_prompt`, `n_gen`, `avg_ts` (tok/sec), `avg_ns` (nanoseconds). Fields like `test`, `n_token`, `t_token_mean`, `t_total` do not exist. Before writing any parser for a CLI tool's output, run the tool with a minimal test and inspect the actual output.
- **All blocking work goes in background threads** — HTTP requests, subprocess calls, and anything that can take more than a few milliseconds must run in `threading.Thread(target=..., daemon=True)`. Post results back to the main thread via `rumps.Timer(fn, 0.05).start()`. Running blocking work on the main thread freezes the entire UI.
- **Honour the config the user set in the panel** — if a `BenchmarkConfig` (or any config object) has a `mode` field that the user chose, the callback must read that field to decide what to do. Never infer the mode from the model type instead — that silently ignores the user's selection.
- **No duplicate code paths** — Qwen wrote the `bench_bin` path resolution and existence check twice in a row. Read the file before writing; if logic already exists, extend it rather than duplicating it.
- **`rumps.Timer` from a background thread silently does nothing** — the only safe main-thread re-entry is setting `self._loading = False` and letting `_on_flash_tick` handle UI work. Never create timers or call AppKit from background threads.
- **Always try to parse llama-bench stdout even on non-zero exit** — sweep runs may partially succeed. Recover incomplete JSON by finding the last `}` and appending `]` before parsing.
- **Use `omlx_stop()` + `pkill -9 omlx` + poll `pgrep` before llama-bench** — `kill_port()` alone does not free unified memory held by the oMLX launchd service.

## File Writing Limit

50 lines max per write tool call — use multiple calls for larger files.
