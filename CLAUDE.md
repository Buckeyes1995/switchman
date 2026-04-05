# Switchman — CLAUDE.md

Agent guidance for working on this project.

## What this is

A macOS menu bar app (`rumps` + PyObjC) that lets the user pick a local LLM from a dropdown. Selecting a model starts the appropriate inference engine (oMLX for MLX models, llama-server for GGUF), writes the selection into opencode's config, and optionally kills and reopens opencode in the same terminal window.

## Running it

```bash
bash run.sh   # creates .venv on first run
```

Venv lives at `.venv/` (Python 3.13 via `/opt/homebrew/bin/python3.13`).

## File layout

```
switchman.py   # entire app — single file by design
run.sh              # bootstrap launcher
requirements.txt    # rumps, pyobjc-framework-Cocoa/WebKit/UserNotifications
CLAUDE.md           # this file
AGENTS.md           # coding agent guidelines
```

Config persists at `~/.config/switchman/config.json`.

## Architecture

### Threading model
- **Main thread** — AppKit/rumps callbacks, all menu rebuilds and title updates. Never block here.
- **Background threads** — engine switching (`_switch_mlx`, `_switch_gguf`, `_do_stop`), benchmarking, metadata parsing, API calls.
- Main-thread re-entry from background threads: set `self._rebuild_pending = True`. A 1s idle timer (`_on_idle_tick`) on the main thread polls this flag and calls `_build_menu()`. **Never create `rumps.Timer` from a background thread — it silently does nothing.**

### Menu rebuilding
`_build_menu()` must only run on the main thread. It calls:
```python
self.menu._menu.removeAllItems()   # clear NSMenu
self.menu.clear()                  # clear rumps OrderedDict
```
before rebuilding. **Never reuse `rumps.MenuItem` instances across rebuilds** — AppKit raises `NSInternalInconsistencyException` if you try to add an item that's already in a menu.

### Loading indicator
A `rumps.Timer` at 0.8s interval alternates the menu bar title between `"Loading model…"` and `"⚡"`. Started by `_start_flash()`, stopped by `_stop_flash()`. The tick callback checks `self._loading` and stops itself if False.

## Inference engine logic

### MLX models → oMLX

oMLX is a multi-model MLX server run as a launchd service. It lazy-loads models on first request using an LRU + paged SSD cache.

**Start sequence:**
1. Check `omlx_is_healthy()` — looks for `"engine_pool"` in the `/health` response (discriminates from llama-server which returns `{"status":"ok"}`)
2. If not healthy: `launchctl bootout` then `launchctl bootstrap && kickstart -k`
3. If something else is on the port: `kill_port()` uses `lsof -ti tcp:<port>` to SIGTERM the holder
4. Send a 1-token completion — retry until `resp["model"] == name`
5. Send a 128-token warm-up to force SSD-cached weights into unified memory

**Stop:** `launchctl bootout` + wait for port free.

### GGUF models → llama-server

**Start sequence:**
1. Kill any running `_gguf_proc`
2. Stop omlx (bootout + wait for port free)
3. Belt-and-suspenders: `kill_port()` if port still held
4. Spawn llama-server with `-m <path> --port <port> -ngl <gpu_layers> -c <context>`
5. `wait_for_port_open()` — poll until accepting connections
6. Query `/v1/models` to get server's reported model ID

## opencode integration

`set_opencode_model()` writes to `~/.config/opencode/opencode.json`:
1. Ensures model ID exists in `provider.omlx.models`
2. Sets `"model": "omlx/<model_id>"`

Both MLX and GGUF point opencode at the `omlx` provider (configurable port, default 8000).

## opencode restart

When `restart_opencode` is enabled:
1. `find_opencode_processes()` — `pgrep -f opencode-ai`, then `lsof` for CWD and `ps` for TTY
2. `_close_iterm_ttys(ttys)` — AppleScript closes matching windows **before** killing the process
3. `pkill -f opencode-ai`
4. `_open_terminal()` — AppleScript opens new iTerm/Terminal window, `cd <cwd> && opencode`

## Config schema

```json
{
  "mlx_dir":         "/path/to/models/mlx/",
  "gguf_dir":        "/path/to/models/gguf/",
  "omlx_port":       8000,
  "omlx_api_key":    "your-key",
  "omlx_service":    "com.yourname.omlx",
  "llama_server":    "~/.local/llama.cpp/build/bin/llama-server",
  "llama_port":      8000,
  "opencode_config": "~/.config/opencode/opencode.json",
  "restart_opencode": false,
  "terminal_app":    "iTerm",
  "aliases":         {},
  "model_notes":     {},
  "hidden_models":   [],
  "model_params":    {},
  "sync_cursor":     false,
  "sync_continue":   false,
  "sync_env":        true,
  "notifications":   true
}
```

`model_params` defaults: `context=32768, max_tokens=8192, gpu_layers=999, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`.

## Sampling presets

Four named presets from Qwen documentation, stored in `SAMPLING_PRESETS`:

| Preset | temp | top_p | top_k | min_p | presence | repetition |
|---|---|---|---|---|---|---|
| Thinking — General | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Thinking — Coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct — General | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |
| Instruct — Reasoning | 1.0 | 1.0 | 40 | 0.0 | 2.0 | 1.0 |

## Sampling parameter translation

Config stores canonical names (`repetition_penalty`, `top_k`).

- **`mlx_sampling_params(p)`** → oMLX API body. oMLX does **not** support `top_k`. Maps `repetition_penalty` → `frequency_penalty`.
- **`llama_sampling_params(p)`** → llama-server / opencode config. Maps `repetition_penalty` → `repeat_penalty`. Keeps `top_k`.

## Model scanning

**MLX:** `scan_mlx()` — immediate subdirs of `mlx_dir`, sorted alphabetically.

**GGUF:** `scan_gguf()` — recursive glob for `*.gguf`, with rules:
- Skip files containing `mmproj`
- Skip non-first shards (`-of-` in name but not `-00001-of-`)
- Skip depth ≥ 3 (HuggingFace cache paths)
- depth 2 → use parent dir name; depth 0-1 → use file stem

## Model metadata

`parse_gguf_metadata(path)` reads 8KB of the GGUF header (using `f.read(8192)` — **not** `path.read_bytes()[:8192]` which loads the entire file). Extracts `arch`, `context`, `quant`.

`parse_mlx_metadata(path)` reads `config.json`. Extracts `arch`, `context`, `quant`.

Metadata is populated in a background thread via `_prime_meta_cache()` and signals completion via `_rebuild_pending = True`.

## Known issues / open work

- **omlx loaded model detection** — `/v1/models/status` was removed from oMLX 0.3.0. Active model is not detected on app restart.
- **Multiple opencode windows** — reopens one window per unique CWD. Tested with single window only.
- **Terminal.app support** — implemented but untested.
- **No error dialog** — if omlx fails to start, loading indicator stops but no user-visible error is shown.
- **Metal KV cache quant** — quantized V-cache (`ctv != f16`) fails for every model tested on Apple Silicon with llama.cpp's Metal backend. The error is "failed to create context" with no further detail — it crashes before llm_load even prints context size info. K-cache quant (`ctk`) works fine on most models except those with unusual head dimensions (e.g. "unpacked" dense variants like Bonsai Q6_K). ctv is hardcoded to f16 in the UI. Root cause unknown — likely a missing Metal kernel for quantized V attention. Worth re-testing after llama.cpp updates.

## Key pitfalls (don't repeat these)

1. **`launchctl stop`** — legacy, does nothing useful. Use `bootout`.
2. **`kickstart` without `bootstrap`** — silently fails if plist isn't loaded.
3. **`omlx_is_healthy` must check for `engine_pool`** — both llama-server and omlx serve `/health`.
4. **Kill window before killing process** — if you `pkill` opencode first, iTerm may auto-close the window.
5. **Don't reuse `MenuItem` instances** — create fresh items on every `_build_menu` call.
6. **Never call `_build_menu` from a background thread** — use `_rebuild_pending = True` instead.
7. **`NSApp.activateIgnoringOtherApps_(True)` before any panel** — menu bar apps are not frontmost.
8. **oMLX does not support `top_k`** — use `mlx_sampling_params()` which drops it.
9. **NSPanel coordinate origin is bottom-left** — use `H - from_top - element_height` for top-relative positions.
10. **Never use `super().init()` in PyObjC NSObject subclasses** — use `objc.super(ClassName, self).init()` or avoid custom `init` entirely.
11. **Never set arbitrary Python attributes on PyObjC ObjC objects** — use `setTag_()` + a dict on the handler instead.
12. **rumps MenuItem callbacks** — use `s.add(rumps.MenuItem("Title", callback=fn))`, not dict-style assignment.
13. **Use `NSWindow` not `NSPanel` for modals** — `NSPanel` may not become key window in accessory apps.
14. **Never assume external tool JSON field names** — verify with a real run before writing parsers.
15. **Background threads for any blocking work** — HTTP, subprocess, disk I/O. Never on main thread.
16. **`rumps.Timer` from a background thread silently does nothing** — set `_rebuild_pending = True` instead.
17. **`kill_port()` does not free unified memory** — call `omlx_stop()` + `pkill -9 omlx` + poll `pgrep` before llama-bench.
18. **Always try to parse llama-bench stdout on non-zero exit** — recover partial JSON by finding last `}` and appending `]`.
19. **`path.read_bytes()[:N]` loads the entire file** — use `open(path, "rb"); f.read(N)` to read only N bytes from large GGUF files.
20. **`UNUserNotificationCenter` completion handlers must be non-nil callables** — passing `None` crashes the run loop.
21. **Notification permission must be requested from the main thread** — do it in the first idle timer tick, not in `__init__`.
