# model-switcher — CLAUDE.md

Agent guidance for working on this project.

## What this is

A macOS menu bar app (`rumps` + PyObjC) that lets the user pick a local LLM from a dropdown. Selecting a model starts the appropriate inference engine (oMLX for MLX models, llama-server for GGUF), writes the selection into opencode's config, and optionally kills and reopens opencode in the same terminal window.

## Running it

```bash
bash ~/projects/model-switcher/run.sh   # creates .venv on first run
```

Venv lives at `~/projects/model-switcher/.venv` (Python 3.13 via `/opt/homebrew/bin/python3.13`).

## File layout

```
model_switcher.py   # entire app — single file by design
run.sh              # bootstrap launcher
requirements.txt    # rumps, pyobjc-framework-Cocoa
CLAUDE.md           # this file
```

Config persists at `~/.config/model-switcher/config.json`.

## Architecture

### Threading model
- **Main thread** — AppKit/rumps callbacks, all menu rebuilds and title updates. Never block here.
- **Background threads** — engine switching (`_switch_mlx`, `_switch_gguf`, `_do_stop`). These set state then call `_schedule_rebuild()` to get back onto the main thread.
- `_schedule_rebuild()` fires a one-shot `rumps.Timer(interval=0.05)` which calls `_on_rebuild_timer` on the main thread. This is the only safe way to touch the menu or title from a background thread.

### Menu rebuilding
`_build_menu()` must only run on the main thread. It calls:
```python
self.menu._menu.removeAllItems()   # clear NSMenu
self.menu.clear()                  # clear rumps OrderedDict
```
before rebuilding. **Never reuse `rumps.MenuItem` instances across rebuilds** — AppKit raises `NSInternalInconsistencyException` if you try to add an item that's already in a menu.

### Loading indicator
A `rumps.Timer` at 0.8s interval alternates the menu bar title between `"Loading model…"` and `"⚡"`. Started by `_start_flash()`, stopped by `_stop_flash()`. The tick callback checks `self._loading` and stops itself if it's False — guards against a queued tick firing after load completes.

## Inference engine logic

### MLX models → oMLX

oMLX is a multi-model MLX server run as a launchd service (`com.jim.omlx`). It lazy-loads models on first request using an LRU + paged SSD cache.

**Start sequence:**
1. Check `omlx_is_healthy()` — looks for `"engine_pool"` in the `/health` response (llama-server also serves `/health` but returns `{"status":"ok"}` without `engine_pool`, so this discriminates correctly)
2. If not healthy: `launchctl bootout gui/<uid> ~/Library/LaunchAgents/com.jim.omlx.plist` then `launchctl bootstrap ... && launchctl kickstart -k ...`
3. If something else is on the port (e.g. llama-server from a prior session): `kill_port(port)` uses `lsof -ti tcp:<port>` to find and SIGTERM whatever is holding it, then waits for port to free
4. Send a 1-token completion — retry loop until `resp["model"] == name` (omlx can return wrong model mid-swap)
5. Send a 128-token warm-up ("Write a short Python hello world function") to force SSD-cached weights into unified memory before opencode opens

**Stop:** `launchctl bootout gui/<uid> ~/Library/LaunchAgents/com.jim.omlx.plist` + wait for port free.

### GGUF models → llama-server

Binary: `~/.local/llama.cpp/build/bin/llama-server`

**Start sequence:**
1. Kill any running `_gguf_proc` (tracked via `self._gguf_proc: subprocess.Popen`)
2. Stop omlx (bootout + wait for port free)
3. Belt-and-suspenders: `kill_port()` if port still held
4. Spawn llama-server with `-m <path> --port <port> -ngl <gpu_layers> -c <context>`
5. `wait_for_port_open()` — poll until llama-server is accepting connections
6. Query `/v1/models` to get the server's reported model ID (always `<filename>.gguf` with extension)

Note: `self._gguf_proc` only tracks processes we spawned in this session. `kill_port()` handles processes from previous sessions that survived an app restart.

## opencode integration

`set_opencode_model()` writes two things to `~/.config/opencode/opencode.json`:
1. Ensures the model ID exists in `provider.omlx.models` (adds it with display name + context/output limits if missing). This is needed because llama-server model IDs like `Qwen3.5-9B-Q4_K_M.gguf` aren't pre-registered in opencode config.
2. Sets `"model": "omlx/<model_id>"`

Both MLX and GGUF point opencode at the `omlx` provider (port 8000).

## opencode restart

When `restart_opencode` is enabled (`cfg["restart_opencode"] = true`):

**Order matters — close window BEFORE killing process:**
1. `find_opencode_processes()` — `pgrep -f opencode-ai`, then `lsof` for CWD and `ps -o tty=` for TTY per pid
2. `_close_iterm_ttys(ttys)` — AppleScript iterates windows/tabs/sessions, matches `tty of s` against the list, closes matching windows while opencode is still alive (if process is killed first, the window may already be gone)
3. `pkill -f opencode-ai`
4. `_open_terminal()` — AppleScript opens new iTerm window, `cd <cwd> && opencode`

## Config schema

```json
{
  "mlx_dir": "/Volumes/DataNVME/models/mlx/",
  "gguf_dir": "/Volumes/DataNVME/models/gguf/",
  "omlx_port": 8000,
  "omlx_api_key": "123456",
  "omlx_service": "com.jim.omlx",
  "llama_server": "~/.local/llama.cpp/build/bin/llama-server",
  "llama_port": 8000,
  "opencode_config": "~/.config/opencode/opencode.json",
  "restart_opencode": false,
  "terminal_app": "iTerm",
  "aliases": {
    "Qwen3-Coder-Next-MLX-6bit": "Qwen3 Coder 6b"
  },
  "hidden_models": [],
  "model_params": {
    "Qwen3-Coder-Next-MLX-6bit": {
      "context": 32768,
      "max_tokens": 8192,
      "gpu_layers": 999,
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "min_p": 0.0,
      "presence_penalty": 0.0,
      "repetition_penalty": 1.0
    }
  }
}
```

`model_params` defaults: `context=32768, max_tokens=8192, gpu_layers=999, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0` (matches "Instruct — General" preset).

## Sampling presets

Four named presets from Qwen documentation, stored in `SAMPLING_PRESETS` constant:

| Preset | temp | top_p | top_k | min_p | presence | repetition |
|---|---|---|---|---|---|---|
| Thinking — General | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Thinking — Coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct — General | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |
| Instruct — Reasoning | 1.0 | 1.0 | 40 | 0.0 | 2.0 | 1.0 |

## Sampling parameter translation

Config stores canonical names (`repetition_penalty`, `top_k`).
At the point of use, two functions translate to server-specific names:

- **`mlx_sampling_params(p)`** → oMLX API body. oMLX does **not** support `top_k`.
  Maps `repetition_penalty` → `frequency_penalty`.
- **`llama_sampling_params(p)`** → llama-server / opencode config.
  Maps `repetition_penalty` → `repeat_penalty`. Keeps `top_k`.

These are used in:
1. `_switch_mlx` warm-up completion calls (oMLX)
2. `set_opencode_model` `parameters` block (written to opencode config on every switch)

## Model scanning

**MLX:** `scan_mlx()` — immediate subdirs of `mlx_dir`, sorted alphabetically.

**GGUF:** `scan_gguf()` — recursive glob for `*.gguf`, with rules:
- Skip files containing `mmproj` (vision projectors)
- Skip non-first shards (`-of-` in name but not `-00001-of-`)
- Skip depth ≥ 3 (HuggingFace cache `snapshots/` paths)
- depth 2 → use parent dir name; depth 0-1 → use file stem

## Hidden models

`hidden_models` is a list of model names excluded from the main MLX/GGUF menu sections.
Hidden models appear in a "⊘ Hidden (N)" submenu (shown only when N > 0) where they
can be unhidden. Each visible model's submenu has a "⊘ Hide" item.

## Settings UI — pending refactor (see PLAN_settings_panels.md)

The current per-model and global settings are edited via individual menu items.
A planned refactor replaces them with two native `NSPanel` modal dialogs built with PyObjC:
- **Global Settings panel** — paths, oMLX config, behavior toggles
- **Per-model Settings panel** — alias, limits, sampling params with preset picker

Implementation plan is in `PLAN_settings_panels.md`. Backup of pre-refactor code is
`model_switcher.py.bak`.

## Known issues / open work

- **Model pre-warming** — the 128-token warm-up helps but doesn't fully eliminate first-prompt latency with the SSD paged cache. A longer or more realistic prompt would help more.
- **omlx loaded model detection** — `/v1/models/status` was removed from the current omlx API (0.3.0). We can't query which model is currently hot without sending a completion. On app startup we don't attempt to detect the active model.
- **Multiple opencode windows** — if user has opencode open in multiple windows/CWDs, we reopen one window per unique CWD. Tested with single window only.
- **Terminal.app support** — implemented but untested. iTerm is the primary target.
- **No error dialog** — if omlx fails to start or llama-server crashes, the loading indicator stops but there's no user-visible error message. The menu bar just returns to `⚡`.

## Key pitfalls (don't repeat these)

1. **`launchctl stop`** — legacy syntax, does nothing useful on modern macOS. Use `bootout`.
2. **`kickstart` without `bootstrap`** — silently fails if plist isn't loaded. Always `bootstrap` first.
3. **`omlx_is_healthy` must check for `engine_pool`** — both llama-server and omlx serve `/health`. Without discriminating, switching to MLX while llama-server is running would skip the start sequence entirely and send the completion to llama-server.
4. **Kill window before killing process** — if you `pkill` opencode first, iTerm may auto-close the window, leaving no TTY to match against.
5. **Don't reuse `MenuItem` instances** — create fresh items on every `_build_menu` call.
6. **Never call `_build_menu` from a background thread** — use `_schedule_rebuild()` instead.
7. **`NSApp.activateIgnoringOtherApps_(True)` before any panel** — menu bar apps are not the frontmost app; without this, NSOpenPanel and NSPanel open behind other windows.
8. **oMLX does not support `top_k`** — passing it is silently ignored or errors. Use `mlx_sampling_params()` which drops it. llama-server uses `repeat_penalty` not `repetition_penalty`.
9. **NSPanel coordinate origin is bottom-left** — use `H - from_top - element_height` to convert top-relative positions to AppKit coordinates.
10. **Never use `super().init()` in PyObjC NSObject subclasses** — Python's `super()` does not work; it raises `AttributeError: 'super' object has no attribute 'init'`. Either avoid custom `init` entirely (preferred), or use `objc.super(ClassName, self).init()`.
11. **Never set arbitrary Python attributes on PyObjC ObjC objects** — `nsbutton._field = x` raises `AttributeError`. Store metadata on the Python handler instead, keyed by the object's `tag()` integer (`b.setTag_(n)` / `b.tag()`).
12. **rumps MenuItem callbacks must use `s.add(rumps.MenuItem("Title", callback=fn))`** — the dict-style `s["Title"] = fn` sets the item's value to the function object and renders it as the title string, not a callback.
13. **Use `NSWindow` not `NSPanel` for modal dialogs in menu bar apps** — `NSPanel` may not become key window in apps with `NSApplicationActivationPolicyAccessory`. Use `NSWindow` + `panel.makeKeyAndOrderFront_(None)` before `NSApp.runModalForWindow_(panel)`.
14. **Never assume external tool JSON field names — always verify** — llama-bench `--output json` does NOT have fields `test`, `n_token`, `t_token_mean`, or `t_total`. The actual fields are `n_prompt`, `n_gen`, `avg_ts` (tok/sec), and `avg_ns` (nanoseconds). Guess wrong and the benchmark silently produces zeros. Run the binary with a tiny test first (`-p 16 -n 16 -r 1`) and inspect the output before writing parsing code.
15. **Background threads for any blocking work** — `run_api_benchmark` and `run_llama_bench` make HTTP calls or spawn subprocesses that can take minutes. Never call these on the main thread. Use `threading.Thread(target=..., daemon=True).start()` and post results back via `rumps.Timer(..., 0.05).start()`.
16. **Branch on `bconfig.mode`, not on `kind`** — for GGUF models the user chooses between API and llama-bench in the config panel. The callback must read `bconfig.mode` to decide which runner to call, not infer it from the model type. Ignoring `bconfig.mode` means the user's selection is silently overridden.
17. **`rumps.Timer` created from a background thread silently does nothing** — timers must be created and started on the main thread. The only proven way to cross from a background thread to the main thread is via the flash timer: set `self._loading = False` and let `_on_flash_tick` (which is already running on the main thread) handle the UI work. Do not try `performSelectorOnMainThread` or creating timers from threads.
18. **`kill_port()` does not free unified memory** — oMLX is a launchd service that keeps model weights loaded in unified memory even after its port process is killed. Before running llama-bench, call `omlx_stop()` (launchctl bootout) AND `pkill -9 -f omlx`, then poll `pgrep` until the process is confirmed gone.
19. **llama-bench partial results on sweep failure** — when benchmarking multiple parameter combinations, one combo failing causes non-zero exit but earlier combos already wrote valid JSON to stdout. Always attempt to parse stdout even on non-zero returncode; recover incomplete JSON by finding the last `}` and appending `]`.
20. **`rumps.Timer` interval is the callback schedule, not a one-shot delay** — `rumps.Timer(fn, 0.05).start()` fires every 50ms. Use `timer.stop()` inside the callback for one-shot behaviour, or let the flash timer's natural polling serve as the main-thread re-entry mechanism.
