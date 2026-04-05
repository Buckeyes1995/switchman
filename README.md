# Switchman

A macOS menu bar app for running local LLM inference. Pick a model from the menu — the right inference engine starts automatically, opencode is reconfigured, and a notification fires when it's ready.

Supports [oMLX](https://github.com/jmorganca/omlx) (MLX models, Apple Silicon) and [llama.cpp](https://github.com/ggerganov/llama.cpp) (GGUF models). Download models directly from HuggingFace. Built with Python + rumps + PyObjC — no Electron, no web views, no cloud.

> Tested on M2 Max (96 GB) running macOS 15. Apple Silicon required for MLX; GGUF works on any Mac with llama.cpp built from source.

---

## Requirements

| Requirement | Notes |
|---|---|
| macOS 13+ | Tested on macOS 14 and 15 |
| Python 3.13 | `/opt/homebrew/bin/python3.13` |
| [oMLX](https://github.com/jmorganca/omlx) | MLX model server — optional if you only use GGUF |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Built from source — required for GGUF and llama-bench |

```bash
brew install python@3.13
```

Python dependencies (installed automatically by `run.sh`): `rumps`, `pyobjc-framework-Cocoa`, `pyobjc-framework-WebKit`, `pyobjc-framework-UserNotifications`, `pyobjc-framework-Quartz`, `huggingface-hub`.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/switchman.git ~/projects/switchman
bash ~/projects/switchman/run.sh
```

The first run creates a virtualenv at `.venv/` and installs all dependencies. A `⚡` icon appears in your menu bar.

### Run as a login item (launchd)

Create `~/Library/LaunchAgents/com.yourname.switchman.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.yourname.switchman</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>/Users/YOUR_USERNAME/projects/switchman/run.sh</string>
  </array>
  <key>RunAtLoad</key>   <true/>
  <key>KeepAlive</key>  <true/>
  <key>StandardOutPath</key>
  <string>/Users/YOUR_USERNAME/Library/Logs/switchman.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOUR_USERNAME/Library/Logs/switchman.log</string>
</dict>
</plist>
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.yourname.switchman.plist
launchctl kickstart gui/$(id -u)/com.yourname.switchman
```

Logs: `~/Library/Logs/switchman.log`

---

## First-Time Setup

1. `bash run.sh` — launches the app
2. Click `⚡` in the menu bar
3. **⚙ Settings → Open Settings…**
4. Set **MLX models directory** and/or **GGUF models directory**
5. Set **oMLX service name** and **API key** (if using MLX)
6. Set **llama-server path** (if using GGUF)
7. **Save**
8. **↻ Refresh Models** — your models appear

Or use **⬇ Download from HuggingFace…** to grab a model directly.

---

## Configuration

Stored at `~/.config/switchman/config.json`. All fields are editable via **⚙ Settings → Open Settings…**.

```json
{
  "mlx_dir":          "/path/to/models/mlx/",
  "gguf_dir":         "/path/to/models/gguf/",
  "omlx_port":        8000,
  "omlx_api_key":     "your-key",
  "omlx_service":     "com.yourname.omlx",
  "llama_server":     "~/.local/llama.cpp/build/bin/llama-server",
  "llama_port":       8000,
  "opencode_config":  "~/.config/opencode/opencode.json",
  "restart_opencode": false,
  "terminal_app":     "iTerm2",
  "default_model":    "",
  "aliases":          {},
  "model_notes":      {},
  "hidden_models":    [],
  "model_params":     {},
  "recent_models":    [],
  "sync_cursor":      false,
  "sync_continue":    false,
  "sync_env":         true,
  "notifications":    true
}
```

---

## Model Directory Layout

### MLX

Each subdirectory is one model:

```
/path/to/mlx/
  Qwen3-Coder-30B-8bit/
    config.json
    model.safetensors
    ...
```

### GGUF

```
/path/to/gguf/
  ModelName/
    ModelName.gguf     ← directory name used as display name
  standalone.gguf      ← file stem used as display name
```

Multi-shard models: only the first shard (`-00001-of-NNNNN`) is shown. Files containing `mmproj` are skipped.

---

## Feature Reference

### Menu Bar Icon

| Title | Meaning |
|---|---|
| `⚡` | Idle — no model loaded |
| `⚡ ModelName` | Model active |
| `⚡ 42 t/s` | Model active — live tokens/second |
| `⚡ 42 t/s 14%ctx` | Tok/s + context usage percentage |
| `⚡🔴 42 t/s` | Active — memory pressure critical |
| `Loading weights…` ↔ `⚡` | Model loading (animated) |
| `Benchmarking…` ↔ `⚡` | Benchmark running (animated) |

---

### Loading Step Detail

The menu bar title shows the current stage while loading:

| Stage | What's happening |
|---|---|
| `Stopping engine…` | Killing previous server, waiting for port |
| `Starting oMLX…` | Bootstrapping the oMLX launchd service |
| `Loading default model…` | Auto-loading the startup default |
| `Starting llama-server…` | Spawning llama-server subprocess |
| `Loading weights…` | Waiting for model to respond to first request |
| `Warming up…` | 128-token generation to pull weights into unified memory (MLX) |

---

### Model Switching

Click any model → **▶ Select**. Selecting a model while another is loading immediately cancels the previous load — no need to wait.

**MLX (via oMLX):**
1. Kill any running llama-server
2. Ensure oMLX is healthy; start/restart if needed
3. Send 1-token completions until the correct model responds (up to 5 min)
4. Send a 128-token warm-up to pull SSD-cached weights into unified memory
5. Write to opencode config, sync clients, fire notification

**GGUF (via llama-server):**
1. Kill any running llama-server
2. Stop oMLX (launchd bootout + wait for port)
3. Spawn `llama-server -m <path> --port <port> -ngl <layers> -c <ctx>`
4. Wait for server to accept connections
5. Query `/v1/models` for the server's reported model ID
6. Write to opencode config, sync clients, fire notification

**⏹ Stop Engine** — kills the active server and frees unified memory.

**↻ Refresh Models** — rescans model directories without restarting.

---

### Default Model at Startup

Set a model to load automatically whenever Switchman starts.

Hover a model → **★ Default at startup** to set it. Click again to clear. The default model shows `★` to its left in the menu.

On launch, if a default is set and no server is already running, Switchman begins loading it immediately using the same path as a manual selection — flash animation, notification when ready.

---

### Live Tokens/Second

While a model is active, a background poll runs every 10 seconds: sends an 8-token completion and measures wall-clock throughput. The menu bar title updates to `⚡ 42 t/s`.

---

### Context Usage Meter

After each request (poll or Quick Test), Switchman shows how much of the context window has been used:

- **Menu bar:** `⚡ 42 t/s 14%ctx` — percentage of context used
- **Quick Test window:** `TTFT 340ms | 42.1 tok/s (128 tokens) | ctx 4,096/32,768 (13%)`

Context usage resets when you switch models.

---

### Server Crash Watchdog

A background timer pings `/health` every 30 seconds while a model is active. If the server stops responding, Switchman:

- Clears the active model
- Resets the menu to idle
- Fires a notification: "Inference server stopped unexpectedly"

Catches oMLX crashes, OOM kills, and manual `pkill` automatically.

---

### Memory Pressure Indicator

Shown in **⚙ Settings** submenu, polled every 30 seconds:

| Badge | Meaning |
|---|---|
| 🟢 nominal | Unified memory available |
| 🟡 warn | Pressure elevated |
| 🔴 critical | System swapping — model performance degraded |

🔴 also appears in the menu bar title when critical.

---

### Model Metadata

Each model shows metadata parsed in a background thread at startup:

- **arch** — model architecture (`qwen3`, `llama`, etc.)
- **ctx** — context window (`32,768`)
- **quant** — quantization type (`Q8_0`, `IQ4_XS`, `6bit`, etc.)
- **size** — file size on disk in GB

GGUF: reads first 8 KB of the binary header only (never loads the whole file). MLX: reads `config.json`.

---

### Model Notes

Attach a free-text note to any model. Appears in the menu as `📝 note text`.

Hover a model → **⚙ Settings…** → **Note** field → **Save**.

---

### Per-Model Settings

Hover any model → **⚙ Settings…**

| Parameter | Default | Engines |
|---|---|---|
| Context length | 32,768 | MLX + GGUF |
| Max tokens | 8,192 | MLX + GGUF |
| GPU layers | 999 (all) | GGUF only |
| Temperature | 0.7 | MLX + GGUF |
| top_p | 0.8 | MLX + GGUF |
| top_k | 20 | GGUF only |
| min_p | 0.0 | MLX + GGUF |
| Presence penalty | 1.5 | MLX + GGUF |
| Repetition penalty | 1.0 | MLX + GGUF |
| Enable thinking | off | MLX only |
| Alias | — | display name |
| Note | — | menu annotation |

---

### Sampling Presets

One-click presets in the per-model settings panel (from Qwen documentation):

| Preset | temp | top_p | top_k | presence | Thinking |
|---|---|---|---|---|---|
| Thinking — General | 1.0 | 0.95 | 20 | 1.5 | ✓ |
| Thinking — Coding | 0.6 | 0.95 | 20 | 0.0 | ✓ |
| Instruct — General | 0.7 | 0.8 | 20 | 1.5 | ✗ |
| Instruct — Reasoning | 1.0 | 1.0 | 40 | 2.0 | ✗ |

---

### Aliases

Rename any model for display without touching files. Appears everywhere: menu, menu bar title, benchmarks, opencode config.

Per-model **⚙ Settings…** → **Alias** field.

---

### Hide / Unhide

Hover a model → **⊘ Hide** to remove it from the main list. Hidden models move to **⊘ Hidden (N)** at the bottom. Click any hidden model to restore it.

---

### Recent Models

The top section of the menu shows the last 5 models you switched to, most recent first. Click any to switch immediately.

---

### Profiles

Save any model's current parameter set as a named profile, then apply it to any other model in one click.

- **Save:** Per-model **⚙ Settings…** → **Save as Profile…**
- **Apply:** Menu bar → **── Profiles ──** → click a profile name

Profiles stored at `~/.config/switchman/profiles.json`.

---

### Copy Model ID

Hover a model → **⎘ Copy model ID** — copies `omlx/ModelName` to the clipboard. Paste directly into opencode, Continue.dev, Cursor, or any OpenAI-compatible client.

---

### Quick Test Prompt

**⚙ Settings → Quick Test Prompt…** — a floating non-modal window. Type a prompt, click **Send** (or press Return). Response streams in real time with live tok/s, TTFT, and context usage shown below.

**Compare mode:** Check **Compare models** to show a second model's response side-by-side. oMLX lazy-loads both on demand.

---

### Download from HuggingFace

**⬇ Download from HuggingFace…** in the main menu.

1. Select **MLX** or **GGUF** — the save directory automatically switches to your configured model dir
2. Type a search query (e.g. `Qwen3`, `llama`, `mistral`) and click **Search**
3. Results are sorted by **source org → model name → parameter count → quant quality**. Each entry shows `[7B, Q4_K_M]` and size in GB
4. Select a result — info line shows total size, download count, and likes
5. Adjust **Save to** if needed (defaults to your MLX or GGUF directory)
6. Click **⬇ Download**

Progress shows `X.XX / Y.YY GB  (N%)  Z.Z MB/s`, updated every second by measuring bytes written to disk. When complete, the model list refreshes automatically and metadata is populated.

---

### opencode Integration

On every model switch, Switchman writes to `~/.config/opencode/opencode.json`:

- Adds the model ID to `provider.omlx.models`
- Sets `"model": "omlx/<model_id>"`
- Writes context length, max tokens, and sampling parameters

Both MLX and GGUF point to the same `omlx` provider (same port).

**Auto-restart opencode:** Enable in Global Settings. On each switch, Switchman finds running opencode processes, closes their terminal windows via AppleScript, kills the processes, then re-opens a new terminal window per unique working directory. Supports iTerm2 and Terminal.app.

---

### Client Sync

| Setting | Effect |
|---|---|
| **Sync env file** (on by default) | Writes `~/.config/switchman/env` with `MODEL_ID=omlx/<name>` |
| **Sync Cursor MCP** | Updates Cursor's MCP config |
| **Sync Continue.dev** | Updates Continue.dev's config |

Configure in **⚙ Settings → Open Settings…**.

---

### macOS Notifications

Fires when a model finishes loading ("Model ready: ModelName") and when the server crashes unexpectedly. Toggle in **⚙ Settings → Open Settings…**.

On first run, macOS prompts for notification permission. If you dismissed it: System Settings → Notifications → Switchman.

---

### Global Hotkey ⌥Space

Press **Option+Space** from anywhere to open the Switchman menu without clicking the menu bar.

**Setup required:** System Settings → Privacy & Security → Accessibility → add Python to the list. Without it, the hotkey silently does nothing.

---

### Benchmarking

Hover any model → **⏱ Benchmark…**

For API benchmarks, the model must already be loaded (GGUF always; MLX lazy-loads on first request). For llama-bench, all servers are stopped first to free unified memory.

#### API Benchmark

Sends completions via the OpenAI-compatible API. Supports:
- Multiple prompts from the built-in library (Short 32t / Medium 128t / Long 512t / Coding / Reasoning)
- Thinking sweep: Off / On / Both (MLX only)
- Gen token sweep: comma-separated values e.g. `128,512,1024`
- Multiple repetitions

Metrics: tokens/second, wall time, TTFT.

#### llama-bench (GGUF only)

Runs llama.cpp's `llama-bench` with full parameter sweeps:

| Parameter | Description |
|---|---|
| Batch sizes (`-b`) | e.g. `512,2048` |
| Ubatch sizes (`-ub`) | e.g. `512` |
| Flash attention | Off / On / Both |
| Cache type K | f16, q8_0, q4_0, q4_1, q5_0, q5_1 |
| Cache type V | Hardcoded f16 — see Known Limitations |
| Prompt tokens | `-np` |
| Gen tokens | `-ng` |
| Repetitions | `-r` |

A live progress window streams llama-bench output. Partial results are recovered even if a combination fails.

#### Edit Prompts

**⚙ Settings → Open Settings…** → **Edit Prompts…** — in-app editor for the benchmark prompt library. Stored at `~/.config/switchman/benchmark_prompts.json`.

---

### Benchmark History

**⚙ Settings → Benchmark History…** — interactive Chart.js bar chart of all past runs, grouped by model and phase. Buttons:

- **Clear History** — deletes all saved results
- **Export CSV** — saves all results to a `.csv` file (date, model, mode, label, tok/s, tokens, time, error)

History stored at `~/.config/switchman/bench_history.json`.

---

### Error Dialogs

Engine failures show a native macOS alert describing what failed and where to find logs, instead of silently stopping the loading animation.

---

## Sampling Parameter Translation

| Config key | oMLX field | llama-server field |
|---|---|---|
| `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `top_p` |
| `top_k` | *(dropped)* | `top_k` |
| `min_p` | `min_p` | `min_p` |
| `presence_penalty` | `frequency_penalty` | `presence_penalty` |
| `repetition_penalty` | `repetition_penalty` | `repeat_penalty` |

---

## Known Limitations

### Metal KV Cache Quantization

Quantized V-cache (`ctv != f16`) crashes the Metal backend on all Apple Silicon models tested ("failed to create context" before loading begins). **ctv is hardcoded to f16**. K-cache quant (`ctk`) works on most models; exceptions include models with unusual head dimensions. Re-test after llama.cpp updates.

### oMLX Active Model Detection

oMLX 0.3.x removed `/v1/models/status`. Switchman cannot detect which MLX model is loaded after an app restart — `_active` starts as None until you select a model. GGUF detection works because llama-server's `/v1/models` returns exactly the one loaded model.

### Multiple opencode Windows

When **Restart opencode** is enabled with multiple windows open across different directories, one window per unique CWD is reopened. Tested with a single window only.

### Global Hotkey Accessibility

⌥Space requires explicit Accessibility permission. System Settings → Privacy & Security → Accessibility.

---

## Architecture

Single-file Python app (`switchman.py`, ~3,600 lines) using `rumps` for the menu bar and `PyObjC` for native macOS panels.

### Threading

All AppKit/rumps callbacks run on the **main thread**. Background work (engine switching, benchmarking, metadata parsing, HTTP, disk polling) runs in daemon threads.

Main-thread re-entry from background threads: set `self._rebuild_pending = True`. A 1-second idle timer (`_on_idle_tick`) polls this flag and calls `_build_menu()`. Never create `rumps.Timer` from a background thread — it silently does nothing.

### Switch Token

Every model selection increments `_switch_token`. Each engine thread captures the token at start and checks it before every blocking operation. If the user selects a different model, the old thread detects the changed token and exits without touching `_active` or `_loading` — no race, no stale state.

### Engine Selection

| Model type | Engine | Default port |
|---|---|---|
| `.../mlx/ModelName/` | oMLX (launchd service) | 8000 |
| `.../gguf/Model.gguf` | llama-server (subprocess) | 8000 |

Both present an OpenAI-compatible API. opencode always points at `omlx/<model_id>`.

---

## Development

```bash
# Syntax check
/opt/homebrew/bin/python3.13 -c "import ast; ast.parse(open('switchman.py').read())"

# Run directly (logs to stdout)
.venv/bin/python3 switchman.py

# Restart launchd instance
launchctl kickstart -k gui/$(id -u)/com.yourname.switchman
```

See `CLAUDE.md` for PyObjC pitfalls and architecture notes.

---

## License

MIT
