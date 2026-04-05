# Switchman

A macOS menu bar app for managing local LLM inference. Pick a model, it loads — no terminal, no config files, no fuss.

Supports **MLX** models via [oMLX](https://github.com/jmorganca/omlx) (Apple Silicon) and **GGUF** models via [llama.cpp](https://github.com/ggerganov/llama.cpp). Built with Python + rumps + PyObjC — no Electron, no web views, no cloud.

> Tested on M2 Max (96 GB), macOS 15. Apple Silicon required for MLX; GGUF works on any Mac.

---

## Features

- **One-click model switching** — MLX and GGUF models in the same menu. Switching while loading cancels the previous load immediately.
- **Live tokens/second** — background poll updates the menu bar title every 10 seconds.
- **Context usage meter** — shows `42 t/s 14%ctx` in the menu bar and exact counts in the test window.
- **Default model at startup** — pick any model to auto-load when the app starts.
- **Download from HuggingFace** — search, preview size, and download directly into your model directory with real-time MB/s progress.
- **Quick Test Prompt** — floating window to send prompts and stream responses with TTFT, tok/s, and context stats.
- **Compare mode** — run the same prompt on two models side by side.
- **Benchmarking** — API benchmark (any model) and llama-bench (GGUF, full parameter sweeps).
- **Benchmark history** — interactive Chart.js bar chart of all past runs, with CSV export.
- **Server crash watchdog** — pings `/health` every 30 s; notifies and resets state if the server dies.
- **Memory pressure indicator** — 🟢/🟡/🔴 badge in the Settings submenu.
- **Per-model settings** — context length, max tokens, temperature, top_p, top_k, min_p, penalties, GPU layers, thinking mode.
- **Sampling presets** — one-click Qwen-recommended presets for thinking and instruct modes.
- **Model notes & aliases** — annotate or rename any model without touching files.
- **Recent models** — last 5 selections pinned to the top of the menu.
- **Profiles** — save a parameter set and apply it to any other model in one click.
- **Hide/unhide models** — declutter the menu without deleting anything.
- **Copy model ID** — one click to copy the OpenAI-compatible model ID to clipboard.
- **Global hotkey ⌥Space** — open the menu from anywhere (requires Accessibility permission).
- **macOS notifications** — fires when a model finishes loading or a server crashes.
- **opencode / Cursor / Continue.dev sync** *(optional)* — automatically updates coding agent configs on each switch.

---

## Requirements

| Requirement | Notes |
|---|---|
| macOS 13+ | Tested on macOS 14 and 15 |
| Python 3.13 | `/opt/homebrew/bin/python3.13` |
| [oMLX](https://github.com/jmorganca/omlx) | Only needed for MLX models |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Only needed for GGUF models and llama-bench |

```bash
brew install python@3.13
```

Python dependencies are installed automatically on first run: `rumps`, `pyobjc-framework-Cocoa`, `pyobjc-framework-WebKit`, `pyobjc-framework-Quartz`, `huggingface-hub`, `requests`.

---

## Installation

```bash
git clone https://github.com/Buckeyes1995/switchman.git ~/projects/switchman
bash ~/projects/switchman/run.sh
```

First run creates `.venv/` and installs dependencies. A `⚡` icon appears in your menu bar.

### Run at login (launchd)

Create `~/Library/LaunchAgents/com.yourname.switchman.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>       <string>com.yourname.switchman</string>
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

1. `bash run.sh` — app launches, `⚡` appears in the menu bar
2. Click `⚡` → **⚙ Settings → Open Settings…**
3. Set your **MLX models directory** and/or **GGUF models directory**
4. For MLX: set **oMLX port** and **API key** (if your server uses one)
5. For GGUF: set the path to your **llama-server** binary
6. Click **Save**
7. Click **↻ Refresh Models** — your models appear in the menu

**Don't have oMLX or llama.cpp yet?** Use **⬇ Download from HuggingFace…** to grab a GGUF model first, then [build llama.cpp from source](https://github.com/ggerganov/llama.cpp#build) to serve it. No opencode or other tools required.

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
    ModelName.gguf      ← directory name used as display name
  standalone.gguf       ← file stem used as display name
```

Multi-shard models: only the first shard (`-00001-of-NNNNN`) is shown. Files containing `mmproj` are skipped.

---

## Feature Reference

### Menu Bar Title

| Title | Meaning |
|---|---|
| `⚡` | Idle — no model loaded |
| `⚡ ModelName` | Model active |
| `⚡ 42 t/s` | Model active, live throughput |
| `⚡ 42 t/s 14%ctx` | Throughput + context usage |
| `⚡🔴 42 t/s` | Active — memory pressure critical |
| `Loading weights…` ↔ `⚡` | Model loading (animated) |
| `Benchmarking…` ↔ `⚡` | Benchmark running (animated) |

Loading stages shown in the title: `Stopping engine…` → `Starting oMLX…` / `Starting llama-server…` → `Loading weights…` → `Warming up…`

---

### Model Switching

Click any model → **▶ Select**. Selecting a model while another is loading immediately supersedes it — no waiting.

**⏹ Stop Engine** — kills the active server and frees unified memory.

**↻ Refresh Models** — rescans directories without restarting the app.

---

### Default Model at Startup

Hover any model → **★ Default at startup** to set it; click again to clear. The starred model loads automatically on every launch. It shows `★` to its left in the menu.

---

### Live Tokens/Second + Context Meter

A background poll runs every 10 seconds while a model is active. The menu bar title updates to `⚡ 42 t/s`. After each request, context usage appears: `⚡ 42 t/s 14%ctx`.

---

### Server Crash Watchdog

Pings `/health` every 30 seconds. If the server stops responding, Switchman clears the active model and fires a notification: *"Inference server stopped unexpectedly"*.

---

### Memory Pressure

Polled every 30 seconds from `sysctl`. Shown as 🟢/🟡/🔴 in the Settings submenu and as 🔴 in the menu bar title when critical.

---

### Model Metadata

Parsed in a background thread at startup — arch, context window, quantization, and disk size. Displayed in the menu under each model name. GGUF metadata is read from the binary header only (first 8 KB); MLX from `config.json`.

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
| Alias | — | display name override |
| Note | — | annotation shown in menu |

**Sampling presets** (one-click, from Qwen docs):

| Preset | temp | top_p | top_k | Thinking |
|---|---|---|---|---|
| Thinking — General | 1.0 | 0.95 | 20 | ✓ |
| Thinking — Coding | 0.6 | 0.95 | 20 | ✓ |
| Instruct — General | 0.7 | 0.8 | 20 | ✗ |
| Instruct — Reasoning | 1.0 | 1.0 | 40 | ✗ |

---

### Quick Test Prompt

**⚙ Settings → Quick Test Prompt…** — floating window. Type a prompt, press Return or click **Send**. Response streams in real time with TTFT, tok/s, and context stats below.

**Compare mode:** Check **Compare models** to run the same prompt on a second model side by side.

---

### Download from HuggingFace

**⬇ Download from HuggingFace…** in the main menu.

1. Select **MLX** or **GGUF** — the save directory switches automatically
2. Type a search query and click **Search**
3. Results are sorted by **source org → model name → parameter count → quant quality**
4. Select a result — info line shows total size, download count, and likes
5. Adjust **Save to** directory if needed
6. Click **⬇ Download**

Progress shows `X.XX / Y.YY GB  (N%)  Z.Z MB/s` updated every second. When complete, the model list refreshes automatically. Downloads resume from where they left off if interrupted. Error text is selectable for copy/paste.

---

### Benchmarking

Hover any model → **⏱ Benchmark…**

#### API Benchmark

Sends completions via the OpenAI-compatible API. Supports prompt library (Short/Medium/Long/Coding/Reasoning), thinking sweep (MLX), gen token sweep, and multiple repetitions. Metrics: tok/s, TTFT, wall time.

#### llama-bench *(GGUF only)*

Runs `llama-bench` with full parameter sweeps — batch size, ubatch, flash attention, KV cache quantization, prompt/gen tokens, repetitions. A live window streams output. Partial results are recovered on failure.

**Edit Prompts:** **⚙ Settings → Open Settings… → Edit Prompts…** — in-app editor for the benchmark prompt library (`~/.config/switchman/benchmark_prompts.json`).

---

### Benchmark History

**⚙ Settings → Benchmark History…** — Chart.js bar chart of all past runs, grouped by model and phase.

- **Clear History** — deletes all saved results
- **Export CSV** — date, model, mode, label, tok/s, tokens, time, error

History stored at `~/.config/switchman/bench_history.json`.

---

### Profiles

Save a model's parameter set as a named profile, then apply it to any model in one click.

- **Save:** Per-model **⚙ Settings…** → **Save as Profile…**
- **Apply:** Menu bar → **── Profiles ──** → click a profile

Stored at `~/.config/switchman/profiles.json`.

---

### Other Menu Actions

| Action | How |
|---|---|
| **Copy model ID** | Hover model → **⎘ Copy model ID** → pastes `omlx/ModelName` |
| **Hide model** | Hover model → **⊘ Hide** (restore from **⊘ Hidden** at bottom) |
| **Recent models** | Last 5 selections pinned to the top of the menu |
| **Model notes** | Hover model → **⚙ Settings…** → Note field |
| **Aliases** | Hover model → **⚙ Settings…** → Alias field |

---

### Global Hotkey ⌥Space

Press **Option+Space** from anywhere to open the Switchman menu.

**One-time setup required:** System Settings → Privacy & Security → Accessibility → add Python.

---

### Notifications

Fires when a model finishes loading and when a server crashes. Toggle in **⚙ Settings → Open Settings…**.

---

## opencode / IDE Integration *(Optional)*

Switchman works standalone — opencode is not required. If you use opencode, Cursor, or Continue.dev, Switchman can keep them in sync automatically.

On each model switch, Switchman can:
- Write the active model to `~/.config/opencode/opencode.json`
- Write `~/.config/switchman/env` (`MODEL_ID=omlx/<name>`) for shell scripts
- Update Cursor MCP config
- Update Continue.dev config
- Kill and relaunch opencode in your terminal (iTerm2 or Terminal.app)

Configure in **⚙ Settings → Open Settings…**. All sync options are off by default.

---

## Configuration

Stored at `~/.config/switchman/config.json`. Edited via **⚙ Settings → Open Settings…**.

```json
{
  "mlx_dir":          "/path/to/models/mlx/",
  "gguf_dir":         "/path/to/models/gguf/",
  "omlx_port":        8000,
  "omlx_api_key":     "",
  "omlx_service":     "com.yourname.omlx",
  "llama_server":     "~/.local/llama.cpp/build/bin/llama-server",
  "llama_port":       8000,
  "default_model":    "",
  "notifications":    true,
  "opencode_config":  "~/.config/opencode/opencode.json",
  "restart_opencode": false,
  "terminal_app":     "iTerm2",
  "sync_cursor":      false,
  "sync_continue":    false,
  "sync_env":         true,
  "aliases":          {},
  "model_notes":      {},
  "hidden_models":    [],
  "model_params":     {},
  "recent_models":    []
}
```

---

## Known Limitations

**Metal KV cache quantization** — quantized V-cache (`ctv != f16`) crashes the Metal backend on all Apple Silicon tested. `ctv` is hardcoded to `f16`. K-cache quant works on most models.

**MLX active model after restart** — oMLX doesn't expose which model is loaded. After restarting Switchman, `_active` is unknown until you select a model. GGUF detection works fine.

**Global hotkey** — requires Accessibility permission; silently does nothing without it.

---

## Architecture

Single-file Python app (`switchman.py`) using `rumps` for the menu bar and `PyObjC` for native macOS panels. All AppKit callbacks run on the main thread. Background threads (switching, benchmarking, metadata, HTTP) communicate back via a `_rebuild_pending` flag polled by a 1-second idle timer. A switch token pattern prevents race conditions when selections overlap.

---

## Development

```bash
# Syntax check
/opt/homebrew/bin/python3.13 -c "import ast; ast.parse(open('switchman.py').read())"

# Run directly (logs to stdout)
.venv/bin/python switchman.py

# Restart launchd instance
launchctl kickstart -k gui/$(id -u)/com.yourname.switchman
```

---

## License

MIT
