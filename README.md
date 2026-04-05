# model-switcher

A macOS menu bar app for switching between local LLM inference engines. Supports [oMLX](https://github.com/jmorganca/omlx) (MLX models) and [llama.cpp](https://github.com/ggerganov/llama.cpp) (GGUF models), with automatic server management, [opencode](https://opencode.ai) integration, and benchmarking tools.

> Built for Apple Silicon. Tested on M2 Max (96GB) running macOS 15+.

---

## Features

### Core
- **One-click model switching** — select any MLX or GGUF model from the menu bar; the right inference engine starts automatically
- **Automatic server management** — starts/stops oMLX (launchd service) and llama-server; handles port conflicts and memory cleanup
- **opencode integration** — writes the active model into `opencode.json` on every switch; optionally restarts opencode in a new iTerm/Terminal window
- **Model warm-up** — sends a completion request after load to pull weights into unified memory before your first prompt

### Models
- **MLX models** — served via oMLX (multi-model LRU + SSD paged cache)
- **GGUF models** — served via llama-server (single model, full GPU offload)
- **Auto-scan** — discovers models by scanning configured directories; refreshable without restart
- **Aliases** — rename any model for display without touching files
- **Notes** — attach a persistent note to any model (e.g. "ctv=f16 only on Metal")
- **Metadata** — arch, context length, and quant type parsed from model files and shown in the menu
- **Hide/unhide** — keep the menu tidy by hiding models you rarely use

### Settings
- **Per-model settings** — context length, max tokens, GPU layers, all sampling parameters, alias, and note
- **Sampling presets** — Thinking/Instruct × General/Coding/Reasoning presets from Qwen documentation
- **Global settings** — paths, oMLX config, behavior toggles, sync options

### Benchmarking
- **API benchmark** — runs prompt completions against the live server; sweeps thinking mode on/off and gen token counts
- **llama-bench** — runs llama.cpp's built-in benchmark with full parameter sweeps: batch size, ubatch, flash attention, cache quant types (K and V independently)
- **Visual results** — HTML table with PP/TG split, tok/s, and total time
- **Benchmark history** — Chart.js bar chart of all past runs, accessible from the menu
- **Custom prompts** — edit the prompt library used by API benchmarks in an in-app editor

### Live monitoring
- **Live tok/s** — menu bar title shows real-time tokens/second while a model is active
- **Memory pressure** — 🟢/🟡/🔴 indicator in the Settings submenu, polled every 30s
- **macOS notifications** — system notification when a model finishes loading

### Profiles & Sync
- **Profiles** — save a model's current params as a named profile; apply to any model with one click
- **Quick Test Prompt** — non-modal streaming window to test the active model without leaving the app
- **Client sync** — optionally writes the active model to Cursor MCP config, Continue.dev config, and a shared env file on every switch

---

## Requirements

- macOS 13+ (Apple Silicon recommended)
- Python 3.13 (`/opt/homebrew/bin/python3.13`)
- [oMLX](https://github.com/jmorganca/omlx) for MLX model serving (optional — GGUF works standalone)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built from source for GGUF + benchmarking

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/model-switcher.git ~/projects/model-switcher
```

That's it. The first launch creates a virtualenv and installs dependencies automatically.

### Running

```bash
bash ~/projects/model-switcher/run.sh
```

Or install as a persistent login item via launchd:

```bash
# Create ~/Library/LaunchAgents/com.yourname.model-switcher.plist
# See docs/launchd.md for a template
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.yourname.model-switcher.plist
launchctl kickstart gui/$(id -u)/com.yourname.model-switcher
```

---

## Configuration

Config is stored at `~/.config/model-switcher/config.json`. All settings are editable via the in-app Settings panel (menu bar → ⚙ Settings → Open Settings…).

```json
{
  "mlx_dir":        "/Volumes/DataNVME/models/mlx/",
  "gguf_dir":       "/Volumes/DataNVME/models/gguf/",
  "omlx_port":      8000,
  "omlx_api_key":   "your-key",
  "omlx_service":   "com.yourname.omlx",
  "llama_server":   "~/.local/llama.cpp/build/bin/llama-server",
  "llama_port":     8000,
  "opencode_config": "~/.config/opencode/opencode.json",
  "restart_opencode": false,
  "terminal_app":   "iTerm",
  "aliases":        {},
  "model_notes":    {},
  "model_params":   {},
  "hidden_models":  [],
  "sync_cursor":    false,
  "sync_continue":  false,
  "sync_env":       true,
  "notifications":  true
}
```

### Per-model parameters

Override sampling defaults per model via menu → hover model → ⚙ Settings…

| Parameter | Default | Notes |
|---|---|---|
| `context` | 32768 | Context window in tokens |
| `max_tokens` | 8192 | Max generation tokens |
| `gpu_layers` | 999 | GGUF only — layers to offload to GPU |
| `temperature` | 0.7 | |
| `top_p` | 0.8 | |
| `top_k` | 20 | MLX ignores this (not supported by oMLX) |
| `min_p` | 0.0 | |
| `presence_penalty` | 1.5 | MLX: `frequency_penalty`. llama: `repeat_penalty` |
| `repetition_penalty` | 1.0 | |
| `enable_thinking` | false | MLX only — enables extended thinking mode |

### Sampling presets

| Preset | temp | top_p | top_k | presence | Notes |
|---|---|---|---|---|---|
| Thinking — General | 1.0 | 0.95 | 20 | 1.5 | Open-ended reasoning |
| Thinking — Coding | 0.6 | 0.95 | 20 | 0.0 | Coding with thinking |
| Instruct — General | 0.7 | 0.8 | 20 | 1.5 | Default |
| Instruct — Reasoning | 1.0 | 1.0 | 40 | 2.0 | Complex reasoning |

---

## Model Directory Layout

### MLX models

```
/path/to/mlx/
  ModelName-1/          ← each subdirectory is one model
    config.json
    model.safetensors
    ...
  ModelName-2/
    ...
```

### GGUF models

```
/path/to/gguf/
  ModelName/
    ModelName.gguf      ← depth 2: uses parent dir name
  standalone.gguf       ← depth 1: uses file stem
```

Multi-shard models: only the first shard (`-00001-of-NNNNN`) is listed. Files containing `mmproj` (vision projectors) are skipped.

---

## Benchmarking

### llama-bench (GGUF models)

Hover any GGUF model → ⏱ Benchmark… → select **llama-bench** mode.

Configure sweep parameters:
- **Batch / Ubatch** — comma-separated values, e.g. `512,2048`
- **Flash attention** — off / on / sweep both
- **Cache type K/V** — select from f16, q8_0, q4_0, q4_1, q5_0, q5_1

> **Note on Metal KV cache quantization:** Quantized V-cache (`ctv != f16`) fails for every model tested on Apple Silicon — llama.cpp's Metal backend crashes with "failed to create context" before loading even begins. ctv is therefore hardcoded to f16. K-cache quant (`ctk`) works on most models; exceptions include models with unusual head dimensions (e.g. unpacked dense variants). Re-test after llama.cpp updates.

### API benchmark (MLX and GGUF)

Runs prompt completions via the OpenAI-compatible API.

Configure:
- **Prompts** — select from the built-in library (editable via Edit Prompts…)
- **Thinking sweep** — off / on / both (MLX only)
- **Gen tokens** — comma-separated values, e.g. `128,512,1024`

### Benchmark history

menu → ⚙ Settings → Benchmark History… opens an interactive Chart.js bar chart of all past runs, grouped by model and phase.

---

## Known Limitations

- **oMLX loaded model detection** — no `/v1/models/status` in oMLX 0.3.0; active model is not detected on app restart
- **Multiple opencode windows** — if opencode is open in multiple CWDs, one window per unique CWD is reopened; untested beyond single window
- **Terminal.app** — implemented but untested; iTerm is the primary target
- **KV cache quant on Metal** — quantized V-cache (`ctv != f16`) fails for many model architectures on Apple Silicon; use ctv=f16 for reliable sweeps
- **No error dialog** — if oMLX fails to start, the loading indicator stops but no user-visible error is shown

---

## Architecture

Single-file Python app (`model_switcher.py`, ~3000 lines) using `rumps` for the menu bar and `PyObjC` for native macOS panels.

### Threading model

All AppKit/rumps callbacks run on the **main thread**. Background work (engine switching, benchmarking, metadata parsing, API calls) runs in daemon threads. The only safe main-thread re-entry from a background thread is setting `self._rebuild_pending = True`, which is polled by a 1s idle timer.

### Engine selection

| Model type | Engine | Port | Config key |
|---|---|---|---|
| `.../mlx/ModelName/` | oMLX (launchd) | `omlx_port` | `com.yourname.omlx` |
| `.../gguf/Model.gguf` | llama-server | `llama_port` | `llama_server` |

Both engines present an OpenAI-compatible API. opencode is pointed at `omlx/<model_id>` in both cases (same port).

---

## Development

```bash
# Syntax check
/opt/homebrew/bin/python3.13 -c "import ast; ast.parse(open('model_switcher.py').read())"

# Run directly (logs to stdout/stderr)
.venv/bin/python3 model_switcher.py
```

Logs from the launchd-managed instance go to `~/Library/Logs/model-switcher.log`.

See `CLAUDE.md` for agent coding guidelines and known PyObjC pitfalls.

---

## License

MIT
