# Features

## Model Switching

Click any model → **▶ Select**. Switchman starts the right inference engine automatically.

**MLX models (via oMLX):**
1. Stops any running llama-server
2. Ensures oMLX is healthy; starts/restarts if needed
3. Waits for the model to respond (up to 5 min — MLX lazy-loads on first request)
4. Sends a 128-token warm-up to pull weights into unified memory
5. Fires a notification and updates the menu bar

**GGUF models (via llama-server):**
1. Stops any running llama-server
2. Stops oMLX (freeing unified memory)
3. Spawns `llama-server -m <path> --port <port>`
4. Waits for server to accept connections
5. Fires a notification and updates the menu bar

**Cancellation:** selecting a new model while one is loading immediately cancels the previous load — no waiting.

**⏹ Stop Engine** — kills the active server and frees unified memory.

**↻ Refresh Models** — rescans model directories without restarting.

---

## Menu Bar Title

| Title | Meaning |
|---|---|
| `⚡` | Idle — no model loaded |
| `⚡ ModelName` | Model active |
| `⚡ 42 t/s` | Live throughput |
| `⚡ 42 t/s 14%ctx` | Throughput + context usage |
| `⚡🔴 42 t/s` | Active — memory pressure critical |
| `Loading weights…` ↔ `⚡` | Model loading (animated) |
| `Benchmarking…` ↔ `⚡` | Benchmark running (animated) |

Loading stages: `Stopping engine…` → `Starting oMLX…` / `Starting llama-server…` → `Loading weights…` → `Warming up…`

---

## Download from HuggingFace

**⬇ Download from HuggingFace…** in the main menu.

1. Select **MLX** or **GGUF** — the save directory switches automatically
2. Search for a model by name (e.g. `Qwen3`, `gemma`, `mistral`)
3. Results sorted by: **source org → model name → parameter count → quant quality**
4. Select a result — info line shows size, download count, and likes
5. Disk space indicator shows free space on the destination volume (🟢/🟡/🔴)
6. Click **⬇ Download**

Progress shows `X.XX / Y.YY GB  (N%)  Z.Z MB/s` updated every second.  
Downloads resume if interrupted — reopen the window to continue.  
Model list refreshes automatically when the download completes.

---

## Quick Test Prompt

**⚙ Settings → Quick Test Prompt…** — floating window. Type a prompt, press Return or **Send**. Response streams in real time with:
- **TTFT** — time to first token
- **tok/s** — generation throughput
- **ctx N/M (P%)** — context window usage

**Compare mode:** check **Compare models** to run the same prompt on a second model simultaneously, results shown side by side.

---

## Benchmarking

Hover any model → **⏱ Benchmark…**

### API Benchmark
Sends completions via the OpenAI-compatible API. Works for any loaded model.

Options:
- Prompt library: Short (32t) / Medium (128t) / Long (512t) / Coding / Reasoning
- Thinking: Off / On / Both (MLX only)
- Gen tokens: comma-separated sweep, e.g. `128,512,1024`
- Repetitions

### llama-bench *(GGUF only)*
Runs `llama-bench` directly with full parameter sweeps:

| Parameter | Example |
|---|---|
| Batch sizes | `512,2048` |
| Ubatch sizes | `512` |
| Flash attention | Off / On / Both |
| Cache type K | f16, q8_0, q4_0, q4_1, q5_0, q5_1 |
| Prompt / gen tokens | |
| Repetitions | |

A live window streams llama-bench output. Partial results are saved even if a combination fails.

### Benchmark History
**⚙ Settings → Benchmark History…** — Chart.js bar chart of all past runs with CSV export.

---

## Default Model at Startup

Hover any model → **★ Default at startup** to set it (click again to clear). The starred model loads automatically on every Switchman launch. Shown with `★` in the menu.

---

## Quick Model Search

**🔍 Search Models…** in the main menu (or press **⌥⌘Space** from anywhere) opens a floating panel:
- **All / MLX / GGUF** segment filter — scope results to one engine type
- Type to filter models instantly; results show context length and size on disk
- Arrow keys to navigate
- Enter or **Load Model** to switch
- Sorted same as HF download results (org → name → params → quant)

---

## Server Crash Watchdog

Pings `/health` every 30 seconds while a model is active. If the server stops responding, Switchman clears state and fires a notification: *"Inference server stopped unexpectedly."*

---

## Per-Model Settings

Hover any model → **⚙ Settings…** to configure context length, max tokens, sampling parameters, GPU layers, thinking mode, alias, and note.

**Sampling presets** (one-click, from Qwen docs):

| Preset | temp | top_p | top_k | Thinking |
|---|---|---|---|---|
| Thinking — General | 1.0 | 0.95 | 20 | ✓ |
| Thinking — Coding | 0.6 | 0.95 | 20 | ✓ |
| Instruct — General | 0.7 | 0.8 | 20 | ✗ |
| Instruct — Reasoning | 1.0 | 1.0 | 40 | ✗ |

---

## Profiles

Save any model's parameter set as a named profile, then apply it to any other model in one click.

- **Save:** Per-model **⚙ Settings…** → **Save as Profile…**
- **Apply:** Menu bar → **── Profiles ──** → click a profile

---

## opencode / IDE Integration *(Optional)*

On each model switch, Switchman can update:
- `~/.config/opencode/opencode.json` (model ID, context, sampling params)
- Cursor MCP config
- Continue.dev config
- `~/.config/switchman/env` (`MODEL_ID=omlx/<name>`)

Can also kill and relaunch opencode in a new terminal window (iTerm2 or Terminal.app).

All sync options default to off. Enable in **⚙ Settings → Open Settings…**.

---

## Model Visibility Manager

**⚙ Settings → Manage Visible Models…** opens a checklist of all discovered models. Check to show in the menu, uncheck to hide. **Select All** and **Clear All** buttons let you reset the list in one click.

**New model detection:** when you click **↻ Refresh Models** and a new model directory is found that wasn't previously seen, Switchman prompts: *"New model found — add to menu?"* Choose **Add to Menu** or **Hide for Now**.

---

## Other Features

| Feature | How |
|---|---|
| **Model deletion** | Hover model → **🗑 Delete model…** — confirms, then deletes from disk |
| **Copy model ID** | Hover model → **⎘ Copy model ID** → copies `omlx/ModelName` |
| **Hide / unhide** | Hover model → **⊘ Hide** / restore from **⊘ Hidden** at bottom of menu; or use **Manage Visible Models…** for a full checklist |
| **Model aliases** | Hover model → **⚙ Settings…** → Alias field |
| **Model notes** | Hover model → **⚙ Settings…** → Note field (shown as 📝 in menu) |
| **Recent models** | Last 5 selections pinned to top of menu |
| **Global hotkeys** | ⌥Space opens menu; ⌥⌘Space opens Quick Search — from anywhere (requires Accessibility permission) |
| **Notifications** | Fires on model ready and server crash — toggle in Settings |
| **Memory pressure** | 🟢/🟡/🔴 in Settings submenu; 🔴 shown in menu bar title when critical |

---

## Known Limitations

**MLX active model after restart** — oMLX doesn't expose which model is currently loaded. After restarting Switchman, the active model is unknown until you select one. GGUF detection works correctly.

**Metal KV cache quantization** — quantized V-cache crashes the Metal backend on all Apple Silicon tested. `ctv` is hardcoded to `f16`. K-cache quant works on most models.

**Global hotkey** — requires Accessibility permission; silently does nothing without it.
