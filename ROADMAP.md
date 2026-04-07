# Switchman Roadmap

Features planned, in rough priority order. Contributions welcome.

---

## In Progress / Next Up

*Nothing currently in progress — see Planned Features below.*

---

## Visual Polish

### ~~Titlebar Transparency~~ ✓ Done
~~Call `setTitlebarAppearsTransparent_(True)` and `setMovableByWindowBackground_(True)`~~
Applied to all 12 panels and windows.

### ~~Accent-Colored Primary Buttons~~ ✓ Done
~~Use `NSColor.controlAccentColor()` on primary action buttons~~
Save / Run Benchmark buttons now use the system accent color.

### ~~NSVisualEffectView Vibrancy Backgrounds~~ ✓ Done
Applied to all 12 panels. Windows also set opaque=False + clearColor to let the effect through.

### ~~NSBox Section Dividers~~ ✓ Done
Bold section headers now include a thin NSBox separator line underneath.

### ~~SF Symbols for Menu Icons~~ ✓ Done
All menu action items use SF Symbol images (gearshape, timer, play.fill, trash, etc.).

### ~~Colored Status Dot for Active Model~~ ✓ Done
Active model shows a green ● prefix instead of a plain checkmark.

### ~~Dark Mode Support in HTML Panels~~ ✓ Done
Both benchmark HTML panels have `@media (prefers-color-scheme: dark)` blocks.

### ~~Quick Test Window Visual Refresh~~ ✓ Done
Separator line between input and output; stats label uses secondaryLabelColor.

---

## Developer Infrastructure

### Test Suite for Pure Logic (requires refactor)
`switchman.py` is a single file where UI and logic are interleaved, making AppKit-free unit testing impossible without first extracting the pure functions. The plan:

1. Extract testable logic into `switchman_core.py`: `load_config`, `save_config`, `model_params`, `mlx_sampling_params`, `llama_sampling_params`, `scan_mlx`, `scan_gguf`, `_hf_parse_params`, `_hf_parse_quant`, `_hf_sort_key`, benchmark result aggregation, `parse_gguf_metadata`, `parse_mlx_metadata`
2. Write a `pytest` suite in `tests/` covering config round-trips, parameter merging, model scanning (temp dir fixtures), and HF sort/parse logic
3. Add a `pytest` step to the CI workflow (runs on Linux — no macOS required since the extracted module has no AppKit imports)

Held until there's appetite for the refactor — the extraction itself is low risk but touches import paths throughout the file.

---

## Planned Features

### ~~Token / Memory Cost Estimator~~ ✓ Done
Warns before loading if estimated model size (disk × 1.15) exceeds total RAM − 6 GB headroom.

### ~~Prompt History in Quick Test~~ ✓ Done
Up/down arrow keys cycle previous prompts. Stored in `~/.config/switchman/prompt_history.json`.

### ~~Export / Import Settings~~ ✓ Done
~~One-click export of the full config to a single JSON file.~~
Available under ⚙ Settings → Export Settings… / Import Settings…

### ~~Scheduled Model Switching~~ ✓ Done
~~Define a schedule (e.g., fast small model during the day, large model overnight).~~
"Model Schedule…" in Settings submenu. Define day-of-week + HH:MM + model name rules. A 60s background timer checks and switches automatically.

### ~~Model Tagging~~ ✓ Done
~~Tag models with free-form labels (e.g., `coding`, `reasoning`, `vision`, `fast`). Filter the menu and search by tag. Stored in config.~~
Tag field in per-model settings (comma-separated). Tags shown as `🏷 coding, fast` in the menu. Search panel matches tags.

### ~~Download Queue~~ ✓ Done
~~Queue multiple HuggingFace downloads. Show overall queue progress in the menu bar. Cancel individual items.~~
Download window now shows a queue table with status column. Multiple repos can be queued; downloads proceed sequentially. "✕ Cancel" button per item.

### ~~HuggingFace Authentication~~ ✓ Done
`HF token` field in Settings → Sync. Passed to `hf_model_info` and `build_hf_headers` for gated model downloads. (Keychain storage is a future enhancement.)

### ~~Model Comparison History~~ ✓ Done
~~Save side-by-side compare results from Quick Test to a history file. Browse and export past comparisons.~~
Results saved to `~/.config/switchman/compare_history.json` (last 50 entries). "Compare History…" in Settings submenu opens a scrollable panel.

### ~~IDE / Client Sync — Aider~~ ✓ Done
Writes `~/.aider.conf.yml` (port + model name) on every switch. Toggle in Settings → Sync.

### ~~IDE / Client Sync — Zed~~ ✓ Done
Writes `~/.config/zed/settings.json` assistant section on every switch. Toggle in Settings → Sync.

### ~~IDE / Client Sync — Fix Continue.dev~~ ✓ Done
Now writes `apiBase`, `model`, and `apiKey` to the "Local LLM" entry. Toggle in Settings → Sync. Needs real-world testing.

### IDE / Client Sync — Cursor (needs rethink)
Previous implementation wrote to `~/.cursor/mcp.json` as an MCP tool server — wrong approach. MCP servers provide tools/context, not the AI model. Removed. Cursor's AI model provider is not easily configurable externally; needs investigation before re-adding. Previous (incorrect) code for reference:

```python
def sync_cursor_config(port: int) -> None:
    path = Path.home() / ".cursor" / "mcp.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(path.read_text()) if path.exists() else {}
        cfg.setdefault("mcpServers", {})["local-llm"] = {
            "url": f"http://localhost:{port}/v1"
        }
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    except Exception:
        pass
```

---

## Larger / Longer Term

### Ollama Backend
Add Ollama as a third inference backend alongside oMLX and llama.cpp. Detect running Ollama instances, list their models, switch without restarting.

### richer Menu Bar Popover
Replace the plain NSMenu with a custom NSPopover showing:
- Mini benchmark chart (last 5 runs)
- Memory pressure graph
- Active model stats (uptime, requests served, avg tok/s)
- One-click model switching without submenus

### Remote Model Management
Connect to a remote Switchman instance (e.g., Mac Studio from a MacBook). Browse and switch models on the remote machine from the local menu bar.

### Benchmark Comparison View
Select two or more past benchmark runs and diff them side by side — useful for comparing quantization levels or context lengths on the same model.

### ~~Plugin / Script Hooks~~ ✓ Done
"On switch script" field in Settings runs any shell command after every switch. Env vars: `SWITCHMAN_MODEL`, `SWITCHMAN_PORT`, `SWITCHMAN_KIND`.

### ~~Auto-Reload on Crash~~ ✓ Done
Watchdog detects server death and optionally reloads the last active model. Toggle in Settings → Sync.

### ~~Server Log Viewer~~ ✓ Done
Settings → Server Logs… opens a floating panel tailing the oMLX launchd log stream.

### ~~Model Load Time~~ ✓ Done
macOS notification includes elapsed load time in seconds (e.g. "Model ready (42s)").

---

## New Planned Features

### ~~Resizable Panels~~ ✓ Done
~~Settings, Quick Test, and Benchmark windows currently have fixed sizes.~~
All major windows (Settings, Quick Test, Benchmark, Compare History, Schedule) use style mask 15 (resizable) with `setMinSize_` floors.

### Model Favorites / Pinning
Let users pin any model to the top of the menu (above Recent) regardless of recency. Stored as `pinned_models: []` in config.

### Context Usage in Menu Bar Title
After a Quick Test run, show ctx% in the menu bar title (e.g. `⚡ Qwen 12%`). Clear on next model switch.

### Multi-line Prompt Support in Quick Test
Shift+Enter inserts a newline in the Quick Test input field. Currently Enter always submits.

### Copy Last Response Button
One-click button in Quick Test to copy the entire output text view to clipboard.

### Quick Test Font Size Controls
`+` / `−` buttons (or a slider) to resize the monospace output font without restarting.

### Thermal State Indicator
Show macOS thermal state (nominal / fair / serious / critical) alongside the memory pressure dot in the Settings submenu.

### Model Notes as Tooltip
Show model notes as `NSToolTip` on the menu item row rather than as a non-interactive submenu entry — reduces menu depth.

### Keyboard Shortcut to Load Default Model
Global hotkey (e.g. ⌥⌘D) that immediately loads the ★ default model, bypassing the menu.

### Export Quick Test Session
Save the current prompt + response pair to a Markdown file via NSSavePanel. Useful for sharing or archiving interesting outputs.

---

## Known Issues / Tech Debt

- MLX active model detection after restart — oMLX doesn't expose which model is loaded; `_active` is unknown until a selection is made
- `windowWillClose_` delegate on download window may not always fire when window is force-quit
- Benchmark list append is not thread-safe (low risk under GIL but not guaranteed)
- `_active` and `_loading` written from background threads without locks

---

## Ideas Parking Lot
> Not committed to — just captured for consideration

- Menu bar icon that reflects model size (small/medium/large dot)
- Token-per-dollar cost display (for hosted model comparison)
- Integration with `llm` CLI tool (Simon Willison's)
- Auto-detect new models dropped into model directories (FSEvents watcher)

---

## New Enhancements (2026-04-06)

### Quick Test / Output
- **Markdown rendering** — replace NSTextView output with WKWebView + marked.js; code blocks get syntax highlighting and a per-block Copy button
- **Conversation mode** — multi-turn chat with message history; "New Conversation" button resets; history saved to `~/.config/switchman/conversations/`
- **System prompt field** — collapsible text area above the input; saved per-model in config
- **Inline parameter controls** — collapsible row: Temperature slider, Max Tokens stepper, Preset dropdown (Precise / Balanced / Creative) — no need to open Settings
- **Token counter as you type** — approximate token count (`~42 tokens`) shown below the prompt input in real time; warns when approaching context limit
- **Response word/char count** — add word count and character count to the stats bar alongside tok/s
- **Prompt template variables** — define `{{variable}}` placeholders in a prompt; Switchman shows a fill-in dialog before sending
- **Draft prompt persistence** — unsent prompt text is saved between sessions so you don't lose it on restart

### Menu Bar / Popover
- **Custom popover** — replace plain NSMenu with a floating NSPanel showing active model, live tok/s, memory pressure bar, thermal state, mini spark chart, and one-click model switching (see UI2_PLAN.md)
- **Mini tok/s spark chart** — bar chart of last 5 tok/s readings drawn in the popover with CoreGraphics
- **Animated model switch** — in-popover progress bar sweeps during load; fades to new model name on completion
- **Model size badge in menu** — colored dot or label next to each model showing disk size (e.g. `63 GB`)

### Settings
- **Tabbed settings panel** — replace single scroll with NSTabView: Models / Inference / Sync / Behavior / Appearance (see UI2_PLAN.md)
- **Model cards with live search** — NSSearchField filters a card-based model list; each card shows size, avg tok/s, last load time, and action icons
- **Per-model context usage history** — track and display the highest ctx% ever reached per model, so you know if you're regularly hitting limits

### System / Infrastructure
- **FSEvents model directory watcher** — auto-detect new models dropped into MLX or GGUF directories without needing a manual Refresh
- **Auto-benchmark on first load** — optionally run a short benchmark automatically the first time a new model is loaded, to seed its performance history
- **Model performance ranking** — sort the model list by avg tok/s (from benchmark history) with a toggle, so you can quickly find the fastest model for a given task
- **Persistent Quick Test layout** — remember window size, compare mode state, font size, and selected Model 2 between sessions
- **Ollama backend** — detect a running Ollama instance, list its models, and switch without restarting; unified with MLX/GGUF in the same menu
