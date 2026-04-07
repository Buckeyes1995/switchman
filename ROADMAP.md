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

### Scheduled Model Switching
Define a schedule (e.g., fast small model during the day, large model overnight). Uses launchd or a background timer. Useful when large models need quiet time to load into unified memory.

### Model Tagging
Tag models with free-form labels (e.g., `coding`, `reasoning`, `vision`, `fast`). Filter the menu and search by tag. Stored in config.

### Download Queue
Queue multiple HuggingFace downloads. Show overall queue progress in the menu bar. Cancel individual items.

### ~~HuggingFace Authentication~~ ✓ Done
`HF token` field in Settings → Sync. Passed to `hf_model_info` and `build_hf_headers` for gated model downloads. (Keychain storage is a future enhancement.)

### Model Comparison History
Save side-by-side compare results from Quick Test to a history file. Browse and export past comparisons.

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

### Resizable Panels
Settings, Quick Test, and Benchmark windows currently have fixed sizes. Allow resizing with `setMinSize_` / `setMaxSize_` and auto-layout so fields stretch with the window.

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
- **Resizable panels** — let the user drag to resize Settings and Quick Test windows
- **Model Favorites / Pinning** — pin any model to the top of the menu, above Recent
- **Context length in title** — show ctx% in menu bar title after Quick Test (e.g. `⚡ Qwen 12%`)
- **Multi-line prompt support** — shift+enter inserts newline in Quick Test input
- **Copy last response** — one-click button to copy entire Quick Test output to clipboard
- **Quick Test font size** — slider or +/- buttons to resize the output monospace font
- **Thermal / CPU stats** — show thermal state (nominal/fair/serious/critical) alongside memory pressure
- **Model notes in tooltip** — show model note as NSToolTip on the menu item instead of a submenu entry
- **Keyboard shortcut to load default model** — global hotkey that immediately loads the ★ default model
- **Export Quick Test session** — save prompt + response pair to a Markdown file
