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

### NSVisualEffectView Vibrancy Backgrounds
Replace the plain gray window backgrounds on all panels (Settings, per-model Settings, Benchmark, Quick Test) with `NSVisualEffectView` using `.sidebar` or `.hudWindow` material. Gives the frosted-glass translucency that native Mac apps use. One `NSVisualEffectView` as the root content view of each panel; existing subviews stay the same.

### NSBox Section Dividers
Replace the bold plain-text section headers in panels (Paths, oMLX, Behavior, Sampling, etc.) with `NSBox` title separators (`boxType = NSBoxSeparator` + a label above). Cleaner visual grouping with less visual noise.

### SF Symbols for Menu Icons
Replace Unicode emoji (⚙, ⬇, ⏱, ▶, ⊘) with proper SF Symbol images via `NSImage.imageWithSystemSymbolName_accessibilityDescription_()`. SF Symbols render crisply at any size, automatically adapt to dark/light mode, and look consistent with the rest of macOS.

### Colored Status Dot for Active Model
In the model submenu, show a filled green circle (`NSImage` or a Unicode `●` with `NSColor.systemGreenColor`) next to the active model's name instead of the plain checkmark state. Makes the active model immediately obvious when scanning the menu.

### Dark Mode Support in HTML Panels
The benchmark history and results panels use hardcoded light-mode colors (`#f5f5f5`, `#1a1a1a`, `#3a5a8a`). Add `@media (prefers-color-scheme: dark)` blocks to all inline CSS so these panels look correct in dark mode. Pass the effective appearance from `NSApp.effectiveAppearance` to inject a `data-theme` attribute on `<html>` as a fallback.

### Quick Test Window Visual Refresh
- Distinct background tint on the response output area (subtle `NSBox` or `NSVisualEffectView` inset) to visually separate it from the input
- Thin `NSBox` separator between the input row and the output area
- Token stats label (`TTFT · tok/s · ctx`) styled with a secondary text color (`NSColor.secondaryLabelColor`) rather than the same color as body text

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

### Token / Memory Cost Estimator
Before loading a model, estimate how much unified memory it will consume based on parameter count, quantization, and context length. Show a warning if it may not fit alongside the OS and other apps.

### Prompt History in Quick Test
Up/down arrow keys in the Quick Test window cycle through previous prompts, like a shell history. Stored in `~/.config/switchman/prompt_history.json`.

### ~~Export / Import Settings~~ ✓ Done
~~One-click export of the full config to a single JSON file.~~
Available under ⚙ Settings → Export Settings… / Import Settings…

### Scheduled Model Switching
Define a schedule (e.g., fast small model during the day, large model overnight). Uses launchd or a background timer. Useful when large models need quiet time to load into unified memory.

### Model Tagging
Tag models with free-form labels (e.g., `coding`, `reasoning`, `vision`, `fast`). Filter the menu and search by tag. Stored in config.

### Download Queue
Queue multiple HuggingFace downloads. Show overall queue progress in the menu bar. Cancel individual items.

### HuggingFace Authentication
Support `HF_TOKEN` for downloading gated models (Llama, Gemma, etc.) and higher rate limits. Token stored in system keychain, not config file.

### Model Comparison History
Save side-by-side compare results from Quick Test to a history file. Browse and export past comparisons.

### IDE / Client Sync — Aider
On model switch, update `~/.aider.conf.yml` with the new `openai-api-base` (port) and model name. Aider is widely used in the local LLM community and reads this file on startup.

### IDE / Client Sync — Zed
On model switch, update `~/.config/zed/settings.json` under `assistant` → `default_model` with the new provider URL and model name. Zed's assistant supports custom OpenAI-compatible providers.

### IDE / Client Sync — Fix Continue.dev
Previous sync only updated `apiBase` (the port) but not the model name. Code was removed pending a proper implementation. Starting point:

```python
def sync_continue_config(port: int, model_name: str) -> None:
    path = Path.home() / ".continue" / "config.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(path.read_text()) if path.exists() else {}
        models = cfg.setdefault("models", [])
        entry = next((m for m in models if m.get("title") == "Local LLM"), None)
        if entry is None:
            entry = {"title": "Local LLM", "provider": "openai"}
            models.insert(0, entry)
        entry["apiBase"] = f"http://localhost:{port}/v1"
        entry["model"] = model_name   # was missing — caused wrong model to be used
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    except Exception:
        pass
```

Needs real-world testing against current Continue.dev config format before re-adding.

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

### Plugin / Script Hooks
Run a user-defined shell script on model switch events. Enables custom integrations (Home Assistant, Raycast, webhooks, etc.).

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
