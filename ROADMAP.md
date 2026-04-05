# Switchman Roadmap

Features planned, in rough priority order. Contributions welcome.

---

## In Progress / Next Up

*Nothing currently in progress — see Planned Features below.*

---

## Visual Polish

### NSVisualEffectView Vibrancy Backgrounds
Replace the plain gray window backgrounds on all panels (Settings, per-model Settings, Benchmark, Quick Test) with `NSVisualEffectView` using `.sidebar` or `.hudWindow` material. Gives the frosted-glass translucency that native Mac apps use. One `NSVisualEffectView` as the root content view of each panel; existing subviews stay the same.

### Accent-Colored Primary Buttons
Use `NSColor.controlAccentColor()` on the Save / Run Benchmark / Close buttons to tint them with the user's system accent color. Currently all buttons are identical plain bezel style — the primary action should stand out visually. Set `bezelColor` on the `NSButton`.

### NSBox Section Dividers
Replace the bold plain-text section headers in panels (Paths, oMLX, Behavior, Sampling, etc.) with `NSBox` title separators (`boxType = NSBoxSeparator` + a label above). Cleaner visual grouping with less visual noise.

### Titlebar Transparency on Panels
Call `setTitlebarAppearsTransparent_(True)` and `setMovableByWindowBackground_(True)` on settings panels to blend the titlebar into the window body — removes the hard visual break at the top that makes panels look like generic dialogs.

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

### Export / Import Settings
One-click export of the full config — model parameters, profiles, aliases, notes, prompts — to a single JSON file. Import on another machine to restore everything.

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
