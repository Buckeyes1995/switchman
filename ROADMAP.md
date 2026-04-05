# Switchman Roadmap

Features planned, in rough priority order. Contributions welcome.

---

## In Progress / Next Up

*Nothing currently in progress — see Planned Features below.*

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
- `_active`, `_loading`, `_ctx_used` written from background threads without locks

---

## Ideas Parking Lot
> Not committed to — just captured for consideration

- Menu bar icon that reflects model size (small/medium/large dot)
- Token-per-dollar cost display (for hosted model comparison)
- Integration with `llm` CLI tool (Simon Willison's)
- Auto-detect new models dropped into model directories (FSEvents watcher)
- Dark/light mode aware panel backgrounds
