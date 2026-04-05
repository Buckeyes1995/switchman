# Development

## Running Locally

```bash
# Syntax check
/opt/homebrew/bin/python3.13 -c "import ast; ast.parse(open('switchman.py').read())"

# Run directly (logs to stdout)
.venv/bin/python switchman.py

# Restart launchd instance
launchctl kickstart -k gui/$(id -u)/com.yourname.switchman

# View logs
tail -f ~/Library/Logs/switchman.log
```

---

## Architecture

Single-file Python app (`switchman.py`) using:
- **rumps** — menu bar app framework
- **PyObjC** — native macOS panels (NSWindow, NSTextField, NSTableView, etc.)
- **requests** — HuggingFace downloads
- **huggingface-hub** — model metadata and file listings

### Threading Model

All AppKit/rumps callbacks run on the **main thread**. Background work (engine switching, benchmarking, metadata parsing, HTTP downloads) runs in daemon threads.

**Main-thread re-entry pattern:** background threads never call UI functions directly. Instead they set `self._rebuild_pending = True`. A 1-second idle timer (`_on_idle_tick`) on the main thread polls this flag and calls `_build_menu()`.

> Never create `rumps.Timer` from a background thread — it silently does nothing.

### Switch Token

Every model selection increments `self._switch_token`. Each engine thread captures the token at start and checks it before every blocking call. If the user selects a different model mid-load, the old thread detects the changed token and exits cleanly without modifying `_active` or `_loading`.

### Engine Selection

| Model location | Engine | Default port |
|---|---|---|
| `.../mlx/ModelName/` | oMLX (launchd service) | 8000 |
| `.../gguf/Model.gguf` | llama-server (subprocess) | 8000 |

Both expose an OpenAI-compatible API at the same port.

---

## PyObjC Gotchas

**No arbitrary Python attributes on ObjC objects.** `win.foo = bar` raises `AttributeError`. Store references on the Python app object instead (`self._foo = bar`).

**Trailing underscore = colon in selector.** `def closeWin_(self, s)` maps to `closeWin:` in ObjC. Button actions passed as strings should use colon form: `"closeWin:"` not `"closeWin_"`. Multi-segment selectors: `def foo_bar_(self, a, b)` → `foo:bar:`.

**`rumps.Timer` from background thread is a no-op.** Use the `_rebuild_pending` flag pattern instead.

**`objc.super(...).init()`** must be used in `__init__` for NSObject subclasses, not Python's `super()`.

---

## File Structure

```
switchman.py          Main app (~4,000 lines)
run.sh                Launcher — creates venv on first run
requirements.txt      Python dependencies
ROADMAP.md            Planned features and known issues
docs/
  installation.md
  configuration.md
  features.md
  development.md      (this file)
```

Config and data live in `~/.config/switchman/`.

---

## Contributing

1. Fork the repo
2. Make changes in a branch
3. Syntax-check before submitting: `.venv/bin/python -c "import ast; ast.parse(open('switchman.py').read())"`
4. Open a PR — describe what changed and why
