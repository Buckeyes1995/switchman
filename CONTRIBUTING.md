# Contributing to Switchman

Thanks for your interest. This is a macOS-only, Apple Silicon app — a few things to know before diving in.

---

## Requirements

- macOS 13+ (Apple Silicon for MLX; any Mac for GGUF)
- Python 3.13 via Homebrew: `brew install python@3.13`
- At least one inference backend to test against:
  - [oMLX](https://github.com/jmorganca/omlx) for MLX models
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-server`) for GGUF models

---

## Setup

```bash
git clone https://github.com/Buckeyes1995/switchman.git
cd switchman
bash run.sh   # creates .venv, installs deps, launches the app
```

For development, run directly so stdout/stderr are visible:

```bash
.venv/bin/python switchman.py
```

---

## Before Submitting a PR

**1. Syntax check** — this is what CI runs:

```bash
python3 -c "import ast; ast.parse(open('switchman.py').read()); print('OK')"
```

**2. Lint check:**

```bash
brew install ruff   # one-time
ruff check switchman.py
```

Both must pass. CI will fail the PR if they don't.

**3. Manual smoke test** — run the app and verify:
- Models still load and the menu bar title updates correctly
- Any UI you touched (panels, menus, download window) opens and closes without crashing
- Config saves and loads correctly (`~/.config/switchman/config.json`)

---

## Code Style

- Single-file app by design — all code stays in `switchman.py`
- All AppKit/rumps callbacks must run on the **main thread**
- Background work goes in daemon threads; use `self._rebuild_pending = True` to trigger a main-thread menu rebuild — never call `_build_menu()` directly from a background thread
- No arbitrary Python attributes on ObjC objects (`NSButton`, `NSTextField`, etc.) — store references on the Python app object or in plain Python dicts/lists
- See [docs/development.md](docs/development.md) for the full architecture and PyObjC gotchas

---

## What Makes a Good PR

- **One thing per PR.** A focused change is much easier to review than a batch of unrelated fixes.
- **Describe the why, not just the what.** What problem does this solve? What did you test?
- **Stay in scope.** Don't clean up surrounding code or add docstrings to functions you didn't change.
- **Apple Silicon confirmed.** If you don't have Apple Silicon, note that in the PR.

---

## What We're Not Looking For (Right Now)

- Ports to other platforms (Linux, Intel Mac) — out of scope for this project
- Rewriting to a multi-file structure — single-file is intentional
- Adding new dependencies without discussion — open an issue first

---

## Opening an Issue

Use the issue templates. For bugs, include:
- macOS version and chip (e.g. M2 Max, macOS 15.3)
- Which backend (oMLX / llama-server / both)
- Steps to reproduce
- What you expected vs. what happened
- Any console output (run `switchman.py` directly to see it)
