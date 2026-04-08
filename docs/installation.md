# Installation

## Requirements

| Requirement | Notes |
|---|---|
| macOS 13+ | Tested on macOS 14 and 15 |
| Python 3.13 | `/opt/homebrew/bin/python3.13` |
| [oMLX](https://github.com/jmorganca/omlx) | Only needed for MLX models |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Only needed for GGUF models and llama-bench |
| [vllm-metal](https://github.com/vllm-project/vllm-metal) | Only needed for vLLM backend (Apple Silicon) |
| [Ollama](https://ollama.com) | Only needed for Ollama backend |

```bash
brew install python@3.13
```

Python dependencies are installed automatically on first run: `rumps`, `pyobjc-framework-Cocoa`, `pyobjc-framework-WebKit`, `pyobjc-framework-Quartz`, `huggingface-hub`, `requests`.

---

## Basic Install

```bash
git clone https://github.com/Buckeyes1995/switchman.git ~/projects/switchman
bash ~/projects/switchman/run.sh
```

`run.sh` creates `.venv/` on first launch, installs dependencies, then starts the app. A `⚡` icon appears in your menu bar.

---

## Run at Login (launchd)

To start Switchman automatically at login and keep it running:

**1. Create the plist**

Save as `~/Library/LaunchAgents/com.yourname.switchman.plist`:

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
  <key>KeepAlive</key>   <true/>
  <key>StandardOutPath</key>
  <string>/Users/YOUR_USERNAME/Library/Logs/switchman.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOUR_USERNAME/Library/Logs/switchman.log</string>
</dict>
</plist>
```

**2. Load it**

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.yourname.switchman.plist
launchctl kickstart gui/$(id -u)/com.yourname.switchman
```

**Useful commands:**

```bash
# Restart
launchctl kickstart -k gui/$(id -u)/com.yourname.switchman

# Stop
launchctl bootout gui/$(id -u)/com.yourname.switchman

# View logs
tail -f ~/Library/Logs/switchman.log
```

---

## First-Time Setup

1. Click `⚡` in the menu bar → **⚙ Settings → Open Settings…**
2. Set **MLX models directory** and/or **GGUF models directory**
3. For MLX: set **oMLX port** (default 8000) and API key if your server uses one
4. For GGUF: set the path to your **llama-server** binary
5. Click **Save**
6. Click **↻ Refresh Models** — your models appear

**Don't have models yet?** Use **⬇ Download from HuggingFace…** to grab a GGUF model, then [build llama.cpp from source](https://github.com/ggerganov/llama.cpp#build).

---

## Optional Backends

### vLLM (Apple Silicon Metal)

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

In Settings → Inference → vLLM:
- Set **vllm binary** to `~/.venv-vllm-metal/bin/vllm`
- Set **vLLM models dir** to your MLX models directory (vllm-metal runs MLX-format models)

### Ollama

```bash
brew install ollama
ollama serve          # start the daemon
ollama pull llama3.2  # download a model
```

Models appear automatically in the `── Ollama ──` menu section. Configure host in Settings → Inference → Ollama (default: `http://localhost:11434`).

---

## Global Hotkey Setup

⌥Space requires Accessibility permission to work:

System Settings → Privacy & Security → Accessibility → add Python to the list.

Without it, the hotkey silently does nothing.
