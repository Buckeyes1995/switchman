# Configuration

Config is stored at `~/.config/switchman/config.json` and edited via **⚙ Settings → Open Settings…**.

```json
{
  "mlx_dir":            "/path/to/models/mlx/",
  "gguf_dir":           "/path/to/models/gguf/",
  "omlx_port":          8000,
  "omlx_api_key":       "",
  "omlx_service":       "com.yourname.omlx",
  "llama_server":       "~/.local/llama.cpp/build/bin/llama-server",
  "llama_port":         8080,
  "vllm_binary":        "~/.venv-vllm-metal/bin/vllm",
  "vllm_port":          8001,
  "vllm_models_dir":    "/path/to/models/mlx/",
  "vllm_extra_args":    "",
  "vllm_api_key":       "",
  "vllm_hf_models":     [],
  "ollama_host":        "http://localhost:11434",
  "ollama_api_key":     "",
  "default_model":      "",
  "notifications":      true,
  "opencode_config":    "~/.config/opencode/opencode.json",
  "restart_opencode":   false,
  "terminal_app":       "iTerm2",
  "sync_cursor":        false,
  "sync_continue":      false,
  "sync_env":           true,
  "aliases":            {},
  "model_notes":        {},
  "hidden_models":      [],
  "model_params":       {},
  "recent_models":      []
}
```

---

## Model Directory Layout

### MLX

Each subdirectory is one model. Switchman reads `config.json` to extract architecture, context window, and quantization info.

```
/path/to/mlx/
  Qwen3-Coder-30B-8bit/
    config.json
    model.safetensors
    tokenizer.json
    ...
  Qwen3.5-VL-7B-4bit/
    config.json
    ...
```

### GGUF

Both directory-based and standalone `.gguf` files are supported:

```
/path/to/gguf/
  ModelName/
    ModelName.gguf      ← directory name used as display name
  standalone.gguf       ← file stem used as display name
```

**Multi-shard models:** only the first shard (`-00001-of-NNNNN`) is shown as a single entry.  
**Vision projectors:** files containing `mmproj` are skipped.

### vLLM

vLLM (via vllm-metal on Apple Silicon) scans `vllm_models_dir` for HuggingFace-format model directories — the same layout as MLX. Any directory containing `config.json` or `tokenizer_config.json` is listed.

Additional models can be added as HuggingFace model IDs in Settings → Inference → vLLM → HF Model IDs (e.g. `mlx-community/Qwen3-8B-4bit`).

### Ollama

No directory configuration needed. Switchman queries the running Ollama daemon at `ollama_host` and lists all installed models automatically. Install models with `ollama pull <model>`.

---

## Per-Model Parameters

Stored in `config["model_params"][name]`. Editable via hover → **⚙ Settings…**.

| Key | Default | Notes |
|---|---|---|
| `context` | 32768 | Context window length |
| `max_tokens` | 8192 | Max output tokens |
| `n_gpu_layers` | 999 | GGUF only — 999 = all layers on GPU |
| `temperature` | 0.7 | |
| `top_p` | 0.8 | |
| `top_k` | 20 | GGUF only |
| `min_p` | 0.0 | |
| `presence_penalty` | 1.5 | |
| `repetition_penalty` | 1.0 | |
| `thinking` | false | MLX only |

---

## Sampling Parameter Translation

| Config key | oMLX field | llama-server field |
|---|---|---|
| `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `top_p` |
| `top_k` | *(dropped)* | `top_k` |
| `min_p` | `min_p` | `min_p` |
| `presence_penalty` | `frequency_penalty` | `presence_penalty` |
| `repetition_penalty` | `repetition_penalty` | `repeat_penalty` |

---

## Other Config Files

| File | Contents |
|---|---|
| `~/.config/switchman/profiles.json` | Saved parameter profiles |
| `~/.config/switchman/bench_history.json` | Benchmark run history |
| `~/.config/switchman/benchmark_prompts.json` | Custom benchmark prompt library |
| `~/.config/switchman/pending_download.json` | In-progress download state (auto-cleared on completion) |
| `~/.config/switchman/env` | `MODEL_ID=omlx/<name>` for shell sourcing |
