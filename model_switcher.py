#!/usr/bin/env python3
"""Model Switcher — macOS menu bar app for switching local LLM inference engines."""
import json
import os
import socket
import struct
import subprocess
import threading
import time
import uuid
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import rumps

try:
    from AppKit import (
        NSApp, NSBackingStoreBuffered, NSBox, NSButton, NSFont,
        NSModalResponseOK, NSOpenPanel, NSPanel, NSPopUpButton,
        NSScrollView, NSTextField, NSTextView, NSURL, NSView, NSWindow,
    )
    from Foundation import NSObject, NSThread
    from WebKit import WKWebView, WKWebViewConfiguration
    try:
        from UserNotifications import (
            UNUserNotificationCenter, UNMutableNotificationContent,
            UNNotificationRequest, UNAuthorizationOptionAlert,
            UNAuthorizationOptionSound,
        )
    except ImportError:
        UNUserNotificationCenter = None
except ImportError:
    pass

# ── Settings panels ───────────────────────────────────────────────────────────

_PAD, _LW, _GAP  = 20, 150, 8   # outer padding, label width, label→field gap
_RH,  _RG        = 22, 8         # row height, row gap
_SH,  _SG        = 14, 6         # section-header height, header→row gap
_DG              = 16             # between-section gap
_BTN_H, _BTN_BOT = 28, 16        # button height, distance from panel bottom

# ── Benchmarking ──────────────────────────────────────────────────────────────

_MS_CONFIG_DIR       = Path.home() / ".config" / "model-switcher"
BENCHMARK_PROMPTS_PATH = _MS_CONFIG_DIR / "benchmark_prompts.json"
BENCH_HISTORY_PATH   = _MS_CONFIG_DIR / "bench_history.json"
PROFILES_PATH        = _MS_CONFIG_DIR / "profiles.json"
CACHE_QUANT_TYPES = ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"]

_DEFAULT_BENCHMARK_PROMPTS: dict[str, str] = {
    "Short (32 tok)":   "Write a Python function that returns the nth Fibonacci number.",
    "Medium (128 tok)": "Explain the difference between a mutex and a semaphore. Give a concrete Python example of each.",
    "Long (512 tok)":   (
        "You are a senior software engineer. Design a production-ready REST API "
        "in Python using FastAPI for a task management system. Include: "
        "authentication with JWT, CRUD endpoints for tasks, pagination, "
        "error handling, and a brief explanation of each design decision."
    ),
    "Coding":           "Implement a binary search tree in Python with insert, search, and in-order traversal.",
    "Reasoning":        "A farmer has 17 sheep. All but 9 die. How many are left? Explain your reasoning step by step.",
}


def load_benchmark_prompts() -> dict[str, str]:
    """Load prompts from disk, writing defaults on first run."""
    if not BENCHMARK_PROMPTS_PATH.exists():
        BENCHMARK_PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        BENCHMARK_PROMPTS_PATH.write_text(
            json.dumps(_DEFAULT_BENCHMARK_PROMPTS, indent=2) + "\n")
        return dict(_DEFAULT_BENCHMARK_PROMPTS)
    try:
        data = json.loads(BENCHMARK_PROMPTS_PATH.read_text())
        if isinstance(data, dict) and data:
            return data
    except Exception:
        pass
    return dict(_DEFAULT_BENCHMARK_PROMPTS)

@dataclass
class BenchmarkConfig:
    mode:            str          # "api" or "llama-bench"
    prompts:         list[str]    # keys from BENCHMARK_PROMPTS (API mode)
    n_prompt:        int          # prompt tokens (llama-bench), default 512
    n_gen:           int          # generation tokens (llama-bench), default 128
    n_reps:          int          # repetitions, default 3
    enable_thinking: bool         # MLX only: single-value fallback (deprecated by thinking_modes)
    # llama-bench sweep params (each list = values to sweep; llama-bench runs all combos)
    batch_sizes:     list[int]    = None   # -b  default [2048]
    ubatch_sizes:    list[int]    = None   # -ub default [512]
    flash_attns:     list[int]    = None   # -fa values to test, e.g. [0] [1] [0,1]
    cache_types_k:   list[str]   = None   # -ctk e.g. ["f16"] or ["f16","q8_0"]
    cache_types_v:   list[str]   = None   # -ctv same
    # MLX API sweep params
    thinking_modes:  list[bool]   = None   # [False], [True], or [False, True]
    n_gen_values:    list[int]    = None   # e.g. [128] or [128, 256, 512]

    def __post_init__(self):
        if self.batch_sizes   is None: self.batch_sizes   = [2048]
        if self.ubatch_sizes  is None: self.ubatch_sizes  = [512]
        if self.flash_attns   is None: self.flash_attns   = [0]
        if self.cache_types_k is None: self.cache_types_k = ["f16"]
        if self.cache_types_v is None: self.cache_types_v = ["f16"]
        if self.thinking_modes is None: self.thinking_modes = [self.enable_thinking]
        if self.n_gen_values   is None: self.n_gen_values   = [self.n_gen]


@dataclass
class BenchmarkResult:
    label:       str     # prompt name (API) or "PP" / "TG" (llama-bench)
    run:         int     # 1-based repetition index
    total_ms:    float   # wall time ms (API) or 0 for llama-bench (it reports tok/s directly)
    tokens_out:  int     # completion tokens (API) or n_gen/n_prompt (llama-bench)
    tok_per_sec: float   # tokens/sec
    error:       str     # non-empty if run failed


def _lbl(text: str, frame, bold: bool = False, right: bool = True) -> NSTextField:
    """Non-editable label."""
    f = NSTextField.alloc().initWithFrame_(frame)
    f.setStringValue_(text)
    f.setBezeled_(False)
    f.setDrawsBackground_(False)
    f.setEditable_(False)
    f.setSelectable_(False)
    f.setAlignment_(1 if right else 0)   # 1 = NSRightTextAlignment
    if bold:
        f.setFont_(NSFont.boldSystemFontOfSize_(11))
    return f


def _fld(text: str, frame) -> NSTextField:
    """Editable text field."""
    f = NSTextField.alloc().initWithFrame_(frame)
    f.setStringValue_(str(text))
    return f


def _btn(title: str, target, action: str, frame,
         key_eq: str | None = None) -> NSButton:
    """Rounded push button."""
    b = NSButton.alloc().initWithFrame_(frame)
    b.setTitle_(title)
    b.setBezelStyle_(1)    # NSBezelStyleRounded
    b.setButtonType_(0)    # NSMomentaryLightButton
    b.setTarget_(target)
    b.setAction_(action)
    if key_eq:
        b.setKeyEquivalent_(key_eq)
    return b


def _browse_btn(target, field: NSTextField,
                choose_dir: bool, frame) -> NSButton:
    """Browse button wired to open a file/folder picker for `field`.
    Uses NSView tag + handler._browse_map to associate button → (field, choose_dir),
    since arbitrary Python attributes cannot be set on PyObjC ObjC objects.
    """
    b = _btn("Browse…", target, "browse:", frame)
    if not hasattr(target, '_browse_map'):
        target._browse_map = {}
    tag = 1000 + len(target._browse_map)
    b.setTag_(tag)
    target._browse_map[tag] = (field, choose_dir)
    return b

class _PanelHandler(NSObject):
    """Shared NSObject action target for modal settings panels."""

    _sf: dict = {}          # sampling field refs — set per-instance by caller after init
    _thinking_chk = None   # NSButton checkbox for enable_thinking

    def modeChanged_(self, sender):
        is_api = sender.titleOfSelectedItem() == "API"
        if hasattr(self, '_bench_container'):
            self._bench_container.setHidden_(is_api)
        if hasattr(self, '_api_container'):
            self._api_container.setHidden_(not is_api)
        if hasattr(self, '_edit_btn_ref'):
            self._edit_btn_ref.setHidden_(not is_api)

    def presetChanged_(self, sender):
        name = sender.titleOfSelectedItem()
        if name not in SAMPLING_PRESETS:
            return
        preset = SAMPLING_PRESETS[name]
        for key, val in preset.items():
            if key in self._sf:
                txt = (f"{val:.2f}".rstrip("0").rstrip(".")
                       if isinstance(val, float) else str(val))
                self._sf[key].setStringValue_(txt)
        if self._thinking_chk is not None and "enable_thinking" in preset:
            self._thinking_chk.setState_(1 if preset["enable_thinking"] else 0)

    def clearHistory_(self, _s):
        try:
            BENCH_HISTORY_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        # Reload the WebView with the now-empty history
        if hasattr(self, '_history_wv'):
            self._history_wv.loadHTMLString_baseURL_(_bench_history_html(), None)

    def editPrompts_(self, _s):
        NSApp.stopModalWithCode_(_EDIT_PROMPTS_CODE)

    def stopOK_(self, _s):
        NSApp.stopModalWithCode_(NSModalResponseOK)

    def stopCancel_(self, _s):
        NSApp.stopModalWithCode_(0)

    def windowShouldClose_(self, _s):
        NSApp.stopModalWithCode_(0)
        return True

    def browse_(self, sender):
        entry = getattr(self, "_browse_map", {}).get(int(sender.tag()))
        if entry is None:
            return
        field, choose_dir = entry
        p = NSOpenPanel.openPanel()
        p.setCanChooseFiles_(not choose_dir)
        p.setCanChooseDirectories_(choose_dir)
        p.setAllowsMultipleSelection_(False)
        p.setResolvesAliases_(True)
        p.setPrompt_("Select")
        cur = Path(field.stringValue()).expanduser()
        initial = cur if cur.is_dir() else cur.parent
        if initial.exists():
            p.setDirectoryURL_(NSURL.fileURLWithPath_(str(initial)))
        NSApp.activateIgnoringOtherApps_(True)
        if p.runModal() == NSModalResponseOK:
            urls = p.URLs()
            if urls:
                field.setStringValue_(str(Path(urls[0].path())))


class _TestPromptHandler(NSObject):
    """Action target for the non-modal Quick Test Prompt window."""

    def send_(self, _s):
        prompt = self._input_fld.stringValue().strip()
        if not prompt or self._streaming:
            return
        self._output_tv.setString_("")
        self._tps_lbl.setStringValue_("…")
        self._streaming = True
        self._buf: list[str] = []
        self._t0 = time.time()
        self._tok_count = 0
        self._drain_timer = rumps.Timer(self.drainTick_, 0.1)
        self._drain_timer.start()
        threading.Thread(target=self._do_stream,
                         args=(prompt,), daemon=True).start()

    def _do_stream(self, prompt):
        try:
            cfg = self._app_ref._cfg
            port = cfg.get("omlx_port", 8000)
            api_key = cfg.get("omlx_api_key", "")
            model = self._app_ref._active or ""
            p = self._app_ref._model_params(model) if hasattr(self._app_ref, "_model_params") else {}
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": p.get("max_tokens", 512) if p else 512,
                "stream": True,
            }).encode()
            req = urllib.request.Request(
                f"http://localhost:{port}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            self._buf.append(delta)
                            self._tok_count += 1
                    except Exception:
                        pass
        except Exception as e:
            self._buf.append(f"\n\n[Error: {e}]")
        finally:
            self._streaming = False

    def drainTick_(self, timer):
        if self._buf:
            chunk = "".join(self._buf)
            self._buf.clear()
            ts = self._output_tv.textStorage()
            from AppKit import NSAttributedString
            ts.appendAttributedString_(
                NSAttributedString.alloc().initWithString_(chunk))
            self._output_tv.scrollRangeToVisible_(
                (self._output_tv.string().length(), 0))
        elapsed = time.time() - self._t0
        if self._tok_count > 0 and elapsed > 0:
            self._tps_lbl.setStringValue_(
                f"{self._tok_count / elapsed:.1f} tok/s  ({self._tok_count} tokens)")
        if not self._streaming and not self._buf:
            timer.stop()

    def clear_(self, _s):
        self._output_tv.setString_("")
        self._tps_lbl.setStringValue_("")
        self._input_fld.setStringValue_("")


def _make_test_prompt_window(app) -> NSWindow:
    """Build and return the Quick Test Prompt NSWindow (non-modal)."""
    W, H = 580, 440
    BOT = 48
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 7, NSBackingStoreBuffered, False)
    win.setTitle_("Quick Test Prompt")
    win.center()
    cv = win.contentView()

    handler = _TestPromptHandler.alloc().init()
    handler._app_ref = app
    handler._streaming = False
    handler._buf = []
    handler._t0 = 0.0
    handler._tok_count = 0
    handler._drain_timer = None
    win._handler = handler   # keep handler alive with window

    INPUT_H = 28
    cv.addSubview_(_lbl("Prompt:", ((_PAD, H - _PAD - INPUT_H), (_LW, INPUT_H)),
                        right=False))
    input_fld = NSTextField.alloc().initWithFrame_(
        ((_PAD + _LW + _GAP, H - _PAD - INPUT_H),
         (W - _PAD*2 - _LW - _GAP - 70, INPUT_H)))
    input_fld.setFont_(NSFont.systemFontOfSize_(13))
    cv.addSubview_(input_fld)
    handler._input_fld = input_fld

    send_btn = _btn("Send", handler, "send:",
                    ((W - _PAD - 64, H - _PAD - INPUT_H), (64, INPUT_H)), "\r")
    cv.addSubview_(send_btn)

    scroll = NSScrollView.alloc().initWithFrame_(
        ((_PAD, BOT), (W - _PAD*2, H - _PAD*2 - INPUT_H - 8 - BOT)))
    scroll.setHasVerticalScroller_(True); scroll.setAutohidesScrollers_(True)
    tv = NSTextView.alloc().initWithFrame_(((0, 0), (W - _PAD*2, 300)))
    tv.setFont_(NSFont.userFixedPitchFontOfSize_(12))
    tv.setEditable_(False)
    scroll.setDocumentView_(tv); cv.addSubview_(scroll)
    handler._output_tv = tv

    tps_lbl = NSTextField.alloc().initWithFrame_(
        ((_PAD, 12), (W - _PAD*2 - 70, 22)))
    tps_lbl.setBezeled_(False); tps_lbl.setDrawsBackground_(False)
    tps_lbl.setEditable_(False)
    tps_lbl.setFont_(NSFont.systemFontOfSize_(11))
    cv.addSubview_(tps_lbl)
    handler._tps_lbl = tps_lbl

    cv.addSubview_(_btn("Clear", handler, "clear:",
                        ((W - _PAD - 64, 12), (64, 22))))
    return win


def run_settings_panel(cfg: dict) -> bool:
    """Show global settings panel. Modifies cfg in-place. Returns True if saved."""
    W   = 520
    BW  = 72                              # browse button width
    x_lbl = _PAD
    x_fld = _PAD + _LW + _GAP
    FWB   = W - x_fld - 6 - BW - _PAD   # field width when Browse follows
    FW    = W - x_fld - _PAD             # field width (no browse)
    x_btn = x_fld + FWB + 6

    H = (_PAD
         + _SH + _SG + 4 * (_RH + _RG)            # Paths (4 rows)
         + _DG + _SH + _SG + 3 * (_RH + _RG)      # oMLX (3 rows)
         + _DG + _SH + _SG + 2 * (_RH + _RG) - _RG  # Behavior (2 rows)
         + _DG + _SH + _SG + 4 * (_RH + _RG)        # Sync (4 checkboxes)
         + _DG + _BTN_H + _BTN_BOT)

    def fy(from_top: int, h: int = _RH) -> int:
        return H - from_top - h

    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
    # style mask 3 = NSWindowStyleMaskTitled (1) | NSWindowStyleMaskClosable (2)
    panel.setTitle_("Model Switcher — Settings")
    panel.setDelegate_(handler)
    panel.center()
    cv = panel.contentView()
    fields: dict[str, NSTextField] = {}
    cur = _PAD  # pixels from top, advances downward

    # ── Paths ──
    cv.addSubview_(_lbl("Paths", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                        bold=True, right=False))
    cur += _SH + _SG
    for key, label, is_dir in [
        ("mlx_dir",         "MLX models dir",     True),
        ("gguf_dir",        "GGUF models dir",     True),
        ("llama_server",    "llama-server binary", False),
        ("opencode_config", "opencode config",     False),
    ]:
        cv.addSubview_(_lbl(label + ":", ((x_lbl, fy(cur)), (_LW, _RH))))
        f = _fld(cfg[key], ((x_fld, fy(cur)), (FWB, _RH)))
        cv.addSubview_(f)
        cv.addSubview_(_browse_btn(handler, f, is_dir,
                                    ((x_btn, fy(cur) - 3), (BW, _RH + 6))))
        fields[key] = f
        cur += _RH + _RG
    cur += _DG

    # ── oMLX ──
    cv.addSubview_(_lbl("oMLX", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                        bold=True, right=False))
    cur += _SH + _SG
    for key, label in [
        ("omlx_port",    "Port"),
        ("omlx_api_key", "API key"),
        ("omlx_service", "Service name"),
    ]:
        cv.addSubview_(_lbl(label + ":", ((x_lbl, fy(cur)), (_LW, _RH))))
        f = _fld(cfg[key], ((x_fld, fy(cur)), (FW, _RH)))
        cv.addSubview_(f)
        fields[key] = f
        cur += _RH + _RG
    cur += _DG

    # ── Behavior ──
    cv.addSubview_(_lbl("Behavior", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                        bold=True, right=False))
    cur += _SH + _SG

    chk = NSButton.alloc().initWithFrame_(((x_fld, fy(cur)), (FW, _RH)))
    chk.setButtonType_(3)   # NSSwitchButton / checkbox
    chk.setTitle_("Restart opencode on model switch")
    chk.setState_(1 if cfg["restart_opencode"] else 0)
    cv.addSubview_(chk)
    cur += _RH + _RG

    cv.addSubview_(_lbl("Terminal app:", ((x_lbl, fy(cur)), (_LW, _RH))))
    term_popup = NSPopUpButton.alloc().initWithFrame_(
        ((x_fld, fy(cur) - 2), (140, _RH + 4)))
    for opt in ["Terminal", "iTerm2"]:
        term_popup.addItemWithTitle_(opt)
    term_popup.selectItemWithTitle_(cfg.get("terminal_app", "Terminal"))
    cv.addSubview_(term_popup)
    cur += _RH + _RG + _DG

    # ── Sync & Notifications ──
    cv.addSubview_(_lbl("Sync & Notifications", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                        bold=True, right=False))
    cur += _SH + _SG
    sync_chks = {}
    for key, title in [
        ("notifications",  "macOS notification when model ready"),
        ("sync_env",       "Write env file (LLM_BASE_URL) on switch"),
        ("sync_cursor",    "Sync Cursor MCP config on switch"),
        ("sync_continue",  "Sync Continue.dev config on switch"),
    ]:
        c = NSButton.alloc().initWithFrame_(((x_fld, fy(cur)), (FW, _RH)))
        c.setButtonType_(3)
        c.setTitle_(title)
        c.setState_(1 if cfg.get(key, True) else 0)
        cv.addSubview_(c)
        sync_chks[key] = c
        cur += _RH + _RG

    # ── Buttons ──
    cv.addSubview_(_btn("Cancel", handler, "stopCancel:",
                        ((W - _PAD - 152, _BTN_BOT), (72, _BTN_H)), "\x1b"))
    cv.addSubview_(_btn("Save", handler, "stopOK:",
                        ((W - _PAD - 74, _BTN_BOT), (66, _BTN_H)), "\r"))

    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    result = NSApp.runModalForWindow_(panel)
    panel.orderOut_(None)

    if result != NSModalResponseOK:
        return False

    # Read back values into cfg
    for key in ("mlx_dir", "gguf_dir", "llama_server", "opencode_config",
                "omlx_api_key", "omlx_service"):
        cfg[key] = fields[key].stringValue()
    try:
        cfg["omlx_port"] = int(fields["omlx_port"].stringValue())
    except ValueError:
        pass
    cfg["restart_opencode"] = bool(chk.state())
    cfg["terminal_app"] = term_popup.titleOfSelectedItem()
    for key, c in sync_chks.items():
        cfg[key] = bool(c.state())
    return True

def run_model_settings_panel(cfg: dict, name: str, kind: str) -> bool:
    """Show per-model settings panel. Modifies cfg in-place. Returns True if saved."""
    W       = 440
    x_lbl   = _PAD
    x_fld   = _PAD + _LW + _GAP
    FW      = W - x_fld - _PAD
    is_gguf = kind == "gguf"
    n_limit = 3 if is_gguf else 2   # context + max_tokens [+ gpu_layers]

    H = (_PAD
         + _SH + _SG + 2 * (_RH + _RG)                    # Identity (alias + note)
         + _DG + _SH + _SG + n_limit * (_RH + _RG)        # Limits
         + _DG + _SH + _SG + 8 * (_RH + _RG) - _RG        # Sampling (preset + 6 + checkbox)
         + _DG + _BTN_H + _BTN_BOT)

    def fy(from_top: int, h: int = _RH) -> int:
        return H - from_top - h

    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
    panel.setTitle_(f"Settings — {name}")
    panel.setDelegate_(handler)
    panel.center()
    cv = panel.contentView()

    p = model_params(cfg, name)
    cur = _PAD

    # ── Identity ──
    cv.addSubview_(_lbl("Identity", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                        bold=True, right=False))
    cur += _SH + _SG
    cv.addSubview_(_lbl("Alias:", ((x_lbl, fy(cur)), (_LW, _RH))))
    alias_fld = _fld(cfg["aliases"].get(name, ""), ((x_fld, fy(cur)), (FW, _RH)))
    cv.addSubview_(alias_fld)
    cur += _RH + _RG
    cv.addSubview_(_lbl("Note:", ((x_lbl, fy(cur)), (_LW, _RH))))
    note_fld = _fld(cfg["model_notes"].get(name, ""), ((x_fld, fy(cur)), (FW, _RH)))
    note_fld.setPlaceholderString_("e.g. ctv=f16 only, good for coding")
    cv.addSubview_(note_fld)
    cur += _RH + _RG + _DG

    # ── Limits ──
    cv.addSubview_(_lbl("Limits", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                        bold=True, right=False))
    cur += _SH + _SG
    limit_flds: dict[str, NSTextField] = {}
    for key, label in [("context", "Context"), ("max_tokens", "Max tokens")]:
        cv.addSubview_(_lbl(label + ":", ((x_lbl, fy(cur)), (_LW, _RH))))
        f = _fld(p[key], ((x_fld, fy(cur)), (FW, _RH)))
        cv.addSubview_(f)
        limit_flds[key] = f
        cur += _RH + _RG
    gpu_fld: NSTextField | None = None
    if is_gguf:
        cv.addSubview_(_lbl("GPU layers:", ((x_lbl, fy(cur)), (_LW, _RH))))
        gpu_fld = _fld(p["gpu_layers"], ((x_fld, fy(cur)), (FW, _RH)))
        cv.addSubview_(gpu_fld)
        cur += _RH + _RG
    cur += _DG

    # ── Sampling ──
    cv.addSubview_(_lbl("Sampling", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                        bold=True, right=False))
    cur += _SH + _SG

    # Preset popup — selecting a preset immediately overwrites all 6 fields below
    cv.addSubview_(_lbl("Preset:", ((x_lbl, fy(cur)), (_LW, _RH))))
    preset_popup = NSPopUpButton.alloc().initWithFrame_(
        ((x_fld, fy(cur) - 2), (FW, _RH + 4)))
    preset_popup.addItemWithTitle_("Custom")
    for pn in SAMPLING_PRESETS:
        preset_popup.addItemWithTitle_(pn)
    cv.addSubview_(preset_popup)
    cur += _RH + _RG

    sf: dict[str, NSTextField] = {}
    for key, label in [
        ("temperature",        "Temperature"),
        ("top_p",              "Top P"),
        ("top_k",              "Top K"),
        ("min_p",              "Min P"),
        ("presence_penalty",   "Presence penalty"),
        ("repetition_penalty", "Repetition penalty"),
    ]:
        cv.addSubview_(_lbl(label + ":", ((x_lbl, fy(cur)), (_LW, _RH))))
        val = p[key]
        txt = (f"{val:.2f}".rstrip("0").rstrip(".")
               if isinstance(val, float) else str(val))
        f = _fld(txt, ((x_fld, fy(cur)), (FW, _RH)))
        cv.addSubview_(f)
        sf[key] = f
        cur += _RH + _RG

    # Enable thinking checkbox
    thinking_chk = NSButton.alloc().initWithFrame_(
        ((x_fld, fy(cur)), (FW, _RH)))
    thinking_chk.setTitle_("Enable thinking mode")
    thinking_chk.setButtonType_(3)    # NSSwitchButton (checkbox)
    thinking_chk.setState_(1 if p.get("enable_thinking", False) else 0)
    cv.addSubview_(thinking_chk)
    cur += _RH + _RG

    # Wire preset popup to handler AFTER sf is populated
    handler._sf = sf
    handler._thinking_chk = thinking_chk
    preset_popup.setTarget_(handler)
    preset_popup.setAction_("presetChanged:")

    # ── Buttons ──
    cv.addSubview_(_btn("Cancel", handler, "stopCancel:",
                        ((W - _PAD - 152, _BTN_BOT), (72, _BTN_H)), "\x1b"))
    cv.addSubview_(_btn("Save", handler, "stopOK:",
                        ((W - _PAD - 74, _BTN_BOT), (66, _BTN_H)), "\r"))

    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    result = NSApp.runModalForWindow_(panel)
    panel.orderOut_(None)

    if result != NSModalResponseOK:
        return False

    # Read alias
    alias_text = alias_fld.stringValue().strip()
    if alias_text:
        cfg["aliases"][name] = alias_text
    else:
        cfg["aliases"].pop(name, None)

    # Read note
    note_text = note_fld.stringValue().strip()
    if note_text:
        cfg["model_notes"][name] = note_text
    else:
        cfg["model_notes"].pop(name, None)

    # Read limits
    mp = cfg["model_params"].setdefault(name, {})
    for key in ("context", "max_tokens"):
        try:
            mp[key] = int(limit_flds[key].stringValue())
        except ValueError:
            pass
    if is_gguf and gpu_fld:
        try:
            mp["gpu_layers"] = int(gpu_fld.stringValue())
        except ValueError:
            pass

    # Read sampling — top_k is int, rest are float
    for key in ("top_k",):
        try:
            mp[key] = int(sf[key].stringValue())
        except ValueError:
            pass
    for key in ("temperature", "top_p", "min_p",
                "presence_penalty", "repetition_penalty"):
        try:
            mp[key] = float(sf[key].stringValue())
        except ValueError:
            pass
    mp["enable_thinking"] = bool(thinking_chk.state())

    return True


# ── Benchmarking ──────────────────────────────────────────────────────────────

_EDIT_PROMPTS_CODE = 99   # stopModalWithCode_ value used by "Edit Prompts…" button


def run_edit_prompts_panel() -> None:
    """Show a card-based editor for benchmark_prompts.json (one card per prompt)."""
    load_benchmark_prompts()
    try:
        prompts_dict = json.loads(BENCHMARK_PROMPTS_PATH.read_text())
        if not isinstance(prompts_dict, dict):
            prompts_dict = {}
    except Exception:
        prompts_dict = {}

    prompts = [[k, v] for k, v in prompts_dict.items()]
    if not prompts:
        prompts = [["New Prompt", ""]]

    W           = 560
    H           = 500
    BOTTOM      = 56      # button bar height
    CARD_PAD    = 14
    NAME_H      = 22
    TEXT_H      = 80
    NAME_GAP    = 5       # gap between name and text
    CARD_BOT    = 12      # bottom margin inside card
    CARD_H      = NAME_H + NAME_GAP + TEXT_H + CARD_BOT   # 119
    SEP_SPACE   = 10      # vertical space for separator between cards

    n = len(prompts)
    content_h = n * CARD_H + max(0, n - 1) * SEP_SPACE
    content_h = max(content_h, H - BOTTOM)

    handler = _PanelHandler.alloc().init()
    handler._name_fields = []
    handler._text_views  = []

    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
    panel.setTitle_("Edit Benchmark Prompts")
    panel.setDelegate_(handler)
    panel.center()
    cv = panel.contentView()

    # ── Outer scroll view ──
    outer_scroll = NSScrollView.alloc().initWithFrame_(((0, BOTTOM), (W, H - BOTTOM)))
    outer_scroll.setHasVerticalScroller_(True)
    outer_scroll.setAutohidesScrollers_(True)

    doc = NSView.alloc().initWithFrame_(((0, 0), (W, content_h)))

    y = content_h   # AppKit origin is bottom-left; we build top-down by subtracting
    for i, (name, text) in enumerate(prompts):
        y -= CARD_H
        name_y = y + CARD_BOT + TEXT_H + NAME_GAP
        text_y = y + CARD_BOT

        # Label above name field
        lbl = NSTextField.alloc().initWithFrame_(
            ((CARD_PAD, name_y), (W - CARD_PAD * 2, NAME_H)))
        lbl.setStringValue_(name)
        lbl.setFont_(NSFont.boldSystemFontOfSize_(12))
        lbl.setEditable_(True)
        lbl.setBezeled_(True)
        lbl.setDrawsBackground_(True)
        doc.addSubview_(lbl)
        handler._name_fields.append(lbl)

        # Text view wrapped in a mini scroll view (gives border + scrolling)
        tv_wrap = NSScrollView.alloc().initWithFrame_(
            ((CARD_PAD, text_y), (W - CARD_PAD * 2, TEXT_H)))
        tv_wrap.setHasVerticalScroller_(True)
        tv_wrap.setAutohidesScrollers_(True)
        tv_wrap.setBorderType_(2)   # NSBezelBorder

        tv = NSTextView.alloc().initWithFrame_(((0, 0), (W - CARD_PAD * 2, TEXT_H)))
        tv.setString_(text)
        tv.setFont_(NSFont.systemFontOfSize_(12))
        tv.setAutomaticQuoteSubstitutionEnabled_(False)
        tv.setAutomaticSpellingCorrectionEnabled_(False)
        tv_wrap.setDocumentView_(tv)
        doc.addSubview_(tv_wrap)
        handler._text_views.append(tv)

        # Separator between cards
        if i < n - 1:
            sep = NSBox.alloc().initWithFrame_(
                ((CARD_PAD, y - SEP_SPACE // 2), (W - CARD_PAD * 2, 1)))
            sep.setBoxType_(2)   # NSBoxSeparator
            doc.addSubview_(sep)
            y -= SEP_SPACE

    outer_scroll.setDocumentView_(doc)
    # Scroll to top
    outer_scroll.documentView().scrollPoint_((0, content_h))
    cv.addSubview_(outer_scroll)

    # ── Bottom buttons ──
    note = NSTextField.alloc().initWithFrame_(((CARD_PAD, 18), (W - 220, 18)))
    note.setStringValue_("Edit label (bold) and prompt text per card.")
    note.setBezeled_(False)
    note.setDrawsBackground_(False)
    note.setEditable_(False)
    note.setFont_(NSFont.systemFontOfSize_(10))
    cv.addSubview_(note)
    cv.addSubview_(_btn("Cancel", handler, "stopCancel:",
                        ((W - _PAD - 162, 14), (82, 28)), "\x1b"))
    cv.addSubview_(_btn("Save", handler, "stopOK:",
                        ((W - _PAD - 74, 14), (60, 28)), "\r"))

    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    result = NSApp.runModalForWindow_(panel)
    panel.orderOut_(None)

    if result != NSModalResponseOK:
        return

    new_prompts = {}
    for name_fld, tv in zip(handler._name_fields, handler._text_views):
        name = name_fld.stringValue().strip()
        text = tv.string().strip()
        if name and text:
            new_prompts[name] = text

    if not new_prompts:
        rumps.alert(title="No prompts", message="At least one prompt is required.")
        return

    BENCHMARK_PROMPTS_PATH.write_text(json.dumps(new_prompts, indent=2) + "\n")


def run_benchmark_config_panel(name: str, kind: str, cfg: dict) -> BenchmarkConfig | None:
    """Show benchmark config modal. GGUF → llama-bench, MLX → API."""
    BENCHMARK_PROMPTS = load_benchmark_prompts()
    W = 460
    is_gguf = kind == "gguf"
    n_prompts = len(BENCHMARK_PROMPTS)

    # SECTION_H: height of the mode-specific content area (same for both GGUF modes)
    # llama-bench: header(20)+3 rows(90)+gap(16)+header(20)+3 rows(90)
    #              +cache_K label(20)+chks(30)+cache_V label(20)+chks(30) = 336
    # API:         header(20)+n_prompts(150)+gap(16)+header(20)+2 rows(60)=266 ≤ 336
    SECTION_H = (_SH + _SG + 3 * (_RH + _RG)
                 + _DG + _SH + _SG + 3 * (_RH + _RG)
                 + 2 * (_SH + _SG + _RH + _RG))   # 336

    if is_gguf:
        H = (_PAD
             + (_RH + _RG) + _DG             # Mode selector row
             + SECTION_H                      # content (llama-bench or API)
             + _DG + _BTN_H + _BTN_BOT)
    else:
        # MLX: API prompts + reps + gen tokens + thinking sweep
        H = (_PAD
             + _SH + _SG + n_prompts * (_RH + _RG)   # prompts
             + _DG + _SH + _SG + 3 * (_RH + _RG)     # reps + gen tokens + thinking
             + _DG + _BTN_H + _BTN_BOT)

    def fy(from_top: int, h: int = _RH) -> int:
        return H - from_top - h

    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
    mode_label = "llama-bench" if is_gguf else "API"
    panel.setTitle_(f"Benchmark — {name}  ({mode_label})")
    panel.setDelegate_(handler)
    panel.center()
    cv = panel.contentView()

    x_lbl = _PAD
    x_fld = _PAD + _LW + _GAP
    FW = W - x_fld - _PAD
    cur = _PAD

    prompt_chks: list[NSButton] = []
    prompt_names = list(BENCHMARK_PROMPTS.keys())
    nprompt_fld = ngen_fld = reps_fld = thinking_chk = thinking_popup = None
    gguf_mode_popup = api_reps_fld = api_ngen_fld = None

    # sweep popup helper — returns NSPopUpButton
    def _popup(options, frame):
        p = NSPopUpButton.alloc().initWithFrame_(frame)
        for o in options:
            p.addItemWithTitle_(o)
        return p

    batch_fld = ubatch_fld = fa_popup = None
    ctk_chks: list = []
    ctv_chks: list = []

    if is_gguf:
        # ── Mode selector ──
        panel.setTitle_(f"Benchmark — {name}")
        cv.addSubview_(_lbl("Mode:", ((x_lbl, fy(cur)), (_LW, _RH))))
        gguf_mode_popup = _popup(["llama-bench", "API"], ((x_fld, fy(cur) - 2), (FW, _RH + 4)))
        gguf_mode_popup.setTarget_(handler)
        gguf_mode_popup.setAction_("modeChanged:")
        cv.addSubview_(gguf_mode_popup)
        cur += _RH + _RG + _DG

        # Both containers share the same frame; only one is visible at a time
        container_frame = ((0, fy(cur, SECTION_H)), (W, SECTION_H))

        def sfy(from_top, h=_RH):
            """Coords within a SECTION_H-tall container (top-down)."""
            return SECTION_H - from_top - h

        # ── bench_container: llama-bench + sweep ──
        bench_container = NSView.alloc().initWithFrame_(container_frame)
        sc = 0
        bench_container.addSubview_(_lbl("llama-bench",
            ((_PAD, sfy(sc, _SH)), (W - 2*_PAD, _SH)), bold=True, right=False))
        sc += _SH + _SG
        bench_container.addSubview_(_lbl("Prompt tokens:", ((x_lbl, sfy(sc)), (_LW, _RH))))
        nprompt_fld = _fld("512", ((x_fld, sfy(sc)), (FW, _RH)))
        bench_container.addSubview_(nprompt_fld); sc += _RH + _RG
        bench_container.addSubview_(_lbl("Gen tokens:", ((x_lbl, sfy(sc)), (_LW, _RH))))
        ngen_fld = _fld("128", ((x_fld, sfy(sc)), (FW, _RH)))
        bench_container.addSubview_(ngen_fld); sc += _RH + _RG
        bench_container.addSubview_(_lbl("Repetitions:", ((x_lbl, sfy(sc)), (_LW, _RH))))
        reps_fld = _fld("3", ((x_fld, sfy(sc)), (FW, _RH)))
        bench_container.addSubview_(reps_fld); sc += _RH + _RG + _DG
        bench_container.addSubview_(_lbl("Sweep",
            ((_PAD, sfy(sc, _SH)), (W - 2*_PAD, _SH)), bold=True, right=False))
        sc += _SH + _SG
        bench_container.addSubview_(_lbl("Batch (-b):", ((x_lbl, sfy(sc)), (_LW, _RH))))
        batch_fld = _fld("2048", ((x_fld, sfy(sc)), (FW, _RH)))
        bench_container.addSubview_(batch_fld); sc += _RH + _RG
        bench_container.addSubview_(_lbl("Ubatch (-ub):", ((x_lbl, sfy(sc)), (_LW, _RH))))
        ubatch_fld = _fld("512", ((x_fld, sfy(sc)), (FW, _RH)))
        bench_container.addSubview_(ubatch_fld); sc += _RH + _RG
        bench_container.addSubview_(_lbl("Flash attn:", ((x_lbl, sfy(sc)), (_LW, _RH))))
        fa_popup = _popup(["off", "on", "sweep off+on"], ((x_fld, sfy(sc)-2), (FW, _RH+4)))
        bench_container.addSubview_(fa_popup); sc += _RH + _RG
        chk_w = (W - 2 * _PAD) // len(CACHE_QUANT_TYPES)
        bench_container.addSubview_(_lbl("Cache type K:",
            ((_PAD, sfy(sc, _SH)), (W - 2*_PAD, _SH)), right=False))
        sc += _SH + _SG
        for j, qt in enumerate(CACHE_QUANT_TYPES):
            chk = NSButton.alloc().initWithFrame_(((_PAD + j*chk_w, sfy(sc)), (chk_w, _RH)))
            chk.setTitle_(qt); chk.setButtonType_(3)
            chk.setState_(1 if qt == "f16" else 0)
            chk.setFont_(NSFont.systemFontOfSize_(11))
            bench_container.addSubview_(chk); ctk_chks.append(chk)
        sc += _RH + _RG
        bench_container.addSubview_(_lbl("Cache type V:",
            ((_PAD, sfy(sc, _SH)), (W - 2*_PAD, _SH)), right=False))
        sc += _SH + _SG
        for j, qt in enumerate(CACHE_QUANT_TYPES):
            chk = NSButton.alloc().initWithFrame_(((_PAD + j*chk_w, sfy(sc)), (chk_w, _RH)))
            chk.setTitle_(qt); chk.setButtonType_(3)
            chk.setState_(1 if qt == "f16" else 0)
            chk.setFont_(NSFont.systemFontOfSize_(11))
            bench_container.addSubview_(chk); ctv_chks.append(chk)
        cv.addSubview_(bench_container)
        handler._bench_container = bench_container

        # ── api_container: prompts + options (no thinking — GGUF doesn't support it) ──
        api_container = NSView.alloc().initWithFrame_(container_frame)
        ac = 0
        api_container.addSubview_(_lbl("Prompts",
            ((_PAD, sfy(ac, _SH)), (W - 2*_PAD, _SH)), bold=True, right=False))
        ac += _SH + _SG
        for pname in prompt_names:
            chk = NSButton.alloc().initWithFrame_(((x_fld, sfy(ac)), (FW, _RH)))
            chk.setTitle_(pname); chk.setButtonType_(3); chk.setState_(1)
            api_container.addSubview_(chk)
            prompt_chks.append(chk); ac += _RH + _RG
        ac += _DG
        api_container.addSubview_(_lbl("Options",
            ((_PAD, sfy(ac, _SH)), (W - 2*_PAD, _SH)), bold=True, right=False))
        ac += _SH + _SG
        api_container.addSubview_(_lbl("Repetitions:", ((x_lbl, sfy(ac)), (_LW, _RH))))
        api_reps_fld = _fld("3", ((x_fld, sfy(ac)), (FW, _RH)))
        api_container.addSubview_(api_reps_fld); ac += _RH + _RG
        api_container.addSubview_(_lbl("Gen tokens:", ((x_lbl, sfy(ac)), (_LW, _RH))))
        api_ngen_fld = _fld("128", ((x_fld, sfy(ac)), (FW, _RH)))
        api_container.addSubview_(api_ngen_fld)
        api_container.setHidden_(True)   # llama-bench is default
        cv.addSubview_(api_container)
        handler._api_container = api_container
    else:
        # ── API prompts ──
        cv.addSubview_(_lbl("Prompts", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                            bold=True, right=False))
        cur += _SH + _SG
        for pname in prompt_names:
            chk = NSButton.alloc().initWithFrame_(((x_fld, fy(cur)), (FW, _RH)))
            chk.setTitle_(pname)
            chk.setButtonType_(3)
            chk.setState_(1)
            cv.addSubview_(chk)
            prompt_chks.append(chk)
            cur += _RH + _RG
        cur += _DG
        # ── Options ──
        cv.addSubview_(_lbl("Options", ((_PAD, fy(cur, _SH)), (W - 2 * _PAD, _SH)),
                            bold=True, right=False))
        cur += _SH + _SG
        cv.addSubview_(_lbl("Repetitions:", ((x_lbl, fy(cur)), (_LW, _RH))))
        reps_fld = _fld("3", ((x_fld, fy(cur)), (FW, _RH)))
        cv.addSubview_(reps_fld)
        cur += _RH + _RG
        cv.addSubview_(_lbl("Gen tokens:", ((x_lbl, fy(cur)), (_LW, _RH))))
        ngen_fld = _fld("128", ((x_fld, fy(cur)), (FW, _RH)))
        cv.addSubview_(ngen_fld)
        cur += _RH + _RG
        cv.addSubview_(_lbl("Thinking mode:", ((x_lbl, fy(cur)), (_LW, _RH))))
        thinking_popup = _popup(["off", "on", "sweep off+on"],
                                ((x_fld, fy(cur) - 2), (FW, _RH + 4)))
        cv.addSubview_(thinking_popup)
        cur += _RH + _RG

    # ── Buttons ──
    edit_btn = _btn("Edit Prompts…", handler, "editPrompts:",
                    ((_PAD, _BTN_BOT), (100, _BTN_H)))
    # For GGUF, Edit Prompts is hidden until user switches to API mode (modeChanged_ shows it)
    edit_btn.setHidden_(is_gguf)
    handler._edit_btn_ref = edit_btn
    cv.addSubview_(edit_btn)
    cv.addSubview_(_btn("Cancel", handler, "stopCancel:",
                        ((W - _PAD - 162, _BTN_BOT), (82, _BTN_H)), "\x1b"))
    cv.addSubview_(_btn("Run Benchmark", handler, "stopOK:",
                        ((W - _PAD - 74, _BTN_BOT), (66, _BTN_H)), "\r"))

    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    result = NSApp.runModalForWindow_(panel)
    panel.orderOut_(None)

    if result == _EDIT_PROMPTS_CODE:
        run_edit_prompts_panel()
        # Re-open benchmark config with freshly loaded prompts
        return run_benchmark_config_panel(name, kind, cfg)

    if result != NSModalResponseOK:
        return None

    if is_gguf and gguf_mode_popup:
        mode = gguf_mode_popup.titleOfSelectedItem().lower()   # "llama-bench" or "api"
        # For GGUF API mode, use the api container's fields
        if mode == "api":
            reps_fld = api_reps_fld
            ngen_fld = api_ngen_fld
    else:
        mode = "api"

    if mode == "api":
        selected = [prompt_names[i] for i, chk in enumerate(prompt_chks) if chk.state()]
        if not selected:
            return None
    else:
        selected = []

    def _ints(fld, default):
        """Parse comma-separated ints from a text field."""
        try:
            return [int(x.strip()) for x in fld.stringValue().split(",") if x.strip()]
        except (ValueError, AttributeError):
            return [default]


    try:
        n_prompt = int(nprompt_fld.stringValue()) if nprompt_fld else 512
    except (ValueError, AttributeError):
        n_prompt = 512
    try:
        n_gen = int(ngen_fld.stringValue().split(",")[0].strip()) if ngen_fld else 128
    except (ValueError, AttributeError):
        n_gen = 128
    try:
        n_reps = int(reps_fld.stringValue()) if reps_fld else 3
    except (ValueError, AttributeError):
        n_reps = 3

    # Sweep params (GGUF only)
    batch_sizes  = _ints(batch_fld, 2048)  if batch_fld  else [2048]
    ubatch_sizes = _ints(ubatch_fld, 512)  if ubatch_fld else [512]

    if fa_popup:
        sel = fa_popup.titleOfSelectedItem()
        flash_attns = [0, 1] if sel == "sweep off+on" else ([1] if sel == "on" else [0])
    else:
        flash_attns = [0]

    ctk_list = [qt for qt, c in zip(CACHE_QUANT_TYPES, ctk_chks) if c.state()] or ["f16"]
    ctv_list = [qt for qt, c in zip(CACHE_QUANT_TYPES, ctv_chks) if c.state()] or ["f16"]

    # API sweep params (MLX and GGUF API mode both support comma-separated gen tokens)
    n_gen_values = _ints(ngen_fld, 128) if (ngen_fld and mode == "api") else [n_gen]

    if thinking_popup:
        t_sel = thinking_popup.titleOfSelectedItem()
        thinking_modes = [False, True] if t_sel == "sweep off+on" else ([True] if t_sel == "on" else [False])
    else:
        thinking_modes = [False]

    return BenchmarkConfig(
        mode=mode,
        prompts=selected,
        n_prompt=n_prompt,
        n_gen=n_gen,
        n_reps=n_reps,
        enable_thinking=thinking_modes[0] if thinking_modes else False,
        batch_sizes=batch_sizes,
        ubatch_sizes=ubatch_sizes,
        flash_attns=flash_attns,
        cache_types_k=ctk_list,
        cache_types_v=ctv_list,
        thinking_modes=thinking_modes,
        n_gen_values=n_gen_values,
    )


def run_api_benchmark(cfg: dict, name: str, kind: str,
                      bconfig: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run API benchmarks in background. Returns list of BenchmarkResult."""
    results = []
    port = cfg.get("omlx_port", 8000)
    api_key = cfg.get("omlx_api_key", "")
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}

    if kind == "gguf":
        # For GGUF API benchmarks, llama-server must already be running (load the model first).
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                pass
        except OSError:
            return [BenchmarkResult(label="error", run=1, total_ms=0, tokens_out=0,
                                    tok_per_sec=0,
                                    error="llama-server not running — load the GGUF model first.")]
    else:
        # For MLX, ensure oMLX is running (may have been stopped by a prior llama-bench run)
        if not omlx_is_healthy(cfg):
            if not omlx_start(cfg):
                return [BenchmarkResult(label="error", run=1, total_ms=0,
                                        tokens_out=0, tok_per_sec=0,
                                        error="oMLX failed to start. Try selecting the model first.")]

    model_id = name

    p = model_params(cfg, name)
    all_prompts = load_benchmark_prompts()

    thinking_modes = bconfig.thinking_modes or [bconfig.enable_thinking]
    n_gen_values   = bconfig.n_gen_values   or [bconfig.n_gen]

    for thinking in thinking_modes:
        for n_gen in n_gen_values:
            # Build a label suffix only when sweeping — keeps single-run labels clean
            sweep_suffix = ""
            if len(thinking_modes) > 1 or len(n_gen_values) > 1:
                sweep_suffix = f"  thinking={'on' if thinking else 'off'} gen={n_gen}"

            for prompt_name in bconfig.prompts:
                prompt_text = all_prompts.get(prompt_name, prompt_name)
                label = prompt_name + sweep_suffix
                for run_i in range(bconfig.n_reps):
                    body = {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": n_gen,
                        "stream": False,
                    }
                    if kind == "mlx":
                        body.update(mlx_sampling_params(
                            {**p, "enable_thinking": thinking}))
                    else:
                        body.update(llama_sampling_params(p))

                    t0 = time.time()
                    resp = http_post(
                        f"http://localhost:{port}/v1/chat/completions",
                        body, headers, timeout=300)
                    elapsed_ms = (time.time() - t0) * 1000

                    if not resp or "choices" not in resp:
                        results.append(BenchmarkResult(
                            label=label, run=run_i + 1,
                            total_ms=elapsed_ms, tokens_out=0,
                            tok_per_sec=0, error="no response"))
                        continue

                    usage = resp.get("usage", {})
                    tokens_out = usage.get("completion_tokens", 0)
                    tok_per_sec = tokens_out / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
                    results.append(BenchmarkResult(
                        label=label, run=run_i + 1,
                        total_ms=elapsed_ms, tokens_out=tokens_out,
                        tok_per_sec=tok_per_sec, error=""))
    return results


def run_llama_bench(cfg: dict, name: str, path: str,
                    bconfig: BenchmarkConfig,
                    progress: list | None = None) -> list[BenchmarkResult]:
    """Run llama-bench for a GGUF model. Stops server first."""
    results = []
    port = cfg.get("omlx_port", 8000)
    bench_bin = (Path(cfg.get("llama_server",
                              "~/.local/llama.cpp/build/bin/llama-server"))
                 .expanduser().parent / "llama-bench")

    if not bench_bin.exists():
        return [BenchmarkResult(label="llama-bench", run=1, total_ms=0,
                                tokens_out=0, tok_per_sec=0,
                                error=f"llama-bench not found: {bench_bin}")]

    # Stop oMLX fully before llama-bench so it frees unified memory.
    # bootout alone may leave the process running briefly; pkill ensures it's gone.
    omlx_stop(cfg)
    subprocess.run(["pkill", "-9", "-f", "omlx"], capture_output=True)
    # Wait for omlx process to fully exit and memory to be reclaimed
    deadline = time.time() + 15
    while time.time() < deadline:
        r = subprocess.run(["pgrep", "-f", "omlx"], capture_output=True)
        if r.returncode != 0:
            break
        time.sleep(0.5)
    kill_port(port)

    try:
        cmd = [str(bench_bin), "-m", str(path), "-r", str(bconfig.n_reps),
               "-p", str(bconfig.n_prompt), "-n", str(bconfig.n_gen),
               "-ngl", "999", "--output", "json", "--progress"]
        for b  in bconfig.batch_sizes:   cmd += ["-b",   str(b)]
        for ub in bconfig.ubatch_sizes:  cmd += ["-ub",  str(ub)]
        for fa in bconfig.flash_attns:   cmd += ["-fa",  str(fa)]
        for ct in bconfig.cache_types_k: cmd += ["-ctk", ct]
        for ct in bconfig.cache_types_v: cmd += ["-ctv", ct]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Drain stderr in a thread so it doesn't block stdout
        stderr_lines: list[str] = []
        def _read_stderr():
            for raw in proc.stderr:
                line = raw.decode("utf-8", errors="replace")
                stderr_lines.append(line)
                if progress is not None:
                    progress.append(line)
        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        stdout_data = proc.stdout.read()
        proc.wait(timeout=600)
        stderr_thread.join(timeout=5)

        # Try to parse whatever JSON we got even if the process failed partway through.
        # llama-bench writes each completed benchmark as a JSON object; on failure
        # the array may be incomplete (no closing ]). Recover partial results by
        # trimming trailing incomplete JSON and appending ].
        raw = stdout_data.decode("utf-8", errors="replace").strip() if isinstance(stdout_data, bytes) else stdout_data.strip()
        if not raw:
            err_tail = "".join(stderr_lines)[-400:].strip()
            return [BenchmarkResult(label="error", run=1, total_ms=0,
                                    tokens_out=0, tok_per_sec=0, error=err_tail)]
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt recovery: find last complete object
            last_brace = raw.rfind("}")
            if last_brace != -1:
                try:
                    rows = json.loads(raw[: last_brace + 1] + "]")
                except json.JSONDecodeError:
                    rows = []
            else:
                rows = []

        if proc.returncode != 0 and not rows:
            err_tail = "".join(stderr_lines)[-400:].strip()
            return [BenchmarkResult(label="error", run=1, total_ms=0,
                                    tokens_out=0, tok_per_sec=0, error=err_tail)]

        # If process failed, build a clear message showing which combo failed and why.
        partial_error = ""
        if proc.returncode != 0:
            import re as _re
            # Find which benchmark number failed, e.g. "llama-bench: benchmark 3/8: starting"
            failed_at = total_combos = None
            for line in stderr_lines:
                m = _re.search(r"benchmark (\d+)/(\d+)", line)
                if m:
                    failed_at, total_combos = int(m.group(1)), int(m.group(2))
            # Find the error reason (strip model path noise)
            reason = "failed to create context (likely out of memory)"
            for line in reversed(stderr_lines):
                if "main: error" in line:
                    reason = _re.sub(r"main: error:\s*", "", line).strip()
                    reason = _re.sub(r"'?/[^']*\.gguf'?", "<model>", reason)
                    break
            if failed_at and total_combos:
                completed = failed_at - 1
                partial_error = (f"Completed {completed}/{total_combos} combinations. "
                                 f"Combination {failed_at} failed: {reason}. "
                                 f"Try fewer sweep options or a smaller model.")
            else:
                partial_error = reason

        # JSON fields: n_prompt/n_gen (token counts), avg_ts (tok/sec), avg_ns (ns/tok)
        # Sweep params per row: n_batch, n_ubatch, flash_attn, type_k, type_v
        for row in rows:
            n_prompt = int(row.get("n_prompt", 0))
            n_gen    = int(row.get("n_gen",    0))
            if n_prompt > 0:
                phase, n_tok = "PP", n_prompt
            elif n_gen > 0:
                phase, n_tok = "TG", n_gen
            else:
                continue
            avg_ts = float(row.get("avg_ts", 0))
            avg_ns = float(row.get("avg_ns", 0))
            # Build a label that shows the combo so results are distinguishable
            b   = row.get("n_batch",    "?")
            ub  = row.get("n_ubatch",   "?")
            fa  = "on" if row.get("flash_attn") else "off"
            ctk = row.get("type_k",     "?")
            ctv = row.get("type_v",     "?")
            label = f"{phase}  b={b} ub={ub} fa={fa} ctk={ctk} ctv={ctv}"
            results.append(BenchmarkResult(
                label=label, run=1,
                total_ms=avg_ns / 1_000_000,
                tokens_out=n_tok,
                tok_per_sec=avg_ts,
                error=""))
        if partial_error:
            results.append(BenchmarkResult(
                label="partial", run=1, total_ms=0, tokens_out=0, tok_per_sec=0,
                error=partial_error))
    except subprocess.TimeoutExpired:
        proc.kill()
        results.append(BenchmarkResult(label="llama-bench", run=1, total_ms=0,
                                       tokens_out=0, tok_per_sec=0,
                                       error="timeout"))
    except Exception as e:
        results.append(BenchmarkResult(label="llama-bench", run=1, total_ms=0,
                                       tokens_out=0, tok_per_sec=0,
                                       error=str(e)))
    return results


def _bench_results_html(name: str, results: list[BenchmarkResult], mode: str) -> str:
    """Build an HTML page for benchmark results."""
    from collections import defaultdict
    css = """
    <style>
      body { font-family: -apple-system, sans-serif; font-size: 13px;
             margin: 16px; color: #1a1a1a; background: #f5f5f5; }
      h2   { font-size: 15px; margin: 0 0 4px 0; }
      .sub { color: #666; font-size: 11px; margin: 0 0 14px 0; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 14px; }
      th       { padding: 6px 10px; text-align: left; font-size: 12px; color: #fff; }
      th.th-in  { background: #3a5a8a; }   /* blue — input parameters */
      th.th-out { background: #2a6a3a; }   /* green — output metrics */
      th.num   { text-align: right; }
      td   { padding: 5px 10px; border-bottom: 1px solid #ddd; font-size: 12px; }
      td.num { text-align: right; font-variant-numeric: tabular-nums; }
      tr:nth-child(even) { background: #ececec; }
      tr:hover { background: #dde8f5; }
      .best { font-weight: 600; color: #1a6c1a; }
      .note { color: #888; font-size: 11px; margin-top: 10px; }
      .err  { color: #c00; font-size: 12px; margin: 6px 0; }
    </style>
    """
    body = f"<h2>Benchmark — {name}</h2><p class='sub'>{mode}</p>"

    if mode == "llama-bench":
        ok  = [r for r in results if not r.error]
        err = [r for r in results if r.error]

        def _parse_label(label: str) -> dict:
            """Parse 'PP  b=2048 ub=512 fa=off ctk=f16 ctv=f16' into a dict."""
            import re as _re
            return {k: v for k, v in _re.findall(r'(\w+)=(\S+)', label)}

        if ok:
            # Split PP and TG into separate tables
            for phase, heading in [("PP", "Prompt Processing (PP)"),
                                    ("TG", "Token Generation (TG)")]:
                rows = [r for r in ok if r.label.startswith(phase)]
                if not rows:
                    continue
                best_tps = max(r.tok_per_sec for r in rows)
                # Determine which param columns actually vary across rows
                all_params = [_parse_label(r.label) for r in rows]
                all_keys = list(all_params[0].keys()) if all_params else []
                # Show a column for a param only if it varies OR there's only one row
                varying = [k for k in all_keys
                           if len({p.get(k) for p in all_params}) > 1] if len(rows) > 1 else all_keys
                # Always show at minimum batch and ubatch
                show_keys = list(dict.fromkeys(["b", "ub"] + varying))

                col_labels = {"b": "Batch", "ub": "Ubatch", "fa": "Flash Attn",
                              "ctk": "Cache K", "ctv": "Cache V"}

                body += f"<h3 style='font-size:13px;margin:10px 0 4px'>{heading}</h3>"
                header_cols = "".join(
                    f"<th class='th-in'>{col_labels.get(k, k)}</th>" for k in show_keys)
                body += (f"<table><tr>{header_cols}"
                         f"<th class='th-out num'>Tok/sec</th>"
                         f"<th class='th-out num'>Tokens</th>"
                         f"<th class='th-out num'>Total time (ms)</th>"
                         f"</tr>")
                for r, params in zip(rows, all_params):
                    best = " class='best'" if r.tok_per_sec == best_tps else ""
                    param_cells = "".join(
                        f"<td{best}>{params.get(k, '—')}</td>" for k in show_keys)
                    body += (f"<tr>{param_cells}"
                             f"<td class='num'{best}>{r.tok_per_sec:.1f}</td>"
                             f"<td class='num'>{r.tokens_out}</td>"
                             f"<td class='num'>{r.total_ms:.0f}</td>"
                             f"</tr>")
                body += "</table>"

        for r in err:
            body += f"<p class='err'>Error: {r.error}</p>"
        body += "<p class='note'>Server was stopped for benchmarking. Re-select model to restart.</p>"

    else:  # API mode
        import re as _re

        def _parse_api_label(label: str):
            """Split 'Prompt name  thinking=off gen=128' into (prompt, params_dict)."""
            m = _re.search(r'\s{2,}(\S.*)', label)
            if m:
                prompt_part = label[:m.start()].strip()
                params = {k: v for k, v in _re.findall(r'(\w+)=(\S+)', m.group(1))}
            else:
                prompt_part = label
                params = {}
            return prompt_part, params

        by_label: dict = defaultdict(list)
        for r in results:
            by_label[r.label].append(r)

        # Determine which sweep columns exist and vary
        all_parsed = {lbl: _parse_api_label(lbl) for lbl in by_label}
        all_sweep_params = [p for _, p in all_parsed.values()]
        sweep_keys = list(all_sweep_params[0].keys()) if all_sweep_params else []
        if len(by_label) > 1:
            varying_sweep = [k for k in sweep_keys
                             if len({p.get(k) for p in all_sweep_params}) > 1]
        else:
            varying_sweep = sweep_keys

        sweep_col_labels = {"thinking": "Thinking", "gen": "Gen tokens"}

        # Build header
        sweep_headers = "".join(
            f"<th class='th-in'>{sweep_col_labels.get(k, k)}</th>"
            for k in varying_sweep)
        col_count = 4 + len(varying_sweep)
        body += (f"<table><tr>"
                 f"<th class='th-in'>Prompt</th>"
                 f"{sweep_headers}"
                 f"<th class='th-out num'>Avg tok/sec</th>"
                 f"<th class='th-out num'>Avg tokens</th>"
                 f"<th class='th-out num'>Avg time (ms)</th>"
                 f"<th class='th-out num'>Runs</th>"
                 f"</tr>")

        all_tps = [sum(r.tok_per_sec for r in runs if not r.error)
                   / max(1, len([r for r in runs if not r.error]))
                   for runs in by_label.values() if any(not r.error for r in runs)]
        best_tps = max(all_tps) if all_tps else 0

        for label, runs in by_label.items():
            prompt_part, params = all_parsed[label]
            ok = [r for r in runs if not r.error]
            sweep_cells = "".join(
                f"<td>{params.get(k, '—')}</td>" for k in varying_sweep)
            if ok:
                avg_tps = sum(r.tok_per_sec for r in ok) / len(ok)
                avg_tok = sum(r.tokens_out  for r in ok) / len(ok)
                avg_ms  = sum(r.total_ms    for r in ok) / len(ok)
                best = " class='best'" if abs(avg_tps - best_tps) < 0.01 else ""
                body += (f"<tr>"
                         f"<td{best}>{prompt_part}</td>"
                         f"{sweep_cells}"
                         f"<td class='num'{best}>{avg_tps:.1f}</td>"
                         f"<td class='num'>{avg_tok:.0f}</td>"
                         f"<td class='num'>{avg_ms:.0f}</td>"
                         f"<td class='num'>{len(ok)}/{len(runs)}</td>"
                         f"</tr>")
            else:
                body += (f"<tr><td>{prompt_part}</td>{sweep_cells}"
                         f"<td colspan='4' class='err'>Error: {runs[0].error}</td></tr>")
        body += "</table>"

    return f"<html><head>{css}</head><body>{body}</body></html>"


def run_benchmark_results_panel(name: str, results: list[BenchmarkResult],
                                 mode: str) -> None:
    """Show benchmark results in a modal with a WKWebView."""
    html = _bench_results_html(name, results, mode)

    W, H = 620, 480
    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
    panel.setTitle_(f"Benchmark Results — {name}")
    panel.setDelegate_(handler)
    panel.center()
    cv = panel.contentView()

    web_rect = ((_PAD, _BTN_BOT + _BTN_H + _PAD),
                (W - 2 * _PAD, H - _PAD - (_BTN_BOT + _BTN_H + _PAD) - _PAD))
    wv = WKWebView.alloc().initWithFrame_configuration_(
        web_rect, WKWebViewConfiguration.alloc().init())
    wv.loadHTMLString_baseURL_(html, None)
    cv.addSubview_(wv)

    cv.addSubview_(_btn("Close", handler, "stopOK:",
                        ((W // 2 - 33, _BTN_BOT), (66, _BTN_H)), "\r"))

    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    NSApp.runModalForWindow_(panel)
    panel.orderOut_(None)


# ── Benchmark history ─────────────────────────────────────────────────────────

def save_bench_run(name: str, mode: str, results: list[BenchmarkResult]) -> None:
    """Append a benchmark run to the history file. Silently swallows all errors."""
    try:
        BENCH_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if BENCH_HISTORY_PATH.exists():
            try:
                history = json.loads(BENCH_HISTORY_PATH.read_text())
            except Exception:
                history = []
        import datetime as _dt
        entry = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "model": name,
            "mode": mode,
            "results": [
                {"label": r.label, "tok_per_sec": r.tok_per_sec,
                 "tokens_out": r.tokens_out, "total_ms": r.total_ms,
                 "error": r.error}
                for r in results
            ],
        }
        history.append(entry)
        BENCH_HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n")
    except Exception:
        pass


def _bench_history_html() -> str:
    """Build an HTML comparison page from all saved benchmark runs."""
    history = []
    if BENCH_HISTORY_PATH.exists():
        try:
            history = json.loads(BENCH_HISTORY_PATH.read_text())
        except Exception:
            pass

    # Gather chart data: model → list of (timestamp, avg tok/s)
    from collections import defaultdict as _dd
    model_series: dict = _dd(list)
    for entry in history:
        ok = [r for r in entry.get("results", []) if not r.get("error") and r.get("tok_per_sec", 0) > 0]
        if ok:
            avg = sum(r["tok_per_sec"] for r in ok) / len(ok)
            model_series[entry["model"]].append((entry["timestamp"], round(avg, 1)))

    colors = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
              "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"]
    datasets = []
    for i, (model, points) in enumerate(sorted(model_series.items())):
        color = colors[i % len(colors)]
        datasets.append({
            "label": model,
            "data": [{"x": ts, "y": tps} for ts, tps in points],
            "borderColor": color, "backgroundColor": color + "33",
            "tension": 0.3, "pointRadius": 4,
        })

    chart_json = json.dumps({"datasets": datasets})
    rows_html = ""
    for entry in reversed(history):
        for r in entry.get("results", []):
            if r.get("error"):
                continue
            rows_html += (
                f"<tr><td>{entry['timestamp']}</td><td>{entry['model']}</td>"
                f"<td>{entry['mode']}</td><td>{r['label']}</td>"
                f"<td class='num'>{r['tok_per_sec']:.1f}</td>"
                f"<td class='num'>{r['tokens_out']}</td>"
                f"<td class='num'>{r['total_ms']:.0f}</td></tr>"
            )

    clear_btn = ("<form style='display:inline'>"
                 "<button onclick=\"window.location='clear:'\" style='margin-left:12px;"
                 "font-size:11px;padding:2px 8px'>Clear history</button></form>")
    return f"""<html><head>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:-apple-system,sans-serif;font-size:13px;margin:16px;color:#1a1a1a;background:#f5f5f5}}
  h2{{font-size:15px;margin:0 0 8px}}
  canvas{{max-height:260px;margin-bottom:16px;background:#fff;border-radius:6px}}
  table{{border-collapse:collapse;width:100%;font-size:12px}}
  th{{background:#3a5a8a;color:#fff;padding:5px 8px;text-align:left}}
  th.num,td.num{{text-align:right}}
  td{{padding:4px 8px;border-bottom:1px solid #ddd}}
  tr:nth-child(even){{background:#ececec}}
</style></head><body>
<h2>Benchmark History {clear_btn}</h2>
<canvas id="c"></canvas>
<script>
new Chart(document.getElementById('c'),{{
  type:'line',
  data:{chart_json},
  options:{{scales:{{x:{{type:'category',title:{{display:true,text:'Run'}}}},
    y:{{title:{{display:true,text:'tok/s'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}}}}
}});
</script>
<table><tr><th>Time</th><th>Model</th><th>Mode</th><th>Label</th>
<th class='num'>tok/s</th><th class='num'>Tokens</th><th class='num'>ms</th></tr>
{rows_html}</table>
</body></html>"""


def run_bench_history_panel() -> None:
    """Show benchmark history in a WKWebView modal."""
    html = _bench_history_html()
    W, H = 820, 580
    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
    panel.setTitle_("Benchmark History")
    panel.setDelegate_(handler)
    panel.center()
    cv = panel.contentView()
    web_h = H - _BTN_BOT - _BTN_H - _PAD * 2
    wv = WKWebView.alloc().initWithFrame_configuration_(
        ((_PAD, _BTN_BOT + _BTN_H + _PAD), (W - _PAD * 2, web_h)),
        WKWebViewConfiguration.alloc().init())
    wv.loadHTMLString_baseURL_(html, None)
    cv.addSubview_(wv)
    cv.addSubview_(_btn("Close", handler, "stopOK:",
                        ((W // 2 - 33, _BTN_BOT), (66, _BTN_H)), "\r"))
    cv.addSubview_(_btn("Clear History", handler, "clearHistory:",
                        ((_PAD, _BTN_BOT), (100, _BTN_H))))
    handler._history_wv = wv
    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    NSApp.runModalForWindow_(panel)
    panel.orderOut_(None)


# ── Profiles ─────────────────────────────────────────────────────────────────

def load_profiles() -> list[dict]:
    try:
        if PROFILES_PATH.exists():
            data = json.loads(PROFILES_PATH.read_text())
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_profiles(profiles: list[dict]) -> None:
    try:
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILES_PATH.write_text(json.dumps(profiles, indent=2) + "\n")
    except Exception:
        pass


def run_create_profile_panel(all_models: list[str],
                             presets: list[str]) -> dict | None:
    """Modal to create or edit a profile. Returns dict or None on cancel."""
    W, H = 420, 180
    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
    panel.setTitle_("New Profile")
    panel.setDelegate_(handler)
    panel.center()
    cv = panel.contentView()

    def fy(t, h=_RH): return H - t - h
    def _popup(opts, frame):
        p = NSPopUpButton.alloc().initWithFrame_(frame)
        for o in opts: p.addItemWithTitle_(o)
        return p

    x_lbl, x_fld = _PAD, _PAD + _LW + _GAP
    FW = W - x_fld - _PAD
    cur = _PAD
    cv.addSubview_(_lbl("Profile name:", ((x_lbl, fy(cur)), (_LW, _RH))))
    name_fld = _fld("My Profile", ((x_fld, fy(cur)), (FW, _RH)))
    cv.addSubview_(name_fld); cur += _RH + _RG
    cv.addSubview_(_lbl("Model:", ((x_lbl, fy(cur)), (_LW, _RH))))
    model_popup = _popup(all_models, ((x_fld, fy(cur) - 2), (FW, _RH + 4)))
    cv.addSubview_(model_popup); cur += _RH + _RG
    cv.addSubview_(_lbl("Sampling preset:", ((x_lbl, fy(cur)), (_LW, _RH))))
    preset_popup = _popup(["(none)"] + presets, ((x_fld, fy(cur) - 2), (FW, _RH + 4)))
    cv.addSubview_(preset_popup)
    cv.addSubview_(_btn("Cancel", handler, "stopCancel:",
                        ((W - _PAD - 162, _BTN_BOT), (82, _BTN_H)), "\x1b"))
    cv.addSubview_(_btn("Save", handler, "stopOK:",
                        ((W - _PAD - 74, _BTN_BOT), (60, _BTN_H)), "\r"))
    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    result = NSApp.runModalForWindow_(panel)
    panel.orderOut_(None)
    if result != NSModalResponseOK:
        return None
    pname = name_fld.stringValue().strip()
    if not pname:
        return None
    preset = preset_popup.titleOfSelectedItem()
    return {
        "name": pname,
        "model": model_popup.titleOfSelectedItem(),
        "preset": preset if preset != "(none)" else None,
    }


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".config" / "model-switcher" / "config.json"

# Default per-model params (overridable per model in cfg["model_params"])
DEFAULT_MODEL_PARAMS = {
    "context":            32768,  # -c  (GGUF) / context limit hint for opencode
    "gpu_layers":         999,    # -ngl (GGUF only)
    "max_tokens":         8192,   # opencode output limit hint
    # Sampling params — written to opencode config and used in warm-up calls
    "temperature":        0.7,
    "top_p":              0.8,
    "top_k":              20,
    "min_p":              0.0,
    "presence_penalty":   1.5,
    "repetition_penalty": 1.0,
    "enable_thinking":    False,  # passed as chat_template_kwargs to oMLX
}

# Qwen-recommended sampling presets
SAMPLING_PRESETS: dict[str, dict] = {
    "Thinking — General":   {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "enable_thinking": True},
    "Thinking — Coding":    {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "enable_thinking": True},
    "Instruct — General":   {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "enable_thinking": False},
    "Instruct — Reasoning": {"temperature": 1.0, "top_p": 1.0,  "top_k": 40, "min_p": 0.0, "presence_penalty": 2.0, "repetition_penalty": 1.0, "enable_thinking": False},
}

DEFAULTS = {
    "mlx_dir":        "/Volumes/DataNVME/models/mlx/",
    "gguf_dir":       "/Volumes/DataNVME/models/gguf/",
    "omlx_port":      8000,
    "omlx_api_key":   "123456",
    "omlx_service":   "com.jim.omlx",
    "llama_server":   str(Path.home() / ".local/llama.cpp/build/bin/llama-server"),
    "llama_port":     8000,
    "opencode_config": str(Path.home() / ".config/opencode/opencode.json"),
    "restart_opencode": False,
    "terminal_app":   "Terminal",   # "Terminal" | "iTerm2"
    "aliases":        {},         # {model_name: alias_string}
    "model_notes":    {},         # {model_name: note_string}
    "model_params":   {},         # {model_name: {context, gpu_layers, max_tokens}}
    "hidden_models":  [],         # [model_name, ...]
    # Feature flags
    "sync_cursor":    False,      # sync Cursor MCP config on model switch
    "sync_continue":  False,      # sync Continue.dev config on model switch
    "sync_env":       True,       # write ~/.config/model-switcher/env on switch
    "notifications":  True,       # macOS notification when model is ready
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text())
            cfg = {**DEFAULTS, **saved}
            for key in ("aliases", "model_notes", "model_params"):
                if not isinstance(cfg.get(key), dict):
                    cfg[key] = {}
            if not isinstance(cfg.get("hidden_models"), list):
                cfg["hidden_models"] = []
            return cfg
        except Exception:
            pass
    return dict(DEFAULTS)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")


def model_params(cfg: dict, name: str) -> dict:
    """Return effective params for a model, merging defaults with per-model overrides."""
    return {**DEFAULT_MODEL_PARAMS, **cfg["model_params"].get(name, {})}


# ── Model scanning ────────────────────────────────────────────────────────────

def scan_mlx(cfg: dict) -> dict[str, Path]:
    d = Path(cfg["mlx_dir"])
    if not d.exists():
        return {}
    return {
        e.name: e
        for e in sorted(d.iterdir())
        if e.is_dir() and not e.name.startswith(".")
    }


def scan_gguf(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["gguf_dir"])
    if not root.exists():
        return {}
    models: dict[str, Path] = {}
    for f in sorted(root.rglob("*.gguf")):
        lower = f.name.lower()
        if "mmproj" in lower:
            continue
        if "-of-" in lower and "-00001-of-" not in lower:
            continue
        depth = len(f.relative_to(root).parts) - 1
        if depth >= 3:
            continue
        display = f.parent.name if depth == 2 else f.stem
        if display not in models:
            models[display] = f
    return dict(sorted(models.items()))


# ── Model metadata ────────────────────────────────────────────────────────────

_GGUF_VALUE_TYPES = {4: "<I", 5: "<q", 6: "<f", 11: "<d", 12: "<B"}

def parse_gguf_metadata(path: Path) -> dict:
    """Read up to 8KB of a GGUF file and extract arch, context, and quant from name."""
    meta: dict = {}
    try:
        import re as _re
        # Quant from filename — e.g. Q4_K_M, Q8_0
        m = _re.search(r'[-_]((?:Q|IQ)\d+[_A-Z0-9]*)', path.stem, _re.I)
        if m:
            meta["quant"] = m.group(1).upper()

        with open(path, "rb") as _f:
            data = _f.read(8192)
        if data[:4] != b"GGUF":
            return meta
        version = struct.unpack_from("<I", data, 4)[0]
        if version not in (2, 3):
            return meta
        kv_count = struct.unpack_from("<Q", data, 16)[0]
        pos = 24
        wanted = {"general.architecture", "llm.context_length",
                  "general.context_length"}
        for _ in range(min(kv_count, 64)):
            if pos + 8 > len(data):
                break
            klen = struct.unpack_from("<Q", data, pos)[0]; pos += 8
            if pos + klen > len(data):
                break
            key = data[pos:pos + klen].decode("utf-8", errors="replace"); pos += klen
            if pos + 4 > len(data):
                break
            vtype = struct.unpack_from("<I", data, pos)[0]; pos += 4
            if vtype == 8:   # string
                slen = struct.unpack_from("<Q", data, pos)[0]; pos += 8
                val = data[pos:pos + slen].decode("utf-8", errors="replace"); pos += slen
            elif vtype in _GGUF_VALUE_TYPES:
                fmt = _GGUF_VALUE_TYPES[vtype]
                sz = struct.calcsize(fmt)
                val = struct.unpack_from(fmt, data, pos)[0]; pos += sz
            else:
                break   # unknown type — stop parsing
            if key in wanted:
                if "architecture" in key:
                    meta["arch"] = str(val)
                elif "context" in key:
                    meta["context"] = int(val)
            if len(meta) >= 3:
                break
    except Exception:
        pass
    return meta


def parse_mlx_metadata(path: Path) -> dict:
    """Read MLX model config.json for arch and context length."""
    meta: dict = {}
    try:
        cfg_path = path / "config.json"
        if not cfg_path.exists():
            return meta
        cfg = json.loads(cfg_path.read_text())
        arch = cfg.get("model_type") or cfg.get("architectures", [None])[0]
        if arch:
            meta["arch"] = str(arch).lower()
        ctx = (cfg.get("max_position_embeddings")
               or cfg.get("context_length")
               or cfg.get("max_seq_len"))
        if ctx:
            meta["context"] = int(ctx)
        quant = cfg.get("quantization", {})
        if isinstance(quant, dict) and quant.get("bits"):
            meta["quant"] = f"mlx-{quant['bits']}bit"
    except Exception:
        pass
    return meta


# ── Network helpers ───────────────────────────────────────────────────────────

def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_port_free(port: int, timeout: int = 20) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_free(port):
            return True
        time.sleep(0.5)
    return False


def wait_for_port_open(port: int, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not port_is_free(port):
            return True
        time.sleep(0.5)
    return False


def mlx_sampling_params(p: dict) -> dict:
    """Translate canonical model params → oMLX API-compatible sampling params.

    oMLX does NOT support top_k.
    oMLX uses frequency_penalty; our canonical name is repetition_penalty.
    """
    return {
        "temperature":          p["temperature"],
        "top_p":                p["top_p"],
        "min_p":                p["min_p"],
        "presence_penalty":     p["presence_penalty"],
        "frequency_penalty":    p["repetition_penalty"],
        "chat_template_kwargs": {"enable_thinking": p.get("enable_thinking", False)},
    }


def llama_sampling_params(p: dict) -> dict:
    """Translate canonical model params → llama-server OpenAI-compat sampling params.

    llama-server uses repeat_penalty; our canonical name is repetition_penalty.
    """
    return {
        "temperature":      p["temperature"],
        "top_p":            p["top_p"],
        "top_k":            p["top_k"],
        "min_p":            p["min_p"],
        "presence_penalty": p["presence_penalty"],
        "repeat_penalty":   p["repetition_penalty"],
    }


def kill_port(port: int) -> None:
    """Kill any process listening on the given port."""
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True,
    )
    for pid in result.stdout.strip().splitlines():
        try:
            subprocess.run(["kill", "-TERM", pid.strip()], capture_output=True)
        except Exception:
            pass
    wait_for_port_free(port, timeout=8)


def get_memory_pressure() -> str:
    """Return 'nominal', 'warn', or 'critical' from the system memory_pressure tool."""
    try:
        r = subprocess.run(["/usr/bin/memory_pressure"], capture_output=True,
                           text=True, timeout=8)
        out = r.stdout.lower()
        if "critical" in out:
            return "critical"
        if "elevated" in out or "warn" in out:
            return "warn"
    except Exception:
        pass
    return "nominal"


# ── omlx helpers ─────────────────────────────────────────────────────────────

def http_post(url: str, body: dict, headers: dict, timeout: int = 300) -> dict | None:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def omlx_is_healthy(cfg: dict) -> bool:
    """Return True only if omlx (not llama-server) is on the port.
    omlx's /health response contains 'engine_pool'; llama-server's does not."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{cfg['omlx_port']}/health", timeout=2
        ) as r:
            data = json.loads(r.read())
            return "engine_pool" in data
    except Exception:
        return False


def _omlx_plist(cfg: dict) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{cfg['omlx_service']}.plist"


def omlx_stop(cfg: dict) -> None:
    """Unload omlx plist and wait for port to free."""
    uid = os.getuid()
    plist = str(_omlx_plist(cfg))
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", plist],
        capture_output=True,
    )
    wait_for_port_free(cfg["omlx_port"], timeout=15)


def omlx_start(cfg: dict) -> bool:
    """Bootstrap omlx plist and wait until healthy."""
    uid = os.getuid()
    plist = str(_omlx_plist(cfg))
    # bootstrap loads the plist; kickstart -k ensures it (re)starts even if already loaded
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", plist],
        capture_output=True,
    )
    subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{uid}/{cfg['omlx_service']}"],
        capture_output=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(1)
        if omlx_is_healthy(cfg):
            return True
    return False


def send_model_ready_notification(name: str) -> None:
    """Fire a local macOS notification. Safe to call from any thread."""
    if UNUserNotificationCenter is None:
        return
    try:
        content = UNMutableNotificationContent.alloc().init()
        content.setTitle_("Model Switcher")
        content.setBody_(f"Model ready: {name}")
        req = UNNotificationRequest.requestWithIdentifier_content_trigger_(
            str(uuid.uuid4()), content, None)
        def _noop(error):
            pass
        UNUserNotificationCenter.currentNotificationCenter() \
            .addNotificationRequest_withCompletionHandler_(req, _noop)
    except Exception:
        pass


def request_notification_permission() -> None:
    """Ask for notification permission once (idempotent — system caches grant).
    Must be called from a background thread — the completion handler block fires
    asynchronously and must not be nil (passing nil crashes NSApplication.run)."""
    if UNUserNotificationCenter is None:
        return
    try:
        opts = UNAuthorizationOptionAlert | UNAuthorizationOptionSound
        # The completion handler MUST be a real callable (nil crashes the run loop)
        def _handler(granted, error):
            pass
        UNUserNotificationCenter.currentNotificationCenter() \
            .requestAuthorizationWithOptions_completionHandler_(opts, _handler)
    except Exception:
        pass


# ── opencode helpers ──────────────────────────────────────────────────────────

def set_opencode_model(cfg: dict, provider: str, model_id: str,
                       display_name: str | None = None,
                       context: int = 8192, max_tokens: int = 4096,
                       sampling: dict | None = None) -> None:
    path = Path(cfg["opencode_config"])
    if not path.exists():
        return
    try:
        config = json.loads(path.read_text())
        provider_entry = config.setdefault("provider", {}).setdefault(provider, {})
        models = provider_entry.setdefault("models", {})
        entry: dict = {
            "name": display_name or model_id,
            "limit": {"context": context, "output": max_tokens},
        }
        if sampling:
            entry["parameters"] = sampling
        models[model_id] = entry
        config["model"] = f"{provider}/{model_id}"
        path.write_text(json.dumps(config, indent=2) + "\n")
    except Exception:
        pass


def sync_cursor_config(port: int) -> None:
    """Update Cursor MCP config to point at the current inference server."""
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


def sync_continue_config(port: int) -> None:
    """Update Continue.dev config to point at the current inference server."""
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
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    except Exception:
        pass


def sync_env_file(port: int) -> None:
    """Write LLM_BASE_URL to ~/.config/model-switcher/env for shell sourcing."""
    try:
        _MS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        (_MS_CONFIG_DIR / "env").write_text(
            f"export LLM_BASE_URL=http://localhost:{port}\n"
            f"export LLM_API_BASE=http://localhost:{port}/v1\n"
        )
    except Exception:
        pass


def sync_clients(cfg: dict, port: int) -> None:
    """Sync all enabled external client configs. Safe to call from any thread."""
    if cfg.get("sync_cursor"):
        sync_cursor_config(port)
    if cfg.get("sync_continue"):
        sync_continue_config(port)
    if cfg.get("sync_env", True):
        sync_env_file(port)


def find_opencode_processes() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["pgrep", "-f", "opencode-ai"], capture_output=True, text=True,
    )
    pids = [int(p) for p in result.stdout.strip().splitlines() if p.strip()]
    out: list[tuple[int, str]] = []
    seen_cwds: set[str] = set()
    for pid in pids:
        r = subprocess.run(
            ["lsof", "-p", str(pid), "-a", "-d", "cwd", "-Fn"],
            capture_output=True, text=True,
        )
        cwd = next(
            (l[1:] for l in r.stdout.splitlines() if l.startswith("n")),
            str(Path.home()),
        )
        if cwd not in seen_cwds:
            seen_cwds.add(cwd)
            out.append((pid, cwd))
    return out


def get_process_tty(pid: int) -> str | None:
    r = subprocess.run(["ps", "-p", str(pid), "-o", "tty="],
                       capture_output=True, text=True)
    tty = r.stdout.strip()
    if tty and tty != "??":
        return f"/dev/{tty}" if not tty.startswith("/") else tty
    return None


def restart_opencode(cfg: dict) -> None:
    procs = find_opencode_processes()
    cwds = [cwd for _, cwd in procs] if procs else [str(Path.home())]
    ttys: list[str] = []
    for pid, _ in procs:
        tty = get_process_tty(pid)
        if tty and tty not in ttys:
            ttys.append(tty)

    # Close terminal windows BEFORE killing the process — once the process dies
    # the window may already be gone and we lose the TTY handle.
    if cfg["terminal_app"] == "iTerm2":
        _close_iterm_ttys(ttys)
    else:
        _close_terminal_ttys(ttys)

    subprocess.run(["pkill", "-f", "opencode-ai"], capture_output=True)
    time.sleep(0.3)

    seen: set[str] = set()
    for cwd in cwds:
        if cwd in seen:
            continue
        seen.add(cwd)
        _open_terminal(cfg["terminal_app"], cwd, "opencode")


def _close_iterm_ttys(ttys: list[str]) -> None:
    if not ttys:
        return
    # Build a list literal for AppleScript
    as_list = "{" + ", ".join(f'"{t}"' for t in ttys) + "}"
    script = f"""
tell application "iTerm"
    set targetTTYs to {as_list}
    repeat with w in windows
        set shouldClose to false
        repeat with t in tabs of w
            repeat with s in sessions of t
                if (tty of s) is in targetTTYs then
                    set shouldClose to true
                end if
            end repeat
        end repeat
        if shouldClose then close w
    end repeat
end tell
"""
    subprocess.run(["osascript", "-e", script], capture_output=True)
    time.sleep(0.3)


def _close_terminal_ttys(ttys: list[str]) -> None:
    if not ttys:
        return
    as_list = "{" + ", ".join(f'"{t}"' for t in ttys) + "}"
    script = f"""
tell application "Terminal"
    set targetTTYs to {as_list}
    repeat with w in windows
        if (tty of w) is in targetTTYs then close w
    end repeat
end tell
"""
    subprocess.run(["osascript", "-e", script], capture_output=True)
    time.sleep(0.3)


def _open_terminal(app: str, cwd: str, command: str) -> None:
    safe_cwd = cwd.replace("'", "\\'")
    safe_cmd = command.replace("'", "\\'")
    if app == "iTerm2":
        script = f"""tell application "iTerm"
    activate
    create window with default profile
    tell current session of current window
        write text "cd '{safe_cwd}' && {safe_cmd}"
    end tell
end tell"""
    else:
        script = f"""tell application "Terminal"
    activate
    do script "cd '{safe_cwd}' && {safe_cmd}"
end tell"""
    subprocess.run(["osascript", "-e", script], capture_output=True)


# ── App ───────────────────────────────────────────────────────────────────────

class ModelSwitcher(rumps.App):
    def __init__(self):
        super().__init__("⚡", quit_button=None)
        self._cfg = load_config()
        self._gguf_proc: subprocess.Popen | None = None
        self._active: str | None = None
        self._loading = False
        self._model_map: dict[str, tuple[Path, str]] = {}
        self._flash_timer: rumps.Timer | None = None
        self._flash_state = False
        self._pending_bench: tuple | None = None  # (name, results, mode) set by bench thread
        self._benchmarking = False
        self._bench_progress: list[str] = []     # lines appended by bench thread
        self._bench_progress_win = None          # NSWindow ref
        self._bench_progress_field = None        # NSTextField ref
        self._bench_progress_timer: rumps.Timer | None = None
        # Feature: live tok/s
        self._last_toks: float | None = None
        self._tps_poll_timer: rumps.Timer | None = None
        self._start_poll_on_rebuild = False
        # Feature: test prompt
        self._test_prompt_win = None
        # Feature: model metadata cache
        self._model_meta_cache: dict[str, list[str]] = {}
        self._rebuild_pending = False   # set from bg thread; polled by idle timer
        # Feature: memory pressure
        self._mem_pressure: str = "nominal"

        self._build_menu()
        self._prime_meta_cache()   # sets _rebuild_pending when done
        threading.Thread(target=self._sync_state, daemon=True).start()
        self._notif_permission_requested = False
        self._mem_pressure_timer = rumps.Timer(self._on_mem_pressure_tick, 30)
        self._mem_pressure_timer.start()
        # Idle timer: polls _rebuild_pending set by background threads that
        # cannot create timers themselves (rumps.Timer from bg thread = no-op).
        self._idle_timer = rumps.Timer(self._on_idle_tick, 1)
        self._idle_timer.start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _alias(self, name: str) -> str | None:
        return self._cfg["aliases"].get(name) or None

    def _display(self, name: str) -> str:
        return self._alias(name) or name

    def _model_kind(self, name: str) -> str:
        entry = self._model_map.get(name)
        return entry[1] if entry else "mlx"

    def _title_label(self, name: str, max_len: int = 22) -> str:
        label = self._display(name)
        return label[:max_len - 1] + "…" if len(label) > max_len else label

    def _params(self, name: str) -> dict:
        return model_params(self._cfg, name)

    def _model_params(self, name: str) -> dict:
        return model_params(self._cfg, name)

    def _get_model_meta(self, name: str) -> list[str]:
        """Return cached metadata. Returns [] if not yet cached (never blocks)."""
        return self._model_meta_cache.get(name, [])

    def _prime_meta_cache(self):
        """Populate model metadata cache in a background thread, then rebuild menu."""
        models = dict(self._model_map)   # snapshot

        def _do():
            changed = False
            for name, (path, kind) in models.items():
                if name in self._model_meta_cache:
                    continue
                try:
                    meta = parse_gguf_metadata(path) if kind == "gguf" \
                           else parse_mlx_metadata(path)
                except Exception:
                    meta = {}
                labels = []
                if meta.get("arch"):
                    labels.append(f"  arch: {meta['arch']}")
                if meta.get("context"):
                    labels.append(f"  ctx: {meta['context']:,}")
                if meta.get("quant"):
                    labels.append(f"  quant: {meta['quant']}")
                self._model_meta_cache[name] = labels
                changed = True
            if changed:
                self._rebuild_pending = True   # picked up by _on_idle_tick on main thread

        threading.Thread(target=_do, daemon=True).start()

    # ── Idle tick (main-thread bridge for background work) ────────────────────

    def _on_idle_tick(self, _timer):
        # Request notification permission once, on the first idle tick after the
        # run loop is live (main thread required for the system prompt to appear).
        if not self._notif_permission_requested:
            self._notif_permission_requested = True
            request_notification_permission()
        if self._rebuild_pending and not self._loading:
            self._rebuild_pending = False
            self._update_title()
            self._build_menu()
            if self._start_poll_on_rebuild:
                self._start_poll_on_rebuild = False
                self._start_tps_poll()

    # ── Memory pressure ───────────────────────────────────────────────────────

    def _on_mem_pressure_tick(self, _timer):
        def _check():
            self._mem_pressure = get_memory_pressure()
        threading.Thread(target=_check, daemon=True).start()

    # ── Live tok/s polling ────────────────────────────────────────────────────

    def _start_tps_poll(self):
        if self._tps_poll_timer is not None:
            return
        self._tps_poll_timer = rumps.Timer(self._on_tps_tick, 10)
        self._tps_poll_timer.start()

    def _stop_tps_poll(self):
        if self._tps_poll_timer:
            self._tps_poll_timer.stop()
            self._tps_poll_timer = None

    def _on_tps_tick(self, _timer):
        if self._loading or not self._active:
            self._stop_tps_poll()
            return
        name, port = self._active, self._cfg.get("omlx_port", 8000)
        api_key = self._cfg.get("omlx_api_key", "")
        def _poll():
            body = {"model": name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8, "stream": False}
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {api_key}"}
            t0 = time.time()
            resp = http_post(f"http://localhost:{port}/v1/chat/completions",
                             body, headers, timeout=20)
            if resp and "usage" in resp:
                elapsed = time.time() - t0
                toks = resp["usage"].get("completion_tokens", 0)
                if elapsed > 0 and toks > 0:
                    self._last_toks = toks / elapsed
            self._rebuild_pending = True
        threading.Thread(target=_poll, daemon=True).start()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self):
        self.menu._menu.removeAllItems()
        self.menu.clear()

        mlx = scan_mlx(self._cfg)
        gguf = scan_gguf(self._cfg)
        self._model_map = {
            **{n: (p, "mlx") for n, p in mlx.items()},
            **{n: (p, "gguf") for n, p in gguf.items()},
        }

        hidden = set(self._cfg["hidden_models"])
        mlx_visible  = {n: p for n, p in mlx.items()  if n not in hidden}
        gguf_visible = {n: p for n, p in gguf.items() if n not in hidden}
        hidden_present = {n for n in hidden if n in self._model_map}

        if self._loading and self._active:
            status_text = f"Loading {self._display(self._active)}…"
        elif self._active:
            status_text = f"● {self._display(self._active)}"
        else:
            status_text = "No model running"

        menu: list = [rumps.MenuItem(status_text, callback=None), None]

        # ── Profiles section ────────────────��────────────────────────────��────
        profiles = load_profiles()
        if profiles:
            menu.append(rumps.MenuItem("── Profiles ──", callback=None))
            for p in profiles:
                item = rumps.MenuItem(f"  {p['name']}", callback=self._on_apply_profile)
                item._profile_data = p
                menu.append(item)
            menu.append(None)

        if mlx_visible:
            menu.append(rumps.MenuItem("── MLX ──", callback=None))
            for name in mlx_visible:
                menu.append(self._make_model_item(name))
            menu.append(None)

        if gguf_visible:
            menu.append(rumps.MenuItem("── GGUF ──", callback=None))
            for name in gguf_visible:
                menu.append(self._make_model_item(name))
            menu.append(None)

        if not mlx_visible and not gguf_visible and not hidden_present:
            menu.append(rumps.MenuItem("No models found", callback=None))
            menu.append(None)

        menu += [
            rumps.MenuItem("⏹  Stop Engine", callback=self._stop),
            rumps.MenuItem("↻  Refresh Models", callback=self._refresh),
            None,
            self._build_settings_menu(),
        ]

        if hidden_present:
            menu.append(self._build_hidden_menu(hidden_present))

        menu += [
            None,
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]
        self.menu = menu

    def _make_model_item(self, name: str) -> rumps.MenuItem:
        alias = self._alias(name)
        is_active = self._active == name and not self._loading

        parent = rumps.MenuItem(f"  {alias or name}")
        if is_active:
            parent.state = 1

        meta = self._get_model_meta(name)
        note = self._cfg["model_notes"].get(name, "")
        for meta_label in meta:
            parent.add(rumps.MenuItem(meta_label, callback=None))
        if note:
            parent.add(rumps.MenuItem(f"  📝 {note}", callback=None))
        if meta or note:
            parent.add(None)

        sel = rumps.MenuItem("▶  Select", callback=self._on_select)
        sel._model_name = name
        parent.add(sel)

        hide_item = rumps.MenuItem("⊘  Hide", callback=self._on_hide_model)
        hide_item._model_name = name
        parent.add(hide_item)
        parent.add(None)

        settings_item = rumps.MenuItem("⚙  Settings…", callback=self._open_model_settings)
        settings_item._model_name = name
        parent.add(settings_item)

        benchmark_item = rumps.MenuItem("⏱  Benchmark…", callback=self._on_benchmark)
        benchmark_item._model_name = name
        parent.add(benchmark_item)

        return parent

    def _build_settings_menu(self) -> rumps.MenuItem:
        s = rumps.MenuItem("⚙  Settings")
        # Memory pressure indicator
        dot = {"nominal": "🟢", "warn": "🟡", "critical": "🔴"}.get(
            self._mem_pressure, "⚪")
        s.add(rumps.MenuItem(f"{dot}  Memory: {self._mem_pressure}", callback=None))
        s.add(None)
        s.add(rumps.MenuItem("  Open Settings…", callback=self._open_settings))
        s.add(rumps.MenuItem("  Manage Profiles…", callback=self._open_profiles))
        s.add(None)
        s.add(rumps.MenuItem("  Quick Test Prompt…", callback=self._open_test_prompt))
        s.add(rumps.MenuItem("  Benchmark History…", callback=self._open_bench_history))
        return s

    

    def _build_hidden_menu(self, hidden_present: set) -> rumps.MenuItem:
        label = f"⊘  Hidden ({len(hidden_present)})"
        s = rumps.MenuItem(label)
        for name in sorted(hidden_present):
            display = self._display(name)
            item = rumps.MenuItem(f"  {display}", callback=self._on_unhide_model)
            item._model_name = name
            s.add(item)
        return s

    # ── Scheduling ────────────────────────────────────────────────────────────

    def _schedule_rebuild(self):
        t = rumps.Timer(self._on_rebuild_timer, 0.05)
        t.start()

    def _on_rebuild_timer(self, timer):
        open("/tmp/ms_bench_log.txt", "a").write(
            f"rebuild_timer fired pending={self._pending_bench is not None}\n")
        timer.stop()
        self._update_title()
        try:
            self._build_menu()
        except Exception:
            import traceback
            open("/tmp/ms_bench_log.txt", "a").write(
                f"\n_build_menu exception:\n{traceback.format_exc()}\n")
        if self._start_poll_on_rebuild:
            self._start_poll_on_rebuild = False
            self._start_tps_poll()
        if self._pending_bench:
            args, self._pending_bench = self._pending_bench, None
            try:
                open("/tmp/ms_bench_log.txt", "a").write("opening results panel\n")
                run_benchmark_results_panel(*args)
            except Exception:
                import traceback
                open("/tmp/ms_bench_log.txt", "a").write(
                    f"\nresults panel exception:\n{traceback.format_exc()}\n")

    def _update_title(self):
        if self._loading:
            self._start_flash()
        else:
            self._stop_flash()
            mem_dot = "🔴" if self._mem_pressure == "critical" else ""
            if self._active:
                if self._last_toks is not None:
                    self.title = f"⚡{mem_dot} {int(self._last_toks)} t/s"
                else:
                    self.title = f"⚡{mem_dot} {self._title_label(self._active)}"
            else:
                self.title = f"⚡{mem_dot}"

    def _start_flash(self):
        if self._flash_timer:
            return
        self._flash_state = True
        self.title = "Benchmarking…" if self._benchmarking else "Loading model…"
        self._flash_timer = rumps.Timer(self._on_flash_tick, 0.8)
        self._flash_timer.start()

    def _stop_flash(self):
        if self._flash_timer:
            self._flash_timer.stop()
            self._flash_timer = None

    def _on_flash_tick(self, _timer):
        if not self._loading:
            self._stop_flash()
            self._close_bench_progress_window()
            self._update_title()
            self._build_menu()
            if self._pending_bench:
                args, self._pending_bench = self._pending_bench, None
                try:
                    run_benchmark_results_panel(*args)
                except Exception:
                    import traceback
                    open("/tmp/ms_bench_log.txt", "a").write(
                        f"\nresults panel exception:\n{traceback.format_exc()}\n")
            return
        self._flash_state = not self._flash_state
        label = "Benchmarking…" if self._benchmarking else "Loading model…"
        self.title = label if self._flash_state else "⚡"

    # ── Model callbacks ───────────────────────────────────────────────────────

    def _on_select(self, sender: rumps.MenuItem):
        if self._loading:
            return
        self._last_toks = None
        self._stop_tps_poll()
        name = getattr(sender, "_model_name", None) or sender.title.strip()
        entry = self._model_map.get(name)
        if entry is None:
            return
        path, kind = entry
        self._active = name
        self._loading = True
        self._update_title()
        if kind == "mlx":
            threading.Thread(target=self._switch_mlx, args=(name,), daemon=True).start()
        else:
            threading.Thread(target=self._switch_gguf, args=(name, path), daemon=True).start()

    def _stop(self, _=None):
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _refresh(self, _):
        self._build_menu()

    # ── Engine switching ──────────────────────────────────────────────────────

    def _switch_mlx(self, name: str):
        """Kill anything on the port, start omlx, trigger model load and wait."""
        self._kill_gguf()

        # Kill any stray process on the port (e.g. llama-server from a prior session)
        if not omlx_is_healthy(self._cfg):
            kill_port(self._cfg["omlx_port"])
            if not omlx_start(self._cfg):
                self._active = None
                self._loading = False
                self._rebuild_pending = True
                return

        p = self._params(name)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg['omlx_api_key']}",
        }
        url = f"http://localhost:{self._cfg['omlx_port']}/v1/chat/completions"

        sampling = mlx_sampling_params(p)

        # Step 1: trigger load — retry until omlx confirms the right model is responding
        deadline = time.time() + 300
        while time.time() < deadline:
            resp = http_post(url, body={
                "model": name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                **sampling,
            }, headers=headers, timeout=300)
            if resp and resp.get("model") == name:
                break
            time.sleep(2)

        # Step 2: warm-up — send a real generation to force SSD-cached weights into
        # unified memory. omlx uses a paged SSD cache; the first 1-token completion
        # only pages in enough to start generating. A longer generation touches more
        # of the model so the first opencode prompt doesn't stall.
        http_post(url, body={
            "model": name,
            "messages": [{"role": "user", "content": "Write a short Python hello world function."}],
            "max_tokens": 128,
            **sampling,
        }, headers=headers, timeout=300)

        set_opencode_model(
            self._cfg, "omlx", name,
            display_name=self._display(name),
            context=p["context"],
            max_tokens=p["max_tokens"],
            sampling=sampling,
        )
        sync_clients(self._cfg, self._cfg.get("omlx_port", 8000))
        if self._cfg.get("notifications", True):
            send_model_ready_notification(name)
        self._start_poll_on_rebuild = True
        self._loading = False
        if self._cfg["restart_opencode"]:
            restart_opencode(self._cfg)
        self._rebuild_pending = True

    def _switch_gguf(self, name: str, path: Path):
        """Stop omlx, wait for port, start llama-server."""
        self._kill_gguf()
        omlx_stop(self._cfg)
        # Belt-and-suspenders: kill anything still holding the port
        if not port_is_free(self._cfg["llama_port"]):
            kill_port(self._cfg["llama_port"])

        p = self._params(name)
        self._gguf_proc = subprocess.Popen(
            [
                self._cfg["llama_server"],
                "-m", str(path),
                "--port", str(self._cfg["llama_port"]),
                "-ngl", str(p["gpu_layers"]),
                "-c",   str(p["context"]),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for llama-server to bind, then get its reported model ID
        model_id = name
        if wait_for_port_open(self._cfg["llama_port"], timeout=30):
            time.sleep(0.5)  # brief settle
            reported = self._query_llama_model_id()
            if reported:
                model_id = reported

        set_opencode_model(
            self._cfg, "omlx", model_id,
            display_name=self._display(name),
            context=p["context"],
            max_tokens=p["max_tokens"],
            sampling=llama_sampling_params(p),
        )
        sync_clients(self._cfg, self._cfg.get("llama_port", 8000))
        if self._cfg.get("notifications", True):
            send_model_ready_notification(name)
        self._start_poll_on_rebuild = True
        self._loading = False
        if self._cfg["restart_opencode"]:
            restart_opencode(self._cfg)
        self._rebuild_pending = True

    def _query_llama_model_id(self) -> str | None:
        """Query llama-server's /v1/models for its reported model id."""
        url = f"http://localhost:{self._cfg['llama_port']}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                data = json.loads(r.read())
                models = data.get("data", [])
                return models[0]["id"] if models else None
        except Exception:
            return None

    def _do_stop(self):
        self._kill_gguf()
        omlx_stop(self._cfg)
        self._active = None
        self._loading = False
        self._rebuild_pending = True

    def _kill_gguf(self):
        if self._gguf_proc and self._gguf_proc.poll() is None:
            self._gguf_proc.terminate()
            try:
                self._gguf_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._gguf_proc.kill()
        self._gguf_proc = None

    # ── Startup sync ──────────────────────────────────────────────────────────

    def _sync_state(self):
        """On startup reflect what's already running."""
        if omlx_is_healthy(self._cfg):
            # Ask omlx what it has via the models list
            try:
                url = f"http://localhost:{self._cfg['omlx_port']}/v1/models"
                req = urllib.request.Request(
                    url, headers={"Authorization": f"Bearer {self._cfg['omlx_api_key']}"}
                )
                with urllib.request.urlopen(req, timeout=3) as r:
                    data = json.loads(r.read())
                    # omlx lists all discoverable models; we can't tell which is loaded
                    # without /v1/models/status — just leave active unset at startup
            except Exception:
                pass
        elif not port_is_free(self._cfg["llama_port"]):
            # llama-server might be running — get its model
            model_id = self._query_llama_model_id()
            if model_id:
                # Match against known gguf display names
                for name, (_, kind) in self._model_map.items():
                    if kind == "gguf" and (model_id.startswith(name) or name in model_id):
                        self._active = name
                        self._rebuild_pending = True
                        break

    # ── Benchmark progress window ─────────────────────────────────────────────

    def _open_bench_progress_window(self, name: str) -> None:
        """Create and show a non-modal progress window. Called on main thread."""
        W, H = 560, 320
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
        win.setTitle_(f"Benchmarking — {name}")
        win.center()
        cv = win.contentView()

        tf = NSTextField.alloc().initWithFrame_(
            ((_PAD, _PAD), (W - 2 * _PAD, H - 2 * _PAD)))
        tf.setEditable_(False)
        tf.setSelectable_(True)
        tf.setBezeled_(True)
        tf.setFont_(NSFont.monospacedSystemFontOfSize_weight_(10, 0.0))
        tf.setStringValue_("Starting llama-bench…")
        cv.addSubview_(tf)

        self._bench_progress_win   = win
        self._bench_progress_field = tf
        self._bench_progress       = []

        NSApp.activateIgnoringOtherApps_(True)
        win.makeKeyAndOrderFront_(None)

        self._bench_progress_timer = rumps.Timer(self._update_bench_progress, 0.2)
        self._bench_progress_timer.start()

    def _update_bench_progress(self, _timer) -> None:
        """Timer callback (main thread) — refresh progress text field."""
        if self._bench_progress_field is None:
            return
        lines = self._bench_progress[-30:]   # keep last 30 lines
        # Strip ggml_metal noise, keep llama-bench lines + short useful lines
        visible = [l.rstrip() for l in lines
                   if not l.startswith("ggml_") and l.strip()]
        self._bench_progress_field.setStringValue_("\n".join(visible))

    def _close_bench_progress_window(self) -> None:
        """Close progress window and stop its timer. Called on main thread."""
        if self._bench_progress_timer:
            self._bench_progress_timer.stop()
            self._bench_progress_timer = None
        if self._bench_progress_win:
            self._bench_progress_win.orderOut_(None)
            self._bench_progress_win   = None
            self._bench_progress_field = None

    # ── Settings panel callbacks ──────────────────────────────────────────────

    def _open_settings(self, _):
        if run_settings_panel(self._cfg):
            save_config(self._cfg)
            self._build_menu()

    def _open_model_settings(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        entry = self._model_map.get(name)
        kind = entry[1] if entry else "mlx"
        if run_model_settings_panel(self._cfg, name, kind):
            save_config(self._cfg)
            if self._active == name:
                self._update_title()
            self._build_menu()

    # ── Hide/unhide callbacks ─────────────────────────────────────────────────

    def _on_hide_model(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        hidden = self._cfg["hidden_models"]
        if name not in hidden:
            hidden.append(name)
            save_config(self._cfg)
        self._build_menu()

    def _on_benchmark(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        kind = self._model_kind(name)
        bconfig = run_benchmark_config_panel(name, kind, self._cfg)
        if bconfig is None:
            return

        entry = self._model_map.get(name)
        path = entry[0] if entry else None

        self._benchmarking = True
        self._loading = True
        self._schedule_rebuild()
        if kind == "gguf":
            self._open_bench_progress_window(name)

        def _do_bench():
            log = open("/tmp/ms_bench_log.txt", "w")
            try:
                log.write(f"bench start: mode={bconfig.mode} name={name}\n"
                          f"  cache_types_k={bconfig.cache_types_k}\n"
                          f"  cache_types_v={bconfig.cache_types_v}\n"
                          f"  batch={bconfig.batch_sizes} ubatch={bconfig.ubatch_sizes}\n"
                          f"  flash={bconfig.flash_attns} n_prompt={bconfig.n_prompt} n_gen={bconfig.n_gen}\n"); log.flush()
                if bconfig.mode == "llama-bench":
                    if not path:
                        results = [BenchmarkResult(label="error", run=1, total_ms=0,
                                                   tokens_out=0, tok_per_sec=0,
                                                   error="model path unknown")]
                    else:
                        log.write(f"calling run_llama_bench path={path}\n"); log.flush()
                        results = run_llama_bench(self._cfg, name, path, bconfig,
                                                  self._bench_progress)
                else:
                    log.write(f"calling run_api_benchmark prompts={bconfig.prompts}\n"); log.flush()
                    results = run_api_benchmark(self._cfg, name, kind, bconfig)
                log.write(f"bench done: {len(results)} results\n")
                for r in results:
                    log.write(f"  {r.label!r} tps={r.tok_per_sec:.1f} err={r.error!r}\n")
                log.flush()
            except Exception as e:
                log.write(f"bench exception: {e}\n"); log.flush()
                results = [BenchmarkResult(label="error", run=1, total_ms=0,
                                           tokens_out=0, tok_per_sec=0,
                                           error=str(e))]
            finally:
                log.close()
            save_bench_run(name, bconfig.mode, results)
            open("/tmp/ms_bench_log.txt", "a").write("setting pending_bench\n")
            self._pending_bench = (name, results, bconfig.mode)
            self._benchmarking = False
            self._loading = False   # flash timer sees this and fires _on_flash_tick on main thread

        threading.Thread(target=_do_bench, daemon=True).start()

    def _open_bench_history(self, _):
        run_bench_history_panel()

    def _open_test_prompt(self, _):
        if self._test_prompt_win is not None:
            self._test_prompt_win.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
        else:
            self._test_prompt_win = _make_test_prompt_window(self)
            NSApp.activateIgnoringOtherApps_(True)
            self._test_prompt_win.makeKeyAndOrderFront_(None)

    def _open_profiles(self, _):
        all_models = sorted(self._model_map.keys())
        presets = list(SAMPLING_PRESETS.keys())
        profiles = load_profiles()
        # Show a simple manage panel: list existing + Add button
        new_profile = run_create_profile_panel(all_models, presets)
        if new_profile:
            profiles.append(new_profile)
            save_profiles(profiles)
            self._build_menu()

    def _on_apply_profile(self, sender):
        p = getattr(sender, "_profile_data", None)
        if not p:
            return
        model = p.get("model")
        if not model or model not in self._model_map:
            rumps.alert("Profile Error", f"Model '{model}' not found.")
            return
        # Apply preset sampling params if specified
        if p.get("preset") and p["preset"] in SAMPLING_PRESETS:
            mp = dict(self._params(model))
            mp.update(SAMPLING_PRESETS[p["preset"]])
            self._cfg.setdefault("model_params", {})[model] = mp
            save_config(self._cfg)
        # Switch to the model
        class _Sender:
            _model_name = model
        self._on_select(_Sender())

    def _on_unhide_model(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        hidden = self._cfg["hidden_models"]
        if name in hidden:
            hidden.remove(name)
            save_config(self._cfg)
        self._build_menu()


if __name__ == "__main__":
    ModelSwitcher().run()
