#!/usr/bin/env python3
"""Switchman — macOS menu bar app for switching local LLM inference engines."""
import json
import os
import socket
import struct
import subprocess
import threading
import time
import uuid
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import rumps
import objc

try:
    from AppKit import (
        NSApp, NSBackingStoreBuffered, NSBox, NSButton, NSFont,
        NSModalResponseOK, NSOpenPanel, NSPanel, NSPopUpButton,
        NSScrollView, NSTextField, NSTextView, NSURL, NSView, NSVisualEffectView,
        NSWindow,
    )
    from Foundation import NSObject
    from WebKit import WKWebView, WKWebViewConfiguration
    try:
        from UserNotifications import (
            UNUserNotificationCenter, UNMutableNotificationContent,
            UNNotificationRequest,
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

_MS_CONFIG_DIR       = Path.home() / ".config" / "switchman"
BENCHMARK_PROMPTS_PATH = _MS_CONFIG_DIR / "benchmark_prompts.json"
BENCH_HISTORY_PATH   = _MS_CONFIG_DIR / "bench_history.json"
PROFILES_PATH        = _MS_CONFIG_DIR / "profiles.json"
PENDING_DOWNLOAD_PATH = _MS_CONFIG_DIR / "pending_download.json"
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
    from AppKit import NSColor
    f = NSTextField.alloc().initWithFrame_(frame)
    f.setStringValue_(text)
    f.setBezeled_(False)
    f.setDrawsBackground_(False)
    f.setEditable_(False)
    f.setSelectable_(False)
    f.setAlignment_(1 if right else 0)   # 1 = NSRightTextAlignment
    if bold:
        f.setFont_(NSFont.boldSystemFontOfSize_(12))
        f.setTextColor_(NSColor.controlAccentColor())
    else:
        f.setTextColor_(NSColor.labelColor())
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


def _menu_header(label: str) -> "rumps.MenuItem":
    """Menu section header with accent color."""
    from AppKit import (NSAttributedString, NSColor, NSFont,
                        NSForegroundColorAttributeName, NSFontAttributeName)
    item = rumps.MenuItem(label, callback=None)
    attrs = {
        NSForegroundColorAttributeName: NSColor.colorWithRed_green_blue_alpha_(0.03, 0.15, 0.50, 1.0),
        NSFontAttributeName:            NSFont.boldSystemFontOfSize_(13),
    }
    item._menuitem.setAttributedTitle_(
        NSAttributedString.alloc().initWithString_attributes_(label, attrs))
    return item


def _primary_btn(title: str, target, action: str, frame,
                 key_eq: str | None = None) -> NSButton:
    """Primary action button with system accent color."""
    from AppKit import NSColor
    b = _btn(title, target, action, frame, key_eq)
    b.setBezelColor_(NSColor.controlAccentColor())
    return b


def _vibrancy_content_view(window) -> NSVisualEffectView:
    """Replace the window's plain content view with a vibrant NSVisualEffectView.
    Returns the view so callers can use it as `cv` and add subviews normally."""
    from AppKit import NSColor
    vfx = NSVisualEffectView.alloc().initWithFrame_(window.contentView().frame())
    vfx.setMaterial_(12)   # NSVisualEffectMaterialWindowBackground
    vfx.setBlendingMode_(0)  # NSVisualEffectBlendingModeBehindWindow
    vfx.setState_(1)  # NSVisualEffectStateActive
    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setContentView_(vfx)
    return vfx


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

def show_error_alert(title: str, message: str) -> None:
    """Show an NSAlert on the main thread. Call only from main thread."""
    from AppKit import NSAlert
    alert = NSAlert.alloc().init()
    alert.setMessageText_(title)
    alert.setInformativeText_(message)
    alert.addButtonWithTitle_("OK")
    NSApp.activateIgnoringOtherApps_(True)
    alert.runModal()


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

    def exportCSV_(self, _s):
        try:
            data = json.loads(BENCH_HISTORY_PATH.read_text()) if BENCH_HISTORY_PATH.exists() else []
        except Exception:
            return
        lines = ["date,model,mode,label,tok_per_sec,tokens,total_ms,error"]
        for run in data:
            for r in run.get("results", []):
                lines.append(",".join([
                    run.get("timestamp", run.get("date", "")),
                    run.get("model", ""), run.get("mode", ""),
                    f'"{r.get("label","")}"',
                    str(r.get("tok_per_sec", 0)), str(r.get("tokens_out", 0)),
                    str(r.get("total_ms", 0)), f'"{r.get("error","")}"',
                ]))
        from AppKit import NSSavePanel
        p = NSSavePanel.savePanel()
        p.setNameFieldStringValue_("bench_history.csv")
        p.setAllowedFileTypes_(["csv"])
        NSApp.activateIgnoringOtherApps_(True)
        if p.runModal() == NSModalResponseOK:
            Path(p.URL().path()).write_text("\n".join(lines) + "\n")

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
        self._t_first = None
        self._ttft_shown = False
        self._prompt_tokens = 0
        self._ctx_max = 0
        self._drain_timer = rumps.Timer(self.drainTick_, 0.1)
        self._drain_timer.start()
        threading.Thread(target=self._do_stream,
                         args=(prompt,), daemon=True).start()
        if hasattr(self, '_buf2'):
            self._buf2.clear()
            self._streaming2 = False
            self._tok_count2 = 0
            self._t_first2 = None
            self._ttft_shown2 = False
            model2_name = None
            if hasattr(self, '_model2_popup') and hasattr(self, '_compare_chk'):
                if self._compare_chk.state():
                    model2_name = self._model2_popup.titleOfSelectedItem()
            if model2_name:
                self._streaming2 = True
                threading.Thread(target=self._do_stream2,
                                 args=(prompt, model2_name), daemon=True).start()

    def _do_stream(self, prompt):
        try:
            cfg = self._app_ref._cfg
            port = cfg.get("omlx_port", 8000)
            api_key = cfg.get("omlx_api_key", "")
            model = self._app_ref._active or ""
            p = self._app_ref._model_params(model) if hasattr(self._app_ref, "_model_params") else {}
            self._prompt_tokens = 0
            self._ctx_max = (p.get("context", 32768) if p else 32768)
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": p.get("max_tokens", 512) if p else 512,
                "stream": True,
                "stream_options": {"include_usage": True},
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
                        # Final usage chunk (stream_options: include_usage)
                        if "usage" in chunk and chunk.get("choices") == []:
                            usage = chunk["usage"]
                            self._prompt_tokens = usage.get("prompt_tokens", 0)
                            self._tok_count = usage.get("completion_tokens", self._tok_count)
                            continue
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if self._t_first is None:
                                self._t_first = time.time()
                            self._buf.append(delta)
                            self._tok_count += 1
                    except Exception:
                        pass
        except Exception as e:
            self._buf.append(f"\n\n[Error: {e}]")
        finally:
            self._streaming = False

    def drainTick_(self, timer):
        from AppKit import NSAttributedString
        if self._buf:
            chunk = "".join(self._buf)
            self._buf.clear()
            ts = self._output_tv.textStorage()
            ts.appendAttributedString_(
                NSAttributedString.alloc().initWithString_(chunk))
            self._output_tv.scrollRangeToVisible_(
                (self._output_tv.string().length(), 0))
        # Drain buf2 into second output view if compare mode is active
        if hasattr(self, '_buf2') and self._buf2 and hasattr(self, '_output_tv2'):
            chunk2 = "".join(self._buf2)
            self._buf2.clear()
            ts2 = self._output_tv2.textStorage()
            ts2.appendAttributedString_(
                NSAttributedString.alloc().initWithString_(chunk2))
            self._output_tv2.scrollRangeToVisible_(
                (self._output_tv2.string().length(), 0))
        elapsed = time.time() - self._t0
        if self._tok_count > 0 and elapsed > 0:
            ttft_part = ""
            if self._t_first is not None and not self._ttft_shown:
                ttft_ms = (self._t_first - self._t0) * 1000
                ttft_part = f"TTFT {ttft_ms:.0f}ms  |  "
                self._ttft_shown = True
            elif self._ttft_shown and self._t_first is not None:
                ttft_ms = (self._t_first - self._t0) * 1000
                ttft_part = f"TTFT {ttft_ms:.0f}ms  |  "
            ctx_part = ""
            prompt_toks = getattr(self, '_prompt_tokens', 0)
            ctx_max = getattr(self, '_ctx_max', 0)
            if prompt_toks > 0 and ctx_max > 0:
                used = prompt_toks + self._tok_count
                pct = used / ctx_max * 100
                ctx_part = f"  |  ctx {used:,}/{ctx_max:,} ({pct:.0f}%)"
            self._tps_lbl.setStringValue_(
                f"{ttft_part}{self._tok_count / elapsed:.1f} tok/s  ({self._tok_count} tokens){ctx_part}")
        streaming2 = getattr(self, '_streaming2', False)
        buf2_empty = not getattr(self, '_buf2', [])
        if not self._streaming and not self._buf and not streaming2 and buf2_empty:
            timer.stop()

    def clear_(self, _s):
        self._output_tv.setString_("")
        self._tps_lbl.setStringValue_("")
        self._input_fld.setStringValue_("")
        if hasattr(self, '_output_tv2'):
            self._output_tv2.setString_("")

    def compareChanged_(self, sender):
        on = bool(sender.state())
        if hasattr(self, '_model2_popup'):
            self._model2_popup.setHidden_(not on)
        if hasattr(self, '_model2_lbl'):
            self._model2_lbl.setHidden_(not on)
        if hasattr(self, '_scroll2'):
            self._scroll2.setHidden_(not on)

    def _do_stream2(self, prompt, model2_name):
        try:
            cfg = self._app_ref._cfg
            port = cfg.get("omlx_port", 8000)
            api_key = cfg.get("omlx_api_key", "")
            p = self._app_ref._model_params(model2_name) if hasattr(self._app_ref, "_model_params") else {}
            body = json.dumps({
                "model": model2_name,
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
                            if self._t_first2 is None:
                                self._t_first2 = time.time()
                            self._buf2.append(delta)
                            self._tok_count2 += 1
                    except Exception:
                        pass
        except Exception as e:
            self._buf2.append(f"\n\n[Error: {e}]")
        finally:
            self._streaming2 = False


def _make_test_prompt_window(app) -> NSWindow:
    """Build and return the Quick Test Prompt NSWindow (non-modal)."""
    W, H = 760, 480
    BOT = 48
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 7, NSBackingStoreBuffered, False)
    win.setTitle_("Quick Test Prompt")
    win.center()
    win.setTitlebarAppearsTransparent_(True)
    win.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(win)

    handler = _TestPromptHandler.alloc().init()
    handler._app_ref = app
    handler._streaming = False
    handler._buf = []
    handler._t0 = 0.0
    handler._tok_count = 0
    handler._t_first = None
    handler._ttft_shown = False
    handler._drain_timer = None
    handler._prompt_tokens = 0
    handler._ctx_max = 0
    # Compare mode state
    handler._buf2 = []
    handler._streaming2 = False
    handler._tok_count2 = 0
    handler._t_first2 = None
    handler._ttft_shown2 = False
    # handler kept alive by caller storing it alongside the window

    INPUT_H = 28
    ROW2_Y = H - _PAD - INPUT_H - _GAP - INPUT_H  # y for second row
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

    # Compare checkbox
    compare_chk = NSButton.alloc().initWithFrame_(
        ((_PAD, ROW2_Y), (140, INPUT_H)))
    compare_chk.setButtonType_(3)
    compare_chk.setTitle_("Compare models")
    compare_chk.setState_(0)
    compare_chk.setTarget_(handler)
    compare_chk.setAction_("compareChanged:")
    cv.addSubview_(compare_chk)
    handler._compare_chk = compare_chk

    # Model2 label + popup (hidden by default)
    model2_lbl = _lbl("Model 2:", ((_PAD + 145, ROW2_Y), (70, INPUT_H)), right=False)
    model2_lbl.setHidden_(True)
    cv.addSubview_(model2_lbl)
    handler._model2_lbl = model2_lbl

    all_models = sorted(app._model_map.keys()) if hasattr(app, '_model_map') else []
    model2_popup = NSPopUpButton.alloc().initWithFrame_(
        ((_PAD + 145 + 75, ROW2_Y - 2), (W - _PAD*2 - 145 - 75 - 70, INPUT_H + 4)))
    for m in (all_models or ["(no models)"]):
        model2_popup.addItemWithTitle_(m)
    model2_popup.setHidden_(True)
    cv.addSubview_(model2_popup)
    handler._model2_popup = model2_popup

    # Output scroll area — left panel (always visible)
    output_top = ROW2_Y - _GAP
    output_h = output_top - BOT
    half_w = (W - _PAD*2 - _GAP) // 2
    scroll = NSScrollView.alloc().initWithFrame_(
        ((_PAD, BOT), (half_w, output_h)))
    scroll.setHasVerticalScroller_(True); scroll.setAutohidesScrollers_(True)
    tv = NSTextView.alloc().initWithFrame_(((0, 0), (half_w, output_h)))
    tv.setFont_(NSFont.userFixedPitchFontOfSize_(12))
    tv.setEditable_(False)
    scroll.setDocumentView_(tv); cv.addSubview_(scroll)
    handler._output_tv = tv

    # Right panel for compare (hidden by default)
    scroll2 = NSScrollView.alloc().initWithFrame_(
        ((_PAD + half_w + _GAP, BOT), (half_w, output_h)))
    scroll2.setHasVerticalScroller_(True); scroll2.setAutohidesScrollers_(True)
    tv2 = NSTextView.alloc().initWithFrame_(((0, 0), (half_w, output_h)))
    tv2.setFont_(NSFont.userFixedPitchFontOfSize_(12))
    tv2.setEditable_(False)
    scroll2.setDocumentView_(tv2)
    scroll2.setHidden_(True)
    cv.addSubview_(scroll2)
    handler._output_tv2 = tv2
    handler._scroll2 = scroll2

    tps_lbl = NSTextField.alloc().initWithFrame_(
        ((_PAD, 12), (W - _PAD*2 - 70, 22)))
    tps_lbl.setBezeled_(False); tps_lbl.setDrawsBackground_(False)
    tps_lbl.setEditable_(False)
    tps_lbl.setFont_(NSFont.systemFontOfSize_(11))
    cv.addSubview_(tps_lbl)
    handler._tps_lbl = tps_lbl

    cv.addSubview_(_btn("Clear", handler, "clear:",
                        ((W - _PAD - 64, 12), (64, 22))))
    return win, handler


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
    panel.setTitle_("Switchman — Settings")
    panel.setDelegate_(handler)
    panel.center()
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)
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
    cv.addSubview_(_primary_btn("Save", handler, "stopOK:",
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
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)

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
    cv.addSubview_(_primary_btn("Save", handler, "stopOK:",
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
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)

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
    cv.addSubview_(_primary_btn("Save", handler, "stopOK:",
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
    #              +cache_K label(20)+chks(30)+cache_V label(20) = 306
    #              (ctv hardcoded to f16 — no checkbox row)
    # API:         header(20)+n_prompts(150)+gap(16)+header(20)+2 rows(60)=266 ≤ 306
    SECTION_H = (_SH + _SG + 3 * (_RH + _RG)
                 + _DG + _SH + _SG + 3 * (_RH + _RG)
                 + _SH + _SG + _RH + _RG   # cache K label + checkboxes
                 + _SH)                     # cache V label only (no checkboxes)

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
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)

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
        # ctv is hardcoded to f16 — quantized V-cache is not supported by the
        # Metal backend for most model architectures on Apple Silicon.
        bench_container.addSubview_(_lbl(
            "Cache type V: f16 (Metal only)",
            ((_PAD, sfy(sc, _SH)), (W - 2*_PAD, _SH)), right=False))
        sc += _SH + _SG
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
    cv.addSubview_(_primary_btn("Run Benchmark", handler, "stopOK:",
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
    ctv_list = ["f16"]  # V-cache quant unsupported on Metal — always f16

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
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)

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
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)
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
    cv.addSubview_(_btn("Export CSV…", handler, "exportCSV:",
                        ((_PAD + 108, _BTN_BOT), (96, _BTN_H))))
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


def save_pending_download(repo_id: str, dest_dir: str, filter_tag: str) -> None:
    try:
        PENDING_DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        PENDING_DOWNLOAD_PATH.write_text(json.dumps(
            {"repo_id": repo_id, "dest_dir": dest_dir, "filter": filter_tag}))
    except Exception:
        pass


def clear_pending_download() -> None:
    try:
        PENDING_DOWNLOAD_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def load_pending_download() -> dict | None:
    try:
        if PENDING_DOWNLOAD_PATH.exists():
            return json.loads(PENDING_DOWNLOAD_PATH.read_text())
    except Exception:
        pass
    return None


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
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)

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
    cv.addSubview_(_primary_btn("Save", handler, "stopOK:",
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

CONFIG_PATH = Path.home() / ".config" / "switchman" / "config.json"

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
    "known_models":   [],         # [model_name, ...] all models ever seen (for new-model detection)
    "recent_models":  [],         # [model_name, ...] max 5, most recent first
    "default_model":  "",         # model name to auto-load on startup (empty = none)
    # Feature flags
    "sync_env":       True,       # write ~/.config/switchman/env on switch
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
            if not isinstance(cfg.get("known_models"), list):
                cfg["known_models"] = []
            if not isinstance(cfg.get("recent_models"), list):
                cfg["recent_models"] = []
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
    """Fire a local macOS notification via osascript.

    UNUserNotificationCenter requires a bundle ID, which a bare Python script
    doesn't have — permission requests silently fail.  osascript works from any
    context with no bundle or permission prompt needed.
    """
    try:
        import subprocess
        safe_name = name.replace('"', '\\"')
        subprocess.Popen(
            ["osascript", "-e",
             f'display notification "Model ready: {safe_name}" with title "Switchman"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def request_notification_permission() -> None:
    """No-op — osascript needs no permission prompt."""
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


def sync_env_file(port: int) -> None:
    """Write LLM_BASE_URL to ~/.config/switchman/env for shell sourcing."""
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
        # Capture the new window reference so we write into the right session,
        # not whatever session happens to be frontmost.
        script = f"""tell application "iTerm"
    activate
    set newWin to (create window with default profile)
    tell current session of newWin
        write text "cd '{safe_cwd}' && {safe_cmd}"
    end tell
end tell"""
    else:
        # do script without a target always opens a new Terminal window.
        script = f"""tell application "Terminal"
    activate
    do script "cd '{safe_cwd}' && {safe_cmd}"
end tell"""
    subprocess.run(["osascript", "-e", script], capture_output=True)


# ── Quick model search ────────────────────────────────────────────────────────

class _ModelSearchDelegate(objc.lookUpClass("NSObject")):
    """Data source + delegate for the quick-search results table."""

    def initWithApp_(self, app):
        self = objc.super(_ModelSearchDelegate, self).init()
        if self is None:
            return None
        self._app = app
        self._all_names: list[str] = []
        self._filtered: list[str] = []
        self._table = None
        self._panel = None
        self._engine_filter = "All"   # "All" | "MLX" | "GGUF"
        return self

    def setNames_(self, names):
        self._all_names = names
        self._filtered = names[:]

    def applyFilter_(self, query: str):
        q = query.lower()
        result = []
        for n in self._all_names:
            # Engine filter
            if self._engine_filter != "All":
                entry = self._app._model_map.get(n)
                kind = (entry[1] if entry else "mlx").upper()
                if kind != self._engine_filter:
                    continue
            # Text filter
            if q and q not in (self._app._alias(n) or n).lower() and q not in n.lower():
                continue
            result.append(n)
        self._filtered = result
        if self._table:
            self._table.reloadData()
            if self._filtered:
                self._table.selectRowIndexes_byExtendingSelection_(
                    objc.lookUpClass("NSIndexSet").indexSetWithIndex_(0), False)

    # NSTableViewDataSource
    def numberOfRowsInTableView_(self, tv):
        return len(self._filtered)

    def tableView_objectValueForTableColumn_row_(self, tv, col, row):
        if row >= len(self._filtered):
            return ""
        name = self._filtered[row]
        alias = self._app._alias(name)
        display = alias or name
        is_active = self._app._active == name
        is_default = self._app._cfg.get("default_model") == name
        prefix = "▶ " if is_active else ("★ " if is_default else "  ")
        # Pull size and context from meta cache
        meta = self._app._get_model_meta(name)
        size_str = next((m.strip() for m in meta if "size:" in m), "")
        ctx_str  = next((m.strip() for m in meta if "ctx:"  in m), "")
        detail = "  —  " + "  ".join(filter(None, [ctx_str, size_str])) if (size_str or ctx_str) else ""
        return f"{prefix}{display}{detail}"

    def tableViewSelectionDidChange_(self, note):
        pass

    def selectCurrent(self):
        if not self._table:
            return
        row = self._table.selectedRow()
        if 0 <= row < len(self._filtered):
            name = self._filtered[row]
            self._panel.orderOut_(None)
            class _FakeSender:
                pass
            s = _FakeSender()
            s._model_name = name
            self._app._on_select(s)


def _open_model_search(app) -> None:
    """Show a floating search panel for quick model switching.
    ⌥⌘Space opens it directly without going through the menu."""
    from AppKit import NSTableView, NSTableColumn, NSSegmentedControl

    W, H = 520, 340
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)),
        # titled | closable | resizable | nonactivating
        0b1000000000000 | 2 | 1 | 8,
        NSBackingStoreBuffered, False)
    panel.setTitle_("Switch Model")
    panel.center()
    panel.setLevel_(8)  # NSFloatingWindowLevel
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)

    y = H - _PAD

    # ── Search field ──────────────────────────────────────────────────────────
    y -= 28
    sf = NSTextField.alloc().initWithFrame_(((_PAD, y), (W - _PAD*2, 28)))
    sf.setPlaceholderString_("Type to filter models…  (⌥⌘Space to open)")
    sf.setBezeled_(True)
    sf.setBezelStyle_(1)
    sf.setFont_(NSFont.systemFontOfSize_(14.0))
    cv.addSubview_(sf)

    # ── Engine filter (All / MLX / GGUF) ─────────────────────────────────────
    y -= _RG + 24
    seg = NSSegmentedControl.alloc().initWithFrame_(((_PAD, y), (180, 24)))
    seg.setSegmentCount_(3)
    seg.setLabel_("All",  0)
    seg.setLabel_("MLX",  1)
    seg.setLabel_("GGUF", 2)
    seg.setSelectedSegment_(0)
    seg.setSegmentStyle_(1)   # NSSegmentStyleRounded
    cv.addSubview_(seg)

    # ── Results table ─────────────────────────────────────────────────────────
    y -= _RG + 4
    table_h = y - _PAD - 36  # leave room for button row
    scroll = NSScrollView.alloc().initWithFrame_(((_PAD, _PAD + 36), (W - _PAD*2, table_h)))
    scroll.setHasVerticalScroller_(True)
    scroll.setAutohidesScrollers_(True)

    tv = NSTableView.alloc().initWithFrame_(((0, 0), (W - _PAD*2, table_h)))
    col = NSTableColumn.alloc().initWithIdentifier_("model")
    col.setWidth_(W - _PAD*2 - 20)
    tv.addTableColumn_(col)
    tv.setHeaderView_(None)
    tv.setUsesAlternatingRowBackgroundColors_(True)
    tv.setRowHeight_(22)
    scroll.setDocumentView_(tv)
    cv.addSubview_(scroll)

    # ── Load Model button ─────────────────────────────────────────────────────
    sel_btn = NSButton.alloc().initWithFrame_(
        ((W - _PAD - 110, _PAD), (110, 28)))
    sel_btn.setTitle_("Load Model")
    sel_btn.setBezelStyle_(1)
    sel_btn.setKeyEquivalent_("\r")
    cv.addSubview_(sel_btn)

    # ── Wire delegate ─────────────────────────────────────────────────────────
    all_names = sorted(
        [n for n in app._model_map if n not in app._cfg.get("hidden_models", [])],
        key=lambda n: _hf_sort_key({"modelId": app._alias(n) or n})
    )
    delegate = _ModelSearchDelegate.alloc().initWithApp_(app)
    delegate.setNames_(all_names)
    delegate._table = tv
    delegate._panel = panel
    tv.setDataSource_(delegate)
    tv.setDelegate_(delegate)
    tv.reloadData()
    if all_names:
        tv.selectRowIndexes_byExtendingSelection_(
            objc.lookUpClass("NSIndexSet").indexSetWithIndex_(0), False)

    # ── Search field delegate (text change + keyboard nav) ────────────────────
    class _SFDelegate(objc.lookUpClass("NSObject")):
        def controlTextDidChange_(self, note):
            delegate.applyFilter_(sf.stringValue())

        def control_textView_doCommandBySelector_(self, ctrl, tv2, sel):
            if "insertNewline" in str(sel):
                delegate.selectCurrent()
                return True
            if "moveDown" in str(sel):
                row = min(self._tv.selectedRow() + 1, len(delegate._filtered) - 1)
                self._tv.selectRowIndexes_byExtendingSelection_(
                    objc.lookUpClass("NSIndexSet").indexSetWithIndex_(row), False)
                self._tv.scrollRowToVisible_(row)
                return True
            if "moveUp" in str(sel):
                row = max(self._tv.selectedRow() - 1, 0)
                self._tv.selectRowIndexes_byExtendingSelection_(
                    objc.lookUpClass("NSIndexSet").indexSetWithIndex_(row), False)
                self._tv.scrollRowToVisible_(row)
                return True
            return False

    sfd = _SFDelegate.alloc().init()
    sfd._tv = tv
    sf.setDelegate_(sfd)

    # ── Segmented control → engine filter ─────────────────────────────────────
    class _SegTarget(objc.lookUpClass("NSObject")):
        def changed_(self, sender):
            labels = ["All", "MLX", "GGUF"]
            idx = sender.selectedSegment()
            delegate._engine_filter = labels[idx] if 0 <= idx < 3 else "All"
            delegate.applyFilter_(sf.stringValue())

    seg_target = _SegTarget.alloc().init()
    seg.setTarget_(seg_target)
    seg.setAction_(objc.selector(seg_target.changed_, signature=b"v@:@"))

    # ── Load Model button ─────────────────────────────────────────────────────
    class _BtnTarget(objc.lookUpClass("NSObject")):
        def clicked_(self, _s):
            delegate.selectCurrent()

    btn_target = _BtnTarget.alloc().init()
    sel_btn.setTarget_(btn_target)
    sel_btn.setAction_(objc.selector(btn_target.clicked_, signature=b"v@:@"))

    # Keep strong refs on the app (NSPanel won't accept Python attrs)
    app._search_panel = panel
    app._search_delegate = delegate
    app._search_sfd = sfd
    app._search_seg = seg_target
    app._search_btn = btn_target

    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    panel.makeFirstResponder_(sf)


# ── App ───────────────────────────────────────────────────────────────────────

class Switchman(rumps.App):
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
        # Feature: live tok/s + context usage
        self._last_toks: float | None = None  # unused; kept for compat
        self._ctx_used: int | None = None   # prompt+completion tokens from last poll
        self._ctx_max: int | None = None    # context window for active model
        self._tps_poll_timer: rumps.Timer | None = None
        self._start_poll_on_rebuild = False
        # Feature: test prompt
        self._test_prompt_win = None
        self._test_prompt_handler = None
        self._hf_download_win = None
        self._hf_download_handler = None
        # Feature: model metadata cache
        self._model_meta_cache: dict[str, list[str]] = {}
        self._rebuild_pending = False   # set from bg thread; polled by idle timer
        # Feature: memory pressure
        self._mem_pressure: str = "nominal"
        # Feature: error dialog
        self._pending_error: tuple | None = None
        # Feature: server crash watchdog
        self._watchdog_timer: rumps.Timer | None = None
        # Feature: loading step detail
        self._load_status: str = "Loading model…"
        # Switch token — incremented on every new selection; threads check it
        # to detect supersession and exit without clobbering _active/_loading.
        self._switch_token: int = 0

        self._build_menu()
        self._init_known_models()  # seed known_models on first run (no prompts)
        self._prime_meta_cache()   # sets _rebuild_pending when done
        threading.Thread(target=self._sync_state, daemon=True).start()
        self._notif_permission_requested = False
        self._mem_pressure_timer = rumps.Timer(self._on_mem_pressure_tick, 30)
        self._mem_pressure_timer.start()
        # Idle timer: polls _rebuild_pending set by background threads that
        # cannot create timers themselves (rumps.Timer from bg thread = no-op).
        self._idle_timer = rumps.Timer(self._on_idle_tick, 1)
        self._idle_timer.start()
        # Global hotkey ⌥Space — open the status bar menu from keyboard
        threading.Thread(target=self._register_hotkey, daemon=True).start()

    # ── Global hotkey ─────────────────────────────────────────────────────────

    def _register_hotkey(self):
        """Register ⌥Space as a global hotkey to pop the menu bar icon.
        Runs in a background thread with its own CFRunLoop.
        Requires Accessibility permission (System Settings → Privacy → Accessibility).
        Silently no-ops if Quartz is unavailable or permission is denied."""
        try:
            from Quartz import (CGEventTapCreate, CGEventTapEnable,
                                CFMachPortCreateRunLoopSource, CFRunLoopGetCurrent,
                                CFRunLoopAddSource, CFRunLoopRun,
                                kCGSessionEventTap, kCGHeadInsertEventTap,
                                kCGEventFlagMaskAlternate, kCGEventFlagMaskCommand, kCGEventKeyDown,
                                CGEventGetIntegerValueField, kCGKeyboardEventKeycode)
            from CoreFoundation import kCFRunLoopCommonModes
        except ImportError:
            return

        SPACE_KEYCODE = 49

        def _callback(proxy, etype, event, refcon):
            try:
                flags = event.intValueForField_(kCGEventFlagMaskAlternate) if hasattr(
                    event, 'intValueForField_') else 0
                keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                mods = event.flags() if hasattr(event, 'flags') else 0
                if keycode == SPACE_KEYCODE and (mods & kCGEventFlagMaskAlternate):
                    if mods & kCGEventFlagMaskCommand:
                        self._open_model_search(None)
                    else:
                        self._open_menu_from_hotkey()
            except Exception:
                pass
            return event

        try:
            tap = CGEventTapCreate(
                kCGSessionEventTap,
                kCGHeadInsertEventTap,
                0,
                1 << kCGEventKeyDown,
                _callback,
                None,
            )
            if tap is None:
                return  # Accessibility permission not granted
            src = CFMachPortCreateRunLoopSource(None, tap, 0)
            loop = CFRunLoopGetCurrent()
            CFRunLoopAddSource(loop, src, kCFRunLoopCommonModes)
            CGEventTapEnable(tap, True)
            CFRunLoopRun()
        except Exception:
            pass

    def _open_menu_from_hotkey(self):
        """Simulate a click on the status bar item to open the menu."""
        try:
            from AppKit import NSStatusBar
            bar = NSStatusBar.systemStatusBar()
            # rumps stores the status item; trigger via performClick on the button
            if hasattr(self, '_status_item'):
                self._status_item.button().performClick_(None)
            else:
                # Walk status bar items to find ours by title
                for item in (bar.statusItemWithLength_(-1),):
                    pass
        except Exception:
            pass

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
                try:
                    if path.is_dir():
                        # MLX model — sum all files in directory
                        size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    else:
                        size_bytes = path.stat().st_size
                    size_gb = size_bytes / 1_073_741_824
                    if size_gb >= 0.1:
                        labels.append(f"  size: {size_gb:.1f} GB")
                except Exception:
                    pass
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
        if self._pending_error:
            title, msg = self._pending_error
            self._pending_error = None
            show_error_alert(title, msg)

    # ── Memory pressure ───────────────────────────────────────────────────────

    def _on_mem_pressure_tick(self, _timer):
        def _check():
            self._mem_pressure = get_memory_pressure()
        threading.Thread(target=_check, daemon=True).start()

    # ── Live tok/s polling ────────────────────────────────────────────────────

    def _start_tps_poll(self):
        # tok/s probe removed — inaccurate (measures 8-token idle ping,
        # not real generation speed). Watchdog still runs independently.
        self._start_watchdog()

    def _stop_tps_poll(self):
        self._stop_watchdog()

    def _start_watchdog(self):
        if self._watchdog_timer is not None:
            return
        self._watchdog_timer = rumps.Timer(self._on_watchdog_tick, 30)
        self._watchdog_timer.start()

    def _stop_watchdog(self):
        if self._watchdog_timer:
            self._watchdog_timer.stop()
            self._watchdog_timer = None

    def _on_watchdog_tick(self, _timer):
        if not self._active or self._loading:
            return
        def _check():
            try:
                port = self._cfg.get("omlx_port", 8000)
                api_key = self._cfg.get("omlx_api_key", "")
                req = urllib.request.Request(
                    f"http://localhost:{port}/health",
                    headers={"Authorization": f"Bearer {api_key}"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    r.read()
                # Server is alive — do nothing
            except Exception:
                # Server is down
                self._active = None
                self._last_toks = None
                self._stop_tps_poll()
                self._rebuild_pending = True
                if self._cfg.get("notifications", True):
                    try:
                        content = UNMutableNotificationContent.alloc().init()
                        content.setTitle_("Switchman")
                        content.setBody_("Inference server stopped unexpectedly")
                        req2 = UNNotificationRequest.requestWithIdentifier_content_trigger_(
                            str(uuid.uuid4()), content, None)
                        def _noop(e): pass
                        UNUserNotificationCenter.currentNotificationCenter() \
                            .addNotificationRequest_withCompletionHandler_(req2, _noop)
                    except Exception:
                        pass
        threading.Thread(target=_check, daemon=True).start()

    def _on_tps_tick(self, _timer):
        pass  # probe removed; method kept so any lingering timer refs are harmless

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

        # ── Recent models section ─────────────────────────────────────────────────
        default_model = self._cfg.get("default_model")
        recent_names = [
            n for n in self._cfg.get("recent_models", [])
            if n in self._model_map and (n not in hidden or n == default_model)
        ]
        if recent_names:
            menu.append(_menu_header("── Recent ──"))
            for rname in recent_names:
                ritem = rumps.MenuItem(f"  {self._display(rname)}", callback=self._on_select)
                ritem._model_name = rname
                if self._active == rname and not self._loading:
                    ritem.state = 1
                menu.append(ritem)
            menu.append(None)

        # ── Profiles section ────────────────��────────────────────────────��────
        profiles = load_profiles()
        if profiles:
            menu.append(_menu_header("── Profiles ──"))
            for p in profiles:
                item = rumps.MenuItem(f"  {p['name']}", callback=self._on_apply_profile)
                item._profile_data = p
                menu.append(item)
            menu.append(None)

        if mlx_visible:
            menu.append(_menu_header("── MLX ──"))
            for name in mlx_visible:
                menu.append(self._make_model_item(name))
            menu.append(None)

        if gguf_visible:
            menu.append(_menu_header("── GGUF ──"))
            for name in gguf_visible:
                menu.append(self._make_model_item(name))
            menu.append(None)

        if not mlx_visible and not gguf_visible and not hidden_present:
            menu.append(rumps.MenuItem("No models found", callback=None))
            menu.append(None)

        menu += [
            rumps.MenuItem("⏹  Stop Engine", callback=self._stop),
            rumps.MenuItem("↻  Refresh Models", callback=self._refresh),
            rumps.MenuItem("🔍  Search Models…", callback=self._open_model_search),
            rumps.MenuItem("⬇  Download from HuggingFace…", callback=self._open_hf_download),
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
        is_default = self._cfg.get("default_model") == name

        prefix = "★ " if is_default else "  "
        parent = rumps.MenuItem(f"{prefix}{alias or name}")
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

        copy_item = rumps.MenuItem("⎘  Copy model ID", callback=self._on_copy_model_id)
        copy_item._model_name = name
        parent.add(copy_item)

        is_default = self._cfg.get("default_model") == name
        default_item = rumps.MenuItem(
            "★  Default at startup" if not is_default else "★  Default at startup ✓",
            callback=self._on_set_default)
        default_item._model_name = name
        parent.add(default_item)

        hide_item = rumps.MenuItem("⊘  Hide", callback=self._on_hide_model)
        hide_item._model_name = name
        parent.add(hide_item)

        delete_item = rumps.MenuItem("🗑  Delete model…", callback=self._on_delete_model)
        delete_item._model_name = name
        parent.add(delete_item)
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
        s.add(rumps.MenuItem("  Manage Visible Models…", callback=self._open_manage_models))
        s.add(None)
        s.add(rumps.MenuItem("  Quick Test Prompt…", callback=self._open_test_prompt))
        s.add(rumps.MenuItem("  Benchmark History…", callback=self._open_bench_history))
        s.add(None)
        s.add(rumps.MenuItem("  Export Settings…", callback=self._export_settings))
        s.add(rumps.MenuItem("  Import Settings…", callback=self._import_settings))
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
        timer.stop()
        self._update_title()
        try:
            self._build_menu()
        except Exception:
            import traceback
            import logging
            logging.error("_build_menu exception:\n%s", traceback.format_exc())
        if self._start_poll_on_rebuild:
            self._start_poll_on_rebuild = False
            self._start_tps_poll()
        if self._pending_bench:
            args, self._pending_bench = self._pending_bench, None
            try:
                run_benchmark_results_panel(*args)
            except Exception:
                import traceback
                import logging
                logging.error("results panel exception:\n%s", traceback.format_exc())

    def _update_title(self):
        if self._loading:
            self._start_flash()
        else:
            self._stop_flash()
            mem_dot = "🔴" if self._mem_pressure == "critical" else ""
            if self._active:
                self.title = f"⚡{mem_dot} {self._title_label(self._active)}"
            else:
                self.title = f"⚡{mem_dot}"

    def _start_flash(self):
        if self._flash_timer:
            return
        self._flash_state = True
        self.title = "Benchmarking…" if self._benchmarking else self._load_status
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
                    import logging
                    logging.error("results panel exception:\n%s", traceback.format_exc())
            return
        self._flash_state = not self._flash_state
        label = "Benchmarking…" if self._benchmarking else self._load_status
        self.title = label if self._flash_state else "⚡"

    # ── Model callbacks ───────────────────────────────────────────────────────

    def _update_recent(self, name: str):
        recent = self._cfg.setdefault("recent_models", [])
        if name in recent:
            recent.remove(name)
        recent.insert(0, name)
        self._cfg["recent_models"] = recent[:5]
        save_config(self._cfg)

    def _on_select(self, sender: rumps.MenuItem):
        self._last_toks = None
        self._ctx_used = None
        self._ctx_max = None
        self._stop_tps_poll()
        name = getattr(sender, "_model_name", None) or sender.title.strip()
        entry = self._model_map.get(name)
        if entry is None:
            return
        path, kind = entry
        self._update_recent(name)
        # Increment token — any in-progress load will see it changed and abort
        self._switch_token += 1
        token = self._switch_token
        self._active = name
        self._loading = True
        self._update_title()
        if kind == "mlx":
            threading.Thread(target=self._switch_mlx, args=(name, token), daemon=True).start()
        else:
            threading.Thread(target=self._switch_gguf, args=(name, path, token), daemon=True).start()

    def _stop(self, _=None):
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _refresh(self, _):
        self._build_menu()
        self._check_new_models()

    # ── Engine switching ──────────────────────────────────────────────────────

    def _superseded(self, token: int) -> bool:
        """Return True if a newer switch request has come in since this thread started."""
        return self._switch_token != token

    def _switch_mlx(self, name: str, token: int):
        """Kill anything on the port, start omlx, trigger model load and wait."""
        self._load_status = "Stopping engine…"
        self._kill_gguf()
        if self._superseded(token): return

        if not omlx_is_healthy(self._cfg):
            self._load_status = "Starting oMLX…"
            kill_port(self._cfg["omlx_port"])
            if not omlx_start(self._cfg):
                if self._superseded(token): return
                self._active = None
                self._pending_error = (
                    "Model failed to load",
                    "oMLX failed to start. Check ~/Library/Logs/switchman.log",
                )
                self._loading = False
                self._rebuild_pending = True
                return

        if self._superseded(token): return
        p = self._params(name)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg['omlx_api_key']}",
        }
        url = f"http://localhost:{self._cfg['omlx_port']}/v1/chat/completions"
        sampling = mlx_sampling_params(p)

        self._load_status = "Loading weights…"
        deadline = time.time() + 300
        while time.time() < deadline:
            if self._superseded(token): return
            resp = http_post(url, body={
                "model": name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                **sampling,
            }, headers=headers, timeout=300)
            if resp and resp.get("model") == name:
                break
            time.sleep(2)

        if self._superseded(token): return
        self._load_status = "Warming up…"
        http_post(url, body={
            "model": name,
            "messages": [{"role": "user", "content": "Write a short Python hello world function."}],
            "max_tokens": 128,
            **sampling,
        }, headers=headers, timeout=300)

        if self._superseded(token): return
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

    def _switch_gguf(self, name: str, path: Path, token: int):
        """Stop omlx, wait for port, start llama-server."""
        self._load_status = "Stopping engine…"
        self._kill_gguf()
        omlx_stop(self._cfg)
        if self._superseded(token): return

        if not port_is_free(self._cfg["llama_port"]):
            kill_port(self._cfg["llama_port"])

        if self._superseded(token): return
        self._load_status = "Starting llama-server…"
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

        self._load_status = "Loading weights…"
        model_id = name
        if not wait_for_port_open(self._cfg["llama_port"], timeout=30):
            if self._superseded(token): return
            self._active = None
            self._pending_error = (
                "Model failed to load",
                "llama-server failed to start. Check ~/Library/Logs/switchman.log",
            )
            self._loading = False
            self._rebuild_pending = True
            return
        if self._superseded(token): return
        time.sleep(0.5)
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
        self._stop_tps_poll()
        self._stop_watchdog()
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
        """On startup: detect GGUF if llama-server is already running, then
        auto-load the default model if one is configured and nothing is active."""
        if not port_is_free(self._cfg["llama_port"]) and not omlx_is_healthy(self._cfg):
            model_id = self._query_llama_model_id()
            if model_id:
                for name, (_, kind) in self._model_map.items():
                    if kind == "gguf" and (model_id.startswith(name) or name in model_id):
                        self._active = name
                        self._rebuild_pending = True
                        break

        default = self._cfg.get("default_model", "")
        if default and default in self._model_map and self._active is None:
            # Trigger the normal select flow from the background thread
            entry = self._model_map.get(default)
            if entry:
                path, kind = entry
                self._switch_token += 1
                token = self._switch_token
                self._active = default
                self._loading = True
                self._load_status = "Loading default model…"
                self._rebuild_pending = True
                if kind == "mlx":
                    threading.Thread(target=self._switch_mlx,
                                     args=(default, token), daemon=True).start()
                else:
                    threading.Thread(target=self._switch_gguf,
                                     args=(default, path, token), daemon=True).start()

    # ── Benchmark progress window ─────────────────────────────────────────────

    def _open_bench_progress_window(self, name: str) -> None:
        """Create and show a non-modal progress window. Called on main thread."""
        W, H = 560, 320
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (W, H)), 3, NSBackingStoreBuffered, False)
        win.setTitle_(f"Benchmarking — {name}")
        win.center()
        win.setTitlebarAppearsTransparent_(True)
        win.setMovableByWindowBackground_(True)
        cv = _vibrancy_content_view(win)

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

    def _export_settings(self, _):
        from AppKit import NSSavePanel
        panel = NSSavePanel.savePanel()
        panel.setTitle_("Export Switchman Settings")
        panel.setNameFieldStringValue_("switchman-settings.json")
        panel.setAllowedFileTypes_(["json"])
        NSApp.activateIgnoringOtherApps_(True)
        if panel.runModal() == NSModalResponseOK:
            path = Path(panel.URL().path())
            path.write_text(json.dumps(self._cfg, indent=2, default=list))

    def _import_settings(self, _):
        from AppKit import NSOpenPanel
        panel = NSOpenPanel.openPanel()
        panel.setTitle_("Import Switchman Settings")
        panel.setAllowedFileTypes_(["json"])
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        NSApp.activateIgnoringOtherApps_(True)
        if panel.runModal() == NSModalResponseOK:
            path = Path(panel.URL().path())
            try:
                imported = json.loads(path.read_text())
                if not isinstance(imported, dict):
                    raise ValueError("not a dict")
                # Merge: preserve current paths, overlay everything else
                for key, val in imported.items():
                    if key not in ("mlx_dir", "gguf_dir", "llama_server",
                                   "omlx_port", "omlx_api_key", "omlx_service"):
                        self._cfg[key] = val
                save_config(self._cfg)
                self._build_menu()
            except Exception as e:
                show_error_alert("Import failed", str(e))

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

    def _on_copy_model_id(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        model_id = f"omlx/{name}"
        from AppKit import NSPasteboard, NSStringPboardType
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(model_id, NSStringPboardType)

    def _on_set_default(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        if self._cfg.get("default_model") == name:
            self._cfg["default_model"] = ""   # toggle off
        else:
            self._cfg["default_model"] = name
        save_config(self._cfg)
        self._build_menu()

    def _on_hide_model(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        hidden = self._cfg["hidden_models"]
        if name not in hidden:
            hidden.append(name)
            save_config(self._cfg)
        self._build_menu()

    def _on_delete_model(self, sender: rumps.MenuItem):
        import shutil
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        entry = self._model_map.get(name)
        if not entry:
            return
        path = Path(entry[0])  # entry is (path, kind)
        # For a directory model, delete the dir; for a bare .gguf, delete the file
        if path.is_dir():
            target = path
        elif path.is_file():
            # If the file sits alone in its own directory, delete the directory
            target = path.parent if list(path.parent.glob("*.gguf")) == [path] else path
        else:
            return

        display = self._alias(name) or name
        size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9 \
            if target.is_dir() else target.stat().st_size / 1e9
        confirm = rumps.alert(
            title=f"Delete {display}?",
            message=f"This will permanently delete:\n{target}\n\n{size_gb:.1f} GB will be freed. This cannot be undone.",
            ok="Delete", cancel="Cancel",
        )
        if confirm != 1:
            return
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except Exception as e:
            show_error_alert("Delete failed", str(e))
            return
        # Clean up config references
        for d in (self._cfg["hidden_models"],):
            if name in d:
                d.remove(name)
        self._cfg["model_notes"].pop(name, None)
        self._cfg["model_params"].pop(name, None)
        self._cfg["aliases"].pop(name, None)
        if self._cfg.get("default_model") == name:
            self._cfg["default_model"] = ""
        if name in self._cfg.get("recent_models", []):
            self._cfg["recent_models"].remove(name)
        if self._active == name:
            self._active = None
        save_config(self._cfg)
        self._model_meta_cache.pop(name, None)
        self._rebuild_pending = True

    def _open_model_search(self, _):
        _open_model_search(self)

    def _on_benchmark(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        kind = self._model_kind(name)
        bconfig = run_benchmark_config_panel(name, kind, self._cfg)
        if bconfig is None:
            return

        # Pre-flight: for API benchmarks the right server must be reachable.
        # MLX: oMLX lazy-loads models — just needs oMLX healthy (checked in run_api_benchmark).
        # GGUF: llama-server is single-model — must have this model loaded.
        if bconfig.mode == "api" and kind == "gguf":
            if self._gguf_proc is None or self._gguf_proc.poll() is not None:
                show_error_alert(
                    "Model not loaded",
                    f"Select \"{self._display(name)}\" from the menu to load it first, "
                    f"then run the API benchmark.",
                )
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
            self._test_prompt_win, self._test_prompt_handler = _make_test_prompt_window(self)
            NSApp.activateIgnoringOtherApps_(True)
            self._test_prompt_win.makeKeyAndOrderFront_(None)

    def _open_hf_download(self, _):
        if self._hf_download_win is not None:
            self._hf_download_win.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
        else:
            self._hf_download_win, self._hf_download_handler = _make_hf_download_window(self)
            NSApp.activateIgnoringOtherApps_(True)
            self._hf_download_win.makeKeyAndOrderFront_(None)
            h = self._hf_download_handler
            h._updateDiskSpace()
            # Restore interrupted download if one exists
            pending = load_pending_download()
            if pending:
                h._filter_popup.selectItemWithTitle_(pending.get("filter", "MLX"))
                h.filterChanged_(None)
                h._dest_fld.setStringValue_(pending.get("dest_dir", ""))
                h._updateDiskSpace()
                h._query_fld.setStringValue_(pending.get("repo_id", ""))
                h._status_lbl.setStringValue_(
                    "⟳  Resuming interrupted download — click ⬇ Download to continue")
                h._auto_select_first = True
                h.search_(None)

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

    # ── Model visibility management ───────────────────────────────────────────

    def _init_known_models(self):
        """Seed known_models on startup without prompting the user."""
        known = set(self._cfg.get("known_models", []))
        current = set(self._model_map.keys())
        updated = known | current
        if updated != known:
            self._cfg["known_models"] = sorted(updated)
            save_config(self._cfg)

    def _check_new_models(self):
        """After a manual refresh, prompt for any newly discovered models."""
        known = set(self._cfg.get("known_models", []))
        current = set(self._model_map.keys())
        new_models = current - known
        # Mark all current models as known regardless of user choice
        self._cfg["known_models"] = sorted(known | current)
        save_config(self._cfg)
        if not new_models:
            return
        hidden = self._cfg["hidden_models"]
        changed = False
        for name in sorted(new_models):
            kind = self._model_map[name][1].upper()
            display = self._alias(name) or name
            resp = rumps.alert(
                title="New Model Found",
                message=f"[{kind}]  {display}\n\nAdd this model to the menu?",
                ok="Add to Menu",
                cancel="Hide for Now",
            )
            if resp != 1:  # user chose "Hide for Now"
                if name not in hidden:
                    hidden.append(name)
                    changed = True
        if changed:
            save_config(self._cfg)
            self._build_menu()

    def _open_manage_models(self, _=None):
        """Open a checklist panel to show/hide models and clear the model list."""
        from AppKit import (
            NSWindow, NSScrollView, NSView, NSButton,
            NSMakeRect,
            NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
            NSBackingStoreBuffered, NSSwitchButton,
        )

        if getattr(self, "_manage_win", None):
            try:
                self._manage_win.makeKeyAndOrderFront_(None)
                return
            except Exception:
                self._manage_win = None

        all_models = sorted(
            self._model_map.keys(),
            key=lambda n: _hf_sort_key({"modelId": self._alias(n) or n}),
        )
        hidden_set = set(self._cfg["hidden_models"])

        ROW_H = 24
        W, PADDING = 480, 16
        inner_h = max(len(all_models) * ROW_H + PADDING * 2, 80)
        SCROLL_H = min(inner_h, 380)
        FOOTER_H = 44
        H = SCROLL_H + FOOTER_H + 36  # scroll + footer + title bar

        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (W, H)),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered, False,
        )
        win.setTitle_("Manage Visible Models")
        win.setLevel_(3)
        win.center()
        win.setTitlebarAppearsTransparent_(True)
        win.setMovableByWindowBackground_(True)
        cv = _vibrancy_content_view(win)

        # ── Scroll area with checkboxes ──────────────────────────────────────
        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(PADDING, FOOTER_H + 8, W - 2 * PADDING, SCROLL_H)
        )
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(1)

        inner = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, W - 2 * PADDING - 4, inner_h)
        )

        self._manage_checkboxes = {}
        y = inner_h - PADDING
        for name in all_models:
            y -= ROW_H
            kind = self._model_map[name][1].upper()
            label = f"[{kind}]  {self._alias(name) or name}"
            cb = NSButton.alloc().initWithFrame_(
                NSMakeRect(6, y, W - 2 * PADDING - 20, ROW_H)
            )
            cb.setButtonType_(NSSwitchButton)
            cb.setTitle_(label)
            cb.setState_(0 if name in hidden_set else 1)
            inner.addSubview_(cb)
            self._manage_checkboxes[name] = cb

        scroll.setDocumentView_(inner)
        cv.addSubview_(scroll)

        # ── Footer buttons ────────────────────────────────────────────────────
        handler = _ManageModelsHandler.alloc().initWithApp_(self)
        self._manage_handler = handler  # keep strong ref

        for (label, action, x, w) in [
            ("Select All",  "selectAll:",   PADDING,           84),
            ("Clear All",   "clearAll:",    PADDING + 94,      84),
            ("Save",        "save:",        W - PADDING - 84,  84),
            ("Cancel",      "cancel:",      W - PADDING - 176, 84),
        ]:
            btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, 8, w, 28))
            btn.setTitle_(label)
            btn.setBezelStyle_(1)
            btn.setTarget_(handler)
            btn.setAction_(action)
            cv.addSubview_(btn)

        self._manage_win = win
        win.makeKeyAndOrderFront_(None)


# ── HuggingFace download ──────────────────────────────────────────────────────

def _hf_search(query: str, tag: str, limit: int = 30) -> list[dict]:
    """Search HuggingFace models API. Returns list of model dicts."""
    try:
        import urllib.parse
        params = urllib.parse.urlencode({
            "search": query, "filter": tag, "limit": limit,
            "sort": "downloads", "direction": -1,
        })
        req = urllib.request.Request(
            f"https://huggingface.co/api/models?{params}",
            headers={"User-Agent": "switchman/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return []


def _hf_model_size_gb(model_info: dict) -> float:
    """Extract total size in GB from HF model info, or 0 if unknown."""
    try:
        siblings = model_info.get("siblings", [])
        total = sum(s.get("size", 0) for s in siblings if s.get("size"))
        return total / 1_073_741_824
    except Exception:
        return 0.0


def _hf_parse_params(model_id: str) -> float:
    """Parse parameter count in billions from a model ID string.
    Returns a float (e.g. 7.0, 70.0, 0.5, 235.0) or 0.0 if not found."""
    import re
    # Patterns: 70B, 7b, 0.5B, 235B, 3x8B (MoE total = 24), 8x7B
    moe = re.search(r'(\d+)[xX](\d+(?:\.\d+)?)[Bb]', model_id)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    m = re.search(r'(\d+(?:\.\d+)?)[Bb](?:[^a-zA-Z]|$)', model_id)
    if m:
        return float(m.group(1))
    return 0.0


# Quant ordering: higher quality = higher index (sorted ascending = worst first,
# so we negate for "best quant first" ordering).
_QUANT_ORDER = {
    "f32": 10, "f16": 9, "bf16": 9,
    "8bit": 8, "q8_0": 8, "q8": 8,
    "6bit": 7, "q6_k": 7, "q6": 7,
    "5bit": 6, "q5_k_m": 6, "q5_k_s": 6, "q5_0": 5, "q5_1": 5,
    "4bit": 4, "q4_k_m": 4, "q4_k_s": 4, "q4_0": 3, "q4_1": 3,
    "iq4_xs": 4, "iq4_nl": 4,
    "3bit": 2, "q3_k_m": 2, "q3_k_s": 2, "q3_k_l": 2,
    "2bit": 1, "q2_k": 1,
}


def _hf_parse_quant(model_id: str) -> int:
    """Return a quant quality score (higher = better). 0 if unknown."""
    lower = model_id.lower()
    # Try longest match first
    for key in sorted(_QUANT_ORDER, key=len, reverse=True):
        if key in lower:
            return _QUANT_ORDER[key]
    return 0


def _hf_sort_key(model_info: dict):
    """Sort key: source org, then base model name, then param count asc, then quant desc."""
    import re
    model_id = model_info.get("modelId", model_info.get("id", ""))
    # Split org/model-name
    if "/" in model_id:
        org, base = model_id.split("/", 1)
    else:
        org, base = "", model_id
    # Strip quant/bit/format suffixes for clean alphabetic model grouping
    base_clean = re.sub(
        r'[-_]?(?:\d+(?:\.\d+)?[Bb]it|[QqIi][Qq]?\d+[_\-]?[A-Za-z0-9]*|[Ff]16|[Ff]32|[Bb][Ff]16|MLX|GGUF|CRACK).*$',
        '', base, flags=re.IGNORECASE).strip("-_ ")
    params = _hf_parse_params(model_id)
    quant = _hf_parse_quant(model_id)
    return (org.lower(), base_clean.lower(), params, -quant)


class _HFDownloadHandler(NSObject):
    """Action target for the HuggingFace download window."""

    # ── Search ────────────────────────────────────────────────────────────────

    def search_(self, _s):
        query = self._query_fld.stringValue().strip()
        tag = "mlx" if self._filter_popup.titleOfSelectedItem() == "MLX" else "gguf"
        self._info_lbl.setStringValue_("Searching…")
        self._results = []
        self._pending_items = None   # set by bg thread, picked up by poll timer

        def _do():
            models = _hf_search(query, tag)
            models.sort(key=_hf_sort_key)
            items = []
            for m in models:
                sz = _hf_model_size_gb(m)
                sz_str = f"  {sz:.1f} GB" if sz >= 0.01 else ""
                params = _hf_parse_params(m.get("modelId", m.get("id", "")))
                quant = _hf_parse_quant(m.get("modelId", m.get("id", "")))
                quant_str = next((k for k, v in _QUANT_ORDER.items()
                                  if v == quant and quant > 0), "")
                parts = []
                if params > 0:
                    parts.append(f"{params:g}B")
                if quant_str:
                    parts.append(quant_str.upper())
                meta_str = f"  [{', '.join(parts)}]" if parts else ""
                items.append(f"{m.get('modelId', m.get('id', '?'))}{meta_str}{sz_str}")
            self._results = models
            self._pending_items = items   # signal main thread

        threading.Thread(target=_do, daemon=True).start()
        # Poll for results on the main thread
        if self._search_poll is not None:
            self._search_poll.stop()
        self._search_poll = rumps.Timer(self.pollSearch_, 0.2)
        self._search_poll.start()

    def pollSearch_(self, timer):
        if self._pending_items is None:
            return
        timer.stop()
        self._search_poll = None
        items = self._pending_items
        self._pending_items = None
        self._results_popup.removeAllItems()
        if items:
            for it in items:
                self._results_popup.addItemWithTitle_(it)
            self._results_popup.selectItemAtIndex_(0)
            self._updateInfo()
            if getattr(self, '_auto_select_first', False):
                self._auto_select_first = False
        else:
            self._results_popup.addItemWithTitle_("No results")
            self._info_lbl.setStringValue_("No results found.")

    def filterChanged_(self, _s):
        tag = self._filter_popup.titleOfSelectedItem()
        d = self._app_ref._cfg.get(
            "mlx_dir" if tag == "MLX" else "gguf_dir",
            str(Path.home() / "models"))
        self._dest_fld.setStringValue_(d)
        self._updateDiskSpace()

    def _updateDiskSpace(self):
        try:
            import shutil
            path = Path(self._dest_fld.stringValue().strip()).expanduser()
            if path.exists():
                usage = shutil.disk_usage(path)
                free_gb = usage.free / 1_073_741_824
                total_gb = usage.total / 1_073_741_824
                pct_used = (usage.used / usage.total) * 100
                color_str = "🔴" if pct_used > 90 else ("🟡" if pct_used > 75 else "🟢")
                self._disk_lbl.setStringValue_(
                    f"{color_str}  {free_gb:.1f} GB free of {total_gb:.0f} GB")
            else:
                self._disk_lbl.setStringValue_("⚠️  Directory not found")
        except Exception:
            self._disk_lbl.setStringValue_("")

    def resultChanged_(self, _s):
        self._updateInfo()

    def _updateInfo(self):
        idx = self._results_popup.indexOfSelectedItem()
        if idx < 0 or idx >= len(self._results):
            return
        m = self._results[idx]
        repo_id = m.get("modelId", m.get("id", ""))
        downloads = m.get("downloads", 0)
        likes = m.get("likes", 0)
        sz = _hf_model_size_gb(m)
        sz_str = f"{sz:.1f} GB  |  " if sz >= 0.01 else ""
        self._info_lbl.setStringValue_(
            f"{sz_str}↓ {downloads:,}  ❤ {likes:,}  —  {repo_id}")

    # ── Download ──────────────────────────────────────────────────────────────

    def download_(self, _s):
        if self._downloading:
            return
        idx = self._results_popup.indexOfSelectedItem()
        if idx < 0 or idx >= len(self._results):
            self._status_lbl.setStringValue_("Select a result first.")
            return
        m = self._results[idx]
        repo_id = m.get("modelId", m.get("id", ""))
        dest_dir = Path(self._dest_fld.stringValue().strip()).expanduser()
        if not dest_dir.exists():
            self._status_lbl.setStringValue_(f"Directory not found: {dest_dir}")
            return
        model_dir = dest_dir / repo_id.split("/")[-1]
        # Byte counter updated directly by our tqdm shim on every HTTP chunk.
        byte_counter = type('_C', (), {'bytes_done': 0})()
        tqdm_cls = _make_hf_tqdm_class(byte_counter)
        self._downloading = True
        self._dl_error = None
        self._dl_success_path = None
        self._dl_model_dir = model_dir
        self._dl_byte_counter = byte_counter
        self._dl_bytes_total = 0
        self._dl_size_fetched = False
        self._dl_prev_done = 0
        self._dl_prev_time = 0.0
        self._dl_btn.setEnabled_(False)
        self._progress.setIndeterminate_(False)
        self._progress.setMinValue_(0.0)
        self._progress.setMaxValue_(100.0)
        self._progress.setDoubleValue_(0.0)
        self._status_lbl.setStringValue_(f"Fetching size for {repo_id}…")
        save_pending_download(repo_id, str(dest_dir),
                              self._filter_popup.titleOfSelectedItem())

        def _do():
            try:
                import fnmatch
                import requests
                from huggingface_hub import model_info as hf_model_info, hf_hub_url
                from huggingface_hub.utils import build_hf_headers

                ignore = ["*.bin", "*.pt", "original/*"]

                # Fetch file list + sizes
                try:
                    info = hf_model_info(repo_id, files_metadata=True)
                    siblings = info.siblings or []
                except Exception:
                    siblings = []

                # Filter ignored patterns
                files = [
                    s for s in siblings
                    if not any(fnmatch.fnmatch(s.rfilename, pat) for pat in ignore)
                ]

                total_bytes = sum((s.size or 0) for s in files)
                self._dl_bytes_total = total_bytes
                self._dl_size_fetched = True

                headers = build_hf_headers()
                model_dir.mkdir(parents=True, exist_ok=True)

                for sib in files:
                    fname = sib.rfilename
                    url = hf_hub_url(repo_id, fname)
                    dest = model_dir / fname
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    expected = sib.size or 0
                    resume = dest.stat().st_size if dest.exists() else 0
                    # Skip files already fully downloaded
                    if expected and resume >= expected:
                        byte_counter.bytes_done += expected
                        continue
                    req_headers = dict(headers)
                    if resume:
                        req_headers["Range"] = f"bytes={resume}-"
                        byte_counter.bytes_done += resume
                    with requests.get(url, headers=req_headers, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        mode = "ab" if resume else "wb"
                        with open(dest, mode) as fh:
                            for chunk in r.iter_content(chunk_size=256 * 1024):
                                if chunk:
                                    fh.write(chunk)
                                    byte_counter.bytes_done += len(chunk)

                self._dl_success_path = str(model_dir)
            except Exception as e:
                self._dl_error = str(e)

        threading.Thread(target=_do, daemon=True).start()
        self._dl_poll = rumps.Timer(self.pollDownload_, 1.0)
        self._dl_poll.start()

    def pollDownload_(self, timer):
        total = self._dl_bytes_total
        size_fetched = getattr(self, '_dl_size_fetched', False)

        # Still fetching size
        if not size_fetched and self._dl_error is None and self._dl_success_path is None:
            return

        # Read bytes received directly from the tqdm shim — no disk polling.
        now = time.time()
        counter = getattr(self, '_dl_byte_counter', None)
        done = counter.bytes_done if counter is not None else 0

        # Compute MB/s from delta since last poll
        prev_done = getattr(self, '_dl_prev_done', 0)
        prev_time = getattr(self, '_dl_prev_time', now)
        dt = now - prev_time
        speed_str = ""
        if dt > 0 and done > prev_done:
            mbps = (done - prev_done) / dt / 1_048_576
            speed_str = f"  {mbps:.1f} MB/s"
        self._dl_prev_done = done
        self._dl_prev_time = now

        if total > 0:
            pct = min(done / total * 100, 99.0)
            self._progress.setDoubleValue_(pct)
            done_gb = done / 1_073_741_824
            total_gb = total / 1_073_741_824
            self._status_lbl.setStringValue_(
                f"{done_gb:.2f} / {total_gb:.2f} GB  ({pct:.0f}%){speed_str}")
        elif size_fetched:
            done_gb = done / 1_073_741_824
            self._status_lbl.setStringValue_(f"{done_gb:.2f} GB downloaded…{speed_str}")

        if self._dl_success_path is None and self._dl_error is None:
            return   # still running

        timer.stop()
        self._dl_poll = None
        self._dl_btn.setEnabled_(True)
        self._downloading = False
        clear_pending_download()
        if self._dl_error:
            self._progress.setDoubleValue_(0.0)
            self._status_lbl.setStringValue_(f"Error: {self._dl_error}")
            self._dl_error = None
        else:
            success_path = self._dl_success_path
            self._progress.setDoubleValue_(100.0)
            self._status_lbl.setStringValue_(f"✓ Done — {success_path}")
            self._dl_success_path = None
            if self._app_ref is not None:
                app = self._app_ref
                model_name = Path(success_path).name

                # Hide by default — ask the user what they want
                if model_name not in app._cfg["hidden_models"]:
                    app._cfg["hidden_models"].append(model_name)
                    save_config(app._cfg)

                app._model_meta_cache.clear()
                app._rebuild_pending = True
                app._prime_meta_cache()

                # Ask whether to add to menu / set as default / keep hidden
                from AppKit import NSAlert
                alert = NSAlert.alloc().init()
                alert.setMessageText_(f'"{model_name}" downloaded')
                alert.setInformativeText_(
                    "Add it to the model menu, set it as your default (★), "
                    "or keep it hidden until you're ready.")
                alert.addButtonWithTitle_("Set as Default ★")
                alert.addButtonWithTitle_("Add to Menu")
                alert.addButtonWithTitle_("Keep Hidden")
                NSApp.activateIgnoringOtherApps_(True)
                resp = alert.runModal()

                if resp in (1000, 1001):   # Set as Default OR Add to Menu
                    if model_name in app._cfg["hidden_models"]:
                        app._cfg["hidden_models"].remove(model_name)
                    if resp == 1000:       # Set as Default
                        app._cfg["default_model"] = model_name
                    save_config(app._cfg)
                    app._rebuild_pending = True

    # ── Browse / Close ────────────────────────────────────────────────────────

    def browse_(self, _s):
        from AppKit import NSOpenPanel
        p = NSOpenPanel.openPanel()
        p.setCanChooseFiles_(False)
        p.setCanChooseDirectories_(True)
        p.setAllowsMultipleSelection_(False)
        NSApp.activateIgnoringOtherApps_(True)
        if p.runModal() == NSModalResponseOK:
            self._dest_fld.setStringValue_(str(Path(p.URL().path())))
            self._updateDiskSpace()

    def closeWin_(self, _s):
        self._win_ref.orderOut_(None)
        if self._app_ref is not None:
            self._app_ref._hf_download_win = None
            self._app_ref._hf_download_handler = None


def _make_hf_tqdm_class(counter):
    """Return a minimal tqdm-compatible class that tallies downloaded bytes
    into counter.bytes_done.  snapshot_download calls update(chunk_bytes)
    for every HTTP chunk it writes, giving true real-time byte progress."""
    import threading as _th

    class _HFTqdm:
        _lock = _th.Lock()

        @classmethod
        def get_lock(cls):
            return cls._lock

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock

        def __init__(self, iterable=None, total=None, unit=None,
                     unit_scale=None, initial=0, desc=None, **kw):
            self._iterable = iterable
            self.total = total or 0
            self.n = initial or 0

        # iterable support (used for the file-list loop inside snapshot_download)
        def __iter__(self):
            if self._iterable is None:
                return iter([])
            for item in self._iterable:
                yield item

        def __len__(self):
            if self._iterable is not None:
                try:
                    return len(self._iterable)
                except TypeError:
                    pass
            return self.total or 0

        def __bool__(self):
            return True

        def update(self, n=1):
            self.n += n
            counter.bytes_done += n

        def refresh(self, nolock=False, **kw):
            pass

        def clear(self, nolock=False):
            pass

        def reset(self, total=None):
            if total is not None:
                self.total = total
            self.n = 0

        def close(self):
            pass

        def set_description(self, desc=None, refresh=True):
            pass

        def set_description_str(self, s=None, refresh=True):
            pass

        def set_postfix(self, *a, **kw):
            pass

        def write(self, s, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    return _HFTqdm


class _HFWindowDelegate(NSObject):
    """Clears app refs when the download window is closed via the X button."""
    def initWithApp_(self, app):
        self = objc.super(_HFWindowDelegate, self).init()
        self._app = app
        return self

    def windowWillClose_(self, note):
        self._app._hf_download_win = None
        self._app._hf_download_handler = None


class _ManageModelsHandler(NSObject):
    """Action target for the Manage Visible Models panel."""

    def initWithApp_(self, app):
        self = objc.super(_ManageModelsHandler, self).init()
        self._app = app
        return self

    def selectAll_(self, _s):
        for cb in self._app._manage_checkboxes.values():
            cb.setState_(1)

    def clearAll_(self, _s):
        for cb in self._app._manage_checkboxes.values():
            cb.setState_(0)

    def save_(self, _s):
        app = self._app
        hidden = []
        for name, cb in app._manage_checkboxes.items():
            if cb.state() == 0:
                hidden.append(name)
        app._cfg["hidden_models"] = hidden
        save_config(app._cfg)
        app._build_menu()
        if app._manage_win:
            app._manage_win.orderOut_(None)
            app._manage_win = None

    def cancel_(self, _s):
        app = self._app
        if app._manage_win:
            app._manage_win.orderOut_(None)
            app._manage_win = None


def _make_hf_download_window(app):
    """Build the HuggingFace download NSWindow. Returns (win, handler)."""
    W, H = 640, 320
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 7, NSBackingStoreBuffered, False)
    win.setTitle_("Download from HuggingFace")
    win.center()
    win.setTitlebarAppearsTransparent_(True)
    win.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(win)

    win_delegate = _HFWindowDelegate.alloc().initWithApp_(app)
    win.setDelegate_(win_delegate)
    app._hf_win_delegate = win_delegate  # keep strong ref on app (NSWindow won't accept Python attrs)

    handler = _HFDownloadHandler.alloc().init()
    handler._app_ref = app
    handler._win_ref = win
    handler._results = []
    handler._downloading = False
    handler._dl_error = None
    handler._dl_success_path = None
    handler._dl_bytes_total = 0
    handler._dl_size_fetched = False
    handler._dl_model_dir = Path(".")
    handler._dl_byte_counter = None
    handler._dl_prev_done = 0
    handler._dl_prev_time = 0.0
    handler._search_poll = None
    handler._dl_poll = None
    handler._pending_items = None
    handler._auto_select_first = False

    y = H - _PAD
    ROW = _RH + _RG

    # ── Row 1: Filter selector (full width, prominent) ────────────────────────
    y -= _RH
    cv.addSubview_(_lbl("Type:", ((_PAD, y), (50, _RH)), right=False))
    fp = NSPopUpButton.alloc().initWithFrame_(
        ((_PAD + 55, y - 2), (100, _RH + 4)))
    fp.addItemWithTitle_("MLX")
    fp.addItemWithTitle_("GGUF")
    fp.setTarget_(handler)
    fp.setAction_("filterChanged:")
    cv.addSubview_(fp)
    handler._filter_popup = fp

    # ── Row 2: Search field + button ─────────────────────────────────────────
    y -= ROW
    cv.addSubview_(_lbl("Search:", ((_PAD, y), (50, _RH)), right=False))
    qf = NSTextField.alloc().initWithFrame_(
        ((_PAD + 55, y), (W - _PAD*2 - 55 - 80, _RH)))
    qf.setPlaceholderString_("e.g. Qwen3, llama, mistral")
    cv.addSubview_(qf)
    handler._query_fld = qf
    cv.addSubview_(_btn("Search", handler, "search:",
                        ((W - _PAD - 74, y), (74, _RH)), "\r"))

    # ── Row 3: Results dropdown ───────────────────────────────────────────────
    y -= ROW
    cv.addSubview_(_lbl("Result:", ((_PAD, y), (50, _RH)), right=False))
    rp = NSPopUpButton.alloc().initWithFrame_(
        ((_PAD + 55, y - 2), (W - _PAD*2 - 55, _RH + 4)))
    rp.addItemWithTitle_("— search above —")
    rp.setTarget_(handler)
    rp.setAction_("resultChanged:")
    cv.addSubview_(rp)
    handler._results_popup = rp

    # ── Row 4: Info line (size / downloads / likes) ───────────────────────────
    y -= ROW
    info = _lbl("", ((_PAD, y), (W - _PAD*2, _RH)), right=False)
    cv.addSubview_(info)
    handler._info_lbl = info

    # ── Row 5: Destination dir ────────────────────────────────────────────────
    y -= ROW
    cv.addSubview_(_lbl("Save to:", ((_PAD, y), (50, _RH)), right=False))
    default_dir = app._cfg.get("mlx_dir", str(Path.home() / "models"))
    df = NSTextField.alloc().initWithFrame_(
        ((_PAD + 55, y), (W - _PAD*2 - 55 - 74, _RH)))
    df.setStringValue_(default_dir)
    cv.addSubview_(df)
    handler._dest_fld = df
    cv.addSubview_(_btn("Browse…", handler, "browse:",
                        ((W - _PAD - 68, y), (68, _RH))))

    # ── Disk space indicator ───────────────────────────────────────────────────
    y -= _RH + 2
    disk_lbl = NSTextField.alloc().initWithFrame_(((_PAD, y), (W - _PAD*2, _RH)))
    disk_lbl.setStringValue_("")
    disk_lbl.setBezeled_(False)
    disk_lbl.setDrawsBackground_(False)
    disk_lbl.setEditable_(False)
    disk_lbl.setSelectable_(False)
    disk_lbl.setFont_(NSFont.systemFontOfSize_(10.0))
    cv.addSubview_(disk_lbl)
    handler._disk_lbl = disk_lbl

    # ── Progress bar ──────────────────────────────────────────────────────────
    y -= ROW
    from AppKit import NSProgressIndicator
    prog = NSProgressIndicator.alloc().initWithFrame_(
        ((_PAD, y + 4), (W - _PAD*2, 12)))
    prog.setStyle_(0)
    prog.setIndeterminate_(False)
    prog.setDoubleValue_(0.0)
    cv.addSubview_(prog)
    handler._progress = prog

    # ── Status label ──────────────────────────────────────────────────────────
    y -= 22
    sl = NSTextField.alloc().initWithFrame_(((_PAD, y), (W - _PAD*2, _RH)))
    sl.setStringValue_("")
    sl.setBezeled_(False)
    sl.setDrawsBackground_(False)
    sl.setEditable_(False)
    sl.setSelectable_(True)   # allows copy-paste of error text
    sl.setLineBreakMode_(0)   # NSLineBreakByWordWrapping
    cv.addSubview_(sl)
    handler._status_lbl = sl

    # ── Buttons ───────────────────────────────────────────────────────────────
    cv.addSubview_(_btn("Close", handler, "closeWin:",
                        ((_PAD, _BTN_BOT), (80, _BTN_H))))
    dl_btn = _btn("⬇  Download", handler, "download:",
                  ((W - _PAD - 120, _BTN_BOT), (120, _BTN_H)))
    cv.addSubview_(dl_btn)
    handler._dl_btn = dl_btn

    return win, handler


if __name__ == "__main__":
    Switchman().run()
