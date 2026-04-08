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


def _lbl(text: str, frame, bold: bool = False, right: bool = True):
    """Non-editable label. When bold=True returns an NSView with label + separator line."""
    from AppKit import NSColor, NSBox, NSView
    if bold:
        (x, y), (w, h) = frame
        container = NSView.alloc().initWithFrame_(((x, y), (w, h)))
        f = NSTextField.alloc().initWithFrame_(((0, 2), (w, h)))
        f.setStringValue_(text)
        f.setBezeled_(False); f.setDrawsBackground_(False)
        f.setEditable_(False); f.setSelectable_(False)
        f.setAlignment_(1 if right else 0)
        f.setFont_(NSFont.boldSystemFontOfSize_(12))
        f.setTextColor_(NSColor.controlAccentColor())
        sep = NSBox.alloc().initWithFrame_(((0, 0), (w, 2)))
        sep.setBoxType_(2)  # NSBoxSeparator
        container.addSubview_(f)
        container.addSubview_(sep)
        return container
    f = NSTextField.alloc().initWithFrame_(frame)
    f.setStringValue_(text)
    f.setBezeled_(False); f.setDrawsBackground_(False)
    f.setEditable_(False); f.setSelectable_(False)
    f.setAlignment_(1 if right else 0)
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


def _sf_item(title: str, symbol: str, callback=None) -> "rumps.MenuItem":
    """Create a rumps MenuItem with an SF Symbol image."""
    from AppKit import NSImage
    item = rumps.MenuItem(title, callback=callback)
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, None)
    if img:
        item._menuitem.setImage_(img)
    return item


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


def _constrain_to_screen(window) -> None:
    """After center(), nudge the window so it's fully inside the visible screen area."""
    from AppKit import NSScreen
    vis = NSScreen.mainScreen().visibleFrame()
    fr = window.frame()
    ox, oy = fr.origin.x, fr.origin.y
    sw, sh = fr.size.width, fr.size.height
    # Clamp bottom edge
    if oy < vis.origin.y:
        oy = vis.origin.y
    # Clamp top edge (if window taller than screen, align to top of visible area)
    if oy + sh > vis.origin.y + vis.size.height:
        oy = max(vis.origin.y, vis.origin.y + vis.size.height - sh)
    window.setFrameOrigin_((ox, oy))


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


_PROMPT_HISTORY_PATH = Path.home() / ".config" / "switchman" / "prompt_history.json"
_COMPARE_HISTORY_PATH = Path.home() / ".config" / "switchman" / "compare_history.json"
_PROMPT_HISTORY_MAX  = 200


_MARKED_JS = '/**\n * marked v9.1.6 - a markdown parser\n * Copyright (c) 2011-2023, Christopher Jeffrey. (MIT Licensed)\n * https://github.com/markedjs/marked\n */\n!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?t(exports):"function"==typeof define&&define.amd?define(["exports"],t):t((e="undefined"!=typeof globalThis?globalThis:e||self).marked={})}(this,(function(e){"use strict";function t(){return{async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null}}function n(t){e.defaults=t}e.defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};const s=/[&<>"\']/,r=new RegExp(s.source,"g"),i=/[<>"\']|&(?!(#\\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\\w+);)/,l=new RegExp(i.source,"g"),o={"&":"&amp;","<":"&lt;",">":"&gt;",\'"\':"&quot;","\'":"&#39;"},a=e=>o[e];function c(e,t){if(t){if(s.test(e))return e.replace(r,a)}else if(i.test(e))return e.replace(l,a);return e}const h=/&(#(?:\\d+)|(?:#x[0-9A-Fa-f]+)|(?:\\w+));?/gi;const p=/(^|[^\\[])\\^/g;function u(e,t){e="string"==typeof e?e:e.source,t=t||"";const n={replace:(t,s)=>(s=(s="object"==typeof s&&"source"in s?s.source:s).replace(p,"$1"),e=e.replace(t,s),n),getRegex:()=>new RegExp(e,t)};return n}function g(e){try{e=encodeURI(e).replace(/%25/g,"%")}catch(e){return null}return e}const k={exec:()=>null};function f(e,t){const n=e.replace(/\\|/g,((e,t,n)=>{let s=!1,r=t;for(;--r>=0&&"\\\\"===n[r];)s=!s;return s?"|":" |"})).split(/ \\|/);let s=0;if(n[0].trim()||n.shift(),n.length>0&&!n[n.length-1].trim()&&n.pop(),t)if(n.length>t)n.splice(t);else for(;n.length<t;)n.push("");for(;s<n.length;s++)n[s]=n[s].trim().replace(/\\\\\\|/g,"|");return n}function d(e,t,n){const s=e.length;if(0===s)return"";let r=0;for(;r<s;){const i=e.charAt(s-r-1);if(i!==t||n){if(i===t||!n)break;r++}else r++}return e.slice(0,s-r)}function x(e,t,n,s){const r=t.href,i=t.title?c(t.title):null,l=e[1].replace(/\\\\([\\[\\]])/g,"$1");if("!"!==e[0].charAt(0)){s.state.inLink=!0;const e={type:"link",raw:n,href:r,title:i,text:l,tokens:s.inlineTokens(l)};return s.state.inLink=!1,e}return{type:"image",raw:n,href:r,title:i,text:c(l)}}class b{options;rules;lexer;constructor(t){this.options=t||e.defaults}space(e){const t=this.rules.block.newline.exec(e);if(t&&t[0].length>0)return{type:"space",raw:t[0]}}code(e){const t=this.rules.block.code.exec(e);if(t){const e=t[0].replace(/^ {1,4}/gm,"");return{type:"code",raw:t[0],codeBlockStyle:"indented",text:this.options.pedantic?e:d(e,"\\n")}}}fences(e){const t=this.rules.block.fences.exec(e);if(t){const e=t[0],n=function(e,t){const n=e.match(/^(\\s+)(?:```)/);if(null===n)return t;const s=n[1];return t.split("\\n").map((e=>{const t=e.match(/^\\s+/);if(null===t)return e;const[n]=t;return n.length>=s.length?e.slice(s.length):e})).join("\\n")}(e,t[3]||"");return{type:"code",raw:e,lang:t[2]?t[2].trim().replace(this.rules.inline._escapes,"$1"):t[2],text:n}}}heading(e){const t=this.rules.block.heading.exec(e);if(t){let e=t[2].trim();if(/#$/.test(e)){const t=d(e,"#");this.options.pedantic?e=t.trim():t&&!/ $/.test(t)||(e=t.trim())}return{type:"heading",raw:t[0],depth:t[1].length,text:e,tokens:this.lexer.inline(e)}}}hr(e){const t=this.rules.block.hr.exec(e);if(t)return{type:"hr",raw:t[0]}}blockquote(e){const t=this.rules.block.blockquote.exec(e);if(t){const e=d(t[0].replace(/^ *>[ \\t]?/gm,""),"\\n"),n=this.lexer.state.top;this.lexer.state.top=!0;const s=this.lexer.blockTokens(e);return this.lexer.state.top=n,{type:"blockquote",raw:t[0],tokens:s,text:e}}}list(e){let t=this.rules.block.list.exec(e);if(t){let n=t[1].trim();const s=n.length>1,r={type:"list",raw:"",ordered:s,start:s?+n.slice(0,-1):"",loose:!1,items:[]};n=s?`\\\\d{1,9}\\\\${n.slice(-1)}`:`\\\\${n}`,this.options.pedantic&&(n=s?n:"[*+-]");const i=new RegExp(`^( {0,3}${n})((?:[\\t ][^\\\\n]*)?(?:\\\\n|$))`);let l="",o="",a=!1;for(;e;){let n=!1;if(!(t=i.exec(e)))break;if(this.rules.block.hr.test(e))break;l=t[0],e=e.substring(l.length);let s=t[2].split("\\n",1)[0].replace(/^\\t+/,(e=>" ".repeat(3*e.length))),c=e.split("\\n",1)[0],h=0;this.options.pedantic?(h=2,o=s.trimStart()):(h=t[2].search(/[^ ]/),h=h>4?1:h,o=s.slice(h),h+=t[1].length);let p=!1;if(!s&&/^ *$/.test(c)&&(l+=c+"\\n",e=e.substring(c.length+1),n=!0),!n){const t=new RegExp(`^ {0,${Math.min(3,h-1)}}(?:[*+-]|\\\\d{1,9}[.)])((?:[ \\t][^\\\\n]*)?(?:\\\\n|$))`),n=new RegExp(`^ {0,${Math.min(3,h-1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\\\* *){3,})(?:\\\\n+|$)`),r=new RegExp(`^ {0,${Math.min(3,h-1)}}(?:\\`\\`\\`|~~~)`),i=new RegExp(`^ {0,${Math.min(3,h-1)}}#`);for(;e;){const a=e.split("\\n",1)[0];if(c=a,this.options.pedantic&&(c=c.replace(/^ {1,4}(?=( {4})*[^ ])/g,"  ")),r.test(c))break;if(i.test(c))break;if(t.test(c))break;if(n.test(e))break;if(c.search(/[^ ]/)>=h||!c.trim())o+="\\n"+c.slice(h);else{if(p)break;if(s.search(/[^ ]/)>=4)break;if(r.test(s))break;if(i.test(s))break;if(n.test(s))break;o+="\\n"+c}p||c.trim()||(p=!0),l+=a+"\\n",e=e.substring(a.length+1),s=c.slice(h)}}r.loose||(a?r.loose=!0:/\\n *\\n *$/.test(l)&&(a=!0));let u,g=null;this.options.gfm&&(g=/^\\[[ xX]\\] /.exec(o),g&&(u="[ ] "!==g[0],o=o.replace(/^\\[[ xX]\\] +/,""))),r.items.push({type:"list_item",raw:l,task:!!g,checked:u,loose:!1,text:o,tokens:[]}),r.raw+=l}r.items[r.items.length-1].raw=l.trimEnd(),r.items[r.items.length-1].text=o.trimEnd(),r.raw=r.raw.trimEnd();for(let e=0;e<r.items.length;e++)if(this.lexer.state.top=!1,r.items[e].tokens=this.lexer.blockTokens(r.items[e].text,[]),!r.loose){const t=r.items[e].tokens.filter((e=>"space"===e.type)),n=t.length>0&&t.some((e=>/\\n.*\\n/.test(e.raw)));r.loose=n}if(r.loose)for(let e=0;e<r.items.length;e++)r.items[e].loose=!0;return r}}html(e){const t=this.rules.block.html.exec(e);if(t){return{type:"html",block:!0,raw:t[0],pre:"pre"===t[1]||"script"===t[1]||"style"===t[1],text:t[0]}}}def(e){const t=this.rules.block.def.exec(e);if(t){const e=t[1].toLowerCase().replace(/\\s+/g," "),n=t[2]?t[2].replace(/^<(.*)>$/,"$1").replace(this.rules.inline._escapes,"$1"):"",s=t[3]?t[3].substring(1,t[3].length-1).replace(this.rules.inline._escapes,"$1"):t[3];return{type:"def",tag:e,raw:t[0],href:n,title:s}}}table(e){const t=this.rules.block.table.exec(e);if(t){if(!/[:|]/.test(t[2]))return;const e={type:"table",raw:t[0],header:f(t[1]).map((e=>({text:e,tokens:[]}))),align:t[2].replace(/^\\||\\| *$/g,"").split("|"),rows:t[3]&&t[3].trim()?t[3].replace(/\\n[ \\t]*$/,"").split("\\n"):[]};if(e.header.length===e.align.length){let t,n,s,r,i=e.align.length;for(t=0;t<i;t++){const n=e.align[t];n&&(/^ *-+: *$/.test(n)?e.align[t]="right":/^ *:-+: *$/.test(n)?e.align[t]="center":/^ *:-+ *$/.test(n)?e.align[t]="left":e.align[t]=null)}for(i=e.rows.length,t=0;t<i;t++)e.rows[t]=f(e.rows[t],e.header.length).map((e=>({text:e,tokens:[]})));for(i=e.header.length,n=0;n<i;n++)e.header[n].tokens=this.lexer.inline(e.header[n].text);for(i=e.rows.length,n=0;n<i;n++)for(r=e.rows[n],s=0;s<r.length;s++)r[s].tokens=this.lexer.inline(r[s].text);return e}}}lheading(e){const t=this.rules.block.lheading.exec(e);if(t)return{type:"heading",raw:t[0],depth:"="===t[2].charAt(0)?1:2,text:t[1],tokens:this.lexer.inline(t[1])}}paragraph(e){const t=this.rules.block.paragraph.exec(e);if(t){const e="\\n"===t[1].charAt(t[1].length-1)?t[1].slice(0,-1):t[1];return{type:"paragraph",raw:t[0],text:e,tokens:this.lexer.inline(e)}}}text(e){const t=this.rules.block.text.exec(e);if(t)return{type:"text",raw:t[0],text:t[0],tokens:this.lexer.inline(t[0])}}escape(e){const t=this.rules.inline.escape.exec(e);if(t)return{type:"escape",raw:t[0],text:c(t[1])}}tag(e){const t=this.rules.inline.tag.exec(e);if(t)return!this.lexer.state.inLink&&/^<a /i.test(t[0])?this.lexer.state.inLink=!0:this.lexer.state.inLink&&/^<\\/a>/i.test(t[0])&&(this.lexer.state.inLink=!1),!this.lexer.state.inRawBlock&&/^<(pre|code|kbd|script)(\\s|>)/i.test(t[0])?this.lexer.state.inRawBlock=!0:this.lexer.state.inRawBlock&&/^<\\/(pre|code|kbd|script)(\\s|>)/i.test(t[0])&&(this.lexer.state.inRawBlock=!1),{type:"html",raw:t[0],inLink:this.lexer.state.inLink,inRawBlock:this.lexer.state.inRawBlock,block:!1,text:t[0]}}link(e){const t=this.rules.inline.link.exec(e);if(t){const e=t[2].trim();if(!this.options.pedantic&&/^</.test(e)){if(!/>$/.test(e))return;const t=d(e.slice(0,-1),"\\\\");if((e.length-t.length)%2==0)return}else{const e=function(e,t){if(-1===e.indexOf(t[1]))return-1;let n=0;for(let s=0;s<e.length;s++)if("\\\\"===e[s])s++;else if(e[s]===t[0])n++;else if(e[s]===t[1]&&(n--,n<0))return s;return-1}(t[2],"()");if(e>-1){const n=(0===t[0].indexOf("!")?5:4)+t[1].length+e;t[2]=t[2].substring(0,e),t[0]=t[0].substring(0,n).trim(),t[3]=""}}let n=t[2],s="";if(this.options.pedantic){const e=/^([^\'"]*[^\\s])\\s+([\'"])(.*)\\2/.exec(n);e&&(n=e[1],s=e[3])}else s=t[3]?t[3].slice(1,-1):"";return n=n.trim(),/^</.test(n)&&(n=this.options.pedantic&&!/>$/.test(e)?n.slice(1):n.slice(1,-1)),x(t,{href:n?n.replace(this.rules.inline._escapes,"$1"):n,title:s?s.replace(this.rules.inline._escapes,"$1"):s},t[0],this.lexer)}}reflink(e,t){let n;if((n=this.rules.inline.reflink.exec(e))||(n=this.rules.inline.nolink.exec(e))){let e=(n[2]||n[1]).replace(/\\s+/g," ");if(e=t[e.toLowerCase()],!e){const e=n[0].charAt(0);return{type:"text",raw:e,text:e}}return x(n,e,n[0],this.lexer)}}emStrong(e,t,n=""){let s=this.rules.inline.emStrong.lDelim.exec(e);if(!s)return;if(s[3]&&n.match(/[\\p{L}\\p{N}]/u))return;if(!(s[1]||s[2]||"")||!n||this.rules.inline.punctuation.exec(n)){const n=[...s[0]].length-1;let r,i,l=n,o=0;const a="*"===s[0][0]?this.rules.inline.emStrong.rDelimAst:this.rules.inline.emStrong.rDelimUnd;for(a.lastIndex=0,t=t.slice(-1*e.length+n);null!=(s=a.exec(t));){if(r=s[1]||s[2]||s[3]||s[4]||s[5]||s[6],!r)continue;if(i=[...r].length,s[3]||s[4]){l+=i;continue}if((s[5]||s[6])&&n%3&&!((n+i)%3)){o+=i;continue}if(l-=i,l>0)continue;i=Math.min(i,i+l+o);const t=[...s[0]][0].length,a=e.slice(0,n+s.index+t+i);if(Math.min(n,i)%2){const e=a.slice(1,-1);return{type:"em",raw:a,text:e,tokens:this.lexer.inlineTokens(e)}}const c=a.slice(2,-2);return{type:"strong",raw:a,text:c,tokens:this.lexer.inlineTokens(c)}}}}codespan(e){const t=this.rules.inline.code.exec(e);if(t){let e=t[2].replace(/\\n/g," ");const n=/[^ ]/.test(e),s=/^ /.test(e)&&/ $/.test(e);return n&&s&&(e=e.substring(1,e.length-1)),e=c(e,!0),{type:"codespan",raw:t[0],text:e}}}br(e){const t=this.rules.inline.br.exec(e);if(t)return{type:"br",raw:t[0]}}del(e){const t=this.rules.inline.del.exec(e);if(t)return{type:"del",raw:t[0],text:t[2],tokens:this.lexer.inlineTokens(t[2])}}autolink(e){const t=this.rules.inline.autolink.exec(e);if(t){let e,n;return"@"===t[2]?(e=c(t[1]),n="mailto:"+e):(e=c(t[1]),n=e),{type:"link",raw:t[0],text:e,href:n,tokens:[{type:"text",raw:e,text:e}]}}}url(e){let t;if(t=this.rules.inline.url.exec(e)){let e,n;if("@"===t[2])e=c(t[0]),n="mailto:"+e;else{let s;do{s=t[0],t[0]=this.rules.inline._backpedal.exec(t[0])[0]}while(s!==t[0]);e=c(t[0]),n="www."===t[1]?"http://"+t[0]:t[0]}return{type:"link",raw:t[0],text:e,href:n,tokens:[{type:"text",raw:e,text:e}]}}}inlineText(e){const t=this.rules.inline.text.exec(e);if(t){let e;return e=this.lexer.state.inRawBlock?t[0]:c(t[0]),{type:"text",raw:t[0],text:e}}}}const m={newline:/^(?: *(?:\\n|$))+/,code:/^( {4}[^\\n]+(?:\\n(?: *(?:\\n|$))*)?)+/,fences:/^ {0,3}(`{3,}(?=[^`\\n]*(?:\\n|$))|~{3,})([^\\n]*)(?:\\n|$)(?:|([\\s\\S]*?)(?:\\n|$))(?: {0,3}\\1[~`]* *(?=\\n|$)|$)/,hr:/^ {0,3}((?:-[\\t ]*){3,}|(?:_[ \\t]*){3,}|(?:\\*[ \\t]*){3,})(?:\\n+|$)/,heading:/^ {0,3}(#{1,6})(?=\\s|$)(.*)(?:\\n+|$)/,blockquote:/^( {0,3}> ?(paragraph|[^\\n]*)(?:\\n|$))+/,list:/^( {0,3}bull)([ \\t][^\\n]+?)?(?:\\n|$)/,html:"^ {0,3}(?:<(script|pre|style|textarea)[\\\\s>][\\\\s\\\\S]*?(?:</\\\\1>[^\\\\n]*\\\\n+|$)|comment[^\\\\n]*(\\\\n+|$)|<\\\\?[\\\\s\\\\S]*?(?:\\\\?>\\\\n*|$)|<![A-Z][\\\\s\\\\S]*?(?:>\\\\n*|$)|<!\\\\[CDATA\\\\[[\\\\s\\\\S]*?(?:\\\\]\\\\]>\\\\n*|$)|</?(tag)(?: +|\\\\n|/?>)[\\\\s\\\\S]*?(?:(?:\\\\n *)+\\\\n|$)|<(?!script|pre|style|textarea)([a-z][\\\\w-]*)(?:attribute)*? */?>(?=[ \\\\t]*(?:\\\\n|$))[\\\\s\\\\S]*?(?:(?:\\\\n *)+\\\\n|$)|</(?!script|pre|style|textarea)[a-z][\\\\w-]*\\\\s*>(?=[ \\\\t]*(?:\\\\n|$))[\\\\s\\\\S]*?(?:(?:\\\\n *)+\\\\n|$))",def:/^ {0,3}\\[(label)\\]: *(?:\\n *)?([^<\\s][^\\s]*|<.*?>)(?:(?: +(?:\\n *)?| *\\n *)(title))? *(?:\\n+|$)/,table:k,lheading:/^(?!bull )((?:.|\\n(?!\\s*?\\n|bull ))+?)\\n {0,3}(=+|-+) *(?:\\n+|$)/,_paragraph:/^([^\\n]+(?:\\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\\n)[^\\n]+)*)/,text:/^[^\\n]+/,_label:/(?!\\s*\\])(?:\\\\.|[^\\[\\]\\\\])+/,_title:/(?:"(?:\\\\"?|[^"\\\\])*"|\'[^\'\\n]*(?:\\n[^\'\\n]+)*\\n?\'|\\([^()]*\\))/};m.def=u(m.def).replace("label",m._label).replace("title",m._title).getRegex(),m.bullet=/(?:[*+-]|\\d{1,9}[.)])/,m.listItemStart=u(/^( *)(bull) */).replace("bull",m.bullet).getRegex(),m.list=u(m.list).replace(/bull/g,m.bullet).replace("hr","\\\\n+(?=\\\\1?(?:(?:- *){3,}|(?:_ *){3,}|(?:\\\\* *){3,})(?:\\\\n+|$))").replace("def","\\\\n+(?="+m.def.source+")").getRegex(),m._tag="address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul",m._comment=/<!--(?!-?>)[\\s\\S]*?(?:-->|$)/,m.html=u(m.html,"i").replace("comment",m._comment).replace("tag",m._tag).replace("attribute",/ +[a-zA-Z:_][\\w.:-]*(?: *= *"[^"\\n]*"| *= *\'[^\'\\n]*\'| *= *[^\\s"\'=<>`]+)?/).getRegex(),m.lheading=u(m.lheading).replace(/bull/g,m.bullet).getRegex(),m.paragraph=u(m._paragraph).replace("hr",m.hr).replace("heading"," {0,3}#{1,6}(?:\\\\s|$)").replace("|lheading","").replace("|table","").replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\\\n]*\\\\n)|~{3,})[^\\\\n]*\\\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",m._tag).getRegex(),m.blockquote=u(m.blockquote).replace("paragraph",m.paragraph).getRegex(),m.normal={...m},m.gfm={...m.normal,table:"^ *([^\\\\n ].*)\\\\n {0,3}((?:\\\\| *)?:?-+:? *(?:\\\\| *:?-+:? *)*(?:\\\\| *)?)(?:\\\\n((?:(?! *\\\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\\\n|$))*)\\\\n*|$)"},m.gfm.table=u(m.gfm.table).replace("hr",m.hr).replace("heading"," {0,3}#{1,6}(?:\\\\s|$)").replace("blockquote"," {0,3}>").replace("code"," {4}[^\\\\n]").replace("fences"," {0,3}(?:`{3,}(?=[^`\\\\n]*\\\\n)|~{3,})[^\\\\n]*\\\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",m._tag).getRegex(),m.gfm.paragraph=u(m._paragraph).replace("hr",m.hr).replace("heading"," {0,3}#{1,6}(?:\\\\s|$)").replace("|lheading","").replace("table",m.gfm.table).replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\\\n]*\\\\n)|~{3,})[^\\\\n]*\\\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",m._tag).getRegex(),m.pedantic={...m.normal,html:u("^ *(?:comment *(?:\\\\n|\\\\s*$)|<(tag)[\\\\s\\\\S]+?</\\\\1> *(?:\\\\n{2,}|\\\\s*$)|<tag(?:\\"[^\\"]*\\"|\'[^\']*\'|\\\\s[^\'\\"/>\\\\s]*)*?/?> *(?:\\\\n{2,}|\\\\s*$))").replace("comment",m._comment).replace(/tag/g,"(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\\\b)\\\\w+(?!:|[^\\\\w\\\\s@]*@)\\\\b").getRegex(),def:/^ *\\[([^\\]]+)\\]: *<?([^\\s>]+)>?(?: +(["(][^\\n]+[")]))? *(?:\\n+|$)/,heading:/^(#{1,6})(.*)(?:\\n+|$)/,fences:k,lheading:/^(.+?)\\n {0,3}(=+|-+) *(?:\\n+|$)/,paragraph:u(m.normal._paragraph).replace("hr",m.hr).replace("heading"," *#{1,6} *[^\\n]").replace("lheading",m.lheading).replace("blockquote"," {0,3}>").replace("|fences","").replace("|list","").replace("|html","").getRegex()};const w={escape:/^\\\\([!"#$%&\'()*+,\\-./:;<=>?@\\[\\]\\\\^_`{|}~])/,autolink:/^<(scheme:[^\\s\\x00-\\x1f<>]*|email)>/,url:k,tag:"^comment|^</[a-zA-Z][\\\\w:-]*\\\\s*>|^<[a-zA-Z][\\\\w-]*(?:attribute)*?\\\\s*/?>|^<\\\\?[\\\\s\\\\S]*?\\\\?>|^<![a-zA-Z]+\\\\s[\\\\s\\\\S]*?>|^<!\\\\[CDATA\\\\[[\\\\s\\\\S]*?\\\\]\\\\]>",link:/^!?\\[(label)\\]\\(\\s*(href)(?:\\s+(title))?\\s*\\)/,reflink:/^!?\\[(label)\\]\\[(ref)\\]/,nolink:/^!?\\[(ref)\\](?:\\[\\])?/,reflinkSearch:"reflink|nolink(?!\\\\()",emStrong:{lDelim:/^(?:\\*+(?:((?!\\*)[punct])|[^\\s*]))|^_+(?:((?!_)[punct])|([^\\s_]))/,rDelimAst:/^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)[punct](\\*+)(?=[\\s]|$)|[^punct\\s](\\*+)(?!\\*)(?=[punct\\s]|$)|(?!\\*)[punct\\s](\\*+)(?=[^punct\\s])|[\\s](\\*+)(?!\\*)(?=[punct])|(?!\\*)[punct](\\*+)(?!\\*)(?=[punct])|[^punct\\s](\\*+)(?=[^punct\\s])/,rDelimUnd:/^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)[punct](_+)(?=[\\s]|$)|[^punct\\s](_+)(?!_)(?=[punct\\s]|$)|(?!_)[punct\\s](_+)(?=[^punct\\s])|[\\s](_+)(?!_)(?=[punct])|(?!_)[punct](_+)(?!_)(?=[punct])/},code:/^(`+)([^`]|[^`][\\s\\S]*?[^`])\\1(?!`)/,br:/^( {2,}|\\\\)\\n(?!\\s*$)/,del:k,text:/^(`+|[^`])(?:(?= {2,}\\n)|[\\s\\S]*?(?:(?=[\\\\<!\\[`*_]|\\b_|$)|[^ ](?= {2,}\\n)))/,punctuation:/^((?![*_])[\\spunctuation])/,_punctuation:"\\\\p{P}$+<=>`^|~"};w.punctuation=u(w.punctuation,"u").replace(/punctuation/g,w._punctuation).getRegex(),w.blockSkip=/\\[[^[\\]]*?\\]\\([^\\(\\)]*?\\)|`[^`]*?`|<[^<>]*?>/g,w.anyPunctuation=/\\\\[punct]/g,w._escapes=/\\\\([punct])/g,w._comment=u(m._comment).replace("(?:--\\x3e|$)","--\\x3e").getRegex(),w.emStrong.lDelim=u(w.emStrong.lDelim,"u").replace(/punct/g,w._punctuation).getRegex(),w.emStrong.rDelimAst=u(w.emStrong.rDelimAst,"gu").replace(/punct/g,w._punctuation).getRegex(),w.emStrong.rDelimUnd=u(w.emStrong.rDelimUnd,"gu").replace(/punct/g,w._punctuation).getRegex(),w.anyPunctuation=u(w.anyPunctuation,"gu").replace(/punct/g,w._punctuation).getRegex(),w._escapes=u(w._escapes,"gu").replace(/punct/g,w._punctuation).getRegex(),w._scheme=/[a-zA-Z][a-zA-Z0-9+.-]{1,31}/,w._email=/[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/,w.autolink=u(w.autolink).replace("scheme",w._scheme).replace("email",w._email).getRegex(),w._attribute=/\\s+[a-zA-Z:_][\\w.:-]*(?:\\s*=\\s*"[^"]*"|\\s*=\\s*\'[^\']*\'|\\s*=\\s*[^\\s"\'=<>`]+)?/,w.tag=u(w.tag).replace("comment",w._comment).replace("attribute",w._attribute).getRegex(),w._label=/(?:\\[(?:\\\\.|[^\\[\\]\\\\])*\\]|\\\\.|`[^`]*`|[^\\[\\]\\\\`])*?/,w._href=/<(?:\\\\.|[^\\n<>\\\\])+>|[^\\s\\x00-\\x1f]*/,w._title=/"(?:\\\\"?|[^"\\\\])*"|\'(?:\\\\\'?|[^\'\\\\])*\'|\\((?:\\\\\\)?|[^)\\\\])*\\)/,w.link=u(w.link).replace("label",w._label).replace("href",w._href).replace("title",w._title).getRegex(),w.reflink=u(w.reflink).replace("label",w._label).replace("ref",m._label).getRegex(),w.nolink=u(w.nolink).replace("ref",m._label).getRegex(),w.reflinkSearch=u(w.reflinkSearch,"g").replace("reflink",w.reflink).replace("nolink",w.nolink).getRegex(),w.normal={...w},w.pedantic={...w.normal,strong:{start:/^__|\\*\\*/,middle:/^__(?=\\S)([\\s\\S]*?\\S)__(?!_)|^\\*\\*(?=\\S)([\\s\\S]*?\\S)\\*\\*(?!\\*)/,endAst:/\\*\\*(?!\\*)/g,endUnd:/__(?!_)/g},em:{start:/^_|\\*/,middle:/^()\\*(?=\\S)([\\s\\S]*?\\S)\\*(?!\\*)|^_(?=\\S)([\\s\\S]*?\\S)_(?!_)/,endAst:/\\*(?!\\*)/g,endUnd:/_(?!_)/g},link:u(/^!?\\[(label)\\]\\((.*?)\\)/).replace("label",w._label).getRegex(),reflink:u(/^!?\\[(label)\\]\\s*\\[([^\\]]*)\\]/).replace("label",w._label).getRegex()},w.gfm={...w.normal,escape:u(w.escape).replace("])","~|])").getRegex(),_extended_email:/[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/,url:/^((?:ftp|https?):\\/\\/|www\\.)(?:[a-zA-Z0-9\\-]+\\.?)+[^\\s<]*|^email/,_backpedal:/(?:[^?!.,:;*_\'"~()&]+|\\([^)]*\\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_\'"~)]+(?!$))+/,del:/^(~~?)(?=[^\\s~])([\\s\\S]*?[^\\s~])\\1(?=[^~]|$)/,text:/^([`~]+|[^`~])(?:(?= {2,}\\n)|(?=[a-zA-Z0-9.!#$%&\'*+\\/=?_`{\\|}~-]+@)|[\\s\\S]*?(?:(?=[\\\\<!\\[`*~_]|\\b_|https?:\\/\\/|ftp:\\/\\/|www\\.|$)|[^ ](?= {2,}\\n)|[^a-zA-Z0-9.!#$%&\'*+\\/=?_`{\\|}~-](?=[a-zA-Z0-9.!#$%&\'*+\\/=?_`{\\|}~-]+@)))/},w.gfm.url=u(w.gfm.url,"i").replace("email",w.gfm._extended_email).getRegex(),w.breaks={...w.gfm,br:u(w.br).replace("{2,}","*").getRegex(),text:u(w.gfm.text).replace("\\\\b_","\\\\b_| {2,}\\\\n").replace(/\\{2,\\}/g,"*").getRegex()};class _{tokens;options;state;tokenizer;inlineQueue;constructor(t){this.tokens=[],this.tokens.links=Object.create(null),this.options=t||e.defaults,this.options.tokenizer=this.options.tokenizer||new b,this.tokenizer=this.options.tokenizer,this.tokenizer.options=this.options,this.tokenizer.lexer=this,this.inlineQueue=[],this.state={inLink:!1,inRawBlock:!1,top:!0};const n={block:m.normal,inline:w.normal};this.options.pedantic?(n.block=m.pedantic,n.inline=w.pedantic):this.options.gfm&&(n.block=m.gfm,this.options.breaks?n.inline=w.breaks:n.inline=w.gfm),this.tokenizer.rules=n}static get rules(){return{block:m,inline:w}}static lex(e,t){return new _(t).lex(e)}static lexInline(e,t){return new _(t).inlineTokens(e)}lex(e){let t;for(e=e.replace(/\\r\\n|\\r/g,"\\n"),this.blockTokens(e,this.tokens);t=this.inlineQueue.shift();)this.inlineTokens(t.src,t.tokens);return this.tokens}blockTokens(e,t=[]){let n,s,r,i;for(e=this.options.pedantic?e.replace(/\\t/g,"    ").replace(/^ +$/gm,""):e.replace(/^( *)(\\t+)/gm,((e,t,n)=>t+"    ".repeat(n.length)));e;)if(!(this.options.extensions&&this.options.extensions.block&&this.options.extensions.block.some((s=>!!(n=s.call({lexer:this},e,t))&&(e=e.substring(n.raw.length),t.push(n),!0)))))if(n=this.tokenizer.space(e))e=e.substring(n.raw.length),1===n.raw.length&&t.length>0?t[t.length-1].raw+="\\n":t.push(n);else if(n=this.tokenizer.code(e))e=e.substring(n.raw.length),s=t[t.length-1],!s||"paragraph"!==s.type&&"text"!==s.type?t.push(n):(s.raw+="\\n"+n.raw,s.text+="\\n"+n.text,this.inlineQueue[this.inlineQueue.length-1].src=s.text);else if(n=this.tokenizer.fences(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.heading(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.hr(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.blockquote(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.list(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.html(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.def(e))e=e.substring(n.raw.length),s=t[t.length-1],!s||"paragraph"!==s.type&&"text"!==s.type?this.tokens.links[n.tag]||(this.tokens.links[n.tag]={href:n.href,title:n.title}):(s.raw+="\\n"+n.raw,s.text+="\\n"+n.raw,this.inlineQueue[this.inlineQueue.length-1].src=s.text);else if(n=this.tokenizer.table(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.lheading(e))e=e.substring(n.raw.length),t.push(n);else{if(r=e,this.options.extensions&&this.options.extensions.startBlock){let t=1/0;const n=e.slice(1);let s;this.options.extensions.startBlock.forEach((e=>{s=e.call({lexer:this},n),"number"==typeof s&&s>=0&&(t=Math.min(t,s))})),t<1/0&&t>=0&&(r=e.substring(0,t+1))}if(this.state.top&&(n=this.tokenizer.paragraph(r)))s=t[t.length-1],i&&"paragraph"===s.type?(s.raw+="\\n"+n.raw,s.text+="\\n"+n.text,this.inlineQueue.pop(),this.inlineQueue[this.inlineQueue.length-1].src=s.text):t.push(n),i=r.length!==e.length,e=e.substring(n.raw.length);else if(n=this.tokenizer.text(e))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===s.type?(s.raw+="\\n"+n.raw,s.text+="\\n"+n.text,this.inlineQueue.pop(),this.inlineQueue[this.inlineQueue.length-1].src=s.text):t.push(n);else if(e){const t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}return this.state.top=!0,t}inline(e,t=[]){return this.inlineQueue.push({src:e,tokens:t}),t}inlineTokens(e,t=[]){let n,s,r,i,l,o,a=e;if(this.tokens.links){const e=Object.keys(this.tokens.links);if(e.length>0)for(;null!=(i=this.tokenizer.rules.inline.reflinkSearch.exec(a));)e.includes(i[0].slice(i[0].lastIndexOf("[")+1,-1))&&(a=a.slice(0,i.index)+"["+"a".repeat(i[0].length-2)+"]"+a.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex))}for(;null!=(i=this.tokenizer.rules.inline.blockSkip.exec(a));)a=a.slice(0,i.index)+"["+"a".repeat(i[0].length-2)+"]"+a.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);for(;null!=(i=this.tokenizer.rules.inline.anyPunctuation.exec(a));)a=a.slice(0,i.index)+"++"+a.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);for(;e;)if(l||(o=""),l=!1,!(this.options.extensions&&this.options.extensions.inline&&this.options.extensions.inline.some((s=>!!(n=s.call({lexer:this},e,t))&&(e=e.substring(n.raw.length),t.push(n),!0)))))if(n=this.tokenizer.escape(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.tag(e))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===n.type&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(n=this.tokenizer.link(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.reflink(e,this.tokens.links))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===n.type&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(n=this.tokenizer.emStrong(e,a,o))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.codespan(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.br(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.del(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.autolink(e))e=e.substring(n.raw.length),t.push(n);else if(this.state.inLink||!(n=this.tokenizer.url(e))){if(r=e,this.options.extensions&&this.options.extensions.startInline){let t=1/0;const n=e.slice(1);let s;this.options.extensions.startInline.forEach((e=>{s=e.call({lexer:this},n),"number"==typeof s&&s>=0&&(t=Math.min(t,s))})),t<1/0&&t>=0&&(r=e.substring(0,t+1))}if(n=this.tokenizer.inlineText(r))e=e.substring(n.raw.length),"_"!==n.raw.slice(-1)&&(o=n.raw.slice(-1)),l=!0,s=t[t.length-1],s&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(e){const t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}else e=e.substring(n.raw.length),t.push(n);return t}}class y{options;constructor(t){this.options=t||e.defaults}code(e,t,n){const s=(t||"").match(/^\\S*/)?.[0];return e=e.replace(/\\n$/,"")+"\\n",s?\'<pre><code class="language-\'+c(s)+\'">\'+(n?e:c(e,!0))+"</code></pre>\\n":"<pre><code>"+(n?e:c(e,!0))+"</code></pre>\\n"}blockquote(e){return`<blockquote>\\n${e}</blockquote>\\n`}html(e,t){return e}heading(e,t,n){return`<h${t}>${e}</h${t}>\\n`}hr(){return"<hr>\\n"}list(e,t,n){const s=t?"ol":"ul";return"<"+s+(t&&1!==n?\' start="\'+n+\'"\':"")+">\\n"+e+"</"+s+">\\n"}listitem(e,t,n){return`<li>${e}</li>\\n`}checkbox(e){return"<input "+(e?\'checked="" \':"")+\'disabled="" type="checkbox">\'}paragraph(e){return`<p>${e}</p>\\n`}table(e,t){return t&&(t=`<tbody>${t}</tbody>`),"<table>\\n<thead>\\n"+e+"</thead>\\n"+t+"</table>\\n"}tablerow(e){return`<tr>\\n${e}</tr>\\n`}tablecell(e,t){const n=t.header?"th":"td";return(t.align?`<${n} align="${t.align}">`:`<${n}>`)+e+`</${n}>\\n`}strong(e){return`<strong>${e}</strong>`}em(e){return`<em>${e}</em>`}codespan(e){return`<code>${e}</code>`}br(){return"<br>"}del(e){return`<del>${e}</del>`}link(e,t,n){const s=g(e);if(null===s)return n;let r=\'<a href="\'+(e=s)+\'"\';return t&&(r+=\' title="\'+t+\'"\'),r+=">"+n+"</a>",r}image(e,t,n){const s=g(e);if(null===s)return n;let r=`<img src="${e=s}" alt="${n}"`;return t&&(r+=` title="${t}"`),r+=">",r}text(e){return e}}class ${strong(e){return e}em(e){return e}codespan(e){return e}del(e){return e}html(e){return e}text(e){return e}link(e,t,n){return""+n}image(e,t,n){return""+n}br(){return""}}class z{options;renderer;textRenderer;constructor(t){this.options=t||e.defaults,this.options.renderer=this.options.renderer||new y,this.renderer=this.options.renderer,this.renderer.options=this.options,this.textRenderer=new $}static parse(e,t){return new z(t).parse(e)}static parseInline(e,t){return new z(t).parseInline(e)}parse(e,t=!0){let n="";for(let s=0;s<e.length;s++){const r=e[s];if(this.options.extensions&&this.options.extensions.renderers&&this.options.extensions.renderers[r.type]){const e=r,t=this.options.extensions.renderers[e.type].call({parser:this},e);if(!1!==t||!["space","hr","heading","code","table","blockquote","list","html","paragraph","text"].includes(e.type)){n+=t||"";continue}}switch(r.type){case"space":continue;case"hr":n+=this.renderer.hr();continue;case"heading":{const e=r;n+=this.renderer.heading(this.parseInline(e.tokens),e.depth,this.parseInline(e.tokens,this.textRenderer).replace(h,((e,t)=>"colon"===(t=t.toLowerCase())?":":"#"===t.charAt(0)?"x"===t.charAt(1)?String.fromCharCode(parseInt(t.substring(2),16)):String.fromCharCode(+t.substring(1)):"")));continue}case"code":{const e=r;n+=this.renderer.code(e.text,e.lang,!!e.escaped);continue}case"table":{const e=r;let t="",s="";for(let t=0;t<e.header.length;t++)s+=this.renderer.tablecell(this.parseInline(e.header[t].tokens),{header:!0,align:e.align[t]});t+=this.renderer.tablerow(s);let i="";for(let t=0;t<e.rows.length;t++){const n=e.rows[t];s="";for(let t=0;t<n.length;t++)s+=this.renderer.tablecell(this.parseInline(n[t].tokens),{header:!1,align:e.align[t]});i+=this.renderer.tablerow(s)}n+=this.renderer.table(t,i);continue}case"blockquote":{const e=r,t=this.parse(e.tokens);n+=this.renderer.blockquote(t);continue}case"list":{const e=r,t=e.ordered,s=e.start,i=e.loose;let l="";for(let t=0;t<e.items.length;t++){const n=e.items[t],s=n.checked,r=n.task;let o="";if(n.task){const e=this.renderer.checkbox(!!s);i?n.tokens.length>0&&"paragraph"===n.tokens[0].type?(n.tokens[0].text=e+" "+n.tokens[0].text,n.tokens[0].tokens&&n.tokens[0].tokens.length>0&&"text"===n.tokens[0].tokens[0].type&&(n.tokens[0].tokens[0].text=e+" "+n.tokens[0].tokens[0].text)):n.tokens.unshift({type:"text",text:e+" "}):o+=e+" "}o+=this.parse(n.tokens,i),l+=this.renderer.listitem(o,r,!!s)}n+=this.renderer.list(l,t,s);continue}case"html":{const e=r;n+=this.renderer.html(e.text,e.block);continue}case"paragraph":{const e=r;n+=this.renderer.paragraph(this.parseInline(e.tokens));continue}case"text":{let i=r,l=i.tokens?this.parseInline(i.tokens):i.text;for(;s+1<e.length&&"text"===e[s+1].type;)i=e[++s],l+="\\n"+(i.tokens?this.parseInline(i.tokens):i.text);n+=t?this.renderer.paragraph(l):l;continue}default:{const e=\'Token with "\'+r.type+\'" type was not found.\';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return n}parseInline(e,t){t=t||this.renderer;let n="";for(let s=0;s<e.length;s++){const r=e[s];if(this.options.extensions&&this.options.extensions.renderers&&this.options.extensions.renderers[r.type]){const e=this.options.extensions.renderers[r.type].call({parser:this},r);if(!1!==e||!["escape","html","link","image","strong","em","codespan","br","del","text"].includes(r.type)){n+=e||"";continue}}switch(r.type){case"escape":{const e=r;n+=t.text(e.text);break}case"html":{const e=r;n+=t.html(e.text);break}case"link":{const e=r;n+=t.link(e.href,e.title,this.parseInline(e.tokens,t));break}case"image":{const e=r;n+=t.image(e.href,e.title,e.text);break}case"strong":{const e=r;n+=t.strong(this.parseInline(e.tokens,t));break}case"em":{const e=r;n+=t.em(this.parseInline(e.tokens,t));break}case"codespan":{const e=r;n+=t.codespan(e.text);break}case"br":n+=t.br();break;case"del":{const e=r;n+=t.del(this.parseInline(e.tokens,t));break}case"text":{const e=r;n+=t.text(e.text);break}default:{const e=\'Token with "\'+r.type+\'" type was not found.\';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return n}}class T{options;constructor(t){this.options=t||e.defaults}static passThroughHooks=new Set(["preprocess","postprocess"]);preprocess(e){return e}postprocess(e){return e}}class R{defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};options=this.setOptions;parse=this.#e(_.lex,z.parse);parseInline=this.#e(_.lexInline,z.parseInline);Parser=z;Renderer=y;TextRenderer=$;Lexer=_;Tokenizer=b;Hooks=T;constructor(...e){this.use(...e)}walkTokens(e,t){let n=[];for(const s of e)switch(n=n.concat(t.call(this,s)),s.type){case"table":{const e=s;for(const s of e.header)n=n.concat(this.walkTokens(s.tokens,t));for(const s of e.rows)for(const e of s)n=n.concat(this.walkTokens(e.tokens,t));break}case"list":{const e=s;n=n.concat(this.walkTokens(e.items,t));break}default:{const e=s;this.defaults.extensions?.childTokens?.[e.type]?this.defaults.extensions.childTokens[e.type].forEach((s=>{n=n.concat(this.walkTokens(e[s],t))})):e.tokens&&(n=n.concat(this.walkTokens(e.tokens,t)))}}return n}use(...e){const t=this.defaults.extensions||{renderers:{},childTokens:{}};return e.forEach((e=>{const n={...e};if(n.async=this.defaults.async||n.async||!1,e.extensions&&(e.extensions.forEach((e=>{if(!e.name)throw new Error("extension name required");if("renderer"in e){const n=t.renderers[e.name];t.renderers[e.name]=n?function(...t){let s=e.renderer.apply(this,t);return!1===s&&(s=n.apply(this,t)),s}:e.renderer}if("tokenizer"in e){if(!e.level||"block"!==e.level&&"inline"!==e.level)throw new Error("extension level must be \'block\' or \'inline\'");const n=t[e.level];n?n.unshift(e.tokenizer):t[e.level]=[e.tokenizer],e.start&&("block"===e.level?t.startBlock?t.startBlock.push(e.start):t.startBlock=[e.start]:"inline"===e.level&&(t.startInline?t.startInline.push(e.start):t.startInline=[e.start]))}"childTokens"in e&&e.childTokens&&(t.childTokens[e.name]=e.childTokens)})),n.extensions=t),e.renderer){const t=this.defaults.renderer||new y(this.defaults);for(const n in e.renderer){const s=e.renderer[n],r=n,i=t[r];t[r]=(...e)=>{let n=s.apply(t,e);return!1===n&&(n=i.apply(t,e)),n||""}}n.renderer=t}if(e.tokenizer){const t=this.defaults.tokenizer||new b(this.defaults);for(const n in e.tokenizer){const s=e.tokenizer[n],r=n,i=t[r];t[r]=(...e)=>{let n=s.apply(t,e);return!1===n&&(n=i.apply(t,e)),n}}n.tokenizer=t}if(e.hooks){const t=this.defaults.hooks||new T;for(const n in e.hooks){const s=e.hooks[n],r=n,i=t[r];T.passThroughHooks.has(n)?t[r]=e=>{if(this.defaults.async)return Promise.resolve(s.call(t,e)).then((e=>i.call(t,e)));const n=s.call(t,e);return i.call(t,n)}:t[r]=(...e)=>{let n=s.apply(t,e);return!1===n&&(n=i.apply(t,e)),n}}n.hooks=t}if(e.walkTokens){const t=this.defaults.walkTokens,s=e.walkTokens;n.walkTokens=function(e){let n=[];return n.push(s.call(this,e)),t&&(n=n.concat(t.call(this,e))),n}}this.defaults={...this.defaults,...n}})),this}setOptions(e){return this.defaults={...this.defaults,...e},this}lexer(e,t){return _.lex(e,t??this.defaults)}parser(e,t){return z.parse(e,t??this.defaults)}#e(e,t){return(n,s)=>{const r={...s},i={...this.defaults,...r};!0===this.defaults.async&&!1===r.async&&(i.silent||console.warn("marked(): The async option was set to true by an extension. The async: false option sent to parse will be ignored."),i.async=!0);const l=this.#t(!!i.silent,!!i.async);if(null==n)return l(new Error("marked(): input parameter is undefined or null"));if("string"!=typeof n)return l(new Error("marked(): input parameter is of type "+Object.prototype.toString.call(n)+", string expected"));if(i.hooks&&(i.hooks.options=i),i.async)return Promise.resolve(i.hooks?i.hooks.preprocess(n):n).then((t=>e(t,i))).then((e=>i.walkTokens?Promise.all(this.walkTokens(e,i.walkTokens)).then((()=>e)):e)).then((e=>t(e,i))).then((e=>i.hooks?i.hooks.postprocess(e):e)).catch(l);try{i.hooks&&(n=i.hooks.preprocess(n));const s=e(n,i);i.walkTokens&&this.walkTokens(s,i.walkTokens);let r=t(s,i);return i.hooks&&(r=i.hooks.postprocess(r)),r}catch(e){return l(e)}}}#t(e,t){return n=>{if(n.message+="\\nPlease report this to https://github.com/markedjs/marked.",e){const e="<p>An error occurred:</p><pre>"+c(n.message+"",!0)+"</pre>";return t?Promise.resolve(e):e}if(t)return Promise.reject(n);throw n}}}const S=new R;function A(e,t){return S.parse(e,t)}A.options=A.setOptions=function(e){return S.setOptions(e),A.defaults=S.defaults,n(A.defaults),A},A.getDefaults=t,A.defaults=e.defaults,A.use=function(...e){return S.use(...e),A.defaults=S.defaults,n(A.defaults),A},A.walkTokens=function(e,t){return S.walkTokens(e,t)},A.parseInline=S.parseInline,A.Parser=z,A.parser=z.parse,A.Renderer=y,A.TextRenderer=$,A.Lexer=_,A.lexer=_.lex,A.Tokenizer=b,A.Hooks=T,A.parse=A;const I=A.options,E=A.setOptions,Z=A.use,q=A.walkTokens,L=A.parseInline,D=A,P=z.parse,v=_.lex;e.Hooks=T,e.Lexer=_,e.Marked=R,e.Parser=z,e.Renderer=y,e.TextRenderer=$,e.Tokenizer=b,e.getDefaults=t,e.lexer=v,e.marked=A,e.options=I,e.parse=D,e.parseInline=L,e.parser=P,e.setOptions=E,e.use=Z,e.walkTokens=q}));\n'

_QT_HTML_TEMPLATE = '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n<meta name="color-scheme" content="light dark">\n<style>\n* { box-sizing: border-box; }\nhtml, body {\n  margin: 0; padding: 10px 14px;\n  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;\n  font-size: 13px;\n  line-height: 1.6;\n  background: transparent;\n}\n@media (prefers-color-scheme: dark) {\n  body { color: #e0e0e0; }\n  pre  { background: rgba(255,255,255,0.07); }\n  code { background: rgba(255,255,255,0.10); }\n  .copy-btn { background: #444; color: #ccc; border-color: #555; }\n  .copy-btn:hover { background: #555; }\n}\n@media (prefers-color-scheme: light) {\n  body { color: #1a1a1a; }\n  pre  { background: rgba(0,0,0,0.05); }\n  code { background: rgba(0,0,0,0.07); }\n  .copy-btn { background: #e0e0e0; color: #333; border-color: #bbb; }\n  .copy-btn:hover { background: #ccc; }\n}\npre {\n  position: relative;\n  padding: 10px 12px;\n  border-radius: 6px;\n  overflow-x: auto;\n  font-size: 12px;\n}\ncode {\n  font-family: "SF Mono", "Menlo", "Monaco", monospace;\n  font-size: 12px;\n  padding: 1px 4px;\n  border-radius: 3px;\n}\npre code { background: none; padding: 0; }\n.copy-btn {\n  position: absolute;\n  top: 5px; right: 6px;\n  font-size: 10px;\n  padding: 2px 7px;\n  border-radius: 4px;\n  border: 1px solid;\n  cursor: pointer;\n  opacity: 0.75;\n}\n.copy-btn:active { opacity: 1; }\nblockquote {\n  border-left: 3px solid #888;\n  margin: 8px 0;\n  padding: 2px 12px;\n  color: #888;\n}\ntable { border-collapse: collapse; width: 100%; }\nth, td { border: 1px solid #555; padding: 5px 10px; text-align: left; }\nh1,h2,h3,h4 { margin: 12px 0 6px; }\np { margin: 6px 0; }\n</style>\n</head>\n<body>\n<div id="out"></div>\n<script>\n/**\n * marked v9.1.6 - a markdown parser\n * Copyright (c) 2011-2023, Christopher Jeffrey. (MIT Licensed)\n * https://github.com/markedjs/marked\n */\n!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?t(exports):"function"==typeof define&&define.amd?define(["exports"],t):t((e="undefined"!=typeof globalThis?globalThis:e||self).marked={})}(this,(function(e){"use strict";function t(){return{async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null}}function n(t){e.defaults=t}e.defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};const s=/[&<>"\']/,r=new RegExp(s.source,"g"),i=/[<>"\']|&(?!(#\\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\\w+);)/,l=new RegExp(i.source,"g"),o={"&":"&amp;","<":"&lt;",">":"&gt;",\'"\':"&quot;","\'":"&#39;"},a=e=>o[e];function c(e,t){if(t){if(s.test(e))return e.replace(r,a)}else if(i.test(e))return e.replace(l,a);return e}const h=/&(#(?:\\d+)|(?:#x[0-9A-Fa-f]+)|(?:\\w+));?/gi;const p=/(^|[^\\[])\\^/g;function u(e,t){e="string"==typeof e?e:e.source,t=t||"";const n={replace:(t,s)=>(s=(s="object"==typeof s&&"source"in s?s.source:s).replace(p,"$1"),e=e.replace(t,s),n),getRegex:()=>new RegExp(e,t)};return n}function g(e){try{e=encodeURI(e).replace(/%25/g,"%")}catch(e){return null}return e}const k={exec:()=>null};function f(e,t){const n=e.replace(/\\|/g,((e,t,n)=>{let s=!1,r=t;for(;--r>=0&&"\\\\"===n[r];)s=!s;return s?"|":" |"})).split(/ \\|/);let s=0;if(n[0].trim()||n.shift(),n.length>0&&!n[n.length-1].trim()&&n.pop(),t)if(n.length>t)n.splice(t);else for(;n.length<t;)n.push("");for(;s<n.length;s++)n[s]=n[s].trim().replace(/\\\\\\|/g,"|");return n}function d(e,t,n){const s=e.length;if(0===s)return"";let r=0;for(;r<s;){const i=e.charAt(s-r-1);if(i!==t||n){if(i===t||!n)break;r++}else r++}return e.slice(0,s-r)}function x(e,t,n,s){const r=t.href,i=t.title?c(t.title):null,l=e[1].replace(/\\\\([\\[\\]])/g,"$1");if("!"!==e[0].charAt(0)){s.state.inLink=!0;const e={type:"link",raw:n,href:r,title:i,text:l,tokens:s.inlineTokens(l)};return s.state.inLink=!1,e}return{type:"image",raw:n,href:r,title:i,text:c(l)}}class b{options;rules;lexer;constructor(t){this.options=t||e.defaults}space(e){const t=this.rules.block.newline.exec(e);if(t&&t[0].length>0)return{type:"space",raw:t[0]}}code(e){const t=this.rules.block.code.exec(e);if(t){const e=t[0].replace(/^ {1,4}/gm,"");return{type:"code",raw:t[0],codeBlockStyle:"indented",text:this.options.pedantic?e:d(e,"\\n")}}}fences(e){const t=this.rules.block.fences.exec(e);if(t){const e=t[0],n=function(e,t){const n=e.match(/^(\\s+)(?:```)/);if(null===n)return t;const s=n[1];return t.split("\\n").map((e=>{const t=e.match(/^\\s+/);if(null===t)return e;const[n]=t;return n.length>=s.length?e.slice(s.length):e})).join("\\n")}(e,t[3]||"");return{type:"code",raw:e,lang:t[2]?t[2].trim().replace(this.rules.inline._escapes,"$1"):t[2],text:n}}}heading(e){const t=this.rules.block.heading.exec(e);if(t){let e=t[2].trim();if(/#$/.test(e)){const t=d(e,"#");this.options.pedantic?e=t.trim():t&&!/ $/.test(t)||(e=t.trim())}return{type:"heading",raw:t[0],depth:t[1].length,text:e,tokens:this.lexer.inline(e)}}}hr(e){const t=this.rules.block.hr.exec(e);if(t)return{type:"hr",raw:t[0]}}blockquote(e){const t=this.rules.block.blockquote.exec(e);if(t){const e=d(t[0].replace(/^ *>[ \\t]?/gm,""),"\\n"),n=this.lexer.state.top;this.lexer.state.top=!0;const s=this.lexer.blockTokens(e);return this.lexer.state.top=n,{type:"blockquote",raw:t[0],tokens:s,text:e}}}list(e){let t=this.rules.block.list.exec(e);if(t){let n=t[1].trim();const s=n.length>1,r={type:"list",raw:"",ordered:s,start:s?+n.slice(0,-1):"",loose:!1,items:[]};n=s?`\\\\d{1,9}\\\\${n.slice(-1)}`:`\\\\${n}`,this.options.pedantic&&(n=s?n:"[*+-]");const i=new RegExp(`^( {0,3}${n})((?:[\\t ][^\\\\n]*)?(?:\\\\n|$))`);let l="",o="",a=!1;for(;e;){let n=!1;if(!(t=i.exec(e)))break;if(this.rules.block.hr.test(e))break;l=t[0],e=e.substring(l.length);let s=t[2].split("\\n",1)[0].replace(/^\\t+/,(e=>" ".repeat(3*e.length))),c=e.split("\\n",1)[0],h=0;this.options.pedantic?(h=2,o=s.trimStart()):(h=t[2].search(/[^ ]/),h=h>4?1:h,o=s.slice(h),h+=t[1].length);let p=!1;if(!s&&/^ *$/.test(c)&&(l+=c+"\\n",e=e.substring(c.length+1),n=!0),!n){const t=new RegExp(`^ {0,${Math.min(3,h-1)}}(?:[*+-]|\\\\d{1,9}[.)])((?:[ \\t][^\\\\n]*)?(?:\\\\n|$))`),n=new RegExp(`^ {0,${Math.min(3,h-1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\\\* *){3,})(?:\\\\n+|$)`),r=new RegExp(`^ {0,${Math.min(3,h-1)}}(?:\\`\\`\\`|~~~)`),i=new RegExp(`^ {0,${Math.min(3,h-1)}}#`);for(;e;){const a=e.split("\\n",1)[0];if(c=a,this.options.pedantic&&(c=c.replace(/^ {1,4}(?=( {4})*[^ ])/g,"  ")),r.test(c))break;if(i.test(c))break;if(t.test(c))break;if(n.test(e))break;if(c.search(/[^ ]/)>=h||!c.trim())o+="\\n"+c.slice(h);else{if(p)break;if(s.search(/[^ ]/)>=4)break;if(r.test(s))break;if(i.test(s))break;if(n.test(s))break;o+="\\n"+c}p||c.trim()||(p=!0),l+=a+"\\n",e=e.substring(a.length+1),s=c.slice(h)}}r.loose||(a?r.loose=!0:/\\n *\\n *$/.test(l)&&(a=!0));let u,g=null;this.options.gfm&&(g=/^\\[[ xX]\\] /.exec(o),g&&(u="[ ] "!==g[0],o=o.replace(/^\\[[ xX]\\] +/,""))),r.items.push({type:"list_item",raw:l,task:!!g,checked:u,loose:!1,text:o,tokens:[]}),r.raw+=l}r.items[r.items.length-1].raw=l.trimEnd(),r.items[r.items.length-1].text=o.trimEnd(),r.raw=r.raw.trimEnd();for(let e=0;e<r.items.length;e++)if(this.lexer.state.top=!1,r.items[e].tokens=this.lexer.blockTokens(r.items[e].text,[]),!r.loose){const t=r.items[e].tokens.filter((e=>"space"===e.type)),n=t.length>0&&t.some((e=>/\\n.*\\n/.test(e.raw)));r.loose=n}if(r.loose)for(let e=0;e<r.items.length;e++)r.items[e].loose=!0;return r}}html(e){const t=this.rules.block.html.exec(e);if(t){return{type:"html",block:!0,raw:t[0],pre:"pre"===t[1]||"script"===t[1]||"style"===t[1],text:t[0]}}}def(e){const t=this.rules.block.def.exec(e);if(t){const e=t[1].toLowerCase().replace(/\\s+/g," "),n=t[2]?t[2].replace(/^<(.*)>$/,"$1").replace(this.rules.inline._escapes,"$1"):"",s=t[3]?t[3].substring(1,t[3].length-1).replace(this.rules.inline._escapes,"$1"):t[3];return{type:"def",tag:e,raw:t[0],href:n,title:s}}}table(e){const t=this.rules.block.table.exec(e);if(t){if(!/[:|]/.test(t[2]))return;const e={type:"table",raw:t[0],header:f(t[1]).map((e=>({text:e,tokens:[]}))),align:t[2].replace(/^\\||\\| *$/g,"").split("|"),rows:t[3]&&t[3].trim()?t[3].replace(/\\n[ \\t]*$/,"").split("\\n"):[]};if(e.header.length===e.align.length){let t,n,s,r,i=e.align.length;for(t=0;t<i;t++){const n=e.align[t];n&&(/^ *-+: *$/.test(n)?e.align[t]="right":/^ *:-+: *$/.test(n)?e.align[t]="center":/^ *:-+ *$/.test(n)?e.align[t]="left":e.align[t]=null)}for(i=e.rows.length,t=0;t<i;t++)e.rows[t]=f(e.rows[t],e.header.length).map((e=>({text:e,tokens:[]})));for(i=e.header.length,n=0;n<i;n++)e.header[n].tokens=this.lexer.inline(e.header[n].text);for(i=e.rows.length,n=0;n<i;n++)for(r=e.rows[n],s=0;s<r.length;s++)r[s].tokens=this.lexer.inline(r[s].text);return e}}}lheading(e){const t=this.rules.block.lheading.exec(e);if(t)return{type:"heading",raw:t[0],depth:"="===t[2].charAt(0)?1:2,text:t[1],tokens:this.lexer.inline(t[1])}}paragraph(e){const t=this.rules.block.paragraph.exec(e);if(t){const e="\\n"===t[1].charAt(t[1].length-1)?t[1].slice(0,-1):t[1];return{type:"paragraph",raw:t[0],text:e,tokens:this.lexer.inline(e)}}}text(e){const t=this.rules.block.text.exec(e);if(t)return{type:"text",raw:t[0],text:t[0],tokens:this.lexer.inline(t[0])}}escape(e){const t=this.rules.inline.escape.exec(e);if(t)return{type:"escape",raw:t[0],text:c(t[1])}}tag(e){const t=this.rules.inline.tag.exec(e);if(t)return!this.lexer.state.inLink&&/^<a /i.test(t[0])?this.lexer.state.inLink=!0:this.lexer.state.inLink&&/^<\\/a>/i.test(t[0])&&(this.lexer.state.inLink=!1),!this.lexer.state.inRawBlock&&/^<(pre|code|kbd|script)(\\s|>)/i.test(t[0])?this.lexer.state.inRawBlock=!0:this.lexer.state.inRawBlock&&/^<\\/(pre|code|kbd|script)(\\s|>)/i.test(t[0])&&(this.lexer.state.inRawBlock=!1),{type:"html",raw:t[0],inLink:this.lexer.state.inLink,inRawBlock:this.lexer.state.inRawBlock,block:!1,text:t[0]}}link(e){const t=this.rules.inline.link.exec(e);if(t){const e=t[2].trim();if(!this.options.pedantic&&/^</.test(e)){if(!/>$/.test(e))return;const t=d(e.slice(0,-1),"\\\\");if((e.length-t.length)%2==0)return}else{const e=function(e,t){if(-1===e.indexOf(t[1]))return-1;let n=0;for(let s=0;s<e.length;s++)if("\\\\"===e[s])s++;else if(e[s]===t[0])n++;else if(e[s]===t[1]&&(n--,n<0))return s;return-1}(t[2],"()");if(e>-1){const n=(0===t[0].indexOf("!")?5:4)+t[1].length+e;t[2]=t[2].substring(0,e),t[0]=t[0].substring(0,n).trim(),t[3]=""}}let n=t[2],s="";if(this.options.pedantic){const e=/^([^\'"]*[^\\s])\\s+([\'"])(.*)\\2/.exec(n);e&&(n=e[1],s=e[3])}else s=t[3]?t[3].slice(1,-1):"";return n=n.trim(),/^</.test(n)&&(n=this.options.pedantic&&!/>$/.test(e)?n.slice(1):n.slice(1,-1)),x(t,{href:n?n.replace(this.rules.inline._escapes,"$1"):n,title:s?s.replace(this.rules.inline._escapes,"$1"):s},t[0],this.lexer)}}reflink(e,t){let n;if((n=this.rules.inline.reflink.exec(e))||(n=this.rules.inline.nolink.exec(e))){let e=(n[2]||n[1]).replace(/\\s+/g," ");if(e=t[e.toLowerCase()],!e){const e=n[0].charAt(0);return{type:"text",raw:e,text:e}}return x(n,e,n[0],this.lexer)}}emStrong(e,t,n=""){let s=this.rules.inline.emStrong.lDelim.exec(e);if(!s)return;if(s[3]&&n.match(/[\\p{L}\\p{N}]/u))return;if(!(s[1]||s[2]||"")||!n||this.rules.inline.punctuation.exec(n)){const n=[...s[0]].length-1;let r,i,l=n,o=0;const a="*"===s[0][0]?this.rules.inline.emStrong.rDelimAst:this.rules.inline.emStrong.rDelimUnd;for(a.lastIndex=0,t=t.slice(-1*e.length+n);null!=(s=a.exec(t));){if(r=s[1]||s[2]||s[3]||s[4]||s[5]||s[6],!r)continue;if(i=[...r].length,s[3]||s[4]){l+=i;continue}if((s[5]||s[6])&&n%3&&!((n+i)%3)){o+=i;continue}if(l-=i,l>0)continue;i=Math.min(i,i+l+o);const t=[...s[0]][0].length,a=e.slice(0,n+s.index+t+i);if(Math.min(n,i)%2){const e=a.slice(1,-1);return{type:"em",raw:a,text:e,tokens:this.lexer.inlineTokens(e)}}const c=a.slice(2,-2);return{type:"strong",raw:a,text:c,tokens:this.lexer.inlineTokens(c)}}}}codespan(e){const t=this.rules.inline.code.exec(e);if(t){let e=t[2].replace(/\\n/g," ");const n=/[^ ]/.test(e),s=/^ /.test(e)&&/ $/.test(e);return n&&s&&(e=e.substring(1,e.length-1)),e=c(e,!0),{type:"codespan",raw:t[0],text:e}}}br(e){const t=this.rules.inline.br.exec(e);if(t)return{type:"br",raw:t[0]}}del(e){const t=this.rules.inline.del.exec(e);if(t)return{type:"del",raw:t[0],text:t[2],tokens:this.lexer.inlineTokens(t[2])}}autolink(e){const t=this.rules.inline.autolink.exec(e);if(t){let e,n;return"@"===t[2]?(e=c(t[1]),n="mailto:"+e):(e=c(t[1]),n=e),{type:"link",raw:t[0],text:e,href:n,tokens:[{type:"text",raw:e,text:e}]}}}url(e){let t;if(t=this.rules.inline.url.exec(e)){let e,n;if("@"===t[2])e=c(t[0]),n="mailto:"+e;else{let s;do{s=t[0],t[0]=this.rules.inline._backpedal.exec(t[0])[0]}while(s!==t[0]);e=c(t[0]),n="www."===t[1]?"http://"+t[0]:t[0]}return{type:"link",raw:t[0],text:e,href:n,tokens:[{type:"text",raw:e,text:e}]}}}inlineText(e){const t=this.rules.inline.text.exec(e);if(t){let e;return e=this.lexer.state.inRawBlock?t[0]:c(t[0]),{type:"text",raw:t[0],text:e}}}}const m={newline:/^(?: *(?:\\n|$))+/,code:/^( {4}[^\\n]+(?:\\n(?: *(?:\\n|$))*)?)+/,fences:/^ {0,3}(`{3,}(?=[^`\\n]*(?:\\n|$))|~{3,})([^\\n]*)(?:\\n|$)(?:|([\\s\\S]*?)(?:\\n|$))(?: {0,3}\\1[~`]* *(?=\\n|$)|$)/,hr:/^ {0,3}((?:-[\\t ]*){3,}|(?:_[ \\t]*){3,}|(?:\\*[ \\t]*){3,})(?:\\n+|$)/,heading:/^ {0,3}(#{1,6})(?=\\s|$)(.*)(?:\\n+|$)/,blockquote:/^( {0,3}> ?(paragraph|[^\\n]*)(?:\\n|$))+/,list:/^( {0,3}bull)([ \\t][^\\n]+?)?(?:\\n|$)/,html:"^ {0,3}(?:<(script|pre|style|textarea)[\\\\s>][\\\\s\\\\S]*?(?:</\\\\1>[^\\\\n]*\\\\n+|$)|comment[^\\\\n]*(\\\\n+|$)|<\\\\?[\\\\s\\\\S]*?(?:\\\\?>\\\\n*|$)|<![A-Z][\\\\s\\\\S]*?(?:>\\\\n*|$)|<!\\\\[CDATA\\\\[[\\\\s\\\\S]*?(?:\\\\]\\\\]>\\\\n*|$)|</?(tag)(?: +|\\\\n|/?>)[\\\\s\\\\S]*?(?:(?:\\\\n *)+\\\\n|$)|<(?!script|pre|style|textarea)([a-z][\\\\w-]*)(?:attribute)*? */?>(?=[ \\\\t]*(?:\\\\n|$))[\\\\s\\\\S]*?(?:(?:\\\\n *)+\\\\n|$)|</(?!script|pre|style|textarea)[a-z][\\\\w-]*\\\\s*>(?=[ \\\\t]*(?:\\\\n|$))[\\\\s\\\\S]*?(?:(?:\\\\n *)+\\\\n|$))",def:/^ {0,3}\\[(label)\\]: *(?:\\n *)?([^<\\s][^\\s]*|<.*?>)(?:(?: +(?:\\n *)?| *\\n *)(title))? *(?:\\n+|$)/,table:k,lheading:/^(?!bull )((?:.|\\n(?!\\s*?\\n|bull ))+?)\\n {0,3}(=+|-+) *(?:\\n+|$)/,_paragraph:/^([^\\n]+(?:\\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\\n)[^\\n]+)*)/,text:/^[^\\n]+/,_label:/(?!\\s*\\])(?:\\\\.|[^\\[\\]\\\\])+/,_title:/(?:"(?:\\\\"?|[^"\\\\])*"|\'[^\'\\n]*(?:\\n[^\'\\n]+)*\\n?\'|\\([^()]*\\))/};m.def=u(m.def).replace("label",m._label).replace("title",m._title).getRegex(),m.bullet=/(?:[*+-]|\\d{1,9}[.)])/,m.listItemStart=u(/^( *)(bull) */).replace("bull",m.bullet).getRegex(),m.list=u(m.list).replace(/bull/g,m.bullet).replace("hr","\\\\n+(?=\\\\1?(?:(?:- *){3,}|(?:_ *){3,}|(?:\\\\* *){3,})(?:\\\\n+|$))").replace("def","\\\\n+(?="+m.def.source+")").getRegex(),m._tag="address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul",m._comment=/<!--(?!-?>)[\\s\\S]*?(?:-->|$)/,m.html=u(m.html,"i").replace("comment",m._comment).replace("tag",m._tag).replace("attribute",/ +[a-zA-Z:_][\\w.:-]*(?: *= *"[^"\\n]*"| *= *\'[^\'\\n]*\'| *= *[^\\s"\'=<>`]+)?/).getRegex(),m.lheading=u(m.lheading).replace(/bull/g,m.bullet).getRegex(),m.paragraph=u(m._paragraph).replace("hr",m.hr).replace("heading"," {0,3}#{1,6}(?:\\\\s|$)").replace("|lheading","").replace("|table","").replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\\\n]*\\\\n)|~{3,})[^\\\\n]*\\\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",m._tag).getRegex(),m.blockquote=u(m.blockquote).replace("paragraph",m.paragraph).getRegex(),m.normal={...m},m.gfm={...m.normal,table:"^ *([^\\\\n ].*)\\\\n {0,3}((?:\\\\| *)?:?-+:? *(?:\\\\| *:?-+:? *)*(?:\\\\| *)?)(?:\\\\n((?:(?! *\\\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\\\n|$))*)\\\\n*|$)"},m.gfm.table=u(m.gfm.table).replace("hr",m.hr).replace("heading"," {0,3}#{1,6}(?:\\\\s|$)").replace("blockquote"," {0,3}>").replace("code"," {4}[^\\\\n]").replace("fences"," {0,3}(?:`{3,}(?=[^`\\\\n]*\\\\n)|~{3,})[^\\\\n]*\\\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",m._tag).getRegex(),m.gfm.paragraph=u(m._paragraph).replace("hr",m.hr).replace("heading"," {0,3}#{1,6}(?:\\\\s|$)").replace("|lheading","").replace("table",m.gfm.table).replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\\\n]*\\\\n)|~{3,})[^\\\\n]*\\\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",m._tag).getRegex(),m.pedantic={...m.normal,html:u("^ *(?:comment *(?:\\\\n|\\\\s*$)|<(tag)[\\\\s\\\\S]+?</\\\\1> *(?:\\\\n{2,}|\\\\s*$)|<tag(?:\\"[^\\"]*\\"|\'[^\']*\'|\\\\s[^\'\\"/>\\\\s]*)*?/?> *(?:\\\\n{2,}|\\\\s*$))").replace("comment",m._comment).replace(/tag/g,"(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\\\b)\\\\w+(?!:|[^\\\\w\\\\s@]*@)\\\\b").getRegex(),def:/^ *\\[([^\\]]+)\\]: *<?([^\\s>]+)>?(?: +(["(][^\\n]+[")]))? *(?:\\n+|$)/,heading:/^(#{1,6})(.*)(?:\\n+|$)/,fences:k,lheading:/^(.+?)\\n {0,3}(=+|-+) *(?:\\n+|$)/,paragraph:u(m.normal._paragraph).replace("hr",m.hr).replace("heading"," *#{1,6} *[^\\n]").replace("lheading",m.lheading).replace("blockquote"," {0,3}>").replace("|fences","").replace("|list","").replace("|html","").getRegex()};const w={escape:/^\\\\([!"#$%&\'()*+,\\-./:;<=>?@\\[\\]\\\\^_`{|}~])/,autolink:/^<(scheme:[^\\s\\x00-\\x1f<>]*|email)>/,url:k,tag:"^comment|^</[a-zA-Z][\\\\w:-]*\\\\s*>|^<[a-zA-Z][\\\\w-]*(?:attribute)*?\\\\s*/?>|^<\\\\?[\\\\s\\\\S]*?\\\\?>|^<![a-zA-Z]+\\\\s[\\\\s\\\\S]*?>|^<!\\\\[CDATA\\\\[[\\\\s\\\\S]*?\\\\]\\\\]>",link:/^!?\\[(label)\\]\\(\\s*(href)(?:\\s+(title))?\\s*\\)/,reflink:/^!?\\[(label)\\]\\[(ref)\\]/,nolink:/^!?\\[(ref)\\](?:\\[\\])?/,reflinkSearch:"reflink|nolink(?!\\\\()",emStrong:{lDelim:/^(?:\\*+(?:((?!\\*)[punct])|[^\\s*]))|^_+(?:((?!_)[punct])|([^\\s_]))/,rDelimAst:/^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)[punct](\\*+)(?=[\\s]|$)|[^punct\\s](\\*+)(?!\\*)(?=[punct\\s]|$)|(?!\\*)[punct\\s](\\*+)(?=[^punct\\s])|[\\s](\\*+)(?!\\*)(?=[punct])|(?!\\*)[punct](\\*+)(?!\\*)(?=[punct])|[^punct\\s](\\*+)(?=[^punct\\s])/,rDelimUnd:/^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)[punct](_+)(?=[\\s]|$)|[^punct\\s](_+)(?!_)(?=[punct\\s]|$)|(?!_)[punct\\s](_+)(?=[^punct\\s])|[\\s](_+)(?!_)(?=[punct])|(?!_)[punct](_+)(?!_)(?=[punct])/},code:/^(`+)([^`]|[^`][\\s\\S]*?[^`])\\1(?!`)/,br:/^( {2,}|\\\\)\\n(?!\\s*$)/,del:k,text:/^(`+|[^`])(?:(?= {2,}\\n)|[\\s\\S]*?(?:(?=[\\\\<!\\[`*_]|\\b_|$)|[^ ](?= {2,}\\n)))/,punctuation:/^((?![*_])[\\spunctuation])/,_punctuation:"\\\\p{P}$+<=>`^|~"};w.punctuation=u(w.punctuation,"u").replace(/punctuation/g,w._punctuation).getRegex(),w.blockSkip=/\\[[^[\\]]*?\\]\\([^\\(\\)]*?\\)|`[^`]*?`|<[^<>]*?>/g,w.anyPunctuation=/\\\\[punct]/g,w._escapes=/\\\\([punct])/g,w._comment=u(m._comment).replace("(?:--\\x3e|$)","--\\x3e").getRegex(),w.emStrong.lDelim=u(w.emStrong.lDelim,"u").replace(/punct/g,w._punctuation).getRegex(),w.emStrong.rDelimAst=u(w.emStrong.rDelimAst,"gu").replace(/punct/g,w._punctuation).getRegex(),w.emStrong.rDelimUnd=u(w.emStrong.rDelimUnd,"gu").replace(/punct/g,w._punctuation).getRegex(),w.anyPunctuation=u(w.anyPunctuation,"gu").replace(/punct/g,w._punctuation).getRegex(),w._escapes=u(w._escapes,"gu").replace(/punct/g,w._punctuation).getRegex(),w._scheme=/[a-zA-Z][a-zA-Z0-9+.-]{1,31}/,w._email=/[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/,w.autolink=u(w.autolink).replace("scheme",w._scheme).replace("email",w._email).getRegex(),w._attribute=/\\s+[a-zA-Z:_][\\w.:-]*(?:\\s*=\\s*"[^"]*"|\\s*=\\s*\'[^\']*\'|\\s*=\\s*[^\\s"\'=<>`]+)?/,w.tag=u(w.tag).replace("comment",w._comment).replace("attribute",w._attribute).getRegex(),w._label=/(?:\\[(?:\\\\.|[^\\[\\]\\\\])*\\]|\\\\.|`[^`]*`|[^\\[\\]\\\\`])*?/,w._href=/<(?:\\\\.|[^\\n<>\\\\])+>|[^\\s\\x00-\\x1f]*/,w._title=/"(?:\\\\"?|[^"\\\\])*"|\'(?:\\\\\'?|[^\'\\\\])*\'|\\((?:\\\\\\)?|[^)\\\\])*\\)/,w.link=u(w.link).replace("label",w._label).replace("href",w._href).replace("title",w._title).getRegex(),w.reflink=u(w.reflink).replace("label",w._label).replace("ref",m._label).getRegex(),w.nolink=u(w.nolink).replace("ref",m._label).getRegex(),w.reflinkSearch=u(w.reflinkSearch,"g").replace("reflink",w.reflink).replace("nolink",w.nolink).getRegex(),w.normal={...w},w.pedantic={...w.normal,strong:{start:/^__|\\*\\*/,middle:/^__(?=\\S)([\\s\\S]*?\\S)__(?!_)|^\\*\\*(?=\\S)([\\s\\S]*?\\S)\\*\\*(?!\\*)/,endAst:/\\*\\*(?!\\*)/g,endUnd:/__(?!_)/g},em:{start:/^_|\\*/,middle:/^()\\*(?=\\S)([\\s\\S]*?\\S)\\*(?!\\*)|^_(?=\\S)([\\s\\S]*?\\S)_(?!_)/,endAst:/\\*(?!\\*)/g,endUnd:/_(?!_)/g},link:u(/^!?\\[(label)\\]\\((.*?)\\)/).replace("label",w._label).getRegex(),reflink:u(/^!?\\[(label)\\]\\s*\\[([^\\]]*)\\]/).replace("label",w._label).getRegex()},w.gfm={...w.normal,escape:u(w.escape).replace("])","~|])").getRegex(),_extended_email:/[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/,url:/^((?:ftp|https?):\\/\\/|www\\.)(?:[a-zA-Z0-9\\-]+\\.?)+[^\\s<]*|^email/,_backpedal:/(?:[^?!.,:;*_\'"~()&]+|\\([^)]*\\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_\'"~)]+(?!$))+/,del:/^(~~?)(?=[^\\s~])([\\s\\S]*?[^\\s~])\\1(?=[^~]|$)/,text:/^([`~]+|[^`~])(?:(?= {2,}\\n)|(?=[a-zA-Z0-9.!#$%&\'*+\\/=?_`{\\|}~-]+@)|[\\s\\S]*?(?:(?=[\\\\<!\\[`*~_]|\\b_|https?:\\/\\/|ftp:\\/\\/|www\\.|$)|[^ ](?= {2,}\\n)|[^a-zA-Z0-9.!#$%&\'*+\\/=?_`{\\|}~-](?=[a-zA-Z0-9.!#$%&\'*+\\/=?_`{\\|}~-]+@)))/},w.gfm.url=u(w.gfm.url,"i").replace("email",w.gfm._extended_email).getRegex(),w.breaks={...w.gfm,br:u(w.br).replace("{2,}","*").getRegex(),text:u(w.gfm.text).replace("\\\\b_","\\\\b_| {2,}\\\\n").replace(/\\{2,\\}/g,"*").getRegex()};class _{tokens;options;state;tokenizer;inlineQueue;constructor(t){this.tokens=[],this.tokens.links=Object.create(null),this.options=t||e.defaults,this.options.tokenizer=this.options.tokenizer||new b,this.tokenizer=this.options.tokenizer,this.tokenizer.options=this.options,this.tokenizer.lexer=this,this.inlineQueue=[],this.state={inLink:!1,inRawBlock:!1,top:!0};const n={block:m.normal,inline:w.normal};this.options.pedantic?(n.block=m.pedantic,n.inline=w.pedantic):this.options.gfm&&(n.block=m.gfm,this.options.breaks?n.inline=w.breaks:n.inline=w.gfm),this.tokenizer.rules=n}static get rules(){return{block:m,inline:w}}static lex(e,t){return new _(t).lex(e)}static lexInline(e,t){return new _(t).inlineTokens(e)}lex(e){let t;for(e=e.replace(/\\r\\n|\\r/g,"\\n"),this.blockTokens(e,this.tokens);t=this.inlineQueue.shift();)this.inlineTokens(t.src,t.tokens);return this.tokens}blockTokens(e,t=[]){let n,s,r,i;for(e=this.options.pedantic?e.replace(/\\t/g,"    ").replace(/^ +$/gm,""):e.replace(/^( *)(\\t+)/gm,((e,t,n)=>t+"    ".repeat(n.length)));e;)if(!(this.options.extensions&&this.options.extensions.block&&this.options.extensions.block.some((s=>!!(n=s.call({lexer:this},e,t))&&(e=e.substring(n.raw.length),t.push(n),!0)))))if(n=this.tokenizer.space(e))e=e.substring(n.raw.length),1===n.raw.length&&t.length>0?t[t.length-1].raw+="\\n":t.push(n);else if(n=this.tokenizer.code(e))e=e.substring(n.raw.length),s=t[t.length-1],!s||"paragraph"!==s.type&&"text"!==s.type?t.push(n):(s.raw+="\\n"+n.raw,s.text+="\\n"+n.text,this.inlineQueue[this.inlineQueue.length-1].src=s.text);else if(n=this.tokenizer.fences(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.heading(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.hr(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.blockquote(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.list(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.html(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.def(e))e=e.substring(n.raw.length),s=t[t.length-1],!s||"paragraph"!==s.type&&"text"!==s.type?this.tokens.links[n.tag]||(this.tokens.links[n.tag]={href:n.href,title:n.title}):(s.raw+="\\n"+n.raw,s.text+="\\n"+n.raw,this.inlineQueue[this.inlineQueue.length-1].src=s.text);else if(n=this.tokenizer.table(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.lheading(e))e=e.substring(n.raw.length),t.push(n);else{if(r=e,this.options.extensions&&this.options.extensions.startBlock){let t=1/0;const n=e.slice(1);let s;this.options.extensions.startBlock.forEach((e=>{s=e.call({lexer:this},n),"number"==typeof s&&s>=0&&(t=Math.min(t,s))})),t<1/0&&t>=0&&(r=e.substring(0,t+1))}if(this.state.top&&(n=this.tokenizer.paragraph(r)))s=t[t.length-1],i&&"paragraph"===s.type?(s.raw+="\\n"+n.raw,s.text+="\\n"+n.text,this.inlineQueue.pop(),this.inlineQueue[this.inlineQueue.length-1].src=s.text):t.push(n),i=r.length!==e.length,e=e.substring(n.raw.length);else if(n=this.tokenizer.text(e))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===s.type?(s.raw+="\\n"+n.raw,s.text+="\\n"+n.text,this.inlineQueue.pop(),this.inlineQueue[this.inlineQueue.length-1].src=s.text):t.push(n);else if(e){const t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}return this.state.top=!0,t}inline(e,t=[]){return this.inlineQueue.push({src:e,tokens:t}),t}inlineTokens(e,t=[]){let n,s,r,i,l,o,a=e;if(this.tokens.links){const e=Object.keys(this.tokens.links);if(e.length>0)for(;null!=(i=this.tokenizer.rules.inline.reflinkSearch.exec(a));)e.includes(i[0].slice(i[0].lastIndexOf("[")+1,-1))&&(a=a.slice(0,i.index)+"["+"a".repeat(i[0].length-2)+"]"+a.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex))}for(;null!=(i=this.tokenizer.rules.inline.blockSkip.exec(a));)a=a.slice(0,i.index)+"["+"a".repeat(i[0].length-2)+"]"+a.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);for(;null!=(i=this.tokenizer.rules.inline.anyPunctuation.exec(a));)a=a.slice(0,i.index)+"++"+a.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);for(;e;)if(l||(o=""),l=!1,!(this.options.extensions&&this.options.extensions.inline&&this.options.extensions.inline.some((s=>!!(n=s.call({lexer:this},e,t))&&(e=e.substring(n.raw.length),t.push(n),!0)))))if(n=this.tokenizer.escape(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.tag(e))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===n.type&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(n=this.tokenizer.link(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.reflink(e,this.tokens.links))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===n.type&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(n=this.tokenizer.emStrong(e,a,o))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.codespan(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.br(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.del(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.autolink(e))e=e.substring(n.raw.length),t.push(n);else if(this.state.inLink||!(n=this.tokenizer.url(e))){if(r=e,this.options.extensions&&this.options.extensions.startInline){let t=1/0;const n=e.slice(1);let s;this.options.extensions.startInline.forEach((e=>{s=e.call({lexer:this},n),"number"==typeof s&&s>=0&&(t=Math.min(t,s))})),t<1/0&&t>=0&&(r=e.substring(0,t+1))}if(n=this.tokenizer.inlineText(r))e=e.substring(n.raw.length),"_"!==n.raw.slice(-1)&&(o=n.raw.slice(-1)),l=!0,s=t[t.length-1],s&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(e){const t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}else e=e.substring(n.raw.length),t.push(n);return t}}class y{options;constructor(t){this.options=t||e.defaults}code(e,t,n){const s=(t||"").match(/^\\S*/)?.[0];return e=e.replace(/\\n$/,"")+"\\n",s?\'<pre><code class="language-\'+c(s)+\'">\'+(n?e:c(e,!0))+"</code></pre>\\n":"<pre><code>"+(n?e:c(e,!0))+"</code></pre>\\n"}blockquote(e){return`<blockquote>\\n${e}</blockquote>\\n`}html(e,t){return e}heading(e,t,n){return`<h${t}>${e}</h${t}>\\n`}hr(){return"<hr>\\n"}list(e,t,n){const s=t?"ol":"ul";return"<"+s+(t&&1!==n?\' start="\'+n+\'"\':"")+">\\n"+e+"</"+s+">\\n"}listitem(e,t,n){return`<li>${e}</li>\\n`}checkbox(e){return"<input "+(e?\'checked="" \':"")+\'disabled="" type="checkbox">\'}paragraph(e){return`<p>${e}</p>\\n`}table(e,t){return t&&(t=`<tbody>${t}</tbody>`),"<table>\\n<thead>\\n"+e+"</thead>\\n"+t+"</table>\\n"}tablerow(e){return`<tr>\\n${e}</tr>\\n`}tablecell(e,t){const n=t.header?"th":"td";return(t.align?`<${n} align="${t.align}">`:`<${n}>`)+e+`</${n}>\\n`}strong(e){return`<strong>${e}</strong>`}em(e){return`<em>${e}</em>`}codespan(e){return`<code>${e}</code>`}br(){return"<br>"}del(e){return`<del>${e}</del>`}link(e,t,n){const s=g(e);if(null===s)return n;let r=\'<a href="\'+(e=s)+\'"\';return t&&(r+=\' title="\'+t+\'"\'),r+=">"+n+"</a>",r}image(e,t,n){const s=g(e);if(null===s)return n;let r=`<img src="${e=s}" alt="${n}"`;return t&&(r+=` title="${t}"`),r+=">",r}text(e){return e}}class ${strong(e){return e}em(e){return e}codespan(e){return e}del(e){return e}html(e){return e}text(e){return e}link(e,t,n){return""+n}image(e,t,n){return""+n}br(){return""}}class z{options;renderer;textRenderer;constructor(t){this.options=t||e.defaults,this.options.renderer=this.options.renderer||new y,this.renderer=this.options.renderer,this.renderer.options=this.options,this.textRenderer=new $}static parse(e,t){return new z(t).parse(e)}static parseInline(e,t){return new z(t).parseInline(e)}parse(e,t=!0){let n="";for(let s=0;s<e.length;s++){const r=e[s];if(this.options.extensions&&this.options.extensions.renderers&&this.options.extensions.renderers[r.type]){const e=r,t=this.options.extensions.renderers[e.type].call({parser:this},e);if(!1!==t||!["space","hr","heading","code","table","blockquote","list","html","paragraph","text"].includes(e.type)){n+=t||"";continue}}switch(r.type){case"space":continue;case"hr":n+=this.renderer.hr();continue;case"heading":{const e=r;n+=this.renderer.heading(this.parseInline(e.tokens),e.depth,this.parseInline(e.tokens,this.textRenderer).replace(h,((e,t)=>"colon"===(t=t.toLowerCase())?":":"#"===t.charAt(0)?"x"===t.charAt(1)?String.fromCharCode(parseInt(t.substring(2),16)):String.fromCharCode(+t.substring(1)):"")));continue}case"code":{const e=r;n+=this.renderer.code(e.text,e.lang,!!e.escaped);continue}case"table":{const e=r;let t="",s="";for(let t=0;t<e.header.length;t++)s+=this.renderer.tablecell(this.parseInline(e.header[t].tokens),{header:!0,align:e.align[t]});t+=this.renderer.tablerow(s);let i="";for(let t=0;t<e.rows.length;t++){const n=e.rows[t];s="";for(let t=0;t<n.length;t++)s+=this.renderer.tablecell(this.parseInline(n[t].tokens),{header:!1,align:e.align[t]});i+=this.renderer.tablerow(s)}n+=this.renderer.table(t,i);continue}case"blockquote":{const e=r,t=this.parse(e.tokens);n+=this.renderer.blockquote(t);continue}case"list":{const e=r,t=e.ordered,s=e.start,i=e.loose;let l="";for(let t=0;t<e.items.length;t++){const n=e.items[t],s=n.checked,r=n.task;let o="";if(n.task){const e=this.renderer.checkbox(!!s);i?n.tokens.length>0&&"paragraph"===n.tokens[0].type?(n.tokens[0].text=e+" "+n.tokens[0].text,n.tokens[0].tokens&&n.tokens[0].tokens.length>0&&"text"===n.tokens[0].tokens[0].type&&(n.tokens[0].tokens[0].text=e+" "+n.tokens[0].tokens[0].text)):n.tokens.unshift({type:"text",text:e+" "}):o+=e+" "}o+=this.parse(n.tokens,i),l+=this.renderer.listitem(o,r,!!s)}n+=this.renderer.list(l,t,s);continue}case"html":{const e=r;n+=this.renderer.html(e.text,e.block);continue}case"paragraph":{const e=r;n+=this.renderer.paragraph(this.parseInline(e.tokens));continue}case"text":{let i=r,l=i.tokens?this.parseInline(i.tokens):i.text;for(;s+1<e.length&&"text"===e[s+1].type;)i=e[++s],l+="\\n"+(i.tokens?this.parseInline(i.tokens):i.text);n+=t?this.renderer.paragraph(l):l;continue}default:{const e=\'Token with "\'+r.type+\'" type was not found.\';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return n}parseInline(e,t){t=t||this.renderer;let n="";for(let s=0;s<e.length;s++){const r=e[s];if(this.options.extensions&&this.options.extensions.renderers&&this.options.extensions.renderers[r.type]){const e=this.options.extensions.renderers[r.type].call({parser:this},r);if(!1!==e||!["escape","html","link","image","strong","em","codespan","br","del","text"].includes(r.type)){n+=e||"";continue}}switch(r.type){case"escape":{const e=r;n+=t.text(e.text);break}case"html":{const e=r;n+=t.html(e.text);break}case"link":{const e=r;n+=t.link(e.href,e.title,this.parseInline(e.tokens,t));break}case"image":{const e=r;n+=t.image(e.href,e.title,e.text);break}case"strong":{const e=r;n+=t.strong(this.parseInline(e.tokens,t));break}case"em":{const e=r;n+=t.em(this.parseInline(e.tokens,t));break}case"codespan":{const e=r;n+=t.codespan(e.text);break}case"br":n+=t.br();break;case"del":{const e=r;n+=t.del(this.parseInline(e.tokens,t));break}case"text":{const e=r;n+=t.text(e.text);break}default:{const e=\'Token with "\'+r.type+\'" type was not found.\';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return n}}class T{options;constructor(t){this.options=t||e.defaults}static passThroughHooks=new Set(["preprocess","postprocess"]);preprocess(e){return e}postprocess(e){return e}}class R{defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};options=this.setOptions;parse=this.#e(_.lex,z.parse);parseInline=this.#e(_.lexInline,z.parseInline);Parser=z;Renderer=y;TextRenderer=$;Lexer=_;Tokenizer=b;Hooks=T;constructor(...e){this.use(...e)}walkTokens(e,t){let n=[];for(const s of e)switch(n=n.concat(t.call(this,s)),s.type){case"table":{const e=s;for(const s of e.header)n=n.concat(this.walkTokens(s.tokens,t));for(const s of e.rows)for(const e of s)n=n.concat(this.walkTokens(e.tokens,t));break}case"list":{const e=s;n=n.concat(this.walkTokens(e.items,t));break}default:{const e=s;this.defaults.extensions?.childTokens?.[e.type]?this.defaults.extensions.childTokens[e.type].forEach((s=>{n=n.concat(this.walkTokens(e[s],t))})):e.tokens&&(n=n.concat(this.walkTokens(e.tokens,t)))}}return n}use(...e){const t=this.defaults.extensions||{renderers:{},childTokens:{}};return e.forEach((e=>{const n={...e};if(n.async=this.defaults.async||n.async||!1,e.extensions&&(e.extensions.forEach((e=>{if(!e.name)throw new Error("extension name required");if("renderer"in e){const n=t.renderers[e.name];t.renderers[e.name]=n?function(...t){let s=e.renderer.apply(this,t);return!1===s&&(s=n.apply(this,t)),s}:e.renderer}if("tokenizer"in e){if(!e.level||"block"!==e.level&&"inline"!==e.level)throw new Error("extension level must be \'block\' or \'inline\'");const n=t[e.level];n?n.unshift(e.tokenizer):t[e.level]=[e.tokenizer],e.start&&("block"===e.level?t.startBlock?t.startBlock.push(e.start):t.startBlock=[e.start]:"inline"===e.level&&(t.startInline?t.startInline.push(e.start):t.startInline=[e.start]))}"childTokens"in e&&e.childTokens&&(t.childTokens[e.name]=e.childTokens)})),n.extensions=t),e.renderer){const t=this.defaults.renderer||new y(this.defaults);for(const n in e.renderer){const s=e.renderer[n],r=n,i=t[r];t[r]=(...e)=>{let n=s.apply(t,e);return!1===n&&(n=i.apply(t,e)),n||""}}n.renderer=t}if(e.tokenizer){const t=this.defaults.tokenizer||new b(this.defaults);for(const n in e.tokenizer){const s=e.tokenizer[n],r=n,i=t[r];t[r]=(...e)=>{let n=s.apply(t,e);return!1===n&&(n=i.apply(t,e)),n}}n.tokenizer=t}if(e.hooks){const t=this.defaults.hooks||new T;for(const n in e.hooks){const s=e.hooks[n],r=n,i=t[r];T.passThroughHooks.has(n)?t[r]=e=>{if(this.defaults.async)return Promise.resolve(s.call(t,e)).then((e=>i.call(t,e)));const n=s.call(t,e);return i.call(t,n)}:t[r]=(...e)=>{let n=s.apply(t,e);return!1===n&&(n=i.apply(t,e)),n}}n.hooks=t}if(e.walkTokens){const t=this.defaults.walkTokens,s=e.walkTokens;n.walkTokens=function(e){let n=[];return n.push(s.call(this,e)),t&&(n=n.concat(t.call(this,e))),n}}this.defaults={...this.defaults,...n}})),this}setOptions(e){return this.defaults={...this.defaults,...e},this}lexer(e,t){return _.lex(e,t??this.defaults)}parser(e,t){return z.parse(e,t??this.defaults)}#e(e,t){return(n,s)=>{const r={...s},i={...this.defaults,...r};!0===this.defaults.async&&!1===r.async&&(i.silent||console.warn("marked(): The async option was set to true by an extension. The async: false option sent to parse will be ignored."),i.async=!0);const l=this.#t(!!i.silent,!!i.async);if(null==n)return l(new Error("marked(): input parameter is undefined or null"));if("string"!=typeof n)return l(new Error("marked(): input parameter is of type "+Object.prototype.toString.call(n)+", string expected"));if(i.hooks&&(i.hooks.options=i),i.async)return Promise.resolve(i.hooks?i.hooks.preprocess(n):n).then((t=>e(t,i))).then((e=>i.walkTokens?Promise.all(this.walkTokens(e,i.walkTokens)).then((()=>e)):e)).then((e=>t(e,i))).then((e=>i.hooks?i.hooks.postprocess(e):e)).catch(l);try{i.hooks&&(n=i.hooks.preprocess(n));const s=e(n,i);i.walkTokens&&this.walkTokens(s,i.walkTokens);let r=t(s,i);return i.hooks&&(r=i.hooks.postprocess(r)),r}catch(e){return l(e)}}}#t(e,t){return n=>{if(n.message+="\\nPlease report this to https://github.com/markedjs/marked.",e){const e="<p>An error occurred:</p><pre>"+c(n.message+"",!0)+"</pre>";return t?Promise.resolve(e):e}if(t)return Promise.reject(n);throw n}}}const S=new R;function A(e,t){return S.parse(e,t)}A.options=A.setOptions=function(e){return S.setOptions(e),A.defaults=S.defaults,n(A.defaults),A},A.getDefaults=t,A.defaults=e.defaults,A.use=function(...e){return S.use(...e),A.defaults=S.defaults,n(A.defaults),A},A.walkTokens=function(e,t){return S.walkTokens(e,t)},A.parseInline=S.parseInline,A.Parser=z,A.parser=z.parse,A.Renderer=y,A.TextRenderer=$,A.Lexer=_,A.lexer=_.lex,A.Tokenizer=b,A.Hooks=T,A.parse=A;const I=A.options,E=A.setOptions,Z=A.use,q=A.walkTokens,L=A.parseInline,D=A,P=z.parse,v=_.lex;e.Hooks=T,e.Lexer=_,e.Marked=R,e.Parser=z,e.Renderer=y,e.TextRenderer=$,e.Tokenizer=b,e.getDefaults=t,e.lexer=v,e.marked=A,e.options=I,e.parse=D,e.parseInline=L,e.parser=P,e.setOptions=E,e.use=Z,e.walkTokens=q}));\n\nvar _raw = "";\nvar _out = document.getElementById("out");\n\nfunction _addCopyBtns() {\n  document.querySelectorAll("pre:not([data-copy])").forEach(function(pre) {\n    pre.setAttribute("data-copy","1");\n    var btn = document.createElement("button");\n    btn.className = "copy-btn";\n    btn.textContent = "Copy";\n    btn.addEventListener("click", function() {\n      var code = pre.querySelector("code");\n      navigator.clipboard.writeText(code ? code.textContent : pre.textContent);\n      btn.textContent = "✓";\n      setTimeout(function(){ btn.textContent = "Copy"; }, 1200);\n    });\n    pre.appendChild(btn);\n  });\n}\n\nfunction appendChunk(delta) {\n  _raw += delta;\n  _out.innerHTML = marked.parse(_raw);\n  _addCopyBtns();\n  window.scrollTo(0, document.body.scrollHeight);\n}\n\nfunction setRaw(text) {\n  _raw = text;\n  _out.innerHTML = marked.parse(_raw);\n  _addCopyBtns();\n}\n\nfunction clearOutput() {\n  _raw = "";\n  _out.innerHTML = "";\n}\n\nfunction setPlain(text) {\n  _out.innerHTML = "<pre style=\\"white-space:pre-wrap\\">" + text.replace(/&/g,"&amp;").replace(/</g,"&lt;") + "</pre>";\n}\n</script>\n</body>\n</html>'


_PROMPT_LIBRARY: dict[str, list[str]] = {
    "Coding": [
        "Implement a thread-safe LRU cache in Python with O(1) get and put. Show the full class with tests.",
        "Write a Go function that parses a semi-structured log line (timestamp, level, message, optional key=value pairs) into a struct. Handle malformed input gracefully.",
        "Explain the difference between Go's sync.Mutex and sync.RWMutex. When would you use each? Show a concrete example where the wrong choice causes a bug.",
        "Write a Python decorator that retries a function up to N times with exponential backoff, logging each attempt. Make it work for both sync and async functions.",
        "Design a simple event bus in TypeScript with typed events, subscribe/unsubscribe, and one-time listeners. No external dependencies.",
    ],
    "Reasoning": [
        "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost? Show your reasoning step by step, then verify your answer.",
        "You have 12 balls, one of which is slightly heavier or lighter than the others. Using a balance scale exactly 3 times, identify the odd ball and whether it's heavier or lighter.",
        "Three logicians walk into a bar. The bartender asks 'Does everyone want a drink?' The first says 'I don't know.' The second says 'I don't know.' The third says 'Yes.' Explain why.",
        "A snail climbs a 10-meter pole. Each day it climbs 3 meters, each night it slides back 2 meters. On which day does it reach the top? Show your work.",
        "Explain the Monty Hall problem. Why does switching doors improve your odds? Give both an intuitive explanation and a mathematical proof.",
    ],
    "Architecture": [
        "Design a distributed rate limiter for a high-traffic API gateway handling 500k req/s across 20 nodes. Cover algorithm choice, clock skew, partition behavior, and memory footprint.",
        "You're designing a chat system for 10 million concurrent users. Walk through your data model, message delivery guarantees, presence tracking, and how you'd handle a thundering herd on reconnect.",
        "Compare event sourcing + CQRS vs traditional CRUD for a financial ledger. What are the real tradeoffs in practice, not just theory? When would you NOT use event sourcing?",
        "Explain the CAP theorem with a concrete example for each of the three impossible combinations. Which real databases fall into each category and why?",
        "Design the schema and query patterns for a social graph where you need to find mutual friends within 2 hops for 100M users in under 50ms.",
    ],
    "Explanation": [
        "Explain how transformers work from first principles — attention, positional encoding, and why the architecture replaced RNNs. Assume I understand linear algebra but not ML.",
        "What actually happens when you type a URL in a browser and hit Enter? Go as deep as you can — DNS, TCP, TLS, HTTP, rendering pipeline.",
        "Explain memory ordering in modern CPUs. What is the difference between acquire/release semantics and sequential consistency? Why does this matter in lock-free programming?",
        "How does garbage collection work in Go? Explain the tri-color mark-and-sweep algorithm, write barriers, and how the GC pacer decides when to run.",
        "Explain how Linux handles a page fault from the moment the CPU traps to when the process resumes. Cover demand paging, swap, and OOM.",
    ],
    "Creative": [
        "Write the opening chapter of a hard sci-fi novel where the protagonist discovers that the universe's physical constants are slowly drifting. The tone should be literary, not pulpy.",
        "Write a dialogue between a senior engineer and a junior engineer reviewing code. The senior engineer is trying to explain why the junior's clever solution is actually worse than the boring one.",
        "Write a short story told entirely through git commit messages, Slack messages, and error logs. Something goes badly wrong.",
        "Write a technical post-mortem for a fictional but plausible production outage. Include timeline, root cause, contributing factors, and action items. Make it feel real.",
        "Write a children's book explanation of how the internet works, suitable for a 6-year-old. No analogies involving roads or highways — find something more creative.",
    ],
}

def _load_prompt_history() -> list[str]:
    try:
        return json.loads(_PROMPT_HISTORY_PATH.read_text())
    except Exception:
        return []


def _save_prompt_history(history: list[str], prompt: str) -> list[str]:
    history = [p for p in history if p != prompt]   # dedupe
    history.insert(0, prompt)
    history = history[:_PROMPT_HISTORY_MAX]
    try:
        _PROMPT_HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n")
    except Exception:
        pass
    return history


def _update_spark_chart(spark_view, history):
    """Draw tok/s history as simple colored NSBox bars inside spark_view (NSView)."""
    from AppKit import NSBox, NSColor
    # Remove all subviews
    for sv in list(spark_view.subviews()):
        sv.removeFromSuperview()
    if not history:
        return
    n = len(history)
    bounds = spark_view.bounds()
    W = bounds.size.width
    H = bounds.size.height
    max_val = max(history) or 1.0
    gap = 2
    bar_w = max(4, (W - gap * (n - 1)) / n)
    color = NSColor.systemGreenColor()
    for i, val in enumerate(history):
        bar_h = max(2, (val / max_val) * (H - 2))
        x = i * (bar_w + gap)
        box = NSBox.alloc().initWithFrame_(((x, 0), (bar_w, bar_h)))
        box.setBoxType_(0)  # NSBoxPrimary (filled)
        box.setFillColor_(color)
        box.setBorderColor_(NSColor.clearColor())
        box.setTitlePosition_(0)  # no title
        box.setBorderWidth_(0)
        spark_view.addSubview_(box)
    spark_view.setNeedsDisplay_(True)


class _SuggestionTableSource(NSObject):
    """NSTableView data source/delegate for generated prompt suggestions."""

    def numberOfRowsInTableView_(self, tv):
        return len(self._suggestions)

    def tableView_objectValueForTableColumn_row_(self, tv, col, row):
        return self._suggestions[row] if row < len(self._suggestions) else ""

    def tableView_shouldSelectRow_(self, tv, row):
        return True


class _TestPromptHandler(NSObject):
    """Action target for the non-modal Quick Test Prompt window."""

    # ── Prompt history (up/down arrow navigation) ─────────────────────────────

    def textView_doCommandBySelector_(self, textView, sel):
        name = sel if isinstance(sel, str) else str(sel)
        # Enter (no Shift) → submit; Shift+Enter → let NSTextView insert newline normally
        if "insertNewline" in name:
            from AppKit import NSApp
            mods = NSApp.currentEvent().modifierFlags() if NSApp.currentEvent() else 0
            SHIFT = 1 << 17   # NSEventModifierFlagShift
            if mods & SHIFT:
                return False   # default: NSTextView inserts newline
            self.send_(None)
            return True
        if "moveUp" in name:
            hist = self._history
            if not hist:
                return False
            if self._history_idx == -1:
                self._history_saved = self._input_fld.string()
                self._history_idx = 0
            elif self._history_idx < len(hist) - 1:
                self._history_idx += 1
            self._input_fld.setString_(hist[self._history_idx])
            return True
        if "moveDown" in name:
            if self._history_idx == -1:
                return False
            if self._history_idx == 0:
                self._history_idx = -1
                self._input_fld.setString_(getattr(self, "_history_saved", ""))
            else:
                self._history_idx -= 1
                self._input_fld.setString_(self._history[self._history_idx])
            return True
        return False

    def send_(self, _s):
        prompt = self._input_fld.string().strip()
        if not prompt or self._streaming:
            return
        if self._app_ref and getattr(self._app_ref, '_loading', False):
            if getattr(self, '_output_wv', None):
                self._wv_plain(self._output_wv, "⏳ Model is loading…")
            self._pending_prompt = prompt
            self._load_wait_timer = rumps.Timer(self._waitForLoadThenSend_, 0.5)
            self._load_wait_timer.start()
            return
        self._history = _save_prompt_history(self._history, prompt)
        self._history_idx = -1
        if getattr(self, '_output_wv', None):
            self._wv_plain(self._output_wv, "⏳ Generating…")
        self._first_chunk = True
        self._raw_output = ""
        self._raw_output2 = ""
        self._tps_lbl.setStringValue_("…")
        self._streaming = True
        self._buf: list[str] = []
        self._t0 = time.time()
        self._t_end = None
        self._tok_count = 0
        self._t_first = None
        self._ttft_shown = False
        self._prompt_tokens = 0
        self._ctx_max = 0
        # Show spinner until first token arrives
        if hasattr(self, '_spinner'):
            self._spinner.setHidden_(False)
            self._spinner.startAnimation_(None)
        # Update model 1 label to reflect current active model
        if hasattr(self, '_model1_lbl') and self._app_ref:
            self._model1_lbl.setStringValue_(self._app_ref._active or "Model 1")
        self._drain_timer = rumps.Timer(self.drainTick_, 0.1)
        self._drain_timer.start()
        threading.Thread(target=self._do_stream,
                         args=(prompt,), daemon=True).start()
        self._prompt_for_compare = prompt
        self._sequential_phase = None
        if hasattr(self, '_buf2'):
            self._buf2.clear()
            self._streaming2 = False
            self._tok_count2 = 0
            self._t_first2 = None
            self._ttft_shown2 = False
            self._t02 = time.time()
            self._t_end2 = None
            wv2 = getattr(self, '_output_wv2', None)
            if wv2:
                self._wv_clear(wv2)
            if hasattr(self, '_tps_lbl2'):
                self._tps_lbl2.setStringValue_("")
            compare_on = (hasattr(self, '_compare_chk') and self._compare_chk.state()
                          and hasattr(self, '_model2_popup'))
            if compare_on:
                model2_name = self._model2_popup.titleOfSelectedItem()
                if model2_name:
                    self._model2_name = model2_name
                    self._streaming2 = True
                    self._sequential_phase = "waiting_m1"
                    if wv2:
                        self._wv_plain(wv2, "⏳ Waiting for Model 1 to finish…")

    def _do_stream(self, prompt):
        try:
            cfg = self._app_ref._cfg
            kind = self._app_ref._model_kind(self._app_ref._active or "")
            port, api_key = active_api(cfg, kind)
            base_url = active_base_url(cfg, kind)
            model = self._app_ref._active or ""
            p = self._app_ref._model_params(model) if hasattr(self._app_ref, "_model_params") else {}
            self._prompt_tokens = 0
            self._ctx_max = (p.get("context", 32768) if p else 32768)
            # Inline param controls override per-model defaults when params row is open
            qt_temp = cfg.get("qt_temperature", p.get("temperature", 0.7) if p else 0.7)
            qt_max  = cfg.get("qt_max_tokens",  p.get("max_tokens",  512)  if p else 512)
            body = json.dumps({
                "model": model,
                "messages": self._build_messages(prompt),
                "max_tokens": qt_max,
                "temperature": qt_temp,
                "stream": True,
                "stream_options": {"include_usage": True},
            }).encode()
            req = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
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
            self._t_end = time.time()
            # Record completed assistant reply in conversation history
            full_reply = getattr(self, '_raw_output', '') + "".join(self._buf)
            self._record_assistant_reply(full_reply)

    # ── WKWebView JS helpers ───────────────────────────────────────────────────

    def _wv_clear(self, wv):
        """Clear WKWebView output."""
        wv.evaluateJavaScript_completionHandler_("clearOutput()", None)

    def _wv_plain(self, wv, msg):
        """Show a plain status message in WKWebView (no markdown parse)."""
        safe = msg.replace("\\", "\\\\").replace("`", "\\`")
        wv.evaluateJavaScript_completionHandler_(f"setPlain(`{safe}`)", None)

    def _wv_append(self, wv, chunk):
        """Append a raw text chunk — WKWebView re-renders markdown each call."""
        import json as _json
        js = f"appendChunk({_json.dumps(chunk)})"
        wv.evaluateJavaScript_completionHandler_(js, None)

    # ── Drain timer ────────────────────────────────────────────────────────────

    def drainTick_(self, timer):
        wv = getattr(self, '_output_wv', None)
        wv2 = getattr(self, '_output_wv2', None)

        # Drain primary buffer
        if self._buf and wv:
            chunk = "".join(self._buf)
            self._buf.clear()
            if getattr(self, '_first_chunk', False):
                self._wv_clear(wv)
                self._first_chunk = False
            self._raw_output = getattr(self, '_raw_output', '') + chunk
            self._wv_append(wv, chunk)

        # Stop spinner on first token
        if self._t_first is not None and hasattr(self, '_spinner') and not self._spinner.isHidden():
            self._spinner.stopAnimation_(None)
            self._spinner.setHidden_(True)

        # Sequential compare state machine
        phase = getattr(self, '_sequential_phase', None)
        self._cmp_tick = getattr(self, '_cmp_tick', 0) + 1
        _flash_on = (self._cmp_tick % 20) < 12   # 1.2s on, 0.8s off

        def _wv2_status(msg):
            if not wv2:
                return
            if _flash_on:
                self._wv_plain(wv2, msg)
            # on off-ticks just leave the text as-is (dimming effect via opacity would need CSS)

        if phase == "waiting_m1" and not self._streaming and not self._buf:
            model2_name = getattr(self, '_model2_name', None)
            if model2_name and self._app_ref:
                self._sequential_phase = "loading"
                self._cmp_tick = 0
                if wv2:
                    self._wv_plain(wv2, f"⏳ Switching to {model2_name}…")
                self._app_ref._load_model_by_name(model2_name, skip_memory_check=True)
        elif phase == "waiting_m1":
            _wv2_status("⏳ Waiting for Model 1 to finish…")
        elif phase == "loading":
            app = self._app_ref
            model2_name = getattr(self, '_model2_name', None)
            if app and not app._loading:
                self._sequential_phase = "streaming2"
                self._cmp_tick = 0
                self._tok_count2 = 0
                self._t_first2 = None
                self._ttft_shown2 = False
                self._t02 = time.time()
                self._first_chunk2 = True
                if wv2:
                    self._wv_plain(wv2, "⏳ Generating…")
                if hasattr(self, '_tps_lbl2'):
                    self._tps_lbl2.setStringValue_("…")
                prompt = getattr(self, '_prompt_for_compare', "")
                threading.Thread(target=self._do_stream2,
                                 args=(prompt, model2_name), daemon=True).start()
            elif app and app._loading:
                status = getattr(app, '_load_status', 'Loading…')
                estimate = getattr(app, '_load_time_estimate', None)
                elapsed = time.time() - (app._load_start_time or time.time())
                if estimate and estimate > elapsed:
                    status = f"{status}  (~{int(estimate - elapsed)}s remaining)"
                elif estimate:
                    status = f"{status}  (taking longer than usual)"
                _wv2_status(f"⏳ {status}")

        # Drain compare buffer
        if hasattr(self, '_buf2') and self._buf2 and wv2:
            chunk2 = "".join(self._buf2)
            self._buf2.clear()
            if getattr(self, '_first_chunk2', False):
                self._wv_clear(wv2)
                self._first_chunk2 = False
            self._raw_output2 = getattr(self, '_raw_output2', '') + chunk2
            self._wv_append(wv2, chunk2)

        # Stats bar — model 1
        elapsed = (self._t_end or time.time()) - self._t0
        if self._tok_count > 0 and elapsed > 0 and not getattr(self, '_showing_load_status', False):
            ttft_part = ""
            if self._t_first is not None:
                ttft_ms = (self._t_first - self._t0) * 1000
                ttft_part = f"TTFT {ttft_ms:.0f}ms  |  "
                self._ttft_shown = True
            ctx_part = ""
            prompt_toks = getattr(self, '_prompt_tokens', 0)
            ctx_max = getattr(self, '_ctx_max', 0)
            if prompt_toks > 0 and ctx_max > 0:
                used = prompt_toks + self._tok_count
                pct = used / ctx_max * 100
                ctx_part = f"  |  ctx {used:,}/{ctx_max:,} ({pct:.0f}%)"
                if self._app_ref:
                    self._app_ref._ctx_used = used
                    self._app_ref._ctx_max = ctx_max
                    self._app_ref._update_title()
            self._tps_lbl.setStringValue_(
                f"{ttft_part}{self._tok_count / elapsed:.1f} tok/s  ({self._tok_count} tokens){ctx_part}")

        # Stats bar — model 2
        if hasattr(self, '_tps_lbl2'):
            elapsed2 = (getattr(self, '_t_end2', None) or time.time()) - getattr(self, '_t02', self._t0)
            tok2 = getattr(self, '_tok_count2', 0)
            t_first2 = getattr(self, '_t_first2', None)
            t02 = getattr(self, '_t02', self._t0)
            if tok2 > 0 and elapsed2 > 0:
                ttft2_part = ""
                if t_first2 is not None:
                    ttft2_ms = (t_first2 - t02) * 1000
                    ttft2_part = f"TTFT {ttft2_ms:.0f}ms  |  "
                    self._ttft_shown2 = True
                self._tps_lbl2.setStringValue_(
                    f"{ttft2_part}{tok2 / elapsed2:.1f} tok/s  ({tok2} tokens)")

        streaming2 = getattr(self, '_streaming2', False)
        buf2_empty = not getattr(self, '_buf2', [])
        if not self._streaming and not self._buf and not streaming2 and buf2_empty:
            timer.stop()
            phase = getattr(self, '_sequential_phase', None)
            if phase == "streaming2" and getattr(self, '_model2_name', None):
                self._sequential_phase = None
                self._save_compare_result()

    def clear_(self, _s):
        if getattr(self, '_output_wv', None):
            self._wv_clear(self._output_wv)
        if getattr(self, '_output_wv2', None):
            self._wv_clear(self._output_wv2)
        self._tps_lbl.setStringValue_("")
        self._input_fld.setString_("")
        self._raw_output = ""
        self._raw_output2 = ""
        # Also reset conversation if in chat mode
        self._messages = []

    def copyResponse_(self, _s):
        # Copy raw markdown text (accumulated in _raw_output)
        from AppKit import NSPasteboard, NSStringPboardType
        text = getattr(self, '_raw_output', "") or ""
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSStringPboardType)

    def fontLarger_(self, _s):
        self._output_font_size = min(self._output_font_size + 1, 28)
        self._apply_font()
        if self._app_ref:
            self._app_ref._cfg["qt_font_size"] = self._output_font_size
            save_config(self._app_ref._cfg)

    def fontSmaller_(self, _s):
        self._output_font_size = max(self._output_font_size - 1, 8)
        self._apply_font()
        if self._app_ref:
            self._app_ref._cfg["qt_font_size"] = self._output_font_size
            save_config(self._app_ref._cfg)

    def _apply_font(self):
        # Font size via JS CSS injection into WKWebViews
        size = self._output_font_size
        js = f"document.body.style.fontSize='{size}px'"
        if getattr(self, '_output_wv', None):
            self._output_wv.evaluateJavaScript_completionHandler_(js, None)
        if getattr(self, '_output_wv2', None):
            self._output_wv2.evaluateJavaScript_completionHandler_(js, None)

    def exportSession_(self, _s):
        from AppKit import NSSavePanel
        prompt = self._input_fld.string().strip()
        response = getattr(self, '_raw_output', "") or ""
        if not prompt and not response:
            return
        panel = NSSavePanel.savePanel()
        panel.setTitle_("Export Quick Test Session")
        panel.setNameFieldStringValue_("quick-test.md")
        panel.setAllowedFileTypes_(["md"])
        NSApp.activateIgnoringOtherApps_(True)
        if panel.runModal() == NSModalResponseOK:
            path = Path(panel.URL().path())
            md = f"## Prompt\n\n{prompt}\n\n## Response\n\n{response}\n"
            path.write_text(md)

    def _loadStatusTick_(self, timer):
        """Continuously poll app._loading and show status + estimate in tps_lbl."""
        from AppKit import NSColor, NSForegroundColorAttributeName, NSAttributedString
        app = self._app_ref
        if not app:
            return
        if not app._loading or self._streaming:
            # Clear load status if we were showing it
            if getattr(self, '_showing_load_status', False):
                self._tps_lbl.setStringValue_("")
                self._showing_load_status = False
            return
        self._showing_load_status = True
        status = getattr(app, '_load_status', 'Loading…')
        estimate = getattr(app, '_load_time_estimate', None)
        if estimate and app._load_start_time:
            elapsed = time.time() - app._load_start_time
            if estimate > elapsed:
                remaining = int(estimate - elapsed)
                status = f"{status}  (~{remaining}s remaining)"
            else:
                status = f"{status}  (taking longer than usual)"
        self._load_flash = not getattr(self, '_load_flash', False)
        color = NSColor.labelColor() if self._load_flash else NSColor.secondaryLabelColor()
        astr = NSAttributedString.alloc().initWithString_attributes_(
            f"⏳ {status}", {NSForegroundColorAttributeName: color})
        self._tps_lbl.setAttributedStringValue_(astr)

    def _waitForLoadThenSend_(self, timer):
        app = self._app_ref
        if app and app._loading:
            status = getattr(app, '_load_status', 'Loading…')
            estimate = getattr(app, '_load_time_estimate', None)
            elapsed = time.time() - (app._load_start_time or time.time())
            if estimate and estimate > elapsed:
                remaining = int(estimate - elapsed)
                status = f"{status}  (~{remaining}s remaining)"
            elif estimate:
                status = f"{status}  (taking longer than usual)"
            self._flash_tick = not getattr(self, '_flash_tick', False)
            if getattr(self, '_output_wv', None):
                self._wv_plain(self._output_wv, f"⏳ {status}")
            return
        timer.stop()
        self._load_wait_timer = None
        self._flash_tick = False
        prompt = getattr(self, '_pending_prompt', None)
        self._pending_prompt = None
        if prompt:
            self._input_fld.setString_(prompt)
            self.send_(None)

    def model2PopupChanged_(self, sender):
        if hasattr(self, '_model2_header_lbl'):
            self._model2_header_lbl.setStringValue_(
                sender.titleOfSelectedItem() or "Model 2")

    def compareChanged_(self, sender):
        on = bool(sender.state())
        if hasattr(self, '_model2_popup'):
            self._model2_popup.setHidden_(not on)
        if hasattr(self, '_model2_lbl'):
            self._model2_lbl.setHidden_(not on)
        if getattr(self, '_output_wv2', None):
            self._output_wv2.setHidden_(not on)
        if hasattr(self, '_model2_header_lbl'):
            self._model2_header_lbl.setHidden_(not on)
            if on and hasattr(self, '_model2_popup'):
                self._model2_header_lbl.setStringValue_(
                    self._model2_popup.titleOfSelectedItem() or "Model 2")
        if hasattr(self, '_tps_lbl2'):
            self._tps_lbl2.setHidden_(not on)
        if hasattr(self, '_vdiv'):
            self._vdiv.setHidden_(not on)

    # ── Inline param controls ──────────────────────────────────────────────────

    _PARAM_VIEWS = ('_temp_lbl', '_temp_slider', '_temp_val_lbl',
                    '_maxtok_lbl', '_maxtok_fld', '_maxtok_step',
                    '_tok_counter_lbl', '_preset_popup')

    def paramsToggled_(self, sender):
        on = bool(sender.state())
        for attr in self._PARAM_VIEWS:
            v = getattr(self, attr, None)
            if v:
                v.setHidden_(not on)
        # Start/stop token counter timer
        if on:
            if not getattr(self, '_tok_ctr_timer', None):
                self._tok_ctr_timer = rumps.Timer(self._updateTokenCounter_, 0.5)
                self._tok_ctr_timer.start()
        else:
            t = getattr(self, '_tok_ctr_timer', None)
            if t:
                t.stop()
                self._tok_ctr_timer = None

    def _updateTokenCounter_(self, timer):
        lbl = getattr(self, '_tok_counter_lbl', None)
        if not lbl or lbl.isHidden():
            return
        text = self._input_fld.string() if hasattr(self, '_input_fld') else ""
        approx = max(1, int(len(text) / 3.5))
        lbl.setStringValue_(f"~{approx} tokens")

    def tempChanged_(self, sender):
        val = round(sender.floatValue(), 2)
        lbl = getattr(self, '_temp_val_lbl', None)
        if lbl:
            lbl.setStringValue_(f"{val:.1f}")
        if self._app_ref:
            self._app_ref._cfg["qt_temperature"] = val
            save_config(self._app_ref._cfg)
        # Any manual change → set preset to Custom
        pp = getattr(self, '_preset_popup', None)
        if pp:
            pp.selectItemWithTitle_("Custom")

    def maxTokensChanged_(self, sender):
        val = int(sender.intValue())
        fld = getattr(self, '_maxtok_fld', None)
        if fld:
            fld.setStringValue_(str(val))
        if self._app_ref:
            self._app_ref._cfg["qt_max_tokens"] = val
            save_config(self._app_ref._cfg)
        pp = getattr(self, '_preset_popup', None)
        if pp:
            pp.selectItemWithTitle_("Custom")

    def presetChanged_(self, sender):
        preset = sender.titleOfSelectedItem()
        if self._app_ref:
            self._app_ref._cfg["qt_preset"] = preset
            save_config(self._app_ref._cfg)
        presets = {
            "Precise":  (0.2, 512),
            "Balanced": (0.7, 1024),
            "Creative": (1.2, 2048),
        }
        if preset in presets:
            temp, maxtok = presets[preset]
            sl = getattr(self, '_temp_slider', None)
            if sl:
                sl.setFloatValue_(temp)
            vl = getattr(self, '_temp_val_lbl', None)
            if vl:
                vl.setStringValue_(f"{temp:.1f}")
            st = getattr(self, '_maxtok_step', None)
            if st:
                st.setIntValue_(maxtok)
            fl = getattr(self, '_maxtok_fld', None)
            if fl:
                fl.setStringValue_(str(maxtok))
            if self._app_ref:
                self._app_ref._cfg["qt_temperature"] = temp
                self._app_ref._cfg["qt_max_tokens"] = maxtok
                save_config(self._app_ref._cfg)

    # ── Conversation mode ──────────────────────────────────────────────────────

    def chatToggled_(self, sender):
        on = bool(sender.state())
        nb = getattr(self, '_new_conv_btn', None)
        if nb:
            nb.setHidden_(not on)
        for attr in ('_sys_prompt_lbl', '_sys_prompt_scroll'):
            v = getattr(self, attr, None)
            if v:
                v.setHidden_(not on)
        if on and not getattr(self, '_messages', None):
            self._messages = []
        elif not on:
            self._messages = []

    def newConversation_(self, _s):
        self._messages = []
        wv = getattr(self, '_output_wv', None)
        if wv:
            self._wv_clear(wv)
        self._tps_lbl.setStringValue_("")
        self._raw_output = ""

    def _get_system_prompt(self):
        tv = getattr(self, '_sys_prompt_tv', None)
        if tv:
            return tv.string().strip()
        return self._app_ref._cfg.get("qt_system_prompt", "") if self._app_ref else ""

    def _build_messages(self, prompt):
        """Return messages list for current mode (single or conversation)."""
        chat_on = getattr(self, '_chat_chk', None) and self._chat_chk.state()
        sys_prompt = self._get_system_prompt()
        if not chat_on:
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": prompt})
            return msgs
        # Conversation mode — accumulate history
        if not hasattr(self, '_messages') or self._messages is None:
            self._messages = []
        # Prepend system prompt if set (replace existing system msg)
        history = [m for m in self._messages if m["role"] != "system"]
        if sys_prompt:
            history = [{"role": "system", "content": sys_prompt}] + history
        history.append({"role": "user", "content": prompt})
        return history

    def _record_assistant_reply(self, content):
        """After streaming finishes, record the assistant turn in chat history."""
        chat_on = getattr(self, '_chat_chk', None) and self._chat_chk.state()
        if not chat_on:
            return
        if not hasattr(self, '_messages') or self._messages is None:
            self._messages = []
        prompt = getattr(self, '_prompt_for_compare', "")
        # Add user + assistant turns
        sys_prompt = self._get_system_prompt()
        history = [m for m in self._messages if m["role"] != "system"]
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": content})
        if sys_prompt:
            self._messages = [{"role": "system", "content": sys_prompt}] + history
        else:
            self._messages = history
        # Save sys prompt to config
        if self._app_ref and sys_prompt:
            self._app_ref._cfg["qt_system_prompt"] = sys_prompt
            save_config(self._app_ref._cfg)

    def _save_compare_result(self):
        import datetime
        prompt = getattr(self, '_prompt_for_compare', "")
        model1 = (self._app_ref._active if self._app_ref else None) or "Model 1"
        model2 = getattr(self, '_model2_name', "Model 2")
        response1 = getattr(self, '_raw_output', "") or ""
        response2 = getattr(self, '_raw_output2', "") or ""
        elapsed1 = (getattr(self, '_t_end', None) or self._t0) - self._t0
        elapsed2 = (getattr(self, '_t_end2', None) or getattr(self, '_t02', self._t0)) - getattr(self, '_t02', self._t0)
        tok1 = self._tok_count
        tok2 = getattr(self, '_tok_count2', 0)
        entry = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt": prompt,
            "model1": model1,
            "model2": model2,
            "response1": response1,
            "response2": response2,
            "tps1": round(tok1 / elapsed1, 1) if elapsed1 > 0 and tok1 > 0 else 0,
            "tps2": round(tok2 / elapsed2, 1) if elapsed2 > 0 and tok2 > 0 else 0,
            "tokens1": tok1,
            "tokens2": tok2,
        }
        try:
            _COMPARE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            history = json.loads(_COMPARE_HISTORY_PATH.read_text()) \
                if _COMPARE_HISTORY_PATH.exists() else []
            history.insert(0, entry)
            _COMPARE_HISTORY_PATH.write_text(json.dumps(history[:50], indent=2) + "\n")
        except Exception:
            pass

    def promptLibrarySelected_(self, sender):
        """Fill input from a library prompt menu item."""
        text = sender.representedObject() or sender.title()
        if text:
            self._input_fld.setString_(str(text))

    def generateSuggestions_(self, _s):
        """Ask the loaded model to generate 5 test prompts, then show a picker."""
        if not self._app_ref:
            return
        cfg = self._app_ref._cfg
        model = self._app_ref._active or ""
        if not model:
            return
        kind = self._app_ref._model_kind(model)
        port, api_key = active_api(cfg, kind)
        base_url = active_base_url(cfg, kind)
        # Show a small status panel while generating
        from AppKit import (NSPanel, NSTextField as _NSTextField2,
                            NSProgressIndicator as _NSPIg)
        gwin = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (340, 80)), 15, 2, False)
        gwin.setTitle_("Generating suggestions…")
        gwin.center()
        gcv = _vibrancy_content_view(gwin)
        glbl = _NSTextField2.alloc().initWithFrame_(((_PAD, 28), (300, 22)))
        glbl.setBezeled_(False); glbl.setDrawsBackground_(False)
        glbl.setEditable_(False)
        glbl.setStringValue_("Asking the model for prompt ideas…")
        gcv.addSubview_(glbl)
        gspin = _NSPIg.alloc().initWithFrame_(((300, 30), (16, 16)))
        gspin.setStyle_(1); gspin.startAnimation_(None)
        gcv.addSubview_(gspin)
        gwin.makeKeyAndOrderFront_(None)
        self._gen_win = gwin

        threading.Thread(target=self._do_generate_suggestions,
                         args=(port, api_key, base_url, model, gwin), daemon=True).start()

    def _do_generate_suggestions(self, port, api_key, base_url, model, status_win):
        meta = (
            "Generate exactly 5 diverse, interesting test prompts for evaluating a large language model. "
            "Return ONLY a numbered list (1. ... 2. ... etc.), one prompt per line, no extra commentary. "
            "Make them varied: include coding, reasoning, explanation, and creative tasks. "
            "Each prompt should be self-contained and 1-3 sentences."
        )
        suggestions = []
        try:
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": meta}],
                "max_tokens": 600,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            for line in text.splitlines():
                line = line.strip()
                if line and line[0].isdigit() and ". " in line:
                    suggestions.append(line.split(". ", 1)[1].strip())
        except Exception:
            pass
        # Back on main thread via timer
        self._gen_suggestions = suggestions
        self._gen_status_win = status_win
        t = rumps.Timer(self._showSuggestionsPicker_, 0)
        t.start()

    def _showSuggestionsPicker_(self, timer):
        timer.stop()
        status_win = getattr(self, '_gen_status_win', None)
        if status_win:
            status_win.orderOut_(None)
        suggestions = getattr(self, '_gen_suggestions', [])
        if not suggestions:
            return
        from AppKit import (NSPanel, NSTableView, NSScrollView as _NSScrollViewP,
                            NSTableColumn)
        PW, PH = 560, 260
        picker = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (PW, PH)), 15, 2, False)
        picker.setTitle_("Generated Prompt Suggestions")
        picker.center()
        pcv = _vibrancy_content_view(picker)
        # Label
        lbl = _lbl("Select a prompt to use:", ((_PAD, PH - _PAD - 20), (PW - _PAD*2, 20)))
        pcv.addSubview_(lbl)
        # Table
        tbl_scroll = _NSScrollViewP.alloc().initWithFrame_(
            ((_PAD, 44), (PW - _PAD*2, PH - _PAD - 20 - _GAP - 44)))
        tbl_scroll.setHasVerticalScroller_(True)
        tbl = NSTableView.alloc().initWithFrame_(((0, 0), (PW - _PAD*2, 200)))
        col = NSTableColumn.alloc().initWithIdentifier_("prompt")
        col.setWidth_(PW - _PAD*2 - 20)
        col.headerCell().setStringValue_("Prompt")
        tbl.addTableColumn_(col)
        tbl.setUsesAlternatingRowBackgroundColors_(True)
        tbl.setRowHeight_(36)
        tbl_scroll.setDocumentView_(tbl)
        pcv.addSubview_(tbl_scroll)
        # Use button
        use_btn = _btn("Use This Prompt", self, "useSuggestion:",
                       ((PW - _PAD - 140, 10), (140, 24)))
        pcv.addSubview_(use_btn)
        # Wire data source via simple delegate object
        ds = _SuggestionTableSource.alloc().init()
        ds._suggestions = suggestions
        ds._handler = self
        ds._picker = picker
        tbl.setDataSource_(ds)
        tbl.setDelegate_(ds)
        tbl.reloadData()
        self._suggestion_table = tbl
        self._suggestion_ds = ds  # keep alive
        self._suggestion_picker = picker
        picker.makeKeyAndOrderFront_(None)

    def useSuggestion_(self, _s):
        tbl = getattr(self, '_suggestion_table', None)
        ds = getattr(self, '_suggestion_ds', None)
        if tbl is None or ds is None:
            return
        row = tbl.selectedRow()
        if row >= 0 and row < len(ds._suggestions):
            self._input_fld.setString_(ds._suggestions[row])
        picker = getattr(self, '_suggestion_picker', None)
        if picker:
            picker.orderOut_(None)

    def _do_stream2(self, prompt, model2_name):
        import urllib.error as _ue
        try:
            cfg = self._app_ref._cfg
            kind2 = self._app_ref._model_kind(model2_name)
            port, api_key = active_api(cfg, kind2)
            base_url = active_base_url(cfg, kind2)
            p = self._app_ref._model_params(model2_name) if hasattr(self._app_ref, "_model_params") else {}
            qt_temp = cfg.get("qt_temperature", p.get("temperature", 0.7) if p else 0.7)
            qt_max  = cfg.get("qt_max_tokens",  p.get("max_tokens",  512)  if p else 512)
            body = json.dumps({
                "model": model2_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": qt_max,
                "temperature": qt_temp,
                "stream": True,
            }).encode()
            req = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
            )
            # Retry on 503 — server may still be warming up after load reports done
            max_retries, delay = 8, 3.0
            for attempt in range(max_retries):
                try:
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
                    break  # success
                except _ue.HTTPError as e:
                    if e.code == 503 and attempt < max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 1.5, 10.0)
                    else:
                        raise
        except Exception as e:
            self._buf2.append(f"\n\n[Error: {e}]")
        finally:
            self._streaming2 = False
            self._t_end2 = time.time()

    def windowShouldClose_(self, sender):
        """Hide instead of close — avoids dealloc/timer teardown crashes."""
        sender.orderOut_(None)
        return False  # prevent NSWindow from actually closing


def _make_test_prompt_window(app) -> NSWindow:
    """Build and return the Quick Test Prompt NSWindow (non-modal)."""
    W, H = 760, 520
    BOT = 90   # bottom margin: BTN_Y=12 (buttons h=30), STAT_Y=50 (stats h=26)
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    win.setTitle_("Quick Test Prompt")
    win.center()
    _constrain_to_screen(win)
    win.setTitlebarAppearsTransparent_(True)
    win.setMovableByWindowBackground_(True)
    win.setMinSize_((560, 320))
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
    # Prompt history
    handler._history = _load_prompt_history()
    handler._history_idx = -1
    handler._history_saved = ""
    # handler kept alive by caller storing it alongside the window

    INPUT_H = 64   # multi-line input height (~4 lines)
    CHK_H = 22     # compare row control height
    ROW2_Y = H - _PAD - INPUT_H - _GAP - CHK_H  # y for second row (checkbox row)
    # Autoresizing mask constants
    _TOP   = 8   # NSViewMinYMargin  — sticks to top
    _GROW  = 16  # NSViewHeightSizable — grows with window height
    _WGROW = 2   # NSViewWidthSizable  — grows with window width

    prompt_lbl = _lbl("Prompt:", ((_PAD, H - _PAD - INPUT_H + (INPUT_H - 18) // 2), (_LW, 18)),
                       right=False)
    prompt_lbl.setAutoresizingMask_(_TOP)
    cv.addSubview_(prompt_lbl)

    input_scroll = NSScrollView.alloc().initWithFrame_(
        ((_PAD + _LW + _GAP, H - _PAD - INPUT_H),
         (W - _PAD*2 - _LW - _GAP - 70, INPUT_H)))
    input_scroll.setHasVerticalScroller_(True)
    input_scroll.setAutohidesScrollers_(True)
    input_scroll.setBorderType_(2)  # NSBezelBorder
    input_scroll.setAutoresizingMask_(_TOP | _WGROW)
    input_tv = NSTextView.alloc().initWithFrame_(
        ((0, 0), (W - _PAD*2 - _LW - _GAP - 70, INPUT_H)))
    input_tv.setFont_(NSFont.systemFontOfSize_(13))
    input_tv.setDelegate_(handler)
    input_tv.setRichText_(False)
    input_tv.setAutomaticQuoteSubstitutionEnabled_(False)
    input_tv.setAutomaticDashSubstitutionEnabled_(False)
    input_scroll.setDocumentView_(input_tv)
    cv.addSubview_(input_scroll)
    handler._input_fld = input_tv

    send_btn = _btn("Send", handler, "send:",
                    ((W - _PAD - 64, H - _PAD - INPUT_H), (64, INPUT_H)), "\r")
    send_btn.setAutoresizingMask_(_TOP | 1)  # 1=NSViewMinXMargin (right-anchored)
    cv.addSubview_(send_btn)

    # Compare checkbox
    compare_chk = NSButton.alloc().initWithFrame_(
        ((_PAD, ROW2_Y), (140, CHK_H)))
    compare_chk.setButtonType_(3)
    compare_chk.setTitle_("Compare models")
    compare_chk.setState_(0)
    compare_chk.setTarget_(handler)
    compare_chk.setAction_("compareChanged:")
    compare_chk.setAutoresizingMask_(_TOP)
    cv.addSubview_(compare_chk)
    handler._compare_chk = compare_chk

    # Model2 label + popup (hidden by default)
    model2_lbl = _lbl("Model 2:", ((_PAD + 145, ROW2_Y - 3), (70, CHK_H + 6)), right=False)
    model2_lbl.setHidden_(True)
    model2_lbl.setAutoresizingMask_(_TOP)
    cv.addSubview_(model2_lbl)
    handler._model2_lbl = model2_lbl

    all_models = sorted(app._model_map.keys()) if hasattr(app, '_model_map') else []
    model2_popup = NSPopUpButton.alloc().initWithFrame_(
        ((_PAD + 145 + 75, ROW2_Y - 2), (W - _PAD*2 - 145 - 75 - 70, CHK_H + 4)))
    for m in (all_models or ["(no models)"]):
        model2_popup.addItemWithTitle_(m)
    model2_popup.setTarget_(handler)
    model2_popup.setAction_("model2PopupChanged:")
    model2_popup.setHidden_(True)
    model2_popup.setAutoresizingMask_(_TOP | _WGROW)
    cv.addSubview_(model2_popup)
    handler._model2_popup = model2_popup

    # ── Inline param row (collapsible) ────────────────────────────────────────
    from AppKit import (
        NSSlider as _NSSlider, NSStepper as _NSStepper,
        NSBox as _NSBox,
    )
    PARAM_H = 24
    PARAM_Y = ROW2_Y - _GAP - PARAM_H

    params_chk = NSButton.alloc().initWithFrame_(
        ((_PAD, PARAM_Y), (90, PARAM_H)))
    params_chk.setButtonType_(3)
    params_chk.setTitle_("⚙ Params")
    params_chk.setState_(0)
    params_chk.setTarget_(handler)
    params_chk.setAction_("paramsToggled:")
    params_chk.setAutoresizingMask_(_TOP)
    cv.addSubview_(params_chk)
    handler._params_chk = params_chk

    # Temperature slider  0.0 – 2.0
    saved_temp = app._cfg.get("qt_temperature", 0.7)
    saved_max  = app._cfg.get("qt_max_tokens", 512)
    saved_preset = app._cfg.get("qt_preset", "Custom")

    px = _PAD + 95
    temp_lbl = _lbl("Temp:", ((px, PARAM_Y + 4), (38, 16)), right=False)
    temp_lbl.setHidden_(True); temp_lbl.setAutoresizingMask_(_TOP)
    cv.addSubview_(temp_lbl); handler._temp_lbl = temp_lbl

    px += 42
    temp_slider = _NSSlider.alloc().initWithFrame_(((px, PARAM_Y + 2), (100, 20)))
    temp_slider.setMinValue_(0.0); temp_slider.setMaxValue_(2.0)
    temp_slider.setFloatValue_(saved_temp)
    temp_slider.setTarget_(handler); temp_slider.setAction_("tempChanged:")
    temp_slider.setHidden_(True); temp_slider.setAutoresizingMask_(_TOP)
    cv.addSubview_(temp_slider); handler._temp_slider = temp_slider

    px += 104
    temp_val_lbl = _lbl(f"{saved_temp:.1f}", ((px, PARAM_Y + 4), (32, 16)), right=False)
    temp_val_lbl.setHidden_(True); temp_val_lbl.setAutoresizingMask_(_TOP)
    cv.addSubview_(temp_val_lbl); handler._temp_val_lbl = temp_val_lbl

    # Max tokens stepper
    px += 38
    maxtok_lbl = _lbl("Max:", ((px, PARAM_Y + 4), (34, 16)), right=False)
    maxtok_lbl.setHidden_(True); maxtok_lbl.setAutoresizingMask_(_TOP)
    cv.addSubview_(maxtok_lbl); handler._maxtok_lbl = maxtok_lbl

    px += 38
    maxtok_fld = NSTextField.alloc().initWithFrame_(((px, PARAM_Y + 2), (52, 20)))
    maxtok_fld.setStringValue_(str(saved_max))
    maxtok_fld.setFont_(NSFont.systemFontOfSize_(11))
    maxtok_fld.setHidden_(True); maxtok_fld.setAutoresizingMask_(_TOP)
    cv.addSubview_(maxtok_fld); handler._maxtok_fld = maxtok_fld

    px += 56
    maxtok_step = _NSStepper.alloc().initWithFrame_(((px, PARAM_Y + 2), (19, 20)))
    maxtok_step.setMinValue_(64); maxtok_step.setMaxValue_(32768)
    maxtok_step.setIncrement_(64); maxtok_step.setIntValue_(saved_max)
    maxtok_step.setTarget_(handler); maxtok_step.setAction_("maxTokensChanged:")
    maxtok_step.setHidden_(True); maxtok_step.setAutoresizingMask_(_TOP)
    cv.addSubview_(maxtok_step); handler._maxtok_step = maxtok_step

    # Token counter label (live char÷3.5 estimate)
    px += 24
    tok_counter_lbl = _lbl("", ((px, PARAM_Y + 4), (100, 16)), right=False)
    from AppKit import NSColor as _NSColorPrm
    tok_counter_lbl.setTextColor_(_NSColorPrm.secondaryLabelColor())
    tok_counter_lbl.setFont_(NSFont.systemFontOfSize_(10))
    tok_counter_lbl.setHidden_(True); tok_counter_lbl.setAutoresizingMask_(_TOP)
    cv.addSubview_(tok_counter_lbl); handler._tok_counter_lbl = tok_counter_lbl

    # Preset popup
    px += 104
    preset_popup = NSPopUpButton.alloc().initWithFrame_(((px, PARAM_Y), (120, PARAM_H)))
    for p in ("Precise", "Balanced", "Creative", "Custom"):
        preset_popup.addItemWithTitle_(p)
    preset_popup.selectItemWithTitle_(saved_preset)
    preset_popup.setTarget_(handler); preset_popup.setAction_("presetChanged:")
    preset_popup.setHidden_(True); preset_popup.setAutoresizingMask_(_TOP | _WGROW)
    cv.addSubview_(preset_popup); handler._preset_popup = preset_popup

    # Separator between input rows and output area
    sep_y = PARAM_Y - _GAP - 4
    sep = _NSBox.alloc().initWithFrame_(((_PAD, sep_y), (W - _PAD*2, 1)))
    sep.setBoxType_(2)  # NSBoxSeparator
    sep.setAutoresizingMask_(_TOP | _WGROW)
    cv.addSubview_(sep)

    # Model name labels above output panels (hidden until compare is on)
    LBL_H = 18
    output_top = sep_y - 4
    output_h = output_top - BOT - LBL_H - 2
    half_w = (W - _PAD*2 - _GAP) // 2

    from AppKit import NSColor as _NSColorLbl
    model1_lbl = NSTextField.alloc().initWithFrame_(
        ((_PAD, BOT + output_h + 2), (half_w, LBL_H)))
    model1_lbl.setBezeled_(False); model1_lbl.setDrawsBackground_(False)
    model1_lbl.setEditable_(False); model1_lbl.setSelectable_(False)
    model1_lbl.setFont_(NSFont.boldSystemFontOfSize_(11))
    model1_lbl.setTextColor_(_NSColorLbl.secondaryLabelColor())
    model1_lbl.setStringValue_(app._active or "Model 1")
    model1_lbl.setAutoresizingMask_(_GROW)   # grows with output area
    cv.addSubview_(model1_lbl)
    handler._model1_lbl = model1_lbl

    model2_header_lbl = NSTextField.alloc().initWithFrame_(
        ((_PAD + half_w + _GAP, BOT + output_h + 2), (half_w, LBL_H)))
    model2_header_lbl.setBezeled_(False); model2_header_lbl.setDrawsBackground_(False)
    model2_header_lbl.setEditable_(False); model2_header_lbl.setSelectable_(False)
    model2_header_lbl.setFont_(NSFont.boldSystemFontOfSize_(11))
    model2_header_lbl.setTextColor_(_NSColorLbl.secondaryLabelColor())
    model2_header_lbl.setStringValue_("Model 2")
    model2_header_lbl.setHidden_(True)
    model2_header_lbl.setAutoresizingMask_(_GROW)   # grows with output area
    cv.addSubview_(model2_header_lbl)
    handler._model2_header_lbl = model2_header_lbl

    # WKWebView output — left panel (always visible)
    from WebKit import WKWebView as _WKWebView, WKWebViewConfiguration as _WKCfg
    wv_cfg = _WKCfg.alloc().init()
    wv = _WKWebView.alloc().initWithFrame_configuration_(
        ((_PAD, BOT), (half_w, output_h)), wv_cfg)
    wv.setAutoresizingMask_(_GROW | _WGROW)
    wv.loadHTMLString_baseURL_(_QT_HTML_TEMPLATE, None)
    cv.addSubview_(wv)
    handler._output_wv = wv
    # Keep a stub _output_tv for any legacy code that checks hasattr
    handler._output_tv = None

    # Vertical divider between the two output panels (hidden by default)
    vdiv = _NSBox.alloc().initWithFrame_(
        ((_PAD + half_w + _GAP // 2, BOT), (1, output_h + LBL_H + 2)))
    vdiv.setBoxType_(2)  # NSBoxSeparator
    vdiv.setHidden_(True)
    vdiv.setAutoresizingMask_(_GROW | 1 | 4)
    cv.addSubview_(vdiv)
    handler._vdiv = vdiv

    # WKWebView right panel for compare (hidden by default)
    wv2_cfg = _WKCfg.alloc().init()
    wv2 = _WKWebView.alloc().initWithFrame_configuration_(
        ((_PAD + half_w + _GAP, BOT), (half_w, output_h)), wv2_cfg)
    wv2.setAutoresizingMask_(_GROW | _WGROW | 1)
    wv2.loadHTMLString_baseURL_(_QT_HTML_TEMPLATE, None)
    wv2.setHidden_(True)
    cv.addSubview_(wv2)
    handler._output_wv2 = wv2
    handler._output_tv2 = None  # stub for legacy checks

    # Indeterminate progress spinner (shown while waiting for first token)
    from AppKit import NSProgressIndicator as _NSPI
    spinner = _NSPI.alloc().initWithFrame_(((_PAD, BOT + output_h//2 - 8), (16, 16)))
    spinner.setStyle_(1)   # NSProgressIndicatorStyleSpinning
    spinner.setHidden_(True)
    cv.addSubview_(spinner)
    handler._spinner = spinner

    STAT_Y = 52   # stats row y
    BTN_Y  = 14   # button row y
    BTN_H  = 28   # button height

    tps_lbl = NSTextField.alloc().initWithFrame_(
        ((_PAD, STAT_Y), (half_w, 24)))
    tps_lbl.setBezeled_(False); tps_lbl.setDrawsBackground_(False)
    tps_lbl.setEditable_(False)
    tps_lbl.setFont_(NSFont.systemFontOfSize_(12))
    from AppKit import NSColor as _NSColor
    tps_lbl.setTextColor_(_NSColor.secondaryLabelColor())
    cv.addSubview_(tps_lbl)
    handler._tps_lbl = tps_lbl

    tps_lbl2 = NSTextField.alloc().initWithFrame_(
        ((_PAD + half_w + _GAP, STAT_Y), (half_w, 24)))
    tps_lbl2.setBezeled_(False); tps_lbl2.setDrawsBackground_(False)
    tps_lbl2.setEditable_(False)
    tps_lbl2.setFont_(NSFont.systemFontOfSize_(12))
    tps_lbl2.setTextColor_(_NSColor.secondaryLabelColor())
    tps_lbl2.setHidden_(True)
    cv.addSubview_(tps_lbl2)
    handler._tps_lbl2 = tps_lbl2

    # Bottom-left: Chat toggle + New Conversation
    chat_chk = NSButton.alloc().initWithFrame_(((_PAD, BTN_Y), (80, BTN_H)))
    chat_chk.setButtonType_(3)
    chat_chk.setTitle_("💬 Chat")
    chat_chk.setState_(0)
    chat_chk.setTarget_(handler)
    chat_chk.setAction_("chatToggled:")
    chat_chk.setFont_(NSFont.systemFontOfSize_(13))
    cv.addSubview_(chat_chk)
    handler._chat_chk = chat_chk

    new_conv_btn = _btn("↺ New", handler, "newConversation:",
                        ((_PAD + 84, BTN_Y), (64, BTN_H)))
    new_conv_btn.setHidden_(True)
    cv.addSubview_(new_conv_btn)
    handler._new_conv_btn = new_conv_btn

    # System prompt field (collapsible, above separator when chat is on)
    # We'll overlay it over the sep_y area — show/hide on chat toggle
    sys_prompt_lbl = _lbl("System:", ((_PAD, sep_y - PARAM_H - _GAP + 4), (_LW, 16)), right=False)
    sys_prompt_lbl.setHidden_(True); sys_prompt_lbl.setAutoresizingMask_(_TOP)
    cv.addSubview_(sys_prompt_lbl); handler._sys_prompt_lbl = sys_prompt_lbl

    sys_prompt_scroll = NSScrollView.alloc().initWithFrame_(
        ((_PAD + _LW + _GAP, sep_y - PARAM_H - _GAP),
         (W - _PAD*2 - _LW - _GAP, PARAM_H)))
    sys_prompt_scroll.setHasVerticalScroller_(False)
    sys_prompt_scroll.setAutohidesScrollers_(True)
    sys_prompt_scroll.setBorderType_(2)
    sys_prompt_scroll.setHidden_(True)
    sys_prompt_scroll.setAutoresizingMask_(_TOP | _WGROW)
    sys_prompt_tv = NSTextView.alloc().initWithFrame_(
        ((0, 0), (W - _PAD*2 - _LW - _GAP, PARAM_H)))
    sys_prompt_tv.setFont_(NSFont.systemFontOfSize_(11))
    sys_prompt_tv.setRichText_(False)
    sys_prompt_tv.setString_(app._cfg.get("qt_system_prompt", ""))
    sys_prompt_scroll.setDocumentView_(sys_prompt_tv)
    cv.addSubview_(sys_prompt_scroll)
    handler._sys_prompt_tv = sys_prompt_tv
    handler._sys_prompt_scroll = sys_prompt_scroll

    # Bottom-right button row: Prompts | Export | Copy | − | + | Clear
    bx = W - _PAD - 68
    cv.addSubview_(_btn("Clear",  handler, "clear:",        ((bx,     BTN_Y), (68, BTN_H))))
    bx -= 32
    cv.addSubview_(_btn("+",      handler, "fontLarger:",   ((bx,     BTN_Y), (28, BTN_H))))
    bx -= 32
    cv.addSubview_(_btn("−",      handler, "fontSmaller:",  ((bx,     BTN_Y), (28, BTN_H))))
    bx -= 72
    cv.addSubview_(_btn("Copy",   handler, "copyResponse:", ((bx,     BTN_Y), (68, BTN_H))))
    bx -= 76
    cv.addSubview_(_btn("Export", handler, "exportSession:",((bx,     BTN_Y), (72, BTN_H))))
    bx -= 100

    # Prompts pull-down button (library + generate)
    from AppKit import NSPopUpButton as _NSPUBp, NSMenuItem as _NSMIp, NSMenu as _NSMenup
    prompts_btn = _NSPUBp.alloc().initWithFrame_(((bx, BTN_Y), (96, BTN_H)))
    prompts_btn.setPullsDown_(True)
    prompts_btn.setTarget_(handler)
    prompts_btn.setBezelStyle_(4)   # rounded
    # First item is the button label (shown when pull-down is closed)
    prompts_btn.addItemWithTitle_("📚 Prompts")
    prompts_btn.itemAtIndex_(0).setEnabled_(False)
    # Library categories as submenus
    for cat, prompts in _PROMPT_LIBRARY.items():
        cat_item = _NSMIp.alloc().initWithTitle_action_keyEquivalent_(cat, None, "")
        cat_item.setEnabled_(True)
        sub = _NSMenup.alloc().initWithTitle_(cat)
        for p in prompts:
            short = p[:72] + "…" if len(p) > 72 else p
            pi = _NSMIp.alloc().initWithTitle_action_keyEquivalent_(
                short, "promptLibrarySelected:", "")
            pi.setTarget_(handler)
            pi.setRepresentedObject_(p)   # store full prompt
            sub.addItem_(pi)
        cat_item.setSubmenu_(sub)
        prompts_btn.menu().addItem_(cat_item)
    # Separator + generate option
    prompts_btn.menu().addItem_(_NSMIp.separatorItem())
    gen_item = _NSMIp.alloc().initWithTitle_action_keyEquivalent_(
        "✨ Generate suggestions…", "generateSuggestions:", "")
    gen_item.setTarget_(handler)
    prompts_btn.menu().addItem_(gen_item)
    cv.addSubview_(prompts_btn)

    handler._output_font_size = app._cfg.get("qt_font_size", 12)

    # Persistent load-status poll — updates tps_lbl while a model is loading
    load_poll = rumps.Timer(handler._loadStatusTick_, 0.5)
    load_poll.start()
    handler._load_poll_timer = load_poll

    win.setDelegate_(handler)
    win.setReleasedWhenClosed_(False)

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
         + _DG + _SH + _SG + 8 * (_RH + _RG)        # Sync (6 checkboxes + script + HF token)
         + _DG + _BTN_H + _BTN_BOT)

    def fy(from_top: int, h: int = _RH) -> int:
        return H - from_top - h

    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    panel.setTitle_("Switchman — Settings")
    panel.setMinSize_((W, 400))
    panel.setDelegate_(handler)
    panel.center()
    _constrain_to_screen(panel)
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
        ("notifications",      "macOS notification when model ready"),
        ("auto_reload_on_crash", "Auto-reload model after server crash"),
        ("sync_env",           "Write env file (LLM_BASE_URL) on switch"),
        ("sync_aider",         "Sync ~/.aider.conf.yml on switch"),
        ("sync_zed",           "Sync ~/.config/zed/settings.json on switch"),
        ("sync_continue",      "Sync ~/.continue/config.json on switch"),
    ]:
        c = NSButton.alloc().initWithFrame_(((x_fld, fy(cur)), (FW, _RH)))
        c.setButtonType_(3)
        c.setTitle_(title)
        c.setState_(1 if cfg.get(key, DEFAULTS.get(key, False)) else 0)
        cv.addSubview_(c)
        sync_chks[key] = c
        cur += _RH + _RG

    cv.addSubview_(_lbl("On switch script:", ((x_lbl, fy(cur)), (_LW, _RH))))
    script_fld = _fld(cfg.get("on_switch_script", ""), ((x_fld, fy(cur)), (FW, _RH)))
    script_fld.setPlaceholderString_("/path/to/script.sh  (env: SWITCHMAN_MODEL, _PORT, _KIND)")
    cv.addSubview_(script_fld)
    cur += _RH + _RG

    cv.addSubview_(_lbl("HF token:", ((x_lbl, fy(cur)), (_LW, _RH))))
    hf_token_fld = _fld(cfg.get("hf_token", ""), ((x_fld, fy(cur)), (FW, _RH)))
    hf_token_fld.setPlaceholderString_("hf_… (for gated models — Llama, Gemma, etc.)")
    cv.addSubview_(hf_token_fld)
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
    cfg["on_switch_script"] = script_fld.stringValue().strip()
    cfg["hf_token"] = hf_token_fld.stringValue().strip()
    return True

def run_settings_panel_tabbed(cfg: dict) -> bool:
    """Tabbed settings panel (UI 2.0). Tabs: Inference · Sync · Behavior · Appearance."""
    from AppKit import (
        NSTabView, NSTabViewItem, NSScrollView,
    )
    W = 560
    H = 480
    TAB_H = H - _BTN_BOT - _BTN_H - _GAP - 28  # height available inside each tab

    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    panel.setTitle_("Switchman — Settings")
    panel.setMinSize_((W, 380))
    panel.setDelegate_(handler)
    panel.center()
    _constrain_to_screen(panel)
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(panel)

    # ── Tab view ──────────────────────────────────────────────────────────────
    tab_view = NSTabView.alloc().initWithFrame_(
        ((_PAD, _BTN_BOT + _BTN_H + _GAP), (W - _PAD*2, TAB_H + 28)))
    tab_view.setAutoresizingMask_(2 | 16)  # width+height sizable
    cv.addSubview_(tab_view)

    def _make_tab(label):
        item = NSTabViewItem.alloc().init()
        item.setLabel_(label)
        inner = NSScrollView.alloc().initWithFrame_(((0, 0), (W - _PAD*2 - 16, TAB_H)))
        inner.setHasVerticalScroller_(True)
        inner.setAutohidesScrollers_(True)
        # Content view inside scroll
        content = NSView.alloc().initWithFrame_(((0, 0), (W - _PAD*2 - 32, 0)))
        item._tab_content = content
        item._tab_scroll = inner
        inner.setDocumentView_(content)
        item.setView_(inner)
        tab_view.addTabViewItem_(item)
        return item, content

    BW = 72
    x_lbl = _PAD
    x_fld = _PAD + _LW + _GAP
    FWB   = W - _PAD*2 - 16 - _LW - _GAP - 6 - BW  # field with browse
    FW    = W - _PAD*2 - 16 - _LW - _GAP            # field no browse
    x_btn = x_fld + FWB + 6

    fields: dict[str, NSTextField] = {}

    # ── Tab 1: Inference ──────────────────────────────────────────────────────
    tab_inf, inf_cv = _make_tab("Inference")
    cur = _PAD

    def fy_tab(cv_h, from_top, h=_RH):
        return cv_h - from_top - h

    # Paths section
    INF_H = (_SH + _SG
             + 4 * (_RH + _RG)           # Paths: 4 rows
             + _DG + _SH + _SG
             + 2 * (_RH + _RG)           # vLLM browse rows (dir + binary)
             + 3 * (_RH + _RG)           # vLLM text rows (port + api key + extra)
             + (3 * _RH + _RG) + 14      # vLLM HF model IDs textarea
             + _DG + _SH + _SG
             + 2 * (_RH + _RG)           # Ollama: host + api key
             + _DG + _SH + _SG
             + 3 * (_RH + _RG)           # oMLX: 3 rows
             + _PAD)
    inf_cv.setFrameSize_((W - _PAD*2 - 32, INF_H))

    def fy_i(from_top, h=_RH):
        return INF_H - from_top - h

    inf_cv.addSubview_(_lbl("Paths", ((_PAD, fy_i(cur, _SH)), (W - _PAD*2 - 32 - _PAD, _SH)),
                            bold=True, right=False))
    cur += _SH + _SG
    for key, label, is_dir in [
        ("mlx_dir",         "MLX models dir",     True),
        ("gguf_dir",        "GGUF models dir",     True),
        ("llama_server",    "llama-server binary", False),
        ("opencode_config", "opencode config",     False),
    ]:
        inf_cv.addSubview_(_lbl(label + ":", ((x_lbl, fy_i(cur)), (_LW, _RH))))
        f = _fld(cfg[key], ((x_fld, fy_i(cur)), (FWB, _RH)))
        inf_cv.addSubview_(f)
        inf_cv.addSubview_(_browse_btn(handler, f, is_dir,
                                       ((x_btn, fy_i(cur) - 3), (BW, _RH + 6))))
        fields[key] = f
        cur += _RH + _RG

    cur += _DG
    inf_cv.addSubview_(_lbl("vLLM", ((_PAD, fy_i(cur, _SH)), (W - _PAD*2 - 32 - _PAD, _SH)),
                            bold=True, right=False))
    cur += _SH + _SG
    for key, label, is_dir in [
        ("vllm_models_dir", "vLLM models dir", True),
        ("vllm_binary",     "vllm binary",      False),
    ]:
        inf_cv.addSubview_(_lbl(label + ":", ((x_lbl, fy_i(cur)), (_LW, _RH))))
        f = _fld(cfg.get(key, ""), ((x_fld, fy_i(cur)), (FWB, _RH)))
        inf_cv.addSubview_(f)
        inf_cv.addSubview_(_browse_btn(handler, f, is_dir,
                                       ((x_btn, fy_i(cur) - 3), (BW, _RH + 6))))
        fields[key] = f
        cur += _RH + _RG
    for key, label, placeholder in [
        ("vllm_port",       "Port",       "8001"),
        ("vllm_api_key",    "API Key",    "sk-… (optional)"),
        ("vllm_extra_args", "Extra args", "--dtype float16"),
    ]:
        inf_cv.addSubview_(_lbl(label + ":", ((x_lbl, fy_i(cur)), (_LW, _RH))))
        val = str(cfg.get(key, "")) if key == "vllm_port" else cfg.get(key, "")
        f = _fld(val, ((x_fld, fy_i(cur)), (FW, _RH)))
        f.setPlaceholderString_(placeholder)
        inf_cv.addSubview_(f)
        fields[key] = f
        cur += _RH + _RG
    # HF model IDs — one per line
    hf_ta_h = _RH * 3
    inf_cv.addSubview_(_lbl("HF model IDs:", ((x_lbl, fy_i(cur, hf_ta_h)), (_LW, hf_ta_h))))
    from AppKit import NSScrollView as _NSSV2, NSTextView as _NSTV2
    hf_scroll = _NSSV2.alloc().initWithFrame_(((x_fld, fy_i(cur, hf_ta_h)), (FW, hf_ta_h)))
    hf_scroll.setHasVerticalScroller_(True); hf_scroll.setBorderType_(2)
    hf_tv = _NSTV2.alloc().initWithFrame_(((0, 0), (FW, hf_ta_h)))
    hf_tv.setFont_(NSFont.systemFontOfSize_(11))
    hf_tv.setString_("\n".join(cfg.get("vllm_hf_models", [])))
    hf_tv.setRichText_(False)
    hf_scroll.setDocumentView_(hf_tv)
    inf_cv.addSubview_(hf_scroll)
    fields["vllm_hf_tv"] = hf_tv
    inf_cv.addSubview_(_lbl("one per line", ((x_fld, fy_i(cur, hf_ta_h) - 14), (FW, 12)),
                            right=False))
    cur += hf_ta_h + _RG

    cur += _DG
    inf_cv.addSubview_(_lbl("Ollama", ((_PAD, fy_i(cur, _SH)), (W - _PAD*2 - 32 - _PAD, _SH)),
                            bold=True, right=False))
    cur += _SH + _SG
    for key, label, placeholder in [
        ("ollama_host",    "Host URL", "http://localhost:11434"),
        ("ollama_api_key", "API Key",  "(optional, for remote Ollama)"),
    ]:
        inf_cv.addSubview_(_lbl(label + ":", ((x_lbl, fy_i(cur)), (_LW, _RH))))
        f = _fld(cfg.get(key, ""), ((x_fld, fy_i(cur)), (FW, _RH)))
        f.setPlaceholderString_(placeholder)
        inf_cv.addSubview_(f)
        fields[key] = f
        cur += _RH + _RG

    cur += _DG
    inf_cv.addSubview_(_lbl("oMLX Server", ((_PAD, fy_i(cur, _SH)), (W - _PAD*2 - 32 - _PAD, _SH)),
                            bold=True, right=False))
    cur += _SH + _SG
    for key, label, placeholder in [
        ("omlx_port",    "Port",    "8000"),
        ("omlx_api_key", "API Key", "sk-…"),
        ("omlx_service", "Service", "com.jim.omlx"),
    ]:
        inf_cv.addSubview_(_lbl(label + ":", ((x_lbl, fy_i(cur)), (_LW, _RH))))
        val = str(cfg.get(key, "")) if key == "omlx_port" else cfg.get(key, "")
        f = _fld(val, ((x_fld, fy_i(cur)), (FW, _RH)))
        f.setPlaceholderString_(placeholder)
        inf_cv.addSubview_(f)
        fields[key] = f
        cur += _RH + _RG

    # ── Tab 2: Sync ───────────────────────────────────────────────────────────
    tab_sync, sync_cv = _make_tab("Sync")
    SYNC_H = (_PAD + _SH + _SG + 6 * (_RH + _RG)
              + _DG + 2 * (_RH + _RG) + _PAD)
    sync_cv.setFrameSize_((W - _PAD*2 - 32, SYNC_H))

    def fy_s(from_top, h=_RH):
        return SYNC_H - from_top - h

    cur_s = _PAD
    sync_cv.addSubview_(_lbl("Client Sync", ((_PAD, fy_s(cur_s, _SH)), (W - _PAD*2 - 32 - _PAD, _SH)),
                             bold=True, right=False))
    cur_s += _SH + _SG
    sync_chks = {}
    for key, title in [
        ("notifications",        "macOS notification when model ready"),
        ("auto_reload_on_crash", "Auto-reload model after server crash"),
        ("sync_env",             "Write env file (LLM_BASE_URL) on switch"),
        ("sync_aider",           "Sync ~/.aider.conf.yml on switch"),
        ("sync_zed",             "Sync ~/.config/zed/settings.json on switch"),
        ("sync_continue",        "Sync ~/.continue/config.json on switch"),
    ]:
        c = NSButton.alloc().initWithFrame_(((x_fld, fy_s(cur_s)), (FW, _RH)))
        c.setButtonType_(3); c.setTitle_(title)
        c.setState_(1 if cfg.get(key, DEFAULTS.get(key, False)) else 0)
        sync_cv.addSubview_(c)
        sync_chks[key] = c
        cur_s += _RH + _RG

    cur_s += _DG
    sync_cv.addSubview_(_lbl("On switch script:", ((x_lbl, fy_s(cur_s)), (_LW, _RH))))
    script_fld = _fld(cfg.get("on_switch_script", ""), ((x_fld, fy_s(cur_s)), (FW, _RH)))
    script_fld.setPlaceholderString_("/path/to/script.sh  (env: SWITCHMAN_MODEL, _PORT, _KIND)")
    sync_cv.addSubview_(script_fld)
    cur_s += _RH + _RG

    sync_cv.addSubview_(_lbl("HF token:", ((x_lbl, fy_s(cur_s)), (_LW, _RH))))
    hf_token_fld = _fld(cfg.get("hf_token", ""), ((x_fld, fy_s(cur_s)), (FW, _RH)))
    hf_token_fld.setPlaceholderString_("hf_… (for gated models)")
    sync_cv.addSubview_(hf_token_fld)

    # ── Tab 3: Behavior ───────────────────────────────────────────────────────
    tab_beh, beh_cv = _make_tab("Behavior")
    BEH_H = (_PAD + _SH + _SG + 4 * (_RH + _RG) + _PAD)
    beh_cv.setFrameSize_((W - _PAD*2 - 32, BEH_H))

    def fy_b(from_top, h=_RH):
        return BEH_H - from_top - h

    cur_b = _PAD
    beh_cv.addSubview_(_lbl("Startup & UI", ((_PAD, fy_b(cur_b, _SH)), (W - _PAD*2 - 32 - _PAD, _SH)),
                            bold=True, right=False))
    cur_b += _SH + _SG

    beh_cv.addSubview_(_lbl("Restart opencode:", ((x_lbl, fy_b(cur_b)), (_LW, _RH))))
    chk_roc = NSButton.alloc().initWithFrame_(((x_fld, fy_b(cur_b)), (FW, _RH)))
    chk_roc.setButtonType_(3)
    chk_roc.setTitle_("Restart opencode on model switch")
    chk_roc.setState_(1 if cfg.get("restart_opencode", False) else 0)
    beh_cv.addSubview_(chk_roc)
    cur_b += _RH + _RG

    beh_cv.addSubview_(_lbl("Terminal app:", ((x_lbl, fy_b(cur_b)), (_LW, _RH))))
    term_popup = NSPopUpButton.alloc().initWithFrame_(
        ((x_fld, fy_b(cur_b) - 2), (140, _RH + 4)))
    for opt in ["Terminal", "iTerm2"]:
        term_popup.addItemWithTitle_(opt)
    term_popup.selectItemWithTitle_(cfg.get("terminal_app", "Terminal"))
    beh_cv.addSubview_(term_popup)

    # ── Tab 4: Models ────────────────────────────────────────────────────────
    tab_mod, mod_cv = _make_tab("Models")
    # Inner scroll height is variable; start with a reasonable estimate
    # The actual model list is built dynamically from cfg
    from AppKit import NSSearchField, NSColor, NSView
    MOD_INNER_H = max(400, 28 * (len(cfg.get("known_models", [])) or 10) + 60)
    mod_cv.setFrameSize_((W - _PAD*2 - 32, MOD_INNER_H))

    # Search field at top
    search_fld = NSSearchField.alloc().initWithFrame_(
        ((_PAD, MOD_INNER_H - _PAD - 24), (W - _PAD*2 - 32 - _PAD, 24)))
    search_fld.setPlaceholderString_("Filter models…")
    mod_cv.addSubview_(search_fld)

    # Model card rows: name | kind badge | size | tags | actions
    my = MOD_INNER_H - _PAD - 24 - _GAP
    known = sorted(cfg.get("known_models", []))
    hidden_set = set(cfg.get("hidden_models", []))
    tags_map = cfg.get("model_tags", {})
    aliases = cfg.get("aliases", {})
    card_h = 36
    for name in known:
        my -= card_h + 2
        kind_str = "GGUF" if any(name.endswith(s) for s in (".gguf", "-GGUF")) else "MLX"
        color = NSColor.systemBlueColor() if kind_str == "MLX" else NSColor.systemOrangeColor()
        alias = aliases.get(name, "")
        display = alias if alias else name
        tags = ", ".join(tags_map.get(name, []))
        hidden = name in hidden_set

        row = NSView.alloc().initWithFrame_(((_PAD, my), (W - _PAD*2 - 32 - _PAD, card_h)))
        row.setWantsLayer_(True)
        row.layer().setCornerRadius_(6)
        row.layer().setBackgroundColor_(NSColor.quaternaryLabelColor().CGColor())

        # Kind badge
        badge = NSTextField.alloc().initWithFrame_(((4, 8), (36, 18)))
        badge.setStringValue_(kind_str)
        badge.setBezeled_(False); badge.setDrawsBackground_(False)
        badge.setEditable_(False); badge.setSelectable_(False)
        badge.setFont_(NSFont.boldSystemFontOfSize_(9))
        badge.setTextColor_(color)
        row.addSubview_(badge)

        # Model name
        name_lbl = NSTextField.alloc().initWithFrame_(((42, 12), (W - _PAD*2 - 32 - _PAD - 100 - 42, 18)))
        name_lbl.setStringValue_(display[:55])
        name_lbl.setBezeled_(False); name_lbl.setDrawsBackground_(False)
        name_lbl.setEditable_(False); name_lbl.setSelectable_(False)
        name_lbl.setFont_(NSFont.systemFontOfSize_(12))
        name_lbl.setTextColor_(NSColor.labelColor())
        row.addSubview_(name_lbl)

        # Tags / hidden badge
        sub_text = (f"🏷 {tags}" if tags else "") + (" · 👁 hidden" if hidden else "")
        if sub_text:
            sub_lbl = NSTextField.alloc().initWithFrame_(((42, 2), (W - _PAD*2 - 32 - _PAD - 80 - 42, 10)))
            sub_lbl.setStringValue_(sub_text[:70])
            sub_lbl.setBezeled_(False); sub_lbl.setDrawsBackground_(False)
            sub_lbl.setEditable_(False); sub_lbl.setSelectable_(False)
            sub_lbl.setFont_(NSFont.systemFontOfSize_(9))
            sub_lbl.setTextColor_(NSColor.secondaryLabelColor())
            row.addSubview_(sub_lbl)

        mod_cv.addSubview_(row)

    # ── Tab 5: Appearance ─────────────────────────────────────────────────────
    _make_tab("Appearance")   # placeholder — future font/theme controls

    # ── Buttons (outside tab view) ────────────────────────────────────────────
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

    # Read back into cfg
    for key in ("mlx_dir", "gguf_dir", "llama_server", "opencode_config",
                "omlx_api_key", "omlx_service",
                "vllm_models_dir", "vllm_binary", "vllm_extra_args", "vllm_api_key",
                "ollama_host", "ollama_api_key"):
        cfg[key] = fields[key].stringValue()
    try:
        cfg["omlx_port"] = int(fields["omlx_port"].stringValue())
    except ValueError:
        pass
    try:
        cfg["vllm_port"] = int(fields["vllm_port"].stringValue())
    except ValueError:
        pass
    # HF model IDs — one per line, strip blanks
    raw_hf = fields["vllm_hf_tv"].string()
    cfg["vllm_hf_models"] = [l.strip() for l in raw_hf.splitlines() if l.strip()]
    cfg["restart_opencode"] = bool(chk_roc.state())
    cfg["terminal_app"] = term_popup.titleOfSelectedItem()
    for key, c in sync_chks.items():
        cfg[key] = bool(c.state())
    cfg["on_switch_script"] = script_fld.stringValue().strip()
    cfg["hf_token"] = hf_token_fld.stringValue().strip()
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
         + _SH + _SG + 3 * (_RH + _RG)                    # Identity (alias + note + tags)
         + _DG + _SH + _SG + n_limit * (_RH + _RG)        # Limits
         + _DG + _SH + _SG + 8 * (_RH + _RG) - _RG        # Sampling (preset + 6 + checkbox)
         + _DG + _BTN_H + _BTN_BOT)

    def fy(from_top: int, h: int = _RH) -> int:
        return H - from_top - h

    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    panel.setTitle_(f"Settings — {name}")
    panel.setMinSize_((W, 300))
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
    cur += _RH + _RG
    existing_tags = ", ".join(cfg.get("model_tags", {}).get(name, []))
    cv.addSubview_(_lbl("Tags:", ((x_lbl, fy(cur)), (_LW, _RH))))
    tags_fld = _fld(existing_tags, ((x_fld, fy(cur)), (FW, _RH)))
    tags_fld.setPlaceholderString_("coding, fast, vision  (comma-separated)")
    cv.addSubview_(tags_fld)
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

    # Read tags
    tags_raw = tags_fld.stringValue().strip()
    tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()]
    if tags:
        cfg.setdefault("model_tags", {})[name] = tags
    else:
        cfg.get("model_tags", {}).pop(name, None)

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
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    panel.setTitle_("Edit Benchmark Prompts")
    panel.setMinSize_((W, 300))
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
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    mode_label = "llama-bench" if is_gguf else "API"
    panel.setTitle_(f"Benchmark — {name}  ({mode_label})")
    panel.setMinSize_((W, 300))
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
    port, api_key = active_api(cfg, kind)
    base_url = active_base_url(cfg, kind)
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
                        f"{base_url}/v1/chat/completions",
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
    port = cfg.get("llama_port", 8080)
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
      th.th-in  { background: #3a5a8a; }
      th.th-out { background: #2a6a3a; }
      th.num   { text-align: right; }
      td   { padding: 5px 10px; border-bottom: 1px solid #ddd; font-size: 12px; }
      td.num { text-align: right; font-variant-numeric: tabular-nums; }
      tr:nth-child(even) { background: #ececec; }
      tr:hover { background: #dde8f5; }
      .best { font-weight: 600; color: #1a6c1a; }
      .note { color: #888; font-size: 11px; margin-top: 10px; }
      .err  { color: #c00; font-size: 12px; margin: 6px 0; }
      @media (prefers-color-scheme: dark) {
        body { color: #e8e8e8; background: #1e1e1e; }
        .sub { color: #999; }
        th.th-in  { background: #2a4a78; }
        th.th-out { background: #1a5a2a; }
        td   { border-bottom-color: #3a3a3a; }
        tr:nth-child(even) { background: #2a2a2a; }
        tr:hover { background: #2a3a50; }
        .best { color: #4a9c4a; }
        .note { color: #777; }
        .err  { color: #f55; }
      }
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
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    panel.setTitle_(f"Benchmark Results — {name}")
    panel.setMinSize_((W, 400))
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
  @media(prefers-color-scheme:dark){{
    body{{color:#e8e8e8;background:#1e1e1e}}
    canvas{{background:#2a2a2a}}
    th{{background:#2a4a78}}
    td{{border-bottom-color:#3a3a3a}}
    tr:nth-child(even){{background:#2a2a2a}}
  }}
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


def run_schedule_panel(cfg: dict, model_map: dict) -> None:
    """Edit the model switching schedule (HH:MM → model name pairs)."""
    W, H = 480, 380
    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    panel.setTitle_("Scheduled Model Switching")
    panel.setMinSize_((W, 280))
    panel.center()
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    panel.setDelegate_(handler)
    cv = _vibrancy_content_view(panel)

    # Enable toggle
    chk = NSButton.alloc().initWithFrame_(((_PAD, H - _PAD - 28), (W - _PAD * 2, 24)))
    chk.setButtonType_(3)
    chk.setTitle_("Enable scheduled model switching")
    chk.setState_(1 if cfg.get("schedule_enabled") else 0)
    cv.addSubview_(chk)

    cv.addSubview_(_lbl("Define time → model rules below. The closest past entry wins each minute.",
                        ((_PAD, H - _PAD - 52), (W - _PAD * 2, 18)), right=False))

    # Table: time | model
    from AppKit import NSTableView, NSScrollView as _SVS, NSTableColumn
    tbl_h = H - _PAD - 52 - _GAP - 60 - _GAP - 34
    tbl_scroll = _SVS.alloc().initWithFrame_(((_PAD, 60 + _GAP), (W - _PAD * 2, tbl_h)))
    tbl_scroll.setHasVerticalScroller_(True); tbl_scroll.setAutohidesScrollers_(True)
    tbl = NSTableView.alloc().initWithFrame_(((0, 0), (W - _PAD * 2, tbl_h)))
    for ident, title, width in [("time", "Time (HH:MM)", 110), ("model", "Model", W - _PAD*2 - 120)]:
        c = NSTableColumn.alloc().initWithIdentifier_(ident)
        c.setWidth_(width)
        c.headerCell().setStringValue_(title)
        c.setEditable_(True)
        tbl.addTableColumn_(c)
    tbl.setUsesAlternatingRowBackgroundColors_(True)
    tbl.setRowHeight_(22)
    tbl_scroll.setDocumentView_(tbl)
    cv.addSubview_(tbl_scroll)

    # + / − buttons
    add_btn = _btn("+", handler, "addRow:", ((_PAD, 60), (32, 24)))
    rem_btn = _btn("−", handler, "removeRow:", ((_PAD + 36, 60), (32, 24)))
    cv.addSubview_(add_btn); cv.addSubview_(rem_btn)

    # Save / Cancel
    cv.addSubview_(_btn("Save",   handler, "doOK:",    ((W - _PAD - 72, 14), (72, 26)), "\r"))
    cv.addSubview_(_btn("Cancel", handler, "doCancel:",(( W - _PAD - 148, 14), (68, 26))))

    schedule = [dict(e) for e in cfg.get("model_schedule", [])]

    class _SchedDS(NSObject):
        def numberOfRowsInTableView_(self, tv): return len(schedule)
        def tableView_objectValueForTableColumn_row_(self, tv, col, row):
            ident = col.identifier()
            return schedule[row].get(ident, "")
        def tableView_setObjectValue_forTableColumn_row_(self, tv, val, col, row):
            schedule[row][col.identifier()] = val or ""

    ds = _SchedDS.alloc().init()
    tbl.setDataSource_(ds)
    tbl.setDelegate_(ds)
    tbl.reloadData()

    class _SchedActions(NSObject):
        def addRow_(self, _s):
            all_models = sorted(model_map.keys())
            schedule.append({"time": "09:00", "model": all_models[0] if all_models else ""})
            tbl.reloadData()
        def removeRow_(self, _s):
            row = tbl.selectedRow()
            if 0 <= row < len(schedule):
                schedule.pop(row)
                tbl.reloadData()
        def doOK_(self, _s):
            NSApp.stopModalWithCode_(NSModalResponseOK)
        def doCancel_(self, _s):
            NSApp.stopModalWithCode_(NSModalResponseOK + 1)

    sa = _SchedActions.alloc().init()
    add_btn.setTarget_(sa); rem_btn.setTarget_(sa)
    handler._ds_ref = ds; handler._sa_ref = sa

    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    result = NSApp.runModalForWindow_(panel)
    panel.orderOut_(None)
    if result == NSModalResponseOK:
        cfg["schedule_enabled"] = bool(chk.state())
        cfg["model_schedule"] = [e for e in schedule if e.get("time") and e.get("model")]
        save_config(cfg)


def run_compare_history_panel() -> None:
    """Show compare history as a browsable panel with side-by-side responses."""
    try:
        history = json.loads(_COMPARE_HISTORY_PATH.read_text()) \
            if _COMPARE_HISTORY_PATH.exists() else []
    except Exception:
        history = []

    W, H = 880, 600
    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    panel.setTitle_("Compare History")
    panel.setMinSize_((600, 400))
    panel.center()
    panel.setTitlebarAppearsTransparent_(True)
    panel.setMovableByWindowBackground_(True)
    panel.setDelegate_(handler)
    cv = _vibrancy_content_view(panel)

    if not history:
        lbl = _lbl("No compare history yet. Run a side-by-side compare in Quick Test first.",
                   ((_PAD, H // 2 - 10), (W - _PAD * 2, 20)))
        cv.addSubview_(lbl)
        NSApp.activateIgnoringOtherApps_(True)
        panel.makeKeyAndOrderFront_(None)
        NSApp.runModalForWindow_(panel)
        return

    # Left: scrollable list of runs
    LIST_W = 220
    from AppKit import NSTableView, NSScrollView as _SV2, NSTableColumn, NSIndexSet

    list_scroll = _SV2.alloc().initWithFrame_(((_PAD, 44), (LIST_W, H - _PAD - 44)))
    list_scroll.setHasVerticalScroller_(True); list_scroll.setAutohidesScrollers_(True)
    tbl = NSTableView.alloc().initWithFrame_(((0, 0), (LIST_W, H)))
    col = NSTableColumn.alloc().initWithIdentifier_("run")
    col.setWidth_(LIST_W - 4)
    col.headerCell().setStringValue_("Run")
    tbl.addTableColumn_(col)
    tbl.setUsesAlternatingRowBackgroundColors_(True)
    tbl.setRowHeight_(42)
    list_scroll.setDocumentView_(tbl)
    cv.addSubview_(list_scroll)

    # Right: two NSTextViews showing responses
    RX = _PAD + LIST_W + _GAP
    RW = W - RX - _PAD
    half_rw = (RW - _GAP) // 2

    # Prompt label
    prompt_lbl = NSTextField.alloc().initWithFrame_(((RX, H - _PAD - 18), (RW, 18)))
    prompt_lbl.setBezeled_(False); prompt_lbl.setDrawsBackground_(False)
    prompt_lbl.setEditable_(False)
    prompt_lbl.setFont_(NSFont.boldSystemFontOfSize_(12))
    cv.addSubview_(prompt_lbl)

    # Model name labels
    m1_lbl = NSTextField.alloc().initWithFrame_(((RX, H - _PAD - 38), (half_rw, 18)))
    m1_lbl.setBezeled_(False); m1_lbl.setDrawsBackground_(False); m1_lbl.setEditable_(False)
    m1_lbl.setFont_(NSFont.boldSystemFontOfSize_(11))
    cv.addSubview_(m1_lbl)

    m2_lbl = NSTextField.alloc().initWithFrame_(((RX + half_rw + _GAP, H - _PAD - 38), (half_rw, 18)))
    m2_lbl.setBezeled_(False); m2_lbl.setDrawsBackground_(False); m2_lbl.setEditable_(False)
    m2_lbl.setFont_(NSFont.boldSystemFontOfSize_(11))
    cv.addSubview_(m2_lbl)

    # Stats labels
    stats1_lbl = NSTextField.alloc().initWithFrame_(((RX, 10), (half_rw, 18)))
    stats1_lbl.setBezeled_(False); stats1_lbl.setDrawsBackground_(False); stats1_lbl.setEditable_(False)
    stats1_lbl.setFont_(NSFont.systemFontOfSize_(11))
    from AppKit import NSColor as _NSCh
    stats1_lbl.setTextColor_(_NSCh.secondaryLabelColor())
    cv.addSubview_(stats1_lbl)

    stats2_lbl = NSTextField.alloc().initWithFrame_(((RX + half_rw + _GAP, 10), (half_rw, 18)))
    stats2_lbl.setBezeled_(False); stats2_lbl.setDrawsBackground_(False); stats2_lbl.setEditable_(False)
    stats2_lbl.setFont_(NSFont.systemFontOfSize_(11))
    stats2_lbl.setTextColor_(_NSCh.secondaryLabelColor())
    cv.addSubview_(stats2_lbl)

    # Response text views
    tv_h = H - _PAD - 58 - 32
    sv1 = NSScrollView.alloc().initWithFrame_(((RX, 32), (half_rw, tv_h)))
    sv1.setHasVerticalScroller_(True); sv1.setAutohidesScrollers_(True)
    tv1 = NSTextView.alloc().initWithFrame_(((0, 0), (half_rw, tv_h)))
    tv1.setFont_(NSFont.userFixedPitchFontOfSize_(11))
    tv1.setEditable_(False)
    tv1.setTextColor_(_NSCh.labelColor())
    tv1.setBackgroundColor_(_NSCh.clearColor()); tv1.setDrawsBackground_(False)
    sv1.setDocumentView_(tv1); cv.addSubview_(sv1)

    sv2 = NSScrollView.alloc().initWithFrame_(((RX + half_rw + _GAP, 32), (half_rw, tv_h)))
    sv2.setHasVerticalScroller_(True); sv2.setAutohidesScrollers_(True)
    tv2 = NSTextView.alloc().initWithFrame_(((0, 0), (half_rw, tv_h)))
    tv2.setFont_(NSFont.userFixedPitchFontOfSize_(11))
    tv2.setEditable_(False)
    tv2.setTextColor_(_NSCh.labelColor())
    tv2.setBackgroundColor_(_NSCh.clearColor()); tv2.setDrawsBackground_(False)
    sv2.setDocumentView_(tv2); cv.addSubview_(sv2)

    # Vertical divider
    from AppKit import NSBox as _NSBh
    vdiv = _NSBh.alloc().initWithFrame_(((RX + half_rw + _GAP // 2, 28), (1, tv_h + 4)))
    vdiv.setBoxType_(2); cv.addSubview_(vdiv)

    # Export button
    exp_btn = _btn("Export…", handler, "exportCompare:", ((_PAD, 10), (80, 24)))
    cv.addSubview_(exp_btn)

    # Wire table data source
    class _CompareDS(NSObject):
        def numberOfRowsInTableView_(self, tv): return len(history)
        def tableView_objectValueForTableColumn_row_(self, tv, col, row):
            e = history[row]
            ts = e.get("ts", "")[:16].replace("T", " ")
            return f"{ts}\n{e.get('model1','?')[:20]} vs {e.get('model2','?')[:20]}"
        def tableViewSelectionDidChange_(self, notif):
            tv = notif.object()
            row = tv.selectedRow()
            if row < 0 or row >= len(history): return
            e = history[row]
            prompt_lbl.setStringValue_(e.get("prompt", "")[:120])
            m1_lbl.setStringValue_(e.get("model1", ""))
            m2_lbl.setStringValue_(e.get("model2", ""))
            stats1_lbl.setStringValue_(
                f"{e.get('tps1', 0)} tok/s · {e.get('tokens1', 0)} tokens")
            stats2_lbl.setStringValue_(
                f"{e.get('tps2', 0)} tok/s · {e.get('tokens2', 0)} tokens")
            tv1.setString_(e.get("response1", ""))
            tv2.setString_(e.get("response2", ""))

    ds = _CompareDS.alloc().init()
    ds._history_ref = history
    tbl.setDataSource_(ds)
    tbl.setDelegate_(ds)
    tbl.reloadData()
    if history:
        tbl.selectRowIndexes_byExtendingSelection_(
            NSIndexSet.indexSetWithIndex_(0), False)

    # Export callback
    class _ExportCompare(NSObject):
        def exportCompare_(self, _s):
            row = tbl.selectedRow()
            if row < 0 or row >= len(history): return
            e = history[row]
            from AppKit import NSSavePanel
            sp = NSSavePanel.savePanel()
            sp.setTitle_("Export Compare Result")
            sp.setNameFieldStringValue_("compare.md")
            sp.setAllowedFileTypes_(["md"])
            NSApp.activateIgnoringOtherApps_(True)
            if sp.runModal() == NSModalResponseOK:
                path = Path(sp.URL().path())
                md = (f"# Compare: {e.get('model1')} vs {e.get('model2')}\n\n"
                      f"**Date:** {e.get('ts','')}\n\n"
                      f"## Prompt\n\n{e.get('prompt','')}\n\n"
                      f"## {e.get('model1')} "
                      f"({e.get('tps1',0)} tok/s · {e.get('tokens1',0)} tokens)\n\n"
                      f"{e.get('response1','')}\n\n"
                      f"## {e.get('model2')} "
                      f"({e.get('tps2',0)} tok/s · {e.get('tokens2',0)} tokens)\n\n"
                      f"{e.get('response2','')}\n")
                path.write_text(md)

    ec = _ExportCompare.alloc().init()
    exp_btn.setTarget_(ec)
    handler._ds_ref = ds    # keep alive
    handler._ec_ref = ec    # keep alive

    NSApp.activateIgnoringOtherApps_(True)
    panel.makeKeyAndOrderFront_(None)
    NSApp.runModalForWindow_(panel)


def run_bench_history_panel() -> None:
    """Show benchmark history in a WKWebView modal."""
    html = _bench_history_html()
    W, H = 820, 580
    handler = _PanelHandler.alloc().init()
    panel = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    panel.setTitle_("Benchmark History")
    panel.setMinSize_((W, 400))
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
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
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
    "vllm_binary":    str(Path.home() / ".venv-vllm-metal/bin/vllm"),  # vllm-metal on Apple Silicon
    "vllm_port":      8001,            # default different from omlx/llama
    "vllm_models_dir": str(Path.home() / "models/vllm/"),
    "vllm_extra_args": "",             # extra CLI args passed to `vllm serve`
    "vllm_api_key":   "",              # optional Bearer token for vLLM server
    "vllm_hf_models": [],             # list of HF model IDs to show in vLLM menu
    "ollama_host":    "http://localhost:11434",  # Ollama base URL
    "ollama_api_key": "",             # optional (e.g. for remote Ollama behind auth)
    "opencode_config": str(Path.home() / ".config/opencode/opencode.json"),
    "restart_opencode": False,
    "terminal_app":   "Terminal",   # "Terminal" | "iTerm2"
    "aliases":        {},         # {model_name: alias_string}
    "model_notes":    {},         # {model_name: note_string}
    "model_tags":     {},         # {model_name: [tag, ...]}
    "model_schedule": [],         # [{time: "HH:MM", model: "name"}, ...]
    "schedule_enabled": False,    # whether schedule switching is active
    "model_params":   {},         # {model_name: {context, gpu_layers, max_tokens}}
    "hidden_models":  [],         # [model_name, ...]
    "pinned_models":  [],         # [model_name, ...] pinned to top of menu
    "model_load_times": {},       # {model_name: [t1, t2, t3]} rolling last-3 load times (seconds)
    "known_models":   [],         # [model_name, ...] all models ever seen (for new-model detection)
    "recent_models":  [],         # [model_name, ...] max 5, most recent first
    "default_model":  "",         # model name to auto-load on startup (empty = none)
    "qt_font_size":   12,         # Quick Test output font size
    # Feature flags
    "sync_env":         True,     # write ~/.config/switchman/env on switch
    "sync_aider":       False,    # write ~/.aider.conf.yml on switch
    "sync_zed":         False,    # write ~/.config/zed/settings.json on switch
    "notifications":    True,     # macOS notification when model is ready
    "on_switch_script": "",       # shell command to run after every model switch
    "hf_token":         "",       # HuggingFace token for gated model downloads
    "auto_reload_on_crash": False, # watchdog reloads last model after crash
    "sync_continue":    False,    # write ~/.continue/config.json on switch
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
            if not isinstance(cfg.get("pinned_models"), list):
                cfg["pinned_models"] = []
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


def scan_vllm(cfg: dict) -> dict[str, Path | str]:
    """Return vLLM models: local HF dirs + HF model IDs from config.

    Values are either a Path (local) or a str (HF model ID like 'org/name').
    """
    models: dict[str, Path | str] = {}

    # Local model directories
    root = Path(cfg.get("vllm_models_dir", ""))
    if root.exists():
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                if (entry / "config.json").exists() or (entry / "tokenizer_config.json").exists():
                    models[entry.name] = entry

    # HF model IDs configured manually (e.g. "mistralai/Mistral-7B-Instruct-v0.3")
    for hf_id in cfg.get("vllm_hf_models", []):
        if isinstance(hf_id, str) and hf_id.strip():
            display = hf_id.strip().split("/")[-1]   # use repo name as display
            models[display] = hf_id.strip()           # str = HF ID

    return models


def scan_ollama(cfg: dict) -> dict[str, str]:
    """Query running Ollama for its installed models.

    Returns {display_name: model_tag} where model_tag is the Ollama model
    identifier (e.g. 'llama3.2:latest'). Returns {} if Ollama is not running.
    """
    host = cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
    try:
        import urllib.request as _ur
        import json as _json
        req = _ur.Request(f"{host}/api/tags")
        api_key = cfg.get("ollama_api_key", "")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with _ur.urlopen(req, timeout=2) as r:
            data = _json.loads(r.read())
        models = {}
        for m in data.get("models", []):
            tag = m.get("name", "")
            if not tag:
                continue
            # Display: strip ":latest" suffix for cleaner menu names
            display = tag[:-7] if tag.endswith(":latest") else tag
            models[display] = tag   # value is full tag used in API calls
        return dict(sorted(models.items()))
    except Exception:
        return {}


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


def get_thermal_state() -> str:
    """Return macOS thermal state: nominal / fair / serious / critical."""
    try:
        r = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5)
        out = r.stdout.lower()
        if "cpu_speed_limit" in out:
            import re
            m = re.search(r"cpu_speed_limit\s*=\s*(\d+)", out)
            if m:
                limit = int(m.group(1))
                if limit <= 50:    return "critical"
                if limit <= 75:    return "serious"
                if limit <= 95:    return "fair"
        if "critical" in out:   return "critical"
        if "serious"  in out:   return "serious"
        if "fair"     in out:   return "fair"
    except Exception:
        pass
    return "nominal"


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


def record_model_load_time(cfg: dict, name: str, elapsed: float) -> None:
    """Store elapsed load time for a model, keeping the last 3 samples."""
    times = cfg.setdefault("model_load_times", {}).setdefault(name, [])
    times.insert(0, round(elapsed, 1))
    cfg["model_load_times"][name] = times[:3]
    save_config(cfg)


def get_model_load_estimate(cfg: dict, name: str) -> float | None:
    """Return rolling average of last load times, or None if no history."""
    times = cfg.get("model_load_times", {}).get(name, [])
    return round(sum(times) / len(times), 0) if times else None


def send_model_ready_notification(name: str, elapsed: float = 0.0) -> None:
    """Fire a local macOS notification via osascript.

    UNUserNotificationCenter requires a bundle ID, which a bare Python script
    doesn't have — permission requests silently fail.  osascript works from any
    context with no bundle or permission prompt needed.
    """
    try:
        import subprocess
        safe_name = name.replace('"', '\\"')
        elapsed_str = f" ({elapsed:.0f}s)" if elapsed > 0 else ""
        subprocess.Popen(
            ["osascript", "-e",
             f'display notification "Model ready: {safe_name}{elapsed_str}" with title "Switchman"'],
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


def sync_aider_config(port: int, model_name: str) -> None:
    """Write ~/.aider.conf.yml with the current model and port."""
    path = Path.home() / ".aider.conf.yml"
    try:
        path.write_text(
            f"openai-api-base: http://localhost:{port}/v1\n"
            f"openai-api-key: dummy\n"
            f"model: openai/{model_name}\n"
        )
    except Exception:
        pass


def sync_zed_config(port: int, model_name: str) -> None:
    """Write ~/.config/zed/settings.json assistant section."""
    path = Path.home() / ".config" / "zed" / "settings.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(path.read_text()) if path.exists() else {}
        cfg.setdefault("language_models", {}).setdefault("openai", {}).update({
            "api_url": f"http://localhost:{port}/v1",
            "available_models": [{"name": model_name, "display_name": model_name,
                                   "max_tokens": 32768}],
        })
        cfg.setdefault("assistant", {}).update({
            "default_model": {"provider": "openai", "model": model_name},
            "version": "2",
        })
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    except Exception:
        pass


def run_on_switch_script(script: str, model_name: str, port: int, kind: str) -> None:
    """Run user-defined shell script after model switch. Non-blocking."""
    if not script.strip():
        return
    env = {**os.environ,
           "SWITCHMAN_MODEL": model_name,
           "SWITCHMAN_PORT": str(port),
           "SWITCHMAN_KIND": kind}
    try:
        subprocess.Popen(script, shell=True, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def sync_continue_config(port: int, model_name: str) -> None:
    """Write ~/.continue/config.json with current model and API base."""
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
        entry["model"] = model_name
        entry["apiKey"] = "dummy"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    except Exception:
        pass


def active_api(cfg: dict, kind: str) -> tuple[int, str]:
    """Return (port, api_key) for the given backend kind."""
    if kind == "vllm":
        return cfg.get("vllm_port", 8001), cfg.get("vllm_api_key", "")
    elif kind == "gguf":
        return cfg.get("llama_port", 8080), ""
    elif kind == "ollama":
        host = cfg.get("ollama_host", "http://localhost:11434")
        try:
            port = int(host.rstrip("/").rsplit(":", 1)[-1])
        except (ValueError, IndexError):
            port = 11434
        return port, cfg.get("ollama_api_key", "")
    else:  # mlx / default
        return cfg.get("omlx_port", 8000), cfg.get("omlx_api_key", "")


def active_base_url(cfg: dict, kind: str) -> str:
    """Return full base URL (no trailing slash) for the given backend kind."""
    if kind == "ollama":
        return cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
    port, _ = active_api(cfg, kind)
    return f"http://localhost:{port}"


def sync_clients(cfg: dict, port: int, model_name: str = "", kind: str = "") -> None:
    """Sync all enabled external client configs. Safe to call from any thread."""
    if cfg.get("sync_env", True):
        sync_env_file(port)
    if cfg.get("sync_aider") and model_name:
        sync_aider_config(port, model_name)
    if cfg.get("sync_zed") and model_name:
        sync_zed_config(port, model_name)
    if cfg.get("sync_continue") and model_name:
        sync_continue_config(port, model_name)
    if cfg.get("on_switch_script") and model_name:
        run_on_switch_script(cfg["on_switch_script"], model_name, port, kind)


def estimate_model_memory_gb(path: Path) -> float:
    """Estimate unified memory needed for a model based on disk size (GB)."""
    try:
        if path.is_dir():
            size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            size_bytes = path.stat().st_size
        # Disk size ≈ weights. Add ~15% for runtime overhead + KV cache headroom.
        return (size_bytes / 1_073_741_824) * 1.15
    except Exception:
        return 0.0


def get_total_ram_gb() -> float:
    """Return total unified memory in GB via sysctl."""
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(r.stdout.strip()) / 1_073_741_824
    except Exception:
        return 0.0


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
        self._engine_filter = "All"   # "All" | "MLX" | "GGUF" | "VLLM" | "OLLAMA"
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
            # Text filter (name, alias, and tags)
            if q:
                tags = self._app._cfg.get("model_tags", {}).get(n, [])
                tag_match = any(q in t for t in tags)
                if (q not in (self._app._alias(n) or n).lower()
                        and q not in n.lower()
                        and not tag_match):
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
    seg = NSSegmentedControl.alloc().initWithFrame_(((_PAD, y), (280, 24)))
    seg.setSegmentCount_(5)
    seg.setLabel_forSegment_("All",    0)
    seg.setLabel_forSegment_("MLX",    1)
    seg.setLabel_forSegment_("GGUF",   2)
    seg.setLabel_forSegment_("vLLM",   3)
    seg.setLabel_forSegment_("Ollama", 4)
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
            labels = ["All", "MLX", "GGUF", "VLLM", "OLLAMA"]
            idx = sender.selectedSegment()
            delegate._engine_filter = labels[idx] if 0 <= idx < 5 else "All"
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


# ── Popover button handler ────────────────────────────────────────────────────

class _PopoverHandler(NSObject):
    """NSObject target for all NSPopover button actions.

    Needed because Switchman inherits from rumps.App (not NSObject), so its
    methods cannot be called as ObjC selectors via setTarget_/setAction_.
    """

    def initWithApp_(self, app):
        self = super().init()
        if self is None:
            return None
        self._app = app
        return self

    def popQuickTest_(self, _):
        self._app._open_test_prompt(None)
        if getattr(self._app, '_popover', None):
            self._app._popover.close()

    def popSwitchModel_(self, _):
        self._app._open_model_search(None)
        if getattr(self._app, '_popover', None):
            self._app._popover.close()

    def popBenchmark_(self, _):
        self._app._on_benchmark(None)
        if getattr(self._app, '_popover', None):
            self._app._popover.close()

    def popSettings_(self, _):
        self._app._open_settings(None)
        if getattr(self._app, '_popover', None):
            self._app._popover.close()

    def popLoadRecent_(self, sender):
        idx = sender.tag()
        recent = self._app._cfg.get("recent_models", [])
        if idx < len(recent):
            name = recent[idx]
            if name in self._app._model_map:
                class _S: pass
                s = _S(); s._model_name = name
                self._app._on_select(s)
        if getattr(self._app, '_popover', None):
            self._app._popover.close()


# ── App ───────────────────────────────────────────────────────────────────────

class Switchman(rumps.App):
    def __init__(self):
        super().__init__("⚡", quit_button=None)
        self._cfg = load_config()
        self._gguf_proc: subprocess.Popen | None = None
        self._vllm_proc: subprocess.Popen | None = None
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
        self._mem_pressure:  str = "nominal"
        self._thermal_state: str = "nominal"
        # Feature: error dialog
        self._pending_error: tuple | None = None
        # Feature: server crash watchdog
        self._watchdog_timer: rumps.Timer | None = None
        # Feature: loading step detail + load time
        self._load_status: str = "Loading model…"
        self._load_start_time: float = 0.0
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
        # Schedule timer: checks model schedule every 60 seconds
        self._schedule_timer = rumps.Timer(self._on_schedule_tick, 60)
        self._schedule_timer.start()
        self._schedule_last_fired: str | None = None  # "HH:MM" of last fired entry
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
        D_KEYCODE     = 2   # 'd' key

        def _callback(proxy, etype, event, refcon):
            try:
                keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                mods = event.flags() if hasattr(event, 'flags') else 0
                ALT  = kCGEventFlagMaskAlternate
                CMD  = kCGEventFlagMaskCommand
                if keycode == SPACE_KEYCODE and (mods & ALT):
                    if mods & CMD:
                        self._open_model_search(None)
                    else:
                        self._open_menu_from_hotkey()
                elif keycode == D_KEYCODE and (mods & ALT) and (mods & CMD):
                    self._load_default_model_hotkey()
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

    def _load_default_model_hotkey(self):
        """⌥⌘D — immediately load the ★ default model."""
        default = self._cfg.get("default_model", "")
        if not default or default not in self._model_map or self._loading:
            return
        class _S: pass
        s = _S(); s._model_name = default
        self._on_select(s)

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

    # ── Glass Popover ─────────────────────────────────────────────────────────

    def _make_popover(self):
        """Build a native NSPopover with full glass treatment."""
        from AppKit import (
            NSPopover, NSViewController, NSView, NSVisualEffectView,
            NSColor, NSFont, NSTextField, NSButton, NSProgressIndicator,
            NSBox,
        )

        # Create NSObject handler so button actions can be called as ObjC selectors
        self._pop_handler = _PopoverHandler.alloc().initWithApp_(self)

        PW = 320

        # ── helpers ──────────────────────────────────────────────────────────
        def _tf(text, frame, size=12, bold=False, color=None, align=0):
            f = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
            t = NSTextField.alloc().initWithFrame_(frame)
            t.setStringValue_(text); t.setBezeled_(False)
            t.setDrawsBackground_(False); t.setEditable_(False); t.setSelectable_(False)
            t.setFont_(f)
            t.setTextColor_(color or NSColor.labelColor())
            t.setAlignment_(align)  # 0=left, 1=right, 2=center
            return t

        def _sep(y):
            s = NSBox.alloc().initWithFrame_(((_PAD, y), (PW - _PAD*2, 1)))
            s.setBoxType_(2)
            return s

        def _glass_card(frame, radius=10):
            """Rounded frosted card — NSVisualEffectView inside a clipping view."""
            clip = NSView.alloc().initWithFrame_(frame)
            clip.setWantsLayer_(True)
            clip.layer().setCornerRadius_(radius)
            clip.layer().setMasksToBounds_(True)
            vfx = NSVisualEffectView.alloc().initWithFrame_(
                ((0, 0), frame[1]))
            vfx.setMaterial_(8)   # NSVisualEffectMaterialMenu — denser glass
            vfx.setBlendingMode_(0)   # NSVisualEffectBlendingModeBehindWindow
            vfx.setState_(1)      # NSVisualEffectStateActive
            clip.addSubview_(vfx)
            return clip, vfx

        def _pill_btn(title, target, action, frame, accent=False):
            b = NSButton.alloc().initWithFrame_(frame)
            b.setTitle_(title)
            b.setBezelStyle_(4)   # NSBezelStyleRounded
            b.setTarget_(target)
            b.setAction_(action)
            if accent:
                b.setKeyEquivalent_("\r")
            return b

        # ── content view ─────────────────────────────────────────────────────
        # Calculate total height dynamically
        HEADER_H  = 90   # dot + model name + meta + tps
        SPARK_H   = 36
        STATS_H   = 58   # memory + thermal
        BTNS_H    = 72   # 2 rows × 32 + gaps
        RECENT_H  = 18 + 5 * 32   # header + 5 rows
        PADDING   = _PAD * 2
        PH = HEADER_H + SPARK_H + STATS_H + BTNS_H + RECENT_H + PADDING + 16

        cv = NSView.alloc().initWithFrame_(((0, 0), (PW, PH)))

        y = PH - _PAD

        # ── header card ──────────────────────────────────────────────────────
        card_h = HEADER_H - 4
        hdr_card, hdr_vfx = _glass_card(((_PAD, y - card_h), (PW - _PAD*2, card_h)), radius=12)
        cv.addSubview_(hdr_card)

        # Status dot (green circle via layer)
        dot_view = NSView.alloc().initWithFrame_(((12, card_h - 18), (10, 10)))
        dot_view.setWantsLayer_(True)
        dot_view.layer().setCornerRadius_(5)
        dot_view.layer().setBackgroundColor_(NSColor.systemGreenColor().CGColor())
        hdr_vfx.addSubview_(dot_view)
        self._pop_dot = dot_view

        # Model name — large, bold
        model_lbl = _tf("", ((26, card_h - 22), (PW - _PAD*2 - 28, 20)),
                        size=14, bold=True)
        hdr_vfx.addSubview_(model_lbl); self._pop_model_lbl = model_lbl

        # Meta row
        meta_lbl = _tf("", ((_PAD - 4, card_h - 38), (PW - _PAD*2 - 8, 14)),
                        size=10, color=NSColor.secondaryLabelColor())
        hdr_vfx.addSubview_(meta_lbl); self._pop_meta_lbl = meta_lbl

        # tok/s
        tps_lbl = _tf("", ((_PAD - 4, card_h - 54), (PW - _PAD*2 - 8, 13)),
                      size=10, color=NSColor.secondaryLabelColor())
        hdr_vfx.addSubview_(tps_lbl); self._pop_tps_lbl = tps_lbl

        # Indeterminate load bar inside header card (hidden by default)
        load_bar = NSProgressIndicator.alloc().initWithFrame_(
            ((8, 4), (PW - _PAD*2 - 16, 4)))
        load_bar.setStyle_(0); load_bar.setIndeterminate_(True)
        load_bar.setControlSize_(3)   # NSControlSizeMini
        load_bar.setHidden_(True)
        hdr_vfx.addSubview_(load_bar); self._pop_load_bar = load_bar

        y -= card_h + 8

        # ── spark chart card ─────────────────────────────────────────────────
        spark_card, spark_vfx = _glass_card(((_PAD, y - SPARK_H), (PW - _PAD*2, SPARK_H)), radius=10)
        cv.addSubview_(spark_card)

        spark_lbl = _tf("TOK/S", ((8, SPARK_H - 13), (40, 11)),
                        size=8, bold=True, color=NSColor.tertiaryLabelColor())
        spark_vfx.addSubview_(spark_lbl)

        spark_inner = NSView.alloc().initWithFrame_(((50, 6), (PW - _PAD*2 - 58, SPARK_H - 12)))
        spark_vfx.addSubview_(spark_inner)
        self._pop_spark_view = spark_inner
        self._pop_tps_history = []

        y -= SPARK_H + 6

        # ── stats card (memory + thermal) ────────────────────────────────────
        stats_card, stats_vfx = _glass_card(((_PAD, y - STATS_H), (PW - _PAD*2, STATS_H)), radius=10)
        cv.addSubview_(stats_card)

        # Memory row  — "🟢 Memory" label (left) + bar (center) + pct (right)
        cw = PW - _PAD*2   # card interior width = 304
        mem_lbl = _tf("🟢 Memory", ((8, STATS_H - 18), (90, 14)),
                      size=10, color=NSColor.secondaryLabelColor())
        stats_vfx.addSubview_(mem_lbl); self._pop_mem_lbl = mem_lbl

        mem_bar = NSProgressIndicator.alloc().initWithFrame_(
            ((104, STATS_H - 14), (cw - 104 - 36, 6)))
        mem_bar.setStyle_(0); mem_bar.setIndeterminate_(False)
        mem_bar.setControlSize_(3)
        mem_bar.setMinValue_(0); mem_bar.setMaxValue_(100)
        mem_bar.setDoubleValue_(0)
        stats_vfx.addSubview_(mem_bar); self._pop_mem_bar = mem_bar

        mem_pct_lbl = _tf("0%", ((cw - 32, STATS_H - 18), (30, 14)),
                          size=10, color=NSColor.secondaryLabelColor(), align=2)
        stats_vfx.addSubview_(mem_pct_lbl); self._pop_mem_pct_lbl = mem_pct_lbl

        # Thermal row — full-width label
        thermal_val = _tf("🌡 Thermal  Nominal", ((8, STATS_H - 36), (cw - 8, 14)),
                          size=10, color=NSColor.secondaryLabelColor())
        stats_vfx.addSubview_(thermal_val); self._pop_thermal_lbl = thermal_val

        y -= STATS_H + 8

        # ── action buttons ────────────────────────────────────────────────────
        bw = (PW - _PAD*2 - 8) // 2
        _ph = self._pop_handler
        for i, (title, method_fn, is_accent) in enumerate([
            ("💬 Quick Test",   _ph.popQuickTest_,   True),
            ("⇄ Switch Model", _ph.popSwitchModel_, False),
            ("⏱ Benchmark",    _ph.popBenchmark_,   False),
            ("⚙ Settings",     _ph.popSettings_,    False),
        ]):
            row = i // 2
            col = i % 2
            bx = _PAD + col * (bw + 8)
            by = y - (row + 1) * 32 - row * 6
            b = _pill_btn(title, _ph, objc.selector(method_fn, signature=b"v@:@"),
                          ((bx, by), (bw, 28)), accent=is_accent)
            cv.addSubview_(b)

        y -= BTNS_H

        # ── recent models ─────────────────────────────────────────────────────
        cv.addSubview_(_tf("RECENT", ((_PAD, y - 14), (80, 11)),
                           size=8, bold=True, color=NSColor.tertiaryLabelColor()))
        y -= 18

        self._pop_recent_btns = []
        for i in range(5):
            row_card, row_vfx = _glass_card(((_PAD, y - 28), (PW - _PAD*2, 28)), radius=8)
            cv.addSubview_(row_card)

            name_lbl = _tf("", ((10, 7), (PW - _PAD*2 - 100, 14)),
                           size=11, color=NSColor.labelColor())
            row_vfx.addSubview_(name_lbl)

            kind_lbl = _tf("", ((PW - _PAD*2 - 90, 7), (80, 14)),
                           size=9, color=NSColor.secondaryLabelColor(), align=2)
            row_vfx.addSubview_(kind_lbl)

            # invisible button over the whole card
            tap = NSButton.alloc().initWithFrame_(((0, 0), (PW - _PAD*2, 28)))
            tap.setTitle_(""); tap.setBordered_(False)
            tap.setTarget_(self._pop_handler)
            tap.setAction_(objc.selector(self._pop_handler.popLoadRecent_, signature=b"v@:@"))
            tap.setTag_(i)
            row_vfx.addSubview_(tap)

            row_card.setHidden_(True)
            self._pop_recent_btns.append((row_card, name_lbl, kind_lbl, tap))
            y -= 32

        # ── wire into NSPopover ───────────────────────────────────────────────
        vc = NSViewController.alloc().init()
        vc.setView_(cv)

        popover = NSPopover.alloc().init()
        popover.setContentViewController_(vc)
        popover.setContentSize_((PW, PH))
        popover.setBehavior_(1)   # NSPopoverBehaviorTransient — closes on click outside
        popover.setAnimates_(True)

        self._popover = popover
        self._pop_vc  = vc        # keep alive
        return popover

    def _open_popover(self, _=None):
        """Toggle the glass popover anchored to the status bar button."""
        if not getattr(self, '_popover', None):
            self._make_popover()

        popover = self._popover
        if popover.isShown():
            popover.close()
            return

        self._pop_refresh()

        # Get the status bar button to anchor to.
        # SwitchmanApp inherits from rumps.App so self._status_item IS the NSStatusItem.
        btn = None
        try:
            from AppKit import NSRectEdgeMaxY
            si = getattr(self, '_status_item', None)
            if si is not None:
                btn = si.button()
        except Exception:
            btn = None
        if btn:
            popover.showRelativeToRect_ofView_preferredEdge_(
                btn.bounds(), btn, NSRectEdgeMaxY)
        else:
            # Fallback: use a dummy NSView anchored to screen top-right
            try:
                from AppKit import NSScreen, NSView, NSRectEdgeMaxY as _EDGE
                PW, PH = int(popover.contentSize().width), int(popover.contentSize().height)
                vf = NSScreen.mainScreen().frame()
                # Place anchor 1pt square at top-right of screen
                dummy = NSView.alloc().initWithFrame_(((vf.size.width - PW - 8, vf.size.height - 1), (1, 1)))
                # Must have a window — use content view of a temp panel trick:
                # Instead fall back to showing relative to main window content view
                from AppKit import NSApp
                mw = NSApp.mainWindow() or (NSApp.windows()[0] if NSApp.windows() else None)
                if mw:
                    cv = mw.contentView()
                    popover.showRelativeToRect_ofView_preferredEdge_(
                        cv.bounds(), cv, _EDGE)
            except Exception as e:
                print(f"[popover fallback error] {e}")

        # Start refresh timer
        if not getattr(self, '_pop_timer', None):
            self._pop_timer = rumps.Timer(self._pop_tick_, 2.0)
            self._pop_timer.start()

    def _pop_refresh(self):
        """Update all popover labels with current state."""
        try:
            import subprocess
            result = subprocess.run(
                ['memory_pressure'], capture_output=True, text=True, timeout=2)
            pct_str = ""
            for line in result.stdout.splitlines():
                if "System memory pressure" in line or "percentage" in line.lower():
                    import re
                    m = re.search(r'(\d+)%', line)
                    if m:
                        pct_str = m.group(1)
            mem_pct = int(pct_str) if pct_str else 0
        except Exception:
            mem_pct = 0

        # Loading progress bar
        load_bar = getattr(self, '_pop_load_bar', None)
        if load_bar:
            if self._loading:
                load_bar.setHidden_(False)
                load_bar.startAnimation_(None)
            else:
                load_bar.stopAnimation_(None)
                load_bar.setHidden_(True)

        if self._loading and self._active:
            self._pop_model_lbl.setStringValue_(f"⏳ Loading {self._display(self._active)}…")
            status = getattr(self, '_load_status', 'Loading…')
            estimate = getattr(self, '_load_time_estimate', None)
            if estimate and self._load_start_time:
                elapsed = __import__('time').time() - self._load_start_time
                remaining = max(0, int(estimate - elapsed))
                status = f"~{remaining}s remaining"
            self._pop_meta_lbl.setStringValue_(status)
        elif self._active:
            self._pop_model_lbl.setStringValue_(self._display(self._active))
            kind = self._model_kind(self._active).upper()
            path = self._model_map.get(self._active, (None, None))[0]
            size_str = ""
            if path and path.exists():
                try:
                    size_b = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.is_dir() else path.stat().st_size
                    size_str = f" · {size_b/1e9:.1f} GB"
                except Exception:
                    pass
            ctx_str = ""
            if self._ctx_used and self._ctx_max:
                ctx_str = f" · ctx {int(self._ctx_used/self._ctx_max*100)}%"
            self._pop_meta_lbl.setStringValue_(f"{kind}{size_str}{ctx_str}")
        else:
            self._pop_model_lbl.setStringValue_("No model loaded")
            self._pop_meta_lbl.setStringValue_("")

        # Memory row
        mem_dot = "🔴" if self._mem_pressure == "critical" else ("🟡" if self._mem_pressure == "fair" else "🟢")
        self._pop_mem_lbl.setStringValue_(f"{mem_dot} Memory")
        self._pop_mem_bar.setDoubleValue_(mem_pct)
        self._pop_mem_pct_lbl.setStringValue_(f"{mem_pct}%")

        # Thermal row
        thermal_map = {"nominal": "Nominal ✓", "fair": "Fair ⚠", "serious": "Serious ⚠⚠", "critical": "Critical 🔥"}
        thermal_str = thermal_map.get(self._thermal_state, self._thermal_state.title())
        self._pop_thermal_lbl.setStringValue_(f"🌡 Thermal  {thermal_str}")

        # tok/s from tps poll + spark chart
        tps_val = getattr(self, '_pop_last_tps', 0.0)
        dot = "🟢" if tps_val > 5 else ("🟡" if tps_val > 0 else "⚫")
        self._pop_tps_lbl.setStringValue_(f"{dot} {tps_val:.1f} tok/s")
        # Update spark chart history
        if not hasattr(self, '_pop_tps_history'):
            self._pop_tps_history = []
        if tps_val > 0:
            self._pop_tps_history.append(tps_val)
            if len(self._pop_tps_history) > 10:
                self._pop_tps_history.pop(0)
        if hasattr(self, '_pop_spark_view'):
            _update_spark_chart(self._pop_spark_view, self._pop_tps_history)

        # Recent models — _pop_recent_btns is list of (row_card, name_lbl, kind_lbl, tap)
        recent = self._cfg.get("recent_models", [])
        for i, row_tuple in enumerate(self._pop_recent_btns):
            row_card, name_lbl, kind_lbl, tap = row_tuple
            if i < len(recent):
                name = recent[i]
                display = self._display(name)
                if len(display) > 38:
                    display = display[:36] + "…"
                name_lbl.setStringValue_(display)
                kind_lbl.setStringValue_(self._model_kind(name).upper())
                tap.setTag_(i)
                row_card.setHidden_(False)
            else:
                row_card.setHidden_(True)

    def _pop_tick_(self, timer):
        """Refresh popover every 2s (every 0.5s while loading)."""
        if not getattr(self, '_popover', None) or not self._popover.isShown():
            timer.stop()
            self._pop_timer = None
            return
        # Speed up refresh during model load
        target_interval = 0.5 if self._loading else 2.0
        if abs(timer.timeInterval - target_interval) > 0.1:
            timer.stop()
            self._pop_timer = rumps.Timer(self._pop_tick_, target_interval)
            self._pop_timer.start()
        self._pop_refresh()

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
            self._mem_pressure  = get_memory_pressure()
            self._thermal_state = get_thermal_state()
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

    def _on_schedule_tick(self, _timer):
        if not self._cfg.get("schedule_enabled"):
            return
        schedule = self._cfg.get("model_schedule", [])
        if not schedule:
            return
        import datetime
        now = datetime.datetime.now().strftime("%H:%M")
        for entry in schedule:
            t = entry.get("time", "")
            model = entry.get("model", "")
            if t == now and t != self._schedule_last_fired:
                if model in self._model_map and model != self._active and not self._loading:
                    self._schedule_last_fired = t
                    self._load_model_by_name(model)
                    send_model_ready_notification(model)
                break
        # Reset last_fired when minute changes
        if self._schedule_last_fired and self._schedule_last_fired != now:
            self._schedule_last_fired = None

    def _open_schedule_panel(self, _=None):
        run_schedule_panel(self._cfg, self._model_map)
        self._cfg = load_config()

    def _on_watchdog_tick(self, _timer):
        if not self._active or self._loading:
            return
        def _check():
            try:
                kind = self._model_kind(self._active or "")
                port, api_key = active_api(self._cfg, kind)
                base = active_base_url(self._cfg, kind)
                # For vLLM: also check if the process itself died
                if kind == "vllm" and getattr(self, '_vllm_proc', None):
                    if self._vllm_proc.poll() is not None:
                        raise RuntimeError("vllm process exited")
                req = urllib.request.Request(
                    f"{base}/health",
                    headers={"Authorization": f"Bearer {api_key}"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    r.read()
                # Server is alive — do nothing
            except Exception:
                # Server is down
                crashed_model = self._active
                self._active = None
                self._last_toks = None
                self._stop_tps_poll()
                self._rebuild_pending = True
                if self._cfg.get("notifications", True):
                    try:
                        content = UNMutableNotificationContent.alloc().init()
                        content.setTitle_("Switchman")
                        body = "Inference server stopped unexpectedly"
                        if self._cfg.get("auto_reload_on_crash") and crashed_model:
                            body += f" — reloading {crashed_model}…"
                        content.setBody_(body)
                        req2 = UNNotificationRequest.requestWithIdentifier_content_trigger_(
                            str(uuid.uuid4()), content, None)
                        def _noop(e): pass
                        UNUserNotificationCenter.currentNotificationCenter() \
                            .addNotificationRequest_withCompletionHandler_(req2, _noop)
                    except Exception:
                        pass
                if self._cfg.get("auto_reload_on_crash") and crashed_model \
                        and crashed_model in self._model_map:
                    time.sleep(2)   # brief pause before restart
                    self._switch_token += 1
                    token = self._switch_token
                    self._active = crashed_model
                    self._loading = True
                    self._rebuild_pending = True
                    path, kind = self._model_map[crashed_model]
                    if kind == "mlx":
                        threading.Thread(
                            target=self._switch_mlx,
                            args=(crashed_model, token), daemon=True).start()
                    else:
                        threading.Thread(
                            target=self._switch_gguf,
                            args=(crashed_model, path, token), daemon=True).start()
        threading.Thread(target=_check, daemon=True).start()

    def _on_tps_tick(self, _timer):
        pass  # probe removed; method kept so any lingering timer refs are harmless

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self):
        self.menu._menu.removeAllItems()
        self.menu.clear()

        mlx = scan_mlx(self._cfg)
        gguf = scan_gguf(self._cfg)
        vllm_models = scan_vllm(self._cfg)
        ollama_models = scan_ollama(self._cfg)
        self._model_map = {
            **{n: (p, "mlx") for n, p in mlx.items()},
            **{n: (p, "gguf") for n, p in gguf.items()},
            **{n: (p, "vllm") for n, p in vllm_models.items()},
            **{n: (tag, "ollama") for n, tag in ollama_models.items()},
        }

        hidden = set(self._cfg["hidden_models"])
        mlx_visible    = {n: p for n, p in mlx.items()         if n not in hidden}
        gguf_visible   = {n: p for n, p in gguf.items()        if n not in hidden}
        vllm_visible   = {n: p for n, p in vllm_models.items() if n not in hidden}
        ollama_visible = {n: t for n, t in ollama_models.items() if n not in hidden}
        hidden_present = {n for n in hidden if n in self._model_map}

        if self._loading and self._active:
            status_text = f"Loading {self._display(self._active)}…"
        elif self._active:
            status_text = f"● {self._display(self._active)}"
        else:
            status_text = "No model running"

        menu: list = [rumps.MenuItem(status_text, callback=None), None]

        # ── Pinned models section ─────────────────────────────────────────────────
        pinned = self._cfg.get("pinned_models", [])
        pinned_visible = [n for n in pinned if n in self._model_map and n not in hidden]
        if pinned_visible:
            menu.append(_menu_header("── Pinned ──"))
            for pname in pinned_visible:
                pitem = rumps.MenuItem(f"  {self._display(pname)}", callback=self._on_select)
                pitem._model_name = pname
                if self._active == pname and not self._loading:
                    pitem.state = 1
                menu.append(pitem)
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

        if vllm_visible:
            menu.append(_menu_header("── vLLM ──"))
            for name in vllm_visible:
                menu.append(self._make_model_item(name))
            menu.append(None)

        if ollama_visible:
            menu.append(_menu_header("── Ollama ──"))
            for name in ollama_visible:
                menu.append(self._make_model_item(name))
            menu.append(None)

        if not mlx_visible and not gguf_visible and not vllm_visible and not ollama_visible and not hidden_present:
            menu.append(rumps.MenuItem("No models found", callback=None))
            menu.append(None)

        menu += [
            _sf_item("Stop Engine",               "stop.circle",            self._stop),
            _sf_item("Refresh Models",             "arrow.clockwise",        self._refresh),
            _sf_item("Search Models…",             "magnifyingglass",        self._open_model_search),
            _sf_item("Download from HuggingFace…", "arrow.down.to.line",     self._open_hf_download),
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
            from AppKit import (NSAttributedString, NSColor,
                                NSForegroundColorAttributeName)
            dot_attrs = {NSForegroundColorAttributeName: NSColor.systemGreenColor()}
            dot = NSAttributedString.alloc().initWithString_attributes_("● ", dot_attrs)
            from AppKit import NSMutableAttributedString
            full = NSMutableAttributedString.alloc().initWithString_(
                f"{prefix}{alias or name}")
            full.insertAttributedString_atIndex_(dot, 0)
            parent._menuitem.setAttributedTitle_(full)

        meta = self._get_model_meta(name)
        note = self._cfg["model_notes"].get(name, "")
        tags = self._cfg.get("model_tags", {}).get(name, [])
        for meta_label in meta:
            parent.add(rumps.MenuItem(meta_label, callback=None))
        if tags:
            parent.add(rumps.MenuItem(f"  🏷 {', '.join(tags)}", callback=None))
        if meta or tags:
            parent.add(None)
        if note:
            parent._menuitem.setToolTip_(note)

        sel = _sf_item("Select", "play.fill", self._on_select)
        sel._model_name = name
        parent.add(sel)

        copy_item = _sf_item("Copy model ID", "doc.on.doc", self._on_copy_model_id)
        copy_item._model_name = name
        parent.add(copy_item)

        is_default = self._cfg.get("default_model") == name
        default_item = _sf_item(
            "Default at startup" if not is_default else "Default at startup ✓",
            "star.fill" if is_default else "star",
            self._on_set_default)
        default_item._model_name = name
        parent.add(default_item)

        is_pinned = name in self._cfg.get("pinned_models", [])
        pin_item = _sf_item(
            "Unpin from top" if is_pinned else "Pin to top",
            "pin.slash" if is_pinned else "pin",
            self._on_toggle_pin)
        pin_item._model_name = name
        parent.add(pin_item)

        hide_item = _sf_item("Hide", "eye.slash", self._on_hide_model)
        hide_item._model_name = name
        parent.add(hide_item)

        delete_item = _sf_item("Delete model…", "trash", self._on_delete_model)
        delete_item._model_name = name
        parent.add(delete_item)
        parent.add(None)

        settings_item = _sf_item("Settings…", "gearshape", self._open_model_settings)
        settings_item._model_name = name
        parent.add(settings_item)

        benchmark_item = _sf_item("Benchmark…", "timer", self._on_benchmark)
        benchmark_item._model_name = name
        parent.add(benchmark_item)

        return parent

    def _build_settings_menu(self) -> rumps.MenuItem:
        s = _sf_item("Settings", "gearshape")
        # Memory pressure indicator
        dot = {"nominal": "🟢", "warn": "🟡", "critical": "🔴"}.get(
            self._mem_pressure, "⚪")
        s.add(rumps.MenuItem(f"{dot}  Memory: {self._mem_pressure}", callback=None))
        thermal_dot = {"nominal": "🟢", "fair": "🟡", "serious": "🟠", "critical": "🔴"}.get(
            self._thermal_state, "⚪")
        s.add(rumps.MenuItem(f"{thermal_dot}  Thermal: {self._thermal_state}", callback=None))
        s.add(None)
        s.add(rumps.MenuItem("  ◎ Dashboard…", callback=self._open_popover))
        s.add(rumps.MenuItem("  Open Settings…", callback=self._open_settings))
        s.add(rumps.MenuItem("  Manage Profiles…", callback=self._open_profiles))
        s.add(rumps.MenuItem("  Manage Visible Models…", callback=self._open_manage_models))
        s.add(None)
        s.add(rumps.MenuItem("  Quick Test Prompt…", callback=self._open_test_prompt))
        s.add(rumps.MenuItem("  Compare History…", callback=self._open_compare_history))
        s.add(rumps.MenuItem("  Model Schedule…", callback=self._open_schedule_panel))
        s.add(rumps.MenuItem("  Benchmark History…", callback=self._open_bench_history))
        s.add(rumps.MenuItem("  Server Logs…", callback=self._open_server_logs))
        s.add(None)
        s.add(rumps.MenuItem("  Export Settings…", callback=self._export_settings))
        s.add(rumps.MenuItem("  Import Settings…", callback=self._import_settings))
        s.add(None)
        kind = self._model_kind(self._active) if self._active else "mlx"
        port, _ = active_api(self._cfg, kind)
        copy_item = _sf_item(f"Copy API URL  (:{port}/v1)", "link.circle", self._copy_api_url)
        s.add(copy_item)
        return s

    

    def _build_hidden_menu(self, hidden_present: set) -> rumps.MenuItem:
        s = _sf_item(f"Hidden ({len(hidden_present)})", "eye.slash")
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
                base = f"⚡{mem_dot} {self._title_label(self._active)}"
                if self._ctx_used and self._ctx_max:
                    ctx_pct = int(self._ctx_used / self._ctx_max * 100)
                    base += f" {ctx_pct}%"
                self.title = base
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
        if self._benchmarking:
            label = "Benchmarking…"
        else:
            label = self._load_status
            estimate = getattr(self, '_load_time_estimate', None)
            if estimate and self._load_start_time:
                elapsed = time.time() - self._load_start_time
                if estimate > elapsed:
                    label += f" ~{int(estimate - elapsed)}s"
                else:
                    label += " …"
        self.title = label if self._flash_state else "⚡"

    # ── Model callbacks ───────────────────────────────────────────────────────

    def _update_recent(self, name: str):
        recent = self._cfg.setdefault("recent_models", [])
        if name in recent:
            recent.remove(name)
        recent.insert(0, name)
        self._cfg["recent_models"] = recent[:5]
        save_config(self._cfg)

    def _load_model_by_name(self, name: str, skip_memory_check: bool = False):
        """Programmatically load a model by name (must be called on the main thread)."""
        entry = self._model_map.get(name)
        if entry is None:
            return
        path, kind = entry
        if not skip_memory_check:
            est_gb = estimate_model_memory_gb(path)
            total_gb = get_total_ram_gb()
            if est_gb > 0 and total_gb > 0 and est_gb > total_gb - 6:
                from AppKit import NSAlert
                alert = NSAlert.alloc().init()
                alert.setMessageText_("Model may not fit in memory")
                alert.setInformativeText_(
                    f"{name}\nestimated {est_gb:.1f} GB — your Mac has {total_gb:.0f} GB total.\n\n"
                    "Loading may cause heavy swap or a crash. Continue?")
                alert.addButtonWithTitle_("Load Anyway")
                alert.addButtonWithTitle_("Cancel")
                if alert.runModal() != 1000:
                    return
        self._last_toks = None
        self._ctx_used = None
        self._ctx_max = None
        self._stop_tps_poll()
        self._update_recent(name)
        self._switch_token += 1
        token = self._switch_token
        self._active = name
        self._loading = True
        self._load_start_time = time.time()
        self._load_time_estimate = get_model_load_estimate(self._cfg, name)
        self._update_title()
        if kind == "mlx":
            threading.Thread(target=self._switch_mlx, args=(name, token), daemon=True).start()
        elif kind == "vllm":
            threading.Thread(target=self._switch_vllm, args=(name, path, token), daemon=True).start()
        elif kind == "ollama":
            # path is the Ollama model tag string (e.g. "llama3.2:latest")
            threading.Thread(target=self._switch_ollama, args=(name, str(path), token), daemon=True).start()
        else:
            threading.Thread(target=self._switch_gguf, args=(name, path, token), daemon=True).start()

    def _on_select(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None) or sender.title.strip()
        self._load_model_by_name(name)

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
        deadline = time.time() + 600
        while time.time() < deadline:
            if self._superseded(token): return
            resp = http_post(url, body={
                "model": name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                **sampling,
            }, headers=headers, timeout=600)
            if resp and resp.get("model") == name:
                break
            time.sleep(2)

        if self._superseded(token): return
        self._load_status = "Warming up…"
        http_post(url, body={
            "model": name,
            "messages": [{"role": "user", "content":
                "Write a detailed technical explanation of how transformer attention "
                "mechanisms work, including multi-head attention, key/query/value "
                "projections, and softmax scaling. Include Python code examples. "
                "Be thorough and complete."}],
            "max_tokens": 2048,
            **sampling,
        }, headers=headers, timeout=600)

        if self._superseded(token): return
        set_opencode_model(
            self._cfg, "omlx", name,
            display_name=self._display(name),
            context=p["context"],
            max_tokens=p["max_tokens"],
            sampling=sampling,
        )
        sync_clients(self._cfg, self._cfg.get("omlx_port", 8000), model_name=name, kind="mlx")
        elapsed = time.time() - self._load_start_time if self._load_start_time else 0.0
        record_model_load_time(self._cfg, name, elapsed)
        if self._cfg.get("notifications", True):
            send_model_ready_notification(name, elapsed)
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
        sync_clients(self._cfg, self._cfg.get("llama_port", 8000), model_name=model_id, kind="gguf")
        elapsed = time.time() - self._load_start_time if self._load_start_time else 0.0
        record_model_load_time(self._cfg, name, elapsed)
        if self._cfg.get("notifications", True):
            send_model_ready_notification(name, elapsed)
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
        # Unload Ollama model from memory if one is active
        if self._active and self._model_kind(self._active) == "ollama":
            entry = self._model_map.get(self._active)
            if entry:
                self._unload_ollama(str(entry[0]))
        self._kill_gguf()
        self._kill_vllm()
        omlx_stop(self._cfg)
        self._active = None
        self._loading = False
        self._stop_tps_poll()
        self._stop_watchdog()
        self._rebuild_pending = True

    def _unload_ollama(self, model_tag: str):
        """Ask Ollama to evict a model from memory (keep_alive=0)."""
        host = self._cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
        api_key = self._cfg.get("ollama_api_key", "")
        try:
            import json as _json
            body = _json.dumps({"model": model_tag, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{host}/api/generate", data=body,
                headers={"Content-Type": "application/json"})
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            pass

    def _switch_ollama(self, name: str, model_tag: str, token: int):
        """Switch to an Ollama model — no process management needed.

        Ollama is a system service; we just warm it up with the new model.
        """
        self._load_status = "Checking Ollama…"

        # Unload any currently active Ollama model to free memory first
        if self._active and self._model_kind(self._active) == "ollama":
            entry = self._model_map.get(self._active)
            if entry:
                self._unload_ollama(str(entry[0]))

        # Stop non-Ollama engines
        self._kill_gguf()
        self._kill_vllm()
        omlx_stop(self._cfg)
        if self._superseded(token):
            return

        host = self._cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
        api_key = self._cfg.get("ollama_api_key", "")

        # Verify Ollama is running
        try:
            req = urllib.request.Request(f"{host}/api/tags")
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            if self._superseded(token):
                return
            self._active = None
            self._pending_error = (
                "Ollama not reachable",
                f"Could not connect to Ollama at {host}.\n\n"
                "Make sure Ollama is installed and running:\n  ollama serve",
            )
            self._loading = False
            self._rebuild_pending = True
            return

        if self._superseded(token):
            return

        # Warm up: send a minimal request to trigger model load
        self._load_status = "Loading model…"
        import json as _json
        body = _json.dumps({
            "model": model_tag,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            req = urllib.request.Request(
                f"{host}/v1/chat/completions", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                r.read()
        except Exception as e:
            if self._superseded(token):
                return
            self._active = None
            self._pending_error = (
                "Ollama model failed to load",
                f"Model '{model_tag}' could not be loaded.\n\n"
                f"Try: ollama pull {model_tag}\n\nError: {e}",
            )
            self._loading = False
            self._rebuild_pending = True
            return

        if self._superseded(token):
            return

        p = self._params(name)
        sync_clients(self._cfg, 11434, model_name=model_tag, kind="ollama")
        elapsed = time.time() - self._load_start_time if self._load_start_time else 0.0
        record_model_load_time(self._cfg, name, elapsed)
        if self._cfg.get("notifications", True):
            send_model_ready_notification(name, elapsed)
        self._start_poll_on_rebuild = True
        self._loading = False
        if self._cfg["restart_opencode"]:
            restart_opencode(self._cfg)
        self._rebuild_pending = True

    def _kill_gguf(self):
        if self._gguf_proc and self._gguf_proc.poll() is None:
            self._gguf_proc.terminate()
            try:
                self._gguf_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._gguf_proc.kill()
        self._gguf_proc = None

    def _kill_vllm(self):
        if getattr(self, '_vllm_proc', None) and self._vllm_proc.poll() is None:
            self._vllm_proc.terminate()
            try:
                self._vllm_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._vllm_proc.kill()
        self._vllm_proc = None

    def _switch_vllm(self, name: str, path, token: int):
        """Stop other engines, start vllm serve, wait for readiness.

        path may be a pathlib.Path (local model dir) or a str (HF model ID).
        """
        self._load_status = "Stopping engine…"
        self._kill_gguf()
        self._kill_vllm()
        omlx_stop(self._cfg)
        if self._superseded(token):
            return

        port = self._cfg.get("vllm_port", 8001)
        if not port_is_free(port):
            kill_port(port)

        if self._superseded(token):
            return

        self._load_status = "Starting vLLM…"
        binary = self._cfg.get("vllm_binary", "vllm")
        extra = self._cfg.get("vllm_extra_args", "").split()
        p = self._params(name)
        # path is str (HF model ID) or Path (local dir)
        model_arg = str(path)
        cmd = [
            binary, "serve", model_arg,
            "--port", str(port),
            "--served-model-name", name,   # so API model ID matches Switchman name
            "--max-model-len", str(p.get("context", 4096)),
        ] + extra

        try:
            self._vllm_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            if self._superseded(token):
                return
            self._active = None
            self._pending_error = (
                "vLLM not found",
                f"Could not launch '{binary}'. Set the vLLM binary path in Settings → Inference.",
            )
            self._loading = False
            self._rebuild_pending = True
            return

        self._load_status = "Loading weights…"
        if not wait_for_port_open(port, timeout=300):
            if self._superseded(token):
                return
            self._active = None
            self._pending_error = (
                "vLLM failed to start",
                f"Port {port} did not open within 5 minutes.",
            )
            self._loading = False
            self._rebuild_pending = True
            return

        if self._superseded(token):
            return

        # Query the loaded model ID from /v1/models
        model_id = name
        try:
            import urllib.request as _ur
            import json as _json
            with _ur.urlopen(f"http://localhost:{port}/v1/models", timeout=10) as r:
                data = _json.loads(r.read())
                ids = [m["id"] for m in data.get("data", [])]
                if ids:
                    model_id = ids[0]
        except Exception:
            pass

        sync_clients(self._cfg, port, model_name=model_id, kind="vllm")
        elapsed = time.time() - self._load_start_time if self._load_start_time else 0.0
        record_model_load_time(self._cfg, name, elapsed)
        if self._cfg.get("notifications", True):
            send_model_ready_notification(name, elapsed)
        self._start_poll_on_rebuild = True
        self._loading = False
        if self._cfg["restart_opencode"]:
            restart_opencode(self._cfg)
        self._rebuild_pending = True

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
            ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
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
        if run_settings_panel_tabbed(self._cfg):
            save_config(self._cfg)
            self._build_menu()

    def _copy_api_url(self, _):
        from AppKit import NSPasteboard, NSStringPboardType
        kind = self._model_kind(self._active) if self._active else "mlx"
        port, _ = active_api(self._cfg, kind)
        url = f"http://localhost:{port}/v1"
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(url, NSStringPboardType)

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

    def _on_toggle_pin(self, sender: rumps.MenuItem):
        name = getattr(sender, "_model_name", None)
        if not name:
            return
        pinned = self._cfg.setdefault("pinned_models", [])
        if name in pinned:
            pinned.remove(name)
        else:
            pinned.insert(0, name)
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

    def _open_compare_history(self, _):
        run_compare_history_panel()

    def _open_test_prompt(self, _):
        if self._test_prompt_win is not None:
            self._test_prompt_win.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
        else:
            self._test_prompt_win, self._test_prompt_handler = _make_test_prompt_window(self)
            NSApp.activateIgnoringOtherApps_(True)
            self._test_prompt_win.makeKeyAndOrderFront_(None)

    def _open_server_logs(self, _):
        if hasattr(self, '_log_win') and self._log_win is not None:
            self._log_win.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
            return
        self._log_win = _make_log_window(self)
        NSApp.activateIgnoringOtherApps_(True)
        self._log_win.makeKeyAndOrderFront_(None)

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

    # ── Queue management ──────────────────────────────────────────────────────

    def addToQueue_(self, _s):
        idx = self._results_popup.indexOfSelectedItem()
        if idx < 0 or idx >= len(self._results):
            self._status_lbl.setStringValue_("Select a result first.")
            return
        m = self._results[idx]
        repo_id = m.get("modelId", m.get("id", ""))
        dest_dir = self._dest_fld.stringValue().strip()
        filter_tag = self._filter_popup.titleOfSelectedItem()
        # Avoid duplicates
        if any(e["repo_id"] == repo_id for e in self._dl_queue):
            self._status_lbl.setStringValue_(f"Already queued: {repo_id}")
            return
        self._dl_queue.append({
            "repo_id": repo_id, "dest_dir": dest_dir,
            "filter_tag": filter_tag, "status": "Queued"
        })
        self._queue_tbl.reloadData()
        self._status_lbl.setStringValue_(f"Queued: {repo_id}  ({len(self._dl_queue)} in queue)")
        # Start queue processing if idle
        if not self._downloading:
            self._processQueue()

    def cancelQueueItem_(self, _s):
        row = self._queue_tbl.selectedRow()
        if row < 0 or row >= len(self._dl_queue):
            return
        entry = self._dl_queue[row]
        if entry.get("status") == "Downloading":
            # Signal the active download to abort
            self._dl_cancelled = True
        else:
            self._dl_queue.pop(row)
            self._queue_tbl.reloadData()

    def _processQueue(self):
        """Start the next queued download if not already downloading."""
        if self._downloading:
            return
        # Find next Queued entry
        for entry in self._dl_queue:
            if entry["status"] == "Queued":
                entry["status"] = "Downloading"
                self._queue_tbl.reloadData()
                self._startDownloadEntry_(entry)
                return

    def _startDownloadEntry_(self, entry):
        self._dl_active_entry = entry
        repo_id = entry["repo_id"]
        dest_dir = Path(entry["dest_dir"]).expanduser()
        if not dest_dir.exists():
            entry["status"] = "Error: dir missing"
            self._queue_tbl.reloadData()
            self._processQueue()
            return
        model_dir = dest_dir / repo_id.split("/")[-1]
        byte_counter = type('_C', (), {'bytes_done': 0})()
        self._downloading = True
        self._dl_cancelled = False
        self._dl_error = None
        self._dl_success_path = None
        self._dl_model_dir = model_dir
        self._dl_byte_counter = byte_counter
        self._dl_bytes_total = 0
        self._dl_size_fetched = False
        self._dl_prev_done = 0
        self._dl_prev_time = 0.0
        self._dl_active_entry = entry
        self._dl_btn.setEnabled_(False)
        self._progress.setIndeterminate_(False)
        self._progress.setDoubleValue_(0.0)
        self._status_lbl.setStringValue_(f"Fetching size for {repo_id}…")
        save_pending_download(repo_id, str(dest_dir), entry["filter_tag"])

        def _do():
            try:
                import fnmatch
                import requests
                from huggingface_hub import model_info as hf_model_info, hf_hub_url
                from huggingface_hub.utils import build_hf_headers
                hf_token = self._app_ref._cfg.get("hf_token", "").strip() if self._app_ref else ""
                ignore = ["*.bin", "*.pt", "original/*"]
                try:
                    info = hf_model_info(repo_id, files_metadata=True, token=hf_token or None)
                    siblings = info.siblings or []
                except Exception:
                    siblings = []
                files = [s for s in siblings
                         if not any(fnmatch.fnmatch(s.rfilename, pat) for pat in ignore)]
                total_bytes = sum((s.size or 0) for s in files)
                self._dl_bytes_total = total_bytes
                self._dl_size_fetched = True
                headers = build_hf_headers(token=hf_token or None)
                model_dir.mkdir(parents=True, exist_ok=True)
                for sib in files:
                    if getattr(self, '_dl_cancelled', False):
                        raise Exception("Cancelled")
                    fname = sib.rfilename
                    url = hf_hub_url(repo_id, fname)
                    dest = model_dir / fname
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    expected = sib.size or 0
                    resume = dest.stat().st_size if dest.exists() else 0
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
                                if getattr(self, '_dl_cancelled', False):
                                    raise Exception("Cancelled")
                                if chunk:
                                    fh.write(chunk)
                                    byte_counter.bytes_done += len(chunk)
                self._dl_success_path = str(model_dir)
            except Exception as e:
                self._dl_error = str(e)

        threading.Thread(target=_do, daemon=True).start()
        self._dl_poll = rumps.Timer(self._pollQueueDownload, 1.0)
        self._dl_poll.start()

    def _pollQueueDownload_(self, timer):
        """Poll for active queue item completion, then chain to next."""
        total = self._dl_bytes_total
        size_fetched = getattr(self, '_dl_size_fetched', False)
        if not size_fetched and self._dl_error is None and self._dl_success_path is None:
            return
        now = time.time()
        counter = getattr(self, '_dl_byte_counter', None)
        done = counter.bytes_done if counter else 0
        prev_done = getattr(self, '_dl_prev_done', 0)
        prev_time = getattr(self, '_dl_prev_time', now)
        dt = now - prev_time
        speed_str = ""
        if dt > 0 and done > prev_done:
            speed_str = f"  {(done - prev_done) / dt / 1_048_576:.1f} MB/s"
        self._dl_prev_done = done
        self._dl_prev_time = now
        if total > 0:
            pct = min(done / total * 100, 99.0)
            self._progress.setDoubleValue_(pct)
            self._status_lbl.setStringValue_(
                f"{done/1e9:.2f} / {total/1e9:.2f} GB  ({pct:.0f}%){speed_str}")
        elif size_fetched:
            self._status_lbl.setStringValue_(f"{done/1e9:.2f} GB…{speed_str}")
        if self._dl_success_path is None and self._dl_error is None:
            return
        timer.stop()
        self._dl_poll = None
        self._dl_btn.setEnabled_(True)
        self._downloading = False
        clear_pending_download()
        entry = getattr(self, '_dl_active_entry', None)
        if self._dl_error:
            self._progress.setDoubleValue_(0.0)
            err = self._dl_error
            self._dl_error = None
            if entry:
                entry["status"] = "Cancelled" if "Cancelled" in err else "Error"
            self._queue_tbl.reloadData()
            self._status_lbl.setStringValue_(f"{'Cancelled' if 'Cancelled' in err else f'Error: {err}'}")
        else:
            success_path = self._dl_success_path
            self._dl_success_path = None
            self._progress.setDoubleValue_(100.0)
            if entry:
                entry["status"] = "✓ Done"
            self._queue_tbl.reloadData()
            self._status_lbl.setStringValue_(f"✓ Done — {success_path}")
            if self._app_ref:
                app = self._app_ref
                model_name = Path(success_path).name
                if model_name not in app._cfg["hidden_models"]:
                    app._cfg["hidden_models"].append(model_name)
                    save_config(app._cfg)
                app._model_meta_cache.clear()
                app._rebuild_pending = True
                app._prime_meta_cache()
        # Remove done/cancelled/error items from queue and process next
        self._dl_queue[:] = [e for e in self._dl_queue if e.get("status") == "Queued"]
        self._queue_tbl.reloadData()
        self._processQueue()

    # ── Download (single — kept for backward compat, now delegates to queue) ──

    def download_(self, _s):
        """Download immediately — adds to queue and starts if idle."""
        self.addToQueue_(_s)

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
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    win.setTitle_("Download from HuggingFace")
    win.setMinSize_((W, 400))
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

    # ── Queue list ────────────────────────────────────────────────────────────
    y -= 16
    q_lbl = _lbl("Queue:", ((_PAD, y), (50, 14)), right=False)
    q_lbl.setFont_(NSFont.systemFontOfSize_(11))
    cv.addSubview_(q_lbl)

    from AppKit import NSTableView, NSScrollView as _NSSV2, NSTableColumn
    q_scroll = _NSSV2.alloc().initWithFrame_(((_PAD, y - 80), (W - _PAD*2, 76)))
    q_scroll.setHasVerticalScroller_(True); q_scroll.setAutohidesScrollers_(True)
    q_tbl = NSTableView.alloc().initWithFrame_(((0, 0), (W - _PAD*2, 76)))
    for ident, title, width in [
        ("repo", "Model", W - _PAD*2 - 80 - 80 - 4),
        ("status", "Status", 80),
    ]:
        c = NSTableColumn.alloc().initWithIdentifier_(ident)
        c.setWidth_(width)
        c.headerCell().setStringValue_(title)
        q_tbl.addTableColumn_(c)
    q_tbl.setUsesAlternatingRowBackgroundColors_(True)
    q_tbl.setRowHeight_(18)
    q_scroll.setDocumentView_(q_tbl)
    cv.addSubview_(q_scroll)
    handler._queue_tbl = q_tbl
    handler._dl_queue = []   # list of {repo_id, dest_dir, filter_tag, status}

    class _QueueDS(NSObject):
        def numberOfRowsInTableView_(self, tv):
            return len(handler._dl_queue)
        def tableView_objectValueForTableColumn_row_(self, tv, col, row):
            e = handler._dl_queue[row]
            ident = col.identifier()
            if ident == "repo":
                return e.get("repo_id", "")
            return e.get("status", "Queued")
    q_ds = _QueueDS.alloc().init()
    q_tbl.setDataSource_(q_ds)
    q_tbl.reloadData()
    handler._queue_ds = q_ds

    # ── Buttons ───────────────────────────────────────────────────────────────
    cv.addSubview_(_btn("Close", handler, "closeWin:",
                        ((_PAD, _BTN_BOT), (80, _BTN_H))))
    dl_btn = _btn("⬇  Download", handler, "download:",
                  ((W - _PAD - 120, _BTN_BOT), (120, _BTN_H)))
    cv.addSubview_(dl_btn)
    handler._dl_btn = dl_btn

    queue_btn = _btn("+ Queue", handler, "addToQueue:",
                     ((W - _PAD - 120 - _GAP - 90, _BTN_BOT), (90, _BTN_H)))
    cv.addSubview_(queue_btn)
    handler._queue_btn = queue_btn

    cancel_btn = _btn("✕ Cancel", handler, "cancelQueueItem:",
                      ((W - _PAD - 120 - _GAP - 90 - _GAP - 90, _BTN_BOT), (90, _BTN_H)))
    cv.addSubview_(cancel_btn)
    handler._cancel_btn = cancel_btn

    return win, handler


def _make_log_window(app) -> NSWindow:
    """Open a floating window tailing the oMLX launchd log."""
    W, H = 700, 420
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (W, H)), 15, NSBackingStoreBuffered, False)
    win.setTitle_("Server Logs — oMLX")
    win.setMinSize_((W, 300))
    win.center()
    _constrain_to_screen(win)
    win.setTitlebarAppearsTransparent_(True)
    win.setMovableByWindowBackground_(True)
    cv = _vibrancy_content_view(win)

    scroll = NSScrollView.alloc().initWithFrame_(
        ((_PAD, _BTN_BOT + _BTN_H + _PAD), (W - _PAD*2, H - _PAD*2 - _BTN_H - _BTN_BOT)))
    scroll.setHasVerticalScroller_(True)
    scroll.setAutohidesScrollers_(True)
    tv = NSTextView.alloc().initWithFrame_(
        ((0, 0), (W - _PAD*2, H)))
    tv.setFont_(NSFont.monospacedSystemFontOfSize_weight_(10.5, 0.0))
    tv.setEditable_(False)
    scroll.setDocumentView_(tv)
    cv.addSubview_(scroll)

    # Close button
    class _LogDelegate(NSObject):
        def windowWillClose_(self, notif):
            if self._timer:
                self._timer.stop()
            app._log_win = None

    delegate = _LogDelegate.alloc().init()
    delegate._timer = None
    win.setDelegate_(delegate)
    app._log_delegate = delegate

    cv.addSubview_(_btn("Close", None, None, ((_PAD, _BTN_BOT), (80, _BTN_H))))

    # Tail via log stream — oMLX service logs
    service = app._cfg.get("omlx_service", "com.jim.omlx")

    def _tail():
        try:
            proc = subprocess.Popen(
                ["log", "stream", "--predicate",
                 f'subsystem == "{service}" OR process CONTAINS "omlx" OR process CONTAINS "python"',
                 "--style", "compact"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            delegate._proc = proc
            for line in proc.stdout:
                rumps.Timer(lambda _, l=line: _append(l), 0).start()
        except Exception as e:
            rumps.Timer(lambda _, m=str(e): _append(f"Error: {m}\n"), 0).start()

    def _append(line: str):
        cur = tv.string() or ""
        lines = (cur + line).splitlines(keepends=True)
        if len(lines) > 500:
            lines = lines[-500:]
        tv.setString_("".join(lines))
        tv.scrollToEndOfDocument_(None)

    threading.Thread(target=_tail, daemon=True).start()

    NSApp.activateIgnoringOtherApps_(True)
    win.makeKeyAndOrderFront_(None)
    return win


if __name__ == "__main__":
    Switchman().run()
