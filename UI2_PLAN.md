# Switchman UI 2.0 — Design Plan

> Goal: evolve from "functional utility" to "polished productivity tool" — still lightweight,
> still native, no Electron, no web server. Three pillars: Rich Quick Test, Custom Popover,
> Redesigned Settings.

---

## Pillar 1 — Rich Quick Test (highest impact)

### 1.1 WKWebView Output Pane
Replace the plain NSTextView with a WKWebView that renders markdown in real time.

**Why:** Code blocks, headers, bold/italic, tables — current output is a wall of monospace text.
Models produce well-structured markdown that we're throwing away.

**How:**
- Inject a minimal HTML skeleton on window open (dark/light CSS, monospace code font,
  `@media prefers-color-scheme`)
- Streaming: accumulate raw text in a JS string; call `window.appendChunk(delta)` via
  `evaluateJavaScript_completionHandler_` on every drain tick
- Parse markdown client-side with a single bundled JS library (e.g. `marked.js`, ~50 KB,
  no CDN — bundle inline as a Python string constant)
- Code blocks get a "Copy" button injected by JS
- Auto-scroll: `window.scrollTo(0, document.body.scrollHeight)` on each chunk

**Tradeoff:** WKWebView adds ~5 MB RAM. Worth it.

### 1.2 Conversation Mode (Multi-turn)
Add a toggle: Single Prompt ↔ Conversation. In conversation mode, messages accumulate
and the full history is sent on each turn.

**How:**
- `handler._messages: list[dict]` accumulates `{"role": ..., "content": ...}` pairs
- "New conversation" button clears history and resets the WKWebView
- System prompt field (collapsible, saves per-model in config)
- History persists in `~/.config/switchman/conversations/` as JSON, browsable

### 1.3 Inline Parameter Controls
Collapsible row below the input: Temperature slider (0–2), Max Tokens stepper, a
Preset dropdown (Precise / Balanced / Creative / Custom). Updates live without opening
Settings.

### 1.4 Token Counter as You Type
Real-time approximate token count in the input field (chars ÷ 3.5 heuristic, shown as
`~42 tokens` in small secondary text below the input). Warns if approaching context limit.

### 1.5 Response Stats Bar Upgrade
Current: `TTFT 420ms | 32.1 tok/s (512 tokens)`
Proposed: `TTFT 420ms · 32.1 tok/s · 512 tokens · 847 words · ctx 12%`

---

## Pillar 2 — Custom Menu Bar Popover

Replace the plain `NSMenu` with a custom `NSPopover` (or floating `NSPanel`) that appears
on menu bar click. The NSMenu still exists for keyboard/accessibility fallback.

### 2.1 Popover Layout (320 × 480 pt)

```
┌─────────────────────────────────────┐
│  ⚡ Switchman          ● 32 tok/s   │  ← title bar + live tok/s
├─────────────────────────────────────┤
│  Qwen3-Coder-Next-MLX-6bit          │  ← active model (large)
│  MLX · 63 GB · ctx 0%  🟢 Healthy   │  ← meta row
├─────────────────────────────────────┤
│  Mini spark chart (last 5 tok/s)    │  ← NSView bar chart
├─────────────────────────────────────┤
│  🟢 Memory   ████████░░  67%        │  ← memory pressure bar
│  🌡 Thermal  Nominal                │  ← thermal state
├─────────────────────────────────────┤
│  [Quick Test]  [Switch Model]       │  ← primary action buttons
│  [Benchmark]   [Settings]           │
├─────────────────────────────────────┤
│  Recent / Pinned model list         │  ← click to switch instantly
└─────────────────────────────────────┘
```

### 2.2 Animated Model Switch in Popover
Clicking a model in the popover triggers an in-place loading animation — the model
name fades to secondary color, a progress bar sweeps across, then snaps to the
new model name on completion. No separate flash in the title bar needed.

### 2.3 Mini Spark Chart
Custom NSView drawing last 5 tok/s readings as a bar chart. ~40 lines of CoreGraphics.
Stored in `_tps_history: deque(maxlen=5)`.

---

## Pillar 3 — Redesigned Settings

### 3.1 Tabbed Settings Panel
Replace the single long-scroll panel with an `NSTabView`:

- **Models** — model list with search filter, per-model cards (alias, note, context,
  params, set-default, pin, hide, delete)
- **Inference** — paths to oMLX / llama-server, ports, API keys, warm-up toggle
- **Sync** — Aider, Zed, Continue, opencode, on-switch script, HF token
- **Behavior** — notifications, auto-reload, watchdog, default model, startup behavior
- **Appearance** — font size, Quick Test defaults, system prompt template

### 3.2 Model Cards
Each model gets a card row in the Models tab:
- Left: colored badge (MLX=blue, GGUF=orange), model name, alias if set
- Center: size on disk, last load time, avg tok/s from benchmark history
- Right: action icons (▶ load, ★ default, 📌 pin, 👁 hide, 🗑 delete)
- Expandable for notes / per-model params

### 3.3 Live Search in Model List
NSSearchField at the top of the Models tab filters the card list in real time.
Same search logic as the floating search panel.

---

## Pillar 4 — Polish & Animation Layer

### 4.1 Smooth Loading Progress Bar
During model load, show a real progress bar (indeterminate → determinate once we have
a time estimate). Sits below the active model name in the popover and in Quick Test.
Uses `NSProgressIndicator` style 0 (bar) with `setIndeterminate_(False)` once estimate
is available. Animated by a timer: `progress = elapsed / estimate`.

### 4.2 Status Message Animations
Use `NSAnimationContext` to cross-fade between status strings instead of hard-setting text.
~5 lines: `NSAnimationContext.beginGrouping()`, set duration, `animator().setAlphaValue_`.

### 4.3 Consistent Icon Language
Define a small set of SF Symbol → action mappings used everywhere (menu, popover, cards):
- ▶ play.fill = Load
- ★ star.fill = Default
- 📌 pin = Pin
- 👁 eye.slash = Hide
- ⚙ gearshape = Settings
- ⏱ timer = Benchmark
- 💬 bubble.left = Quick Test
- ⬇ arrow.down.circle = Download

---

## Implementation Order

| Phase | Work | Effort |
|-------|------|--------|
| 1 | WKWebView output + marked.js streaming | Medium |
| 2 | Inline param controls + token counter | Small |
| 3 | Conversation mode + system prompt | Medium |
| 4 | Custom popover shell + layout | Medium |
| 5 | Mini spark chart + memory/thermal bars | Small |
| 6 | Animated model switch in popover | Small |
| 7 | Tabbed settings panel | Large |
| 8 | Model cards with search | Medium |
| 9 | Progress bar + cross-fade animations | Small |

Start with Phase 1 (WKWebView) — highest visible impact, self-contained, doesn't
touch the rest of the app.

---

## Open Questions

- Does WKWebView streaming feel smooth enough? (Test with 50 tok/s — ~1 chunk per 20ms
  drain tick. Should be fine.)
- Popover vs floating NSPanel? Popover is more native but has size constraints.
  NSPanel gives full control. Probably NSPanel styled like a popover with arrow.
- Keep NSMenu as fallback for keyboard nav? Yes — accessibility requirement.
