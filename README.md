# 4-Step Trading Protocol — System

Semi-automated trading assistant for the 4-Step Protocol (EURUSD + GBPUSD on TradeNation via TradingView).

## Architecture

- **Layer 1 — Context Engine** (`01_context_engine.pine`) ← we are here
- Layer 2 — Setup Grader (A+/A/B checklist)
- Layer 3 — Checklist dashboard
- Layer 4 — Execution: manual click on TradeNation chart panel in TradingView
- Layer 5 — Backtest strategy harness

## Layer 1 — Install

1. Open TradingView, load **EURUSD** or **GBPUSD** on the **5-minute** chart.
2. Open **Pine Editor** (bottom panel).
3. Paste the full contents of `01_context_engine.pine`.
4. Click **Save** → name it `4SP - Context Engine`.
5. Click **Add to chart**.

## Inputs cheat-sheet

| Group                | Default          | Notes                                                                 |
|----------------------|------------------|-----------------------------------------------------------------------|
| Timezone             | America/New_York | Switch to Europe/London if you prefer UK clock                        |
| Asia session         | 18:00–02:00 NY   | 23:00–07:00 London                                                    |
| London               | 02:00–05:00 NY   | 07:00–10:00 London                                                    |
| NY AM                | 07:00–12:00 NY   | Covers 08:30 & 10:00 news windows                                     |
| NY Lunch             | 12:00–13:00 NY   |                                                                       |
| NY PM / LO Close     | 13:00–16:00 NY   |                                                                       |
| FVG timeframe        | 60 (H1)          | Blank = use current chart TF; set "240" for H4 FVGs                   |
| Displacement ATR mult| 1.5              | Candle body > 1.5 × ATR(20) to qualify                                |
| Swing pivot length   | 5                | Shorter = more swings, noisier                                        |
| EQH/L tolerance      | 0.10 × ATR       | Stricter = fewer but cleaner clusters                                 |

## What the dashboard shows

| Row          | Meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| Session      | Asia / London / NY-AM / Lunch / NY-PM — drives WHEN rules               |
| Cycle        | Expansion / Consolidation / Neutral (ATR-fast vs ATR-slow)              |
| H1 OF        | Bull / Bear / Neutral — close vs last H1 swing high/low                 |
| ADR%         | % of 5-day ADR consumed today. >100% = overextended                     |
| LQ Above/Below | Nearest tracked liquidity level                                       |
| Last Swing   | Strong / weak label of most recent pivot                                |

## On-chart markers

- `SFP` (orange): sweep of a tracked level, closed back inside
- ▲/▼ (lime/red): displacement candle
- `CIOF↑/↓` (teal/maroon): close through prior swing + displacement
- `SH/SL` = strong high/low (took opposing liquidity first); `wh/wl` = weak
- Green box = bullish FVG; red box = bearish FVG
- `OB+/-` box = order block

## Alerts

In TradingView → right-click chart → **Add alert** → Condition: `4SP - Context Engine` → "Any alert() function call" → save.

You'll receive one alert per event type (sweep / displacement / CIOF / session open). Route to your phone / Discord / Telegram via TradingView notification settings.

## Known limitations (honest)

- **Strong/weak swing** is simplified — production rules use multi-leg context you'd refine by hand.
- **H1 OF** is a proxy (close vs last H1 swing). Real OF = "respecting/disrespecting PDAs." Layer 2 will tighten this.
- **FVG** detection is pure 3-candle gap — doesn't yet track mitigation / inversion (IFVG).
- **Monday Range** resets at exchange midnight, which for NY tz = 00:00 NY. Verify this matches your definition.
- **TradeNation orders** are not placed by this script. Layer 1 is observation + alerts only.

## Next (Layer 2)

Setup grader that consumes Layer 1 events + state and outputs A+/A/B/Skip with:
- Full checklist status (12 items from the PDF)
- Suggested risk (1.5R / 1R / 0.5R / 0)
- Suggested invalidation level + RR to nearest LQ
- Webhook payload (JSON) for auto-logging to a journal sheet
