# Trading Signal Research Findings

All studies run on M15 data across 8 pairs (EURUSD, GBPUSD, AUDUSD, USDCAD,
EURJPY, GBPJPY, USDCHF, NZDUSD) covering Apr 2024 – May 2025 (~13 months).

---

## 1. Key Level Behavior — Base Rates

**Dataset:** 50,918 touch events across 6 level types × 8 pairs  
**Outcome window:** 16 M15 bars (~4h) after each touch

### What counts as a "touch"
- HIGH level: bar's high within 5 pips of level (from below)
- LOW level: bar's low within 5 pips of level (from above)

### Results by level type

| Level | N | Reversal% | Stop Hunt% | Break% | Avg Rev Pips |
|---|---|---|---|---|---|
| **PSH/PSL** (prev session) | ~21,000 | **27%** | 4–5% | 57% | 36p |
| **PDH/PDL** (prev day) | ~12,000 | **18–20%** | 3% | 69% | 37p |
| **PWH/PWL** (prev week) | ~17,000 | **4–5%** | 1% | **93%** | 40p |

**Key insight:** Weekly levels are break levels, not reversal levels.  
By the time price reaches PWH/PWL, momentum is strong enough to push through 93% of the time.
PSH/PSL are the most reactive reversal levels.

### Results by session (all levels)

| Session | Reversal% | Break% |
|---|---|---|
| **London** | **21%** | 72% |
| Asia | 17% | 69% |
| NY | 15% | 75% |

London produces the cleanest reversals. NY is the most break-heavy.

### Best level × session combinations (no OB filter)

| Combo | N | Reversal% |
|---|---|---|
| PSL + London | 3,072 | **30%** |
| PSH + London | 3,531 | **29%** |
| PSL + Asia | 3,421 | 28% |
| PDH + London | 1,604 | 26% |
| PSH + NY | 3,943 | 26% |

---

## 2. OB + H4 Confluence Filters

Adding unmitigated Order Block (M15 or H1) at the level + H4 EMA20 trend alignment
dramatically improves reversal rates.

### Filter impact (PDH/PDL/PSH/PSL combined)

| Filter | N | Rev% | Avg Rev Pips | Empty Weeks |
|---|---|---|---|---|
| Baseline | 33,715 | 24% | 36p | — |
| H4 aligned only | 7,437 | 44% | 40p | — |
| H4 + M15 OB | 109 | **81%** | 51p | 53% |
| **H4 + H1 OB** | **194** | **87%** | **58p** | **39%** |
| H4 + M15 or H1 OB | 291 | 84% | 55p | 29% |
| H4 + M15 AND H1 OB | 12 | **100%** | **76p** | 90% |

**Use H1 OB as the primary filter.** H1 OBs are wider (larger candles), so they
naturally overlap the level with more authority, and they generate 78% more
qualifying events than M15 OBs while improving the reversal rate.

### By level type (H4 + H1 OB)

| Level | Baseline | With H4+H1OB | Avg Rev Pips |
|---|---|---|---|
| PDL | 18% | **91%** | 46p |
| PSL | 27% | **82%** | 49p |
| PSH | 27% | **78%** | 55p |
| PDH | 20% | **75%** | 45p |

### By session (H4 + H1 OB)

| Session | Rev% | Notes |
|---|---|---|
| **New York** | **90%** | Best — NY respects OB+level the most |
| **London** | **84%** | Strong |
| Asia | 68% | Weaker but still trades |

### By pair (H4 + H1 OB)

| Pair | N | Rev% | Avg Rev Pips |
|---|---|---|---|
| GBPUSD | 19 | **100%** | 65p |
| NZDUSD | 25 | **96%** | 42p |
| EURUSD | 64 | **91%** | 45p |
| USDCHF | 16 | 88% | 63p |
| USDCAD | 28 | 86% | 51p |
| GBPJPY | 14 | 79% | **139p** |
| EURJPY | 8 | 75% | **115p** |
| AUDUSD | 20 | 65% | 46p |

JPY pairs move the most in pips when they do reverse.

### Weekly frequency

| Tier | Setups/week avg | Rev% | Empty weeks |
|---|---|---|---|
| CONF (M15+H1+H4) | 0.1 | 100% | 90% |
| H1 + H4 | 1.8 | 87% | 39% |
| M15 + H4 | 1.0 | 81% | 53% |
| Either + H4 | 2.6 | 84% | 29% |

### Why PWH/PWL have zero qualifying OB events
When price travels from PSL/PDL all the way to PWL, it sweeps through every OB
on the way there — those OBs get mitigated before price reaches the weekly level.
Zero qualifying events were found across 110 weeks and 8 pairs.

---

## 3. RSI Divergence at Key Levels

**Counterintuitive finding:** RSI should NOT be used as confirmation at key levels.
It works as a **divergence signal** — the opposite of what most people expect.

### At a LOW level touch

| RSI at touch bar | Break% | Reversal% | Interpretation |
|---|---|---|---|
| RSI 0–30 (very oversold) | **82%** | 13% | Momentum aligned = blows through |
| RSI 30–40 | 66% | 21% | |
| RSI 40–50 | 54% | 27% | |
| RSI 60–70 | 47% | 46% | Rising divergence |
| **RSI 70+** | 17% | **72%** | Strong divergence = bounce |

### At a HIGH level touch

| RSI at touch bar | Break% | Reversal% |
|---|---|---|
| RSI 70+ (very overbought) | **84%** | 12% | Momentum = continuation |
| RSI 60–70 | 67% | 21% | |
| RSI 50–60 | 54% | 28% | |
| RSI 30–40 | 51% | 40% | Rising divergence |
| RSI 0–30 | 35% | **60%** | Strong divergence = reversal |

### Practical application
- If price tests PDL and RSI is at 72 → high probability bounce (divergence)
- If price tests PDH and RSI is at 32 → high probability reversal (divergence)
- If price tests PDL and RSI is at 22 → expect a break through (momentum aligned)
- The LEVEL_OB strategy does NOT use RSI as a filter — the OB+H4 combination
  already implicitly captures divergence because OBs form when momentum pauses

---

## 4. Break + Pullback Study

**23,043 break events | 48% had a pullback within 50 bars**

The break+pullback trade (break through level → price returns to test the flip) 
was studied extensively. Results were weak.

### Results

| Filter | Continuation% | Fail% |
|---|---|---|
| No filter | 37% | 49% |
| H4 aligned | **42%** | 42% |
| OB filter | n/a | OBs are mitigated during pullback |

### Why OBs don't work for break+pullback
When price breaks a level and then pulls back to test it, the pullback candle
itself wicks through any OB sitting at the level — mitigating it in the process.
By the time of the pullback, no qualifying OBs remain near the level.

To use OBs for break+pullback, you would need to scan for OBs in the **new 
territory** (above old resistance for bullish breaks) — a structurally different 
setup requiring a separate study.

### No-pullback rate (price just kept going)

| Level | No pullback% | Implication |
|---|---|---|
| PWH/PWL | **78–80%** | Momentum entries — 4/5 weekly breaks never retrace |
| PDH/PDL | 35–42% | Half retest |
| PSH/PSL | 23–26% | Most come back to test |

**Verdict:** Do not build break+pullback as a strategy. 42% with H4 is below
breakeven after spread/slippage. Use LEVEL_OB (reversal) instead.

---

## 5. PWH/PWL Momentum Breakout Study

**2,659 break events | H4 + strong close filter**

Directly entering on the M15 close that breaks PWH/PWL.

### Filter results (TP=2R)

| Filter | N | WR | Exp/trade | PF |
|---|---|---|---|---|
| No filter | 2,659 | 20% | +0.02R | 1.0 |
| H4 aligned | 1,968 | 22% | +0.10R | 1.2 |
| H4 + strong close | 929 | 26% | +0.09R | 1.2 |
| H4 + strong close + RSI 45-65 | 390 | 25% | **+0.16R** | 1.3 |

### Session breakdown (H4 + strong close, TP=2R)

| Session | WR | Exp | Notes |
|---|---|---|---|
| London | 29% | +0.18R | Best |
| Asia | 29% | +0.17R | Competitive |
| **NY** | **20%** | **-0.07R** | **Loses money — avoid** |

### Day of week (H4 + strong close, TP=2R)

| Day | WR | Exp |
|---|---|---|
| Thursday | 32% | **+0.29R** | Best |
| Monday | 27% | +0.16R | Good |
| Tuesday | 28% | +0.06R | Marginal |
| Wednesday | 26% | +0.03R | Marginal |
| Friday | 18% | -0.08R | **Avoid** |

### Per-pair (H4 + strong close, TP=2R)

| Pair | N | WR | Exp | Verdict |
|---|---|---|---|---|
| **GBPJPY** | 45 | **44%** | **+0.41R** | Only viable pair |
| **GBPUSD** | 123 | **33%** | **+0.33R** | Viable |
| EURUSD | 265 | 22% | +0.09R | Marginal |
| All others | — | 22–27% | -0.08 to +0.07R | Not worth it |

### RSI at break bar (H4 aligned, bullish breaks)

| RSI | WR | Exp |
|---|---|---|
| **RSI 0–40** | **26%** | **+0.42R** | Best — not yet overbought |
| RSI 40–55 | 19% | +0.14R | |
| RSI 80+ | 10% | -0.44R | **Avoid — exhaustion** |

### Verdict
- **Do not build as a standalone strategy.** +0.16R/trade is too thin after costs.
- **Context use only:** if TREND_RSI fires a SHORT at PSH but H4 is breaking
  above PWH with a strong candle on Thursday London → skip the TREND_RSI short.
- **GBPJPY + GBPUSD** are the only pairs where weekly breakouts have real edge.
  If ever adding a breakout strategy, start here with more data.

---

## 6. Strategy Comparison Summary

| Strategy | Setups/wk | WR | Exp/trade | Notes |
|---|---|---|---|---|
| **LEVEL_OB H1+H4** | 1.8 | 87% | ~+1.0R est | High quality, low frequency |
| **LEVEL_OB Either** | 2.6 | 84% | ~+0.85R est | Best balance |
| **LEVEL_OB CONF** | 0.1 | 100% | ~+1.5R est | Rare, size up |
| **TREND_RSI** | ~20 | 61% | +0.51R | Backtested, workhorse |
| Break+pullback | — | 42% | — | No edge — not built |
| PWH/PWL breakout | 17 | 22–26% | +0.04–0.16R | No edge — not built |

---

## 7. What NOT to Trade

Based on the research:

| Setup | Why |
|---|---|
| Level reversal at PWH/PWL | 93% break rate, OBs mitigated before price arrives |
| Break+pullback (any level) | 42% max with H4 — below breakeven after costs |
| PWH/PWL breakout (most pairs) | +0.16R max — not worth the risk |
| Level touch with RSI extreme in test direction | RSI confirms momentum = break, not reversal |
| NY session PWH/PWL breakout | Exp=-0.07R — NY reverses London direction |
| Friday breakouts | WR=18%, Exp=-0.08R — end-of-week positioning distorts |
| TREND_RSI SHORT at PSH when H4 is breaking PWH on Thu/Fri London | Weekly momentum overrides |

---

*Last updated: May 2026 | Data: Apr 2024–May 2025 | 8 pairs | M15*
