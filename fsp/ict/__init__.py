"""ICT (Inner Circle Trader) decision engine.

Combines the structure primitives (swings, FVGs, order blocks, displacement)
into a coherent market-structure state machine and, ultimately, a confluence
engine that produces concise high-probability trade decisions.

Modules:
  structure       — BOS / CHoCH / MSS market-structure state machine (this build)
  liquidity       — liquidity pools + sweep detection            (next)
  premium_discount — dealing range + OTE zones                   (next)
  bias            — HTF directional bias                          (next)
  engine          — confluence scorer → trade decisions           (next)
"""
