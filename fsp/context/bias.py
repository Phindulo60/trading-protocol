"""H1 Order Flow bias — proper BOS / CHoCH walking.

We walk pivots chronologically. At each pivot we update state based on whether
it takes out the MOST RECENT opposing swing (not the all-time extreme):
  - In BULL state, a pivot-low < most-recent prior pivot-low = CHoCH to BEAR.
  - In BEAR state, a pivot-high > most-recent prior pivot-high = CHoCH to BULL.
  - A continuation break of the same-direction prior swing = BOS (stays).
  - First few pivots set a tentative direction by comparing HH/HL sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fsp.data.types import OFBias, Swing
from fsp.structure.swings import find_swings


@dataclass
class BiasState:
    bias: OFBias
    last_hh: float | None
    last_hl: float | None
    last_lh: float | None
    last_ll: float | None
    last_event: str


def compute_of_bias(df: pd.DataFrame, length: int = 3) -> BiasState:
    swings = find_swings(df, length=length)
    if len(swings) < 4:
        return BiasState(OFBias.NEUTRAL, None, None, None, None, "init")

    state = OFBias.NEUTRAL
    last_event = "init"

    # Track most-recent confirmed swing of each kind (not the all-time extreme)
    prev_ph: float | None = None  # most recent pivot high
    prev_pl: float | None = None  # most recent pivot low
    hh = hl = lh = ll = None

    for s in swings:
        if s.kind == "high":
            if prev_ph is not None:
                if s.price > prev_ph:
                    # Broke the most recent high upward
                    if state == OFBias.BEAR:
                        state = OFBias.BULL
                        last_event = "CHoCH-up"
                    else:
                        state = OFBias.BULL
                        last_event = "BOS-up"
                    hh = s.price
                else:
                    lh = s.price  # lower high, structural weakness
            else:
                hh = s.price  # seed
            prev_ph = s.price
        else:  # low
            if prev_pl is not None:
                if s.price < prev_pl:
                    if state == OFBias.BULL:
                        state = OFBias.BEAR
                        last_event = "CHoCH-dn"
                    else:
                        state = OFBias.BEAR
                        last_event = "BOS-dn"
                    ll = s.price
                else:
                    hl = s.price  # higher low, structural strength
            else:
                ll = s.price
            prev_pl = s.price

    return BiasState(state, hh, hl, lh, ll, last_event)
