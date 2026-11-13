#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.

Method (per market, per case):
  * Center: the most recent observed price (persistence, no drift).
  * Volatility: standard deviation of the market's 15-minute price
    changes over the last 120 minutes, expressed as a multiplier on the
    baseline per-minute volatility of 0.0025, clipped to [0.1, 3.0].
  * Spread at horizon h: the 15-minute volatility grown as
    (h / 15) ** 0.30, so the spread widens slower than a random walk.
  * Quantiles: center + z_tau * spread(h), with the standard normal
    z-scores for 5/25/50/75/95%.
  * Empirical blend: the spread of the market's prices over the last
    24 hours (its 5-95% range, as a standard deviation) is scaled down
    to each horizon the same way the model spread grows, and blended
    0.8 * model spread + 0.2 * empirical spread. The interval stays
    symmetric around the center, so the median is exactly the center.
  * Clipped to [0, 1] and forced non-decreasing across the five levels.

Reads every *.in file in $MAKEFUN_INPUT_DIR (default /input) and writes
one *.out file per case to $MAKEFUN_OUTPUT_DIR (default /output).
Standard library only, deterministic, no network.
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

# --- constants -----------------------------------------------------------
BASE_SIGMA = 0.0025          # baseline per-minute volatility (house)
VOL_WINDOW_MIN = 120         # minutes of history used for volatility
VOL_STEP_MIN = 15            # spacing of the changes inside that window
BETA = 0.30                  # spread growth exponent in horizon
MULT_CLIP = (0.1, 3.0)       # clamp on the volatility multiplier
EMP_WINDOW_MIN = 1440        # minutes of history for empirical quantiles
EMP_BLEND = 0.2              # weight of the empirical quantiles
HORIZON = 240                # minutes forecast ahead
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
SQRT_15 = math.sqrt(15.0)


def finite_prices(prices):
    """Keep only finite numbers (drop NaN/inf from a corrupt live tick)."""
    out = []
    for p in prices:
        if isinstance(p, (int, float)) and math.isfinite(p):
            out.append(float(p))
    return out


def sample_sd(values):
    """Population standard deviation; 0.0 when fewer than two values."""
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def interp_quantile(sorted_vals, level):
    """Linear-interpolation quantile of a sorted list at `level` in [0, 1]."""
    n = len(sorted_vals)
    if n == 0:
        return 0.5
    if n == 1:
        return sorted_vals[0]
    pos = level * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def volatility_multiplier(clean):
    """Per-market volatility multiplier from the last 120 minutes.

    Takes the price changes over 15-minute intervals within the last
    120 minutes (every minute's 15-minute change, overlapping) and
    divides their standard deviation by the baseline 15-minute
    standard deviation (0.0025 * sqrt(15)). Clipped to [0.1, 3.0];
    1.0 when there is not enough history to measure a change.
    """
    changes = []
    # every minute's 15-minute change over the last 120 minutes
    start = max(VOL_STEP_MIN, len(clean) - VOL_WINDOW_MIN)
    for i in range(start, len(clean)):
        changes.append(clean[i] - clean[i - VOL_STEP_MIN])
    sd15 = sample_sd(changes)
    if sd15 <= 0.0:
        return 1.0
    mult = sd15 / (BASE_SIGMA * SQRT_15)
    if mult < MULT_CLIP[0]:
        return MULT_CLIP[0]
    if mult > MULT_CLIP[1]:
        return MULT_CLIP[1]
    return mult


def empirical_quantiles(clean):
    """Quantiles of the last 24 hours of prices at the five levels."""
    window = clean[-EMP_WINDOW_MIN:]
    ordered = sorted(window)
    return [interp_quantile(ordered, lv) for lv in LEVELS]


def forecast_market(prices):
    """Return the 240 x 5 forecast rows for one market."""
    clean = finite_prices(prices)
    if not clean:
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(HORIZON)]
    center = clean[-1]
    mult = volatility_multiplier(clean)
    emp = empirical_quantiles(clean)
    # empirical spread implied by the 24h 5-95% range, blended with the
    # model spread; the interval stays symmetric around the center, so
    # the median is exactly the center. The empirical spread is scaled
    # down to the horizon the same way the model spread grows.
    emp_spread = (emp[4] - emp[0]) / (2.0 * 1.644854)
    sigma15 = BASE_SIGMA * mult * SQRT_15
    rows = []
    for h in range(1, HORIZON + 1):
        spread = sigma15 * (h / float(VOL_STEP_MIN)) ** BETA
        emp_h = emp_spread * (h / float(EMP_WINDOW_MIN)) ** BETA
        blended_spread = (1.0 - EMP_BLEND) * spread + EMP_BLEND * emp_h
        row = []
        for qi in range(len(LEVELS)):
            row.append(center + Z_SCORES[qi] * blended_spread)
        # clip to [0, 1] and force non-decreasing across the levels
        row = [min(1.0, max(0.0, v)) for v in row]
        for qi in range(1, len(row)):
            if row[qi] < row[qi - 1]:
                row[qi] = row[qi - 1]
        rows.append([round(v, 6) for v in row])
    return rows


def forecast_case(case):
    """Forecast one case dict; return the output object."""
    forecasts = []
    for market in case["markets"]:
        rows = forecast_market(market["prices"])
        forecasts.append({"id": market["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    in_files = sorted(in_dir.glob("*.in"))
    if not in_files:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    for in_file in in_files:
        with open(in_file, "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_file = out_dir / (in_file.stem + ".out")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"), sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
