# Polymarket NVIDIA CEO forecaster

Entry for make.fun competition `polymarket-nvidia-ceo-forecast-01` ($250).
Forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event at every minute of the 4 hours after a checkpoint, as five
quantiles (5%, 25%, 50%, 75%, 95%).

## Method

The house baseline (persistence + spread widening like a random walk,
sigma 0.0025 per sqrt-minute) is well calibrated for the calm regime, so
this entry keeps its shape and fixes its two weaknesses:

1. **Per-market recent volatility.** The 72h history is dominated by the
   wild market-discovery regime (Nov 10-11) and overstates current
   volatility 2-5x. The spread is scaled per market by the last 120
   minutes' 15-minute moves, relative to what the baseline implies
   (clipped to 0.1x-3x; young markets take the pooled median).
2. **Saturating spread.** The spread grows like h^0.35, not sqrt(h):
   240-minute moves are far smaller than random-walk extrapolation
   (mean reversion). A 20% blend of the last 24h's empirical move
   quantiles adds jump-risk shape.

No drift (every drift tried hurt on practice) and no Student-t
fattening (the answers' tails are thinner than normal at these
horizons).

## Results (local practice scorer)

Practice loss **0.290475** vs house baseline **0.363589** (-20.1%);
wins all 6 practice cases and both 3-case halves. Also beats the
baseline on 12 extra backtest windows from the wilder Nov 10-12 regime
(2.190 vs 2.249). The scorer was validated by reproducing the baseline's
published practice score exactly (0.363589, median error 0.767503,
coverage 0.949642).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic
(two runs byte-identical). ~0.15 s per case, limit is 8 s.
