"""
teamName.py — Algothon 2026 trading strategy.

STRATEGY: Multi-Window Cross-Sectional Contrarian (Mean Reversion)
==================================================================
Statistical analysis of prices.txt reveals strong mean-reversion in the
cross section: instruments with the largest cumulative losses over 17-, 91-
and 100-day horizons tend to outperform peers on the next day, and vice
versa.  This strategy exploits that negative autocorrelation by going long
recent under-performers and short recent out-performers.

Signal construction
    For each lookback window the log-price change (negative sign for
    contrarian direction) is computed, then the three signals are blended
    by MOM_WEIGHTS.  The combined score is cross-sectionally standardized
    and reduced to a ±1 direction via np.sign().

Position sizing
    Each instrument is allocated up to POS_FRACTION × (dollar position
    limit) in the signal direction.  The evaluator enforces hard dollar-
    limit caps, so positions are also clipped to those limits internally.

Risk controls
    1. Max exposure cap: positions are capped at POS_FRACTION × dollar limit.
    2. Drawdown de-risking: cumulative P&L is approximated from daily
       mark-to-market changes (currentPos · Δprice).  When the drawdown from
       the peak cumulative P&L exceeds DRAWDOWN_THRESH, all position sizes
       are scaled linearly down to DRAWDOWN_FLOOR × normal.
       Set DRAWDOWN_THRESH = 1.0 to disable entirely.
    3. Minimum history guard: no positions are opened until MIN_HISTORY days
       of price data are available for the longest signal window.

PARAMETER TUNING (edit the constants in the HYPERPARAMETERS section)
    MOM_WINDOWS   — lookback windows (days) for the contrarian signal
    MOM_WEIGHTS   — blend weights; must sum to 1.0
    POS_FRACTION  — aggression: fraction of dollar limit to target (0–1)
    DRAWDOWN_THRESH — cumulative-P&L drawdown from peak that triggers
                      de-risking (0–1); set to 1.0 to disable
    DRAWDOWN_FLOOR  — minimum position scale during severe drawdown (0–1)
"""

import numpy as np

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
# Contrarian signal lookback windows (trading days) and blend weights.
# Weights are renormalized automatically if some windows lack history.
MOM_WINDOWS = [17, 91, 100]
MOM_WEIGHTS = [0.35, 0.35, 0.30]

# Fraction of per-instrument dollar position limit to target (0 < value ≤ 1).
# Higher values increase P&L but also increase commission drag.
POS_FRACTION = 0.95

# Drawdown de-risking based on cumulative approximate P&L.
# Set DRAWDOWN_THRESH = 1.0 to disable drawdown de-risking entirely.
DRAWDOWN_THRESH = 0.50   # start de-risking when cumulative P&L drops 50 % from peak
DRAWDOWN_FLOOR  = 0.25   # minimum position scale during severe drawdown

# Minimum trading days required before any position is taken.
MIN_HISTORY = max(MOM_WINDOWS) + 2

# ── PER-INSTRUMENT DOLLAR LIMITS (mirroring eval.py) ─────────────────────────
_N_INST = 51
_DLR_LIMIT = np.full(_N_INST, 10_000.0)
_DLR_LIMIT[0] = 100_000.0   # instrument 0 has a 10× higher limit

# ── MODULE STATE ──────────────────────────────────────────────────────────────
nInst       = _N_INST
currentPos  = np.zeros(nInst)
_prev_prices = None   # prices on the previous call (for P&L estimation)
_cum_pnl     = 0.0    # cumulative approximate P&L (mark-to-market)
_peak_pnl    = 0.0    # peak of _cum_pnl for drawdown calculation


def getMyPosition(prcSoFar):
    """
    Return desired integer share positions given the price history so far.

    Args:
        prcSoFar: np.ndarray of shape (nInst, nt).
                  Each row is one instrument; each column is one day.

    Returns:
        np.ndarray of shape (nInst,) with integer share counts (may be
        negative for short positions).
    """
    global currentPos, _prev_prices, _cum_pnl, _peak_pnl

    nins, nt = prcSoFar.shape

    if nt < MIN_HISTORY:
        return np.zeros(nins, dtype=int)

    curPrices = prcSoFar[:, -1]
    logP      = np.log(prcSoFar)   # (nins, nt)

    # ── Cumulative P&L tracking for drawdown control ──────────────────────────
    # Approximate daily gain = previous positions × today's price change.
    # This does not include commissions but is sufficient for drawdown detection.
    if _prev_prices is not None:
        _cum_pnl += currentPos.dot(curPrices - _prev_prices)
    _prev_prices = curPrices.copy()
    _peak_pnl = max(_peak_pnl, _cum_pnl)

    # ── Composite contrarian signal ───────────────────────────────────────────
    # Negative sign: instruments that fell over [window] days get a positive
    # score (bet on reversion upward); recent winners get a negative score.
    composite = np.zeros(nins)
    total_w   = 0.0

    for window, weight in zip(MOM_WINDOWS, MOM_WEIGHTS):
        if nt < window + 2:
            continue   # not enough history — skip this component
        composite += weight * -(logP[:, -1] - logP[:, -(window + 1)])
        total_w   += weight

    if total_w < 1e-9:
        # No signal components available; hold flat
        currentPos = np.zeros(nins, dtype=int)
        return currentPos

    composite /= total_w   # re-weight so the blend always sums to 1

    # ── Cross-sectional z-score standardization ───────────────────────────────
    cs_std = composite.std()
    if cs_std > 1e-10:
        composite = (composite - composite.mean()) / cs_std

    # Direction: +1 = long, −1 = short (binary signal)
    sig_dir = np.sign(composite)

    # ── Drawdown-aware de-risking ─────────────────────────────────────────────
    dd_scale = 1.0
    if DRAWDOWN_THRESH < 1.0 and _peak_pnl > 1_000:
        dd = (_peak_pnl - _cum_pnl) / _peak_pnl
        if dd > DRAWDOWN_THRESH:
            excess   = dd - DRAWDOWN_THRESH
            ramp     = (1.0 - DRAWDOWN_FLOOR) / (1.0 - DRAWDOWN_THRESH)
            dd_scale = max(DRAWDOWN_FLOOR, 1.0 - ramp * excess)

    # ── Position sizing ───────────────────────────────────────────────────────
    # Dollar target for each instrument: POS_FRACTION × limit × direction
    dlr_pos = POS_FRACTION * dd_scale * _DLR_LIMIT * sig_dir

    # Convert to integer shares and enforce hard position limits
    pos_limits = (_DLR_LIMIT / curPrices).astype(int)
    newPos     = np.clip((dlr_pos / curPrices).astype(int), -pos_limits, pos_limits)

    currentPos = newPos
    return currentPos
