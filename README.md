# Algothon 2026 Starter Code

Starter code for the Susquehanna x UNSW FinTech Society Algothon 2026 - the seventh year of Australia's first student-led algorithmic trading hackathon.

Full rules, scoring, schedule, and submission details live on the **[Algothon 2026 Wiki](https://wiki.algothon.au/)** - this README only covers what's in this repo and how to run it. If anything here ever seems to disagree with the wiki, the wiki is correct.

## What's in this repo

| File | What it's for |
| :--- | :--- |
| `teamName.py` | Boilerplate for your algorithm - implement `getMyPosition(prcSoFar)` here. Only renamed to `<YourTeamName>.py` at submission time (see below). |
| `eval.py` | The official evaluation script - scores the last 250 days of whatever `prices.txt` you give it. |
| `prices.txt` | Current stage's price data. |
| `requirements-dev.txt` | The exact package set available at grading time, for setting up a local environment that matches the sandbox. **Do not include this in your submission.** |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements-dev.txt
```

1. Implement `getMyPosition(prcSoFar)` in `teamName.py`. No need to rename anything or touch `eval.py` while developing - `eval.py` imports from `teamName` by default.
2. Run `python eval.py` to backtest locally.

## Strategy in this branch

This branch adds a lightweight regime-aware strategy in `strategy.py` and keeps the submission interface in `teamName.py`.

- **Trend component:** short-vs-long moving-average spread, normalized by recent volatility.
- **Mean-reversion component:** negative rolling z-score versus a rolling mean.
- **Regime detection:** rolling directional persistence decides whether to trust trend more or mean-reversion more.
- **Risk control:** recent volatility scales exposure down when the market gets noisy, then positions are clipped to the competition limits.
- **Turnover control:** EMA smoothing plus a small deadband reduces unnecessary churn.

All features use rolling windows only, so the strategy at time `t` depends only on prices observed up to `t`.

## Running the strategy and backtest

Run the official evaluator:

```bash
python eval.py
```

Run the lightweight research backtest:

```bash
python backtest.py --prices prices.txt
```

Optional walk-forward controls:

```bash
python backtest.py --prices prices.txt --min-train 250 --test-size 125
```

`backtest.py` reports:

- cumulative PnL
- mean daily PnL
- Sharpe-like metric
- score using the competition formula
- max drawdown
- turnover
- expanding-window walk-forward split results

## Robustness / tuning guidance

To keep the strategy generalizable as `prices.txt` grows by roughly 50 rows per day:

- keep the parameter count low and avoid adding narrowly tuned thresholds
- prefer broad, stable window lengths instead of optimizing for one historical segment
- validate on multiple rolling or expanding out-of-sample windows, not just one backtest
- avoid re-optimizing every day on tiny increments of new data
- if you tune anything, change one parameter at a time and confirm the walk-forward profile stays stable

The implementation is intentionally simple and competition-friendly: it uses only `numpy` and `pandas`, preserves the starter-code `getMyPosition(prcSoFar)` interface, and avoids any future-data leakage.

## Submitting

Only when you're ready to submit: copy `teamName.py` to `<YourTeamName>.py` (matching your registered team name) and zip it up - `eval.py` and `prices.txt` are not part of the submission, only your algorithm file (and `requirements.txt`, if you used extra packages). See the [Submission Guide](https://wiki.algothon.au/submission/) for exact packaging requirements. Submit through the [live leaderboard](https://www.algothon.au/leaderboard).

## Questions

Post in the questions forum on our Discord - moderators are there to help.
