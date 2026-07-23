from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from strategy import (
    DEFAULT_COMMISSION,
    DEFAULT_CONFIG,
    INST0_COMMISSION,
    StrategyConfig,
    generate_positions,
    load_prices,
)


def score(mean_pnl: float, std_pnl: float, param: float = 1.0) -> float:
    if mean_pnl <= 0 or std_pnl < 1e-10:
        return mean_pnl
    sharpe = np.sqrt(250.0) * mean_pnl / std_pnl
    return float(mean_pnl * (sharpe**2 / (sharpe**2 + param**2)))


def _commission_rates(n_inst: int) -> np.ndarray:
    rates = np.full(n_inst, DEFAULT_COMMISSION, dtype=float)
    if n_inst > 0:
        rates[0] = INST0_COMMISSION
    return rates


def _max_drawdown(daily_pnl: np.ndarray) -> float:
    equity = np.cumsum(daily_pnl)
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity - running_peak
    return float(drawdown.min(initial=0.0))


def simulate(prices: np.ndarray, positions: np.ndarray) -> dict[str, float | np.ndarray]:
    if prices.shape != positions.shape:
        raise ValueError("Prices and positions must share the same shape")
    if prices.shape[1] < 2:
        raise ValueError("Need at least two observations to backtest")

    commission_rates = _commission_rates(prices.shape[0])[:, None]
    held_positions = positions[:, :-1]
    trades = held_positions.copy()
    if held_positions.shape[1] > 1:
        trades[:, 1:] = held_positions[:, 1:] - held_positions[:, :-1]
    trade_prices = prices[:, :-1]
    next_returns = prices[:, 1:] - prices[:, :-1]

    commissions = np.sum(np.abs(trades) * trade_prices * commission_rates, axis=0)
    daily_pnl = np.sum(held_positions * next_returns, axis=0) - commissions
    turnover = np.sum(np.abs(trades) * trade_prices, axis=0)

    mean_pnl = float(np.mean(daily_pnl))
    std_pnl = float(np.std(daily_pnl))
    sharpe = float(np.sqrt(250.0) * mean_pnl / std_pnl) if std_pnl > 0 else 0.0

    return {
        "daily_pnl": daily_pnl,
        "turnover_series": turnover,
        "cumulative_pnl": float(np.sum(daily_pnl)),
        "mean_pnl": mean_pnl,
        "sharpe": sharpe,
        "score": float(score(mean_pnl, std_pnl)),
        "max_drawdown": _max_drawdown(daily_pnl),
        "avg_turnover": float(np.mean(turnover)),
        "total_turnover": float(np.sum(turnover)),
    }


def walk_forward_splits(
    n_obs: int,
    min_train: int,
    test_size: int,
) -> list[tuple[int, int]]:
    splits: list[tuple[int, int]] = []
    train_end = min_train
    while train_end < n_obs - 1:
        test_end = min(train_end + test_size, n_obs)
        if test_end - train_end < 2:
            break
        splits.append((train_end, test_end))
        train_end = test_end
    return splits


def evaluate_walk_forward(
    prices: np.ndarray,
    config: StrategyConfig = DEFAULT_CONFIG,
    min_train: int = 250,
    test_size: int = 125,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for train_end, test_end in walk_forward_splits(prices.shape[1], min_train=min_train, test_size=test_size):
        prefix_prices = prices[:, :test_end]
        prefix_positions = generate_positions(prefix_prices, config=config)
        simulation = simulate(prefix_prices, prefix_positions)
        test_slice = slice(train_end - 1, test_end - 1)
        split_pnl = simulation["daily_pnl"][test_slice]
        split_turnover = simulation["turnover_series"][test_slice]
        mean_pnl = float(np.mean(split_pnl))
        std_pnl = float(np.std(split_pnl))
        sharpe = float(np.sqrt(250.0) * mean_pnl / std_pnl) if std_pnl > 0 else 0.0
        results.append(
            {
                "train_end": train_end,
                "test_end": test_end,
                "days": float(test_end - train_end),
                "cumulative_pnl": float(np.sum(split_pnl)),
                "mean_pnl": mean_pnl,
                "sharpe": sharpe,
                "score": float(score(mean_pnl, std_pnl)),
                "max_drawdown": _max_drawdown(split_pnl),
                "avg_turnover": float(np.mean(split_turnover)),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the robust regime-aware strategy")
    parser.add_argument("--prices", default="prices.txt", help="Path to price file")
    parser.add_argument("--min-train", type=int, default=250, help="Initial expanding-window train size")
    parser.add_argument("--test-size", type=int, default=125, help="Walk-forward test chunk size")
    args = parser.parse_args()

    prices = load_prices(args.prices)
    positions = generate_positions(prices, config=DEFAULT_CONFIG)
    summary = simulate(prices, positions)
    walk_forward = evaluate_walk_forward(
        prices,
        config=DEFAULT_CONFIG,
        min_train=args.min_train,
        test_size=args.test_size,
    )

    print("Strategy config:", asdict(DEFAULT_CONFIG))
    print(f"Loaded {prices.shape[0]} instruments across {prices.shape[1]} observations")
    print("Full-sample metrics:")
    print(f"  cumulative_pnl: {summary['cumulative_pnl']:.2f}")
    print(f"  mean_pnl:       {summary['mean_pnl']:.2f}")
    print(f"  sharpe:         {summary['sharpe']:.2f}")
    print(f"  score:          {summary['score']:.2f}")
    print(f"  max_drawdown:   {summary['max_drawdown']:.2f}")
    print(f"  avg_turnover:   {summary['avg_turnover']:.2f}")
    print("")
    print("Walk-forward splits:")
    for split in walk_forward:
        print(
            "  "
            f"train<={int(split['train_end'])} test({int(split['days'])}d) "
            f"pnl={split['cumulative_pnl']:.2f} "
            f"sharpe={split['sharpe']:.2f} "
            f"score={split['score']:.2f} "
            f"mdd={split['max_drawdown']:.2f} "
            f"turnover={split['avg_turnover']:.2f}"
        )


if __name__ == "__main__":
    main()
