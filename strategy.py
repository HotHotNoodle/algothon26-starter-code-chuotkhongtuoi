from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_COMMISSION = 0.0001
INST0_COMMISSION = 0.00002
DEFAULT_DOLLAR_LIMIT = 10_000.0
INST0_DOLLAR_LIMIT = 100_000.0


@dataclass(frozen=True)
class StrategyConfig:
    short_window: int = 10
    long_window: int = 40
    mean_reversion_window: int = 20
    volatility_window: int = 20
    regime_window: int = 30
    signal_clip: float = 3.0
    target_volatility: float = 0.02
    max_risk_scale: float = 2.0
    smoothing_alpha: float = 0.35
    deadband: float = 0.05
    trend_weight_floor: float = 0.15
    trend_weight_ceiling: float = 0.55


DEFAULT_CONFIG = StrategyConfig()


def _coerce_history(prices: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected 1D or 2D price history")
    return arr


def load_prices(path: str | Path) -> np.ndarray:
    raw_text = Path(path).read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"No price data found in {path}")

    normalized = raw_text.replace(",", " ").replace(";", " ").replace("\t", " ")
    frame = pd.read_csv(StringIO(normalized), sep=r"\s+", header=None, engine="python")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError(f"Unable to parse price data from {path}")

    values = numeric.to_numpy(dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values.T


def _price_frame(prices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(prices.T)


def _dollar_limits(n_inst: int) -> np.ndarray:
    limits = np.full(n_inst, DEFAULT_DOLLAR_LIMIT, dtype=float)
    if n_inst > 0:
        limits[0] = INST0_DOLLAR_LIMIT
    return limits


def _position_limits(prices: np.ndarray) -> np.ndarray:
    limits = _dollar_limits(prices.shape[0])[:, None]
    return np.maximum(np.floor(limits / np.maximum(prices, EPS)), 1.0)


def _apply_deadband(weights: pd.DataFrame, deadband: float) -> pd.DataFrame:
    arr = weights.to_numpy(dtype=float, copy=True)
    if arr.size == 0:
        return weights
    for t in range(1, arr.shape[0]):
        prev = arr[t - 1]
        cur = arr[t]
        small_move = np.abs(cur - prev) < deadband
        arr[t, small_move] = prev[small_move]
    return pd.DataFrame(arr, index=weights.index, columns=weights.columns)


def generate_position_weights(
    prices: np.ndarray | list[float] | list[list[float]],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    history = _coerce_history(prices)
    price_df = _price_frame(history)
    returns = price_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    short_mean = price_df.rolling(config.short_window, min_periods=config.short_window).mean()
    long_mean = price_df.rolling(config.long_window, min_periods=config.long_window).mean()
    trend_raw = (short_mean - long_mean) / long_mean.clip(lower=EPS)

    volatility = returns.rolling(
        config.volatility_window,
        min_periods=config.volatility_window,
    ).std().clip(lower=1e-4)
    trend_signal = (trend_raw / volatility).clip(-config.signal_clip, config.signal_clip)

    mr_mean = price_df.rolling(
        config.mean_reversion_window,
        min_periods=config.mean_reversion_window,
    ).mean()
    mr_std = price_df.rolling(
        config.mean_reversion_window,
        min_periods=config.mean_reversion_window,
    ).std().clip(lower=1e-4)
    mr_signal = (-(price_df - mr_mean) / mr_std).clip(-config.signal_clip, config.signal_clip)

    directional_persistence = (
        np.sign(returns)
        .rolling(config.regime_window, min_periods=config.regime_window)
        .mean()
        .abs()
    )
    trend_weight = (
        (directional_persistence - config.trend_weight_floor)
        / max(config.trend_weight_ceiling - config.trend_weight_floor, EPS)
    ).clip(0.0, 1.0)

    blended_signal = trend_weight * trend_signal + (1.0 - trend_weight) * mr_signal
    risk_scale = (config.target_volatility / volatility).clip(upper=config.max_risk_scale)
    scaled_signal = blended_signal * risk_scale
    raw_weights = pd.DataFrame(
        np.tanh(scaled_signal.to_numpy(dtype=float)),
        index=price_df.index,
        columns=price_df.columns,
    ).fillna(0.0)

    smooth_weights = raw_weights.ewm(alpha=config.smoothing_alpha, adjust=False).mean()
    smooth_weights = _apply_deadband(smooth_weights, config.deadband)
    smooth_weights = smooth_weights.clip(-1.0, 1.0).fillna(0.0)
    return smooth_weights.to_numpy(dtype=float).T


def generate_positions(
    prices: np.ndarray | list[float] | list[list[float]],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    history = _coerce_history(prices)
    weights = generate_position_weights(history, config=config)
    limits = _position_limits(history)
    shares = np.rint(weights * limits)
    shares = np.clip(shares, -limits, limits)
    return shares.astype(int)


def generate_latest_position(
    prices: np.ndarray | list[float] | list[list[float]],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    return generate_positions(prices, config=config)[:, -1]
