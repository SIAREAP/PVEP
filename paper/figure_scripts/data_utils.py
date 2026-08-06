from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def bootstrap_mean_interval(
    values: np.ndarray | pd.Series,
    *,
    confidence: float = 0.95,
    n_boot: int = 20_000,
    seed: int = 20260803,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("bootstrap input must be a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return tuple(float(x) for x in np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0]))


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0 or not (0 <= successes <= trials):
        raise ValueError(f"invalid binomial counts: {successes}/{trials}")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return float(center - half), float(center + half)


def binomial_error_percent(successes: int, trials: int) -> tuple[float, float, float]:
    rate = 100.0 * successes / trials
    low, high = wilson_interval(successes, trials)
    # Clamp tiny floating-point excursions (for example, -1e-16 at 0/90),
    # because Matplotlib rejects negative error-bar magnitudes.
    return rate, max(0.0, rate - 100.0 * low), max(0.0, 100.0 * high - rate)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asymmetric_yerr(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return np.vstack((values - lows, highs - values))
