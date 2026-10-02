"""Extreme Value Theory estimators for financial tail risk.

Implements, from first principles (numpy/scipy only):
  - Hill estimator of the tail index alpha (and xi = 1/alpha) for a heavy tail
  - Hill plot across the order statistic k
  - Peaks-Over-Threshold (POT) generalised-Pareto (GPD) fit via MLE
  - EVT closed-form VaR / Expected Shortfall from the POT fit
  - Gaussian and Historical VaR for comparison
  - Stationary bootstrap for confidence intervals

Convention: we work with LOSSES. If r_t are simple/log returns, the loss is
L_t = -r_t, and we study the UPPER tail of L (the largest losses). A VaR at
confidence p is the positive loss threshold V with P(L > V) = 1 - p.

References:
  Hill (1975); Pickands (1975); McNeil, Frey & Embrechts, "Quantitative Risk
  Management"; Politis & Romano (1994) stationary bootstrap.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Returns / losses
# --------------------------------------------------------------------------- #
def log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


def losses_from_returns(returns: np.ndarray) -> np.ndarray:
    """Loss series L = -r (positive number = a loss)."""
    return -np.asarray(returns, dtype=float)


# --------------------------------------------------------------------------- #
# Hill estimator
# --------------------------------------------------------------------------- #
def hill_estimator(data: np.ndarray, k: int) -> float:
    """Hill estimator of the tail index alpha using the k largest observations.

    alpha_hat = 1 / ( (1/k) sum_{i=1}^{k} ln X_(i) - ln X_(k+1) )
    where X_(1) >= X_(2) >= ... are the order statistics of the POSITIVE data.
    Only strictly positive observations are used (losses in the upper tail).
    """
    x = np.sort(np.asarray(data, dtype=float))[::-1]  # descending
    x = x[x > 0]
    if k < 2 or k >= len(x):
        return np.nan
    top = x[:k]
    thresh = x[k]  # the (k+1)-th largest, i.e. X_(k+1)
    if thresh <= 0:
        return np.nan
    hill = np.mean(np.log(top)) - np.log(thresh)
    if hill <= 0:
        return np.nan
    return 1.0 / hill


def hill_plot(data: np.ndarray, k_min: int = 10, k_max: int | None = None):
    """Return (ks, alphas) for a Hill plot over a range of k."""
    x = np.sort(np.asarray(data, dtype=float))[::-1]
    x = x[x > 0]
    n = len(x)
    if k_max is None:
        k_max = int(0.25 * n)
    ks = np.arange(k_min, min(k_max, n - 1))
    alphas = np.array([hill_estimator(data, int(k)) for k in ks])
    return ks, alphas


# --------------------------------------------------------------------------- #
# EWMA volatility filter (RiskMetrics) -- for conditional EVT
# --------------------------------------------------------------------------- #
def ewma_vol(returns: np.ndarray, lam: float = 0.94, sigma0: float | None = None) -> np.ndarray:
    """EWMA volatility forecast: sigma[t] is made at the close of day t-1.

    sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2

    Causality: for every t >= 1, sigma[t] is a function of r_0,...,r_{t-1} only
    (strictly ex-ante). sigma[0] is a *seeded prior* (r_0^2 by default, or
    sigma0^2 if given) since no pre-sample data exist; it therefore uses r_0.
    This seed is deliberately harmless: the out-of-sample backtest discards a
    long warm-up (WINDOW=1000 days), so the reported results never depend on it.
    Verified by the causality unit tests in tests_evt.py.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sig2 = np.empty(n)
    # Seed the day-0 prior without using any future data.
    sig2_prev = float(r[0] ** 2) if sigma0 is None else float(sigma0 ** 2)
    for t in range(n):
        sig2[t] = sig2_prev
        sig2_prev = lam * sig2_prev + (1 - lam) * r[t] ** 2
    return np.sqrt(sig2)


# --------------------------------------------------------------------------- #
# Peaks-Over-Threshold / Generalised Pareto
# --------------------------------------------------------------------------- #
def pot_fit(losses: np.ndarray, u: float):
    """Fit a GPD to exceedances of losses over threshold u via MLE.

    Returns dict with xi (shape), beta (scale), u, n, n_u.
    Uses scipy genpareto with the location fixed at 0.
    """
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    exceed = losses[losses > u] - u
    n_u = len(exceed)
    if n_u < 20:
        raise ValueError(f"Too few exceedances ({n_u}) over u={u:.4g}")
    xi, loc, beta = stats.genpareto.fit(exceed, floc=0.0)
    return {"xi": float(xi), "beta": float(beta), "u": float(u),
            "n": int(n), "n_u": int(n_u)}


def pot_var(fit: dict, p: float) -> float:
    """EVT VaR at confidence p (e.g. 0.99) from a POT fit (McNeil et al.)."""
    xi, beta, u, n, n_u = fit["xi"], fit["beta"], fit["u"], fit["n"], fit["n_u"]
    ratio = (n / n_u) * (1.0 - p)
    if abs(xi) < 1e-8:
        return u - beta * np.log(ratio)
    return u + (beta / xi) * (ratio ** (-xi) - 1.0)


def pot_es(fit: dict, p: float) -> float:
    """EVT Expected Shortfall at confidence p from a POT fit (requires xi<1)."""
    xi, beta, u = fit["xi"], fit["beta"], fit["u"]
    var = pot_var(fit, p)
    if xi >= 1:
        return np.nan  # infinite mean tail
    return var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)


# --------------------------------------------------------------------------- #
# Parametric / empirical VaR for comparison
# --------------------------------------------------------------------------- #
def gaussian_var(returns: np.ndarray, p: float) -> float:
    """Gaussian VaR (positive loss) at confidence p from return mean/std."""
    r = np.asarray(returns, dtype=float)
    mu, sig = np.mean(r), np.std(r, ddof=1)
    return -(mu + sig * stats.norm.ppf(1.0 - p))


def historical_var(returns: np.ndarray, p: float) -> float:
    """Historical VaR (positive loss) = -empirical (1-p) quantile of returns."""
    r = np.asarray(returns, dtype=float)
    return -np.quantile(r, 1.0 - p)


def gaussian_es(returns: np.ndarray, p: float) -> float:
    r = np.asarray(returns, dtype=float)
    mu, sig = np.mean(r), np.std(r, ddof=1)
    z = stats.norm.ppf(1.0 - p)
    return -(mu - sig * stats.norm.pdf(z) / (1.0 - p))


# --------------------------------------------------------------------------- #
# Stationary bootstrap (Politis & Romano) for CIs on a statistic
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(n: int, mean_block: float, rng) -> np.ndarray:
    """One resample of indices of length n via the stationary bootstrap."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block
    i = rng.integers(0, n)
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
    return idx


def bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = 500,
                 mean_block: float = 20.0, alpha: float = 0.05, seed: int = 0):
    """Percentile CI for stat_fn(resample) using the stationary bootstrap."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        v = stat_fn(data[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), float(np.mean(vals)), float(np.std(vals))
