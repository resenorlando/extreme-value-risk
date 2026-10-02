"""Out-of-sample VaR backtesting with formal coverage tests.

All forecasts are strictly causal: the VaR used on day t is estimated only
from returns up to and including day t-1 (an expanding window). We then compare
three one-day-ahead VaR models -- Gaussian, Historical, and EVT (POT-GPD) --
using the exception (violation) sequence and two likelihood-ratio tests:

  - Kupiec (1995) Proportion-of-Failures (POF) test  -> unconditional coverage
  - Christoffersen (1998) independence test          -> exception clustering
  - Their sum -> Christoffersen conditional-coverage test

A well-specified VaR model should have an exception rate ~ (1-p) AND exceptions
that arrive independently (no clustering in crises).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

import evt


def _refit_days(n: int, start: int, step: int):
    """Indices (into the return array) at which we re-estimate the model."""
    return set(range(start, n, step))


def rolling_var_forecasts(returns: np.ndarray, p: float,
                          window: int = 1000, refit_step: int = 21,
                          pot_q: float = 0.95, lam: float = 0.94):
    """Expanding-window, causal one-day-ahead VaR for five models.

    Unconditional:  Gaussian, Historical, EVT (static POT-GPD)
    Conditional:    CondGaussian (EWMA vol + normal, i.e. RiskMetrics),
                    CondEVT (McNeil-Frey: EWMA vol + EVT on standardised residuals)

    Returns dict of model -> array of VaR forecasts aligned to returns[window:],
    plus the realised returns for that out-of-sample span.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    oos_idx = np.arange(window, n)

    # EWMA volatility is causal: sigma[t] is the forecast for day t from t-1 data
    sigma = evt.ewma_vol(r, lam=lam)
    resid = r / sigma  # standardised residuals (also causal, std ~ 1)
    z_normal = stats.norm.ppf(p)

    models = {m: [] for m in ["Gaussian", "Historical", "EVT",
                              "CondGaussian", "CondEVT"]}
    last_fit = None       # static (unconditional) POT fit
    last_fit_z = None     # POT fit to standardised-residual losses
    for t in oos_idx:
        hist = r[:t]  # strictly causal
        models["Gaussian"].append(evt.gaussian_var(hist, p))
        models["Historical"].append(evt.historical_var(hist, p))
        models["CondGaussian"].append(sigma[t] * z_normal)
        if last_fit is None or (t - window) % refit_step == 0:
            losses = evt.losses_from_returns(hist)
            try:
                last_fit = evt.pot_fit(losses, np.quantile(losses, pot_q))
            except ValueError:
                last_fit = None
            zloss = evt.losses_from_returns(resid[:t])
            try:
                last_fit_z = evt.pot_fit(zloss, np.quantile(zloss, pot_q))
            except ValueError:
                last_fit_z = None
        models["EVT"].append(evt.pot_var(last_fit, p) if last_fit
                             else evt.historical_var(hist, p))
        if last_fit_z is not None:
            models["CondEVT"].append(sigma[t] * evt.pot_var(last_fit_z, p))
        else:
            models["CondEVT"].append(sigma[t] * z_normal)

    out = {m: np.array(v) for m, v in models.items()}
    out["_realised"] = r[oos_idx]
    return out


def exceptions(realised_returns: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """Binary exception sequence: 1 if the realised loss exceeded VaR."""
    return (np.asarray(realised_returns) < -np.asarray(var_forecast)).astype(int)


def kupiec_pof(exc: np.ndarray, p: float):
    """Kupiec unconditional-coverage LR test. H0: exception rate = 1-p."""
    exc = np.asarray(exc)
    n = len(exc)
    x = int(exc.sum())
    pi_exp = 1.0 - p
    pi_obs = x / n if n else 0.0
    if x == 0 or x == n:  # avoid log(0); use a tiny floor
        pi_obs = min(max(pi_obs, 1e-12), 1 - 1e-12)
    ll_null = (n - x) * np.log(1 - pi_exp) + x * np.log(pi_exp)
    ll_obs = (n - x) * np.log(1 - pi_obs) + x * np.log(pi_obs)
    lr = -2.0 * (ll_null - ll_obs)
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    return {"n": n, "exceptions": x, "rate": x / n if n else np.nan,
            "expected_rate": pi_exp, "LR": float(lr), "pvalue": float(pval)}


def christoffersen_independence(exc: np.ndarray):
    """Christoffersen independence LR test on the exception transitions.

    Degenerate sequences (fewer than 2 points, or no exceptions at all) carry no
    information about clustering, so we return LR=0, p=1 (cannot reject
    independence) rather than dividing by zero.
    """
    exc = np.asarray(exc)
    if len(exc) < 2 or exc.sum() == 0 or exc.sum() == len(exc):
        return {"LR": 0.0, "pvalue": 1.0, "n00": 0, "n01": 0, "n10": 0, "n11": 0,
                "note": "degenerate exception sequence"}
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(exc[:-1], exc[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    eps = 1e-12
    def _l(a, b, pr):
        pr = min(max(pr, eps), 1 - eps)
        return a * np.log(1 - pr) + b * np.log(pr)
    ll_null = _l(n00 + n10, n01 + n11, pi)
    ll_alt = _l(n00, n01, pi01) + _l(n10, n11, pi11)
    lr = -2.0 * (ll_null - ll_alt)
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    return {"LR": float(lr), "pvalue": float(pval),
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def backtest_summary(realised: np.ndarray, var_forecast: np.ndarray, p: float):
    exc = exceptions(realised, var_forecast)
    kp = kupiec_pof(exc, p)
    ci = christoffersen_independence(exc)
    cc_lr = kp["LR"] + ci["LR"]
    cc_p = 1.0 - stats.chi2.cdf(cc_lr, df=2)
    return {"kupiec": kp, "independence": ci,
            "cond_coverage_LR": float(cc_lr), "cond_coverage_p": float(cc_p)}
