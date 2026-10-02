"""Validation of the estimators on synthetic data with KNOWN answers.

Run:  python src/tests_evt.py
These are the checks that make the tail estimates defensible: before trusting
the estimator on markets, show it recovers the truth on data we generated.
"""
import numpy as np
from scipy import stats

import evt
import var_backtest as vb


def test_hill_recovers_pareto_alpha():
    # Pareto(alpha) has a power-law tail with index alpha. Hill should recover it.
    rng = np.random.default_rng(1)
    alpha_true = 3.0
    x = (1 - rng.random(200_000)) ** (-1.0 / alpha_true)  # Pareto, x>=1
    k = int(0.05 * len(x))
    alpha_hat = evt.hill_estimator(x, k)
    assert abs(alpha_hat - alpha_true) < 0.15, alpha_hat
    return alpha_true, alpha_hat


def test_gpd_recovers_shape():
    # Exceedances of a GPD(xi) tail should recover xi via the POT MLE.
    rng = np.random.default_rng(2)
    xi_true, beta_true = 0.3, 1.0
    y = stats.genpareto.rvs(c=xi_true, scale=beta_true, size=100_000, random_state=rng)
    losses = y  # already "exceedances" over 0
    fit = evt.pot_fit(losses, u=0.0)
    assert abs(fit["xi"] - xi_true) < 0.05, fit
    return xi_true, fit["xi"]


def test_kupiec_size_is_calibrated():
    # Under a correctly specified model, exceptions are iid Bernoulli(1-p).
    # The Kupiec test should reject ~5% of the time at the 5% level.
    rng = np.random.default_rng(3)
    p = 0.99
    rejects = 0
    trials = 2000
    for _ in range(trials):
        exc = (rng.random(2500) < (1 - p)).astype(int)
        if vb.kupiec_pof(exc, p)["pvalue"] < 0.05:
            rejects += 1
    size = rejects / trials
    assert 0.02 < size < 0.09, size
    return size


def test_evt_var_monotone_in_p():
    # VaR must increase with confidence level.
    rng = np.random.default_rng(4)
    r = stats.t.rvs(df=4, size=50_000, random_state=rng) * 0.01
    losses = evt.losses_from_returns(r)
    u = np.quantile(losses, 0.95)
    fit = evt.pot_fit(losses, u)
    v99, v999 = evt.pot_var(fit, 0.99), evt.pot_var(fit, 0.999)
    assert v999 > v99 > 0, (v99, v999)
    return v99, v999


def test_ewma_is_causal():
    # sigma[t] must depend only on returns BEFORE t. Perturbing the future must
    # not change any earlier volatility forecast.
    rng = np.random.default_rng(5)
    r = rng.standard_normal(3000) * 0.01
    sig1 = evt.ewma_vol(r)
    r2 = r.copy()
    t0 = 1500
    r2[t0:] += rng.standard_normal(len(r) - t0) * 0.05  # shock the future
    sig2 = evt.ewma_vol(r2)
    # sigma[t] for t <= t0 uses r[:t] (i.e. through t-1) -> unchanged
    assert np.allclose(sig1[: t0 + 1], sig2[: t0 + 1]), "EWMA leaked the future"
    return t0


def test_var_forecasts_are_causal():
    # A one-day-ahead VaR at OOS day i must not move when we perturb returns that
    # occur strictly AFTER its forecast origin.
    rng = np.random.default_rng(6)
    r = stats.t.rvs(df=5, size=1600, random_state=rng) * 0.01
    fc1 = vb.rolling_var_forecasts(r, 0.99, window=1000, refit_step=21)
    r2 = r.copy()
    r2[1300:] += rng.standard_normal(len(r) - 1300) * 0.05  # shock the future
    fc2 = vb.rolling_var_forecasts(r2, 0.99, window=1000, refit_step=21)
    # OOS index 0 == calendar day 1000; forecasts up to day 1300 (idx 300) are
    # made from data before the shock and must be identical for every model.
    for m in ["Gaussian", "Historical", "EVT", "CondGaussian", "CondEVT"]:
        assert np.allclose(fc1[m][:300], fc2[m][:300]), f"{m} leaked the future"
    return True


if __name__ == "__main__":
    print("Running estimator validation tests...\n")
    a_true, a_hat = test_hill_recovers_pareto_alpha()
    print(f"[PASS] Hill on Pareto(alpha=3):     recovered alpha = {a_hat:.3f}")
    xi_true, xi_hat = test_gpd_recovers_shape()
    print(f"[PASS] GPD MLE on GPD(xi=0.3):      recovered xi    = {xi_hat:.3f}")
    size = test_kupiec_size_is_calibrated()
    print(f"[PASS] Kupiec test size @5% level:  empirical size  = {size:.3f}")
    v99, v999 = test_evt_var_monotone_in_p()
    print(f"[PASS] EVT VaR monotone in p:        99%={v99:.4f} < 99.9%={v999:.4f}")
    t0 = test_ewma_is_causal()
    print(f"[PASS] EWMA causality:               future shock @t={t0} left sigma[:t] unchanged")
    test_var_forecasts_are_causal()
    print("[PASS] VaR forecast causality:       future shock left pre-shock forecasts unchanged")
    print("\nAll validation checks passed.")
