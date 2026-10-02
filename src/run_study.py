"""Main study: fat tails & extreme risk across global equity indices.

Outputs (into ../results and ../figures):
  - results.json            machine-readable results
  - results.md              human-readable tables (the research note's numbers)
  - hill_plot_SP500.png     tail-index stability vs threshold
  - gpd_tailfit_SP500.png   empirical vs GPD-modelled exceedance tail
  - var_backtest_SP500.png  OOS VaR lines and exceptions
  - tail_index_bars.png     tail index across indices with bootstrap CIs

Run:  python src/run_study.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evt
import var_backtest as vb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

PRIMARY = "SP500"
LEVELS = [0.99, 0.999]
POT_Q = 0.95          # POT threshold = 95th percentile of losses
HILL_FRAC = 0.02      # Hill k = 2% of sample (stable region; see Hill plot)
BACKTEST_P = 0.99
WINDOW = 1000         # ~4 trading years of warm-up before OOS backtest


def load_returns() -> dict[str, np.ndarray]:
    wide = pd.read_csv(DATA / "prices_wide.csv", index_col=0)
    out = {}
    for col in wide.columns:
        s = wide[col].dropna()
        out[col] = evt.log_returns(s.values)
    return out


def per_index_tail_stats(returns: np.ndarray) -> dict:
    losses = evt.losses_from_returns(returns)
    n = len(losses)
    k = int(HILL_FRAC * n)
    alpha = evt.hill_estimator(losses, k)
    lo, hi, mean, sd = evt.bootstrap_ci(
        losses, lambda d: evt.hill_estimator(d, int(HILL_FRAC * len(d))),
        n_boot=300, mean_block=20.0, seed=7,
    )
    u = np.quantile(losses, POT_Q)
    fit = evt.pot_fit(losses, u)
    row = {
        "n": n, "hill_k": k,
        "alpha_hill": alpha, "alpha_ci": [lo, hi],
        "xi_pot": fit["xi"], "beta_pot": fit["beta"],
        "u": fit["u"], "n_u": fit["n_u"],
    }
    for p in LEVELS:
        row[f"VaR_EVT_{p}"] = evt.pot_var(fit, p)
        row[f"ES_EVT_{p}"] = evt.pot_es(fit, p)
        row[f"VaR_Gauss_{p}"] = evt.gaussian_var(returns, p)
        row[f"VaR_Hist_{p}"] = evt.historical_var(returns, p)
        row[f"ratio_EVT_Gauss_{p}"] = row[f"VaR_EVT_{p}"] / row[f"VaR_Gauss_{p}"]
    return row


def figure_hill(returns, name):
    losses = evt.losses_from_returns(returns)
    ks, alphas = evt.hill_plot(losses, k_min=15, k_max=int(0.20 * len(losses)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, alphas, lw=1.6, color="#1f4e79")
    kstar = int(HILL_FRAC * len(losses))
    a_star = evt.hill_estimator(losses, kstar)
    ax.axvline(kstar, color="#c0392b", ls="--", lw=1,
               label=f"k={HILL_FRAC*100:.0f}% of N -> alpha={a_star:.2f}")
    ax.set_xlabel("k  (number of upper-tail order statistics)")
    ax.set_ylabel("Hill tail index  alpha")
    ax.set_title(f"Hill plot - {name} daily losses (20y)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"hill_plot_{name}.png", dpi=140)
    plt.close(fig)


def figure_tailfit(returns, name):
    losses = evt.losses_from_returns(returns)
    u = np.quantile(losses, POT_Q)
    fit = evt.pot_fit(losses, u)
    exceed = np.sort(losses[losses > u])
    n_u = len(exceed)
    emp = 1.0 - (np.arange(1, n_u + 1) - 0.5) / n_u  # empirical survival
    xi, beta = fit["xi"], fit["beta"]
    model = (1 + xi * (exceed - u) / beta) ** (-1.0 / xi)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(exceed * 100, emp, ".", ms=3, color="#555",
                label="Empirical exceedances")
    ax.semilogy(exceed * 100, model, "-", lw=2, color="#c0392b",
                label=f"GPD fit (xi={xi:.2f})")
    ax.set_xlabel("Daily loss over threshold (%)")
    ax.set_ylabel("P(Loss > x | Loss > u)")
    ax.set_title(f"POT-GPD tail fit - {name} (threshold = 95th pct)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / f"gpd_tailfit_{name}.png", dpi=140)
    plt.close(fig)


def run_backtest(returns, name):
    fc = vb.rolling_var_forecasts(returns, BACKTEST_P, window=WINDOW)
    realised = fc["_realised"]
    summ = {}
    for m in ["Gaussian", "Historical", "EVT", "CondGaussian", "CondEVT"]:
        summ[m] = vb.backtest_summary(realised, fc[m], BACKTEST_P)
    # figure for the primary index
    if name == PRIMARY:
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(realised))
        ax.plot(x, realised * 100, color="#999", lw=0.5, label="Daily return")
        for m, c in [("Gaussian", "#2980b9"), ("CondEVT", "#c0392b")]:
            ax.plot(x, -fc[m] * 100, lw=1.1, color=c, label=f"{m} 99% VaR")
            exc = vb.exceptions(realised, fc[m])
            ax.plot(x[exc == 1], realised[exc == 1] * 100, "o", ms=3, color=c,
                    alpha=0.6)
        ax.set_title(f"Out-of-sample 99% VaR - {name} (exceptions marked)")
        ax.set_ylabel("%")
        ax.set_xlabel("Out-of-sample trading day")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"var_backtest_{name}.png", dpi=140)
        plt.close(fig)
    return summ


def figure_tailindex_bars(stats_by_index):
    names = list(stats_by_index.keys())
    alphas = [stats_by_index[n]["alpha_hill"] for n in names]
    los = [stats_by_index[n]["alpha_ci"][0] for n in names]
    his = [stats_by_index[n]["alpha_ci"][1] for n in names]
    err = [[a - l for a, l in zip(alphas, los)], [h - a for a, h in zip(alphas, his)]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, alphas, yerr=err, capsize=4, color="#1f4e79", alpha=0.85)
    ax.axhline(2, color="#c0392b", ls="--", lw=1,
               label="alpha=2 (finite-variance boundary)")
    ax.set_ylabel("Hill tail index  alpha")
    ax.set_title("Left-tail index across global equity indices (95% CI)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "tail_index_bars.png", dpi=140)
    plt.close(fig)


def main():
    rets = load_returns()
    stats_by_index = {name: per_index_tail_stats(r) for name, r in rets.items()}
    backtests = {name: run_backtest(r, name) for name, r in rets.items()}

    figure_hill(rets[PRIMARY], PRIMARY)
    figure_tailfit(rets[PRIMARY], PRIMARY)
    figure_tailindex_bars(stats_by_index)

    results = {"tail_stats": stats_by_index, "backtests": backtests,
               "config": {"POT_Q": POT_Q, "HILL_FRAC": HILL_FRAC,
                          "BACKTEST_P": BACKTEST_P, "WINDOW": WINDOW}}
    (RES / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    # ---- human-readable markdown ----
    lines = ["# Fat Tails & Extreme Risk ,  Results\n"]
    lines.append("## 1. Tail index (Hill) and GPD shape per index\n")
    lines.append("| Index | N | alpha (Hill) | 95% CI | xi (GPD) |")
    lines.append("|---|---|---|---|---|")
    for n, s in stats_by_index.items():
        lines.append(f"| {n} | {s['n']} | {s['alpha_hill']:.2f} | "
                     f"[{s['alpha_ci'][0]:.2f}, {s['alpha_ci'][1]:.2f}] | {s['xi_pot']:.3f} |")
    lines.append("\n## 2. 1-day VaR/ES: EVT vs Gaussian vs Historical (as % loss)\n")
    lines.append("_Full-sample descriptive estimates (whole 20y), NOT out-of-sample "
                 "forecasts. The OOS evaluation is in Section 3._\n")
    lines.append("| Index | level | EVT VaR | Gauss VaR | Hist VaR | EVT/Gauss | EVT ES |")
    lines.append("|---|---|---|---|---|---|---|")
    for n, s in stats_by_index.items():
        for p in LEVELS:
            lines.append(
                f"| {n} | {p:.3f} | {100*s[f'VaR_EVT_{p}']:.2f} | "
                f"{100*s[f'VaR_Gauss_{p}']:.2f} | {100*s[f'VaR_Hist_{p}']:.2f} | "
                f"{s[f'ratio_EVT_Gauss_{p}']:.2f}x | {100*s[f'ES_EVT_{p}']:.2f} |")
    def fmt_p(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    lines.append(f"\n## 3. OOS 99% VaR backtest (expanding window, {WINDOW}d warm-up)\n")
    lines.append("Target exception rate = 1.00%. These are hypothesis tests: p<0.05 means "
                 "we REJECT the model (evidence against), and p>=0.05 means we FAIL TO "
                 "REJECT it (consistent with, not proof of, correctness). Kupiec tests "
                 "unconditional coverage; Christoffersen tests independence (exceptions "
                 "arriving in clusters). All tuning (95th-pct POT threshold, EWMA "
                 "lambda=0.94, 1000d window, monthly refit) was set to conventional values (not tuned on the backtest).\n")
    lines.append("| Index | Model | Exc. rate % | Kupiec p | Christoffersen p | CondCov p | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for n, bt in backtests.items():
        for m in ["Gaussian", "Historical", "EVT", "CondGaussian", "CondEVT"]:
            k = bt[m]["kupiec"]
            ci = bt[m]["independence"]
            passed = (k["pvalue"] >= 0.05) and (ci["pvalue"] >= 0.05)
            verdict = "not rejected" if passed else "rejected"
            lines.append(f"| {n} | {m} | {100*k['rate']:.2f} | {fmt_p(k['pvalue'])} | "
                         f"{fmt_p(ci['pvalue'])} | {fmt_p(bt[m]['cond_coverage_p'])} | {verdict} |")
    (RES / "results.md").write_text("\n".join(lines))

    # ---- console headline ----
    s = stats_by_index[PRIMARY]
    print("\n================ HEADLINE (S&P 500, 20y daily) ================")
    print(f"Tail index alpha (Hill, k={HILL_FRAC*100:.0f}%): {s['alpha_hill']:.2f}  "
          f"CI[{s['alpha_ci'][0]:.2f},{s['alpha_ci'][1]:.2f}]  (Gaussian => alpha=inf)")
    print(f"GPD shape xi: {s['xi_pot']:.3f}  (>0 => heavy, power-law tail)")
    for p in LEVELS:
        print(f" {p:.3f} VaR: EVT {100*s[f'VaR_EVT_{p}']:.2f}%  vs  "
              f"Gaussian {100*s[f'VaR_Gauss_{p}']:.2f}%  "
              f"=> EVT is {s[f'ratio_EVT_Gauss_{p}']:.2f}x larger")
    bt = backtests[PRIMARY]
    print("\n 99% VaR OOS backtest exception rate (target 1.00%):")
    for m in ["Gaussian", "Historical", "EVT", "CondGaussian", "CondEVT"]:
        k = bt[m]["kupiec"]
        print(f"   {m:12s}: {100*k['rate']:.2f}%  Kupiec p={k['pvalue']:.3f}  "
              f"Christoffersen p={bt[m]['independence']['pvalue']:.3f}")
    print("===============================================================")
    print("\nWrote results.json, results.md and 4 figures.")


if __name__ == "__main__":
    main()
