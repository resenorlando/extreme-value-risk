# Fat tails and extreme risk in global equity indices

An Extreme Value Theory engine for tail estimation and out-of-sample VaR/ES backtesting.

The short version: equity index daily losses follow a power law tail (tail index around 3), so a
Gaussian model badly understates the size of a 1-in-1,000-day loss (by roughly 2x in this
sample). Getting the tail shape right is not enough on its own. Out of sample, a static tail
model still bunches its exceptions together in crises. The only model whose 99% coverage is not
rejected across all five indices is a conditional EVT model that combines a volatility filter
with a fat-tailed model of the extremes. In other words, tail risk is about fat tails and
changing volatility together, not either one alone.

## Questions

1. How heavy are the tails, really? Estimate the tail index and the generalised Pareto shape for
 five global equity indices.
2. How badly does the Gaussian assumption understate extreme risk? Compare VaR and ES across
 Gaussian, Historical and EVT models.
3. Which model actually works out of sample? Backtest one-day 99% VaR over about 4,000 unseen
 days with the Kupiec and Christoffersen coverage tests.

## Data

Yahoo Finance daily adjusted closes (archived in `data/`, pulled with `src/fetch_data.py`).
Universe fixed in advance: S&P 500, Nasdaq Composite, Russell 2000, FTSE 100, Nikkei 225. About
5,030 daily log returns each, roughly 20 years to September 2026.

These are non-tradable indices, not execution vehicles; a real book would use the ETF (SPY, say)
and pay spread and financing. Adjusted closes are a total-return proxy. There is no intraday or
order-book data here, so this is a study of risk, and it makes no market-making or capacity
claim.

## Method

| Component | What it does | File |
|---|---|---|
| Hill estimator | Tail index from the k largest losses | `evt.py::hill_estimator` |
| Hill plot | Tail index vs k, which exposes the threshold-choice problem | `evt.py::hill_plot` |
| Peaks Over Threshold (POT) | Fits a Generalised Pareto Distribution to exceedances by MLE | `evt.py::pot_fit` |
| EVT VaR / ES | Closed-form tail quantile and expected shortfall from the POT fit | `evt.py::pot_var/pot_es` |
| EWMA volatility | RiskMetrics lambda 0.94, causal, used by the conditional models | `evt.py::ewma_vol` |
| Conditional EVT | McNeil-Frey style: EWMA filter, EVT on the residuals, rescale by today's vol | `var_backtest.py` |
| Coverage tests | Kupiec (unconditional coverage) and Christoffersen (clustering) | `var_backtest.py` |
| Stationary bootstrap | Politis-Romano block bootstrap for confidence intervals | `evt.py::bootstrap_ci` |

Everything is causal. The VaR used on day t is estimated only from data up to day t-1 (expanding
window, about four years of warm-up, monthly POT refits). The estimators are first checked on
synthetic data with known answers (`src/tests_evt.py`): Hill recovers 3.00 on Pareto(3), the POT
MLE recovers 0.30 on GPD(0.3), and the Kupiec test rejects at the right rate. Two more tests
shock future returns and confirm past forecasts do not move.

## Results

### The tails are power laws, not Gaussian
Left-tail index across the five indices (Hill, top 2%): about 2.8 to 3.3, the near-cubic tail
documented by Gopikrishnan et al. (1999). This is a regularity at that cutoff, not a universal
law, and Hill estimates move with the threshold. Every generalised Pareto shape is clearly
positive (0.13 to 0.27), so the tails are heavy by a second method too. Figures:
`figures/hill_plot_SP500.png`, `figures/gpd_tailfit_SP500.png`, `figures/tail_index_bars.png`.

### Gaussian VaR understates the extremes
One-day S&P 500 VaR, as a percentage loss (full-sample descriptive estimates):

| Confidence | EVT | Gaussian | ratio |
|---|---|---|---|
| 99% | 3.74% | 2.83% | 1.32x |
| 99.9% | 7.63% | 3.78% | 2.02x |

The gap widens the further into the tail you go, which is exactly where a risk model needs to
hold up. The pattern is consistent across all five indices (roughly 1.8x to 2.0x at 99.9%).

### Out of sample, only the conditional EVT model holds its coverage
One-day 99% VaR, about 4,000 out-of-sample days, S&P 500 (target exception rate 1.00%):

| Model | Exc. rate | Kupiec p | Christoffersen p | verdict |
|---|---|---|---|---|
| Gaussian (static) | 1.14% | 0.38 | <0.001 | rejected |
| Historical (static) | 0.40% | <0.001 | <0.001 | rejected |
| EVT (static) | 0.40% | <0.001 | <0.001 | rejected |
| Conditional Gaussian (RiskMetrics) | 2.26% | <0.001 | 0.07 | rejected |
| Conditional EVT (McNeil-Frey style) | 1.07% | 0.67 | <0.001 | not rejected on coverage |

You need both ingredients. Static EVT gets the shape right but, being unconditional, mistimes
risk and bunches its breaches in crises. Conditional Gaussian scales with volatility but its
thin tail still lets through too many breaches (2.26%). Conditional EVT is the only model whose
unconditional coverage is not rejected on any of the five indices (a test can only fail to
reject, never prove). It is not a clean win: it still fails the independence test on four of the
five (breaches stay somewhat clustered, S&P 500 p below 0.001), and it passes both tests on only
one index, the Russell 2000 (Kupiec 0.57, Christoffersen 0.10). Conditioning on volatility fixes
coverage but not the clustering, which points to a faster GARCH filter next.

## Limitations and next steps
- Conditional EVT still shows some breach clustering. That is the lag in the EWMA volatility
 estimate; a GJR-GARCH filter, which reacts faster and captures the leverage effect, is the
 obvious next step.
- Tail-index point estimates move with the threshold. I report a range and show the Hill plot
 rather than picking a flattering number. The GPD shape is the more stable statistic.
- Hill (about 2.9) and the GPD-implied index (about 5) differ because they weight the tail
 differently, and because unconditional estimates are inflated by volatility clustering.
- Non-tradable indices, no costs modelled. This measures risk; it is not a trading strategy.

## Running it
```bash
pip install numpy pandas scipy matplotlib
python src/fetch_data.py # pull about 20y of daily data into data/
python src/tests_evt.py # check the estimators on synthetic data
python src/run_study.py # full study into results/ and figures/
```

## References
Hill (1975); Pickands (1975); Kupiec (1995); Christoffersen (1998); McNeil and Frey (2000);
McNeil, Frey and Embrechts, Quantitative Risk Management; Politis and Romano (1994);
Gopikrishnan et al. (1999).
