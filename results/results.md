# Fat Tails & Extreme Risk - Results

## 1. Tail index (Hill) and GPD shape per index

| Index | N | alpha (Hill) | 95% CI | xi (GPD) |
|---|---|---|---|---|
| SP500 | 5031 | 2.94 | [2.32, 3.91] | 0.185 |
| NASDAQ | 5031 | 3.33 | [2.74, 4.41] | 0.143 |
| RUSSELL2000 | 5031 | 2.98 | [2.29, 4.17] | 0.254 |
| FTSE100 | 5051 | 3.05 | [2.34, 4.52] | 0.135 |
| NIKKEI225 | 4889 | 2.80 | [2.30, 4.01] | 0.268 |

## 2. 1-day VaR/ES: EVT vs Gaussian vs Historical (as % loss)

_Full-sample descriptive estimates (whole 20y), NOT out-of-sample forecasts. The OOS evaluation is in Section 3._

| Index | level | EVT VaR | Gauss VaR | Hist VaR | EVT/Gauss | EVT ES |
|---|---|---|---|---|---|---|
| SP500 | 0.990 | 3.74 | 2.83 | 3.57 | 1.32x | 5.40 |
| SP500 | 0.999 | 7.63 | 3.78 | 7.92 | 2.02x | 10.17 |
| NASDAQ | 0.990 | 4.07 | 3.21 | 4.08 | 1.27x | 5.54 |
| NASDAQ | 0.999 | 7.51 | 4.28 | 7.55 | 1.76x | 9.56 |
| RUSSELL2000 | 0.990 | 4.42 | 3.64 | 4.26 | 1.21x | 6.43 |
| RUSSELL2000 | 0.999 | 9.11 | 4.85 | 9.82 | 1.88x | 12.72 |
| FTSE100 | 0.990 | 3.36 | 2.58 | 3.25 | 1.30x | 4.70 |
| FTSE100 | 0.999 | 6.49 | 3.43 | 5.86 | 1.89x | 8.32 |
| NIKKEI225 | 0.990 | 4.21 | 3.45 | 4.13 | 1.22x | 6.23 |
| NIKKEI225 | 0.999 | 8.90 | 4.59 | 9.88 | 1.94x | 12.63 |

## 3. OOS 99% VaR backtest (expanding window, 1000d warm-up)

Target exception rate = 1.00%. These are hypothesis tests: p<0.05 means we REJECT the model (evidence against), and p>=0.05 means we FAIL TO REJECT it (consistent with, not proof of, correctness). Kupiec tests unconditional coverage; Christoffersen tests independence (exceptions arriving in clusters). All tuning (95th-pct POT threshold, EWMA lambda=0.94, 1000d window, monthly refit) was set to conventional values (not tuned on the backtest).

| Index | Model | Exc. rate % | Kupiec p | Christoffersen p | CondCov p | Verdict |
|---|---|---|---|---|---|---|
| SP500 | Gaussian | 1.14 | 0.378 | <0.001 | <0.001 | rejected |
| SP500 | Historical | 0.40 | <0.001 | 0.001 | <0.001 | rejected |
| SP500 | EVT | 0.40 | <0.001 | 0.001 | <0.001 | rejected |
| SP500 | CondGaussian | 2.26 | <0.001 | 0.074 | <0.001 | rejected |
| SP500 | CondEVT | 1.07 | 0.674 | <0.001 | 0.004 | rejected |
| NASDAQ | Gaussian | 1.66 | <0.001 | 0.030 | <0.001 | rejected |
| NASDAQ | Historical | 0.69 | 0.039 | 0.015 | 0.006 | rejected |
| NASDAQ | EVT | 0.67 | 0.025 | 0.013 | 0.004 | rejected |
| NASDAQ | CondGaussian | 2.28 | <0.001 | 0.231 | <0.001 | rejected |
| NASDAQ | CondEVT | 1.29 | 0.077 | 0.033 | 0.021 | rejected |
| RUSSELL2000 | Gaussian | 0.84 | 0.305 | 0.034 | 0.062 | rejected |
| RUSSELL2000 | Historical | 0.42 | <0.001 | 0.002 | <0.001 | rejected |
| RUSSELL2000 | EVT | 0.35 | <0.001 | 0.039 | <0.001 | rejected |
| RUSSELL2000 | CondGaussian | 2.11 | <0.001 | 0.399 | <0.001 | rejected |
| RUSSELL2000 | CondEVT | 1.09 | 0.565 | 0.096 | 0.212 | not rejected |
| FTSE100 | Gaussian | 0.91 | 0.574 | 0.004 | 0.014 | rejected |
| FTSE100 | Historical | 0.54 | 0.001 | 0.005 | <0.001 | rejected |
| FTSE100 | EVT | 0.47 | <0.001 | 0.003 | <0.001 | rejected |
| FTSE100 | CondGaussian | 2.27 | <0.001 | <0.001 | <0.001 | rejected |
| FTSE100 | CondEVT | 1.06 | 0.697 | <0.001 | 0.004 | rejected |
| NIKKEI225 | Gaussian | 1.11 | 0.515 | <0.001 | <0.001 | rejected |
| NIKKEI225 | Historical | 0.57 | 0.003 | 0.006 | <0.001 | rejected |
| NIKKEI225 | EVT | 0.64 | 0.017 | <0.001 | <0.001 | rejected |
| NIKKEI225 | CondGaussian | 2.13 | <0.001 | 0.041 | <0.001 | rejected |
| NIKKEI225 | CondEVT | 0.72 | 0.065 | <0.001 | <0.001 | rejected |