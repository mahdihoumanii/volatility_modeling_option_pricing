# Volatility Modeling & Option Pricing

End-to-end quantitative finance project implementing classical **volatility forecasting** models and **European option pricing**.  
The pipeline downloads equity data (SPY by default), computes log returns, estimates time-varying volatility using rolling, EWMA, and GARCH(1,1) models, evaluates forecasts **out-of-sample**, and applies volatility estimates to option pricing via **Black–Scholes** and **Monte Carlo simulation with confidence intervals**.

📄 **Full mathematical report (PDF):** `reports/main.pdf`

---

## Features
- **Data ingestion** via `yfinance` with reproducible preprocessing to adjusted close prices and log returns.
- **Volatility estimation**:
  - Rolling historical volatility  
  - EWMA variance (RiskMetrics-style)  
  - GARCH(1,1) fitted via maximum likelihood (`arch`)
- **Option pricing**:
  - Black–Scholes closed form (calls/puts)
  - Monte Carlo pricing under GBM with convergence diagnostics and 95% confidence intervals
- **Evaluation**:
  - Chronological walk-forward testing (no look-ahead bias)
  - Metrics: MSE (variance), MAE (volatility), QLIKE
  - Smoothed realized volatility using 5-day and 21-day windows
- **Reproducibility**:
  - Modular source code
  - Unit tests (`pytest`)
  - Notebooks for transparent, end-to-end analysis

---

## Quickstart
```bash
# create environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# download example dataset (SPY)
python -m src.data_loader --ticker SPY --start 2015-01-01 --end 2024-01-01
```

---

## Methodology

### 1. Data & returns
Daily adjusted close prices are converted to **log returns**:

```
r_t = ln(P_t / P_{t-1})
```

Volatility is annualized using:

```
annual_vol = daily_vol * sqrt(252)
```

---

### 2. Volatility models

**Historical (rolling volatility)**  
Standard deviation of returns over a fixed window `w` (e.g. 20 days).

**EWMA volatility**  
Recursive variance update:

```
sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
```

with `lambda = 0.94`.

**GARCH(1,1)**  
Conditional variance model:

```
sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
```

Parameters are estimated via maximum likelihood.

---

### 3. Evaluation
Forecasts are generated using **walk-forward (out-of-sample) testing** and evaluated against **windowed realized volatility**:

```
RV_t(w) = sqrt(252 / w * sum_{i=0..w-1} r_{t-i}^2)
```

Windows used:
- 5 days (weekly)
- 21 days (monthly)

Metrics:
- Mean Squared Error (MSE) on variance
- Mean Absolute Error (MAE) on volatility
- QLIKE loss

---

### 4. Option pricing
Forecasted volatility is used as input to:

**Black–Scholes pricing**  
Closed-form pricing for European calls and puts.

**Monte Carlo pricing under GBM**  

Price dynamics:

```
S_{t+dt} = S_t * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
```

with `Z ~ N(0,1)`.

Monte Carlo estimates include **95% confidence intervals**, and convergence to the Black–Scholes price is demonstrated with increasing number of paths.

---

## Running notebooks
The notebooks in `notebooks/` provide a complete workflow:
1. Data ingestion and sanity checks  
2. Volatility modeling and visualization  
3. Out-of-sample forecast evaluation  
4. Option pricing and Monte Carlo convergence  

Figures are saved to `reports/figures/`, and results are summarized in `reports/summary.md`.

---

## Tests
```bash
pytest -q
```

All core functionality is covered by unit tests.

---

## Results

### Forecasted vs. realized volatility (5D / 21D realized volatility)
![Forecast vs Realized](reports/figures/forecast_vs_realized.png)

### Monte Carlo option pricing with 95% confidence intervals
![MC CI Convergence](reports/figures/mc_ci_convergence.png)

---

## Notes & limitations
- Daily data is used; realized volatility is a proxy for latent volatility
- Option pricing assumes lognormal dynamics and constant volatility
- Transaction costs and market microstructure effects are not modeled