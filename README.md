# Volatility Modeling & Option Pricing

End-to-end quantitative finance project implementing classical **volatility forecasting** models and **European option pricing**.  
The pipeline downloads equity data (SPY by default), computes log returns, estimates time-varying volatility using rolling, EWMA, and GARCH(1,1) models, evaluates forecasts **out-of-sample**, and applies volatility estimates to option pricing via **Black–Scholes** and **Monte Carlo simulation with confidence intervals**.

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
1. **Data & returns**  
   Daily adjusted close prices are converted to log returns  
   $begin:math:display$
   r\_t \= \\ln\\left\(\\frac\{P\_t\}\{P\_\{t\-1\}\}\\right\)\,
   $end:math:display$
   with volatility annualized using $begin:math:text$\\sqrt\{252\}$end:math:text$.

2. **Volatility models**
   - **Historical (rolling)**: standard deviation over a fixed window $begin:math:text$w$end:math:text$.
   - **EWMA**:  
     $begin:math:display$
     \\sigma\_t\^2 \= \\lambda \\sigma\_\{t\-1\}\^2 \+ \(1\-\\lambda\) r\_\{t\-1\}\^2\.
     $end:math:display$
   - **GARCH(1,1)**:  
     $begin:math:display$
     \\sigma\_t\^2 \= \\omega \+ \\alpha r\_\{t\-1\}\^2 \+ \\beta \\sigma\_\{t\-1\}\^2\,
     $end:math:display$
     estimated via maximum likelihood.

3. **Evaluation**  
   Forecasts are generated using walk-forward (out-of-sample) testing and evaluated against **windowed realized volatility** (5-day and 21-day) using MSE, MAE, and QLIKE.

4. **Option pricing**  
   Forecasted volatility is used as input to:
   - **Black–Scholes** pricing
   - **Monte Carlo** pricing under GBM, with convergence analysis and 95% confidence intervals demonstrating  
     $begin:math:text$ \\mathcal\{O\}\(1\/\\sqrt\{N\}\) $end:math:text$ behavior.

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