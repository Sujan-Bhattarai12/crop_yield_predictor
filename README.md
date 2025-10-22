---
title: "Agricultural Yield Intelligence System"
subtitle: "Machine Learning-Powered Crop Yield Prediction & Forecasting Platform"
output: github_document
---

> **A data-driven platform combining predictive modeling, causal inference, and time series forecasting to enhance agricultural decision-making under climate uncertainty.**

---

## Problem & Business Value

Climate change threatens global food security. This system provides **predictive intelligence** for:

- **Farmers** – Optimize resource allocation & maximize yield 30 days in advance  
- **Policymakers** – Prioritize subsidies & disaster relief using predictive risk models  
- **Agribusinesses** – Forecast demand and optimize inventory  
- **Researchers** – Study causal impacts of climate and input variables on yield  

---

## Core Capabilities

| Capability | Description |
|-------------|--------------|
| **Predictive Modeling** | LightGBM model achieving **R² = 0.91**, MAE = 0.23 MT/HA |
| **Causal Analysis** | Quantified irrigation and fertilizer impact on yield (+15% per 10% irrigation access) |
| **Time Series Forecasting** | ARIMA model projecting yield trends through 2050 with confidence intervals |
| **Interactive Dashboard** | Built with Streamlit & Plotly for dynamic exploration of predictions and trends |

---

## Technical Highlights

- **Feature Engineering:** 47 engineered variables (interactions, encodings, temporal aggregations)  
- **Model Optimization:** GridSearchCV & cross-validation for hyperparameter tuning  
- **Statistical Validation:** Residual normality and significance testing (p < 0.05)  
- **Forecast Validation:** Walk-forward validation, AIC/BIC optimization, residual diagnostics  
- **Performance:** Yield prediction accuracy within ±0.5 MT/HA for 87% of cases  

---

## Key Insights

- **Top Predictors:** Temperature (28%), Precipitation (23%), Irrigation (18%)  
- **Causal Findings:** Fertilizer effectiveness plateaus beyond 150 kg/HA  
- **Forecast (2025–2050):**
  - Global yields expected to **decline ~8%** by 2050  
  - **Sub-Saharan Africa:** Highest risk (−15%)  
  - **Northern Regions:** +5% yield potential with adaptation  

---

## Architecture Overview

```text
Data Ingestion → Feature Engineering → ML & Forecasting Pipelines → Dashboard
