# Data_Science_Project
# Causal Inference in UK Stock Market Trading

This project investigates the true causal drivers of stock price movements in the UK market by integrating macroeconomic indicators and technical analysis. It leverages causal inference techniques (e.g., Granger causality, DoWhy) and compares models built on causally validated features against those based solely on correlation.

---

## Project Objectives

- Identify macroeconomic and technical indicators with causal influence on UK stock prices.
- Compare predictive performance of models using causally filtered features vs. all features.
- Evaluate the potential for causal inference to improve quantitative trading strategies.

---

## Datasets Used

| Dataset | Source | Frequency |
|--------|--------|-----------|
| FTSE 100 Stocks (e.g., AZN.L, HSBA.L) | [Yahoo Finance](https://finance.yahoo.com/) via `yfinance` | Daily |
| UK Consumer Price Index (CPI) | [ONS](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23) | Monthly |
| Bank of England Base Rate | [BoE](https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp) | Irregular |
| GBP/USD Exchange Rate | [Yahoo Finance](https://finance.yahoo.com/) via `yfinance` | Daily |
| Unemployment Rate (UK) | [ONS](https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms) | Monthly |
