# Usage Guide

This guide explains how to use the Stock Return Analyzer for various analysis tasks, from basic running of the main script to advanced custom analyses.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Using StockReturnCalculator Class](#using-stockreturncalculator-class)
- [Custom Analysis Examples](#custom-analysis-examples)
- [Visualization Functions](#visualization-functions)
- [Working with Results](#working-with-results)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

## Basic Usage

### Running the Main Analysis

The primary analysis script (`main.py`) performs a complete analysis of three assets:

```bash
# Using uv (recommended)
uv run main.py

# Using python directly (if virtual environment is activated)
python main.py
```

**What happens when you run `main.py`:**
1. **Data Fetching**: Downloads historical price data for:
   - S&P 500 (^GSPC): 2015-10-01 to 2025-09-30
   - Apple (AAPL): 2015-10-01 to 2025-09-30
   - NVIDIA (NVDA): 2015-10-01 to 2025-09-30

2. **Metrics Calculation**: Computes for each asset:
   - Daily returns
   - Cumulative returns
   - Annualized Sharpe ratio
   - Maximum drawdown
   - Current market capitalization
   - Historical market capitalization (AAPL & NVDA only)

3. **Console Output**: Displays formatted results:
   ```
   Calculating metrics...
   Metrics calculated:
   S&P 500 - Cumulative return: 120.53%, Sharpe: 0.8521, Max Drawdown: 0.3392, Market Cap: None
   AAPL - Cumulative return: 580.31%, Sharpe: 1.1234, Max Drawdown: 0.4521, Market Cap: 2800000000000
   NVDA - Cumulative return: 12500.45%, Sharpe: 1.4523, Max Drawdown: 0.6789, Market Cap: 1200000000000
   ```

4. **Visualization**: Generates and displays a 4-chart subplot:
   - Top-left: Cumulative returns over time
   - Top-right: Sharpe ratios (bar chart)
   - Bottom-left: Maximum drawdowns (bar chart)
   - Bottom-right: Market capitalization evolution (line chart)

### Running the Sample Analysis

The `sample.py` file contains an earlier version analyzing different stocks:

```bash
uv run sample.py
```

This analyzes:
- S&P 500 (^GSPC)
- McDonald's (MCD)
- Coca-Cola (KO)

## Using StockReturnCalculator Class

### Importing the Class

```python
# From main.py
from main import StockReturnCalculator

# Or import directly if working within the same directory
from main import StockReturnCalculator, plot_metric, plot_cumulative_returns
```

### Basic Class Usage

```python
# Initialize calculator
calculator = StockReturnCalculator(
    ticker_symbol="AAPL",      # Stock symbol
    start_date="2020-01-01",   # Start date (YYYY-MM-DD)
    end_date="2025-01-01"      # End date (YYYY-MM-DD)
)

# Fetch price data (required before calculations)
calculator.fetch_price_data()

# Calculate daily returns
returns = calculator.calculate_returns()
print(f"Number of trading days: {len(returns)}")
print(f"First 5 returns: {returns[:5]}")

# Calculate Sharpe ratio (annualized)
sharpe_ratio = calculator.calculate_sharpe_ratio(risk_free_rate=0.02)  # 2% risk-free rate
print(f"Sharpe ratio: {sharpe_ratio:.4f}")

# Calculate maximum drawdown
max_drawdown = calculator.calculate_max_drawdown()
print(f"Maximum drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")

# Get current market capitalization
market_cap = calculator.get_market_cap()
if market_cap:
    print(f"Market cap: ${market_cap:,.0f}")
    print(f"Market cap (billions): ${market_cap/1e9:,.2f}B")

# Calculate historical market capitalization
market_cap_history = calculator.calculate_market_cap_history()
print(f"Historical market cap data points: {len(market_cap_history)}")
```

### Accessing Internal Data

```python
# Access the fetched price data
price_data = calculator.price_data  # Full pandas DataFrame
close_prices = calculator.close_prices  # Adjusted close prices Series
dates = close_prices.index  # DatetimeIndex of trading days

# Access calculated returns
returns = calculator.returns  # Daily returns list (after calculate_returns())
```

## Custom Analysis Examples

### Analyzing Different Stocks

```python
from main import StockReturnCalculator

# Analyze technology stocks
msft = StockReturnCalculator("MSFT", "2018-01-01", "2024-12-31")
googl = StockReturnCalculator("GOOGL", "2018-01-01", "2024-12-31")
amzn = StockReturnCalculator("AMZN", "2018-01-01", "2024-12-31")

for stock in [msft, googl, amzn]:
    stock.fetch_price_data()
    sharpe = stock.calculate_sharpe_ratio()
    max_dd = stock.calculate_max_drawdown()
    print(f"{stock.ticker_symbol}: Sharpe={sharpe:.3f}, MaxDD={max_dd:.3f}")
```

### Comparing Multiple Time Periods

```python
from main import StockReturnCalculator

# Compare pre-COVID, COVID, and post-COVID periods
periods = {
    "Pre-COVID": ("2017-01-01", "2019-12-31"),
    "COVID": ("2020-01-01", "2021-12-31"),
    "Post-COVID": ("2022-01-01", "2024-12-31")
}

results = {}
for period_name, (start, end) in periods.items():
    calculator = StockReturnCalculator("SPY", start, end)
    calculator.fetch_price_data()
    sharpe = calculator.calculate_sharpe_ratio()
    max_dd = calculator.calculate_max_drawdown()
    results[period_name] = {"sharpe": sharpe, "max_dd": max_dd}

for period, metrics in results.items():
    print(f"{period}: Sharpe={metrics['sharpe']:.3f}, MaxDD={metrics['max_dd']:.3f}")
```

### Batch Processing Multiple Stocks

```python
from main import StockReturnCalculator
import pandas as pd

# List of stocks to analyze
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ", "V"]

results = []
for ticker in stocks:
    try:
        calc = StockReturnCalculator(ticker, "2020-01-01", "2024-12-31")
        calc.fetch_price_data()

        results.append({
            "Ticker": ticker,
            "Sharpe": calc.calculate_sharpe_ratio(),
            "MaxDrawdown": calc.calculate_max_drawdown(),
            "MarketCap": calc.get_market_cap(),
            "CumulativeReturn": (calc.calculate_returns()[-1] - 1) * 100
        })
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)
print(results_df)
```

## Visualization Functions

### Using Built-in Plotting Functions

```python
from main import (
    plot_metric,
    plot_cumulative_returns,
    plot_sharpe_ratios,
    plot_max_drawdowns,
    plot_market_caps,
    plot_market_cap_history
)

# Example: Create custom cumulative returns plot
dates = aapl.close_prices.index  # Get dates from an existing calculator

cumulative_returns_dict = {
    "AAPL": aapl_cumulative,  # Assuming you have calculated these
    "MSFT": msft_cumulative,
    "GOOGL": googl_cumulative
}

# Plot using specialized function
plt = plot_cumulative_returns(dates, cumulative_returns_dict)
plt.savefig("cumulative_returns.png", dpi=300, bbox_inches="tight")
plt.show()

# Or use general plot_metric function
data_dict = {
    "Series 1": [1, 2, 3, 4, 5],
    "Series 2": [2, 3, 4, 5, 6]
}
plt = plot_metric(range(5), data_dict, "Custom Plot", "Y-axis Label", figsize=(10, 6))
plt.show()
```

### Creating Custom Visualizations

```python
import matplotlib.pyplot as plt
import numpy as np

# Example: Create a correlation heatmap of returns
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]
returns_data = {}

for ticker in stocks:
    calc = StockReturnCalculator(ticker, "2020-01-01", "2024-12-31")
    calc.fetch_price_data()
    returns_data[ticker] = calc.calculate_returns()

# Calculate correlation matrix (simplified)
corr_matrix = np.corrcoef([returns_data[ticker] for ticker in stocks])

# Plot heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(stocks)))
ax.set_yticks(range(len(stocks)))
ax.set_xticklabels(stocks)
ax.set_yticklabels(stocks)
plt.colorbar(im)
plt.title("Return Correlations (2020-2024)")
plt.show()
```

## Working with Results

### Exporting Data

```python
import pandas as pd
import json

# Export metrics to CSV
metrics_df = pd.DataFrame([
    {"Ticker": "AAPL", "Sharpe": 1.12, "MaxDD": 0.45, "MarketCap": 2.8e12},
    {"Ticker": "NVDA", "Sharpe": 1.45, "MaxDD": 0.68, "MarketCap": 1.2e12}
])
metrics_df.to_csv("stock_metrics.csv", index=False)

# Export time series data
aapl = StockReturnCalculator("AAPL", "2020-01-01", "2024-12-31")
aapl.fetch_price_data()
returns = aapl.calculate_returns()
dates = aapl.close_prices.index[1:]  # Align with returns (one less date)

returns_series = pd.Series(returns, index=dates, name="Daily Returns")
returns_series.to_csv("aapl_daily_returns.csv")

# Export to JSON
results = {
    "AAPL": {
        "sharpe_ratio": 1.1234,
        "max_drawdown": 0.4521,
        "market_cap": 2800000000000
    }
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Statistical Analysis

```python
import numpy as np
from scipy import stats

# Basic statistics of returns
returns_array = np.array(aapl_daily_returns)  # Assuming you have returns

print(f"Mean daily return: {returns_array.mean():.4%}")
print(f"Std deviation: {returns_array.std():.4%}")
print(f"Skewness: {stats.skew(returns_array):.3f}")
print(f"Kurtosis: {stats.kurtosis(returns_array):.3f}")
print(f"Minimum return: {returns_array.min():.4%}")
print(f"Maximum return: {returns_array.max():.4%}")

# Value at Risk (VaR) calculation
var_95 = np.percentile(returns_array, 5)  # 95% VaR
print(f"95% VaR: {var_95:.4%}")
```

## Advanced Topics

### Handling Missing Data

```python
# The calculator automatically handles some missing data
calculator = StockReturnCalculator("BRK.A", "2020-01-01", "2024-12-31")

try:
    calculator.fetch_price_data()
except Exception as e:
    print(f"Error fetching data: {e}")
    # Implement fallback or alternative data source

# Check if data was fetched successfully
if calculator.price_data is not None:
    print("Data fetched successfully")
else:
    print("No data available")
```

### Custom Risk-Free Rate

```python
# Use current 10-year Treasury yield as risk-free rate
risk_free_rate = 0.0425  # 4.25%
sharpe_ratio = calculator.calculate_sharpe_ratio(risk_free_rate=risk_free_rate)

# Or fetch dynamically (example)
import yfinance as yf
try:
    treasury = yf.download("^TNX", period="1d")
    risk_free_rate = treasury["Close"].iloc[-1] / 100  # Convert percentage to decimal
    sharpe_ratio = calculator.calculate_sharpe_ratio(risk_free_rate=risk_free_rate)
except:
    # Fallback to default
    sharpe_ratio = calculator.calculate_sharpe_ratio()
```

### Performance Optimization

```python
# Cache data to avoid repeated downloads
import pickle
import os

CACHE_FILE = "stock_data_cache.pkl"

def get_cached_calculator(ticker, start, end):
    cache_key = f"{ticker}_{start}_{end}"

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
            if cache_key in cache:
                print(f"Using cached data for {ticker}")
                return cache[cache_key]

    # Not in cache, create new
    calculator = StockReturnCalculator(ticker, start, end)
    calculator.fetch_price_data()

    # Update cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    cache[cache_key] = calculator
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    return calculator
```

## Troubleshooting

### Common Issues

1. **Data Fetching Errors**:
   - Check internet connection
   - Verify ticker symbol is correct
   - Ensure date range is valid (not future dates)
   - Yahoo Finance may have rate limits

2. **Calculation Errors**:
   - Ensure `fetch_price_data()` is called before calculations
   - Check for missing or NaN values in price data
   - Verify date alignment for returns calculation

3. **Visualization Issues**:
   - Ensure matplotlib backend is properly configured
   - Check figure size parameters for your display
   - Close previous plots before creating new ones

### Debugging Tips

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data quality
print(f"Price data shape: {calculator.price_data.shape}")
print(f"Missing values: {calculator.price_data.isna().sum().sum()}")
print(f"Date range: {calculator.price_data.index[0]} to {calculator.price_data.index[-1]}")
```

## Next Steps

- Explore the [API Reference](api-reference.md) for detailed class documentation
- Check [Examples](examples.md) for more analysis patterns
- Modify `main.py` to analyze your favorite stocks
- Extend the `StockReturnCalculator` class with new metrics
- Integrate with other financial libraries like `backtrader` or `zipline`

Remember to always verify calculations and understand the limitations of historical data analysis. Past performance does not guarantee future results.