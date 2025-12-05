# Examples

This document provides practical examples of using the Stock Return Analyzer for various financial analysis tasks.

## Table of Contents
- [Basic Examples](#basic-examples)
- [Comparative Analysis](#comparative-analysis)
- [Portfolio Analysis](#portfolio-analysis)
- [Risk Assessment](#risk-assessment)
- [Data Export & Reporting](#data-export--reporting)
- [Custom Visualizations](#custom-visualizations)
- [Advanced Scenarios](#advanced-scenarios)

## Basic Examples

### Example 1: Single Stock Analysis

```python
from main import StockReturnCalculator
import matplotlib.pyplot as plt

# Analyze Tesla stock
tsla = StockReturnCalculator("TSLA", "2020-01-01", "2024-12-31")
tsla.fetch_price_data()

# Calculate metrics
returns = tsla.calculate_returns()
sharpe = tsla.calculate_sharpe_ratio()
max_dd = tsla.calculate_max_drawdown()
market_cap = tsla.get_market_cap()

print(f"Tesla Analysis (2020-2024)")
print(f"=========================")
print(f"Trading days: {len(returns)}")
print(f"Total return: {(returns[-1] - 1) * 100:.2f}%")
print(f"Sharpe ratio: {sharpe:.3f}")
print(f"Max drawdown: {max_dd:.3f} ({max_dd*100:.1f}%)")
if market_cap:
    print(f"Market cap: ${market_cap/1e9:.1f}B")

# Plot cumulative returns
cumulative = [1.0]
for ret in returns:
    cumulative.append(cumulative[-1] * (1 + ret))

plt.figure(figsize=(10, 6))
plt.plot(tsla.close_prices.index[1:], cumulative[1:])
plt.title("TSLA Cumulative Returns (2020-2024)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Example 2: Comparing Two Stocks

```python
from main import StockReturnCalculator
import matplotlib.pyplot as plt

# Analyze two competitors
amd = StockReturnCalculator("AMD", "2020-01-01", "2024-12-31")
intc = StockReturnCalculator("INTC", "2020-01-01", "2024-12-31")

for stock in [amd, intc]:
    stock.fetch_price_data()

# Calculate cumulative returns
amd_returns = amd.calculate_returns()
intc_returns = intc.calculate_returns()

amd_cumulative = [1.0]
intc_cumulative = [1.0]

for ret in amd_returns:
    amd_cumulative.append(amd_cumulative[-1] * (1 + ret))
for ret in intc_returns:
    intc_cumulative.append(intc_cumulative[-1] * (1 + ret))

# Plot comparison
plt.figure(figsize=(12, 8))
plt.plot(amd.close_prices.index[1:], amd_cumulative[1:], label="AMD", linewidth=2)
plt.plot(intc.close_prices.index[1:], intc_cumulative[1:], label="Intel", linewidth=2)
plt.title("AMD vs Intel Cumulative Returns (2020-2024)", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Print comparison table
print(f"{'Metric':<20} {'AMD':<15} {'Intel':<15}")
print(f"{'-'*50}")
print(f"{'Final Return':<20} {amd_cumulative[-1]-1:>14.2%} {intc_cumulative[-1]-1:>14.2%}")
print(f"{'Sharpe Ratio':<20} {amd.calculate_sharpe_ratio():>14.3f} {intc.calculate_sharpe_ratio():>14.3f}")
print(f"{'Max Drawdown':<20} {amd.calculate_max_drawdown():>14.2%} {intc.calculate_max_drawdown():>14.2%}")
```

### Example 3: Sector Analysis

```python
from main import StockReturnCalculator
import pandas as pd

# Define stocks by sector
sectors = {
    "Technology": ["AAPL", "MSFT", "GOOGL"],
    "Finance": ["JPM", "BAC", "WFC"],
    "Healthcare": ["JNJ", "PFE", "MRK"],
    "Consumer": ["PG", "KO", "PEP"]
}

results = []

for sector, stocks in sectors.items():
    sector_returns = []

    for ticker in stocks:
        try:
            calc = StockReturnCalculator(ticker, "2020-01-01", "2024-12-31")
            calc.fetch_price_data()
            returns = calc.calculate_returns()
            sector_returns.extend(returns)

            results.append({
                "Sector": sector,
                "Ticker": ticker,
                "Sharpe": calc.calculate_sharpe_ratio(),
                "MaxDD": calc.calculate_max_drawdown(),
                "MarketCap": calc.get_market_cap()
            })
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

    # Calculate sector-level statistics
    if sector_returns:
        import numpy as np
        avg_return = np.mean(sector_returns)
        std_return = np.std(sector_returns)
        print(f"{sector}: Avg Return={avg_return:.4%}, Std={std_return:.4%}")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Display summary
print("\nSector Analysis Summary:")
print("="*80)
print(results_df.groupby("Sector").agg({
    "Sharpe": "mean",
    "MaxDD": "mean",
    "MarketCap": "sum"
}).round(3))
```

## Comparative Analysis

### Example 4: Benchmark Comparison

```python
from main import StockReturnCalculator
import numpy as np

# Analyze a stock against its benchmark
stock_symbol = "META"
benchmark_symbol = "^GSPC"  # S&P 500

stock = StockReturnCalculator(stock_symbol, "2020-01-01", "2024-12-31")
benchmark = StockReturnCalculator(benchmark_symbol, "2020-01-01", "2024-12-31")

stock.fetch_price_data()
benchmark.fetch_price_data()

# Calculate returns
stock_returns = np.array(stock.calculate_returns())
benchmark_returns = np.array(benchmark.calculate_returns())

# Calculate alpha and beta (simplified)
covariance = np.cov(stock_returns, benchmark_returns)[0, 1]
benchmark_variance = np.var(benchmark_returns)

beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
alpha = np.mean(stock_returns) - beta * np.mean(benchmark_returns)

# Annualize alpha
annualized_alpha = alpha * 252

print(f"{stock_symbol} vs {benchmark_symbol} Analysis")
print(f"="*40)
print(f"Beta: {beta:.3f}")
print(f"Alpha (annualized): {annualized_alpha:.4f}")
print(f"Correlation: {np.corrcoef(stock_returns, benchmark_returns)[0,1]:.3f}")

# Calculate information ratio
active_returns = stock_returns - benchmark_returns
tracking_error = np.std(active_returns)
information_ratio = np.mean(active_returns) / tracking_error if tracking_error != 0 else 0
annualized_ir = information_ratio * np.sqrt(252)

print(f"Information Ratio (annualized): {annualized_ir:.3f}")
```

### Example 5: Rolling Metrics Analysis

```python
from main import StockReturnCalculator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Analyze rolling Sharpe ratio
symbol = "NVDA"
calculator = StockReturnCalculator(symbol, "2015-01-01", "2024-12-31")
calculator.fetch_price_data()
returns = calculator.calculate_returns()
dates = calculator.close_prices.index[1:]  # Align with returns

# Convert to pandas Series for rolling operations
returns_series = pd.Series(returns, index=dates)

# Calculate 252-day (1 year) rolling Sharpe ratio
rolling_sharpe = returns_series.rolling(window=252).apply(
    lambda x: (np.mean(x) / np.std(x)) * np.sqrt(252) if np.std(x) != 0 else 0
)

# Calculate 252-day rolling maximum drawdown
rolling_max_dd = returns_series.rolling(window=252).apply(
    lambda x: calculate_rolling_drawdown(x)  # Custom function needed
)

def calculate_rolling_drawdown(returns):
    """Calculate maximum drawdown for a return series"""
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown) if len(drawdown) > 0 else 0

# Plot rolling metrics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Rolling Sharpe
ax1.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.set_title(f"{symbol} - 252-Day Rolling Sharpe Ratio", fontsize=14)
ax1.set_ylabel("Sharpe Ratio")
ax1.grid(True, alpha=0.3)

# Rolling Max Drawdown
ax2.plot(rolling_max_dd.index, rolling_max_dd.values * 100, linewidth=1.5, color='red')
ax2.set_title(f"{symbol} - 252-Day Rolling Maximum Drawdown", fontsize=14)
ax2.set_ylabel("Max Drawdown (%)")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Portfolio Analysis

### Example 6: Simple Portfolio Construction

```python
from main import StockReturnCalculator
import numpy as np
import pandas as pd

# Define portfolio weights
portfolio = {
    "AAPL": 0.30,  # 30%
    "MSFT": 0.25,  # 25%
    "GOOGL": 0.20, # 20%
    "AMZN": 0.15,  # 15%
    "NVDA": 0.10   # 10%
}

# Fetch data for all stocks
calculators = {}
returns_data = {}

for ticker in portfolio.keys():
    calc = StockReturnCalculator(ticker, "2020-01-01", "2024-12-31")
    calc.fetch_price_data()
    calculators[ticker] = calc
    returns_data[ticker] = calc.calculate_returns()

# Align return dates (assuming same date range)
# Convert to DataFrame for easier manipulation
returns_df = pd.DataFrame(returns_data)

# Calculate portfolio returns (weighted average)
portfolio_returns = returns_df.dot(pd.Series(portfolio))

# Calculate portfolio metrics
portfolio_mean_return = portfolio_returns.mean()
portfolio_std_return = portfolio_returns.std()
portfolio_sharpe = (portfolio_mean_return / portfolio_std_return) * np.sqrt(252) if portfolio_std_return != 0 else 0

# Calculate cumulative portfolio return
portfolio_cumulative = np.cumprod(1 + portfolio_returns)

print("Portfolio Analysis")
print("="*50)
print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.3f}")
print(f"Portfolio Annualized Return: {portfolio_mean_return * 252:.2%}")
print(f"Portfolio Volatility (annualized): {portfolio_std_return * np.sqrt(252):.2%}")
print(f"Final Portfolio Value: ${portfolio_cumulative.iloc[-1]:.2f} (starting from $1)")

# Compare to equal-weighted portfolio
equal_weights = {ticker: 1/len(portfolio) for ticker in portfolio.keys()}
equal_returns = returns_df.dot(pd.Series(equal_weights))
equal_sharpe = (equal_returns.mean() / equal_returns.std()) * np.sqrt(252) if equal_returns.std() != 0 else 0

print(f"\nEqual-weighted Portfolio Sharpe: {equal_sharpe:.3f}")
print(f"Improvement: {(portfolio_sharpe - equal_sharpe) / equal_sharpe * 100:.1f}%")
```

### Example 7: Efficient Frontier Simulation

```python
from main import StockReturnCalculator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define stocks for portfolio optimization
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B", "JPM", "JNJ", "V"]

# Fetch returns data
returns_data = {}
for ticker in stocks:
    calc = StockReturnCalculator(ticker, "2018-01-01", "2023-12-31")
    calc.fetch_price_data()
    returns_data[ticker] = calc.calculate_returns()

# Convert to DataFrame
returns_df = pd.DataFrame(returns_data)

# Calculate mean returns and covariance matrix
mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

# Generate random portfolios
np.random.seed(42)
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    # Calculate portfolio metrics
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_sharpe = portfolio_return / portfolio_volatility * np.sqrt(252) if portfolio_volatility != 0 else 0

    # Store results
    results[0, i] = portfolio_return * 252  # Annualized return
    results[1, i] = portfolio_volatility * np.sqrt(252)  # Annualized volatility
    results[2, i] = portfolio_sharpe
    weights_record.append(weights)

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])

# Find optimal portfolios
max_sharpe_idx = results_df['Sharpe'].idxmax()
min_vol_idx = results_df['Volatility'].idxmin()

print("Portfolio Optimization Results")
print("="*60)
print(f"Maximum Sharpe Ratio Portfolio:")
print(f"  Sharpe Ratio: {results_df.loc[max_sharpe_idx, 'Sharpe']:.3f}")
print(f"  Annual Return: {results_df.loc[max_sharpe_idx, 'Return']:.2%}")
print(f"  Annual Volatility: {results_df.loc[max_sharpe_idx, 'Volatility']:.2%}")

print(f"\nMinimum Volatility Portfolio:")
print(f"  Sharpe Ratio: {results_df.loc[min_vol_idx, 'Sharpe']:.3f}")
print(f"  Annual Return: {results_df.loc[min_vol_idx, 'Return']:.2%}")
print(f"  Annual Volatility: {results_df.loc[min_vol_idx, 'Volatility']:.2%}")

# Plot efficient frontier
plt.figure(figsize=(12, 8))
plt.scatter(results_df['Volatility'], results_df['Return'],
            c=results_df['Sharpe'], cmap='viridis', alpha=0.5, s=10)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(results_df.loc[max_sharpe_idx, 'Volatility'],
            results_df.loc[max_sharpe_idx, 'Return'],
            marker='*', color='red', s=500, label='Max Sharpe')
plt.scatter(results_df.loc[min_vol_idx, 'Volatility'],
            results_df.loc[min_vol_idx, 'Return'],
            marker='*', color='green', s=500, label='Min Volatility')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.title('Efficient Frontier Simulation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Risk Assessment

### Example 8: Value at Risk (VaR) Calculation

```python
from main import StockReturnCalculator
import numpy as np
import matplotlib.pyplot as plt

# Analyze a stock for risk assessment
symbol = "TSLA"
calculator = StockReturnCalculator(symbol, "2020-01-01", "2024-12-31")
calculator.fetch_price_data()
returns = np.array(calculator.calculate_returns())

# Calculate VaR using historical method
confidence_levels = [0.90, 0.95, 0.99]
var_results = {}

for cl in confidence_levels:
    var = np.percentile(returns, (1 - cl) * 100)
    var_results[cl] = var

# Calculate Expected Shortfall (CVaR)
cvar_results = {}
for cl in confidence_levels:
    var = var_results[cl]
    tail_returns = returns[returns <= var]
    cvar = tail_returns.mean() if len(tail_returns) > 0 else var
    cvar_results[cl] = cvar

print(f"Risk Metrics for {symbol}")
print("="*50)
for cl in confidence_levels:
    print(f"{cl*100:.0f}% VaR:  {var_results[cl]:.4%}")
    print(f"{cl*100:.0f}% CVaR: {cvar_results[cl]:.4%}")
    print("-"*30)

# Plot return distribution with VaR markers
plt.figure(figsize=(12, 6))

# Histogram of returns
plt.hist(returns * 100, bins=50, alpha=0.7, density=True, label='Return Distribution')

# Add VaR lines
colors = ['red', 'orange', 'yellow']
for (cl, var), color in zip(var_results.items(), colors):
    plt.axvline(x=var * 100, color=color, linestyle='--',
                linewidth=2, label=f'{cl*100:.0f}% VaR = {var:.2%}')

plt.xlabel('Daily Return (%)')
plt.ylabel('Density')
plt.title(f'{symbol} - Return Distribution with Value at Risk')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Example 9: Stress Testing

```python
from main import StockReturnCalculator
import pandas as pd

# Define stress periods (historical crises)
stress_periods = {
    "COVID Crash": ("2020-02-19", "2020-03-23"),
    "2018 Q4 Selloff": ("2018-10-01", "2018-12-24"),
    "2015 China Slowdown": ("2015-08-01", "2015-09-30"),
    "2022 Inflation": ("2022-01-01", "2022-10-01")
}

# Analyze a stock during stress periods
symbol = "SPY"  # S&P 500 ETF
stress_results = []

for period_name, (start, end) in stress_periods.items():
    try:
        calc = StockReturnCalculator(symbol, start, end)
        calc.fetch_price_data()

        if len(calc.close_prices) > 10:  # Ensure enough data
            returns = calc.calculate_returns()
            max_dd = calc.calculate_max_drawdown()
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            stress_results.append({
                "Period": period_name,
                "Start": start,
                "End": end,
                "Trading Days": len(returns),
                "Total Return": (calc.close_prices.iloc[-1] / calc.close_prices.iloc[0] - 1) * 100,
                "Max Drawdown": max_dd * 100,
                "Volatility": volatility * 100
            })
    except Exception as e:
        print(f"Error analyzing {period_name}: {e}")

# Display results
stress_df = pd.DataFrame(stress_results)
print("Stress Test Results")
print("="*80)
print(stress_df.to_string(index=False))

# Find worst periods
if not stress_df.empty:
    worst_drawdown = stress_df.loc[stress_df['Max Drawdown'].idxmax()]
    worst_return = stress_df.loc[stress_df['Total Return'].idxmin()]

    print(f"\nWorst Drawdown: {worst_drawdown['Period']} ({worst_drawdown['Max Drawdown']:.1f}%)")
    print(f"Worst Return: {worst_return['Period']} ({worst_return['Total Return']:.1f}%)")
```

## Data Export & Reporting

### Example 10: Comprehensive Report Generation

```python
from main import StockReturnCalculator
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def generate_stock_report(ticker, start_date, end_date, output_dir="reports"):
    """Generate a comprehensive stock analysis report"""

    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Initialize calculator
    calculator = StockReturnCalculator(ticker, start_date, end_date)
    calculator.fetch_price_data()

    # Calculate all metrics
    returns = calculator.calculate_returns()
    sharpe = calculator.calculate_sharpe_ratio()
    max_dd = calculator.calculate_max_drawdown()
    market_cap = calculator.get_market_cap()
    market_cap_history = calculator.calculate_market_cap_history()

    # Calculate additional metrics
    cumulative_returns = [1.0]
    for ret in returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + ret))

    # Create summary DataFrame
    summary_data = {
        "Metric": [
            "Ticker Symbol",
            "Analysis Period",
            "Trading Days",
            "Total Return",
            "Annualized Return",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Current Market Cap",
            "Average Daily Return",
            "Daily Return Std Dev"
        ],
        "Value": [
            ticker,
            f"{start_date} to {end_date}",
            len(returns),
            f"{(cumulative_returns[-1] - 1) * 100:.2f}%",
            f"{np.mean(returns) * 252 * 100:.2f}%",
            f"{sharpe:.3f}",
            f"{max_dd * 100:.2f}%",
            f"${market_cap/1e9:.2f}B" if market_cap else "N/A",
            f"{np.mean(returns) * 100:.4f}%",
            f"{np.std(returns) * 100:.4f}%"
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # Create time series DataFrame
    dates = calculator.close_prices.index[1:]  # Align with returns
    ts_data = {
        "Date": dates,
        "Close Price": calculator.close_prices.values[1:],
        "Daily Return": returns,
        "Cumulative Return": cumulative_returns[1:]
    }

    if not market_cap_history.empty:
        ts_data["Market Cap"] = market_cap_history.values

    ts_df = pd.DataFrame(ts_data)

    # Export to CSV
    summary_file = os.path.join(output_dir, f"{ticker}_summary_{datetime.now().strftime('%Y%m%d')}.csv")
    ts_file = os.path.join(output_dir, f"{ticker}_timeseries_{datetime.now().strftime('%Y%m%d')}.csv")

    summary_df.to_csv(summary_file, index=False)
    ts_df.to_csv(ts_file, index=False)

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Price chart
    axes[0, 0].plot(calculator.close_prices.index, calculator.close_prices.values, linewidth=2)
    axes[0, 0].set_title(f"{ticker} - Price Chart", fontsize=14)
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative returns
    axes[0, 1].plot(dates, cumulative_returns[1:], linewidth=2, color='green')
    axes[0, 1].set_title(f"{ticker} - Cumulative Returns", fontsize=14)
    axes[0, 1].set_ylabel("Cumulative Return")
    axes[0, 1].grid(True, alpha=0.3)

    # Daily returns distribution
    axes[1, 0].hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title(f"{ticker} - Daily Return Distribution", fontsize=14)
    axes[1, 0].set_xlabel("Daily Return (%)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # Market cap evolution (if available)
    if not market_cap_history.empty:
        axes[1, 1].plot(market_cap_history.index, market_cap_history.values / 1e9, linewidth=2, color='purple')
        axes[1, 1].set_title(f"{ticker} - Market Capitalization", fontsize=14)
        axes[1, 1].set_ylabel("Market Cap (Billions $)")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "Market Cap Data\nNot Available",
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title(f"{ticker} - Market Capitalization", fontsize=14)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"{ticker}_charts_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Generate markdown report
    report_file = os.path.join(output_dir, f"{ticker}_report_{datetime.now().strftime('%Y%m%d')}.md")
    with open(report_file, 'w') as f:
        f.write(f"# {ticker} Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Files Generated\n\n")
        f.write(f"- `{os.path.basename(summary_file)}`: Summary metrics\n")
        f.write(f"- `{os.path.basename(ts_file)}`: Time series data\n")
        f.write(f"- `{os.path.basename(plot_file)}`: Analysis charts\n")

    print(f"Report generated for {ticker}")
    print(f"  Summary: {summary_file}")
    print(f"  Time series: {ts_file}")
    print(f"  Charts: {plot_file}")
    print(f"  Report: {report_file}")

    return summary_df, ts_df

# Generate report for Apple
generate_stock_report("AAPL", "2020-01-01", "2024-12-31", "reports")
```

## Custom Visualizations

### Example 11: Interactive Dashboard (Matplotlib)

```python
from main import StockReturnCalculator
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

# Create interactive comparison dashboard
stocks = ["AAPL", "MSFT", "GOOGL"]
calculators = {}

# Fetch data
for ticker in stocks:
    calc = StockReturnCalculator(ticker, "2020-01-01", "2024-12-31")
    calc.fetch_price_data()
    calculators[ticker] = calc

# Prepare data for plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.95)

# Initial plot with first stock
current_stock = stocks[0]
calc = calculators[current_stock]
returns = calc.calculate_returns()
dates = calc.close_prices.index[1:]

# Plot 1: Price chart
axes[0, 0].plot(calc.close_prices.index, calc.close_prices.values, linewidth=2)
axes[0, 0].set_title(f"{current_stock} - Price Chart")
axes[0, 0].set_ylabel("Price ($)")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Returns distribution
axes[0, 1].hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].set_title(f"{current_stock} - Return Distribution")
axes[0, 1].set_xlabel("Daily Return (%)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Cumulative returns
cumulative = [1.0]
for ret in returns:
    cumulative.append(cumulative[-1] * (1 + ret))
axes[1, 0].plot(dates, cumulative[1:], linewidth=2, color='green')
axes[1, 0].set_title(f"{current_stock} - Cumulative Returns")
axes[1, 0].set_ylabel("Cumulative Return")
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Drawdown chart
prices = calc.close_prices.tolist()
peak = prices[0]
drawdowns = []
for price in prices:
    if price > peak:
        peak = price
    drawdown = (peak - price) / peak
    drawdowns.append(drawdown)
axes[1, 1].plot(calc.close_prices.index, drawdowns, linewidth=2, color='red')
axes[1, 1].set_title(f"{current_stock} - Drawdown Chart")
axes[1, 1].set_ylabel("Drawdown")
axes[1, 1].grid(True, alpha=0.3)

# Add radio buttons for stock selection
rax = plt.axes([0.1, 0.05, 0.2, 0.1])
radio = RadioButtons(rax, stocks)

def update_stock(label):
    """Update plots when stock selection changes"""
    global current_stock
    current_stock = label
    calc = calculators[label]
    returns = calc.calculate_returns()
    dates = calc.close_prices.index[1:]

    # Update price chart
    axes[0, 0].clear()
    axes[0, 0].plot(calc.close_prices.index, calc.close_prices.values, linewidth=2)
    axes[0, 0].set_title(f"{label} - Price Chart")
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].grid(True, alpha=0.3)

    # Update returns distribution
    axes[0, 1].clear()
    axes[0, 1].hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f"{label} - Return Distribution")
    axes[0, 1].set_xlabel("Daily Return (%)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Update cumulative returns
    axes[1, 0].clear()
    cumulative = [1.0]
    for ret in returns:
        cumulative.append(cumulative[-1] * (1 + ret))
    axes[1, 0].plot(dates, cumulative[1:], linewidth=2, color='green')
    axes[1, 0].set_title(f"{label} - Cumulative Returns")
    axes[1, 0].set_ylabel("Cumulative Return")
    axes[1, 0].grid(True, alpha=0.3)

    # Update drawdown chart
    axes[1, 1].clear()
    prices = calc.close_prices.tolist()
    peak = prices[0]
    drawdowns = []
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        drawdowns.append(drawdown)
    axes[1, 1].plot(calc.close_prices.index, drawdowns, linewidth=2, color='red')
    axes[1, 1].set_title(f"{label} - Drawdown Chart")
    axes[1, 1].set_ylabel("Drawdown")
    axes[1, 1].grid(True, alpha=0.3)

    plt.draw()

radio.on_clicked(update_stock)

plt.suptitle("Interactive Stock Analysis Dashboard", fontsize=16)
plt.show()
```

## Advanced Scenarios

### Example 12: Monte Carlo Simulation

```python
from main import StockReturnCalculator
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_simulation(ticker, start_date, end_date, num_simulations=1000, days_forward=252):
    """Run Monte Carlo simulation for future stock prices"""

    # Get historical data
    calculator = StockReturnCalculator(ticker, start_date, end_date)
    calculator.fetch_price_data()
    returns = np.array(calculator.calculate_returns())
    last_price = calculator.close_prices.iloc[-1]

    # Calculate parameters from historical returns
    mean_return = returns.mean()
    std_return = returns.std()

    # Run simulations
    simulations = np.zeros((num_simulations, days_forward))

    for i in range(num_simulations):
        # Generate random returns
        random_returns = np.random.normal(mean_return, std_return, days_forward)

        # Calculate price path
        price_path = last_price * np.cumprod(1 + random_returns)
        simulations[i] = price_path

    # Calculate statistics
    final_prices = simulations[:, -1]
    mean_final_price = final_prices.mean()
    median_final_price = np.median(final_prices)
    std_final_price = final_prices.std()

    # Calculate confidence intervals
    ci_95 = np.percentile(final_prices, [2.5, 97.5])
    ci_68 = np.percentile(final_prices, [16, 84])

    print(f"Monte Carlo Simulation for {ticker}")
    print("="*60)
    print(f"Current Price: ${last_price:.2f}")
    print(f"Simulations: {num_simulations}")
    print(f"Time Horizon: {days_forward} trading days (~{days_forward/252:.1f} years)")
    print(f"\nExpected Final Price: ${mean_final_price:.2f}")
    print(f"Median Final Price: ${median_final_price:.2f}")
    print(f"Std Dev of Final Price: ${std_final_price:.2f}")
    print(f"\n95% Confidence Interval: [${ci_95[0]:.2f}, ${ci_95[1]:.2f}]")
    print(f"68% Confidence Interval: [${ci_68[0]:.2f}, ${ci_68[1]:.2f}]")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot sample price paths
    for i in range(min(50, num_simulations)):
        ax1.plot(simulations[i], alpha=0.1, linewidth=0.5)
    ax1.axhline(y=last_price, color='red', linestyle='--', label='Current Price')
    ax1.set_title(f"{ticker} - Monte Carlo Price Paths (50 samples)")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot final price distribution
    ax2.hist(final_prices, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=last_price, color='red', linestyle='--', label='Current Price')
    ax2.axvline(x=mean_final_price, color='blue', linestyle='--', label='Mean Final Price')
    ax2.axvline(x=ci_95[0], color='green', linestyle=':', alpha=0.7, label='95% CI')
    ax2.axvline(x=ci_95[1], color='green', linestyle=':', alpha=0.7)
    ax2.set_title(f"{ticker} - Final Price Distribution")
    ax2.set_xlabel("Final Price ($)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return simulations, final_prices

# Run simulation for NVIDIA
simulations, final_prices = monte_carlo_simulation("NVDA", "2020-01-01", "2024-12-31")
```

### Example 13: Pair Trading Strategy

```python
from main import StockReturnCalculator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Identify correlated pairs (example: JPM and BAC)
pair1 = "JPM"
pair2 = "BAC"

# Fetch historical data
calc1 = StockReturnCalculator(pair1, "2020-01-01", "2024-12-31")
calc2 = StockReturnCalculator(pair2, "2020-01-01", "2024-12-31")

calc1.fetch_price_data()
calc2.fetch_price_data()

# Get price series
prices1 = calc1.close_prices
prices2 = calc2.close_prices

# Align dates (intersection)
common_dates = prices1.index.intersection(prices2.index)
prices1_aligned = prices1.loc[common_dates]
prices2_aligned = prices2.loc[common_dates]

# Calculate spread (ratio)
spread = prices1_aligned / prices2_aligned

# Calculate z-score of spread
spread_mean = spread.mean()
spread_std = spread.std()
zscore = (spread - spread_mean) / spread_std

# Define trading signals
entry_threshold = 2.0  # Enter when z-score > 2 or < -2
exit_threshold = 0.5   # Exit when z-score crosses back to 0.5

# Generate signals
position = 0  # 0: no position, 1: long spread, -1: short spread
signals = []
positions = []

for z in zscore:
    if position == 0:
        if z > entry_threshold:
            position = -1  # Short spread (short pair1, long pair2)
        elif z < -entry_threshold:
            position = 1   # Long spread (long pair1, short pair2)
    elif position == 1:  # Long spread
        if z > -exit_threshold:
            position = 0   # Exit
    elif position == -1:  # Short spread
        if z < exit_threshold:
            position = 0   # Exit

    signals.append(position)
    positions.append(position)

# Calculate returns
# Simplified: Assume we trade $1 of spread
returns = []
for i in range(1, len(positions)):
    if positions[i-1] != 0:
        # Return = position * spread return
        spread_return = (spread.iloc[i] / spread.iloc[i-1] - 1)
        trade_return = positions[i-1] * spread_return
        returns.append(trade_return)
    else:
        returns.append(0)

# Convert to numpy array
returns_array = np.array(returns)

# Calculate strategy metrics
total_return = np.prod(1 + returns_array) - 1
sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(252) if returns_array.std() != 0 else 0
max_drawdown = calculate_max_drawdown_from_returns(returns_array)

def calculate_max_drawdown_from_returns(returns):
    """Calculate maximum drawdown from return series"""
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown) if len(drawdown) > 0 else 0

print(f"Pair Trading Strategy: {pair1} vs {pair2}")
print("="*60)
print(f"Correlation: {np.corrcoef(prices1_aligned, prices2_aligned)[0,1]:.3f}")
print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Trades: {sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])}")

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Plot price ratio
ax1.plot(common_dates, spread, label='Price Ratio', linewidth=2)
ax1.axhline(y=spread_mean, color='red', linestyle='--', alpha=0.7, label='Mean')
ax1.axhline(y=spread_mean + entry_threshold * spread_std, color='orange', linestyle=':', alpha=0.7, label='Entry/Exit Bands')
ax1.axhline(y=spread_mean - entry_threshold * spread_std, color='orange', linestyle=':', alpha=0.7)
ax1.set_title(f"{pair1}/{pair2} Price Ratio")
ax1.set_ylabel("Ratio")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot z-score
ax2.plot(common_dates, zscore, label='Z-Score', linewidth=2)
ax2.axhline(y=entry_threshold, color='green', linestyle='--', alpha=0.7, label='Entry Threshold')
ax2.axhline(y=-entry_threshold, color='green', linestyle='--', alpha=0.7)
ax2.axhline(y=exit_threshold, color='red', linestyle=':', alpha=0.7, label='Exit Threshold')
ax2.axhline(y=-exit_threshold, color='red', linestyle=':', alpha=0.7)
ax2.set_title("Z-Score of Price Ratio")
ax2.set_ylabel("Z-Score")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot strategy returns
cumulative_returns = np.cumprod(1 + returns_array)
ax3.plot(common_dates[1:len(returns_array)+1], cumulative_returns, label='Strategy Returns', linewidth=2)
ax3.set_title("Strategy Cumulative Returns")
ax3.set_xlabel("Date")
ax3.set_ylabel("Cumulative Return")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Conclusion

These examples demonstrate the versatility of the Stock Return Analyzer for various financial analysis tasks. The modular design of the `StockReturnCalculator` class makes it easy to extend and adapt for specific needs.

Remember:
1. Always verify calculations and understand assumptions
2. Consider transaction costs and liquidity in real trading
3. Past performance does not guarantee future results
4. Use appropriate risk management techniques

For more information, refer to:
- [API Reference](api-reference.md) for detailed class documentation
- [Usage Guide](usage.md) for basic to advanced usage patterns
- [Installation Guide](installation.md) for setup instructions