# API Reference

This document provides detailed API documentation for the `StockReturnCalculator` class and related functions in the Stock Return Analyzer.

## Table of Contents
- [StockReturnCalculator Class](#stockreturncalculator-class)
  - [Constructor](#constructor)
  - [Methods](#methods)
  - [Attributes](#attributes)
- [Plotting Functions](#plotting-functions)
- [Helper Functions](#helper-functions)

## StockReturnCalculator Class

The `StockReturnCalculator` class is the core component for stock analysis. It handles data fetching, metric calculations, and provides access to results.

### Constructor

```python
StockReturnCalculator(ticker_symbol, start_date, end_date)
```

Creates a new calculator instance for a specific stock and time period.

**Parameters:**
- `ticker_symbol` (str): Yahoo Finance ticker symbol (e.g., "AAPL", "^GSPC")
- `start_date` (str): Start date in "YYYY-MM-DD" format
- `end_date` (str): End date in "YYYY-MM-DD" format

**Example:**
```python
calculator = StockReturnCalculator("AAPL", "2020-01-01", "2024-12-31")
```

**Notes:**
- Dates should be valid trading days (non-trading days will be adjusted)
- Ticker symbols must be valid Yahoo Finance symbols
- The object stores data internally; call `fetch_price_data()` to populate

### Methods

#### `fetch_price_data()`

```python
fetch_price_data() -> None
```

Downloads historical price data from Yahoo Finance and stores it internally.

**Behavior:**
- Downloads daily price data for the specified date range
- Uses `yf.download()` with `auto_adjust=False` for compatibility
- Stores the full DataFrame in `self.price_data`
- Extracts adjusted close prices to `self.close_prices`

**Raises:**
- `ConnectionError`: If unable to connect to Yahoo Finance
- `ValueError`: If ticker symbol is invalid or data unavailable

**Example:**
```python
calculator.fetch_price_data()
print(f"Fetched {len(calculator.close_prices)} days of data")
```

#### `calculate_returns()`

```python
calculate_returns() -> list[float]
```

Calculates daily percentage returns from adjusted close prices.

**Returns:**
- `list[float]`: Daily returns as decimal values (e.g., 0.01 for 1%)

**Algorithm:**
1. Converts `self.close_prices` to a list
2. Calculates: `(price_t - price_{t-1}) / price_{t-1}` for each day
3. Stores result in `self.returns` for later use

**Notes:**
- Requires `fetch_price_data()` to be called first
- Returns list has one less element than price data (no return for first day)
- Uses adjusted close prices to account for dividends and splits

**Example:**
```python
returns = calculator.calculate_returns()
print(f"Average daily return: {np.mean(returns):.4%}")
```

#### `calculate_sharpe_ratio()`

```python
calculate_sharpe_ratio(risk_free_rate: float = 0.0) -> float
```

Calculates the annualized Sharpe ratio, a risk-adjusted return metric.

**Parameters:**
- `risk_free_rate` (float, optional): Annual risk-free rate as decimal. Default: 0.0

**Returns:**
- `float`: Annualized Sharpe ratio

**Formula:**
```
Sharpe = (E[R] - R_f) / σ(R)
Annualized Sharpe = Sharpe × √252
```
Where:
- `E[R]`: Mean of daily returns
- `R_f`: Daily risk-free rate (annual rate ÷ 252)
- `σ(R)`: Standard deviation of daily returns
- `252`: Approximate number of trading days per year

**Notes:**
- Automatically calls `calculate_returns()` if `self.returns` is None
- Returns 0 if standard deviation is 0 (to avoid division by zero)
- Assumes 252 trading days per year for annualization

**Example:**
```python
# With default risk-free rate
sharpe = calculator.calculate_sharpe_ratio()

# With custom risk-free rate (e.g., 2%)
sharpe = calculator.calculate_sharpe_ratio(risk_free_rate=0.02)
```

#### `calculate_max_drawdown()`

```python
calculate_max_drawdown() -> float
```

Calculates the maximum historical drawdown (peak-to-trough decline).

**Returns:**
- `float`: Maximum drawdown as decimal (e.g., 0.35 for 35% drawdown)

**Algorithm:**
1. Tracks running peak price
2. For each price: `drawdown = (peak - price) / peak`
3. Returns the maximum drawdown encountered

**Notes:**
- Requires `fetch_price_data()` to be called first
- Measures worst historical loss from a peak
- Useful for assessing downside risk

**Example:**
```python
max_dd = calculator.calculate_max_drawdown()
print(f"Maximum drawdown: {max_dd:.2%}")
```

#### `get_market_cap()`

```python
get_market_cap() -> float | None
```

Retrieves current market capitalization from Yahoo Finance.

**Returns:**
- `float | None`: Market capitalization in USD, or `None` if unavailable

**Notes:**
- Fetches real-time data from Yahoo Finance
- For indices (like ^GSPC), returns `None`
- Data may be delayed by 15 minutes for US markets

**Example:**
```python
market_cap = calculator.get_market_cap()
if market_cap:
    print(f"Market cap: ${market_cap/1e9:.2f}B")
```

#### `get_shares_outstanding_history()`

```python
get_shares_outstanding_history() -> pandas.Series
```

Retrieves historical shares outstanding data.

**Returns:**
- `pandas.Series`: Historical shares outstanding with dates as index

**Behavior:**
1. Attempts to fetch full historical shares data via `get_shares_full()`
2. If historical data unavailable, creates a Series with current shares outstanding
3. Returns empty Series if no data available

**Notes:**
- Index is timezone-naive for compatibility with price data
- Uses forward-fill for missing dates in later calculations
- Some stocks may not have historical shares data

**Example:**
```python
shares_series = calculator.get_shares_outstanding_history()
print(f"Shares data points: {len(shares_series)}")
```

#### `calculate_market_cap_history()`

```python
calculate_market_cap_history() -> pandas.Series
```

Calculates historical market capitalization time series.

**Returns:**
- `pandas.Series`: Historical market cap with dates as index

**Algorithm:**
1. Aligns shares outstanding data with price dates using forward fill
2. Calculates: `market_cap = shares × adjusted_close_price`
3. Falls back to current shares if historical data incomplete

**Notes:**
- Requires `fetch_price_data()` to be called first
- Returns empty Series if no shares data available
- For indices, returns empty Series

**Example:**
```python
market_cap_history = calculator.calculate_market_cap_history()
if not market_cap_history.empty:
    print(f"Market cap history from {market_cap_history.index[0]} to {market_cap_history.index[-1]}")
```

### Attributes

#### `ticker_symbol`
- **Type**: `str`
- **Description**: The ticker symbol provided during initialization

#### `start_date`
- **Type**: `str`
- **Description**: Start date in "YYYY-MM-DD" format

#### `end_date`
- **Type**: `str`
- **Description**: End date in "YYYY-MM-DD" format

#### `price_data`
- **Type**: `pandas.DataFrame` or `None`
- **Description**: Full historical price data from Yahoo Finance. Contains columns: "Open", "High", "Low", "Close", "Adj Close", "Volume". `None` until `fetch_price_data()` is called.

#### `close_prices`
- **Type**: `pandas.Series` or `None`
- **Description**: Adjusted close prices extracted from `price_data`. `None` until `fetch_price_data()` is called.

#### `returns`
- **Type**: `list[float]` or `None`
- **Description**: Daily percentage returns as decimal values. `None` until `calculate_returns()` is called.

## Plotting Functions

### `plot_metric()`

```python
plot_metric(dates, data_dict, title, ylabel, figsize=(12, 8)) -> matplotlib.pyplot
```

General-purpose plotting function for time series data.

**Parameters:**
- `dates`: Index or list of dates for x-axis
- `data_dict` (dict): Dictionary mapping series labels to data lists
- `title` (str): Plot title
- `ylabel` (str): Y-axis label
- `figsize` (tuple, optional): Figure size as (width, height). Default: (12, 8)

**Returns:**
- `matplotlib.pyplot`: Pyplot object with the figure

**Example:**
```python
data = {"Series A": [1, 2, 3], "Series B": [2, 3, 4]}
plt = plot_metric([1, 2, 3], data, "My Plot", "Values")
plt.show()
```

### `plot_cumulative_returns()`

```python
plot_cumulative_returns(dates, cumulative_returns_dict) -> matplotlib.pyplot
```

Specialized function for plotting cumulative returns.

**Parameters:**
- `dates`: Index or list of dates for x-axis
- `cumulative_returns_dict` (dict): Dictionary mapping labels to cumulative return lists

**Returns:**
- `matplotlib.pyplot`: Pyplot object with the figure

**Notes:**
- Converts cumulative returns to percentage: `(value - 1) × 100`
- Uses `plot_metric()` internally with appropriate labels

**Example:**
```python
cumulative_returns = {"AAPL": [1.0, 1.1, 1.2], "MSFT": [1.0, 1.05, 1.15]}
plt = plot_cumulative_returns(dates, cumulative_returns)
plt.savefig("cumulative_returns.png")
```

### `plot_sharpe_ratios()`

```python
plot_sharpe_ratios(sharpe_ratios_dict) -> matplotlib.pyplot
```

Creates a bar chart of Sharpe ratios.

**Parameters:**
- `sharpe_ratios_dict` (dict): Dictionary mapping labels to Sharpe ratio values

**Returns:**
- `matplotlib.pyplot`: Pyplot object with the figure

**Example:**
```python
sharpe_ratios = {"AAPL": 1.12, "MSFT": 0.95, "GOOGL": 1.05}
plt = plot_sharpe_ratios(sharpe_ratios)
plt.show()
```

### `plot_max_drawdowns()`

```python
plot_max_drawdowns(max_drawdowns_dict) -> matplotlib.pyplot
```

Creates a bar chart of maximum drawdowns.

**Parameters:**
- `max_drawdowns_dict` (dict): Dictionary mapping labels to max drawdown values

**Returns:**
- `matplotlib.pyplot`: Pyplot object with the figure

**Notes:**
- Converts drawdowns to percentage: `value × 100`

**Example:**
```python
max_drawdowns = {"AAPL": 0.45, "MSFT": 0.38, "GOOGL": 0.42}
plt = plot_max_drawdowns(max_drawdowns)
plt.show()
```

### `plot_market_caps()`

```python
plot_market_caps(market_caps_dict) -> matplotlib.pyplot
```

Creates a bar chart of market capitalizations.

**Parameters:**
- `market_caps_dict` (dict): Dictionary mapping labels to market cap values

**Returns:**
- `matplotlib.pyplot`: Pyplot object with the figure

**Notes:**
- Converts market caps to billions USD for readability

**Example:**
```python
market_caps = {"AAPL": 2.8e12, "MSFT": 3.1e12, "GOOGL": 1.8e12}
plt = plot_market_caps(market_caps)
plt.show()
```

### `plot_market_cap_history()`

```python
plot_market_cap_history(dates, market_cap_history_dict) -> matplotlib.pyplot
```

Creates a line chart of historical market capitalization.

**Parameters:**
- `dates`: Index or list of dates for x-axis
- `market_cap_history_dict` (dict): Dictionary mapping labels to pandas.Series of historical market cap

**Returns:**
- `matplotlib.pyplot`: Pyplot object with the figure

**Notes:**
- Converts market caps to billions USD for readability
- Requires pandas.Series inputs with date indices

**Example:**
```python
market_cap_history = {
    "AAPL": aapl_market_cap_history,  # pandas.Series
    "MSFT": msft_market_cap_history
}
plt = plot_market_cap_history(dates, market_cap_history)
plt.show()
```

## Helper Functions

### `calculate_cumulative_returns()`

```python
calculate_cumulative_returns(daily_returns) -> list[float]
```

Calculates cumulative returns from daily returns.

**Parameters:**
- `daily_returns` (list): List of daily return values as decimals

**Returns:**
- `list[float]`: Cumulative returns starting from 1.0

**Formula:**
```
cumulative[0] = 1.0
cumulative[t] = cumulative[t-1] × (1 + daily_returns[t-1])
```

**Example:**
```python
daily_returns = [0.01, 0.02, -0.01]
cumulative = calculate_cumulative_returns(daily_returns)
# Result: [1.0, 1.01, 1.0302, 1.020098]
```

## Error Handling

### Common Exceptions

1. **DataFetchError**: Raised when unable to download data from Yahoo Finance
2. **CalculationError**: Raised when calculations fail due to invalid data
3. **ParameterError**: Raised when invalid parameters are provided

### Best Practices

```python
try:
    calculator = StockReturnCalculator("INVALID", "2020-01-01", "2024-12-31")
    calculator.fetch_price_data()
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

## Performance Considerations

### Caching
- Consider caching downloaded data to avoid repeated API calls
- Yahoo Finance may rate-limit frequent requests

### Memory Usage
- Historical data for long periods can be memory-intensive
- Use appropriate data types (float32 vs float64) if precision allows

### Computational Complexity
- Most operations are O(n) for n trading days
- Market cap history calculation involves pandas operations with O(n log n) complexity

## Extension Points

### Subclassing
```python
class EnhancedStockCalculator(StockReturnCalculator):
    def calculate_sortino_ratio(self, risk_free_rate=0.0):
        # Implement Sortino ratio calculation
        pass
```

### Adding New Metrics
```python
def calculate_calmar_ratio(calculator, lookback_years=3):
    # Implement Calmar ratio using existing methods
    pass
```

## See Also

- [Usage Guide](usage.md) for practical examples
- [Examples](examples.md) for more analysis patterns
- [Overview](overview.md) for project architecture
- [Installation Guide](installation.md) for setup instructions