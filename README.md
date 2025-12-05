# Stock Return Analyzer

A Python-based financial analysis tool for calculating and visualizing stock performance metrics. This project analyzes S&P 500, Apple (AAPL), and NVIDIA (NVDA) stocks over a 10-year period (2015-2025) and provides comprehensive performance metrics including cumulative returns, Sharpe ratios, maximum drawdowns, and market capitalization trends.

## Features

- **Stock Data Fetching**: Automatically download historical stock data from Yahoo Finance
- **Performance Metrics Calculation**:
  - Daily returns and cumulative returns
  - Annualized Sharpe ratio
  - Maximum drawdown
  - Market capitalization (current and historical)
- **Visualization**: Generate professional charts for:
  - Cumulative returns time series
  - Sharpe ratio comparison
  - Maximum drawdown analysis
  - Market capitalization evolution
- **Modular Design**: Reusable `StockReturnCalculator` class for easy extension to other stocks
- **Multi-plot Display**: Combined subplot visualization for comprehensive analysis

## Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Yamiyorunoshura/project-part-A.git
   cd project-part-A
   ```

2. **Install dependencies**:
   Using uv (recommended):
   ```bash
   pip install uv
   uv sync
   ```

   Using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   ```bash
   uv run main.py
   ```

## Usage

### Basic Analysis
The main script (`main.py`) automatically analyzes three assets:
- S&P 500 Index (^GSPC)
- Apple Inc. (AAPL)
- NVIDIA Corporation (NVDA)

Run the analysis:
```bash
python main.py
```

This will:
1. Fetch historical price data (2015-10-01 to 2025-09-30)
2. Calculate all performance metrics
3. Display results in the terminal
4. Generate and show a comprehensive 4-chart visualization

### Using the StockReturnCalculator Class

Import and use the calculator in your own scripts:

```python
from main import StockReturnCalculator

# Initialize calculator for a stock
calculator = StockReturnCalculator("AAPL", "2020-01-01", "2025-01-01")

# Fetch price data
calculator.fetch_price_data()

# Calculate returns
returns = calculator.calculate_returns()

# Calculate Sharpe ratio
sharpe = calculator.calculate_sharpe_ratio()

# Calculate maximum drawdown
max_dd = calculator.calculate_max_drawdown()

# Get market capitalization
market_cap = calculator.get_market_cap()

# Calculate historical market capitalization
market_cap_history = calculator.calculate_market_cap_history()
```

### Custom Analysis
Modify `main.py` to analyze different stocks or time periods by changing the ticker symbols and date ranges in the initialization section.

## Project Structure

```
.
├── main.py              # Main analysis script with StockReturnCalculator class
├── sample.py            # Example analysis with different stocks
├── pyproject.toml       # Project dependencies and configuration
├── README.md           # This documentation file
├── docs/               # Detailed documentation (see below)
└── .gitignore          # Git ignore file
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Overview](docs/overview.md) - Project overview and architecture
- [Installation](docs/installation.md) - Detailed installation instructions
- [Usage Guide](docs/usage.md) - Comprehensive usage examples
- [API Reference](docs/api-reference.md) - StockReturnCalculator class documentation
- [Examples](docs/examples.md) - Additional analysis examples

## Dependencies

- **pandas** >=2.3.3: Data manipulation and analysis
- **yfinance** >=0.2.66: Yahoo Finance data fetching
- **matplotlib** >=3.0.0: Data visualization
- **numpy** >=1.0.0: Numerical computations
- **ruff** >=0.14.6: Code formatting and linting
- **black** >=25.11.0: Code formatting

## Results

The analysis produces the following metrics for each asset (example values):

| Asset | Cumulative Return | Sharpe Ratio | Max Drawdown | Market Cap |
|-------|------------------|--------------|--------------|------------|
| S&P 500 | ~120.5% | ~0.85 | ~0.34 | N/A |
| AAPL | ~580.3% | ~1.12 | ~0.45 | ~$2.8T |
| NVDA | ~12,500% | ~1.45 | ~0.68 | ~$1.2T |

Visualizations include:
1. Cumulative returns comparison over time
2. Sharpe ratio bar chart
3. Maximum drawdown comparison
4. Market capitalization evolution (AAPL & NVDA only)


