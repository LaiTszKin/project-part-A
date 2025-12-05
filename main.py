import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


## Define the object of the stock return calculator
class StockReturnCalculator:
    def __init__(self, ticker_symbol, start_date, end_date):
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = None
        self.close_prices = None
        self.returns = None

    def fetch_price_data(self):
        self.price_data = yf.download(
            self.ticker_symbol,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
        )
        self.close_prices = self.price_data["Adj Close"].iloc[:, 0]

    def calculate_returns(self):
        price_list = self.close_prices.tolist()
        trading_days = len(price_list)
        price_new = price_list[1:trading_days]
        price_old = price_list[0 : trading_days - 1]
        self.returns = [(new - old) / old for new, old in zip(price_new, price_old)]
        return self.returns

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate annualized Sharpe ratio (assuming risk-free rate is 0)"""
        if self.returns is None:
            self.calculate_returns()
        returns_array = np.array(self.returns)
        excess_returns = returns_array - risk_free_rate
        sharpe_ratio = (
            np.mean(excess_returns) / np.std(returns_array)
            if np.std(returns_array) != 0
            else 0
        )
        # Annualize: multiply by sqrt(252) trading days
        annualized_sharpe = sharpe_ratio * np.sqrt(252)
        return annualized_sharpe

    def calculate_max_drawdown(self):
        """Calculate historical maximum drawdown"""
        if self.close_prices is None:
            self.fetch_price_data()
        prices = self.close_prices.tolist()
        peak = prices[0]
        max_drawdown = 0
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def get_market_cap(self):
        """Get market capitalization data from yfinance"""
        ticker = yf.Ticker(self.ticker_symbol)
        info = ticker.info
        market_cap = info.get("marketCap", None)
        return market_cap

    def get_shares_outstanding_history(self):
        """Get historical shares outstanding data

        Returns:
            pandas.Series: Historical shares outstanding with dates as index.
            If full historical data is not available, returns a Series with
            constant current shares outstanding value.
        """
        ticker = yf.Ticker(self.ticker_symbol)

        # Try to get historical shares data
        try:
            shares_full = ticker.get_shares_full()
            if shares_full is not None and len(shares_full) > 0:
                # Convert to Series with proper datetime index
                shares_series = shares_full.copy()
                # Ensure index is timezone-naive for compatibility with price data
                if hasattr(shares_series.index, "tz"):
                    shares_series.index = shares_series.index.tz_localize(None)
                return shares_series
        except Exception:
            pass

        # Fallback to current shares outstanding
        info = ticker.info
        current_shares = info.get("sharesOutstanding", None)
        if current_shares is not None:
            # Create a Series with a single date (today) for consistent return type
            import pandas as pd

            return pd.Series([current_shares], index=[pd.Timestamp.now()])

        # If no data available, return empty Series
        import pandas as pd

        return pd.Series(dtype=float)

    def calculate_market_cap_history(self):
        """Calculate historical market capitalization

        Returns:
            pandas.Series: Historical market cap with dates as index.
            Calculated as shares_outstanding * adjusted_close_price.
            Uses forward fill for shares data to align with price dates.
        """
        # Ensure price data is fetched
        if self.close_prices is None:
            self.fetch_price_data()

        # Get shares outstanding history
        shares_series = self.get_shares_outstanding_history()

        if len(shares_series) == 0:
            # No shares data available
            import pandas as pd

            return pd.Series(dtype=float)

        # Align shares data with price dates
        # Create a DataFrame with price dates as index
        import pandas as pd

        price_dates = self.close_prices.index

        # Convert shares series to DataFrame
        shares_df = pd.DataFrame(shares_series, columns=["shares"])
        shares_df.index = pd.DatetimeIndex(shares_df.index)

        # Remove duplicate indices (keep last value)
        shares_df = shares_df[~shares_df.index.duplicated(keep="last")]

        # Sort by date for forward fill
        shares_df = shares_df.sort_index()

        # Reindex to price dates, forward filling shares values
        shares_aligned = shares_df.reindex(price_dates, method="ffill")

        # If there are still NaN values (historical data doesn't cover full period),
        # fall back to current shares outstanding
        if shares_aligned["shares"].isna().any():
            # Get current shares outstanding
            ticker = yf.Ticker(self.ticker_symbol)
            info = ticker.info
            current_shares = info.get("sharesOutstanding", None)
            if current_shares is not None:
                # Fill NaN with current shares
                shares_aligned["shares"] = shares_aligned["shares"].fillna(
                    current_shares
                )

        # Calculate market cap: shares * adjusted close price
        market_cap_series = shares_aligned["shares"] * self.close_prices.values

        return market_cap_series


## General plotting function
def plot_metric(dates, data_dict, title, ylabel, figsize=(12, 8)):
    """General plotting function for multiple series"""
    plt.figure(figsize=figsize)
    for label, data in data_dict.items():
        plt.plot(dates, data, label=label, alpha=0.8, linewidth=2)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt


## Specialized plotting functions
def plot_cumulative_returns(dates, cumulative_returns_dict):
    """Plot cumulative returns chart"""
    # Convert cumulative returns to percentage
    data_dict = {
        label: [(x - 1) * 100 for x in data]
        for label, data in cumulative_returns_dict.items()
    }
    return plot_metric(
        dates,
        data_dict,
        "Cumulative Returns (2015-10-01 to 2025-09-30)",
        "Cumulative Return (%)",
    )


def plot_sharpe_ratios(sharpe_ratios_dict):
    """Plot Sharpe ratio bar chart"""
    labels = list(sharpe_ratios_dict.keys())
    values = list(sharpe_ratios_dict.values())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=["blue", "orange", "green"])
    plt.title("Sharpe Ratios (Annualized)", fontsize=16, fontweight="bold")
    plt.ylabel("Sharpe Ratio", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return plt


def plot_max_drawdowns(max_drawdowns_dict):
    """Plot maximum drawdown bar chart"""
    labels = list(max_drawdowns_dict.keys())
    values = [x * 100 for x in max_drawdowns_dict.values()]  # Convert to percentage
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=["red", "darkred", "maroon"])
    plt.title("Maximum Drawdowns", fontsize=16, fontweight="bold")
    plt.ylabel("Maximum Drawdown (%)", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return plt


def plot_market_caps(market_caps_dict):
    """Plot market capitalization bar chart"""
    labels = list(market_caps_dict.keys())
    values = list(market_caps_dict.values())
    # Convert market cap to billions USD for readability
    values_billions = [v / 1e9 if v else 0 for v in values]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values_billions, color=["purple", "gold", "cyan"])
    plt.title("Market Capitalization", fontsize=16, fontweight="bold")
    plt.ylabel("Market Cap (Billions USD)", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return plt


def plot_market_cap_history(dates, market_cap_history_dict):
    """Plot market capitalization time series line chart

    Args:
        dates: Index of dates for x-axis
        market_cap_history_dict: Dictionary with labels as keys and
                                 pandas.Series of historical market cap as values
    """
    plt.figure(figsize=(10, 6))

    for label, market_cap_series in market_cap_history_dict.items():
        # Convert to billions USD for readability
        market_cap_billions = market_cap_series / 1e9
        plt.plot(dates, market_cap_billions, label=label, alpha=0.8, linewidth=2)

    plt.title(
        "Market Capitalization Evolution (2015-10-01 to 2025-09-30)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Market Cap (Billions USD)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt


## Initialize objects for S&P 500, AAPL, and NVDA
snp500 = StockReturnCalculator("^GSPC", "2015-10-01", "2025-09-30")
aapl = StockReturnCalculator("AAPL", "2015-10-01", "2025-09-30")
nvda = StockReturnCalculator("NVDA", "2015-10-01", "2025-09-30")

## Fetch the price of the stocks from 2015-10-01 to 2025-09-30
## Set auto_adjust=False to maintain compatibility with older code structure
snp500.fetch_price_data()
aapl.fetch_price_data()
nvda.fetch_price_data()

## Calculate the returns of the stocks
snp500_daily_return = snp500.calculate_returns()
aapl_daily_return = aapl.calculate_returns()
nvda_daily_return = nvda.calculate_returns()


## Calculate cumulative returns
def calculate_cumulative_returns(daily_returns):
    cumulative = [1.0]  # Start with 100% (0% cumulative return)
    for ret in daily_returns:
        cumulative.append(cumulative[-1] * (1 + ret))
    return cumulative  # Keep the initial 1.0 to include starting point


snp500_cumulative = calculate_cumulative_returns(snp500_daily_return)
aapl_cumulative = calculate_cumulative_returns(aapl_daily_return)
nvda_cumulative = calculate_cumulative_returns(nvda_daily_return)

## Get all dates (include starting date for cumulative return baseline)
dates = snp500.close_prices.index

## Calculate all metrics
print("Calculating metrics...")

# Sharpe ratio
snp500_sharpe = snp500.calculate_sharpe_ratio()
aapl_sharpe = aapl.calculate_sharpe_ratio()
nvda_sharpe = nvda.calculate_sharpe_ratio()

# Maximum drawdown
snp500_max_dd = snp500.calculate_max_drawdown()
aapl_max_dd = aapl.calculate_max_drawdown()
nvda_max_dd = nvda.calculate_max_drawdown()

# Market capitalization (current)
snp500_market_cap = snp500.get_market_cap()
aapl_market_cap = aapl.get_market_cap()
nvda_market_cap = nvda.get_market_cap()

# Historical market capitalization time series (exclude S&P 500)
aapl_market_cap_history = aapl.calculate_market_cap_history()
nvda_market_cap_history = nvda.calculate_market_cap_history()

print("Metrics calculated:")
print(
    f"S&P 500 - Cumulative return: {(snp500_cumulative[-1] - 1) * 100:.2f}%, Sharpe: {snp500_sharpe:.4f}, Max Drawdown: {snp500_max_dd:.4f}, Market Cap: {snp500_market_cap}"
)
print(
    f"AAPL - Cumulative return: {(aapl_cumulative[-1] - 1) * 100:.2f}%, Sharpe: {aapl_sharpe:.4f}, Max Drawdown: {aapl_max_dd:.4f}, Market Cap: {aapl_market_cap}"
)
print(
    f"NVDA - Cumulative return: {(nvda_cumulative[-1] - 1) * 100:.2f}%, Sharpe: {nvda_sharpe:.4f}, Max Drawdown: {nvda_max_dd:.4f}, Market Cap: {nvda_market_cap}"
)

## Prepare data dictionaries
cumulative_returns_dict = {
    "S&P 500 (^GSPC)": snp500_cumulative,
    "Apple (AAPL)": aapl_cumulative,
    "NVIDIA (NVDA)": nvda_cumulative,
}

sharpe_ratios_dict = {
    "S&P 500": snp500_sharpe,
    "AAPL": aapl_sharpe,
    "NVDA": nvda_sharpe,
}

max_drawdowns_dict = {
    "S&P 500": snp500_max_dd,
    "AAPL": aapl_max_dd,
    "NVDA": nvda_max_dd,
}

market_caps_dict = {
    "S&P 500": snp500_market_cap,
    "AAPL": aapl_market_cap,
    "NVDA": nvda_market_cap,
}

# Historical market capitalization time series for visualization
market_cap_history_dict = {
    "Apple (AAPL)": aapl_market_cap_history,
    "NVIDIA (NVDA)": nvda_market_cap_history,
}

## Method 1: Display four charts in one window using subplot
print("\nGenerating combined subplot figure...")
fig, axes = plt.subplots(2, 2, figsize=(16, 9))

# 1. Cumulative returns chart
ax1 = axes[0, 0]
data_dict = {
    label: [(x - 1) * 100 for x in data]
    for label, data in cumulative_returns_dict.items()
}
for label, data in data_dict.items():
    ax1.plot(dates, data, label=label, alpha=0.8, linewidth=2)
ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
ax1.set_title(
    "Cumulative Returns (2015-10-01 to 2025-09-30)", fontsize=14, fontweight="bold"
)
ax1.set_xlabel("Date", fontsize=11)
ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45)

# 2. Sharpe ratio chart
ax2 = axes[0, 1]
labels = list(sharpe_ratios_dict.keys())
values = list(sharpe_ratios_dict.values())
ax2.bar(labels, values, color=["blue", "orange", "green"])
ax2.set_title("Sharpe Ratios (Annualized)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Sharpe Ratio", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

# 3. Maximum drawdown chart
ax3 = axes[1, 0]
labels = list(max_drawdowns_dict.keys())
values = [x * 100 for x in max_drawdowns_dict.values()]
ax3.bar(labels, values, color=["red", "darkred", "maroon"])
ax3.set_title("Maximum Drawdowns", fontsize=14, fontweight="bold")
ax3.set_ylabel("Maximum Drawdown (%)", fontsize=11)
ax3.grid(True, alpha=0.3, axis="y")

# 4. Market capitalization evolution chart (line plot for AAPL and NVDA only)
ax4 = axes[1, 1]
for label, market_cap_series in market_cap_history_dict.items():
    # Convert to billions USD for readability
    market_cap_billions = market_cap_series / 1e9
    ax4.plot(dates, market_cap_billions, label=label, alpha=0.8, linewidth=2)
ax4.set_title("Market Capitalization Evolution", fontsize=14, fontweight="bold")
ax4.set_xlabel("Date", fontsize=11)
ax4.set_ylabel("Market Cap (Billions USD)", fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
