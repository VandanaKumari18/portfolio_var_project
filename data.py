import pandas as pd
import yfinance as yf

def fetch_prices(tickers, start, end=None):
    """
    Download daily adjusted close prices for given tickers.
    Assumption: 'Close' from yfinance with auto_adjust=True already accounts for
    splits/dividends, so we don't need a separate 'Adj Close' column.
    """
    if not tickers or not isinstance(tickers, list):
        raise ValueError("tickers must be a non-empty list, e.g. ['AAPL', 'GOOGL']")

    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        # Network failure, Yahoo Finance downtime, invalid date format, etc.
        raise ConnectionError(f"Failed to fetch data from Yahoo Finance: {e}")

    if raw.empty:
        raise ValueError(
            "No data returned. Check that tickers are spelled correctly and the "
            "date range is valid (not in the future, not a holiday-only range)."
        )

    prices = raw['Close'] if 'Close' in raw.columns.get_level_values(0) else raw

    # Assumption: we drop any day where ANY ticker is missing data, so all series
    # stay aligned on the same trading calendar. This slightly shortens history
    # if one ticker started trading later than others.
    prices = prices.dropna(how='any')

    if prices.empty:
        raise ValueError("Price data is empty after cleaning — check ticker symbols.")

    missing_tickers = set(tickers) - set(prices.columns)
    if missing_tickers:
        raise ValueError(f"No data found for: {missing_tickers}. Check spelling.")

    return prices


def compute_returns(prices):
    """
    Simple daily % returns: (price_today - price_yesterday) / price_yesterday.
    Assumption: we use simple returns, not log returns — fine for 1-day VaR at
    this scale of price moves, where the difference between the two is negligible.
    """
    if prices is None or prices.empty:
        raise ValueError("prices DataFrame is empty — fetch_prices may have failed silently.")

    returns = prices.pct_change().dropna(how='any')

    if returns.empty:
        raise ValueError("Returns DataFrame is empty — need at least 2 days of price data.")

    return returns


def portfolio_returns_series(returns, weights):
    """
    Weighted sum of individual stock returns -> one portfolio return series.
    Assumption: weights are static (don't change day to day) — a simplification;
    real portfolios drift in weight as prices move and would need rebalancing logic.
    """
    if not weights:
        raise ValueError("weights dict is empty.")

    missing = set(weights.keys()) - set(returns.columns)
    if missing:
        raise ValueError(f"Weights reference tickers not in returns data: {missing}")

    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")

    w = pd.Series(weights)[returns.columns]
    return returns.dot(w)
