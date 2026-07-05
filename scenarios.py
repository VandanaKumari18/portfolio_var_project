import numpy as np
import pandas as pd

def stress_scenario(shocks, weights, portfolio_value):
    """
    Deterministic 'what if' P&L calculation.
    Assumption: shocks apply instantly and independently to each stock's weighted
    dollar exposure — no second-order effects (e.g. liquidity, margin calls) modeled.
    """
    if set(shocks.keys()) != set(weights.keys()):
        raise ValueError("shocks and weights must cover the same set of tickers")

    per_ticker_loss = {t: portfolio_value * weights[t] * shocks[t] for t in weights}
    return sum(per_ticker_loss.values()), per_ticker_loss


def correlation_spike_cov(returns, target_corr=0.85):
    """
    Rebuilds the covariance matrix keeping each stock's own volatility the same,
    but forcing all pairwise correlations to target_corr.
    Assumption: in a crisis, individual stock volatility doesn't necessarily change,
    but co-movement does — this isolates that single effect for comparison.
    """
    if not (0 <= target_corr <= 1):
        raise ValueError("target_corr must be between 0 and 1")

    vols = returns.std().values
    n = len(vols)
    corr = np.full((n, n), target_corr)
    np.fill_diagonal(corr, 1.0)
    return np.outer(vols, vols) * corr


def rolling_var_backtest(portfolio_returns, window=250, confidence=0.95):
    """
    Rolls a historical VaR window forward day by day and checks whether each
    day's ACTUAL loss exceeded the VaR predicted using only PRIOR days (no lookahead).
    Assumption: 250 trading days (~1 year) is enough history to estimate VaR without
    the window itself being too short to be statistically meaningful.
    """
    if len(portfolio_returns) <= window:
        raise ValueError(f"Need more than {window} days of data for this window size.")

    alpha = 1 - confidence
    rolling_var = portfolio_returns.rolling(window).apply(
        lambda x: -np.percentile(x, alpha * 100), raw=True
    )
    predicted_var = rolling_var.shift(1)  # only use info available BEFORE that day
    actual_loss = -portfolio_returns
    breach = actual_loss > predicted_var

    df = pd.DataFrame({
        'actual_return': portfolio_returns,
        'predicted_var': predicted_var,
        'breach': breach
    }).dropna()
    return df
