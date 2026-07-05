import numpy as np
from scipy.stats import norm

def historical_var(portfolio_returns, portfolio_value, confidence=0.95):
    """
    Historical Simulation VaR.
    Assumption: the past window of returns is representative of tomorrow's risk.
    No distributional assumption is made — we just read off the empirical quantile.
    Weakness: if the sampled history was unusually calm (or volatile), VaR will be too.
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1")
    if portfolio_returns is None or len(portfolio_returns) == 0:
        raise ValueError("portfolio_returns is empty")
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")

    alpha = 1 - confidence
    var_return = np.percentile(portfolio_returns, alpha * 100)  # a negative number = a loss
    return -var_return * portfolio_value


def parametric_var(returns, weights, portfolio_value, confidence=0.95, cov_override=None):
    """
    Variance-Covariance (Normal) VaR.
    Assumption: portfolio returns are Normally distributed. This is the method's
    biggest weakness — real markets have fatter tails than a Normal curve predicts,
    so this can understate true risk in extreme scenarios.
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")

    w = np.array([weights[t] for t in returns.columns])
    cov = cov_override if cov_override is not None else returns.cov().values
    port_vol = np.sqrt(w @ cov @ w)
    z = norm.ppf(confidence)  # e.g. 1.645 for 95%, 2.33 for 99% under a Normal curve
    return z * port_vol * portfolio_value, port_vol


def parametric_cvar(returns, weights, portfolio_value, confidence=0.95, cov_override=None):
    """
    Expected Shortfall (CVaR) under the Normal assumption.
    This is the average loss GIVEN we're already past the VaR cutoff — it answers
    "how bad, on average, is the tail" rather than just "where does the tail start."
    """
    w = np.array([weights[t] for t in returns.columns])
    cov = cov_override if cov_override is not None else returns.cov().values
    port_vol = np.sqrt(w @ cov @ w)
    z = norm.ppf(confidence)
    return portfolio_value * port_vol * norm.pdf(z) / (1 - confidence)


def monte_carlo_var(returns, weights, portfolio_value, confidence=0.95, n_sim=100_000, seed=42):
    """
    Monte Carlo VaR.
    Assumption: same as Parametric (Normal, using historical mean/covariance) —
    we're simulating FROM that assumption rather than solving it in closed form.
    That's why Parametric and Monte Carlo should converge closely; if they don't,
    something is wrong (too few simulations, or a data/weights mismatch).
    Seed is fixed so results are reproducible run to run.
    """
    if n_sim < 1000:
        raise ValueError("n_sim should be at least 1000 for a stable quantile estimate")

    w = np.array([weights[t] for t in returns.columns])
    mean = returns.mean().values
    cov = returns.cov().values
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mean, cov, size=n_sim)
    pnl = (sims @ w) * portfolio_value
    alpha = 1 - confidence
    return -np.percentile(pnl, alpha * 100), pnl
