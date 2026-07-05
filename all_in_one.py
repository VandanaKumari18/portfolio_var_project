import subprocess
import sys

try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============ CONFIG ============
TICKERS = ['AAPL', 'GOOGL', 'MSFT']
WEIGHTS = {'AAPL': 0.40, 'GOOGL': 0.35, 'MSFT': 0.25}
PORTFOLIO_VALUE = 1_000_000
START_DATE = '2022-01-01'
CONFIDENCE_LEVELS = [0.95, 0.99]

# ============ DATA ============
def fetch_prices(tickers, start, end=None):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No data returned — check tickers/dates/internet.")
    prices = raw['Close'] if 'Close' in raw.columns.get_level_values(0) else raw
    prices = prices.dropna(how='any')
    if prices.empty:
        raise ValueError("Price data empty after cleaning.")
    return prices

def compute_returns(prices):
    returns = prices.pct_change().dropna(how='any')
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    return returns

def portfolio_returns_series(returns, weights):
    missing = set(weights.keys()) - set(returns.columns)
    if missing:
        raise ValueError(f"Weights reference tickers not in returns data: {missing}")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")
    w = pd.Series(weights)[returns.columns]
    return returns.dot(w)

# ============ RISK ============
def historical_var(portfolio_returns, portfolio_value, confidence=0.95):
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1")
    alpha = 1 - confidence
    return -np.percentile(portfolio_returns, alpha * 100) * portfolio_value

def parametric_var(returns, weights, portfolio_value, confidence=0.95, cov_override=None):
    w = np.array([weights[t] for t in returns.columns])
    cov = cov_override if cov_override is not None else returns.cov().values
    port_vol = np.sqrt(w @ cov @ w)
    z = norm.ppf(confidence)
    return z * port_vol * portfolio_value, port_vol

def parametric_cvar(returns, weights, portfolio_value, confidence=0.95, cov_override=None):
    w = np.array([weights[t] for t in returns.columns])
    cov = cov_override if cov_override is not None else returns.cov().values
    port_vol = np.sqrt(w @ cov @ w)
    z = norm.ppf(confidence)
    return portfolio_value * port_vol * norm.pdf(z) / (1 - confidence)

def monte_carlo_var(returns, weights, portfolio_value, confidence=0.95, n_sim=100_000, seed=42):
    if n_sim < 1000:
        raise ValueError("n_sim should be at least 1000")
    w = np.array([weights[t] for t in returns.columns])
    mean = returns.mean().values
    cov = returns.cov().values
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mean, cov, size=n_sim)
    pnl = (sims @ w) * portfolio_value
    alpha = 1 - confidence
    return -np.percentile(pnl, alpha * 100), pnl

# ============ SCENARIOS ============
def stress_scenario(shocks, weights, portfolio_value):
    if set(shocks.keys()) != set(weights.keys()):
        raise ValueError("shocks and weights must cover the same tickers")
    per_ticker_loss = {t: portfolio_value * weights[t] * shocks[t] for t in weights}
    return sum(per_ticker_loss.values()), per_ticker_loss

def correlation_spike_cov(returns, target_corr=0.85):
    if not (0 <= target_corr <= 1):
        raise ValueError("target_corr must be between 0 and 1")
    vols = returns.std().values
    n = len(vols)
    corr = np.full((n, n), target_corr)
    np.fill_diagonal(corr, 1.0)
    return np.outer(vols, vols) * corr

def rolling_var_backtest(portfolio_returns, window=250, confidence=0.95):
    if len(portfolio_returns) <= window:
        raise ValueError(f"Need more than {window} days of data.")
    alpha = 1 - confidence
    rolling_var = portfolio_returns.rolling(window).apply(
        lambda x: -np.percentile(x, alpha * 100), raw=True)
    predicted_var = rolling_var.shift(1)
    actual_loss = -portfolio_returns
    breach = actual_loss > predicted_var
    return pd.DataFrame({'actual_return': portfolio_returns,
                          'predicted_var': predicted_var, 'breach': breach}).dropna()

# ============ EXPLANATIONS ============
def explain_var(method_name, var_dollar, portfolio_value, confidence):
    pct = var_dollar / portfolio_value * 100
    bad_days = round((1 - confidence) * 100)
    print(f"{method_name} — {int(confidence*100)}% VaR: ${var_dollar:,.0f} ({pct:.2f}% of the book)")
    print(f"  Out of 100 typical trading days, only about {bad_days} of them should")
    print(f"  produce a loss bigger than this. It's a threshold, not a ceiling.\n")

def explain_cvar(cvar_dollar, var_dollar, portfolio_value):
    pct = cvar_dollar / portfolio_value * 100
    print(f"Expected Shortfall / CVaR: ${cvar_dollar:,.0f} ({pct:.2f}%)")
    print(f"  This is the average loss if you're already past the VaR line — always")
    print(f"  worse than VaR (${var_dollar:,.0f}) since it's conditioned on the bad tail.\n")

def explain_scenario(name, total, portfolio_value):
    pct = total / portfolio_value * 100
    print(f"{name}: ${total:,.0f} ({pct:.2f}%)")
    print(f"  If this shock landed today, the ${portfolio_value:,.0f} book would be worth")
    print(f"  ${portfolio_value + total:,.0f} by the close. No probability attached —")
    print(f"  just a fixed 'what if' calculation.\n")

def explain_correlation_spike(normal_var, spiked_var):
    increase_pct = (spiked_var / normal_var - 1) * 100
    print(f"Correlation spike: VaR moves from ${normal_var:,.0f} to ${spiked_var:,.0f}, "
          f"about {increase_pct:.1f}% worse.")
    print("  Normally these stocks don't fall in lockstep, so there's a diversification")
    print("  discount on risk. Crises tend to erase that — everything falls together.\n")

def explain_backtest(breach_count, total_days, breach_rate, expected_rate=0.05):
    print(f"Backtest: {breach_count} breaches across {total_days} days, rate {breach_rate:.2%}.")
    diff = breach_rate - expected_rate
    if abs(diff) <= 0.015:
        print(f"  Close to the expected {expected_rate:.0%} — model looks well-calibrated.\n")
    elif diff > 0.015:
        print(f"  Higher than expected — a sign of fatter tails than the model assumes.\n")
    else:
        print(f"  Lower than expected — model may be a bit conservative here.\n")

# ============ TESTS ============
def make_fake_returns(n_days=500, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    data = rng.normal(loc=0.0005, scale=0.01, size=(n_days, 2))
    return pd.DataFrame(data, index=dates, columns=['FAKEA', 'FAKEB'])

def run_all_tests():
    returns = make_fake_returns()
    weights = {'FAKEA': 0.5, 'FAKEB': 0.5}
    port_returns = portfolio_returns_series(returns, weights)

    var_95 = historical_var(port_returns, 1_000_000, 0.95)
    var_99 = historical_var(port_returns, 1_000_000, 0.99)
    assert var_99 >= var_95
    print("PASS: test_var_increases_with_confidence")

    var, _ = parametric_var(returns, weights, 1_000_000, 0.95)
    cvar = parametric_cvar(returns, weights, 1_000_000, 0.95)
    assert cvar >= var
    print("PASS: test_cvar_exceeds_var")

    p_var, _ = parametric_var(returns, weights, 1_000_000, 0.95)
    mc_var, _ = monte_carlo_var(returns, weights, 1_000_000, 0.95, n_sim=200_000)
    assert abs(mc_var - p_var) / p_var < 0.05
    print("PASS: test_parametric_and_montecarlo_agree")

    total, _ = stress_scenario({'FAKEA': -0.10, 'FAKEB': -0.10}, weights, 1_000_000)
    assert abs(total - (-100_000)) < 1e-6
    print("PASS: test_stress_scenario_known_result")

    normal_var, _ = parametric_var(returns, weights, 1_000_000, 0.95)
    spiked_cov = correlation_spike_cov(returns, target_corr=0.85)
    spiked_var, _ = parametric_var(returns, weights, 1_000_000, 0.95, cov_override=spiked_cov)
    assert spiked_var >= normal_var
    print("PASS: test_correlation_spike_increases_risk")

    try:
        portfolio_returns_series(returns, {'FAKEA': 0.5, 'FAKEB': 0.6})
        assert False
    except ValueError:
        print("PASS: test_bad_weights_raise_error")

    print("\nAll tests passed.\n")

# ============ MAIN ============
def run():
    print("="*70)
    print("STEP 0: RUNNING UNIT TESTS ON SYNTHETIC DATA")
    print("="*70)
    run_all_tests()

    prices = fetch_prices(TICKERS, START_DATE)
    returns = compute_returns(prices)
    port_returns = portfolio_returns_series(returns, WEIGHTS)

    print("="*70)
    print(f"PORTFOLIO: ${PORTFOLIO_VALUE:,.0f} | Weights: {WEIGHTS}")
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} trading days)")
    print("="*70 + "\n")

    (prices / prices.iloc[0] * 100).plot(figsize=(10,5), title="Normalized Price Performance (base=100)")
    plt.show()

    print("Correlation matrix:")
    print(returns.corr().round(3), "\n")

    print("-"*70)
    print("PART A: VALUE AT RISK")
    print("-"*70 + "\n")

    results = []
    for cl in CONFIDENCE_LEVELS:
        hvar = historical_var(port_returns, PORTFOLIO_VALUE, cl)
        pvar, _ = parametric_var(returns, WEIGHTS, PORTFOLIO_VALUE, cl)
        mvar, _ = monte_carlo_var(returns, WEIGHTS, PORTFOLIO_VALUE, cl)
        cvar = parametric_cvar(returns, WEIGHTS, PORTFOLIO_VALUE, cl)

        print(f"--- {int(cl*100)}% Confidence ---")
        explain_var("Historical Simulation", hvar, PORTFOLIO_VALUE, cl)
        explain_var("Parametric (Normal)", pvar, PORTFOLIO_VALUE, cl)
        explain_var("Monte Carlo", mvar, PORTFOLIO_VALUE, cl)
        explain_cvar(cvar, pvar, PORTFOLIO_VALUE)

        results.append({'Confidence': f"{int(cl*100)}%", 'Historical': round(hvar,2),
                         'Parametric': round(pvar,2), 'MonteCarlo': round(mvar,2), 'CVaR': round(cvar,2)})

    results_df = pd.DataFrame(results).set_index('Confidence')
    print("=== Summary Table ===")
    print(results_df, "\n")

    plt.figure(figsize=(10,5))
    plt.hist(port_returns * PORTFOLIO_VALUE, bins=60, alpha=0.7, color='steelblue')
    for cl, color in zip(CONFIDENCE_LEVELS, ['orange','red']):
        hvar = historical_var(port_returns, PORTFOLIO_VALUE, cl)
        plt.axvline(-hvar, color=color, linestyle='--', linewidth=2, label=f'{int(cl*100)}% VaR = -${hvar:,.0f}')
    plt.legend(); plt.title("Portfolio Daily P&L — VaR marks the tail cutoff")
    plt.xlabel("Daily P&L ($)"); plt.ylabel("Frequency")
    plt.show()

    print("-"*70)
    print("PART B: STRESS TESTING")
    print("-"*70 + "\n")

    scenarios = {
        "2008-style shock (AAPL -8%, GOOGL -7%, MSFT -6%)": {'AAPL': -0.08, 'GOOGL': -0.07, 'MSFT': -0.06},
        "Tech drawdown (all -10%)": {'AAPL': -0.10, 'GOOGL': -0.10, 'MSFT': -0.10},
    }
    scenario_totals = {}
    for name, shocks in scenarios.items():
        total, _ = stress_scenario(shocks, WEIGHTS, PORTFOLIO_VALUE)
        scenario_totals[name] = total
        explain_scenario(name, total, PORTFOLIO_VALUE)

    plt.figure(figsize=(8,4))
    names, vals = list(scenario_totals.keys()), list(scenario_totals.values())
    bars = plt.bar(names, vals, color=['firebrick','darkorange'])
    plt.axhline(0, color='black', linewidth=0.8)
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v, f"${v:,.0f}", ha='center',
                  va='top' if v<0 else 'bottom')
    plt.title("Stress Scenario Impact"); plt.xticks(rotation=15, ha='right')
    plt.tight_layout(); plt.show()

    stressed_cov = correlation_spike_cov(returns, 0.85)
    print("--- Correlation Spike Scenario ---")
    for cl in CONFIDENCE_LEVELS:
        normal_var, _ = parametric_var(returns, WEIGHTS, PORTFOLIO_VALUE, cl)
        spiked_var, _ = parametric_var(returns, WEIGHTS, PORTFOLIO_VALUE, cl, cov_override=stressed_cov)
        explain_correlation_spike(normal_var, spiked_var)

    print("-"*70)
    print("BONUS: ROLLING VaR BACKTEST")
    print("-"*70 + "\n")

    bt = rolling_var_backtest(port_returns, 250, 0.95)
    breach_count = bt['breach'].sum()
    total_days = len(bt)
    breach_rate = breach_count / total_days
    explain_backtest(breach_count, total_days, breach_rate)

    plt.figure(figsize=(11,5))
    plt.plot(bt.index, -bt['actual_return']*PORTFOLIO_VALUE, label='Actual daily loss', color='steelblue', linewidth=1)
    plt.plot(bt.index, bt['predicted_var']*PORTFOLIO_VALUE, label='Rolling VaR(95%)', color='black', linestyle='--')
    breaches = bt[bt['breach']]
    plt.scatter(breaches.index, -breaches['actual_return']*PORTFOLIO_VALUE, color='red', zorder=5,
                label=f'Breaches ({len(breaches)})')
    plt.legend(); plt.title("Rolling VaR Backtest"); plt.tight_layout(); plt.show()

    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("1. VaR only marks a threshold — it never tells you HOW bad the tail could get.")
    print("2. Parametric/Monte Carlo assume Normal returns — real markets have fatter tails.")
    print("3. Correlations spike in crises, eroding diversification when you need it most.")

run()
