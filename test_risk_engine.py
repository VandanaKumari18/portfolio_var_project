import numpy as np
import pandas as pd

from risk import historical_var, parametric_var, parametric_cvar, monte_carlo_var
from scenarios import stress_scenario, correlation_spike_cov, rolling_var_backtest
from data import portfolio_returns_series

def make_fake_returns(n_days=500, seed=1):
    """Synthetic returns for 2 fake stocks — deterministic, so tests are reproducible."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    data = rng.normal(loc=0.0005, scale=0.01, size=(n_days, 2))
    return pd.DataFrame(data, index=dates, columns=['FAKEA', 'FAKEB'])


def test_var_increases_with_confidence():
    """A 99% VaR should always be >= 95% VaR — higher confidence means a further tail."""
    returns = make_fake_returns()
    weights = {'FAKEA': 0.5, 'FAKEB': 0.5}
    port_returns = portfolio_returns_series(returns, weights)

    var_95 = historical_var(port_returns, 1_000_000, 0.95)
    var_99 = historical_var(port_returns, 1_000_000, 0.99)
    assert var_99 >= var_95, f"Expected 99% VaR >= 95% VaR, got {var_99} < {var_95}"
    print("PASS: test_var_increases_with_confidence")


def test_cvar_exceeds_var():
    """CVaR (average tail loss) must always be >= VaR (tail threshold) by definition."""
    returns = make_fake_returns()
    weights = {'FAKEA': 0.5, 'FAKEB': 0.5}
    var, _ = parametric_var(returns, weights, 1_000_000, 0.95)
    cvar = parametric_cvar(returns, weights, 1_000_000, 0.95)
    assert cvar >= var, f"Expected CVaR >= VaR, got CVaR={cvar} < VaR={var}"
    print("PASS: test_cvar_exceeds_var")


def test_parametric_and_montecarlo_agree():
    """Since Monte Carlo simulates from the same mean/cov as Parametric, they
    should land within a small tolerance of each other."""
    returns = make_fake_returns()
    weights = {'FAKEA': 0.5, 'FAKEB': 0.5}
    p_var, _ = parametric_var(returns, weights, 1_000_000, 0.95)
    mc_var, _ = monte_carlo_var(returns, weights, 1_000_000, 0.95, n_sim=200_000)
    pct_diff = abs(mc_var - p_var) / p_var
    assert pct_diff < 0.05, f"Parametric and Monte Carlo differ by {pct_diff:.1%}, expected < 5%"
    print("PASS: test_parametric_and_montecarlo_agree")


def test_stress_scenario_known_result():
    """A known, hand-calculable shock should match manual arithmetic exactly."""
    weights = {'FAKEA': 0.5, 'FAKEB': 0.5}
    shocks = {'FAKEA': -0.10, 'FAKEB': -0.10}
    total, _ = stress_scenario(shocks, weights, 1_000_000)
    expected = -100_000  # -10% on the whole $1M, since both halves shocked equally
    assert abs(total - expected) < 1e-6, f"Expected {expected}, got {total}"
    print("PASS: test_stress_scenario_known_result")


def test_correlation_spike_increases_risk():
    """Forcing correlation to 0.85 should never DECREASE portfolio VaR vs the
    natural (lower) correlation — more co-movement means less diversification."""
    returns = make_fake_returns()
    weights = {'FAKEA': 0.5, 'FAKEB': 0.5}
    normal_var, _ = parametric_var(returns, weights, 1_000_000, 0.95)
    spiked_cov = correlation_spike_cov(returns, target_corr=0.85)
    spiked_var, _ = parametric_var(returns, weights, 1_000_000, 0.95, cov_override=spiked_cov)
    assert spiked_var >= normal_var, "Correlation spike should not reduce VaR"
    print("PASS: test_correlation_spike_increases_risk")


def test_bad_weights_raise_error():
    """Weights that don't sum to 1 should fail loudly, not silently give wrong numbers."""
    returns = make_fake_returns()
    bad_weights = {'FAKEA': 0.5, 'FAKEB': 0.6}  # sums to 1.1
    try:
        portfolio_returns_series(returns, bad_weights)
        assert False, "Expected ValueError for weights not summing to 1"
    except ValueError:
        print("PASS: test_bad_weights_raise_error")


def run_all_tests():
    test_var_increases_with_confidence()
    test_cvar_exceeds_var()
    test_parametric_and_montecarlo_agree()
    test_stress_scenario_known_result()
    test_correlation_spike_increases_risk()
    test_bad_weights_raise_error()
    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all_tests()
