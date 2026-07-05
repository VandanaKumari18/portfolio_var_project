import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import fetch_prices, compute_returns, portfolio_returns_series
from risk import historical_var, parametric_var, parametric_cvar, monte_carlo_var
from scenarios import stress_scenario, correlation_spike_cov, rolling_var_backtest
from explain import (explain_var, explain_cvar, compare_methods,
                      explain_scenario, explain_correlation_spike, explain_backtest)

TICKERS = ['AAPL', 'GOOGL', 'MSFT']
WEIGHTS = {'AAPL': 0.40, 'GOOGL': 0.35, 'MSFT': 0.25}
PORTFOLIO_VALUE = 1_000_000
START_DATE = '2022-01-01'
CONFIDENCE_LEVELS = [0.95, 0.99]

def run():
    prices = fetch_prices(TICKERS, START_DATE)
    returns = compute_returns(prices)
    port_returns = portfolio_returns_series(returns, WEIGHTS)

    print("="*70)
    print(f"PORTFOLIO: ${PORTFOLIO_VALUE:,.0f} | Weights: {WEIGHTS}")
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} trading days)")
    print("="*70 + "\n")

    (prices / prices.iloc[0] * 100).plot(figsize=(10,5), title="Normalized Price Performance (base=100)")
    plt.show()

    print("📈 Correlation matrix (1.0 = always move together, 0 = independent):")
    print(returns.corr().round(3), "\n")

    print("─"*70)
    print("PART A: VALUE AT RISK (statistical estimate on a 'normal' bad day)")
    print("─"*70 + "\n")

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
    for cl in CONFIDENCE_LEVELS:
        compare_methods(results_df, f"{int(cl*100)}%")

    plt.figure(figsize=(10,5))
    plt.hist(port_returns * PORTFOLIO_VALUE, bins=60, alpha=0.7, color='steelblue')
    for cl, color in zip(CONFIDENCE_LEVELS, ['orange','red']):
        hvar = historical_var(port_returns, PORTFOLIO_VALUE, cl)
        plt.axvline(-hvar, color=color, linestyle='--', linewidth=2,
                                           label=f'{int(cl*100)}% VaR = -${hvar:,.0f}')
    plt.legend(); plt.title("Portfolio Daily P&L — VaR marks the tail cutoff")
    plt.xlabel("Daily P&L ($")
    plt.ylabel("Frequency")
    plt.show()

    print("\n" + "─"*70)
    print("PART B: STRESS TESTING (deterministic 'what if this happens' scenarios)")
    print("─"*70 + "\n")

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
    plt.title("Stress Scenario Impact")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()

    stressed_cov = correlation_spike_cov(returns, 0.85)
    print("\n--- Correlation Spike Scenario ---")
    for cl in CONFIDENCE_LEVELS:
        normal_var, _ = parametric_var(returns, WEIGHTS, PORTFOLIO_VALUE, cl)
        spiked_var, _ = parametric_var(returns, WEIGHTS, PORTFOLIO_VALUE, cl, cov_override=stressed_cov)
        explain_correlation_spike(normal_var, spiked_var)

    print("─"*70)
    print("BONUS: ROLLING VaR BACKTEST (is our model actually reliable historically?)")
    print("─"*70 + "\n")

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
    plt.legend()
    plt.title("Rolling VaR Backtest")
    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS FOR DISCUSSION")
    print("="*70)
    print("1. VaR only marks a threshold — it never tells you HOW bad the tail could get.")
    print("   That's why we also compute Expected Shortfall and run stress tests.")
    print("2. Parametric/Monte Carlo assume Normal returns — real markets have fatter tails,")
    print("   so these can understate true risk (the backtest checks this directly).")
    print("3. Correlations aren't stable — they spike in crises, eroding diversification")
    print("   exactly when you need it most.")

if __name__ == "__main__":
    run()
