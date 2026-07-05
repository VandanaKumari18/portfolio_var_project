
def explain_var(method_name, var_dollar, portfolio_value, confidence):
    pct = var_dollar / portfolio_value * 100
    bad_days = round((1 - confidence) * 100)
    print(f"{method_name} — {int(confidence*100)}% VaR: ${var_dollar:,.0f} ({pct:.2f}% of the book)")
    print(f"  In other words: out of 100 typical trading days, only about {bad_days} of them")
    print(f"  should produce a loss bigger than this. It's a threshold, not a ceiling.\n")


def explain_cvar(cvar_dollar, var_dollar, portfolio_value):
    pct = cvar_dollar / portfolio_value * 100
    print(f"Expected Shortfall / CVaR: ${cvar_dollar:,.0f} ({pct:.2f}%)")
    print(f"  This is the average loss you'd see if you're already past the VaR line — always")
    print(f"  worse than VaR (${var_dollar:,.0f}) since it's conditioned on being in the bad tail.\n")


def compare_methods(results_df, confidence_label):
    row = results_df.loc[confidence_label]
    spread = row['MonteCarlo'] - row['Parametric']
    pct_diff = abs(spread) / row['Parametric'] * 100
    print(f"At {confidence_label}, Parametric and Monte Carlo land within {pct_diff:.1f}% of each other.")
    if pct_diff < 5:
        print("  That's expected — Monte Carlo is just simulating from the same mean/covariance")
        print("  that Parametric uses in closed form, so close agreement is a decent internal check.\n")
    else:
        print("  Bigger gap than expected — worth rerunning with more simulations before trusting it.\n")


def explain_scenario(name, total, portfolio_value):
    pct = total / portfolio_value * 100
    print(f"{name}: ${total:,.0f} ({pct:.2f}%)")
    print(f"  If this shock landed today, the $1,000,000 book would be worth")
    print(f"  ${portfolio_value + total:,.0f} by the close. No probability attached here —")
    print(f"  just a fixed 'what if' calculation.\n")


def explain_correlation_spike(normal_var, spiked_var):
    increase_pct = (spiked_var / normal_var - 1) * 100
    print(f"Correlation spike: VaR moves from ${normal_var:,.0f} to ${spiked_var:,.0f}, "
          f"about {increase_pct:.1f}% worse.")
    print("  Normally these 3 stocks don't fall in lockstep, so there's some built-in diversification")
    print("  discount on risk. Crises tend to erase that — everything starts moving together —")
    print("  which is exactly what this scenario is simulating.\n")


def explain_backtest(breach_count, total_days, breach_rate, expected_rate=0.05):
    print(f"Backtest: {breach_count} breaches across {total_days} days, a rate of {breach_rate:.2%}.")
    diff = breach_rate - expected_rate
    if abs(diff) <= 0.015:
        print(f"  That's close enough to the {expected_rate:.0%} we'd expect from a 95% VaR model —")
        print("  no strong evidence the model is miscalibrated on this window.\n")
    elif diff > 0.015:
        print(f"  That's higher than the {expected_rate:.0%} we'd expect. Actual losses are breaching")
        print("  the VaR line more often than the Normal-distribution assumption predicts — a sign")
        print("  of fatter tails in the real data than the model accounts for.\n")
    else:
        print(f"  That's lower than the {expected_rate:.0%} we'd expect, so if anything the model")
        print("  looks a bit conservative over this particular window.\n")
