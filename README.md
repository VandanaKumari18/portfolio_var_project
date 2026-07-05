# Portfolio Risk Engine — VaR + Stress Testing

## What is this?

This is a small tool I built to answer one practical question: **"If I had $1,000,000
invested in Apple, Google, and Microsoft, how much could I realistically lose on a bad
day — and how bad could my worst days actually get?"**

There are two parts to the answer:

1. **Value at Risk (VaR)** — a statistical estimate of "normal" bad-day losses, calculated
   three different ways so I could cross-check the answer instead of trusting one method blindly.
2. **Stress Testing** — instead of statistics, I asked "what if a specific crisis hit today?"
   and calculated the exact dollar damage.

## How to run it
pip install -r requirements.txt
python main.py

Or in Google Colab: run each file cell in order (`data.py` → `risk.py` → `scenarios.py`
→ `explain.py` → `main.py`), then run `main.py`. To check the tests pass:
python test_risk_engine.py


## What's inside

| File | What it does |
|---|---|
| `data.py` | Pulls stock prices from Yahoo Finance and turns them into daily returns |
| `risk.py` | The three VaR methods (Historical, Parametric, Monte Carlo) plus Expected Shortfall |
| `scenarios.py` | Stress scenarios, a correlation-crisis simulation, and a backtest |
| `explain.py` | Turns raw numbers into plain-language explanations |
| `main.py` | Runs everything end-to-end and prints/plots the full report |
| `test_risk_engine.py` | Automated checks that prove the math behaves correctly on known inputs |

## What I actually found

Running this on ~4.5 years of real data (Jan 2022 – Jul 2026):

- **At 95% confidence, all three VaR methods agreed closely** — around $25,000-26,000
  a day. When three independent approaches land in the same place, that's a strong signal
  the underlying math and data are sound, not a coincidence.
- **At 99% confidence, the methods disagreed more** — Historical VaR came out to about
  $40,000, noticeably higher than Parametric/Monte Carlo (~$36,000-36,600). This wasn't
  a bug — it's the real market showing "fatter tails" than a clean bell-curve model expects.
  Extreme days happen more often in reality than a Normal distribution predicts.
- **Expected Shortfall was always meaningfully higher than VaR** (e.g. $41,962 vs $36,627
  at 99%) — exactly as it should be, since it measures the average loss *inside* the worst-case
  tail, not just where that tail begins.
- **A 2008-style shock (-8%/-7%/-6%) would cost about $71,500.** A flat 10% tech drawdown
  across all three stocks would cost exactly $100,000 — a useful gut-check that the math
  lines up with plain arithmetic.
- **Forcing correlations up to 0.85 (simulating a crisis where everything falls together)
  increased VaR by about 12.6%** at both confidence levels. This is the part I find most
  interesting: my portfolio's built-in diversification is worth roughly 12.6% of risk
  reduction — and that protection shrinks exactly when markets are most stressed.
- **I backtested the model against real history**, not just theory: rolling a 250-day
  VaR window forward and checking how often actual losses broke through it. Result: 43
  breaches out of 877 days, a 4.9% breach rate — almost exactly the 5% you'd expect from
  a well-calibrated 95% VaR model. That's the strongest evidence in this project that the
  approach actually works, not just that the formulas are typed in correctly.

## Why I built it this way

I split the code into small, single-purpose files instead of one long script, so each
piece — fetching data, calculating VaR, running scenarios — can be tested and understood
on its own. I also wrote automated tests using synthetic data (not live market data), so
the logic can be verified instantly without depending on an internet connection or the
market being open.

## What this model can't tell you (and I think that matters)

- It assumes the recent past is a decent guide to tomorrow — it isn't always.
- Two of the three VaR methods assume returns follow a clean bell curve. Real markets
  don't fully cooperate with that assumption, especially in crashes.
- VaR only tells you where a "bad day" threshold sits — it doesn't tell you how bad
  the very worst days could get. That's exactly why I added Expected Shortfall and
  stress testing alongside it, rather than relying on VaR by itself.
- Correlations between stocks aren't fixed — they can spike during real crises, which
  is what the correlation-spike scenario is designed to expose.

I think a risk report that quietly hides these limitations is more dangerous than one
that states them plainly — so I built this to show its own edges, not just its output.
