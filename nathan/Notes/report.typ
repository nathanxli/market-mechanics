= Introduction: What Statistical Arbitrage Is (and Isn’t)

Statistical arbitrage (stat arb) exploits predictable statistical regularities in relationships among securities. Instead of betting on the direction of one asset, we trade the relationship (pairs, baskets, factors). The thesis: prices sometimes temporarily diverge from an equilibrium relationship and then mean‑revert.

Key contrasts with classical (riskless) arbitrage:
- Classical arbitrage locks in a riskless profit with identical payoffs performed simultaneously.  
- Statistical arbitrage accepts residual risk, aiming for positive expected profit from mean reversion or relative‑value mispricings.

Alpha sketch:

Profit ≈ (speed of convergence) × (position size) − costs − risk penalties ]

Historical note: institutional quant teams in the 1980s–1990s popularized convergence trades; modern variants persist across horizons, often market‑neutral or embedded in higher‑frequency frameworks.

= Mathematical Foundation

== Correlation vs. Cointegration

Correlation measures short‑run co‑movement; it does not guarantee a stable long‑run relationship. Cointegration asserts that a linear combination of nonstationary prices is stationary. For two assets with log‑prices X_t and Y_t, consider:

#equation( X_t = α + β · Y_t + ε_t )

with ε_t stationary. If this holds, deviations from the mean of ε_t tend to decay — creating a mean‑reversion trade.

== Spread Construction and Normalization

After estimating α and β, define the spread and a standardized z‑score:

#equation( spread_t = X_t − (α + β · Y_t) )
#equation( z_t = ( spread_t − mean(spread) ) / stdev(spread) )

A basic rule: enter when |z_t| > θ and exit when z_t reverts toward 0.

== Mean‑Reversion Dynamics (OU Process)

A common continuous‑time model for a stationary spread S_t is the Ornstein–Uhlenbeck (OU) process:

#equation( dS_t = κ ( μ − S_t ) dt + σ dW_t )

Here μ is the mean level, κ > 0 the speed, and σ the volatility. Larger κ implies faster expected reversion (shorter holding periods). Discretizations allow parameter estimation from time series.

== Hypothesis‑Testing View

Each trade is a small hypothesis test on a deviation:

#equation( H_0: spread within equilibrium noise )
#equation( H_1: spread deviates significantly )

Thresholds (for example, |z_t| > θ) balance false positives (overtrading) versus missed opportunities; cross‑validation or walk‑forward testing can select θ.

= Varieties of Statistical Arbitrage

== Classical Pairs Trading

- Identify candidate pairs within sectors or economic peers.  
- Test cointegration; estimate hedge ratio β.  
- Trade on z‑score bands; target market‑ and beta‑neutral sizing.

== Multi‑Asset or Basket Relative Value

Model an asset’s return with factors, peers, or ETFs, and trade residuals as mean‑reverting signals:

#equation( R_(i,t) = α_i + ∑_k β_(i,k) F_(k,t) + ε_(i,t) )

Portfolio constraints can enforce neutrality (market, sector, factor loadings).

== Factor‑Neutral or Style‑Neutral Stat Arb

Within a defined universe (industry, size bucket), rank on residuals or valuation spreads. Go long undervalued, short overvalued; neutralize exposures (market, size, value, momentum).

== ML‑Infused Residual Trading

Replace linear spread modeling with nonlinear predictions of residuals or reversion horizons (trees and boosting, simple RNN or LSTM). Use leakage‑safe validation (purged and embargoed cross‑validation) and maintain neutrality.

= Implementation Considerations

== Data Hygiene and Preprocessing

- Synchronized timestamps; adjust for splits, dividends, and delistings.  
- Prefer log‑prices or returns; winsorize outliers when justified.  
- Avoid survivorship bias (include delisted names when possible).

== Backtesting Pipeline (Minimal)

1. Signal generation: estimate spread, compute z_t, compare to thresholds.  
2. Portfolio construction: size legs to target neutrality (dollar‑ or beta‑neutral).  
3. Execution model: apply transaction costs and slippage; respect liquidity.  
4. PnL attribution: mark‑to‑market; compute cumulative PnL.

Core metrics: Sharpe, Sortino, hit rate, average holding period, turnover, max drawdown, and capacity (PnL decay versus size).

== Risk Management

- Neutrality: target portfolio beta ≈ 0 and bounded factor exposures.  
- Stops: exit if |z_t| or loss exceeds limits, or if max holding time is breached.  
- Sizing: inverse‑volatility weighting or conservative Kelly‑style fractions.

== Costs, Slippage, and Capacity

- Explicit fees plus implicit slippage erode thin edges; model both.  
- Capacity limits: stronger impact for illiquid names or larger size.  
- Regime risk: relationships can break; use rolling tests and change‑point detection.

= Historical Context and Real‑World Notes

- 1990s–2000s: strong returns; profitability thinned post‑2008.  
- Major quant firms (Renaissance, D. E. Shaw, Two Sigma, etc.) still employ relative‑value frameworks, often with richer microstructure features.  
- Cautionary tale: leveraged convergence trades can fail when correlations spike or cointegration regimes shift (for example, LTCM in 1998).

= Discussion Prompts (for the Meeting)

- What constitutes a statistical edge beyond anecdotal correlation?  
- When is correlation adequate, and when is cointegration essential?  
- How do we estimate thresholds θ without overfitting?  
- Which horizons (intra‑day versus multi‑day) best suit pairs trading for our data access?  
- How do we validate neutrality and control hidden factor bets?  
- Where could ML genuinely improve the spread model versus add noise and overfit?

= References and Further Reading

- Gatev, Goetzmann, Rouwenhorst (2006) — Pairs Trading: Performance of a Relative‑Value Arbitrage Rule.  
- Vidyamurthy (2004) — Pairs Trading: Quantitative Methods and Analysis.  
- Pole (2007) — Statistical Arbitrage: Algorithmic Trading Insights and Techniques.  
- Elliott, van der Hoek, Malcolm (2005) — Pairs Trading.

#pagebreak()

= Appendix: Minimal Pair‑Trader Sketch (Pseudo‑Algorithm)

1. Select candidate pairs (sector or peer based).  
2. Fit X_t = α + β · Y_t + ε_t on a rolling window.  
3. Compute z_t from ε_t.  
4. Enter long‑spread if z_t < −θ (long X, short β · Y); enter short‑spread if z_t > θ (short X, long β · Y).  
5. Exit when |z_t| < θ_exit or when max holding time is reached.  
6. Track PnL with costs; monitor neutrality, drawdown, and turnover.
