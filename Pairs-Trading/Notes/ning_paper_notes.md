# Ning Dissertation Summary

---

## 1. Introduction

Explains the foundations of arbitrage and stat arb.
- Arbitrage enforces price efficiency (“no free lunch”).
- Statistical arbitrage exploits mean-reverting spreads between correlated assets (eg. pairs trading).
- Process: identify co-moving securities -> construct spreads -> design entry/exit trading rules.
- Reviews classic methods:  
  - Distance Method (Gatev)  
  - Cointegration approaches (Engle–Granger, Johansen),  
  - Stochastic spread models (Ornstein–Uhlenbeck, CIR),  
  - Modern extensions using machine learning, stochastic control, copulas, PCA, and reinforcement learning.  
- Identifies gaps: existing work focuses on single-pair analysis and assumes model structure. The dissertation generalizes to multiple pairs and model-free timing.

---

## 2. Diversification Framework for Multiple Pairs Trading

Proposes a multi-pair trading and portfolio allocation framework.
- **Goal:** Extend pairs trading to multiple uncorrelated spreads to enhance diversification and returns.  
- **Methodology:**
  1. Fit Ornstein–Uhlenbeck (OU) models to each spread with MLE to obtain mean reversion rate ($\mu$), volatility ($\sigma$), and long-term mean ($\theta$).
  2. Simulate trading behavior under various OU parameters to study performance sensitivity to $\mu$ and $\sigma$.
  3. Develop allocation rules:
     - Mean-Variance Analysis (MVA): maximize historical Sharpe ratio.
     - Mean Reversion Budgeting (MRB): weight pairs by normalized mean reversion speed and OU likelihood fit.
     - Mean Reversion Ranking (MRR): rank pairs by mean reversion strength to avoid concentration.
  4. Implement rebalancing stages and backtest 
- **Results:**  
  - MRB improved returns significantly;  
  - MRR achieved highest Sharpe ratio with reduced volatility;  
  - Both outperform equal-weight and mean-variance benchmarks.  
- **Conclusion:** Incorporating mean reversion characteristics into portfolio design yields superior diversification and stability.

---

## 3. Optimal Entry and Exit with Signature Method

Introduces a Signature Optimal Stopping approach using rough path theory.
- **Overview:** The *signature* of a path is a sequence of iterated integrals that encode its shape; used to model temporal patterns without assuming specific dynamics.
- **Framework:**
  - Formulate sequential optimal stopping to determine both entry and exit times for mean-reverting spreads.
  - Replace traditional parametric models with a data-driven signature representation.
  - Estimate stopping rules by learning mappings from path signatures to stopping decisions.
- **Experiments:**
  - Simulated OU spreads validate accuracy vs analytical optima.
  - Real-market backtests show higher cumulative returns and better timing precision than conventional threshold-based mean reversion trading.
- **Conclusion:** Signature-based stopping captures non-Markovian features and generalizes to unknown dynamics, outperforming standard rule-based strategies.

---

## 4. Advanced Statistical Arbitrage with Reinforcement Learning

Develops a model-free RL trading framework.
- Phase 1: Spread Construction
  - Define Empirical Mean Reversion Time (EMRT) to measure how fast a spread reverts empirically.
  - Optimize asset weights to minimize EMRT—yielding faster mean-reverting synthetic spreads.
- Phase 2: Trading via Reinforcement Learning
  - Formulate the problem as a Markov Decision Process:
    - State encodes recent spread and trend information.
    - Actions = trading decisions (long/short/hold).
    - Rewards = PnL adjusted for mean reversion characteristics.
  - Train the RL agent to maximize long-term profit using simulated and real data.
- **Results:**  
  - On simulated OU data, RL achieved higher Sharpe ratios and smoother returns.  
  - On real data, it outperformed both simple OU and cointegration baselines.  
- **Conclusion:** RL provides flexible, adaptive decision-making for mean-reversion trading without assuming parametric spread dynamics.

---

## 5. Summary and Contributions

- **Three key innovations:**
  1. Diversification and dynamic allocation among multiple mean-reverting pairs.
  2. Signature-based optimal stopping for data-driven timing decisions.
  3. Reinforcement learning framework for model-free statistical arbitrage.
- **Future directions:**  
  - Extending signature and RL methods to multi-asset and high-frequency settings.  
  - Integrating transaction costs, market impact, and regime-switching dynamics.  
