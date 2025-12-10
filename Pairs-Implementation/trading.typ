= Part 2. Trading
Given hedge ratio $hat(B)$, how to trade the spread $X_t = S^1_t - hat(B) dot S^2_t$.

When the spread deviates far from its recent mean, bet that it will revert.

Define components & indicators:
- $C_t:=$ Capital available at time
- $P O S(S^i_t) :=$ current position of asset $i$ in the spread
- $M A(X_t) :=$ $M$-day moving average of spread
- $S D(X_t) :=$ Standard deviation of spread over past $M$ days
- $K :=$ Threshold parameter. How many SDs away from MA are required to enter a trade
- $r :=$ Stop-loss parameter. Defines acceptable adverse move from entry price before forcing exit

== 2.1. Trading Procedure
The spread is treated like a single instrument.

1. Estimate $hat(B)$ at time $0$

Use past $L$ days of $S^1$ and $S^2$ to compute
$
  hat(B) = arg max_b l^*(b)
$
This $hat(B)$ is fixed for the upcoming trading period.

2. Compute the spread $X_t = S^1_t - hat(B) dot S^2_t$ at each $t$, and track $M A(X_t)$ and $S D(X_t)$

3. Entry & Exit

*Entry*:

If spread is significantly low, then expect reversion upwards and enter a long position on the spread, which is long on asset $1$ and short on asset $2$:
$
  &P O S(X_t) = 0 " and " X_t < M A(X_t) - K dot S D(X_t) \
  &==>P O S(S^1_t) = C_t / S^1_t " and " P O S(S^2_t) = - hat(beta) dot P O S (S^1_t)
$

If spread is significantly high, then expect reversion downwards, and enter a short position on the spread, which is short on asset $1$ and long on asset $2$:
$
  &P O S(X_t) = 0 " and " X_t > M A(X_t) - K dot S D(X_t) \
  &==>P O S(S^1_t) = -C_t / S^1_t " and " P O S(S^2_t) = - hat(beta) dot P O S (S^1_t)
$

*Exit*:

If a long/short spread reverts back to equilibrium, reset positions and update capital:
$
  &P O S(X_t) > 0 " and " X_t > M A (X_t) "(for long position)"\
  &P O S(X_t) < 0 " and " X_t < M A (X_t) "(for short position)"\
  &==> C_t = C_t + X_t dot P O S (X_t) " and "P O S (X_t, S^1_t, S^2_t) = 0
$

#pagebreak()
*Stop-Loss*:

Protects against large deviations in spread before mean reversion occurs.

In a long/short position, when the spread deviates too far below/above entry, cut the loss:
$
  &P O S(X_t) > 0 " and " X_t < X_0 (1-r) "(for long position)"\
  &P O S(X_t) < 0 " and " X_t < X_0 (1+r) "(for short position)"\
  &==> C_t = C_t + X_t dot P O S (X_t) " and "P O S (X_t, S^1_t, S^2_t) = 0
$

*Stay Put*:

If no entry or exit condition is met, hold the position:
$
  P O S(X_t) = P O S(X_t) " and " C_t = C_t
$