= Strategy Example

Making a market on product XYZ.

=== 1. Observe current market state

Note down:
- Best bid/ask ($b_t, a_t$)
- Midprice ($m_t = (a_t + b_t) \/ 2$)
- Spread ($s_t = a_t - b_t$)

Record relevant microstructure features:
- trade imbalance
- short term volatility
- depth

=== 2. Estimate Relative Values
Use the stat arb model to compare XYZ to a related basket (eg. other stocks in ETF, industry, etc.)

Compute mispricing residual and standardized z-score
$
  r_t = P_"XYZ" - (alpha + beta P_"basket")\
  z_t = (r_t - mu) / sigma
$
- Estimate $alpha$ (baseline difference between prices) and $beta$ (hedge ratio) using some regression method on historical data

Interpret:
- $z_t > 0$: product looks overvalued, likely to mean-revert downwards
- $z_t < 0$: product looks undervalued, likely to mean-revert upwards


=== 3. Translate signal into quoting stance
Adjust fair value estimate
$
  hat(p)_t = m_t - k_p dot z_t
$
where $k_p$ is a constant that controls how aggresively we act on the signal

Adjust spread:
- Start from observed spread $s_t$
- Compute bid ask
$
  "bid" = hat(p) - s_t/2, 
  #h(3em)
  "ask" = hat(p) + s_t/2
$

Skew spread towards signal direction:
- If $z_t > 0$ (overvalued): widen ask, tighten bid (encourages selling inventory)
- If $z_t < 0$ (undervalued): tighten ask, widen bid (accumulates inventory)

