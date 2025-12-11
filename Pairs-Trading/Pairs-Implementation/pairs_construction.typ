= Part 1. Pairs Construction 
Create a single time series that combines two assets so that it can be traded like a single instrument. 
The spread should be stationary and means reverting, meaning it drifts away from and returns to equilibrium predictably.


== 1.1. Spreads
A spread is defined as
$
  X_t = S^1_t - B dot S^2_t
$
where
- $X_t$ is the portfolio value
- $S^1_t, S^2_t$ are two highly correlated assets
- $B$ is the hedge ratio

The hedge ratio defines the spread.

A common way to estimate the hedge ratio is to select the value such that the spread resembles an Ornstein-Uhlenbeck process as much as possible.

#v(1em)
== 1.2. OU process

$
  "d"  X_t = mu (theta - X_t) "d"t + sigma "d" W_t
$
- $mu$: speed of mean reversion. How quickly deviations appear. Large $mu$ means trades close quickly.
- $theta$: long term mean of the spread
- $sigma$: volatility. Even if spreads revert, there's noise. High $sigma$ means spread is wide & high fluctuation.
- $W_t$; standard Brownian motion under historical measure $PP$

The OU process is a simple continuous time reverting model, with parameters that are directly relevant to stat-arb related features. By interpreting a spread as an OU process, a lot of critical information is gained.


#v(1em)
== 1.3. Parameter Estimation
Estimate OU parameters to choose the optimal hedge ratio estimator $hat(B)$.

1. Consider a list of possible hedge ratio values
$
(b_1, ..., b_n)
$

2. For each $b_i$, find the OU parameters that define the OU model that fits the data most closely

With MLE,
$
  (mu^*(b_i), theta^*(b_i), sigma^*(b_i)) = arg max_(mu, theta, sigma) l(mu, theta, sigma | X_t (b_i))\
  l^*(b_i) = l(mu^*(b_i), theta^*(b_i), sigma^*(b_i) | X_t (b_i))
$
where $l^*(b_i)$ is the best OU likelihood achieved under $b_i$.


4. Find the final estimate, $hat(B)$, by selecting the $b_i$ with the OU model with the highest likelihood
$
  hat(B) = arg max_b l^*(b)
$



== 1.4. MLE Implementation
1. Solve the OU process for discrete time steps.
An OU process has the continuous stochastic differential equation
$
  "d" X_t = mu (theta - X_t) "d" t + sigma "d" W_t
$

Spreads are observed at discrete times $t_0, t_1, ..., t_n$ with a fixed step $Delta t = t_i - t_(i-1)$.

Solving the SDE for discrete times,
$
  X_i = X_(i-1) dot e^(-mu Delta t) + theta (1 - e^(-mu Delta t)) + epsilon_i
$
where $epsilon$ is a normally distributed random variable with a mean of $0$ and variance
$
  delta_epsilon^2 = delta^2 (1 - e^(-2 mu Delta t))/(2 mu)
$


So, given $X_(i-1)$, the next value $X_i$ is normally distributed with
$
  "Mean:" m_i (mu, theta) = X_(i-1) dot e^(-mu Delta t) + theta (1 - e^(-mu Delta t))\
  "Variance:" delta_epsilon^2 (mu, sigma) = delta^2 (1 - e^(-2 mu Delta t))/(2 mu)\
  X_i | (X_(i-1), mu, theta, sigma) ~ "Normal"(m_i (mu, theta), delta_epsilon^2 (mu, sigma))
$

2. Under a specific hedge ratio, use MLE to estimate for the OU parameters. 

For a fixed hedge ratio $b_i$, form the spread series
$
  x_i = X_(t_i) (b_i)
$

For the past $L$ observations, the probability density of seeing the sequence $x_(t_0 - L+1), ..., x_t_0$ is
$
  product_(i= t_0 - L + 1)^(t_0) f(x_i | x_(i-1); mu, theta, sigma)\
$
Since each step is gaussian, the density for one step is 
$
  f(x_i | x_(i-1)) = 1/sqrt(2 pi sigma_epsilon^2) dot exp(- (x_i - m_i)^2 / (2 sigma_epsilon^2))
$
Log likelihood:
$
  l(mu, theta, sigma, b) &= 1/L sum_(i = t_0 - L + 1)^(t_0 -1) log f(x_i | x_(i-1); mu, theta, sigma)\
  &= - 1/2 ln(2 pi) - ln(sigma_epsilon) - 1/(2 L sigma_epsilon^2) sum_(i=t-L+1)^(t_0) [x_i - x_(i-1) e^(-mu Delta t) - theta(1 - e^(-mu Delta t))]^2
$

#pagebreak()
Focus on the residual sum:
$
  S := sum_i [x_i - x_(i-1) e^(-mu Delta t) - theta(1 - e^(-mu Delta t))]^2
$
Let $phi.alt = e^(-mu Delta t)$ and $c = 1 - phi.alt$.
$
  S &= sum_i [x_i - phi.alt x_(i-1) - theta c]^2\
  &= sum_i [x_i^2 - 2 phi.alt x_i x_(i-1) - 2 theta c x_i + phi.alt^2 x_(i-1)^2 + 2 phi.alt theta c x_(i-1) + theta^2 c^2]
$

This yields several sufficient statistics.
$
  &X_x := sum x_(i-1) &&X_y := sum x_i\
  &X_(x x) := sum x_(i-1)^2#h(3em) &&X_(y y) := sum x_i^2 #h(3em) X_(x y) := sum x_i  x_(i-1)
$

The series can be evaluated with these sufficient statistics.
$
  l(mu, theta, sigma, b) = -ln(sigma_epsilon) - 1/(2 L sigma_epsilon^2) S(phi.alt, theta; X_x, X_y, X_(x x), X_(y y), X_(x y))
$

Optimal parameters are given by:
$
  theta^* &= (X_y X_(x x) - X_x X_(x y))/(n (X_(x x) - X_(x y)) - (X_x^2 - X_x X_y))\
  mu^* &= -1/(Delta t)  ln (X_(x y) - theta^* X_x - theta^* X_y + n(theta^*)^2)/(X_(x x) - 2 theta^* X_x + n(theta^*)^2)\
  (sigma^*)^2 &= (2 mu^*)/(n (c^*)) dot 
  (X_(y y) - 2phi.alt^2 X_(x y) + phi.alt^2 X_(x x) - 2 theta^* c^* (X_y - phi.alt X_x) + n(theta^*)^2 c^*^2)
$

3. Find the maximized average log-likelihood.

Maximize the average log-likelihood over $(mu, theta, sigma)$ for the specific $b_i$.
$
  l^*(b_i) = l(mu^*, theta^*, sigma^*, b^*)
$

4. Compare the maximized average log-likelihood over all $b$'s.

$
  B = arg max_b l^*(b)
$

The maximizer $B$ is the hedge ratio that best fits the spread to an OU process. 

In his paper, Ning suggests doing a grid search over values of $b$, computing the maximized log-likelihood function for each.
$
  b = {-2, -1.99, ..., 1.99, 2}
$