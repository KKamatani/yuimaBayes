
## 1) Build the CIR model
#   dX_t = κ(θ − X_t) dt + σ √X_t dW_t
#     • κ, θ, σ > 0
#   Note: This uses a Gaussian Euler transition; positivity is encouraged by
#   bounds/priors but not hard-enforced by the likelihood approximation.

mod <- setModel(
  drift          = "kappa*(theta - x)",
  diffusion      = "sigma*sqrt(x)",
  time.variable  = "t",
  state.variable = "x",
  solve.variable = "x"
)

## 2) Sampling scheme: 250 points on [0, 2]
yui <- setYuima(
  model    = mod,
  sampling = setSampling(Terminal = 2, n = 250)
)

## 3) Simulate “observed” data with known parameters
set.seed(42)
sim <- simulate(
  yui,
  xinit          = 0.5,  # must be > 0
  true.parameter = list(
    kappa = 3.0,
    theta = 1.2,
    sigma = 0.4
  )
)

## 4) Bayesian estimation via Stan (bounds + priors)
bounds <- list(
  kappa = c(0, Inf),
  theta = c(0, Inf),
  sigma = c(0, Inf)
)

priors <- list(
  # Half-t priors via lower=0 bounds
  kappa = "student_t(3, 0, 2)",
  theta = "student_t(3, 0, 2)",
  sigma = "student_t(3, 0, 0.5)"
)

fit <- bayes(
  sim,
  bounds        = bounds,
  priors        = priors,
  default_prior = NULL,  # only the priors above are applied
  chains        = 4,
  iter          = 2000
)

print(fit@fit)
