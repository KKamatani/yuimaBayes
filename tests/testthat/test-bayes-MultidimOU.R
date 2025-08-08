# tests/testthat/test-MultidimOU.R
test_that("2D OU runs end-to-end", {
  state     <- c("x1", "x2")
  driftExpr <- c(
    "theta11*(mu1 - x1) + theta12*(mu2 - x2)",
    "theta21*(mu1 - x1) + theta22*(mu2 - x2)"
  )
  diffMat <- matrix(
    c("sigma1", "0",
      "0",      "sigma2"),
    nrow = 2, byrow = TRUE
  )

  mod <- yuima::setModel(
    drift          = driftExpr,
    diffusion      = diffMat,            # ← matrix, not "c(...)"
    time.variable  = "t",
    state.variable = state,
    solve.variable = state
  )

  samp <- suppressWarnings(  # avoid “delta (re)defined”
    yuima::setSampling(Initial = 0, delta = 1/200, n = 200)
  )
  yui <- yuima::setYuima(model = mod, sampling = samp)

  set.seed(1)
  sim <- yuima::simulate(
    yui,
    xinit = c(0, 0),
    true.parameter = list(
      theta11 = 1, theta12 = 0,
      theta21 = 0, theta22 = 1,
      mu1 = 0, mu2 = 0,
      sigma1 = 0.3, sigma2 = 0.3
    )
  )

  fit <- yuimaBayes::bayes(
    sim,
    bounds = list(sigma1 = c(0, Inf), sigma2 = c(0, Inf)),
    priors = list(sigma1 = "student_t(3, 0, 0.5)", sigma2 = "student_t(3, 0, 0.5)"),
    default_prior = "normal(0, 5)",
    chains = 2, iter = 800, control = list(adapt_delta = 0.95)
  )

  expect_true(inherits(fit, "bayes.yuima"))
  expect_true(inherits(fit@fit, "stanfit"))
})
