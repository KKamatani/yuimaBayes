# tests/testthat/test-OU.R
test_that("1D OU fits without pathologies", {
  mod <- yuima::setModel(
    drift          = "theta*(mu - x)",
    diffusion      = "sigma",
    time.variable  = "t",
    state.variable = "x",
    solve.variable = "x"
  )
  samp <- suppressWarnings(yuima::setSampling(Initial = 0, delta = 1/200, n = 200))
  yui  <- yuima::setYuima(model = mod, sampling = samp)

  set.seed(1)
  sim <- yuima::simulate(
    yui, xinit = 0,
    true.parameter = list(theta = 1, mu = 0, sigma = 0.3)
  )

  fit <- yuimaBayes::bayes(
    sim,
    bounds = list(sigma = c(0, Inf)),
    priors = list(sigma = "student_t(3, 0, 0.5)"),
    default_prior = "student_t(3, 0, 2.5)",
    chains = 2, iter = 1000,
    control = list(adapt_delta = 0.95)
  )

  expect_true(inherits(fit, "bayes.yuima"))
  expect_true(inherits(fit@fit, "stanfit"))
})
