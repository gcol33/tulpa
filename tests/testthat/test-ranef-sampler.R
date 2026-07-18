# ranef() on a ModelData sampler-tier fit must return exactly the random-effect
# coefficients (the `re[...]` columns of the joint draws), not the whole latent
# block. The draws are [fixed, latent (field + RE), variance-component
# hyperparameters]; a regression would re-include the latent field and a
# spurious `log_sigma_re` row.

test_that("ranef() on a sampler-tier fit returns exactly the RE coefficients", {
  skip_if_not_slow()
  set.seed(3)
  ng <- 8L; per <- 10L
  g <- rep(seq_len(ng), each = per)
  x <- rnorm(ng * per)
  b <- rnorm(ng, 0, 0.6)
  eta <- 0.2 + 0.5 * x + b[g]
  d <- data.frame(y = rpois(ng * per, exp(eta)), x = x, g = factor(g))

  fit <- tulpa(y ~ x + (1 | g), d, family = "poisson", mode = "mala",
               sigma_re = 0.6, control = list(n_iter = 300L, warmup = 150L))

  re <- ranef(fit)
  expect_equal(nrow(re), ng)                              # one row per group
  expect_true(all(grepl("^g\\[", re$term)))               # named by group level
  expect_false(any(grepl("sigma|phi|tau|L_re|log_", re$term)))  # no hyperparameters
  expect_true(all(is.finite(re$estimate)))
})
