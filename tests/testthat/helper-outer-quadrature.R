expect_outer_quadrature_reference <- function(fit, reference) {
  state <- tulpa:::.tulpa_outer_integration_state(fit)
  expect_identical(state$method, "aghq_log_spline")
  expect_true(state$converged)
  for (i in seq_along(state$reads)) {
    read <- state$reads[[i]]
    hyper <- max(abs(sweep(t(read$hyper) - reference$quantiles,
                          1L, reference$sd, "/")))
    fixed <- max(abs(sweep(read$fixed - reference$fixed_quantiles,
                          1L, reference$fixed_sd, "/")))
    evidence <- abs(state$history[i, "log_evidence"] - reference$log_evidence)
    expect_gte(read$error$max, max(hyper, fixed))
    expect_gte(read$error$log_evidence, evidence)
  }
  expect_lte(hyper, .1)
  expect_lte(fixed, .1)
  expect_lte(evidence, .015)
  expect_lte(state$error$max, state$tol)
  expect_lte(state$error$log_evidence, state$evidence_tol)
}
