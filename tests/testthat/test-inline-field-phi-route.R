skip_on_cran()

for (correlated in c(FALSE, TRUE)) {
  test_that(sprintf("inline fields integrate an omitted variance under laplace (correlated = %s)", correlated), {
    skip_on_cran()
    set.seed(11)
    n_s <- 16L
    W <- 1 * (abs(outer(seq_len(n_s), seq_len(n_s), "-")) == 1L)
    cell <- rep(seq_len(n_s), each = 12L)
    x <- rnorm(length(cell))
    y <- 0.3 + 0.5 * x + rnorm(n_s, sd = 0.6)[cell] +
      rnorm(length(cell), sd = 0.4)
    d <- data.frame(y = y, x = x, cell = cell)
    formula <- if (correlated) {
      y ~ spatial(graph = W, formula = ~ 1 + x | cell)
    } else {
      y ~ x + spatial(graph = W, formula = ~ 1 || cell)
    }
    warnings <- character()
    omitted <- withCallingHandlers(tulpa(
      formula, d,
      family = "gaussian", mode = "laplace",
      control = list(n_draws = 100L, progress = FALSE, diagnose_k = FALSE)),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      })
    expect_false(any(grepl("`phi` not supplied", warnings, fixed = TRUE)))
    expect_true(omitted$phi_estimated)
    expect_true("phi_y" %in% colnames(omitted$theta_grid))
    expect_gt(length(unique(omitted$theta_grid[, "phi_y"])), 1L)
    expect_equal(omitted$phi, omitted$theta_mean[["phi_y"]])
    expect_true(omitted$theta_ci_lo[["phi_y"]] < 0.4^2 &&
                  omitted$theta_ci_hi[["phi_y"]] > 0.4^2)

    stated <- suppressWarnings(tulpa(
      formula, d,
      family = "gaussian", mode = "laplace", phi = 0.4^2,
      control = list(n_draws = 100L, progress = FALSE, diagnose_k = FALSE)))
    expect_false(isTRUE(stated$phi_estimated))
    expect_false("phi_y" %in% colnames(stated$theta_grid))
    expect_equal(stated$phi, 0.4^2)

    if (!correlated) {
      conditional_warnings <- character()
      conditional <- withCallingHandlers(
        tulpa(y ~ x, d, family = "gaussian", mode = "laplace"),
        warning = function(w) {
          conditional_warnings <<- c(conditional_warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        })
      expect_identical(conditional$backend, "laplace")
      expect_equal(conditional$phi, 1)
      expect_true(any(grepl("`phi` not supplied", conditional_warnings, fixed = TRUE)))
    }
  })
}
