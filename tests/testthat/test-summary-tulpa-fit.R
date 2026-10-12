# summary() is the full report of a fit and print() the short one
# (gcol33/tulpa#953); tulpa_outer_grid() is the one door to a nested fit's
# hyperparameter grid (gcol33/tulpa#958).

.sm_laplace_shaped <- function() {
  structure(list(
    mode = c(0.5, 0.3), H_beta = diag(c(100, 50)), n_fixed = 2L,
    fixed_names = c("(Intercept)", "x"), backend = "laplace",
    inference_tier = 2L, inference_mode = "structured",
    selection_reason = "requested", N = 60L, family = "poisson",
    draws_kind = "point", call = quote(tulpa(y ~ x)),
    beta_prior = list(mean = 0, sd = 2.5)
  ), class = "tulpa_fit")
}

.sm_grid_shaped <- function() {
  modes <- rbind(c(0.1, 0.5, -1, 1), c(0.2, 0.6, -2, 2), c(0.3, 0.7, -3, 3))
  structure(list(
    theta_grid = c(0.5, 1, 2), theta_names = "tau",
    weights = c(1, 2, 1), log_marginal = c(-3, -2, -3),
    axis_support = list(tau = c(0.4, 2.5)), modes = modes, n_fixed = 2L,
    fixed_names = c("(Intercept)", "x"), backend = "nested_laplace_test",
    theta_mean = c(tau = 1.125), theta_sd = c(tau = 0.5),
    theta_median = c(tau = 1), theta_ci_lo = c(tau = 0.5),
    theta_ci_hi = c(tau = 2)
  ), class = "tulpa_fit")
}

test_that("summary() returns a summary.tulpa_fit carrying the whole report", {
  fit <- .sm_laplace_shaped()
  s <- summary(fit)
  expect_s3_class(s, "summary.tulpa_fit")
  expect_identical(s$inference$backend, "laplace")
  expect_identical(s$inference$inference_tier, 2L)
  expect_identical(s$inference$selection_reason, "requested")
  expect_identical(s$n_obs, 60L)
  expect_identical(s$family, "poisson")
  expect_identical(s$call, quote(tulpa(y ~ x)))
  expect_identical(s$reliability$kind, "point")
  expect_null(s$random)
  expect_null(s$hyperparameters)
  expect_equal(s$prior$beta$sd, 2.5)
})

test_that("coef(summary()) is the fixed-effect table confint() reads", {
  fit <- .sm_laplace_shaped()
  tab <- coef(summary(fit))
  expect_s3_class(tab, "data.frame")
  expect_identical(rownames(tab), c("(Intercept)", "x"))
  expect_identical(names(tab), c("estimate", "std.error", "2.5 %", "97.5 %"))
  expect_equal(tab$estimate, c(0.5, 0.3))
  expect_equal(tab$std.error, sqrt(c(1 / 100, 1 / 50)))
  expect_equal(unname(as.matrix(tab[, 3:4])), unname(confint(fit)),
               tolerance = 0)
  expect_identical(names(coef(summary(fit, level = 0.9)))[3:4],
                   c("5 %", "95 %"))
  expect_identical(summary(fit)$coefficients, tab)
})

test_that("print() is the short display and points at summary()", {
  fit <- .sm_laplace_shaped()
  out <- utils::capture.output(print(fit))
  expect_match(out[1L], "tulpa fit  (laplace, tier 2)", fixed = TRUE)
  expect_true(any(grepl("Call: tulpa(y ~ x)", out, fixed = TRUE)))
  expect_true(any(grepl("Coefficients:", out, fixed = TRUE)))
  expect_true(any(grepl("Use summary()", out, fixed = TRUE)))
  expect_false(any(grepl("std.error", out, fixed = TRUE)))
  expect_invisible(print(fit))
})

test_that("print(summary()) renders every section the fit carries", {
  s <- summary(.sm_grid_shaped())
  out <- utils::capture.output(print(s))
  expect_true(any(grepl("Fixed effects:", out, fixed = TRUE)))
  expect_true(any(grepl("Hyperparameters:", out, fixed = TRUE)))
  expect_true(any(grepl("tau", out, fixed = TRUE)))

  out_l <- utils::capture.output(print(summary(.sm_laplace_shaped())))
  expect_true(any(grepl("selected because: requested", out_l, fixed = TRUE)))
  expect_true(any(grepl("Fixed-effect prior: normal(0, 2.5)", out_l,
                        fixed = TRUE)))
  expect_true(any(grepl("Reliability (point draws):", out_l, fixed = TRUE)))
  expect_true(any(grepl("point summary", out_l, fixed = TRUE)))
})

test_that("a reliability record with nothing to report prints no header", {
  bare <- list(kind = "iid", verdict = NA_character_,
               convergence = list(ok = NA, rhat_max = NA, ess_bulk_min = NA))
  expect_identical(utils::capture.output(.print_reliability(bare, 4L)),
                   character())

  out <- utils::capture.output(
    .print_reliability(utils::modifyList(bare, list(elbo = -12.5)), 4L))
  expect_identical(out, c("", "Reliability (iid draws):", "  ELBO: -12.5"))
})

test_that("the hyperparameter table reads the outer-grid moments", {
  h <- summary(.sm_grid_shaped())$hyperparameters
  expect_identical(names(h), c("parameter", "mean", "sd", "median",
                               "ci_lo", "ci_hi"))
  expect_identical(h$parameter, "tau")
  expect_equal(h$mean, 1.125)
  expect_equal(c(h$ci_lo, h$ci_hi), c(0.5, 2))
})

test_that("tulpa_outer_grid() returns nodes, normalized weights and modes", {
  fit <- .sm_grid_shaped()
  g <- tulpa_outer_grid(fit)
  expect_identical(names(g), c("nodes", "weights", "log_marginal", "support",
                               "modes", "n_fixed"))
  expect_identical(dim(g$nodes), c(3L, 1L))
  expect_identical(colnames(g$nodes), "tau")
  expect_equal(as.numeric(g$nodes), c(0.5, 1, 2))
  expect_equal(g$weights, c(0.25, 0.5, 0.25))
  expect_equal(g$log_marginal, c(-3, -2, -3))
  expect_equal(g$support, list(tau = c(0.4, 2.5)))
  expect_identical(g$n_fixed, 2L)
  expect_equal(unname(colSums(g$weights * g$modes)), c(0.2, 0.6, -2, 2))
})

test_that("tulpa_outer_grid() refuses a fit with no outer grid", {
  expect_error(tulpa_outer_grid(.sm_laplace_shaped()),
               "no outer hyperparameter grid")
})

test_that("a real Laplace fit summarizes and prints end to end", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(x = rnorm(60))
  d$y <- rpois(60, exp(0.5 + 0.3 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  s <- summary(fit)
  expect_s3_class(s, "summary.tulpa_fit")
  expect_equal(coef(s)$estimate, unname(coef(fit)), tolerance = 0)
  expect_false(is.null(s$inference$backend))
  expect_output(print(s), "Fixed effects:")
  expect_output(print(fit), "Use summary()", fixed = TRUE)
})

test_that("a real nested-Laplace fit reads its grid through tulpa_outer_grid()", {
  skip_on_cran()
  set.seed(1)
  S <- 30L
  nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
  nn <- lengths(nb)
  field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
  idx <- rep(seq_len(S), each = 6L); n <- length(idx); x <- rnorm(n)
  y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
  prior <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
                adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
                n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4, 8))
  fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                              family = "binomial",
                              control = list(progress = FALSE))
  g <- tulpa_outer_grid(fit)
  expect_identical(g$nodes, tulpa_theta_matrix(fit))
  expect_equal(sum(g$weights), 1)
  expect_identical(nrow(g$nodes), length(g$weights))
  expect_identical(nrow(g$modes), length(g$weights))
  h <- summary(fit)$hyperparameters
  expect_identical(h$parameter, colnames(g$nodes))
  expect_equal(h$mean, unname(as.numeric(fit$theta_mean)))
})

test_that("tulpa_fit_summary() wraps a model package's own coefficient table", {
  fit <- .sm_laplace_shaped()
  tab <- data.frame(estimate = c(1, 2), std.error = c(0.1, 0.2),
                    `2.5 %` = c(0.8, 1.6), `97.5 %` = c(1.2, 2.4),
                    row.names = c("a_x", "b_x"), check.names = FALSE)
  s <- tulpa_fit_summary(fit, tab)
  expect_s3_class(s, "summary.tulpa_fit")
  expect_identical(coef(s), tab)
  expect_identical(s$inference, summary(fit)$inference)
  expect_identical(
    unclass(s)[setdiff(names(s), "coefficients")],
    unclass(summary(fit))[setdiff(names(s), "coefficients")])
  expect_error(tulpa_fit_summary(fit, as.matrix(tab)), "data frame")
})
