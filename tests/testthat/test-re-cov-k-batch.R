# The outer Pareto-k of tulpa_re_cov_nested() evaluates its importance draws
# through one compiled batch (cpp_laplace_log_marginal_multi_re_batch) rather
# than one tulpa_laplace() call per draw (gcol33/tulpa#934). The batch has to be
# the same target the per-draw path was: the same log-marginal at a cold start,
# the same zero weight where a draw cannot be solved, and the same value at any
# thread width.

k_batch_fixture <- function(seed = 401L, G = 30L, per = 3L) {
  set.seed(seed)
  grp <- rep(seq_len(G), each = per)
  n <- length(grp)
  x <- rnorm(n)
  b <- matrix(rnorm(2L * G, 0, c(0.7, 0.4)), G, 2L, byrow = TRUE)
  y <- rbinom(n, 1L, plogis(-1 + 0.7 * x + b[grp, 1L] + b[grp, 2L] * x))
  list(y = y, n_trials = rep(1L, n), X = cbind(1, x), x = x, grp = grp, G = G)
}

k_batch_call <- function(args, packs, x_init = NULL, width = 1L, tol = 1e-8) {
  do.call(cpp_laplace_log_marginal_multi_re_batch, c(args, list(
    re_sigma_batch = packs, tol = tol, x_init = x_init,
    n_threads_outer = width)))
}

test_that("the log-marginal batch is the per-point tulpa_laplace() target", {
  skip_on_cran()
  d <- k_batch_fixture()
  re_of <- function(s) list(list(idx = d$grp, n_groups = d$G, n_coefs = 1L,
                                 sigma = s))
  sigs <- c(Inf, 0, exp(log(0.6) + 0.8 * stats::rnorm(40)))
  one <- vapply(sigs, function(s) tryCatch(
    tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X, re_list = re_of(s),
                  family = "binomial", return_hessian = FALSE,
                  tol = 1e-8)$log_marginal,
    error = function(e) -Inf), numeric(1))
  one[!is.finite(one)] <- -Inf
  args <- .laplace_multi_re_model_args(d$y, d$n_trials, d$X, re_of(1),
                                       "binomial", 1)
  packs <- lapply(sigs, function(s) list(s))

  cold <- k_batch_call(args, packs)
  # A draw whose covariance cannot be converted is a zero-weight draw, and the
  # rest of the batch is unaffected by it.
  expect_identical(cold[1:2], c(-Inf, -Inf))
  expect_identical(cold, one)

  m0 <- tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                      re_list = re_of(0.6), family = "binomial",
                      return_hessian = FALSE, tol = 1e-8)$mode
  warm <- k_batch_call(args, packs, x_init = m0)
  expect_identical(is.finite(warm), is.finite(one))
  expect_lt(max(abs(warm - one)[is.finite(one)]), 1e-10)
  # Every point starts from the same x_init, so the width cannot move a value.
  expect_identical(k_batch_call(args, packs, x_init = m0, width = 2L), warm)
})

test_that("the batch reads correlated and multi-term covariances as the fit does", {
  skip_on_cran()
  d <- k_batch_fixture(G = 25L)
  Z <- cbind(1, d$x)
  re_of <- function(L, s2) list(
    list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = Z, L = L),
    list(idx = (d$grp %% 5L) + 1L, n_groups = 5L, n_coefs = 1L, sigma = s2))
  pts <- list(list(L = matrix(c(0.7, 0.1, 0, 0.4), 2L), s2 = 0.3),
              list(L = matrix(c(1.2, -0.3, 0, 0.2), 2L), s2 = 0.9),
              list(L = matrix(c(0.3, 0.05, 0, 0.6), 2L), s2 = 0.05))
  one <- vapply(pts, function(p) tulpa_laplace(
    y = d$y, n_trials = d$n_trials, X = d$X, re_list = re_of(p$L, p$s2),
    family = "binomial", return_hessian = FALSE, tol = 1e-8)$log_marginal,
    numeric(1))
  args <- .laplace_multi_re_model_args(d$y, d$n_trials, d$X,
                                       re_of(diag(2), 1), "binomial", 1)
  packs <- lapply(pts, function(p)
    .laplace_multi_re_sigma_list(re_of(p$L, p$s2)))
  expect_identical(k_batch_call(args, packs), one)
})

test_that("plain and subspace-debiased fits read one outer k-hat", {
  skip_on_cran()
  # Both fits score the mode-Hessian proposal against the same target, and the
  # debias consumes random numbers the plain fit does not. The diagnostic draws
  # from the stream the fit started on, so the two report one k-hat.
  set.seed(11)
  G <- 20L; grp <- rep(seq_len(G), each = 3L); x <- rnorm(length(grp))
  y <- rbinom(length(grp), 1L, plogis(-2.5 + 0.7 * x + rnorm(G, 0, 0.7)[grp]))
  rt <- list(idx = grp, n_groups = G, n_coefs = 1L)
  fit <- function(...) tulpa_re_cov_nested(y, rep(1L, length(y)), cbind(1, x),
                                          rt, family = "binomial",
                                          control = list(seed = 7L, ...))
  fp <- fit()
  fs <- fit(subspace_debias = TRUE)
  expect_true(is.finite(fp$pareto_k))
  expect_identical(fs$pareto_k, fp$pareto_k)
  expect_identical(fs$pareto_k_first_pass, fp$pareto_k_first_pass)

  # The width is wall-clock only.
  expect_identical(fit(k_threads = 2L)$pareto_k, fp$pareto_k)
  # The diagnostic leaves the draws as they are without it.
  expect_identical(fit(diagnose_k = FALSE)$draws, fp$draws)
})
