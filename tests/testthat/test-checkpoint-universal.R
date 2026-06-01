# Universal grid-cell / per-unit checkpoint/resume (gcol33/tulpa#50).
#
# The joint nested-Laplace path has its own file
# (test-nested-laplace-joint-checkpoint.R). This covers the other fitters the
# checkpoint was extended to: the single-block nested-Laplace grid kernels (all
# of which share run_multi_block_nested_laplace, so ICAR exercises the path),
# the RE-covariance CCD integration, and the per-chain NUTS producer. For each:
# a checkpointed fit equals an un-checkpointed one, a resume loads completed
# units and re-appends nothing, a torn tail from a killed write is discarded and
# re-solved to the same answer, and a file written for different inputs is
# rejected (fingerprint mismatch).

skip_on_cran()

.ck_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s), function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# --- single-block nested Laplace (ICAR) -------------------------------------

test_that("single-block nested-Laplace checkpoint == un-checkpointed + resumes", {
  set.seed(11L)
  n_s <- 20L; N <- 160L
  adj <- .ck_chain_adj(n_s)
  w <- as.numeric(arima.sim(n = n_s, list(ar = 0.6))) * 0.7; w <- w - mean(w)
  sidx <- sample(n_s, N, replace = TRUE); x <- rnorm(N)
  y <- rbinom(N, 1, plogis(0.3 + 0.5 * x + w[sidx]))
  X <- cbind(1, x)
  prior <- c(list(type = "icar", sigma_grid = c(0.5, 0.8, 1.2, 1.8),
                  spatial_idx = sidx), adj)
  fit <- function(ck = NULL) {
    ctrl <- list(diagnose_k = FALSE)
    if (!is.null(ck)) ctrl$checkpoint <- ck
    tulpa_nested_laplace(y = y, n_trials = rep(1L, N), X = X, prior = prior,
                         family = "binomial", control = ctrl)
  }
  path <- tempfile(fileext = ".ckpt"); on.exit(unlink(path), add = TRUE)

  f_plain <- fit()
  f_ck    <- fit(list(path = path, resume = FALSE))
  expect_equal(f_ck$log_marginal, f_plain$log_marginal, tolerance = 1e-9)
  expect_true(file.exists(path))
  sz <- file.size(path)

  f_re <- fit(list(path = path, resume = TRUE))
  expect_equal(f_re$log_marginal, f_plain$log_marginal, tolerance = 1e-9)
  expect_equal(file.size(path), sz)  # resume re-appended nothing

  b <- readBin(path, "raw", n = file.size(path))
  writeBin(b[seq_len(as.integer(length(b) * 0.6))], path)
  f_torn <- fit(list(path = path, resume = TRUE))
  expect_equal(f_torn$log_marginal, f_plain$log_marginal, tolerance = 1e-9)

  y2 <- rbinom(N, 1, plogis(0.3 + 0.5 * x + w[sidx] + 0.6))
  expect_error(
    tulpa_nested_laplace(y = y2, n_trials = rep(1L, N), X = X, prior = prior,
                         family = "binomial",
                         control = list(diagnose_k = FALSE,
                                        checkpoint = list(path = path, resume = TRUE))),
    "fingerprint")
})

# --- RE-covariance nested CCD -----------------------------------------------

test_that("RE-cov nested CCD checkpoint == un-checkpointed + resumes", {
  set.seed(1L)
  G <- 40L; npg <- 10L; N <- G * npg
  grp <- rep(seq_len(G), each = npg); x <- rnorm(N)
  X <- cbind(1, x); Z <- cbind(1, x)
  Sigma <- matrix(c(0.64, 0.24, 0.24, 0.36), 2)
  u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), 2))
  eta <- as.numeric(X %*% c(-0.3, 0.7)) + rowSums(Z * u[grp, ])
  y <- rbinom(N, 1L, plogis(eta))
  re_term <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = Z)
  fit <- function(ck = NULL)
    tulpa_re_cov_nested(y, rep(1L, N), X, re_term, family = "binomial",
                        diagnose_k = FALSE, checkpoint = ck)
  path <- tempfile(fileext = ".ckpt"); on.exit(unlink(path), add = TRUE)

  f0 <- fit()
  f1 <- fit(list(path = path, resume = FALSE))
  expect_equal(f1$weights, f0$weights, tolerance = 1e-9)
  expect_equal(f1$posterior$mean, f0$posterior$mean, tolerance = 1e-9)
  f2 <- fit(list(path = path, resume = TRUE))
  expect_equal(f2$posterior$mean, f0$posterior$mean, tolerance = 1e-9)

  y2 <- rbinom(N, 1L, plogis(eta + 0.6))
  expect_error(
    tulpa_re_cov_nested(y2, rep(1L, N), X, re_term, family = "binomial",
                        diagnose_k = FALSE,
                        checkpoint = list(path = path, resume = TRUE)),
    "fingerprint")
})

# --- per-chain NUTS ---------------------------------------------------------

test_that("per-chain NUTS checkpoint reproduces draws and resumes after a crash", {
  set.seed(7L)
  N <- 120L; p <- 3L
  X <- cbind(1, matrix(rnorm(N * (p - 1)), N))
  y <- as.numeric(X %*% c(0.5, -0.8, 0.4) + rnorm(N, sd = 0.7))
  run <- function(ck = "")
    cpp_tulpa_fit_generic_chains(y, X, n_chains = 4L, n_iter = 300L,
                                 n_warmup = 150L, seed = 123L, verbose = FALSE,
                                 checkpoint_path = ck)
  path <- tempfile(fileext = ".ckpt"); on.exit(unlink(path), add = TRUE)

  f_plain <- run("")
  f_ck    <- run(path)
  # Determinism: per-chain checkpoint must not perturb the draws.
  expect_equal(f_ck$draws, f_plain$draws, tolerance = 1e-10)

  # Crash mid-run: keep half the records, resume re-runs only the lost chains.
  b <- readBin(path, "raw", n = file.size(path))
  writeBin(b[seq_len(as.integer(length(b) * 0.5))], path)
  f_re <- run(path)
  expect_equal(f_re$draws, f_plain$draws, tolerance = 1e-10)

  # A full resume loads every chain and appends nothing.
  sz <- file.size(path)
  f_re2 <- run(path)
  expect_equal(file.size(path), sz)
  expect_equal(f_re2$draws, f_plain$draws, tolerance = 1e-10)

  # Different seed -> different fit -> fingerprint mismatch.
  expect_error(
    cpp_tulpa_fit_generic_chains(y, X, n_chains = 4L, n_iter = 300L,
                                 n_warmup = 150L, seed = 999L, verbose = FALSE,
                                 checkpoint_path = path),
    "fingerprint")
})
