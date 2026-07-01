# tulpa_pit() and the LOO-PIT weighting (cpp_psis_loo_pit) run in C++. The
# randomization draws from R's RNG stream in the same index order as the former
# R bodies (the PSIS core is deterministic), so under a fixed seed both are
# byte-identical to the R reference reconstructed here.

.old_tulpa_pit <- function(cdf, cdf_lower = NULL, jitter = TRUE) {
  bar <- function(z) if (is.matrix(z)) colMeans(z) else as.numeric(z)
  Fu <- bar(cdf); N <- length(Fu)
  if (!is.null(cdf_lower)) { Fl <- bar(cdf_lower); U <- runif(N); pit <- Fl + U * (Fu - Fl) }
  else { pit <- Fu; if (jitter) pit <- pit + runif(N, 0, 1e-6) }
  pmin(1, pmax(0, pit))
}

test_that("tulpa_pit matches the R reference under a fixed seed", {
  set.seed(5); S <- 40L; N <- 60L
  cu <- matrix(runif(S * N, 0.3, 1), S, N); cl <- matrix(runif(S * N, 0, 0.3), S, N)
  set.seed(9); R1 <- .old_tulpa_pit(cu, cl); set.seed(9); C1 <- tulpa_pit(cu, cl)
  set.seed(9); R2 <- .old_tulpa_pit(cu);     set.seed(9); C2 <- tulpa_pit(cu)
  expect_equal(C1, R1, tolerance = 1e-12)
  expect_equal(C2, R2, tolerance = 1e-12)
})

test_that("cpp_psis_loo_pit matches the per-column R loop, thread invariant", {
  old_loo <- function(ll, Fl, Fu) {
    Nn <- ncol(ll); Ss <- nrow(ll); fl <- numeric(Nn); fu <- numeric(Nn)
    for (i in seq_len(Nn)) {
      ps <- tulpa_psis(-ll[, i]); lw <- ps$log_weights
      w <- if (length(lw) == Ss) exp(lw) else rep(1 / Ss, Ss)
      fl[i] <- sum(w * Fl[, i]); fu[i] <- sum(w * Fu[, i])
    }
    u <- runif(Nn); pmin(1, pmax(0, fl + u * (fu - fl)))
  }
  set.seed(3); Sd <- 200L; Nn <- 80L
  ll <- matrix(rnorm(Sd * Nn, -2, 1), Sd, Nn)
  Fl <- matrix(runif(Sd * Nn, 0, 0.4), Sd, Nn)
  Fu <- Fl + matrix(runif(Sd * Nn, 0, 0.6), Sd, Nn)
  tail_len <- .psis_tail_len(Sd, NULL)
  set.seed(11); R <- old_loo(ll, Fl, Fu)
  set.seed(11); C1 <- cpp_psis_loo_pit(ll, Fl, Fu, as.integer(tail_len), 1L)
  set.seed(11); C4 <- cpp_psis_loo_pit(ll, Fl, Fu, as.integer(tail_len), 4L)
  expect_equal(C1, R, tolerance = 1e-12)
  expect_identical(C4, C1)
})
