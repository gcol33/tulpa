# The deterministic core of tulpa_psis() (GPD tail fit + Pareto smoothing) runs
# in cpp_tulpa_psis. The R helpers (.tulpa_gpd_fit / .tulpa_qgpd /
# .tulpa_logsumexp) are the oracle here: this test reconstructs the former R
# body and asserts the C++-backed tulpa_psis() reproduces it to libm rounding
# across sample sizes, tail settings, and heavy-tailed inputs. (test-psis.R
# separately pins the k-hat against loo::psis.)

.psis_R_oracle <- function(log_ratios, tail_points = NULL) {
  log_ratios <- log_ratios[is.finite(log_ratios)]; S <- length(log_ratios)
  if (S < 5L) return(list(pareto_k = NA_real_, is_ess = NA_real_,
                          log_weights = numeric(0), tail_len = 0L,
                          tail_smoothed = FALSE))
  lw <- log_ratios - max(log_ratios)
  tail_len <- .psis_tail_len(S, tail_points); k_hat <- NA_real_
  smoothed_tail <- FALSE
  if (tail_len >= 5L && tail_len < S && S >= 25L) {
    ord <- order(lw); cut_idx <- S - tail_len; cutoff <- lw[ord[cut_idx]]
    tail_ord <- ord[(cut_idx + 1L):S]; exceed <- exp(lw[tail_ord]) - exp(cutoff)
    if (sum(exceed > 0) >= 5L) {
      fit <- .tulpa_gpd_fit(exceed); k_hat <- fit$k
      # loo::psis smooths only when the tail fit is usable (loo:::psis_smooth_tail
      # guards on is.finite(k)); otherwise the tail keeps its raw log ratios.
      if (is.finite(fit$k) && is.finite(fit$sigma) && fit$sigma > 0) {
        smoothed_tail <- TRUE
        pp <- (seq_len(tail_len) - 0.5) / tail_len
        smoothed <- log(exp(cutoff) + .tulpa_qgpd(pp, fit$k, fit$sigma))
        smoothed <- pmin(smoothed, max(lw)); lw[tail_ord] <- smoothed
      }
    }
  }
  lw <- lw - .tulpa_logsumexp(lw); w <- exp(lw)
  list(pareto_k = k_hat, is_ess = 1 / sum(w^2), log_weights = lw,
       tail_len = tail_len, tail_smoothed = smoothed_tail)
}

test_that("cpp_tulpa_psis reproduces the R PSIS core (k, is_ess, log-weights)", {
  set.seed(61)
  wk <- wi <- ww <- 0
  for (rep in 1:150) {
    S  <- sample(c(6, 25, 60, 200, 800), 1)
    lr <- switch(sample(1:3, 1),
                 rnorm(S), rexp(S) - 1, c(rnorm(S - 3), 20, 25, 40))
    tp <- if (runif(1) < 0.3) sample(5:max(6, floor(0.2 * S)), 1) else NULL
    R  <- .psis_R_oracle(lr, tp)
    C  <- tulpa_psis(lr, tp)
    expect_equal(C$tail_len, R$tail_len)
    dk <- if (is.na(R$pareto_k) && is.na(C$pareto_k)) 0
          else abs(R$pareto_k - C$pareto_k)
    wk <- max(wk, dk); wi <- max(wi, abs(R$is_ess - C$is_ess))
    if (length(R$log_weights) == length(C$log_weights))
      ww <- max(ww, max(abs(R$log_weights - C$log_weights)))
  }
  expect_lt(wk, 1e-9)
  expect_lt(wi, 1e-9)
  expect_lt(ww, 1e-9)
})

# A flat region in the log ratios -- an observation whose removal barely moves
# the posterior -- makes exp(lw) - exp(cutoff) exactly 0 for every tail draw
# tied with the cutoff. With a quarter of the exceedances at 0 the Zhang-Stephens
# profile's scale anchor `xstar` is 0, the whole theta grid is -Inf, and the fit
# comes back with a non-finite shape and a NaN scale. The tail then keeps its raw
# log ratios (loo:::psis_smooth_tail's own `is.finite(k)` behaviour) rather than
# being written with NaN quantiles that the normalizer spreads to all S weights.
.psis_tied_tail_ratios <- function(n_flat = 175L, n_top = 25L) {
  c(rep(0, n_flat), seq(0.5, 5, length.out = n_top))
}

test_that("a tail tied at the cutoff leaves every weight finite and unsmoothed", {
  lr <- .psis_tied_tail_ratios()
  ps <- tulpa_psis(lr)

  expect_false(any(is.nan(ps$log_weights)))
  expect_true(all(is.finite(ps$log_weights)))
  expect_equal(sum(exp(ps$log_weights)), 1, tolerance = 1e-12)
  expect_true(is.finite(ps$is_ess))
  expect_false(is.nan(ps$pareto_k))
  expect_false(ps$tail_smoothed)

  # Unsmoothed means the weights are the raw normalized log ratios.
  raw <- lr - .tulpa_logsumexp(lr)
  expect_equal(ps$log_weights, raw, tolerance = 1e-12)

  # And the R oracle carrying the same guard agrees, k included.
  R <- .psis_R_oracle(lr)
  expect_equal(ps$is_ess, R$is_ess, tolerance = 1e-9)
  expect_equal(ps$log_weights, R$log_weights, tolerance = 1e-9)
  expect_equal(ps$tail_smoothed, R$tail_smoothed)
  expect_equal(is.finite(ps$pareto_k), is.finite(R$pareto_k))
})

test_that("a usable tail fit still reports tail_smoothed", {
  set.seed(7)
  ps <- tulpa_psis(rnorm(500))
  expect_true(ps$tail_smoothed)
  expect_true(is.finite(ps$pareto_k))
})

test_that("cpp_tulpa_psis rejects a tail_len at or past the sample size", {
  set.seed(9)
  lr <- rnorm(60)
  expect_error(cpp_tulpa_psis(lr, 60L), "tail_len")
  expect_error(cpp_tulpa_psis(lr, 61L), "tail_len")
  expect_error(cpp_tulpa_psis(lr, 0L),  "tail_len")
  expect_error(cpp_tulpa_psis(lr, -1L), "tail_len")
  expect_silent(cpp_tulpa_psis(lr, 59L))
})

test_that("cpp_psis_loo_pit rejects a tail_len at or past the draw count", {
  set.seed(8)
  S <- 40L; N <- 3L
  ll <- matrix(stats::rnorm(S * N), S, N)
  Fl <- matrix(stats::runif(S * N, 0, 0.5), S, N)
  Fu <- Fl + 0.25
  expect_error(cpp_psis_loo_pit(ll, Fl, Fu, S, 1L), "tail_len")
  expect_error(cpp_psis_loo_pit(ll, Fl, Fu, S + 1L, 1L), "tail_len")
  expect_error(cpp_psis_loo_pit(ll, Fl, Fu, 0L, 1L), "tail_len")
  expect_true(all(is.finite(cpp_psis_loo_pit(ll, Fl, Fu, 8L, 1L))))
})
