# The deterministic core of tulpa_psis() (GPD tail fit + Pareto smoothing) runs
# in cpp_tulpa_psis. The R helpers (.tulpa_gpd_fit / .tulpa_qgpd /
# .tulpa_logsumexp) are the oracle here: this test reconstructs the former R
# body and asserts the C++-backed tulpa_psis() reproduces it to libm rounding
# across sample sizes, tail settings, and heavy-tailed inputs. (test-psis.R
# separately pins the k-hat against loo::psis.)

.psis_R_oracle <- function(log_ratios, tail_points = NULL) {
  log_ratios <- log_ratios[is.finite(log_ratios)]; S <- length(log_ratios)
  if (S < 5L) return(list(pareto_k = NA_real_, is_ess = NA_real_,
                          log_weights = numeric(0), tail_len = 0L))
  lw <- log_ratios - max(log_ratios)
  tail_len <- .psis_tail_len(S, tail_points); k_hat <- NA_real_
  if (tail_len >= 5L && S >= 25L) {
    ord <- order(lw); cut_idx <- S - tail_len; cutoff <- lw[ord[cut_idx]]
    tail_ord <- ord[(cut_idx + 1L):S]; exceed <- exp(lw[tail_ord]) - exp(cutoff)
    if (sum(exceed > 0) >= 5L) {
      fit <- .tulpa_gpd_fit(exceed); k_hat <- fit$k
      pp <- (seq_len(tail_len) - 0.5) / tail_len
      smoothed <- log(exp(cutoff) + .tulpa_qgpd(pp, fit$k, fit$sigma))
      smoothed <- pmin(smoothed, max(lw)); lw[tail_ord] <- smoothed
    }
  }
  lw <- lw - .tulpa_logsumexp(lw); w <- exp(lw)
  list(pareto_k = k_hat, is_ess = 1 / sum(w^2), log_weights = lw, tail_len = tail_len)
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
