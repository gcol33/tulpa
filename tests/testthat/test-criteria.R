# Criteria layer: WAIC / PSIS-LOO reproduce the loo package; CPO/LPML/DIC obey
# their defining identities; streaming equals materialized; PIT is uniform under
# a correct model.

make_loglik <- function(S = 300L, N = 80L, seed = 1L) {
  set.seed(seed)
  levels <- stats::rnorm(N, -2, 0.5)
  list(ll = matrix(stats::rnorm(S * N, rep(levels, each = S), 0.3), S, N),
       levels = levels, S = S, N = N)
}

test_that("WAIC matches loo::waic", {
  skip_if_not_installed("loo")
  d <- make_loglik()
  cr <- tulpa_criteria(d$ll, criteria = "waic")
  lw <- loo::waic(d$ll)
  expect_equal(cr$waic, lw$estimates["waic", "Estimate"], tolerance = 1e-8)
  expect_equal(cr$p_waic, lw$estimates["p_waic", "Estimate"], tolerance = 1e-8)
  expect_equal(cr$elpd_waic, lw$estimates["elpd_waic", "Estimate"],
               tolerance = 1e-8)
  expect_equal(cr$se_elpd_waic, lw$estimates["elpd_waic", "SE"],
               tolerance = 1e-6)
})

test_that("PSIS-LOO matches loo::loo (elpd, p_loo, pareto_k)", {
  skip_if_not_installed("loo")
  d <- make_loglik()
  cr <- tulpa_criteria(d$ll, criteria = "loo", pointwise = TRUE)
  lo <- suppressWarnings(loo::loo(d$ll, r_eff = NA))
  expect_equal(cr$elpd_loo, lo$estimates["elpd_loo", "Estimate"],
               tolerance = 1e-8)
  expect_equal(cr$p_loo, lo$estimates["p_loo", "Estimate"], tolerance = 1e-8)
  expect_equal(max(abs(cr$pointwise$pareto_k - lo$diagnostics$pareto_k)),
               0, tolerance = 1e-6)
})

test_that("CPO and LPML share the LOO computation", {
  d <- make_loglik()
  cr <- tulpa_criteria(d$ll, criteria = c("loo", "cpo", "lpml"),
                       pointwise = TRUE)
  expect_equal(cr$lpml, cr$elpd_loo)
  expect_equal(cr$lpml, sum(log(cr$pointwise$cpo)), tolerance = 1e-10)
  expect_equal(cr$pointwise$cpo, exp(cr$pointwise$elpd_loo), tolerance = 1e-12)
})

test_that("DIC obeys its identities and needs loglik_at_mean", {
  d <- make_loglik()
  cr <- tulpa_criteria(d$ll, criteria = "dic", loglik_at_mean = d$levels)
  expect_equal(cr$dbar, -2 * mean(rowSums(d$ll)), tolerance = 1e-10)
  expect_equal(cr$dhat, -2 * sum(d$levels), tolerance = 1e-10)
  expect_equal(cr$p_dic, cr$dbar - cr$dhat, tolerance = 1e-12)
  expect_equal(cr$dic, cr$dbar + cr$p_dic, tolerance = 1e-12)

  cr_na <- tulpa_criteria(d$ll, criteria = "dic")
  expect_true(is.na(cr_na$dic))
  expect_true(is.na(cr_na$p_dic))
  expect_false(is.na(cr_na$dbar))
})

test_that("streaming over column blocks equals the materialized matrix", {
  d <- make_loglik(N = 53L)
  full <- tulpa_criteria(d$ll, criteria = c("waic", "loo", "dic"),
                         loglik_at_mean = d$levels)
  gen <- function(cols) d$ll[, cols, drop = FALSE]
  streamed <- tulpa_criteria(
    tulpa_loglik(gen, n_obs = d$N, n_draws = d$S),
    criteria = c("waic", "loo", "dic"), loglik_at_mean = d$levels,
    chunk_size = 5L
  )
  expect_equal(streamed$waic, full$waic, tolerance = 1e-12)
  expect_equal(streamed$elpd_loo, full$elpd_loo, tolerance = 1e-12)
  expect_equal(streamed$dic, full$dic, tolerance = 1e-12)
})

test_that("tulpa_loglik requires dimensions for a generator", {
  expect_error(tulpa_loglik(function(cols) NULL), "n_obs")
})

test_that("tulpa_pit is uniform under a correct continuous model", {
  set.seed(7)
  # Correct model: the predictive CDF at the observation is exactly U(0,1).
  N <- 2000L
  pit <- tulpa_pit(stats::runif(N))
  ks <- suppressWarnings(stats::ks.test(pit, "punif"))
  expect_gt(ks$p.value, 0.01)
  expect_true(all(pit >= 0 & pit <= 1))
})

test_that("randomized PIT interpolates between the CDF limits", {
  set.seed(3)
  S <- 200L; N <- 50L
  Fl <- matrix(stats::runif(S * N, 0, 0.4), S, N)
  Fu <- Fl + matrix(stats::runif(S * N, 0, 0.4), S, N)
  pit <- tulpa_pit(Fu, cdf_lower = Fl)
  expect_true(all(pit >= colMeans(Fl) - 1e-9))
  expect_true(all(pit <= colMeans(Fu) + 1e-9))
})

test_that("waic.tulpa_fit / loo.tulpa_fit route through tulpa_criteria", {
  skip_if_not_installed("loo")
  d <- make_loglik()
  fit <- structure(list(draws = list(log_lik = d$ll)), class = "tulpa_fit")
  w <- waic.tulpa_fit(fit)
  expect_s3_class(w, "waic")
  expect_equal(w$estimates["waic", "Estimate"],
               loo::waic(d$ll)$estimates["waic", "Estimate"], tolerance = 1e-8)
  l <- loo.tulpa_fit(fit)
  expect_s3_class(l, "psis_loo")
})
