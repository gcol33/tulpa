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

test_that("group = NULL is byte-identical to the ungrouped call", {
  d <- make_loglik(N = 40L)
  base <- tulpa_criteria(d$ll, criteria = c("waic", "loo", "cpo", "lpml"),
                         pointwise = TRUE)
  same <- tulpa_criteria(d$ll, criteria = c("waic", "loo", "cpo", "lpml"),
                         group = NULL, pointwise = TRUE)
  expect_identical(base$n_obs, same$n_obs)
  expect_null(same$n_groups)
  expect_equal(same$waic, base$waic, tolerance = 0)
  expect_equal(same$elpd_loo, base$elpd_loo, tolerance = 0)
  expect_equal(same$pointwise$pareto_k, base$pointwise$pareto_k, tolerance = 0)
})

test_that("grouped LOGO-CV equals criteria on the hand-aggregated matrix", {
  d <- make_loglik(N = 60L, seed = 7L)
  # 60 columns -> 12 groups of 5 (each column is one observation within a cell).
  group <- rep(seq_len(12L), each = 5L)
  grouped <- tulpa_criteria(d$ll, criteria = c("waic", "loo", "cpo", "lpml"),
                            group = group, pointwise = TRUE)
  # Hand-aggregate: a fold's joint conditional log-lik is the within-group sum.
  G <- sapply(split(seq_len(60L), group), function(cols) rowSums(d$ll[, cols, drop = FALSE]))
  ref <- tulpa_criteria(G, criteria = c("waic", "loo", "cpo", "lpml"),
                        pointwise = TRUE)
  expect_equal(grouped$n_groups, 12L)
  expect_equal(grouped$n_obs, 60L)
  expect_equal(grouped$waic, ref$waic, tolerance = 1e-12)
  expect_equal(grouped$elpd_loo, ref$elpd_loo, tolerance = 1e-12)
  expect_equal(grouped$se_elpd_loo, ref$se_elpd_loo, tolerance = 1e-12)
  expect_equal(grouped$pointwise$pareto_k, ref$pointwise$pareto_k, tolerance = 1e-12)
  expect_equal(nrow(grouped$pointwise), 12L)
})

test_that("grouped LOO streams over column blocks identically", {
  d <- make_loglik(N = 57L, seed = 9L)
  group <- ((seq_len(57L) - 1L) %% 9L) + 1L      # 9 interleaved groups
  full <- tulpa_criteria(d$ll, criteria = c("waic", "loo"), group = group)
  gen <- function(cols) d$ll[, cols, drop = FALSE]
  streamed <- tulpa_criteria(
    tulpa_loglik(gen, n_obs = d$N, n_draws = d$S),
    criteria = c("waic", "loo"), group = group, chunk_size = 4L)
  expect_equal(streamed$waic, full$waic, tolerance = 1e-12)
  expect_equal(streamed$elpd_loo, full$elpd_loo, tolerance = 1e-12)
})

test_that("group must match n_obs", {
  d <- make_loglik(N = 20L)
  expect_error(tulpa_criteria(d$ll, criteria = "loo", group = rep(1L, 19L)),
               "length n_obs")
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

test_that("compare_models computes native WAIC / LOO from a fit's pointwise log-lik", {
  skip_if_not_installed("loo")
  d1 <- make_loglik(seed = 1L)
  d2 <- make_loglik(seed = 2L)
  d2$ll <- d2$ll + 0.1                              # a slightly better fit
  fit1 <- structure(list(draws = list(log_lik = d1$ll)), class = "tulpa_fit")
  fit2 <- structure(list(draws = list(log_lik = d2$ll)), class = "tulpa_fit")

  cmp_loo <- compare_models(a = fit1, b = fit2, criterion = "loo")
  expect_equal(nrow(cmp_loo), 2L)
  expect_true(all(c("elpd", "se_elpd", "ic", "delta", "weight") %in% names(cmp_loo)))
  expect_equal(cmp_loo$delta[1], 0)                 # best model first, delta 0
  # elpd matches loo::loo on the same pointwise matrix.
  el <- setNames(cmp_loo$elpd, cmp_loo$model)
  expect_equal(el[["a"]], suppressWarnings(loo::loo(d1$ll, r_eff = NA))$
                 estimates["elpd_loo", "Estimate"], tolerance = 1e-8)

  cmp_waic <- compare_models(a = fit1, b = fit2, criterion = "waic")
  ew <- setNames(cmp_waic$elpd, cmp_waic$model)
  expect_equal(ew[["a"]], loo::waic(d1$ll)$estimates["elpd_waic", "Estimate"],
               tolerance = 1e-8)
})

test_that("model_average stacking weights reproduce loo::stacking_weights", {
  skip_if_not_installed("loo")
  d1 <- make_loglik(seed = 1L); d2 <- make_loglik(seed = 2L)
  fit1 <- structure(list(draws = list(log_lik = d1$ll), fitted = rep(0, d1$N)),
                    class = "tulpa_fit")
  fit2 <- structure(list(draws = list(log_lik = d2$ll), fitted = rep(1, d2$N)),
                    class = "tulpa_fit")
  fitted_fn <- function(f) f$fitted

  avg <- model_average(a = fit1, b = fit2, weights = "loo", fitted_fn = fitted_fn)
  expect_equal(sum(avg$weights), 1, tolerance = 1e-6)

  # Cross-check the stacking weights against loo on the same pointwise elpd.
  lpd <- cbind(
    suppressWarnings(loo::loo(d1$ll, r_eff = NA))$pointwise[, "elpd_loo"],
    suppressWarnings(loo::loo(d2$ll, r_eff = NA))$pointwise[, "elpd_loo"])
  ref <- as.numeric(loo::stacking_weights(lpd))
  expect_equal(unname(avg$weights), ref, tolerance = 1e-3)
})

test_that("model_average pseudo-BMA weights reproduce loo::pseudobma_weights", {
  skip_if_not_installed("loo")
  d1 <- make_loglik(seed = 1L); d2 <- make_loglik(seed = 2L)
  fit1 <- structure(list(draws = list(log_lik = d1$ll), fitted = rep(0, d1$N)),
                    class = "tulpa_fit")
  fit2 <- structure(list(draws = list(log_lik = d2$ll), fitted = rep(1, d2$N)),
                    class = "tulpa_fit")
  avg <- model_average(a = fit1, b = fit2, weights = "pbma",
                      fitted_fn = function(f) f$fitted)
  lpd <- cbind(
    suppressWarnings(loo::loo(d1$ll, r_eff = NA))$pointwise[, "elpd_loo"],
    suppressWarnings(loo::loo(d2$ll, r_eff = NA))$pointwise[, "elpd_loo"])
  ref <- as.numeric(loo::pseudobma_weights(lpd, BB = FALSE))
  expect_equal(unname(avg$weights), ref, tolerance = 1e-6)
})
