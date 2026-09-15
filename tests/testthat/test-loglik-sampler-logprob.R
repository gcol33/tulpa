# Every sampler backend records a per-draw log posterior, so logLik() reports
# `log_posterior_mean` on it instead of declining (gcol33/tulpa#750).
#
# Three arbiters, none of them the recording code:
#   * the ModelData samplers (ess, sghmc, sgld, mclmc, smc, vi) report the
#     helper's value, and that helper evaluated on a NUTS fit's own draws has
#     to reproduce the log_prob NUTS recorded while sampling;
#   * the RE-covariance Gibbs sweep is scored against its joint density written
#     out in R (dpois, a Gaussian through chol, the inverse-gamma / Wishart);
#   * each Polya-Gamma route is scored against its own target written out in R
#     from the draws the kernel stores.

lp_poisson_fixture <- function(seed = 1L, N = 60L) {
  set.seed(seed)
  x <- rnorm(N)
  X <- cbind("(Intercept)" = 1, x = x)
  y <- rpois(N, exp(0.3 + 0.5 * x))
  list(y = y, X = X, N = N)
}

lp_sample <- function(d, backend, control) {
  tulpa_sample_glmm(d$y, NULL, d$X, family = "poisson", backend = backend,
                    control = control)
}

lp_helper_at <- function(fit) {
  mi <- fit$model_inputs
  cpp_tulpa_glmm_log_prob_draws(
    draws = unname(as.matrix(fit$draws)), y = mi$y, n_trials = mi$n_trials,
    X = mi$X, family = mi$family, phi = mi$phi, sigma_beta = mi$sigma_beta,
    offset_nullable = mi$offset, re_spec = mi$re_spec,
    spatial_spec = mi$spatial_spec, temporal_spec = mi$temporal_spec,
    sigma_re_scale = mi$sigma_re_scale, phi2 = mi$phi2,
    svc_spec = mi$svc_spec, tvc_spec = mi$tvc_spec, zi_spec = mi$zi_spec)
}

expect_log_posterior_mean <- function(fit) {
  expect_false(is.null(fit$log_prob))
  expect_length(fit$log_prob, nrow(as.matrix(fit$draws)))
  expect_true(all(is.finite(fit$log_prob)))
  ll <- logLik(fit)
  expect_identical(attr(ll, "quantity"), "log_posterior_mean")
  expect_null(attr(ll, "declined"))
  expect_equal(as.numeric(ll), mean(fit$log_prob))
}

# ---------------------------------------------------------------------------
# ModelData samplers
# ---------------------------------------------------------------------------

test_that("the per-draw helper reproduces the log_prob NUTS records", {
  d <- lp_poisson_fixture()
  fit <- lp_sample(d, "hmc", list(n_iter = 200L, warmup = 100L, n_chains = 1L,
                                  seed = 3L))
  expect_log_posterior_mean(fit)
  expect_lt(max(abs(lp_helper_at(fit) - fit$log_prob)), 1e-10)
})

backend_controls <- list(
  ess   = list(n_iter = 200L, warmup = 100L, seed = 5L),
  sghmc = list(n_iter = 200L, warmup = 100L, seed = 5L),
  sgld  = list(n_iter = 200L, warmup = 100L, seed = 5L),
  mclmc = list(n_iter = 200L, warmup = 100L, seed = 5L),
  smc   = list(n_particles = 200L, n_mcmc_steps = 3L, seed = 5L),
  vi    = list(vi_max_iter = 300L, n_draws = 200L, seed = 5L)
)

for (backend in names(backend_controls)) {
  local({
    b <- backend
    test_that(sprintf("backend '%s' reports the helper's log posterior", b), {
      d <- lp_poisson_fixture()
      fit <- lp_sample(d, b, backend_controls[[b]])
      expect_log_posterior_mean(fit)
      expect_lt(max(abs(lp_helper_at(fit) - fit$log_prob)), 1e-10)
      # A kernel that also keeps a running log posterior per stored draw agrees
      # with the one definition.
      if (!is.null(fit$log_prob_kernel)) {
        expect_lt(max(abs(fit$log_prob_kernel - fit$log_prob)), 1e-8)
      }
    })
  })
}

# ---------------------------------------------------------------------------
# RE-covariance Gibbs
# ---------------------------------------------------------------------------

# log IW(S; nu, Lambda) for a p x p block.
ref_log_iw <- function(S, nu, Lambda) {
  p <- nrow(S)
  lgp <- p * (p - 1) / 4 * log(pi) + sum(lgamma((nu + 1 - seq_len(p)) / 2))
  nu / 2 * as.numeric(determinant(Lambda)$modulus) - nu * p / 2 * log(2) - lgp -
    (nu + p + 1) / 2 * as.numeric(determinant(S)$modulus) -
    sum(diag(Lambda %*% solve(S))) / 2
}

ref_log_mvn0 <- function(B, S) {
  L <- t(chol(S))
  nc <- nrow(S)
  sum(apply(B, 1, function(b) {
    w <- forwardsolve(L, b)
    -0.5 * nc * log(2 * pi) - sum(log(diag(L))) - 0.5 * sum(w^2)
  }))
}

test_that("re_cov_gibbs records its joint density at each retained sweep", {
  set.seed(11)
  G <- 15L; npg <- 6L; N <- G * npg
  g <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  X <- cbind(1, x)
  Z <- cbind(1, x)
  u <- cbind(rnorm(G, 0, 0.5), rnorm(G, 0, 0.3))
  y <- rpois(N, exp(0.2 + 0.4 * x + rowSums(Z * u[g, ])))

  for (corr in c(TRUE, FALSE)) {
    rt <- list(idx = g, n_groups = G, n_coefs = 2L, Z = Z, correlated = corr)
    fit <- tulpa_re_cov_gibbs(y, NULL, X, rt, family = "poisson",
                              control = list(n_iter = 60L, warmup = 40L,
                                             seed = 2L))
    expect_log_posterior_mean(fit)

    k <- 37L
    beta <- fit$beta_draws[k, ]
    B <- matrix(fit$re[k, ], ncol = 2L, byrow = TRUE)
    S <- fit$Sigma_draws[[k]]
    pr <- fit$prior[[1L]]
    bp <- tulpa:::.normalize_beta_prior(
      tulpa:::.tulpa_default_beta_prior("re_cov_gibbs"), 2L)
    eta <- as.numeric(X %*% beta) + rowSums(Z * B[g, ])
    ref <- sum(dpois(y, exp(eta), log = TRUE)) +
      sum(dnorm(beta, bp$mean, bp$sd, log = TRUE)) +
      ref_log_mvn0(B, S) +
      if (pr$full) ref_log_iw(S, pr$nu0, pr$Lambda0) else
        sum(vapply(1:2, function(i)
          ref_log_iw(S[i, i, drop = FALSE], pr$nu0,
                     matrix(pr$lambda0[i])), 0))
    expect_lt(abs(fit$log_prob[k] - ref), 1e-8)
  }
})

# ---------------------------------------------------------------------------
# Polya-Gamma Gibbs routes
# ---------------------------------------------------------------------------

PG_ITER <- 120L
PG_WARM <- 60L

ref_log_hc <- function(s, scale) log(2) + dcauchy(s, 0, scale, log = TRUE)

ref_log_gmrf <- function(quad, rank, tau) {
  0.5 * rank * (log(tau) - log(2 * pi)) - 0.5 * tau * quad
}

ref_icar_Q <- function(W) diag(rowSums(W)) - W

pg_areal <- function(seed = 4L, nr = 4L, nc = 4L, reps = 3L) {
  set.seed(seed)
  W <- rook_adj(nr, nc)
  J <- nr * nc
  unit <- rep(seq_len(J), each = reps)
  N <- length(unit)
  X <- cbind(1, rnorm(N))
  grp <- ((unit - 1L) %% 4L) + 1L
  ntr <- rep(4L, N)
  y <- rbinom(N, ntr, plogis(-0.2 + 0.6 * X[, 2] + rnorm(J, 0, 0.4)[unit]))
  al <- tulpa:::adjacency_to_list_tulpa(W)
  list(W = as.matrix(W), J = J, unit = unit, N = N, X = X, grp = grp,
       ntr = ntr, y = as.integer(y), al = al)
}

pg_common_ref <- function(y, ntr, eta, beta, re, sigma_re, beta_sd, re_scale) {
  sum(dbinom(y, ntr, plogis(eta), log = TRUE)) +
    sum(dnorm(beta, 0, beta_sd, log = TRUE)) +
    sum(dnorm(re, 0, sigma_re, log = TRUE)) + ref_log_hc(sigma_re, re_scale)
}

test_that("PG binomial iid route records its joint density", {
  d <- pg_areal()
  res <- cpp_pg_binomial_gibbs(d$y, d$ntr, d$X, d$grp, 4L, PG_ITER, PG_WARM, 1L,
                               prior_beta_sd = 3, prior_sigma_scale = 2,
                               verbose = FALSE)
  k <- nrow(res$beta)
  beta <- res$beta[k, ]; re <- res$re[k, ]
  eta <- as.numeric(d$X %*% beta) + re[d$grp]
  ref <- pg_common_ref(d$y, d$ntr, eta, beta, re, res$sigma_re[k], 3, 2)
  expect_lt(abs(res$log_prob[k] - ref), 1e-8)
})

test_that("PG binomial ICAR route records its joint density", {
  d <- pg_areal()
  res <- cpp_pg_binomial_gibbs_spatial(
    d$y, d$ntr, d$X, d$grp, 4L, d$unit, d$J, d$al$adj_list, d$al$n_neighbors,
    PG_ITER, PG_WARM, 1L, prior_beta_sd = 3, prior_sigma_re_scale = 2,
    prior_tau_shape = 1.5, prior_tau_rate = 0.2, verbose = FALSE)
  k <- nrow(res$beta)
  beta <- res$beta[k, ]; re <- res$re[k, ]; phi <- res$spatial[k, ]
  tau <- res$tau[k]
  eta <- as.numeric(d$X %*% beta) + re[d$grp] + phi[d$unit]
  quad <- drop(crossprod(phi, ref_icar_Q(d$W) %*% phi))
  ref <- pg_common_ref(d$y, d$ntr, eta, beta, re, res$sigma_re[k], 3, 2) +
    ref_log_gmrf(quad, d$J - 1L, tau) + dgamma(tau, 1.5, 0.2, log = TRUE)
  expect_lt(abs(res$log_prob[k] - ref), 1e-8)
})

test_that("PG binomial BYM2 route records its joint density", {
  d <- pg_areal()
  sf <- 0.7; a <- 0.8; b <- 1.3
  res <- cpp_pg_binomial_gibbs_bym2(
    d$y, d$ntr, d$X, d$grp, 4L, d$unit, d$J, d$al$adj_list, d$al$n_neighbors,
    sf, PG_ITER, PG_WARM, 1L, prior_beta_sd = 3, prior_sigma_re_scale = 2,
    prior_sigma_spatial_scale = 1.5, prior_rho_alpha = a, prior_rho_beta = b,
    verbose = FALSE)
  k <- nrow(res$beta)
  beta <- res$beta[k, ]; re <- res$re[k, ]
  ph <- res$phi_scaled[k, ]; th <- res$theta[k, ]
  s <- res$sigma_spatial[k]; rho <- res$rho[k]
  eps <- 1e-10
  u <- s * (sqrt(rho + eps) * ph * sf + sqrt(1 - rho + eps) * th)
  eta <- as.numeric(d$X %*% beta) + re[d$grp] + u[d$unit]
  kern <- function(r) (a - 1) * log(r + eps) + (b - 1) * log(1 - r + eps)
  nodes <- (seq_len(20L) - 0.5) / 20
  kv <- kern(nodes)
  log_mass <- kern(rho) - (max(kv) + log(sum(exp(kv - max(kv)))))
  quad <- drop(crossprod(ph, ref_icar_Q(d$W) %*% ph))
  ref <- pg_common_ref(d$y, d$ntr, eta, beta, re, res$sigma_re[k], 3, 2) +
    ref_log_gmrf(quad, d$J - 1L, 1) + sum(dnorm(th, log = TRUE)) +
    log(2) + dnorm(s, 0, 1.5, log = TRUE) + log_mass
  expect_true(rho %in% nodes)
  expect_lt(abs(res$log_prob[k] - ref), 1e-8)
})

test_that("PG binomial RSR route records its joint density at the raw field", {
  d <- pg_areal(reps = 1L)
  Xu <- d$X
  P <- tulpa:::compute_rsr_projection(Xu)
  res <- cpp_pg_binomial_gibbs_rsr(
    d$y, d$ntr, d$X, rep(1L, d$N), 0L, d$unit, d$J, d$al$adj_list,
    d$al$n_neighbors, as.numeric(t(P)), d$J, PG_ITER, PG_WARM, 1L,
    prior_beta_sd = 3, prior_sigma_re_scale = 2, prior_tau_shape = 1.5,
    prior_tau_rate = 0.2, verbose = FALSE)
  k <- nrow(res$beta)
  beta <- res$beta[k, ]; phi <- res$spatial_raw[k, ]; tau <- res$tau[k]
  eta <- as.numeric(d$X %*% beta) + as.numeric(P %*% phi)[d$unit]
  quad <- drop(crossprod(phi, ref_icar_Q(d$W) %*% phi))
  ref <- sum(dbinom(d$y, d$ntr, plogis(eta), log = TRUE)) +
    sum(dnorm(beta, 0, 3, log = TRUE)) +
    ref_log_gmrf(quad, d$J - 1L, tau) + dgamma(tau, 1.5, 0.2, log = TRUE)
  expect_lt(abs(res$log_prob[k] - ref), 1e-8)
})

test_that("PG binomial temporal route records its joint density", {
  set.seed(8)
  n_times <- 12L; period <- 4L
  tt <- rep(seq_len(n_times), each = 4L)
  N <- length(tt)
  X <- cbind(1, rnorm(N))
  ntr <- rep(5L, N)
  y <- as.integer(rbinom(N, ntr, plogis(0.1 + 0.4 * X[, 2] + sin(tt))))
  res <- cpp_pg_binomial_gibbs_temporal(
    y, ntr, X, rep(1L, N), 0L, tt, n_times, period, 1L, 1L, PG_ITER, PG_WARM,
    1L, prior_beta_sd = 3, prior_sigma_re_scale = 2,
    prior_sigma_trend_scale = 0.7, prior_sigma_seasonal_scale = 0.9,
    prior_sigma_short_scale = 1.1, verbose = FALSE)
  k <- nrow(res$beta)
  beta <- res$beta[k, ]
  tr <- res$trend[k, ]; se <- res$seasonal[k, ]; sh <- res$short_term[k, ]
  st <- res$sigma_trend[k]; ss <- res$sigma_seasonal[k]; sr <- res$sigma_short[k]
  rho <- res$rho_short[k]
  eta <- as.numeric(X %*% beta) + tr[tt] + se[((tt - 1L) %% period) + 1L] + sh[tt]
  q_trend <- sum(diff(tr)^2)
  q_seas <- sum(diff(c(se, se[1]))^2)
  q_ar1 <- (1 - rho^2) * sh[1]^2 + sum((sh[-1] - rho * sh[-n_times])^2)
  tau_s <- 1 / sr^2
  ref <- sum(dbinom(y, ntr, plogis(eta), log = TRUE)) +
    sum(dnorm(beta, 0, 3, log = TRUE)) +
    ref_log_gmrf(q_trend, n_times - 1L, 1 / st^2) + ref_log_hc(st, 0.7) +
    ref_log_gmrf(q_seas, period - 1L, 1 / ss^2) + ref_log_hc(ss, 0.9) +
    0.5 * n_times * (log(tau_s) - log(2 * pi)) + 0.5 * log(1 - rho^2) -
    0.5 * tau_s * q_ar1 - log(2 * 0.999) + ref_log_hc(sr, 1.1)
  expect_lt(abs(res$log_prob[k] - ref), 1e-8)
})

nb_r_log_prior <- function(r, a, b) {
  dgamma(r, a, b, log = TRUE) - log(pgamma(500, a, b) - pgamma(0.1, a, b))
}

test_that("PG negative-binomial ICAR route records its joint density", {
  d <- pg_areal()
  set.seed(9)
  yc <- as.integer(rnbinom(d$N, size = 4, mu = exp(0.5 + 0.3 * d$X[, 2])))
  res <- cpp_pg_negbin_gibbs_spatial(
    yc, d$X, d$grp, 4L, d$unit, d$J, d$al$adj_list, d$al$n_neighbors,
    PG_ITER, PG_WARM, 1L, 3, 2, 1.5, 0.2, 1.2, 0.3, 5, FALSE, FALSE, 1L)
  k <- nrow(res$beta)
  r <- res$r[k]
  beta_nb2 <- res$beta[k, ]
  beta_zhou <- beta_nb2 - c(log(r), 0)
  re <- res$re[k, ]; phi <- res$spatial[k, ]; tau <- res$tau[k]
  eta_nb2 <- as.numeric(d$X %*% beta_nb2) + re[d$grp] + phi[d$unit]
  quad <- drop(crossprod(phi, ref_icar_Q(d$W) %*% phi))
  ref <- sum(dnbinom(yc, size = r, mu = exp(eta_nb2), log = TRUE)) +
    sum(dnorm(beta_zhou, 0, 3, log = TRUE)) +
    sum(dnorm(re, 0, res$sigma_re[k], log = TRUE)) +
    ref_log_hc(res$sigma_re[k], 2) +
    ref_log_gmrf(quad, d$J - 1L, tau) + dgamma(tau, 1.5, 0.2, log = TRUE) +
    nb_r_log_prior(r, 1.2, 0.3)
  expect_lt(abs(res$log_prob[k] - ref), 1e-8)
})

test_that("PG negative-binomial route records its joint density, with and without an iid block", {
  d <- pg_areal()
  set.seed(10)
  yc <- as.integer(rnbinom(d$N, size = 4, mu = exp(0.5 + 0.3 * d$X[, 2])))
  res0 <- cpp_pg_negbin_gibbs(yc, d$X, rep(1L, d$N), 0L, PG_ITER, PG_WARM, 1L,
                              prior_beta_sd = 3, prior_sigma_scale = 2,
                              prior_r_shape = 1.2, prior_r_rate = 0.3,
                              r_init = 5, store_eta = FALSE, verbose = FALSE,
                              n_threads = 1L)
  k <- nrow(res0$beta)
  r <- res0$r[k]
  beta_nb2 <- res0$beta[k, ]
  ref <- sum(dnbinom(yc, size = r, mu = exp(as.numeric(d$X %*% beta_nb2)),
                     log = TRUE)) +
    sum(dnorm(beta_nb2 - c(log(r), 0), 0, 3, log = TRUE)) +
    nb_r_log_prior(r, 1.2, 0.3)
  expect_lt(abs(res0$log_prob[k] - ref), 1e-8)

  # The iid block's level is drawn rather than removed (gcol33/tulpa#761), so
  # the chain has a stated target and records it.
  res1 <- cpp_pg_negbin_gibbs(yc, d$X, d$grp, 4L, PG_ITER, PG_WARM, 1L,
                              prior_beta_sd = 3, prior_sigma_scale = 2,
                              prior_r_shape = 1.2, prior_r_rate = 0.3,
                              r_init = 5, store_eta = FALSE, verbose = FALSE,
                              n_threads = 1L)
  k <- nrow(res1$beta)
  r <- res1$r[k]
  beta_nb2 <- res1$beta[k, ]; re <- res1$re[k, ]; s_re <- res1$sigma_re[k]
  ref <- sum(dnbinom(yc, size = r,
                     mu = exp(as.numeric(d$X %*% beta_nb2) + re[d$grp]),
                     log = TRUE)) +
    sum(dnorm(beta_nb2 - c(log(r), 0), 0, 3, log = TRUE)) +
    sum(dnorm(re, 0, s_re, log = TRUE)) + ref_log_hc(s_re, 2) +
    nb_r_log_prior(r, 1.2, 0.3)
  expect_lt(abs(res1$log_prob[k] - ref), 1e-8)
  # The block is no longer forced to mean zero.
  expect_gt(max(abs(rowMeans(res1$re))), 1e-6)
})

# A one-neighbour sequential NNGP over a 1-D ordering: position i's parent is
# position i - 1, which is what the kernel's `nn_idx` encodes 1-based.
pg_nngp_chain <- function(seed, n) {
  set.seed(seed)
  coords <- cbind(runif(n), runif(n))
  ord <- order(coords[, 1])
  nn_idx <- matrix(c(0L, seq_len(n - 1L)), ncol = 1L)
  d1 <- c(0, sqrt(rowSums((coords[ord[-1], , drop = FALSE] -
                           coords[ord[-n], , drop = FALSE])^2)))
  list(coords = coords, ord = ord, nn_idx = nn_idx, nn_dist = matrix(d1, ncol = 1L))
}

# The sequential NNGP density at unit-variance kriging weights, with the
# engine's diagonal nugget in the neighbour correlation, times the PC prior on
# sigma2 and the uniform prior on phi.
ref_log_nngp_chain <- function(w, s2, phi, g, U, alpha, lo, hi) {
  nug <- 1e-8
  o <- g$ord
  lp <- dnorm(w[o[1]], 0, sqrt(s2), log = TRUE)
  for (i in 2:length(o)) {
    rho <- exp(-g$nn_dist[i, 1] / phi)
    B <- rho / (1 + nug)
    F <- max(1 - rho^2 / (1 + nug), 1e-10)
    lp <- lp + dnorm(w[o[i]], B * w[o[i - 1]], sqrt(s2 * F), log = TRUE)
  }
  lam <- -log(alpha) / U
  s <- sqrt(s2)
  lp + log(lam) - lam * s - log(2 * s) - log(hi - lo)
}

test_that("PG NNGP route records its joint density, and logLik() reads it", {
  # gcol33/tulpa#761: the field's level is drawn from its conditional rather
  # than removed into the intercept, so the chain's target is the model's.
  n <- 20L
  g <- pg_nngp_chain(12L, n)
  X <- cbind(1, rnorm(n))
  y <- as.integer(rbinom(n, 5L, 0.4))
  res <- cpp_pg_binomial_gibbs_gp(
    y, rep(5L, n), X, rep(1L, n), 0L, g$coords, g$nn_idx, g$nn_dist,
    as.integer(g$ord - 1L), n, 1L, 1.0, 0.5, 0L, PG_ITER, PG_WARM, 1L,
    prior_beta_sd = 3, prior_sigma_gp_U = 1.5, prior_sigma_gp_alpha = 0.05,
    prior_phi_lower = 0.02, prior_phi_upper = 4, verbose = FALSE)
  k <- nrow(res$beta)
  beta <- res$beta[k, ]; w <- res$gp[k, ]
  ref <- sum(dbinom(y, 5L, plogis(as.numeric(X %*% beta) + w), log = TRUE)) +
    sum(dnorm(beta, 0, 3, log = TRUE)) +
    ref_log_nngp_chain(w, res$sigma2_gp[k], res$phi_gp[k], g, 1.5, 0.05, 0.02, 4)
  expect_lt(abs(res$log_prob[k] - ref), 1e-8)
  expect_gt(max(abs(rowMeans(res$gp))), 1e-6)

  chain <- tulpa:::.pg_as_chain(res, "gp", X)
  expect_false(is.null(chain$log_prob))
  fit <- tulpa:::.finalize_fit(chain, backend = "gibbs", n_fixed = 2L,
                               fixed_names = chain$param_names[1:2])
  expect_log_posterior_mean(fit)
})

test_that("PG multiscale NNGP route records its joint density", {
  n <- 20L
  g <- pg_nngp_chain(13L, n)
  X <- cbind(1, rnorm(n))
  y <- as.integer(rbinom(n, 5L, 0.4))
  res <- cpp_pg_binomial_gibbs_multiscale_gp(
    y, rep(5L, n), X, rep(1L, n), 0L, g$coords,
    g$nn_idx, g$nn_dist, as.integer(g$ord - 1L), 1L,
    g$nn_idx, g$nn_dist, as.integer(g$ord - 1L), 1L,
    n, 0.5, 0.2, 0.8, 2.0, 0L, PG_ITER, PG_WARM, 1L,
    prior_beta_sd = 3, verbose = FALSE)
  k <- nrow(res$beta)
  beta <- res$beta[k, ]; wl <- res$w_local[k, ]; wr <- res$w_regional[k, ]
  ref <- sum(dbinom(y, 5L, plogis(as.numeric(X %*% beta) + wl + wr), log = TRUE)) +
    sum(dnorm(beta, 0, 3, log = TRUE)) +
    ref_log_nngp_chain(wl, res$sigma2_local[k], res$phi_local[k], g,
                       1, 0.01, 0.01, 5) +
    ref_log_nngp_chain(wr, res$sigma2_regional[k], res$phi_regional[k], g,
                       1, 0.01, 0.1, 20)
  expect_lt(abs(res$log_prob[k] - ref), 1e-8)
})

test_that("a Polya-Gamma fit carries log_prob through the chain assembly", {
  d <- pg_areal()
  fit <- tulpa_gibbs(d$y, d$ntr, d$X, d$grp, 4L, family = "binomial",
                     control = list(n_iter = PG_ITER, warmup = PG_WARM,
                                    seed = 1L))
  expect_log_posterior_mean(fit)
})
