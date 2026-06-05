# Latent structure threaded through the ModelData sampler kernels (gcol33/tulpa#75):
# random effects (all forms), areal spatial (ICAR / BYM2), and temporal (RW1 /
# RW2 / AR1) now reach the kernels through the same ModelData builder the Laplace
# fit uses, so the variance components are sampled jointly with the latent field.
# These tests check parameter recovery (NUTS), shape / finiteness for the
# approximate kernels (SMC / VI) on the variance funnel, and the ESS guard.

# Random intercepts (poisson) through NUTS: the slope, intercept, and the RE
# standard deviation all recover, and the layout exposes log_sigma_re + b[G].
test_that("NUTS recovers a poisson random-intercept model", {
  skip_on_cran()
  skip_if_fast()
  set.seed(1)
  G <- 25L; n_per <- 18L; N <- G * n_per
  group <- rep(seq_len(G), each = n_per)
  x <- rnorm(N)
  beta_true <- c(0.5, -0.8)
  sigma_re_true <- 0.7
  b <- rnorm(G, 0, sigma_re_true)
  y <- rpois(N, exp(beta_true[1] + beta_true[2] * x + b[group]))
  X <- cbind(1, x)
  re_spec <- list(idx = list(as.integer(group)), ngroups = G, ncoefs = 1L,
                  correlated = FALSE, Z = list(NULL))
  fit <- tulpa_sample_glmm(y, NULL, X, "poisson", "hmc", re_spec = re_spec,
                           fixed_names = c("(Intercept)", "x"),
                           control = list(n_iter = 1500L, warmup = 750L,
                                          n_chains = 2L, seed = 11L))
  m <- fit$means
  expect_equal(fit$n_params, ncol(X) + 1L + G)
  expect_true("log_sigma_re" %in% fit$param_names)
  expect_lt(abs(m[["(Intercept)"]] - beta_true[1]), 0.25)
  expect_lt(abs(m[["x"]] - beta_true[2]), 0.15)
  expect_lt(abs(exp(m[["log_sigma_re"]]) - sigma_re_true), 0.30)
})

# The SMC and VI kernels run on the same RE ModelData. The variance funnel is
# hard for both approximate kernels (SMC collapses sigma toward 0), so we assert
# only that they return a finite, correctly shaped full latent draw matrix and
# recover the fixed-effect slope -- not the variance component.
test_that("SMC and VI return finite full-latent draws on a binomial RE model", {
  skip_on_cran()
  skip_if_fast()
  set.seed(2)
  G <- 20L; n_per <- 20L; N <- G * n_per
  group <- rep(seq_len(G), each = n_per)
  x <- rnorm(N)
  beta_true <- c(0.2, 1.0)
  b <- rnorm(G, 0, 0.8)
  y <- rbinom(N, 1L, plogis(beta_true[1] + beta_true[2] * x + b[group]))
  X <- cbind(1, x)
  re_spec <- list(idx = list(as.integer(group)), ngroups = G, ncoefs = 1L,
                  correlated = FALSE, Z = list(NULL))
  for (backend in c("smc", "vi")) {
    fit <- tulpa_sample_glmm(y, rep(1L, N), X, "binomial", backend,
                             re_spec = re_spec,
                             fixed_names = c("(Intercept)", "x"),
                             control = list(n_particles = 1500L, n_mcmc_steps = 8L,
                                            vi_max_iter = 8000L, n_draws = 2000L,
                                            seed = 5L))
    expect_equal(ncol(fit$draws), ncol(X) + 1L + G, info = backend)
    expect_true(all(is.finite(fit$means)), info = backend)
    expect_lt(abs(fit$means[["x"]] - beta_true[2]), 0.30, label = backend)
  }
})

# Areal ICAR spatial field (poisson) through NUTS: the covariate slope recovers
# and the posterior-mean field tracks the simulated truth (positive correlation),
# with log_tau_spatial sampled jointly.
test_that("NUTS recovers a poisson ICAR spatial model", {
  skip_on_cran()
  skip_if_fast()
  set.seed(3)
  S <- 30L
  adj <- matrix(0, S, S)
  for (i in 1:(S - 1)) { adj[i, i + 1] <- 1; adj[i + 1, i] <- 1 }
  phi <- cumsum(rnorm(S, 0, 0.3)); phi <- phi - mean(phi)
  n_per_u <- 8L; Ns <- S * n_per_u
  unit <- rep(seq_len(S), each = n_per_u)
  xs <- rnorm(Ns)
  bs <- c(0.3, 0.5)
  ys <- rpois(Ns, exp(bs[1] + bs[2] * xs + phi[unit]))
  Xs <- cbind(1, xs)
  csr <- adjacency_to_csr_tulpa(adj)
  sp_spec <- list(type = "icar", spatial_idx = as.integer(unit),
                  n_spatial_units = S, adj_row_ptr = as.integer(csr$row_ptr),
                  adj_col_idx = as.integer(csr$col_idx),
                  n_neighbors = as.integer(csr$n_neighbors), scale_factor = 1.0)
  fit <- tulpa_sample_glmm(ys, NULL, Xs, "poisson", "hmc", spatial_spec = sp_spec,
                           fixed_names = c("(Intercept)", "xs"),
                           control = list(n_iter = 1200L, warmup = 600L,
                                          n_chains = 2L, seed = 7L))
  m <- fit$means
  expect_true("log_tau_spatial" %in% fit$param_names)
  expect_lt(abs(m[["xs"]] - bs[2]), 0.20)
  phi_idx <- grep("^phi_spatial", names(m))
  phi_est <- as.numeric(m[phi_idx]); phi_est <- phi_est - mean(phi_est)
  expect_equal(length(phi_est), S)
  expect_gt(cor(phi_est, phi), 0.5)
})

# Temporal AR1 field (gaussian) through NUTS: the covariate slope recovers and the
# AR1 dependence parameter is sampled (logit_rho_ar1 present in the layout).
test_that("NUTS recovers a gaussian temporal AR1 model", {
  skip_on_cran()
  skip_if_fast()
  set.seed(4)
  Tn <- 60L; rho <- 0.7; sdq <- 0.5
  ft <- numeric(Tn); ft[1] <- rnorm(1, 0, sdq / sqrt(1 - rho^2))
  for (t in 2:Tn) ft[t] <- rho * ft[t - 1] + rnorm(1, 0, sdq)
  xt <- rnorm(Tn); bt <- c(1.0, -0.5)
  yt <- bt[1] + bt[2] * xt + ft + rnorm(Tn, 0, 0.3)
  Xt <- cbind(1, xt)
  tp_spec <- list(type = "ar1", time_idx = as.integer(seq_len(Tn)),
                  n_times = Tn, n_groups = 1L, group_idx = NULL, cyclic = FALSE)
  fit <- tulpa_sample_glmm(yt, NULL, Xt, "gaussian", "hmc", temporal_spec = tp_spec,
                           phi = 0.3, fixed_names = c("(Intercept)", "xt"),
                           control = list(n_iter = 1200L, warmup = 600L,
                                          n_chains = 2L, seed = 9L))
  m <- fit$means
  expect_true("logit_rho_ar1" %in% fit$param_names)
  expect_lt(abs(m[["xt"]] - bt[2]), 0.20)
})

# An end-to-end RE fit through the tulpa() front door routes to the sampler and
# carries the full latent layout (regression guard for the routing in
# .tulpa_fitter_args + select_inference_mode).
test_that("tulpa(mode = 'hmc') routes a random-intercept model through the sampler", {
  skip_on_cran()
  skip_if_fast()
  set.seed(5)
  G <- 18L; n_per <- 16L; N <- G * n_per
  g <- rep(seq_len(G), each = n_per); x <- rnorm(N)
  b <- rnorm(G, 0, 0.7)
  y <- rpois(N, exp(0.4 + 0.6 * x + b[g]))
  d <- data.frame(y = y, x = x, g = factor(g))
  fit <- suppressMessages(tulpa(y ~ x + (1 | g), d, family = "poisson",
                                mode = "hmc",
                                control = list(n_iter = 1200L, warmup = 600L,
                                               n_chains = 2L, seed = 11L)))
  expect_true("log_sigma_re" %in% fit$param_names)
  expect_equal(fit$n_params, 2L + 1L + G)
  expect_lt(abs(fit$means[["x"]] - 0.6), 0.18)
  expect_lt(abs(exp(fit$means[["log_sigma_re"]]) - 0.7), 0.35)
})

# ESS carries random effects (its joint_sigma_re block samples the RE SD) but its
# isotropic Gaussian-prior block cannot carry a graph-precision spatial / temporal
# field, so a structured spatial or temporal spec is declined with guidance.
test_that("ESS declines a structured spatial or temporal field", {
  skip_on_cran()
  set.seed(6)
  S <- 12L
  adj <- matrix(0, S, S)
  for (i in 1:(S - 1)) { adj[i, i + 1] <- 1; adj[i + 1, i] <- 1 }
  Ns <- S * 4L; unit <- rep(seq_len(S), each = 4L)
  xs <- rnorm(Ns); ys <- rpois(Ns, exp(0.2 + 0.3 * xs))
  Xs <- cbind(1, xs)
  csr <- adjacency_to_csr_tulpa(adj)
  sp_spec <- list(type = "icar", spatial_idx = as.integer(unit),
                  n_spatial_units = S, adj_row_ptr = as.integer(csr$row_ptr),
                  adj_col_idx = as.integer(csr$col_idx),
                  n_neighbors = as.integer(csr$n_neighbors), scale_factor = 1.0)
  expect_error(
    tulpa_sample_glmm(ys, NULL, Xs, "poisson", "ess", spatial_spec = sp_spec,
                      control = list(n_iter = 200L, warmup = 100L)),
    "cannot carry")
})
