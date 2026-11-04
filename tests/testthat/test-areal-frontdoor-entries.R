# What an areal spec states reaches the nested-Laplace block it becomes, or the
# call stops: its grid axes, its PC anchor, the BYM2 scaling, the unit labels
# of an adjacency() graph, and the outer-grid control knobs.

.fe_lattice <- function(k = 5L) {
  K <- k * k
  A <- matrix(0, K, K)
  for (i in seq_len(k)) for (j in seq_len(k)) {
    s <- (i - 1L) * k + j
    if (j < k) A[s, s + 1L] <- A[s + 1L, s] <- 1
    if (i < k) A[s, s + k] <- A[s + k, s] <- 1
  }
  A
}

.fe_data <- function(seed = 2) {
  set.seed(seed)
  A <- .fe_lattice(); K <- nrow(A)
  n <- 150L; reg <- sample(seq_len(K), n, TRUE); x <- rnorm(n)
  u <- 0.5 * as.numeric(scale(cumsum(rnorm(K))))
  list(A = A, d = data.frame(y = rpois(n, exp(0.3 + 0.4 * x + u[reg])),
                             x = x, region = reg))
}

.fe_ctl <- list(progress = FALSE)

test_that("check_adjacency() returns its report visibly and prints nothing", {
  A <- .fe_lattice(3L)
  out <- capture.output(r <- withVisible(check_adjacency(A)))
  expect_length(out, 0L)
  expect_true(r$visible)
  expect_s3_class(r$value, "tulpa_adjacency_check")
  expect_output(print(r$value), "adjacency check")
})

test_that("an adjacency() graph names its matrix by the unit identifiers", {
  ids <- as.character(37000 + 3 * seq_len(9))
  g <- adjacency(data.frame(x = rep(1:3, 3), y = rep(1:3, each = 3), id = ids),
                 id = "id", type = "rook")
  expect_identical(rownames(g$adjacency), ids)
  expect_identical(colnames(as.matrix(g$adjacency)), ids)
  lab <- factor(ids[c(9, 1, 5, 5)])
  expect_identical(
    tulpa:::.resolve_spatial_idx(lab, 9L, as.matrix(g$adjacency), "fips"),
    c(9L, 1L, 5L, 5L))
})

test_that("an areal spec refuses a grid axis its field does not have", {
  A <- .fe_lattice(3L)
  sp <- list(type = "icar", adjacency = A, spatial_idx = 1:9, sigma_grid = 1)
  expect_error(tulpa:::.spatial_spec_to_nl_prior(sp),
               "`sigma_grid` is not an outer-grid axis of a nested-Laplace 'icar'")
  sp <- list(type = "bym2", adjacency = A, spatial_idx = 1:9, tau_grid = 1)
  expect_error(tulpa:::.spatial_spec_to_nl_prior(sp), "'bym2'")
  sp <- list(type = "icar", adjacency = A, spatial_idx = 1:9,
             tau_grid = c(1, 2), prior_sigma = c(0.5, 0.05))
  blk <- tulpa:::.spatial_spec_to_nl_prior(sp)
  expect_identical(blk$tau_grid, c(1, 2))
  expect_identical(blk$prior_sigma, c(0.5, 0.05))
  expect_error(tulpa:::.spatial_spec_to_nl_prior(
    modifyList(sp, list(prior_sigma = c(0.5, 2)))), "prior_sigma")
  expect_error(spatial_car(A, group_var = "g", prior_sigma = 1), "prior_sigma")
  expect_identical(spatial_bym2(A, group_var = "g",
                                prior_sigma = c(1, 0.01))$prior_sigma, c(1, 0.01))
})

test_that("an outer-grid knob a fitter does not read stops the call", {
  ks <- tulpa:::.CONTROL_KEYS
  expect_error(tulpa:::.control_subset(list(n_grid = 3), ks$nested_laplace,
                                       "nested_laplace"),
               "`n_grid`.*'nested_laplace'.*`spatial[$]tau_grid`")
  expect_identical(tulpa:::.control_subset(list(integration = "qmc"),
                                           ks$nested_laplace),
                   list(integration = "qmc"))
  expect_error(tulpa_nested_laplace(c(0, 1), c(1L, 1L), matrix(1, 2L, 1L),
    prior = list(list(type = "iid", n_units = 2L, obs_idx = 1:2)),
    family = "gaussian", phi = 1, control = list(integration = "ccd")),
    "grid.*qmc")
  expect_error(tulpa:::.control_subset(list(n_per_axis = 3), ks$gibbs, "gibbs"),
               "integrates no outer grid")
  expect_identical(tulpa:::.control_subset(list(adaptive_grid = TRUE),
                                           ks$nested_laplace),
                   list(adaptive_grid = TRUE))
  expect_error(tulpa:::.control_subset(list(adaptive_grid = TRUE, n_iter = 5),
                                       ks$nested_laplace, "nested_laplace"),
               "Unknown control knob.*nested_laplace.*n_iter")
  expect_identical(tulpa:::.control_subset(list(re_cov = "gibbs", seed = 1L),
                                           ks$re_cov_gibbs, "re_cov_gibbs",
                                           reads = "re_cov"),
                   list(seed = 1L))
  expect_identical(tulpa:::.control_subset(list(integration = "ccd"),
                                           ks$re_cov_nested),
                   list(integration = "ccd"))
  expect_identical(tulpa:::.control_subset(list(n_grid = 5), ks$spde),
                   list(n_grid = 5))
})

test_that("tulpa() lays an areal field's stated grid and refuses an unread knob", {
  skip_on_cran()
  fx <- .fe_data()
  f <- tulpa(y ~ x + spatial(region), fx$d, family = "poisson",
             spatial = list(type = "icar", adjacency = fx$A,
                            tau_grid = c(2, 4, 8)),
             mode = "nested_laplace",
             control = c(.fe_ctl, list(auto_recenter = FALSE,
                                       var_of_means_consistency = FALSE)))
  expect_equal(sort(unique(as.numeric(tulpa_outer_grid(f)$nodes[, 1]))),
               c(2, 4, 8))
  expect_error(
    tulpa(y ~ x + spatial(region), fx$d, family = "poisson",
          spatial = list(type = "icar", adjacency = fx$A),
          mode = "nested_laplace", control = c(.fe_ctl, list(n_grid = 3))),
    "`n_grid`")
  expect_error(
    tulpa(y ~ x + spatial(region), fx$d, family = "poisson",
          spatial = list(type = "icar", adjacency = fx$A, tau_grid = c(1, 2)),
          mode = "hmc"),
    "nested-Laplace hyperparameter integration")
})

test_that("an areal prior_sigma is the PC anchor of the fit's hyperprior", {
  skip_on_cran()
  fx <- .fe_data()
  f <- tulpa(y ~ x + spatial(region), fx$d, family = "poisson",
             spatial = spatial_car(fx$A, group_var = "region",
                                   prior_sigma = c(0.2, 0.05)),
             mode = "nested_laplace",
             control = c(.fe_ctl, list(auto_recenter = FALSE,
                                       var_of_means_consistency = FALSE)))
  tau <- as.numeric(tulpa:::.nl_theta_matrix(f)[, "tau"])
  expect_equal(as.numeric(f$log_hyperprior),
               tulpa:::.hp_log_scale_density(tau, "precision", c(0.2, 0.05)),
               tolerance = 1e-10)
})

test_that("a bare-list BYM2 field carries the scaling spatial_bym2() computes", {
  skip_on_cran()
  fx <- .fe_data()
  ctl <- c(.fe_ctl, list(auto_recenter = FALSE, var_of_means_consistency = FALSE))
  f_list <- tulpa(y ~ x + spatial(region), fx$d, family = "poisson",
                  spatial = list(type = "bym2", adjacency = fx$A),
                  mode = "nested_laplace", control = ctl)
  f_spec <- tulpa(y ~ x + spatial(region), fx$d, family = "poisson",
                  spatial = spatial_bym2(fx$A, group_var = "region"),
                  mode = "nested_laplace", control = ctl)
  expect_equal(f_list$prior$scale_factor,
               tulpa:::.bym2_component_scaling(fx$A)$scale_factor)
  expect_equal(as.numeric(logLik(f_list)), as.numeric(logLik(f_spec)))
  expect_equal(coef(f_list), coef(f_spec))
})

test_that("a nested BYM2 fit scores its inner Laplace and its k-hat is reproducible", {
  skip_on_cran()
  fx <- .fe_data()
  f2 <- tulpa(y ~ x + spatial(region), fx$d, family = "poisson",
              spatial = list(type = "bym2", adjacency = fx$A),
              mode = "nested_laplace", control = .fe_ctl)
  expect_true(all(is.finite(f2$inner_skew)))
  expect_true(is.na(f2$inner_skew_declined))
  run <- function() {
    set.seed(sample.int(1e6, 1L))
    tulpa(y ~ x + spatial(region), fx$d, family = "poisson",
          spatial = list(type = "icar", adjacency = fx$A),
          mode = "nested_laplace", control = .fe_ctl)$pareto_k
  }
  k1 <- run(); k2 <- run()
  expect_true(is.finite(k1))
  expect_identical(k1, k2)
})

test_that("adaptive_grid refines an NNGP grid's variance and range axes", {
  skip_on_cran()
  # The GP variance and range carry transform tags, so the registry door
  # refines them: the pass adds levels on both, recorded on the fit.
  set.seed(1)
  d <- data.frame(lon = runif(120), lat = runif(120), x = rnorm(120))
  d$y <- rpois(120, exp(0.5 + 0.3 * d$x))
  fit <- suppressWarnings(
    tulpa(y ~ x, d, family = "poisson", spatial = spatial_gp(~ lon + lat),
          mode = "nested_laplace",
          control = c(.fe_ctl, list(adaptive_grid = TRUE))))
  info <- fit$adaptive_grid_info
  expect_false(is.null(info))
  expect_setequal(strsplit(info$triggered_axes, ",")[[1]], c("sigma2", "phi_gp"))
  expect_true(all(c("sigma2", "phi_gp") %in% fit$refining_axis))
})
