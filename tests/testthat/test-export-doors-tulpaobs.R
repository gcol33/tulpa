# Public doors for internals tulpaObs (and other consumer packages) reached
# through `tulpa:::`, tracked in gcol33/tulpaObs#357. Each wrapper is a
# literal pass-through to the existing internal, so the test asserts (1) the
# door is exported and (2) it is equivalent to the internal it wraps -- not a
# re-test of the internal's own behaviour, which is covered elsewhere.

test_that("the nine tulpaObs-357 doors are exported", {
  doors <- c("tulpa_grid_axis", "tulpa_spde_precision_Q",
             "tulpa_spde_log_hyperprior", "tulpa_batched_pareto_k",
             "tulpa_theta_matrix", "tulpa_grid_log_quad",
             "tulpa_normalise_weights_safe", "tulpa_joint_axis_specs_from_grid",
             "tulpa_hyper_slice_home", "tulpa_hyper_grid_supports",
             "tulpa_hyper_copy_slab_density", "tulpa_hyper_check_copy_slab",
             "tulpa_joint_grid_batch")
  expect_true(all(doors %in% getNamespaceExports("tulpa")))
})

test_that("tulpa_grid_axis(#829) matches the internal default axis", {
  expect_identical(tulpa_grid_axis("field_sd"), tulpa:::.nl_grid_axis("field_sd"))
  expect_identical(tulpa_grid_axis("copy_alpha", n = 9L),
                    tulpa:::.nl_grid_axis("copy_alpha", 9L))
  expect_error(tulpa_grid_axis("not_a_key"), "Unknown default grid axis")
})

test_that("tulpa_spde_precision_Q / tulpa_spde_log_hyperprior(#830) match the internals", {
  set.seed(1)
  n <- 25
  L <- cbind(lon = stats::runif(n, 0, 10), lat = stats::runif(n, 0, 10))
  G <- data.frame(L, x = stats::rnorm(n))
  sp <- spatial_spde(~ lon + lat, data = G)
  kt <- tulpa:::.spde_kappa_tau(range = 2, sigma = 1, nu = sp$nu)

  Q1 <- tulpa_spde_precision_Q(sp, kt$kappa, kt$tau_spde)
  Q2 <- tulpa:::.spde_precision_Q(sp, kt$kappa, kt$tau_spde)
  expect_equal(as.matrix(Q1), as.matrix(Q2))

  lp1 <- tulpa_spde_log_hyperprior(range = c(1, 2, 3), sigma = c(0.5, 1, 1.5), sp = sp)
  lp2 <- tulpa:::.spde_log_hyperprior(range = c(1, 2, 3), sigma = c(0.5, 1, 1.5), sp = sp)
  expect_identical(lp1, lp2)
})

test_that("tulpa_batched_pareto_k(#831) matches .nested_is_pareto_k", {
  lt <- function(U) -0.5 * rowSums(U^2)
  set.seed(2)
  got <- tulpa_batched_pareto_k(theta_hat = c(0, 0), L_scale = diag(2),
                                 log_target_batched = lt, n_samples = 300,
                                 Z = matrix(stats::rnorm(600), 300, 2))
  set.seed(2)
  ref <- tulpa:::.nested_is_pareto_k(theta_hat = c(0, 0), L_scale = diag(2),
                                     log_target_batched = lt, n_samples = 300,
                                     Z = matrix(stats::rnorm(600), 300, 2))
  expect_identical(got, ref)
})

test_that("tulpa_theta_matrix / tulpa_grid_log_quad / tulpa_normalise_weights_safe (#832) reconstruct a fit's own weights", {
  skip_on_cran()
  set.seed(3)
  Tn <- 16L; per <- 3L
  ti <- rep(seq_len(Tn), each = per)
  d <- data.frame(tidx = ti, x = stats::rnorm(length(ti)),
                   g = rep(1:6, length.out = length(ti)))
  d$y <- stats::rpois(nrow(d), exp(0.3 + 0.5 * d$x + stats::rnorm(6, 0, 0.4)[d$g]))
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
               mode = "nested_laplace", temporal = temporal_rw1("tidx"))

  M1 <- tulpa_theta_matrix(fit)
  M2 <- tulpa:::.nl_theta_matrix(fit)
  expect_identical(M1, M2)

  lq1 <- tulpa_grid_log_quad(M1)
  lq2 <- tulpa:::.nl_grid_log_quad(M1)
  expect_identical(lq1, lq2)

  w1 <- tulpa_normalise_weights_safe(fit$log_marginal, log_quad = lq1)
  w2 <- tulpa:::.nl_normalise_weights_safe(fit$log_marginal, log_quad = lq2)
  expect_identical(w1, w2)
  expect_equal(w1, fit$weights)
})

test_that("tulpa_joint_axis_specs_from_grid / tulpa_hyper_slice_home / tulpa_hyper_grid_supports (#833) match the internals", {
  skip_on_cran()
  set.seed(4)
  Tn <- 16L; per <- 3L
  ti <- rep(seq_len(Tn), each = per)
  d <- data.frame(tidx = ti, x = stats::rnorm(length(ti)),
                   g = rep(1:6, length.out = length(ti)))
  d$y <- stats::rpois(nrow(d), exp(0.3 + 0.5 * d$x + stats::rnorm(6, 0, 0.4)[d$g]))
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
               mode = "nested_laplace", temporal = temporal_rw1("tidx"))
  M <- tulpa_theta_matrix(fit)

  specs1 <- tulpa_joint_axis_specs_from_grid(M)
  specs2 <- tulpa:::.joint_axis_specs_from_grid(M)
  expect_identical(specs1, specs2)

  home1 <- tulpa_hyper_slice_home(fit$refining, nrow(M))
  home2 <- tulpa:::.hyper_slice_home(fit$refining, nrow(M))
  expect_identical(home1, home2)

  sup1 <- tulpa_hyper_grid_supports(M, specs1, refining = fit$refining)
  sup2 <- tulpa:::.hyper_grid_supports(M, specs2, refining = fit$refining)
  expect_identical(sup1, sup2)
})

test_that("tulpa_hyper_copy_slab_density / tulpa_hyper_check_copy_slab (#833) match the internals", {
  f1 <- tulpa_hyper_copy_slab_density(3)
  f2 <- tulpa:::.hyper_copy_slab_density(3)
  expect_equal(f1(c(0, 1, 2)), f2(c(0, 1, 2)))
  expect_null(tulpa_hyper_copy_slab_density(-1))

  expect_identical(tulpa_hyper_check_copy_slab(NULL), "exponential")
  expect_identical(tulpa_hyper_check_copy_slab("flat"), "flat")
  expect_error(tulpa_hyper_check_copy_slab("bogus"), "copy_slab")
})

test_that("tulpa_joint_grid_batch(#834) is exported and validates its argument", {
  expect_error(tulpa_joint_grid_batch(list()), "non-empty list")
  expect_error(tulpa_joint_grid_batch(list(1)), "non-empty list")
})
