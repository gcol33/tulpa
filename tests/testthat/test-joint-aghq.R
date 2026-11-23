# The joint door's adaptive Gauss-Hermite inner marginal (`control$n_quad`): at
# a fixed covariance cell it is the profile tulpa_re_cov_nested(n_quad = k)
# solves, through the same compiled oracle, for a scalar, a diagonal and a
# correlated random-effect block.

skip_on_cran()

.jaq_data <- fixture_once(function() {
  set.seed(11)
  G <- 25L; per <- 8L; n <- G * per
  grp <- rep(seq_len(G), each = per); x <- rnorm(n)
  b <- cbind(rnorm(G, 0, 0.8), rnorm(G, 0, 0.5))
  y <- rbinom(n, 1L, plogis(-0.3 + 0.6 * x + b[grp, 1] + b[grp, 2] * x))
  list(y = y, x = x, grp = grp, G = G, n = n, X = cbind(1, x))
})

.jaq_arm <- function(d) {
  list(y = d$y, X = d$X, family = "binomial", n_trials = rep(1L, d$n),
       beta_prior_mean = rep(0, 2), beta_prior_prec = rep(1 / 2.5^2, 2))
}

.jaq_compare <- function(d, nq, re_terms, L_list, blocks) {
  core <- .eb_theta_fit(
    y = d$y, n_trials = rep(1L, d$n), X = d$X, re_terms = re_terms,
    family = "binomial", phi = 1, prior_sigma = NULL, eta = NULL,
    log_prior_theta = NULL, beta_prior = list(mean = 0, sd = 2.5),
    n_quad = nq, max_iter = 100L, tol = 1e-8, n_threads = 1L, caller = "t")
  fit <- tulpa_nested_laplace_joint(
    list(y = .jaq_arm(d)), prior = blocks,
    control = list(n_quad = nq, diagnose_k = FALSE, axis_refine = "none",
                   var_of_means_consistency = FALSE, progress = FALSE))
  c(re_cov = core$inner_logmarg(L_list),
    joint = fit$log_marginal - fit$log_hyperprior)
}

test_that("a fixed cell's quadrature marginal matches tulpa_re_cov_nested()", {
  skip_if_fast()
  d <- .jaq_data()
  L <- matrix(c(0.7, 0.2, 0, 0.45), 2, 2)
  th <- .re_L_to_logchol(L, 2L)
  corr <- .jaq_compare(d, 5L,
    list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$X, correlated = TRUE),
    list(L),
    list(list(type = "miid", obs_idx = d$grp, n_groups = d$G, n_fields = 2L,
              field_weight = list(list(rep(1, d$n)), list(d$x)),
              logchol_grid = matrix(th, 1L, dimnames = list(
                NULL, c("L11", "L21", "L22"))))))
  scal <- .jaq_compare(d, 7L, list(idx = d$grp, n_groups = d$G),
    list(matrix(0.7)),
    list(list(type = "iid", obs_idx = d$grp, n_units = d$G, sigma_grid = 0.7)))
  diagl <- .jaq_compare(d, 5L,
    list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$X, correlated = FALSE),
    list(diag(c(0.7, 0.45))),
    list(list(type = "iid", obs_idx = d$grp, n_units = d$G, sigma_grid = 0.7),
         list(type = "iid", obs_idx = d$grp, n_units = d$G, sigma_grid = 0.45,
              svc_weight = list(d$x))))
  for (r in list(corr, scal, diagl)) {
    expect_lt(abs(r[["joint"]] - r[["re_cov"]]) / abs(r[["re_cov"]]), 1e-8)
  }
})

test_that("a quadrature joint fit reports why it carries no random effects", {
  skip_if_fast()
  d <- .jaq_data()
  fit <- tulpa_nested_laplace_joint(
    list(y = .jaq_arm(d)),
    prior = list(list(type = "iid", obs_idx = d$grp, n_units = d$G)),
    control = list(n_quad = 3L, diagnose_k = FALSE, progress = FALSE))
  expect_identical(fit$n_quad, 3L)
  expect_true(all(is.finite(coef(fit))))
  expect_error(ranef(fit), "integrates each group's random effects out")
})

test_that("n_quad > 1 is refused where the groups do not factorize", {
  skip_if_fast()
  d <- .jaq_data()
  g2 <- rep(seq_len(10L), length.out = d$n)
  expect_error(tulpa_nested_laplace_joint(
    list(y = .jaq_arm(d)),
    prior = list(list(type = "iid", obs_idx = d$grp, n_units = d$G),
                 list(type = "iid", obs_idx = g2, n_units = 10L)),
    control = list(n_quad = 3L, progress = FALSE)),
    "more than one grouping factor")
  expect_error(tulpa_nested_laplace_joint(
    list(y = .jaq_arm(d)),
    prior = list(list(type = "iid", obs_idx = d$grp, n_units = d$G)),
    control = list(n_quad = 3L, subspace_debias = TRUE, progress = FALSE)),
    "need the Laplace inner solve")
})
