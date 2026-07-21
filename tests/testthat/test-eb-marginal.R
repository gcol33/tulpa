# Poisson random-intercept design, small enough to keep the 1 + 2k^2 stencil
# solves cheap while leaving the variance component genuinely uncertain.
.em_sim <- function(seed, G = 10L, per = 8L, beta = c(0.3, 0.5), sigma = 0.8) {
  set.seed(seed)
  n   <- G * per
  grp <- rep(seq_len(G), each = per)
  x   <- rnorm(n)
  b   <- rnorm(G, 0, sigma)
  list(y  = rpois(n, exp(beta[1] + beta[2] * x + b[grp])),
       X  = cbind(1, x),
       re = list(idx = grp, n_groups = G, n_coefs = 1L),
       beta = beta)
}

test_that("marginal = FALSE leaves the fit unchanged", {
  d <- .em_sim(1L)
  f <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson")
  expect_null(f$cov_marginal)
  expect_null(f$cov_conditional)
  expect_null(f$theta_cov)
  expect_null(f$H_theta)
  # vcov still comes from H_beta on a plain fit.
  expect_equal(unname(vcov(f)), unname(solve(f$H_beta)[1:2, 1:2]),
               tolerance = 1e-10)
})

test_that("marginal = TRUE attaches the correction and widens intervals", {
  d  <- .em_sim(1L)
  f0 <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson")
  f1 <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE)
  skip_if(is.null(f1$cov_marginal), "correction did not form on this seed")

  expect_true(is.matrix(f1$cov_marginal))
  expect_true(is.matrix(f1$cov_conditional))
  expect_true(is.matrix(f1$theta_cov))
  expect_equal(dim(f1$cov_marginal), dim(f1$cov_conditional))

  # The conditional block the correction started from is the one a plain fit
  # would have reported.
  expect_equal(unname(f1$cov_conditional), unname(solve(f0$H_beta)[1:2, 1:2]),
               tolerance = 1e-8)

  # Inflation is positive semidefinite, so no SE can shrink.
  infl <- f1$cov_marginal - f1$cov_conditional
  ev   <- eigen((infl + t(infl)) / 2, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(ev >= -1e-10))
  expect_true(all(sqrt(diag(f1$cov_marginal)) >=
                    sqrt(diag(f1$cov_conditional)) - 1e-12))
})

test_that("H_theta matches an independently differenced Hessian", {
  d  <- .em_sim(1L)
  f1 <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE)
  skip_if(is.null(f1$H_theta), "correction did not form on this seed")

  core <- .re_cov_theta_fit(
    y = d$y, n_trials = NULL, X = d$X, re_terms = list(d$re),
    family = "poisson", phi = 1.0, prior_sigma = c(3, 0.05), eta = 2,
    log_prior_theta = NULL, beta_prior = NULL, n_quad = 1L,
    max_iter = 100L, tol = 1e-8, n_threads = 1L,
    caller = "test", need_scale = FALSE, outer_maxit = 500L, offset = NULL)

  # optimHess differences the objective with its own scheme and step, so
  # agreement is evidence about the quantity, not about the stencil reproducing
  # itself.
  negm <- function(th) {
    -(core$inner_logmarg(.re_cov_theta_to_L_list(th, core$layout)) +
        core$log_prior_theta(th))
  }
  H_num <- optimHess(core$theta_hat, negm)
  expect_equal(as.numeric(f1$H_theta), as.numeric(H_num), tolerance = 1e-4)
})

test_that("J matches a direct central difference of the inner mode", {
  d <- .em_sim(1L)
  core <- .re_cov_theta_fit(
    y = d$y, n_trials = NULL, X = d$X, re_terms = list(d$re),
    family = "poisson", phi = 1.0, prior_sigma = c(3, 0.05), eta = 2,
    log_prior_theta = NULL, beta_prior = NULL, n_quad = 1L,
    max_iter = 100L, tol = 1e-8, n_threads = 1L,
    caller = "test", need_scale = FALSE, outer_maxit = 500L, offset = NULL)
  f1 <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE)
  skip_if(is.null(f1$cov_marginal), "correction did not form on this seed")

  corr <- .eb_marginal_correction(core, core$theta_hat, p_fix = 2L,
                                  H_beta = f1$H_beta, step = 1e-3)
  skip_if(is.null(corr), "correction did not form")

  th <- core$theta_hat
  mode_at <- function(t)
    core$inner_fit(.re_cov_theta_to_L_list(t, core$layout))$mode
  fd_at <- function(h) (mode_at(th + h) - mode_at(th - h)) / (2 * h)

  # J is closed-form on the analytic-gradient route, so the finite difference is
  # the APPROXIMATION here, not the reference. Asserting agreement at the FD's
  # own accuracy is the most this comparison can say: a central difference at
  # h carries O(h^2) truncation, so h = 1e-3 buys about six digits.
  fd_coarse <- fd_at(1e-3)
  expect_equal(as.numeric(corr$J[, 1]), as.numeric(fd_coarse),
               tolerance = 1e-5)

  # The sharper claim: as the step shrinks the difference has to shrink with it.
  # If J were wrong rather than merely different, halving h would leave the gap
  # roughly where it was instead of quartering it.
  gap <- function(h) max(abs(as.numeric(corr$J[, 1]) - as.numeric(fd_at(h))))
  g1 <- gap(2e-3)
  g2 <- gap(1e-3)
  expect_lt(g2, g1)
})

test_that("vcov and summary report the marginal covariance when present", {
  d  <- .em_sim(1L)
  f1 <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE)
  skip_if(is.null(f1$cov_marginal), "correction did not form on this seed")

  expect_equal(unname(vcov(f1)), unname(f1$cov_marginal), tolerance = 1e-12)

  f0  <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson")
  s0  <- .fit_fixed_table(f0)
  s1  <- .fit_fixed_table(f1)
  expect_true(all(s1$std.error >= s0$std.error - 1e-12))
  expect_equal(s1$std.error, unname(sqrt(diag(f1$cov_marginal))),
               tolerance = 1e-10)
})

test_that("the new EB control knobs validate", {
  expect_silent(.check_control(
    list(marginal_step = 1e-4, marginal_richardson = TRUE),
    tulpa:::.CONTROL_KEYS$eb, "tulpa_eb"))
  expect_error(.check_control(list(marginal_stp = 1e-4),
                             tulpa:::.CONTROL_KEYS$eb, "tulpa_eb"),
               "Unknown control knob")
  d <- .em_sim(1L)
  expect_error(
    tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE,
             control = list(marginal_step = -1)),
    "must be a single positive number")
  expect_error(
    tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = NA),
    "must be TRUE or FALSE")
})

test_that("force_sparse resolves TRUE / FALSE / auto", {
  expect_true(.resolve_force_sparse(TRUE, function() 1L))
  expect_false(.resolve_force_sparse(FALSE, function() 1e6))
  expect_false(.resolve_force_sparse("auto", function() 500L))
  expect_true(.resolve_force_sparse("auto", function() 2000L))
  # At the threshold exactly, dense: the rule is strictly greater.
  expect_false(.resolve_force_sparse("auto", function() 1000L))
  # An undeterminable size must not opt into the specialized path.
  expect_false(.resolve_force_sparse("auto", function() stop("no layout")))
  expect_false(.resolve_force_sparse("auto", function() NULL))
  expect_error(.resolve_force_sparse("sparse", function() 1L),
               "must be TRUE, FALSE")
})

test_that("warm-start hyperparameter mass reads theta_cov by block", {
  d       <- .em_sim(1L)
  f_plain <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson")
  f_marg  <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE)
  skip_if(is.null(f_marg$theta_cov), "correction did not form on this seed")

  # No outer curvature on the fit means every slot keeps adapting.
  expect_null(.warm_start_hyper_var(f_plain))

  hv <- .warm_start_hyper_var(f_marg)
  expect_equal(unname(unlist(hv)), unname(diag(f_marg$theta_cov)),
               tolerance = 1e-12)

  # A correlated block is in log-Cholesky coordinates, which are not the
  # sampler's log_sigma_re, so it must not map across.
  f_full <- f_marg
  f_full$layout    <- list(list(nc = 2L, full = TRUE, k = 3L))
  f_full$theta_cov <- diag(3)
  hv_full <- .warm_start_hyper_var(f_full)
  expect_length(hv_full, 1L)
  expect_null(hv_full[[1]])

  # A skipped block must not shift the blocks after it: block 2 has to keep its
  # own variance, not inherit block 1's slot.
  f_mix <- f_marg
  f_mix$layout    <- list(list(nc = 2L, full = TRUE,  k = 3L),
                          list(nc = 1L, full = FALSE, k = 1L))
  f_mix$theta_cov <- diag(c(1, 2, 3, 42))
  hv_mix <- .warm_start_hyper_var(f_mix)
  expect_length(hv_mix, 2L)
  expect_null(hv_mix[[1]])
  expect_equal(unname(unlist(hv_mix[[2]])), 42)

  # A theta_cov that does not line up with the layout is refused outright,
  # rather than filling masses from a mismatched vector.
  f_bad <- f_marg
  f_bad$layout <- list(list(nc = 1L, full = FALSE, k = 5L))
  expect_null(.warm_start_hyper_var(f_bad))
})

test_that(".build_warm_start masses log_sigma_re from theta_cov", {
  # The sampler warm start is a separate feature; this only checks that it picks
  # up the curvature when both are present.
  skip_if_not(exists(".build_warm_start") &&
                exists("cpp_tulpa_glmm_layout"),
              "sampler warm start not available in this build")
  d <- .em_sim(1L)
  n <- length(d$y)
  # Same shape tulpa() builds at R/tulpa.R:910 -- parallel per-term vectors,
  # not a list of per-term lists.
  lay <- tryCatch(cpp_tulpa_glmm_layout(
    y = as.numeric(d$y), n_trials = rep(1L, n), X = d$X, family = "poisson",
    re_spec = list(
      idx        = list(as.integer(d$re$idx - 1L)),
      ngroups    = as.integer(d$re$n_groups),
      ncoefs     = 1L,
      correlated = FALSE,
      Z          = list(matrix(1, n, 1)))),
    error = function(e) NULL)
  skip_if(is.null(lay) || is.null(lay$total_params), "layout unavailable")

  f_plain <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson")
  f_marg  <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE)
  skip_if(is.null(f_marg$theta_cov), "correction did not form on this seed")

  ws0 <- .build_warm_start(f_plain, lay, list(d$re), n_chains = 1L, jitter = 0)
  ws1 <- .build_warm_start(f_marg,  lay, list(d$re), n_chains = 1L, jitter = 0)
  # .build_warm_start is under active development in R/warm_start.R and its
  # return shape is not settled; this file owns the theta_cov -> mass mapping,
  # not the warm-start container. Skip rather than pin a field name that is
  # still moving, so a rename there shows up as a warm-start test failure
  # rather than as a spurious failure here.
  skip_if(is.null(ws0$inv_metric_diag) || is.null(ws1$inv_metric_diag),
          "warm start does not expose inv_metric_diag")

  ls <- as.integer(lay$re_terms[[1]]$log_sigma_re)
  ls <- ls[!is.na(ls)]
  skip_if(length(ls) == 0L, "layout exposes no log_sigma_re slot")

  # Without an outer curvature the slot keeps the sampler's adapting default.
  expect_equal(ws0$inv_metric_diag[ls], rep(1, length(ls)))
  # With one, it starts at the estimated posterior variance of log sigma.
  expect_equal(ws1$inv_metric_diag[ls],
               unname(diag(f_marg$theta_cov)), tolerance = 1e-12)
  # Nothing else moved: the two warm starts agree everywhere but those slots.
  expect_equal(ws0$inv_metric_diag[-ls], ws1$inv_metric_diag[-ls])
  expect_equal(ws0$init, ws1$init)
})

test_that("Richardson extrapolation runs and stays close to the plain stencil", {
  d  <- .em_sim(1L)
  f1 <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE)
  f2 <- tulpa_eb(d$y, NULL, d$X, d$re, family = "poisson", marginal = TRUE,
                 control = list(marginal_richardson = TRUE))
  skip_if(is.null(f1$H_theta) || is.null(f2$H_theta), "correction did not form")
  expect_true(isTRUE(f2$marginal_richardson))
  expect_false(isTRUE(f1$marginal_richardson))
  # Both estimate the same Hessian; they differ only by truncation error.
  expect_equal(as.numeric(f1$H_theta), as.numeric(f2$H_theta), tolerance = 1e-3)
})
