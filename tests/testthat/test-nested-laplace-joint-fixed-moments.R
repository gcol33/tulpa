# test-nested-laplace-joint-fixed-moments.R
# Per-cell fixed-effect retention on the joint nested-Laplace tier
# (gcol33/tulpa#305).
#
# Before this, `summary()` / `confint()` / `vcov()` on a
# `tulpa_nested_laplace_joint()` fit reported the point estimates and NA for
# every standard error and both bounds: `.nested_fixed_moments()` reads
# `$grid_modes` / `$grid_hessians`, and the joint driver stored neither. The
# tests here judge what it now stores against references computed OUTSIDE the
# engine, and pin that adding the retention changed no inference.
#
# Cross-tier recovery and coverage for the same feature live in
# test-nested-laplace-recovery.R, which runs the joint fitter through the one
# recovery sweep rather than a second harness.

# --------------------------------------------------------------------------- #
# (1) Multi-block joint, genuinely coupled arms: against the exact posterior   #
# --------------------------------------------------------------------------- #

# The #300 coupled fixture writes its own log posterior in R
# (`coupled_occ_log_post`, asserted equal to the compiled spec in
# test-cell-coupling-occupancy-mixture.R). Its latent block is reached by
# neither arm, so the conditional posterior factorises into the
# (beta_occ, beta_det) target and an independent Gaussian, and the inverse
# numerical Hessian of that target at the mode IS the fixed-effect covariance
# the inner Laplace is defined to produce -- an arbiter with no engine code in
# it, exact to the differencing error rather than to a Laplace error.
.j305_coupled_fit <- function(beta_prec = 0.25, seed = 4L, skew_correct = FALSE) {
  d <- coupled_occ_data(seed = seed, n_cells = 40L, n_visits = 3L,
                        b_occ = 0.3, b_det = -0.6)
  fit <- tulpa_nested_laplace_joint(
    responses = coupled_occ_arms(d, beta_prec = beta_prec),
    prior = coupled_occ_flat_prior(d),
    cell_coupling = "test_occupancy_mixture",
    control = list(max_iter = 100L, tol = 1e-10, diagnose_k = FALSE,
                   skew_correct = skew_correct))
  list(d = d, fit = fit, beta_prec = beta_prec)
}

test_that("a coupled multi-block joint fit reports the exact fixed-effect covariance", {
  skip_on_cran()
  skip_if_not_installed("numDeriv")
  coupled_occ_register()
  o <- .j305_coupled_fit()
  fit <- o$fit
  expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
  expect_true(is.na(fit$grid_fixed_declined))

  tab <- tulpa:::.fit_fixed_table(fit)
  expect_true(all(is.finite(tab$std.error)))
  expect_true(all(is.finite(tab$conf.low)))
  expect_true(all(is.finite(tab$conf.high)))

  lp <- coupled_occ_log_post(o$d, o$beta_prec)
  mode_ab <- fit$modes[1L, 1:2]
  # The engine's mode is a stationary point of that same density.
  expect_lt(max(abs(numDeriv::grad(function(v) lp(v[1], v[2]), mode_ab))), 1e-6)

  V_ref <- solve(-numDeriv::hessian(function(v) lp(v[1], v[2]), mode_ab))
  expect_equal(unname(vcov(fit)), V_ref, tolerance = 1e-6)
  expect_equal(tab$std.error, sqrt(diag(V_ref)), tolerance = 1e-6)
})

# --------------------------------------------------------------------------- #
# (2) Single-block joint over a real multi-cell grid: the mixture is the law   #
#     of total covariance, carrying BOTH within-cell curvature and between-    #
#     cell hyperparameter spread.                                             #
# --------------------------------------------------------------------------- #

.j305_icar_data <- function(seed = 21L, n_s = 24L, N = 260L) {
  set.seed(seed)
  rp <- integer(n_s + 1L); ci <- integer(0); nb <- integer(n_s)
  for (i in seq_len(n_s)) {
    nbrs <- c(if (i > 1L) i - 1L, if (i < n_s) i + 1L)
    ci <- c(ci, nbrs - 1L); nb[i] <- length(nbrs); rp[i + 1L] <- length(ci)
  }
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  phi_s <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.4))))
  x <- rnorm(N); Xocc <- cbind(1, x)
  occur <- rbinom(N, 1, plogis(as.numeric(Xocc %*% c(-0.3, 0.5)) +
                                phi_s[spatial_idx]))
  is_pos <- occur == 1L
  Xpos <- Xocc[is_pos, , drop = FALSE]; spi <- spatial_idx[is_pos]
  y_pos <- rnorm(sum(is_pos),
                 as.numeric(Xpos %*% c(0.2, -0.4)) + phi_s[spi], 0.5)
  list(
    prior = list(type = "icar", n_spatial_units = n_s, adj_row_ptr = rp,
                 adj_col_idx = ci, n_neighbors = nb,
                 sigma_grid = c(0.4, 0.9, 1.4)),
    responses = list(
      occ = list(y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
                 spatial_idx = spatial_idx, re_idx = rep(0, N),
                 n_re_groups = 0L, sigma_re = 1.0, family = "binomial",
                 phi = 1.0),
      pos = list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
                 spatial_idx = spi, re_idx = rep(0, length(y_pos)),
                 n_re_groups = 0L, sigma_re = 1.0, family = "gaussian",
                 phi = 0.5)))
}

.j305_icar_fit <- function(d, ...) {
  ctrl <- utils::modifyList(
    list(max_iter = 100L, tol = 1e-8, diagnose_k = FALSE), list(...))
  tulpa_nested_laplace_joint(d$responses, d$prior, control = ctrl)
}

test_that("the single-block joint mixture matches the independent law-of-total-covariance", {
  skip_on_cran()
  d <- .j305_icar_data()
  fit <- .j305_icar_fit(d)
  expect_s3_class(fit, "tulpa_nested_laplace_joint")
  expect_true(is.na(fit$grid_fixed_declined))
  # A grid that actually spreads its weight, so the between-cell term is real.
  expect_gt(length(fit$weights), 1L)
  expect_gt(sort(fit$weights, decreasing = TRUE)[2L], 0.01)

  # `.joint_mixture_moments()` is a separate R implementation of the same
  # mixture (Matrix sparse solves rather than the compiled per-cell extraction),
  # so agreeing with it exercises both the constrained-block extraction and the
  # marginalization against code that shares none of it.
  fq <- .j305_icar_fit(d, store_Q = TRUE)
  p <- fit$n_fixed
  mm <- tulpa:::.joint_mixture_moments(fq, idx = seq_len(p))
  expect_equal(unname(vcov(fit)), unname(mm$Sigma), tolerance = 1e-8)
  expect_equal(unname(coef(fit)), unname(mm$mean), tolerance = 1e-10)

  # The reported variance strictly exceeds the modal cell's own conditional
  # variance: the hyperparameter uncertainty is integrated in, not dropped.
  k_map <- which.max(fit$weights)
  v_cond <- diag(solve(fit$grid_hessians[[k_map]]))
  expect_true(all(diag(vcov(fit)) > v_cond))
})

test_that("the joint fixed-effect covariance is the one its own posterior draws carry", {
  skip_if_not_slow()
  d <- .j305_icar_data()
  fit <- .j305_icar_fit(d, store_Q = TRUE)
  p <- fit$n_fixed
  set.seed(99L)
  dr <- tulpa_posterior_draws(fit, idx = seq_len(p), n = 200000L)
  expect_equal(unname(apply(dr, 2, stats::sd)),
               unname(sqrt(diag(vcov(fit)))), tolerance = 0.02)
})

# --------------------------------------------------------------------------- #
# (3) The retention adds reporting and changes no inference.                   #
# --------------------------------------------------------------------------- #

test_that("keep_grid_hessians leaves modes, weights and log_marginal bit-for-bit", {
  skip_on_cran()
  d <- .j305_icar_data()
  on  <- .j305_icar_fit(d)
  off <- .j305_icar_fit(d, keep_grid_hessians = FALSE)

  expect_identical(on$modes,        off$modes)
  expect_identical(on$weights,      off$weights)
  expect_identical(on$log_marginal, off$log_marginal)
  expect_identical(on$theta_grid,   off$theta_grid)
  expect_identical(on$theta_mean,   off$theta_mean)
  expect_identical(on$theta_sd,     off$theta_sd)
  expect_identical(on$inner_skew,   off$inner_skew)

  # Switching it off is the pre-#305 report, and says so rather than going
  # quiet about it.
  expect_identical(off$grid_fixed_declined, "not_requested")
  expect_true(all(is.na(tulpa:::.fit_fixed_table(off)$std.error)))
  expect_null(tulpa:::.nested_fixed_moments(off))
})

test_that("the cell precision is not carried home unless store_Q asked for it", {
  skip_on_cran()
  d <- .j305_icar_data()
  fit <- .j305_icar_fit(d)
  expect_null(fit$Q_csc_p_per_grid)
  expect_null(fit$Q_csc_n)
  # The retained pieces are the fixed-effect block only, one per cell.
  expect_length(fit$grid_hessians, length(fit$weights))
  expect_length(fit$grid_modes,    length(fit$weights))
  for (k in seq_along(fit$weights)) {
    expect_identical(dim(fit$grid_hessians[[k]]),
                     c(fit$n_fixed, fit$n_fixed))
    expect_length(fit$grid_modes[[k]], fit$n_fixed)
  }
  expect_true(!is.null(.j305_icar_fit(d, store_Q = TRUE)$Q_csc_p_per_grid))
})

# --------------------------------------------------------------------------- #
# (4) What #305 was blocking: the #302 skew correction is now visible.         #
# --------------------------------------------------------------------------- #

test_that("the inner-Laplace skew correction reaches a separable joint fit's intervals", {
  skip_on_cran()
  # Both arms of this fixture sum per observation, so the location term
  # gcol33/tulpa#354 requires is a complete sum and the correction runs.
  d <- .j305_icar_data()
  gauss <- .j305_icar_fit(d, skew_correct = FALSE)
  skew  <- .j305_icar_fit(d, skew_correct = TRUE)

  sc <- skew$skew_correction
  expect_true(all(is.finite(sc$gamma3)))
  expect_true(all(is.finite(sc$gamma1)))
  expect_true(all(sc$eligible))
  # Both terms sum over EVERY arm, so a gaussian-arm coefficient is not scored
  # at 0 here: it reads the binomial arm's third derivatives through the shared
  # ICAR field. Its own arm contributes nothing, which is why the values are
  # an order of magnitude below the binomial arm's.
  expect_lt(max(abs(sc$gamma3[3:4])), 0.1 * max(abs(sc$gamma3[1:2])))

  ci_g <- confint(gauss)
  ci_s <- confint(skew)
  expect_true(all(is.finite(ci_g)))
  expect_true(all(is.finite(ci_s)))
  expect_identical(unname(attr(ci_s, "skew_applied")), sc$eligible)
  expect_false(any(attr(ci_g, "skew_applied")))
  # The corrected interval is a DIFFERENT interval on the binomial arm, not a
  # silently identical one.
  expect_gt(max(abs(ci_s[1:2, ] - ci_g[1:2, ])), 1e-6)

  # Correcting the report leaves the fit that produced it untouched.
  expect_identical(gauss$modes,   skew$modes)
  expect_identical(gauss$weights, skew$weights)
})

test_that("a coupled joint fit declines the correction for want of a location term", {
  skip_on_cran()
  coupled_occ_register()
  gauss <- .j305_coupled_fit(skew_correct = FALSE)$fit
  skew  <- .j305_coupled_fit(skew_correct = TRUE)$fit

  # Every arm is coupled through the cell spec, so the cubic term is scored
  # through the widened contraction (gcol33/tulpa#301) but the location term's
  # contraction against a covariance BLOCK is not reachable from that oracle.
  # The correction declines rather than reading the absent gamma_1 as zero.
  sc <- skew$skew_correction
  expect_true(all(is.finite(sc$gamma3)))
  expect_true(all(is.na(sc$gamma1)))
  expect_identical(sc$gamma1_declined, "multi_eta_unit")
  expect_identical(sc$reason, rep("gamma1_not_computable", length(sc$reason)))
  expect_false(any(sc$eligible))

  ci_g <- confint(gauss)
  ci_s <- confint(skew)
  expect_true(all(is.finite(ci_s)))
  expect_false(any(attr(ci_s, "skew_applied")))

  # With the correction off the bounds are exactly the Gaussian ones, and a
  # declined correction reports the same numbers rather than a partial one.
  tab <- tulpa:::.fit_fixed_table(gauss)
  z <- stats::qnorm(0.975)
  expect_equal(tab$conf.low,  tab$estimate - z * tab$std.error, tolerance = 1e-12)
  expect_equal(tab$conf.high, tab$estimate + z * tab$std.error, tolerance = 1e-12)
  expect_equal(matrix(as.numeric(ci_s), nrow(ci_s)),
               matrix(as.numeric(ci_g), nrow(ci_g)), tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# (5) Which cells the shared marginalizer sums over, and under what weights.   #
# --------------------------------------------------------------------------- #

test_that(".nested_fixed_moments ignores zero-weight and empty cells", {
  H <- matrix(c(4, 1, 1, 2), 2, 2)
  base <- list(weights = c(0.25, 0.75),
               grid_modes = list(c(1, 2), c(3, 4)),
               grid_hessians = list(H, H))
  ref <- tulpa:::.nested_fixed_moments(base)

  # A third cell with zero weight contributes nothing, whether or not it
  # carries a block.
  padded <- list(weights = c(0.25, 0.75, 0),
                 grid_modes = list(c(1, 2), c(3, 4), NULL),
                 grid_hessians = list(H, H, NULL))
  got <- tulpa:::.nested_fixed_moments(padded)
  expect_equal(got$mean, ref$mean)
  expect_equal(got$cov,  ref$cov)

  # A per-cell array that does not describe the grid is refused outright
  # rather than recycled against it.
  expect_null(tulpa:::.nested_fixed_moments(
    list(weights = c(0.5, 0.5), grid_modes = list(c(1, 2)),
         grid_hessians = list(H))))

  # A complete grid retains all of its weight.
  expect_equal(ref$mass, 1, tolerance = 1e-14)
  expect_equal(got$mass, 1, tolerance = 1e-14)
})

# A cell that carries POSITIVE weight and no retained block is the other case
# (gcol33/tulpa#342). Its mass is gone, so the moments are renormalized over
# what survived and report the posterior conditional on those cells; summing the
# whole-grid weights over the survivors instead would shrink the mean toward the
# origin by exactly the dropped mass.
test_that(".nested_fixed_moments renormalizes over the cells it kept", {
  # The symmetric reproduction from the issue: two identical cells, one of them
  # dropped. The truth is unmoved, and the pre-#342 answer was half of it.
  H <- matrix(c(4, 1, 1, 2), 2, 2)
  full <- list(weights = c(0.5, 0.5),
               grid_modes = list(c(1, 2), c(1, 2)),
               grid_hessians = list(H, H))
  holed <- full
  holed$grid_hessians[2] <- list(NULL)

  expect_equal(tulpa:::.nested_fixed_moments(full)$mean, c(1, 2),
               tolerance = 1e-14)
  expect_equal(tulpa:::.nested_fixed_moments(holed)$mean, c(1, 2),
               tolerance = 1e-14)
  expect_equal(tulpa:::.nested_fixed_moments(holed)$cov, solve(H),
               tolerance = 1e-12)

  # That fixture passes under any renormalization, because its survivors share
  # one mode. The discriminating case is a dropped cell whose mode and weight
  # both differ from the survivors': there the answer is the grid that never
  # held it, entry for entry.
  H1 <- matrix(c(4,  1.0,  1.0, 2.0), 2, 2)
  H2 <- matrix(c(3, -0.5, -0.5, 5.0), 2, 2)
  H3 <- matrix(c(6,  0.25, 0.25, 1.5), 2, 2)
  grid <- list(weights = c(0.2, 0.5, 0.3),
               grid_modes = list(c(-1, 0.5), c(2, -3), c(0.25, 4)),
               grid_hessians = list(H1, H2, H3))
  dropped <- grid
  dropped$grid_hessians[3] <- list(NULL)
  never <- list(weights = c(0.2, 0.5),
                grid_modes = grid$grid_modes[1:2],
                grid_hessians = list(H1, H2))

  a <- tulpa:::.nested_fixed_moments(dropped)
  b <- tulpa:::.nested_fixed_moments(never)
  expect_equal(a$mean, b$mean, tolerance = 1e-14)
  expect_equal(a$cov,  b$cov,  tolerance = 1e-14)
  expect_equal(a$mu,   b$mu,   tolerance = 1e-14)
  expect_equal(a$var,  b$var,  tolerance = 1e-14)
  expect_equal(a$w,    b$w,    tolerance = 1e-14)

  # Against the law of total covariance by hand at the renormalized weights.
  wk <- c(0.2, 0.5) / 0.7
  mu <- rbind(c(-1, 0.5), c(2, -3))
  m  <- as.numeric(crossprod(wk, mu))
  S  <- wk[1] * (solve(H1) + tcrossprod(mu[1, ])) +
        wk[2] * (solve(H2) + tcrossprod(mu[2, ]))
  expect_equal(a$mean, m, tolerance = 1e-14)
  expect_equal(a$cov, S - tcrossprod(m), tolerance = 1e-14)
  expect_equal(a$w, wk, tolerance = 1e-14)

  # The weights the moments were formed under sum to one; the mass they were
  # renormalized from is the reporting channel, and separates the repaired grid
  # from the one that never held the cell.
  expect_equal(sum(a$w), 1, tolerance = 1e-14)
  expect_equal(a$mass, 0.7, tolerance = 1e-14)
  expect_equal(b$mass, 1,   tolerance = 1e-14)

  # Dropping every cell but one leaves that cell's own Gaussian.
  one <- grid
  one$grid_modes[c(1, 3)] <- list(NULL)
  o <- tulpa:::.nested_fixed_moments(one)
  solo <- tulpa:::.nested_fixed_moments(
    list(weights = 1, grid_modes = list(c(2, -3)), grid_hessians = list(H2)))
  expect_equal(o$mean, solo$mean, tolerance = 1e-14)
  expect_equal(o$cov,  solo$cov,  tolerance = 1e-14)
  expect_equal(o$mean, c(2, -3),  tolerance = 1e-14)
  expect_equal(o$cov,  solve(H2), tolerance = 1e-12)
  expect_equal(o$mass, 0.5, tolerance = 1e-14)
})

# --------------------------------------------------------------------------- #
# (6) The block is extracted inside each cell's own solve (gcol33/tulpa#307).  #
# --------------------------------------------------------------------------- #

# Reference: the block read the way #305 read it -- off the STORED per-cell
# precision, through the R-level `cpp_joint_inner_vcov_blocks()` entry. The
# in-loop extraction hands the same cell CSC to the same routine, so the two
# agree entry for entry, not to a tolerance.
.j307_reference_blocks <- function(fq) {
  p <- fq$n_fixed
  n_x <- as.integer(fq$Q_csc_n)
  cpp_joint_inner_vcov_blocks(
    fq$Q_csc_p_per_grid, fq$Q_csc_i_per_grid, fq$Q_csc_x_per_grid,
    n_x = n_x, idx = seq_len(p), n_dense = p,
    A_cols_list = tulpa:::.joint_constraint_cols(fq$arm_layout, n_x),
    field_marginal = FALSE, n_threads = 1L)
}

test_that("the in-loop block is the stored-precision block, entry for entry", {
  skip_on_cran()
  d <- .j305_icar_data()
  fq <- .j305_icar_fit(d, store_Q = TRUE)
  V <- .j307_reference_blocks(fq)
  expect_length(V, length(fq$weights))

  for (k in seq_along(V)) {
    expect_identical(fq$grid_hessians[[k]], solve(V[[k]]))
  }

  # The probe is live: the same routine WITHOUT the field sum-to-zero groups
  # returns a materially different block, so the exact agreement above is the
  # constrained extraction matching, not two ways of reading nothing.
  n_x <- as.integer(fq$Q_csc_n)
  V_unc <- cpp_joint_inner_vcov_blocks(
    fq$Q_csc_p_per_grid, fq$Q_csc_i_per_grid, fq$Q_csc_x_per_grid,
    n_x = n_x, idx = seq_len(fq$n_fixed), n_dense = fq$n_fixed,
    A_cols_list = list(), field_marginal = FALSE, n_threads = 1L)
  expect_gt(max(abs(V_unc[[1L]] - V[[1L]])), 1e-6)
})

test_that("a coupled multi-block joint fit extracts the same block in the loop", {
  skip_on_cran()
  coupled_occ_register()
  d <- coupled_occ_data(seed = 4L, n_cells = 40L, n_visits = 3L,
                        b_occ = 0.3, b_det = -0.6)
  fq <- tulpa_nested_laplace_joint(
    responses = coupled_occ_arms(d, beta_prec = 0.25),
    prior = coupled_occ_flat_prior(d),
    cell_coupling = "test_occupancy_mixture",
    control = list(max_iter = 100L, tol = 1e-10, diagnose_k = FALSE,
                   store_Q = TRUE))
  V <- .j307_reference_blocks(fq)
  for (k in seq_along(V)) expect_identical(fq$grid_hessians[[k]], solve(V[[k]]))
})

test_that("the retention no longer needs the precision materialised at all", {
  skip_on_cran()
  d <- .j305_icar_data()
  fit <- .j305_icar_fit(d)
  # Nothing anywhere in the pipeline stored a precision, and the block is still
  # there: before #307 the kernel had to keep every cell's precision to reach it.
  expect_null(fit$Q_csc_p_per_grid)
  expect_null(fit$cov_block_per_grid)
  expect_true(is.na(fit$grid_fixed_declined))
  expect_true(all(is.finite(vcov(fit))))

  # And the reported numbers do not depend on whether the precision was kept.
  fq <- .j305_icar_fit(d, store_Q = TRUE)
  expect_identical(vcov(fit), vcov(fq))
  expect_identical(confint(fit), confint(fq))
  expect_identical(fit$grid_hessians, fq$grid_hessians)
})

# The sparse joint Newton loop takes a different route to the same block: it
# reads the builder's own CSC instead of converting a dense H, and its live
# factor after the final PD-enforced solve is not one the block could be read
# off -- which is why the extraction factorizes the cell's own CSC.
test_that("the sparse joint loop extracts the same block as the dense one", {
  skip_on_cran()
  d <- .j305_icar_data()
  dense  <- .j305_icar_fit(d, force_sparse = FALSE, store_Q = TRUE)
  sparse <- .j305_icar_fit(d, force_sparse = TRUE,  store_Q = TRUE)
  expect_equal(unname(vcov(dense)), unname(vcov(sparse)), tolerance = 1e-8)
  for (k in seq_along(sparse$weights)) {
    expect_identical(sparse$grid_hessians[[k]],
                     solve(.j307_reference_blocks(sparse)[[k]]))
  }
})

# --------------------------------------------------------------------------- #
# (7) The bounds this retention feeds are the mixture's, not the collapse's    #
#     (gcol33/tulpa#336).                                                      #
# --------------------------------------------------------------------------- #

test_that("a real multi-cell fit reports mixture-CDF bounds beside its moments", {
  skip_on_cran()
  d   <- .j305_icar_data()
  fit <- .j305_icar_fit(d)
  expect_gt(length(fit$weights), 1L)

  ci <- confint(fit)
  expect_identical(attr(ci, "interval_source"), "mixture_cdf")
  expect_true(is.na(attr(ci, "interval_declined")))

  # The bounds solve the mixture CDF assembled here from the fit's own retained
  # cells, so what `summary()` reports is checked against the definition rather
  # than against the helper that produced it.
  p <- fit$n_fixed
  w <- fit$weights / sum(fit$weights)
  mu <- vapply(fit$grid_modes, function(m) m[seq_len(p)], numeric(p))
  sd <- vapply(fit$grid_hessians, function(H) sqrt(diag(solve(H))), numeric(p))
  for (j in seq_len(p)) {
    cdf <- function(b) sum(w * stats::pnorm(b, mu[j, ], sd[j, ]))
    expect_equal(cdf(ci[j, 1]), 0.025, tolerance = 1e-8)
    expect_equal(cdf(ci[j, 2]), 0.975, tolerance = 1e-8)
  }

  # The moments are untouched, so the two reads differ only in the quantile
  # step -- and on a grid this spread they do differ.
  est <- coef(fit); se <- summary(fit)$std.error
  gauss <- cbind(est + stats::qnorm(0.025) * se, est + stats::qnorm(0.975) * se)
  expect_gt(max(abs(matrix(as.numeric(ci), p, 2L) - gauss)), 1e-6)
})

test_that("the mixture bounds are the ones the fit's own posterior draws carry", {
  skip_if_not_slow()
  d   <- .j305_icar_data()
  fit <- .j305_icar_fit(d, store_Q = TRUE)
  p   <- fit$n_fixed
  # `tulpa_posterior_draws()` samples a cell by weight and then that cell's
  # Gaussian, so it realizes the same mixture the bounds invert -- an arbiter
  # that shares no code with the quantile path.
  set.seed(101L)
  dr <- tulpa_posterior_draws(fit, idx = seq_len(p), n = 400000L)
  emp <- t(apply(dr, 2L, stats::quantile, c(0.025, 0.975)))
  ci  <- matrix(as.numeric(confint(fit)), p, 2L)
  expect_equal(ci, unname(emp), tolerance = 0.02)

  # And the collapsed Gaussian read is the one that misses them.
  est <- coef(fit); se <- summary(fit)$std.error
  gauss <- cbind(est + stats::qnorm(0.025) * se, est + stats::qnorm(0.975) * se)
  expect_gt(max(abs(gauss - emp)), max(abs(ci - emp)))
})
