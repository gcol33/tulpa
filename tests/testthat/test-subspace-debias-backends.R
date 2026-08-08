# test-subspace-debias-backends.R
# The subspace debias on the GRID and JOINT nested backends (gcol33/tulpa#306),
# the follow-up to #304's `tulpa_re_cov_nested()` / `tulpa_laplace()` wiring.
#
# What is different here, and what these tests have to establish because of it:
#
#  * The selector's input is already on the fit. `control$diagnose_skew` (on by
#    default) re-dispatches the kernel at the fitted MAP cell and attaches both
#    inner scores, so `.subspace_bands()` reads them rather than paying for a
#    probe of its own. That the bands a nested fit reports are the bands the
#    selector acts on is asserted, not assumed.
#  * The correction is applied PER CELL and the fit then reports draws instead
#    of the Gaussian-mixture moments, so the arbiter is the exact conditional
#    posterior, computed by quadrature outside the engine.
#  * Empty S has to leave every backend bit-for-bit as it was.
#
# Both fixtures put the latent block where no observation reaches it, so the
# conditional posterior factorises into the fixed-effect target and an
# independent Gaussian and a two-dimensional quadrature of that target is EXACT
# for the reported coefficients -- the same construction the #300 coupled
# fixture uses, here on the single-arm grid driver as well.

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

# Rare-event binomial logit, unreached iid block: the regime the inner Laplace
# is skewed in, on the multi-block grid driver (run_multi_block_nested_laplace).
.sd306_rare_data <- function(seed = 3L, n = 120L, b = c(-3.2, 0.9)) {
  set.seed(seed)
  x <- rnorm(n)
  y <- rbinom(n, 1L, stats::plogis(b[1] + b[2] * x))
  list(y = y, n = n, x = x, X = cbind(`(Intercept)` = 1, x = x),
       prior = list(list(type = "iid", n_units = 1L, sigma_grid = 1.0,
                         obs_idx = rep(0L, n))))
}

.sd306_rare_fit <- function(d, ctrl = list()) {
  tulpa_nested_laplace(
    d$y, rep(1L, d$n), d$X, prior = d$prior, family = "binomial",
    control = utils::modifyList(
      list(max_iter = 100L, tol = 1e-10, keep_grid_hessians = TRUE,
           diagnose_k = FALSE, progress = FALSE), ctrl))
}

# The engine's own target for that fit: binomial log-lik plus the driver's
# N(0, 100^2) ridge on beta (sigma_beta = 100 in the multi-block driver).
.sd306_rare_log_post <- function(d) {
  function(g0, g1) {
    M <- matrix(NA_real_, length(g0), length(g1))
    for (i in seq_along(g0)) for (j in seq_along(g1)) {
      eta <- g0[i] + g1[j] * d$x
      M[i, j] <- sum(d$y * eta - log1p(exp(eta))) -
        0.5e-4 * (g0[i]^2 + g1[j]^2)
    }
    M
  }
}

# Exact marginal quantiles of a two-dimensional log posterior, by quadrature.
# The quantile-side companion of `coupled_occ_quadrature()`, which returns the
# moments of the same construction.
.sd306_exact_quantiles <- function(log_post_grid, center, probs,
                                   half = 6, n_grid = 1201L) {
  g0 <- seq(center[1] - half, center[1] + half, length.out = n_grid)
  g1 <- seq(center[2] - half, center[2] + half, length.out = n_grid)
  W <- exp(log_post_grid(g0, g1) - max(log_post_grid(g0, g1)))
  W <- W / sum(W)
  qf <- function(g, wg) {
    cw <- cumsum(wg) - wg / 2
    suppressWarnings(stats::approx(cw, g, xout = probs)$y)
  }
  list(a = qf(g0, rowSums(W)), b = qf(g1, colSums(W)))
}

# Total absolute error of a fit's reported interval endpoints against a
# reference, over both coefficients.
.sd306_endpoint_err <- function(fit, ref, level = 0.95) {
  s <- summary(fit, level = level)
  sum(abs(c(s[1L, 3L] - ref$a[1L], s[1L, 4L] - ref$a[2L],
            s[2L, 3L] - ref$b[1L], s[2L, 4L] - ref$b[2L])))
}

# --------------------------------------------------------------------------- #
# (1) The selector reads the bands the fit already reports                     #
# --------------------------------------------------------------------------- #

test_that("a nested fit's own inner scores are what the selector bands", {
  skip_on_cran()
  d <- .sd306_rare_data()
  fit <- .sd306_rare_fit(d)

  bands <- tulpa:::.subspace_bands(fit)
  expect_equal(nrow(bands), 2L)
  expect_identical(bands$idx, as.integer(fit$inner_skew_idx))
  # Read off the fit, not re-fitted: a grid fit carries the per-index k-hat
  # `.inner_k_attach()` stored and NOT the raw importance curve, so a selector
  # that only knew how to re-fit the curve would band every index NA here.
  expect_identical(bands$gamma3, as.numeric(fit$inner_skew))
  expect_identical(bands$pareto_k, as.numeric(fit$inner_pareto_k))
  expect_identical(bands$rel_ess, as.numeric(fit$inner_pareto_k_rel_ess))
  expect_true(all(!is.na(bands$band)))
})

# --------------------------------------------------------------------------- #
# (2) Empty S, per backend                                                     #
# --------------------------------------------------------------------------- #

# `idx = integer(0)` pins the selection to nothing and skips the selector, which
# is the cleanest way to ask "does requesting the correction and selecting
# nothing change any number?". The plain comparison keeps `keep_grid_hessians`
# on, since requesting the correction turns it on.
test_that("an empty S leaves the grid backend bit-for-bit", {
  skip_on_cran()
  d <- .sd306_rare_data()
  plain <- .sd306_rare_fit(d)
  empty <- .sd306_rare_fit(d, list(subspace_debias = list(idx = integer(0))))

  expect_null(empty$draws)
  expect_identical(empty$subspace_debias$idx, integer(0))
  expect_identical(plain$log_marginal, empty$log_marginal)
  expect_identical(plain$weights, empty$weights)
  expect_identical(plain$modes, empty$modes)
  expect_identical(plain$grid_modes, empty$grid_modes)
  expect_identical(plain$grid_hessians, empty$grid_hessians)
  expect_identical(summary(plain), summary(empty))
  expect_identical(vcov(plain), vcov(empty))
})

test_that("an empty S leaves the single-block grid kernel bit-for-bit", {
  skip_on_cran()
  set.seed(5)
  S <- 24L; reps <- 4L; n <- S * reps
  idx <- rep(seq_len(S), each = reps)
  nb <- lapply(seq_len(S),
               function(s) setdiff(intersect(c(s - 1L, s + 1L), seq_len(S)), s))
  nn <- vapply(nb, length, integer(1))
  x <- rnorm(n)
  field <- as.numeric(scale(cumsum(rnorm(S, sd = 0.3)), scale = FALSE))
  y <- rbinom(n, 1L, stats::plogis(-3 + 0.8 * x + field[idx]))
  prior <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
                adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
                n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4))
  fit <- function(ctrl) {
    tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                         family = "binomial",
                         control = utils::modifyList(
                           list(keep_grid_hessians = TRUE, diagnose_k = FALSE,
                                progress = FALSE), ctrl))
  }
  plain <- fit(list())
  empty <- fit(list(subspace_debias = list(idx = integer(0))))
  expect_null(empty$draws)
  expect_identical(plain$log_marginal, empty$log_marginal)
  expect_identical(plain$modes, empty$modes)
  expect_identical(summary(plain), summary(empty))

  # And with the selector on, the correction reaches every integrated cell.
  set.seed(9)
  deb <- fit(list(subspace_debias = TRUE))
  if (length(deb$subspace_debias$idx)) {
    expect_true(is.matrix(deb$draws))
    expect_equal(nrow(deb$draws), tulpa:::.nl_diag("debias_n_draws"))
    expect_equal(length(deb$subspace_debias$accept), length(deb$weights))
    expect_true(all(is.finite(deb$subspace_debias$accept)))
  }
})

test_that("an empty S leaves the joint backends bit-for-bit", {
  skip_on_cran()
  coupled_occ_register()
  d <- coupled_occ_data(seed = 4L, n_cells = 40L, n_visits = 3L,
                        b_occ = 0.3, b_det = -0.6)
  fit <- function(ctrl) {
    tulpa_nested_laplace_joint(
      responses = coupled_occ_arms(d, beta_prec = 0.25),
      prior = coupled_occ_flat_prior(d),
      cell_coupling = "test_occupancy_mixture",
      control = utils::modifyList(
        list(max_iter = 100L, tol = 1e-10, diagnose_k = FALSE,
             progress = FALSE), ctrl))
  }
  plain <- fit(list())
  empty <- fit(list(subspace_debias = list(idx = integer(0))))
  expect_s3_class(plain, "tulpa_nested_laplace_joint_multi")
  expect_null(empty$draws)
  expect_identical(plain$log_marginal, empty$log_marginal)
  expect_identical(plain$modes, empty$modes)
  expect_identical(plain$grid_modes, empty$grid_modes)
  expect_identical(plain$grid_hessians, empty$grid_hessians)
  expect_identical(summary(plain), summary(empty))
})

test_that("an empty S leaves a single-block joint fit bit-for-bit", {
  skip_on_cran()
  set.seed(21)
  n_s <- 20L; N <- 200L
  rp <- integer(n_s + 1L); ci <- integer(0); nb <- integer(n_s)
  for (i in seq_len(n_s)) {
    nbrs <- c(if (i > 1L) i - 1L, if (i < n_s) i + 1L)
    ci <- c(ci, nbrs - 1L); nb[i] <- length(nbrs); rp[i + 1L] <- length(ci)
  }
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  phi_s <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.4))))
  x <- rnorm(N); Xocc <- cbind(1, x)
  occur <- rbinom(N, 1, stats::plogis(as.numeric(Xocc %*% c(-0.3, 0.5)) +
                                        phi_s[spatial_idx]))
  prior <- list(type = "icar", n_spatial_units = n_s, adj_row_ptr = rp,
                adj_col_idx = ci, n_neighbors = nb,
                sigma_grid = c(0.4, 0.9, 1.4))
  responses <- list(occ = list(
    y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
    spatial_idx = spatial_idx, re_idx = rep(0, N), n_re_groups = 0L,
    sigma_re = 1.0, family = "binomial", phi = 1.0))
  fit <- function(ctrl) {
    tulpa_nested_laplace_joint(
      responses, prior,
      control = utils::modifyList(
        list(max_iter = 100L, tol = 1e-8, diagnose_k = FALSE,
             progress = FALSE), ctrl))
  }
  plain <- fit(list())
  empty <- fit(list(subspace_debias = list(idx = integer(0))))
  expect_null(empty$draws)
  expect_identical(plain$log_marginal, empty$log_marginal)
  expect_identical(plain$modes, empty$modes)
  expect_identical(summary(plain), summary(empty))
})

# --------------------------------------------------------------------------- #
# (3) Against the exact conditional posterior                                  #
# --------------------------------------------------------------------------- #

test_that("the grid backend's corrected interval moves toward exact quadrature", {
  skip_on_cran()
  d <- .sd306_rare_data()
  plain <- .sd306_rare_fit(d)
  ref <- .sd306_exact_quantiles(.sd306_rare_log_post(d),
                                plain$modes[1L, 1:2], c(0.025, 0.975))

  # The intercept is the misfit coordinate and the selector finds it.
  set.seed(201)
  deb <- .sd306_rare_fit(d, list(subspace_debias = TRUE))
  expect_identical(deb$subspace_debias$idx, 1L)
  expect_identical(deb$subspace_debias$selected_by, "band")
  expect_true(is.matrix(deb$draws))

  e_plain <- .sd306_endpoint_err(plain, ref)
  e_deb <- .sd306_endpoint_err(deb, ref)
  # Measured over 5 seeds while writing this: 0.5229 -> 0.1883 (sd 0.0672),
  # a 64.0% cut. Pinned loosely so Monte Carlo noise on 4000 draws cannot make
  # it flap, but tightly enough that losing the correction fails it.
  expect_lt(e_deb, 0.5 * e_plain)

  # The corrected centre and scale move toward the exact ones too, not only the
  # endpoints: the Gaussian mode is at -2.526 against an exact mean of -2.635.
  expect_lt(abs(coef(deb)[1L] + 2.6346), abs(coef(plain)[1L] + 2.6346))
})

test_that("the joint backend's corrected interval moves toward exact quadrature", {
  skip_on_cran()
  coupled_occ_register()
  beta_prec <- 0.25
  d <- coupled_occ_data(seed = 4L, n_cells = 40L, n_visits = 3L,
                        b_occ = 0.3, b_det = -0.6)
  fit <- function(ctrl) {
    tulpa_nested_laplace_joint(
      responses = coupled_occ_arms(d, beta_prec = beta_prec),
      prior = coupled_occ_flat_prior(d),
      cell_coupling = "test_occupancy_mixture",
      control = utils::modifyList(
        list(max_iter = 100L, tol = 1e-10, diagnose_k = FALSE,
             progress = FALSE), ctrl))
  }
  plain <- fit(list())
  lpf <- coupled_occ_log_post(d, beta_prec)
  ref <- .sd306_exact_quantiles(
    function(g0, g1) outer(g0, g1, lpf), plain$modes[1L, 1:2],
    c(0.025, 0.975), half = 12)

  # Every coordinate of this fully coupled fit bands `good` on both inner
  # scores, so the band selector takes nothing -- see the note in NEWS about
  # gamma_3 reading 0.256 against an exact skewness of 1.198 here. Pinning the
  # set is what the `idx` setting exists for, and it is what tests the sampler
  # rather than the selector.
  set.seed(101)
  pinned <- fit(list(subspace_debias = list(idx = 1:2)))
  expect_identical(pinned$subspace_debias$idx, 1:2)
  expect_identical(pinned$subspace_debias$selected_by, "idx")
  expect_true(is.matrix(pinned$draws))

  e_plain <- .sd306_endpoint_err(plain, ref)
  e_pin <- .sd306_endpoint_err(pinned, ref)
  # Measured over 3 seeds: 0.6189 -> 0.163 / 0.341 / 0.313.
  expect_lt(e_pin, 0.7 * e_plain)
  # And the reported centre: plain -0.2400 against an exact mean of -0.1437.
  expect_lt(abs(coef(pinned)[1L] + 0.1437), abs(coef(plain)[1L] + 0.1437))
})

# --------------------------------------------------------------------------- #
# (4) Settings and declines                                                    #
# --------------------------------------------------------------------------- #

test_that("the coupling closure declines on a grid fit rather than silently finding nothing", {
  skip_on_cran()
  d <- .sd306_rare_data()
  set.seed(202)
  fit <- .sd306_rare_fit(d, list(subspace_debias = list(closure = TRUE)))
  # A grid fit retains no joint precision, so the precision-graph neighbours the
  # closure grows S by cannot be read. That is recorded, not passed over.
  expect_identical(fit$subspace_debias$declined, "closure_needs_joint_hessian")
  expect_identical(fit$subspace_debias$closure_added, integer(0))
})

test_that("control$subspace_debias is validated on the grid front doors", {
  skip_on_cran()
  d <- .sd306_rare_data()
  expect_error(.sd306_rare_fit(d, list(subspace_debias = list(nope = 1))),
               "Unknown `control\\$subspace_debias` setting")
  expect_error(.sd306_rare_fit(d, list(subspace_debias = list(band = "meh"))),
               "must be one of")
})
