# Restricted temporal regression: temporal_rtr() flags an RW1 / RW2 / AR1 walk,
# tulpa() builds the time-level projector P_perp orthogonal to the restrict_to
# design, and the nested-Laplace multi-block drivers read the walk through
# A = S P_perp in place of its time index.

# Confounded fixture: x carries the temporal trend f between time points plus
# within-time variation; y = 1 + b_c x + f + noise (variance 0.09). The
# conditional slope is b_c; the marginal one is b_c plus the slope of f[t_i] on
# x_i, the covariate's projection onto the field.
.rtr_sim <- function(seed, n_t = 50L, m = 4L, b_c = 1.0) {
  set.seed(seed)
  f <- as.numeric(stats::arima.sim(list(ar = 0.85), n_t, sd = 0.45))
  f <- f - mean(f)
  t_i <- rep(seq_len(n_t), each = m)
  u <- stats::rnorm(n_t, 0, 0.4)
  x <- 0.8 * f[t_i] + u[t_i] + stats::rnorm(n_t * m, 0, 0.5)
  y <- 1 + b_c * x + f[t_i] + stats::rnorm(n_t * m, 0, 0.3)
  list(d = data.frame(year = t_i, x = x, y = y), b_c = b_c,
       b_m = b_c + unname(stats::coef(stats::lm(f[t_i] ~ x))[2]))
}

test_that("temporal_rtr() flags a single walk and refuses what no projector carries", {
  for (tf in list(temporal_rw1("year"), temporal_rw2("year", cyclic = TRUE),
                  temporal_ar1("year"))) {
    r <- temporal_rtr(tf, ~ x)
    expect_true(isTRUE(r$rtr))
    expect_identical(r$type, tf$type)
    expect_s3_class(r, "tulpa_rtr")
    expect_s3_class(r, "tulpa_temporal")
  }
  expect_output(print(temporal_rtr(temporal_rw1("year"), ~ x)),
                "Orthogonal to: ~x")
  expect_error(temporal_rtr(list(), ~ x), "tulpa temporal specification")
  expect_error(temporal_rtr(temporal_rw1("year"), y ~ x), "one-sided formula")
  expect_error(temporal_rtr(temporal_gp("year"), ~ x), "got 'gp'")
  expect_error(temporal_rtr(temporal_multiscale("year", trend = "rw1"), ~ x),
               "got 'multiscale'")
  expect_error(temporal_rtr(temporal_tvc("year", terms = ~ x), ~ x),
               "got 'tvc'")
  expect_error(temporal_rtr(temporal_rw1("year", group_var = "site"), ~ x),
               "grouped \\(panel\\)")
})

test_that("the nested prior reads a restricted walk through S P_perp", {
  sm <- .rtr_sim(1L, n_t = 12L, m = 3L)
  spec <- tulpa:::validate_temporal(temporal_rtr(temporal_rw1("year"), ~ x), sm$d)
  X <- stats::model.matrix(~ x, sm$d)
  spec <- tulpa:::.attach_rtr_projection(spec, sm$d, X)
  P <- spec$rtr_projection
  xbar <- tapply(sm$d$x, sm$d$year, mean)
  expect_equal(dim(P), c(12L, 12L))
  expect_lt(max(abs(P %*% cbind(1, xbar))), 1e-10)

  blk <- tulpa:::.temporal_spec_to_nl_prior(spec)
  expect_null(blk$temporal_idx)
  expect_equal(blk$projector, P[sm$d$year, , drop = FALSE])
  expect_true(tulpa:::.nl_block_is_projected(blk))
  # An intercept in restrict_to annihilates the walk's constant: A 1 = 0.
  expect_lt(max(abs(rowSums(blk$projector))), 1e-10)

  cpp <- tulpa:::.nl_block_spec_for_cpp(blk, n_obs = nrow(sm$d),
                                        block_index = 1L)
  expect_null(cpp$temporal_idx)
  expect_length(cpp$projector$A_p[[1]], 13L)

  both <- blk; both$temporal_idx <- sm$d$year
  expect_error(tulpa:::.nl_block_projector(both, 1L, 1L, nrow(sm$d)),
               "either `projector` or `temporal_idx`")
})

test_that("a restriction spanning an RW2 walk's trend, or a sampler mode, is refused", {
  sm <- .rtr_sim(2L, n_t = 12L, m = 3L)
  expect_error(
    tulpa(y ~ x, data = sm$d, family = "gaussian", phi = 0.09,
          temporal = temporal_rtr(temporal_rw2("year"), ~ x + year)),
    "spans the linear time trend")
  expect_error(
    tulpa(y ~ x, data = sm$d, family = "gaussian", phi = 0.09, mode = "exact",
          temporal = temporal_rtr(temporal_rw1("year"), ~ x)),
    "fit by nested Laplace")
})

test_that("the restricted fit recovers the marginal slope, the unrestricted the conditional", {
  skip_on_cran()
  # Measured over 5 seeds at b_c = 1.0, mean b_m 1.51: restricted 0.018 from
  # b_m and 0.51 from b_c under each structure, unrestricted 0.45 / 0.06-0.08.
  err <- NULL
  for (st in c("rw1", "rw2", "ar1")) for (s in 101:103) {
    sm <- .rtr_sim(s)
    tf <- switch(st, rw1 = temporal_rw1("year"), rw2 = temporal_rw2("year"),
                 ar1 = temporal_ar1("year"))
    fr <- tulpa(y ~ x, data = sm$d, family = "gaussian", phi = 0.09,
                temporal = temporal_rtr(tf, ~ x),
                control = list(progress = FALSE))
    fu <- tulpa(y ~ x, data = sm$d, family = "gaussian", phi = 0.09,
                temporal = tf, control = list(progress = FALSE))
    expect_identical(fr$backend, "nested_laplace")
    err <- rbind(err, data.frame(
      st = st,
      r_m = abs(coef(fr)[["x"]] - sm$b_m), r_c = abs(coef(fr)[["x"]] - sm$b_c),
      u_m = abs(coef(fu)[["x"]] - sm$b_m), u_c = abs(coef(fu)[["x"]] - sm$b_c)))
  }
  for (st in unique(err$st)) {
    e <- colMeans(err[err$st == st, -1])
    expect_lt(e[["r_m"]], 0.05)
    expect_gt(e[["r_c"]], 0.3)
    expect_lt(e[["u_c"]], 0.15)
    expect_gt(e[["u_m"]], 0.3)
  }
})

test_that("temporal() reports the restricted field P z, orthogonal to the design", {
  skip_on_cran()
  sm <- .rtr_sim(7L, n_t = 30L)
  fr <- tulpa(y ~ x, data = sm$d, family = "gaussian", phi = 0.09,
              temporal = temporal_rtr(temporal_rw1("year"), ~ x),
              control = list(progress = FALSE))
  tp <- temporal(fr, summary = TRUE)
  xbar <- tapply(sm$d$x, sm$d$year, mean)
  expect_equal(nrow(tp), 30L)
  expect_lt(abs(sum(tp$mean)), 1e-8)
  expect_lt(abs(sum(xbar * tp$mean)), 1e-8)
  expect_true(all(tp$sd > 0))
})

test_that("a restricted walk stacks beside an areal field on the multi-block driver", {
  skip_on_cran()
  sm <- .rtr_sim(3L, n_t = 30L)
  K <- 8L
  W <- matrix(0, K, K)
  for (i in seq_len(K - 1L)) W[i, i + 1L] <- W[i + 1L, i] <- 1
  set.seed(3)
  d <- sm$d
  d$region <- sample(seq_len(K), nrow(d), replace = TRUE)
  fs <- tulpa(y ~ x + spatial(region), data = d, family = "gaussian",
              phi = 0.09, spatial = list(type = "icar", adjacency = W),
              temporal = temporal_rtr(temporal_ar1("year"), ~ x),
              control = list(progress = FALSE))
  expect_lt(abs(coef(fs)[["x"]] - sm$b_m), 0.1)
})

test_that("a temporal projector equal to the time incidence fits the gathered walk", {
  skip_on_cran()
  # With A = S (no restriction) the projected block is the gathered one, on
  # both multi-block drivers: same log marginal, fixed effects and covariance.
  set.seed(11)
  n_t <- 15L
  t_i <- sample.int(n_t, 120L, replace = TRUE)
  x <- stats::rnorm(120L)
  X <- cbind(`(Intercept)` = 1, x = x)
  y <- stats::rpois(120L, exp(0.3 + 0.4 * x + sin(t_i / 3) * 0.5))
  S <- matrix(0, 120L, n_t)
  S[cbind(seq_along(t_i), t_i)] <- 1
  ctl <- list(axis_refine = "none", prune = FALSE, diagnose_k = FALSE)
  resp <- list(y = list(y = y, n_trials = rep(1L, 120L), X = X,
                        family = "poisson"))
  for (type in c("rw1", "rw2", "ar1")) {
    blk <- list(type = type, n_times = n_t)
    single_g <- tulpa_nested_laplace(
      y = y, n_trials = rep(1L, 120L), X = X, family = "poisson",
      prior = list(c(blk, list(temporal_idx = t_i))), control = ctl)
    single_p <- tulpa_nested_laplace(
      y = y, n_trials = rep(1L, 120L), X = X, family = "poisson",
      prior = c(blk, list(projector = S)), control = ctl)
    expect_equal(single_p$log_marginal, single_g$log_marginal,
                 tolerance = 1e-6, info = type)
    expect_equal(coef(single_p), coef(single_g), tolerance = 1e-6, info = type)
    joint_g <- tulpa_nested_laplace_joint(
      responses = resp, prior = list(c(blk, list(temporal_idx = list(t_i)))),
      control = ctl)
    joint_p <- tulpa_nested_laplace_joint(
      responses = resp, prior = list(c(blk, list(projector = list(S)))),
      control = ctl)
    expect_equal(joint_p$log_marginal, joint_g$log_marginal,
                 tolerance = 1e-6, info = type)
    expect_equal(coef(joint_p), coef(joint_g), tolerance = 1e-6, info = type)
    expect_equal(vcov(joint_p), vcov(joint_g), tolerance = 1e-6, info = type)
  }
})


test_that("temporal() summarizes a restricted field past the retained-covariance limit against its dense posterior (#1125)", {
  skip_on_cran()
  # The limit at zero keeps no per-cell covariance on the fit, as a wide field
  # does; the summary reads each cell's covariance off its solved precision.
  old <- tulpa:::.NL_FIELD_COV_MAX
  assignInNamespace(".NL_FIELD_COV_MAX", 0, "tulpa")
  on.exit(assignInNamespace(".NL_FIELD_COV_MAX", old, "tulpa"), add = TRUE)
  sm <- .rtr_sim(7L, n_t = 30L)
  fr <- tulpa(y ~ x, data = sm$d, family = "gaussian", phi = 0.09,
              temporal = temporal_rtr(temporal_rw1("year"), ~ x),
              control = list(progress = FALSE))
  expect_null(fr$grid_field_cov)
  tp <- temporal(fr, summary = TRUE)
  # Dense reference: each cell's precision written out, inverted, conditioned
  # on the fit's sum-to-zero rows, mapped by P and mixed over the weights.
  src <- tulpa:::.field_cells(fr)
  on <- which(fr$weights > 0)
  sol <- tulpa:::.field_cell_solve(fr, src, on)
  n_x <- ncol(fr$modes)
  cols <- tulpa:::.nl_temporal_field_cols(fr)
  P <- fr$temporal$rtr_projection
  w <- fr$weights[on] / sum(fr$weights[on])
  A <- if (length(sol$constraint_cols)) {
    as.matrix(tulpa:::.constraint_rows_matrix(sol$constraint_cols, n_x))
  }
  mus <- vars <- NULL
  for (i in seq_along(on)) {
    Q <- Matrix::sparseMatrix(i = sol$Q_i[[i]] + 1L, p = sol$Q_p[[i]],
                              x = sol$Q_x[[i]], dims = c(n_x, n_x),
                              symmetric = TRUE)
    S <- as.matrix(solve(Q))
    if (!is.null(A)) {
      SA <- S %*% A
      S <- S - SA %*% solve(t(A) %*% SA, t(SA))
    }
    mus <- rbind(mus, as.numeric(P %*% fr$modes[on[i], cols]))
    vars <- rbind(vars, diag(P %*% S[cols, cols] %*% t(P)))
  }
  m <- colSums(w * mus)
  ref_sd <- sqrt(colSums(w * (vars + mus^2)) - m^2)
  expect_equal(tp$mean, m, tolerance = 1e-8)
  expect_equal(tp$sd, ref_sd, tolerance = 1e-8)
})
