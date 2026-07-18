# Guards that the single-block outer Pareto-k importance target matches the
# integrator's own weighting (#221). The default single positive-scale grid is
# geometric (uniform in u = log theta); the integrator weights it with plain
# softmax(log_marginal) and NO volume element, so exp(log_marginal) IS the
# u-space posterior density. .nested_grid_pareto_k fits its Gaussian proposal in
# that same u coordinate, so the importance target must be `log_marginal` with
# NO change-of-variables Jacobian -- matching the SPDE Pareto-k path. A spurious
# + sum(u) (the pre-#221 target) tilts the certified target away from the
# posterior the fit reports and can flip the usable/unusable verdict.

test_that(".nested_grid_pareto_k reports a matched Gaussian proposal as reliable", {
  # log_marginal exactly Gaussian in u = log(theta): the integrator's grid
  # weights recover N(mu, s^2) and the proposal fit to that grid matches it, so
  # with the correct (no-Jacobian) target k-hat sits below the 0.7 usable line.
  mu <- 0.2; s <- 0.5
  u_grid <- matrix(seq(mu - 3 * s, mu + 3 * s, length.out = 15L), ncol = 1L)
  lm_grid <- -0.5 * ((u_grid[, 1] - mu) / s)^2
  w <- exp(lm_grid - max(lm_grid)); w <- w / sum(w)
  refit <- function(theta_mat) -0.5 * ((log(theta_mat[, 1]) - mu) / s)^2

  set.seed(101)
  kd <- tulpa:::.nested_grid_pareto_k(u_grid, w, refit, n_samples = 800L)
  expect_true(is.finite(kd$pareto_k))
  expect_lt(kd$pareto_k, 0.7)
})

test_that(".nested_grid_pareto_k target carries no change-of-variables Jacobian", {
  # Exact check: rebuild the proposal + radius cap the way .nested_grid_pareto_k
  # does, then drive .nested_is_pareto_k with an explicit no-Jacobian target and
  # the SAME seed. The grid path must reproduce the no-Jacobian result bit for
  # bit, and the with-Jacobian target (the pre-#221 lm + sum(u)) must give a
  # different k-hat -- so the fix is neither a no-op nor an accidental match.
  mu <- 0.2; s <- 0.7
  u_grid <- matrix(seq(mu - 3 * s, mu + 3 * s, length.out = 21L), ncol = 1L)
  lm_grid <- -0.5 * ((u_grid[, 1] - mu) / s)^2
  w <- exp(lm_grid - max(lm_grid)); w <- w / sum(w)
  refit <- function(theta_mat) -0.5 * ((log(theta_mat[, 1]) - mu) / s)^2

  u_hat <- as.numeric(crossprod(w, u_grid))
  cen   <- sweep(u_grid, 2L, u_hat)
  Su    <- crossprod(cen * w, cen); Su <- (Su + t(Su)) / 2
  L     <- t(chol(Su))
  rcap  <- tulpa:::.nested_grid_radius_cap(u_grid, u_hat, L)

  lt_nojac   <- function(U) refit(exp(U))
  lt_withjac <- function(U) refit(exp(U)) + rowSums(U)

  set.seed(202)
  got <- tulpa:::.nested_grid_pareto_k(u_grid, w, refit, n_samples = 600L)$pareto_k
  set.seed(202)
  ref_nojac <- tulpa:::.nested_is_pareto_k(u_hat, L, lt_nojac, n_samples = 600L,
                                           radius_cap = rcap)$pareto_k
  set.seed(202)
  ref_withjac <- tulpa:::.nested_is_pareto_k(u_hat, L, lt_withjac, n_samples = 600L,
                                             radius_cap = rcap)$pareto_k

  expect_equal(got, ref_nojac, tolerance = 1e-9)
  expect_false(isTRUE(all.equal(ref_nojac, ref_withjac, tolerance = 1e-6)))
})

test_that("joint outer Pareto-k target carries no log-axis Jacobian (#221)", {
  # Same fix on the joint path (.joint_pareto_inv). Ground truth is
  # target-agnostic: PSIS the grid-node importance ratios log(w_k) - log q(u_k)
  # for a geometric log axis (w_k proportional to p(u_k), so this is the true
  # importance-weight tail). The corrected joint diagnostic must track it; the
  # pre-#221 target (log_marginal + sum(u)) overstates the tail materially.
  sg <- exp(seq(-3, 3, length.out = 61)); u <- log(sg)
  mu <- 0.4; s <- 1.3
  lm_grid <- -0.5 * ((u - mu) / s)^2
  w <- exp(lm_grid - max(lm_grid)); w <- w / sum(w)
  res <- list(theta_grid = matrix(sg, ncol = 1, dimnames = list(NULL, "sigma")),
              weights = w, prior = list(type = "icar"))
  refit <- function(tm) -0.5 * ((log(tm[, "sigma"]) - mu) / s)^2

  prep <- tulpa:::.joint_pareto_prepare(res, refit, 4000L, NULL)
  logq <- stats::dnorm(u, prep$u_hat[1], sqrt(prep$Su[1, 1]), log = TRUE)
  gt   <- tulpa:::tulpa_psis(log(w) - logq)$pareto_k          # grid-node ground truth

  set.seed(11)
  kd <- tulpa:::.joint_pareto_k(res, refit, n_samples = 4000L, proposal = NULL)

  # The corrected joint target reproduces the ground truth; the with-Jacobian
  # target (the pre-#221 code) sits materially further from it.
  L <- t(chol(prep$Su))
  lt_withjac <- function(U) refit(matrix(exp(U[, 1]), ncol = 1,
                                         dimnames = list(NULL, "sigma"))) + rowSums(U)
  set.seed(11)
  k_withjac <- tulpa:::.nested_is_pareto_k(prep$u_hat, L, lt_withjac, 4000L)$pareto_k

  expect_lt(abs(kd$pareto_k - gt), 0.35)
  expect_gt(abs(k_withjac - gt), abs(kd$pareto_k - gt))
})
