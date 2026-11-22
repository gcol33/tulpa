# Guards that the outer Pareto-k importance target matches the integrator's own
# weighting (#221). A geometric positive-scale grid is uniform in u = log theta;
# the integrator weights it with plain softmax(log_marginal) and NO volume
# element, so exp(log_marginal) IS the u-space posterior density. The Gaussian
# proposal is fit in that same u coordinate, so the importance target must be
# `log_marginal` with NO change-of-variables Jacobian -- matching the SPDE
# Pareto-k path. A spurious + sum(u) tilts the certified target away from the
# posterior the fit reports and can flip the usable/unusable verdict.

skip_on_cran()

test_that("joint outer Pareto-k target carries no log-axis Jacobian (#221)", {
  skip_on_cran()
  # The joint path (.joint_pareto_inv). Ground truth is
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
