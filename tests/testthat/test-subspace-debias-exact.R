test_that("the exact subspace method validates its composition and indices", {
  expect_identical(tulpa:::.subspace_debias_config(TRUE)$method, "fast")
  expect_identical(tulpa:::.subspace_debias_config(list(method = "exact"))$method, "exact")
  expect_error(tulpa:::.subspace_debias_config(list(method = "other")), "method")
  expect_error(tulpa:::.subspace_debias_config(list(method = list("exact"))), "method")
  expect_error(tulpa:::.subspace_debias_config(list(method = "exact"), TRUE), "choose one")
  expect_error(tulpa:::.subspace_debias_config(list(n_iter = 1.5)), "whole number")
  expect_error(tulpa:::.subspace_debias_config(list(idx = 1.5)), "whole numbers")
  expect_error(tulpa:::.subspace_debias_request(list(method = "other", idx = 1L)), "method")
  expect_error(tulpa:::.subspace_debias_request(list(idx = c(1, 1))), "distinct")
  expect_error(tulpa:::.subspace_debias_request(list(idx = 1.5)), "whole numbers")
  expect_error(tulpa:::.subspace_debias_request(list(idx = 1L,
    n_iter = .Machine$integer.max, warmup = 0L)), "native integer range")
  expect_null(tulpa:::.subspace_debias_request(list(idx = integer(0))))
  expect_identical(tulpa:::.subspace_debias_request(
    list(method = "exact", idx = integer(0)))$method, "exact")
})

test_that("outer cell reweighting uses raw marginal estimates and the PSIS gate", {
  e <- list(c(log(2), 40, 256, .1, 4096, .1, .5),
            c(log(.5), 60, 256, .1, 4096, .2, .5))
  set.seed(117600)
  ratios <- lapply(c(2, .5), function(value) {
    lr <- rnorm(256, 0, .2)
    lr - tulpa:::.tulpa_logsumexp(lr) + log(256) + log(value)
  })
  got <- tulpa:::.subspace_outer_correction(c(.25, .75), e, ratios)
  expect_true(got$applied)
  expect_equal(got$weights, c(.5, .375) / .875)
  expect_equal(got$log_normalizer, log(.875))
  ratios[[2]] <- -2 * log((seq_len(256) - .5) / 256)
  e[[2]][1] <- tulpa:::.tulpa_logsumexp(ratios[[2]]) - log(256)
  declined <- tulpa:::.subspace_outer_correction(c(.25, .75), e, ratios)
  expect_false(declined$applied)
  expect_identical(declined$declined, "pareto_k_above_threshold")
  expect_identical(declined$laplace_weights, c(.25, .75))
})

test_that("matrix and operator proposals share the same importance estimator", {
  skip_on_cran()
  Q <- matrix(c(2, .7, .7, 1.1), 2)
  log_joint <- function(W) -colSums(W * (Q %*% W)) / 2
  matrix_proposal <- list(mean = c(0, 0), precision = Matrix::Matrix(Q, sparse = TRUE))
  operator_proposal <- list(mean = c(0, 0), log_determinant = as.numeric(determinant(Q)$modulus),
    transform = function(Z) backsolve(chol(Q), Z))
  set.seed(117601)
  a <- tulpa:::.tgmrf_is_estimate(matrix_proposal, 256L, log_joint)
  sa <- .Random.seed
  set.seed(117601)
  b <- tulpa:::.tgmrf_is_estimate(operator_proposal, 256L, log_joint)
  expect_equal(a, b, tolerance = 1e-12)
  expect_identical(sa, .Random.seed)
})

test_that("cell correction uses the retained Laplace denominator and folded prior", {
  res <- list(weights = c(.4, .6), log_marginal = log(c(2, 3)) + c(5, 7),
    log_hyperprior = c(6, 9), log_hyperprior_in_kernel = c(1, 2))
  out <- list(log_marginal = log(c(4, 6)), log_hyperprior = c(11, 22),
    log_hyperprior_in_kernel = c(11, 22),
    debias_importance_per_grid = rep(list(c(log(.5), 1, 2, 3, 4, 5, 6)), 2),
    debias_importance_log_weights_per_grid = rep(list(c(-1, 1)), 2))
  rebased <- tulpa:::.subspace_rebase_importance(res, out)
  expect_equal(rebased$debias_reference_shift, rep(log(2), 2))
  expect_equal(vapply(rebased$debias_importance_per_grid, `[`, numeric(1), 1L), c(0, 0))
  expect_equal(rebased$debias_importance_log_weights_per_grid[[1]], c(-1, 1) + log(2))
  res$weights[2] <- 0
  res$log_marginal[2] <- -Inf
  excluded <- tulpa:::.subspace_rebase_importance(res, out)
  expect_true(is.na(excluded$debias_reference_shift[2]))
  expect_true(is.na(excluded$debias_importance_per_grid[[2]][1]))
})

test_that("corrected diagnostics retain the original proposal scores separately", {
  fit <- list(pareto_k = .9, pareto_k_tail_points = 12L,
    pareto_k_by_arm = c(a = .9), subspace_debias = list())
  outer <- list(pareto_k = .2, is_ess = 300, scope = "corrected cells")
  fit <- tulpa:::.subspace_attach_diagnostics(fit, outer)
  expect_identical(fit$pareto_k, .2)
  expect_null(fit$pareto_k_tail_points)
  expect_null(fit$pareto_k_by_arm)
  expect_identical(fit$subspace_debias$laplace_diagnostics$pareto_k_tail_points, 12L)
})

skip_on_cran()

test_that("the joint Gaussian proposal satisfies detailed balance", {
  skip_on_cran()
  load_cpp_fixture("subspace_density_work")
  expect_true(cpp_subspace_checkpoint_roundtrip())
  set.seed(117604)
  tuned <- cpp_subspace_conditional_scale()
  expect_gt(tuned, .5)
  expect_lt(tuned, 10)
  Q <- matrix(c(2, .7, .7, 1.1), 2)
  V <- solve(Q)
  slope <- V[2, 1] / V[1, 1]
  conditional_sd <- sqrt(V[2, 2] - V[2, 1]^2 / V[1, 1])
  log_target <- function(x) -sum(x * (Q %*% x)) / 2
  log_proposal <- function(to, from) {
    dnorm(to[1], from[1], .8, log = TRUE) +
      dnorm(to[2], slope * to[1], conditional_sd, log = TRUE)
  }
  energy <- function(x) ((x[2] - slope * x[1]) / conditional_sd)^2
  states <- rbind(c(-1, .3), c(.8, -1.2), c(.2, 1.4))
  for (i in seq_len(nrow(states))) for (j in seq_len(nrow(states))) {
    x <- states[i, ]; y <- states[j, ]
    alpha <- cpp_subspace_log_accept(log_target(y), log_target(x), energy(y), energy(x))
    independent <- log_target(y) + log_proposal(x, y) -
      log_target(x) - log_proposal(y, x)
    expect_equal(alpha, independent, tolerance = 1e-12)
    left <- log_target(x) + log_proposal(y, x) + min(0, alpha)
    right <- log_target(y) + log_proposal(x, y) + min(0, -alpha)
    expect_equal(left, right, tolerance = 1e-12)
  }
})

test_that("the exact conditional kernel samples nuisance coordinates and retains rejections", {
  skip_on_cran()
  set.seed(117602)
  X <- matrix(1, 6, 1, dimnames = list(NULL, "intercept"))
  y <- c(-1, .2, 1, .4, -.7, .3)
  re <- list(list(idx = rep(1:3, each = 2), n_groups = 3L, sigma = 1.1))
  fit <- tulpa_laplace(y, rep(1L, 6), X, re_list = re, family = "gaussian",
    return_joint_hessian = TRUE, debias = list(method = "exact", idx = 1L,
      n_iter = 8000L, warmup = 1000L))
  expect_identical(fit$debias_idx, 1:4)
  expect_identical(fit$inference_tier, 1L)
  expect_match(fit$target, "conditional on supplied block hyperparameters", fixed = TRUE)
  expect_equal(dim(fit$joint_latent_draws), c(8000L, 4L))
  expect_equal(unname(coef(fit)), unname(colMeans(fit$draws)))
  expect_equal(vcov(fit), cov(fit$draws), ignore_attr = TRUE)
  expect_lt(max(abs(colMeans(fit$joint_latent_draws) - fit$mode)), .09)
  expect_equal(cov(fit$joint_latent_draws), solve(as.matrix(fit$H_joint)),
               tolerance = .08, ignore_attr = TRUE)
  same <- rowSums(abs(fit$joint_latent_draws[-1, ] -
                     fit$joint_latent_draws[-nrow(fit$joint_latent_draws), ])) == 0
  expect_gt(sum(same), 100L)
  expect_equal(mean(!same), fit$debias_accept, tolerance = .001)
  empty <- tulpa_laplace(y, rep(1L, 6), X, re_list = re, family = "gaussian",
    debias = list(method = "exact", idx = integer(0), n_iter = 1000L, warmup = 0L))
  expect_equal(empty$debias_accept, 1, tolerance = 1e-8)
  expect_equal(ncol(empty$joint_latent_draws), 4L)
})

test_that("nested exact correction reports its finite outer approximation", {
  skip_on_cran()
  args <- list(y = c(0, .2, -.3, .1), n_trials = rep(1L, 4), X = matrix(1, 4, 1),
    re_terms = list(idx = rep(1:2, each = 2), n_groups = 2L), family = "gaussian")
  fit <- do.call(tulpa_re_cov_nested, c(args, list(control = list(
    seed = 117603L, diagnose_k = FALSE, n_draws = 300L,
    subspace_debias = list(method = "exact", idx = 1L, n_iter = 300L, warmup = 100L)))))
  expect_identical(fit$inference_tier, 2L)
  expect_true(fit$subspace_debias$outer$applied)
  expect_identical(fit$weights_source, "subspace_is")
  expect_match(fit$target, "finite-grid IS")
  expect_equal(fit$joint_latent_draws[, 1], unname(fit$draws[, 1]))
  expect_equal(fit$joint_latent_draws[, -1], fit$re_debias_draws, ignore_attr = TRUE)
  expect_equal(fit$draws_theta, fit$theta_grid[fit$draw_nodes, , drop = FALSE])
  expect_equal(sum(fit$weights), 1)
  expect_identical(fit$pareto_k, fit$subspace_debias$outer$pareto_k)
  expect_match(fit$pareto_k_scope, "stratified latent importance")
  expect_identical(fit$draws_kind, "mcmc_mixture")
  expect_equal(unname(coef(fit)), unname(colMeans(fit$draws)))
  expect_equal(vcov(fit), cov(fit$draws), ignore_attr = TRUE)
  diag <- diagnostics(fit)
  expect_true(is.data.frame(diag))
  expect_true(all(c("cell", "weight", "rhat", "ess_bulk") %in% names(diag)))
  expect_true(is.data.frame(attr(diag, "subspace_outer_correction")))
  expect_identical(attr(diag, "pareto_k"), fit$pareto_k)
  expect_match(tulpa:::.tulpa_non_chain_msg(fit), "source-chain mixing")
  ds <- diagnostic_summary(fit, quiet = TRUE)
  expect_identical(ds$pareto_k, fit$pareto_k)
  expect_null(ds$inner_skew_max)
  expect_true(all(diagnostics(fit, pars = fit$fixed_names)$parameter %in% fit$fixed_names))
})

test_that("a sparse exact Gaussian kernel and its IS operator use the live factor", {
  skip_on_cran()
  set.seed(117605)
  G <- 210L
  group <- rep(seq_len(G), each = 2L)
  y <- sin(seq_along(group) / 10)
  request <- tulpa:::.subspace_debias_request(list(method = "exact", idx = integer(0),
    correct_outer = TRUE, n_iter = 2000L, warmup = 0L))
  fit <- tulpa_laplace(y, rep(1L, length(y)), matrix(1, length(y), 1),
    re_list = list(list(idx = group, n_groups = G, sigma = .7)), family = "gaussian",
    return_joint_hessian = TRUE, debias = request)
  expect_equal(dim(fit$joint_latent_draws), c(2000L, 211L))
  expect_equal(fit$debias_accept, 1, tolerance = 1e-8)
  V <- as.matrix(Matrix::solve(fit$H_joint))[1:4, 1:4]
  z <- fit$joint_latent_draws[, 1:4]
  expect_true(max(abs(colMeans(z) - fit$mode[1:4]) / sqrt(diag(V) / nrow(z))) < 5)
  covariance_mcse <- sqrt((V^2 + outer(diag(V), diag(V))) / (nrow(z) - 1L))
  expect_lt(max(abs(cov(z) - V) / covariance_mcse), 5)
  ratios <- exp(fit$debias_importance_log_weights)
  expect_lt(abs(mean(ratios) - 1), 5 * sd(ratios) / sqrt(length(ratios)))
})

test_that("joint response kernels retain complete corrected source rows", {
  arms <- list(a = list(y = c(0, .2, -.3, .1), X = matrix(1, 4, 1),
      family = "gaussian", beta_prior_prec = .16),
    b = list(y = c(-.1, .4, .2, 0), X = matrix(1, 4, 1),
      family = "gaussian", beta_prior_prec = .16))
  prior <- list(list(type = "iid", obs_idx = rep(1:2, each = 2),
    n_units = 2L, sigma_grid = c(.5, 1)))
  for (sparse in c(FALSE, TRUE)) {
    set.seed(117606)
    fit <- tulpa_nested_laplace_joint(arms, prior, control = list(
      force_sparse = sparse, diagnose_k = FALSE, diagnose_skew = FALSE,
      auto_recenter = FALSE, axis_refine = "none", progress = FALSE,
      subspace_debias = list(method = "exact", idx = 1L,
        n_iter = 400L, warmup = 100L, n_draws = 100L)))
    expect_true(fit$subspace_debias$outer$applied)
    expect_identical(fit$inference_tier, 2L)
    expect_equal(ncol(fit$joint_latent_draws), 4L)
    expect_equal(fit$draws, fit$joint_latent_draws[, 1:2], ignore_attr = TRUE)
    set.seed(117607)
    draws <- tulpa_posterior_draws(fit, n = 20L)
    cells <- attr(draws, "cells")
    rows <- attr(draws, "source_rows")
    expected <- do.call(rbind, lapply(seq_len(nrow(draws)), function(i)
      fit$subspace_debias$cell_modes[cells[i], ] +
        fit$subspace_debias$cell_draws[[cells[i]]][rows[i], ]))
    expect_equal(draws, expected, ignore_attr = TRUE)
    expect_equal(attr(draws, "theta"), fit$theta_grid[cells, , drop = FALSE])
  }
})
