# The exact joint density at fixed theta. The fixed-effect prior is the same
# engine default the Laplace solver and joint NUTS read.
.tgmrf_log_joint <- function(theta, y, n_trials, X, block, obs_idx,
                              re_idx, n_re_groups, sigma_re, family, phi) {
  p <- ncol(X)
  z_cols <- p + n_re_groups + seq_len(block$n_latent)
  Q <- block$Q(theta)
  Q <- Matrix::forceSymmetric(Q)
  logdet <- as.numeric(Matrix::determinant(Q, logarithm = TRUE)$modulus)
  Matrix::Cholesky(Q, LDL = FALSE)
  mu <- if (is.null(block$mu)) numeric(block$n_latent) else block$mu(theta)
  if (length(mu) != block$n_latent || any(!is.finite(mu))) {
    stop("`mu(theta)` must have n_latent finite entries.", call. = FALSE)
  }
  function(W) {
    beta <- W[seq_len(p), , drop = FALSE]
    z <- W[z_cols, , drop = FALSE]
    eta <- X %*% beta + z[obs_idx, , drop = FALSE]
    lp_re <- rep(0, ncol(W))
    if (n_re_groups) {
      re <- W[p + seq_len(n_re_groups), , drop = FALSE]
      active <- which(re_idx > 0L)
      eta[active, ] <- eta[active, , drop = FALSE] + re[re_idx[active], , drop = FALSE]
      lp_re <- colSums(matrix(stats::dnorm(re, 0, sigma_re, log = TRUE),
                              nrow = n_re_groups))
    }
    residual <- sweep(z, 1L, mu, `-`)
    lp_z <- logdet / 2 - block$n_latent / 2 * log(2 * pi) -
      colSums(residual * as.matrix(Q %*% residual)) / 2
    lp_beta <- colSums(matrix(stats::dnorm(beta, 0,
                              .tulpa_prior_sd("laplace"), log = TRUE), nrow = p))
    cpp_family_loglik_batch(as.numeric(y), as.integer(n_trials),
                            as.matrix(eta), family, .phi_to_kernel(family, phi)) +
      lp_z + lp_beta + lp_re
  }
}

# Calibrate at one deterministic outer mode, then freeze the IS budget for
# every proposal. Neither the retained likelihood estimate nor its particles
# are refreshed on an MH rejection (Andrieu and Roberts 2009).
.tgmrf_make_is <- function(lm, theta_mode, y, n_trials, X, block, obs_idx,
                            re_idx, n_re_groups, sigma_re, family, phi) {
  nt <- as.integer(n_trials %||% 1L)
  ri <- .validate_re_idx(re_idx %||% rep(0L, length(y)), n_re_groups,
                         length(y), "tulpa_tgmrf")
  density <- function(theta) .tgmrf_log_joint(theta, y, nt, X, block, obs_idx,
                                    ri, n_re_groups, sigma_re, family, phi)
  proposal_mode <- lm$conditional(theta_mode)
  joint_mode <- density(theta_mode)
  calibration <- .laplace_is_calibrate(proposal_mode, joint_mode)
  M <- calibration$n_particles
  log_sd <- calibration$log_estimate_sd
  n_evaluations <- calibration$likelihood_evaluations
  evaluate <- function(theta) {
    lp <- block$prior(theta)
    if (length(lp) != 1L || is.na(lp) || lp == Inf) {
      stop("`prior(theta)` must return one log density.", call. = FALSE)
    }
    if (lp == -Inf) return(list(log_prob = -Inf, draw = NULL))
    proposal <- lm$conditional(theta)
    n_evaluations <<- n_evaluations + M
    est <- .tgmrf_is_estimate(proposal, M, density(theta))
    est$log_likelihood <- est$log_prob
    est$log_prob <- est$log_likelihood + lp
    est$log_laplace <- proposal$log_marginal
    est
  }
  list(evaluate = evaluate, n_particles = M, log_estimate_sd = log_sd,
       target_log_sd = .TGMRF_IS$target_log_sd,
       likelihood_evaluations = function() n_evaluations)
}

# Put joint draws on the shared tulpa_fit layout, with fixed effects first.
.tgmrf_with_joint <- function(fit, theta_draws, latent_draws, X, n_re_groups) {
  p <- ncol(X)
  n_z <- ncol(latent_draws) - p - n_re_groups
  colnames(latent_draws) <- c(.tgmrf_beta_names(X),
                              if (n_re_groups) paste0("re_", seq_len(n_re_groups)),
                              paste0("z_", seq_len(n_z)))
  draws <- cbind(latent_draws, theta_draws)
  fit$draws <- draws
  fit$means <- colMeans(draws)
  fit$sds <- apply(draws, 2L, stats::sd)
  fit$param_names <- colnames(draws)
  fit$n_params <- ncol(draws)
  fit$n_samples <- nrow(draws)
  fit$n_fixed <- p
  fit$fixed_names <- colnames(latent_draws)[seq_len(p)]
  parts <- list(beta = seq_len(p), z = p + n_re_groups + seq_len(n_z))
  if (n_re_groups) parts$re <- p + seq_len(n_re_groups)
  for (nm in names(parts)) {
    dr <- latent_draws[, parts[[nm]], drop = FALSE]
    fit[[paste0("draws_", nm)]] <- dr
    fit[[paste0("means_", nm)]] <- colMeans(dr)
    fit[[paste0("sds_", nm)]] <- apply(dr, 2L, stats::sd)
  }
  fit$draws_theta <- theta_draws
  fit$means_theta <- colMeans(theta_draws)
  fit$sds_theta <- apply(theta_draws, 2L, stats::sd)
  fit
}

# PSIS corrects the deterministic Laplace chain on its extended particle
# space. Resampling whole (theta, selected particle) rows preserves dependence.
.tgmrf_correct_nuts <- function(fit, theta, lm, pl, block, X,
                                 y, n_trials, obs_idx, re_idx, n_re_groups,
                                 sigma_re, family, phi, requested) {
  fit$correction <- "none"
  fit$correction_applied <- FALSE
  fit$correction_declined <- "not_requested"
  fit$target <- "Laplace-approximate theta marginal with Gaussian conditional beta"
  fit$pareto_k <- NA_real_
  fit$pareto_k_scope <- "Laplace marginal to joint posterior correction"
  fit$draws_kind <- "chain"
  if (isTRUE(requested)) {
    om <- .tgmrf_outer_mode(pl, lm, block)
    estimator <- .tgmrf_make_is(lm, om$mode, y, n_trials, X, block, obs_idx,
                                re_idx, n_re_groups, sigma_re, family, phi)
    estimates <- lapply(seq_len(nrow(theta)), function(i)
      estimator$evaluate(theta[i, ]))
    ratios <- vapply(estimates, function(e) e$log_prob - e$log_laplace, numeric(1))
    ps <- tulpa_psis(ratios)
    fit$is_particles <- estimator$n_particles
    fit$log_estimate_sd <- estimator$log_estimate_sd
    fit$is_target_log_sd <- estimator$target_log_sd
    fit$is_likelihood_evaluations <- estimator$likelihood_evaluations()
    fit$pareto_k <- ps$pareto_k
    fit$pareto_k_is_ess <- ps$is_ess
    fit$correction_log_ratios <- ratios
    fit$correction_log_weights <- ps$log_weights
    fit$correction_k_threshold <- .ps_khat_threshold(nrow(theta))
    fit$correction_declined <- if (nrow(theta) < .PSIS_MIN_EVAL) "draws_too_few"
      else if (!is.finite(ps$pareto_k)) "nonfinite_pareto_k"
      else if (ps$pareto_k >= fit$correction_k_threshold) "pareto_k_above_threshold"
      else NA_character_
    if (is.na(fit$correction_declined)) {
      index <- sample.int(nrow(theta), nrow(theta), replace = TRUE,
                           prob = exp(ps$log_weights))
      joint <- do.call(rbind, lapply(estimates, `[[`, "draw"))
      fit <- .tgmrf_with_joint(fit, theta[index, , drop = FALSE],
                                joint[index, , drop = FALSE], X, n_re_groups)
      fit$correction <- "psis"
      fit$correction_applied <- TRUE
      fit$target <- "PSIS importance approximation to joint posterior p(beta, re, z, theta | y)"
      fit$draws_kind <- "iid"
      fit$source_theta_draws <- theta
      fit$source_index <- index
      fit$source_sampler_diagnostics <- fit[c("tree_depth", "divergent",
                                              "n_divergent", "energy", "mean_accept")]
      fit[c("tree_depth", "divergent", "n_divergent", "energy")] <- NULL
    }
  }
  if (!fit$correction_applied) fit <- .tgmrf_with_fixed(fit, theta, lm, X)
  fit$selection_reason <- if (fit$correction_applied) {
    sprintf("PSIS joint importance correction applied (k = %.3f); Tier 2 finite importance approximation",
              fit$pareto_k)
  } else {
    paste0("Laplace-approximate theta marginal; posterior correction not applied: ",
             fit$correction_declined)
  }
  fit$inner_solves <- lm$inner_solves()
  fit
}
