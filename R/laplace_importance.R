# Full-dimensional t importance sampling with a precision matrix or a live
# inverse-Cholesky operator. The estimator includes every proposed particle.
.tgmrf_is_estimate <- function(proposal, n_particles, log_joint,
                                df = .TGMRF_IS$df) {
  D <- length(proposal$mean)
  if (is.null(proposal$transform)) {
    fac <- Matrix::Cholesky(methods::as(proposal$precision, "symmetricMatrix"),
                            perm = FALSE, LDL = FALSE)
  }
  Z <- matrix(stats::rnorm(D * n_particles), D, n_particles)
  radii <- stats::rchisq(n_particles, df) / df
  Z <- sweep(Z, 2L, sqrt(radii), `/`)
  if (is.null(proposal$transform)) {
    delta <- as.matrix(Matrix::solve(fac, Z, system = "Lt"))
    logdet <- as.numeric(Matrix::determinant(proposal$precision,
                                            logarithm = TRUE)$modulus)
  } else {
    delta <- proposal$transform(Z)
    logdet <- proposal$log_determinant
  }
  particles <- sweep(delta, 1L, proposal$mean, `+`)
  log_q <- lgamma((df + D) / 2) - lgamma(df / 2) + logdet / 2 -
    D / 2 * log(df * pi) - (df + D) / 2 * log1p(colSums(Z^2) / df)
  log_w <- log_joint(particles) - log_q
  if (anyNA(log_w) || any(log_w == Inf)) {
    stop("Non-finite importance ratio in the likelihood estimator.", call. = FALSE)
  }
  lse <- .tulpa_logsumexp(log_w)
  if (!is.finite(lse)) {
    out <- list(log_prob = -Inf, draw = rep(NA_real_, D), particle_ess = 0)
  } else {
    selected <- sample.int(n_particles, 1L, prob = exp(log_w - lse))
    out <- list(log_prob = lse - log(n_particles), draw = particles[, selected],
                particle_ess = exp(2 * lse - .tulpa_logsumexp(2 * log_w)))
  }
  if (isTRUE(proposal$keep_weights)) out$log_weights <- log_w
  out
}

# Independent pilot estimates set a fixed final particle count.
.laplace_is_calibrate <- function(proposal, log_joint,
                                   initial = .TGMRF_IS$initial_particles) {
  M <- initial
  n_evaluations <- 0
  repeat {
    log_estimates <- replicate(.TGMRF_IS$pilot_batches,
      .tgmrf_is_estimate(proposal, M, log_joint)$log_prob)
    n_evaluations <- n_evaluations + .TGMRF_IS$pilot_batches * M
    log_sd <- stats::sd(log_estimates)
    if (is.finite(log_sd) && log_sd <= .TGMRF_IS$target_log_sd) break
    if (M >= .TGMRF_IS$max_particles) {
      stop("IS calibration could not meet the log-estimate SD target.", call. = FALSE)
    }
    M <- as.integer(min(.TGMRF_IS$max_particles,
      max(2 * M, if (is.finite(log_sd))
        ceiling(M * (log_sd / .TGMRF_IS$target_log_sd)^2) else 2 * M)))
  }
  list(n_particles = M, log_estimate_sd = log_sd,
       likelihood_evaluations = n_evaluations)
}

# The final estimate is independent of the calibration pilots. Raw IS weights
# define the marginal correction; PSIS supplies its tail reliability gate.
.subspace_importance <- function(proposal, log_joint) {
  proposal$keep_weights <- TRUE
  calibration <- .laplace_is_calibrate(proposal, log_joint,
    initial = max(.TGMRF_IS$initial_particles * .TGMRF_IS$pilot_batches,
                    .PSIS_MIN_EVAL))
  M <- calibration$n_particles
  estimate <- .tgmrf_is_estimate(proposal, M, log_joint)
  ps <- tulpa_psis(estimate$log_weights)
  c(estimate[c("log_prob", "particle_ess")], calibration,
    list(pareto_k = ps$pareto_k, k_threshold = .ps_khat_threshold(M),
         log_weights = estimate$log_weights))
}
