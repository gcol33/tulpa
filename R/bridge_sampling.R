#' Bridge sampling for marginal likelihood
#'
#' @description
#' Estimate the log marginal likelihood \eqn{\log Z = \log p(y)} from
#' posterior draws via the Meng-Wong / Gronau bridge-sampling identity.
#' Useful for Bayes factors, model comparison, and as a sanity check on
#' Laplace-approximated marginal likelihoods.
#'
#' The classical iterative scheme (Meng & Wong 1996; Gronau et al. 2017)
#' is used with a multivariate-normal proposal fit to half of the
#' posterior draws -- the other half is used for the bridge ratio so the
#' proposal is independent of the samples that score it. All
#' computations are done in log-space via `logsumexp` for numerical
#' stability.
#'
#' @section Tier:
#' Bridge sampling is a *post-hoc* marginal-likelihood estimator, not
#' a sampling backend, so it does not appear in the
#' [INFERENCE_TIERS] registry. It operates on draws from any Tier-1
#' (Exact) backend.
#'
#' @param draws Numeric matrix of posterior draws, one parameter per
#'   column. Typically `fit$draws` from a tulpa Tier-1 fit.
#' @param log_posterior Function `function(theta) -> numeric` returning
#'   the *unnormalized* log posterior density `log p(theta, y)` at a
#'   single parameter vector. Must accept a numeric vector of length
#'   `ncol(draws)` and return a finite scalar.
#' @param n_proposal Number of proposal draws. Default `nrow(draws)`.
#' @param max_iter Maximum bridge iterations (default 1000).
#' @param tol Convergence tolerance on `|log r_{t+1} - log r_t|`
#'   (default 1e-10).
#' @param split Logical: split the posterior draws in half so the
#'   proposal is fit on one half and scored on the other (default
#'   `TRUE`, recommended). Set `FALSE` only for diagnostic comparison.
#' @param verbose Print iteration history (default `FALSE`).
#'
#' @return A list with:
#'   * `log_marginal`: the bridge-sampling estimate of \eqn{\log Z}.
#'   * `n_iter`: bridge iterations to convergence.
#'   * `converged`: logical.
#'   * `re_sd`: relative MSE estimate (Fruehwirth-Schnatter 2004); a
#'     rough quality check, smaller is better.
#'   * `proposal`: the fitted proposal `list(mean, cov)`.
#'
#' @references
#' Meng, X.-L., & Wong, W. H. (1996). Simulating ratios of normalizing
#' constants via a simple identity: a theoretical exploration.
#' *Statistica Sinica*, 6, 831-860.
#'
#' Gronau, Q. F., Sarafoglou, A., Matzke, D., Ly, A., Boehm, U., Marsman,
#' M., ... & Steingroever, H. (2017). A tutorial on bridge sampling.
#' *Journal of Mathematical Psychology*, 81, 80-97.
#'
#' @examples
#' # Toy: marginal likelihood of N(theta | 0, 1) under Y = theta + eps,
#' # eps ~ N(0, 1), prior theta ~ N(0, 10). Closed-form available.
#' y <- 1.5
#' log_post <- function(theta) {
#'   dnorm(y, theta, 1, log = TRUE) + dnorm(theta, 0, 10, log = TRUE)
#' }
#' draws <- matrix(rnorm(2000, mean = y * 100 / 101, sd = sqrt(100 / 101)),
#'                 ncol = 1)
#' bs <- bridge_sampling(draws, log_post)
#' bs$log_marginal  # should match dnorm(y, 0, sqrt(101), log = TRUE)
#'
#' @export
bridge_sampling <- function(draws,
                            log_posterior,
                            n_proposal = nrow(draws),
                            max_iter = 1000L,
                            tol = 1e-10,
                            split = TRUE,
                            verbose = FALSE) {

  if (!is.matrix(draws)) draws <- as.matrix(draws)
  if (!is.function(log_posterior)) {
    stop("`log_posterior` must be a function(theta) -> numeric.",
         call. = FALSE)
  }
  N <- nrow(draws)
  d <- ncol(draws)
  if (N < 4L) {
    stop("Need at least 4 posterior draws for bridge sampling.",
         call. = FALSE)
  }

  if (split) {
    half <- N %/% 2L
    fit_idx <- seq_len(half)
    score_idx <- (half + 1L):N
    fit_draws <- draws[fit_idx, , drop = FALSE]
    score_draws <- draws[score_idx, , drop = FALSE]
  } else {
    fit_draws <- draws
    score_draws <- draws
  }

  # Multivariate-normal proposal fit to fit_draws.
  prop_mean <- colMeans(fit_draws)
  prop_cov <- stats::cov(fit_draws)
  prop_cov <- prop_cov + diag(1e-10, d)

  # Cholesky for sampling and density eval.
  R <- tryCatch(chol(prop_cov),
                error = function(e) {
                  stop("Posterior covariance is singular; cannot fit ",
                       "Gaussian proposal. ",
                       "Drop redundant parameters or add jitter.",
                       call. = FALSE)
                })
  log_det_cov <- 2 * sum(log(diag(R)))

  log_prop <- function(theta_mat) {
    z <- t(forwardsolve(t(R), t(sweep(theta_mat, 2L, prop_mean, "-"))))
    -0.5 * d * log(2 * pi) - 0.5 * log_det_cov -
      0.5 * rowSums(z^2)
  }

  proposal_draws <- matrix(rnorm(n_proposal * d), nrow = n_proposal) %*% R
  proposal_draws <- sweep(proposal_draws, 2L, prop_mean, "+")

  # Numerator log densities: at proposal samples.
  # Denominator log densities: at posterior samples.
  log_p_prop <- vapply(seq_len(n_proposal),
                       function(i) log_posterior(proposal_draws[i, ]),
                       numeric(1L))
  log_g_prop <- log_prop(proposal_draws)

  log_p_post <- vapply(seq_len(nrow(score_draws)),
                       function(i) log_posterior(score_draws[i, ]),
                       numeric(1L))
  log_g_post <- log_prop(score_draws)

  # Drop any draws where log_posterior is non-finite (lets the user
  # supply log_post that errors on out-of-support draws).
  ok_prop <- is.finite(log_p_prop) & is.finite(log_g_prop)
  ok_post <- is.finite(log_p_post) & is.finite(log_g_post)
  if (sum(ok_prop) < 4L || sum(ok_post) < 4L) {
    stop("Too few finite log-posterior evaluations to run bridge sampling.",
         call. = FALSE)
  }
  log_p_prop <- log_p_prop[ok_prop]
  log_g_prop <- log_g_prop[ok_prop]
  log_p_post <- log_p_post[ok_post]
  log_g_post <- log_g_post[ok_post]

  N1 <- length(log_p_post)   # posterior sample size for ratio
  N2 <- length(log_p_prop)   # proposal sample size for ratio
  s1 <- N1 / (N1 + N2)
  s2 <- N2 / (N1 + N2)

  l1 <- log_p_post - log_g_post   # log p / log g at posterior draws
  l2 <- log_p_prop - log_g_prop   # log p / log g at proposal draws

  # Iterative scheme in log-space (Gronau et al. 2017 eqn 6):
  #   r_{t+1} = mean_j [ exp(l2_j) / (s1 exp(l2_j) + s2 r_t) ]
  #             /
  #             mean_i [ 1 / (s1 exp(l1_i) + s2 r_t) ]
  # Take logs and use logsumexp throughout.

  log_r <- 0
  iter_history <- numeric(0L)
  converged <- FALSE
  for (iter in seq_len(max_iter)) {
    # numerator: log mean exp(l2) / (s1 exp(l2) + s2 r) over j
    log_denom_j <- logsumexp_pair(log(s1) + l2, log(s2) + log_r)
    log_num <- logmeanexp(l2 - log_denom_j)

    # denominator: log mean 1 / (s1 exp(l1) + s2 r) over i
    log_denom_i <- logsumexp_pair(log(s1) + l1, log(s2) + log_r)
    log_den <- logmeanexp(-log_denom_i)

    log_r_new <- log_num - log_den
    iter_history <- c(iter_history, log_r_new)

    if (verbose) {
      cat(sprintf("  iter %3d: log r = %.10f\n", iter, log_r_new))
    }
    if (is.finite(log_r) && abs(log_r_new - log_r) < tol) {
      converged <- TRUE
      log_r <- log_r_new
      break
    }
    log_r <- log_r_new
  }

  # Relative MSE (Fruehwirth-Schnatter 2004 / bridgesampling::error_measures):
  # an approximate variance check on log r. Small re_sd => stable estimate.
  e2 <- exp(l2 - log_denom_j)
  re_sd <- sqrt(stats::var(e2) / N2) / mean(e2)

  list(
    log_marginal = log_r,
    n_iter = length(iter_history),
    converged = converged,
    re_sd = re_sd,
    proposal = list(mean = prop_mean, cov = prop_cov),
    iter_history = iter_history
  )
}


#' Numerically stable log(mean(exp(x)))
#' @keywords internal
logmeanexp <- function(x) {
  m <- max(x)
  m + log(mean(exp(x - m)))
}


#' Numerically stable log(exp(a) + exp(b)) elementwise (vectors / scalars)
#' @keywords internal
logsumexp_pair <- function(a, b) {
  m <- pmax(a, b)
  m + log(exp(a - m) + exp(b - m))
}
