#' Independence Metropolis-Hastings with Laplace proposal
#'
#' @description
#' Sample from a posterior using independence Metropolis-Hastings with
#' a multivariate-normal proposal centred at the Laplace mode and
#' scaled by the inverse Hessian. For posteriors that are well-
#' approximated by a Gaussian near the mode this is dramatically
#' cheaper than HMC: each iteration is one log-posterior evaluation
#' plus one accept/reject step. Embarrassingly parallel across chains.
#'
#' Use cases:
#' * Quick draws when Laplace is "almost right" but you want
#'   asymptotically-correct samples.
#' * Cross-validation, simulation studies, or other repeated-fit
#'   workflows where HMC startup cost dominates.
#' * Sanity check on the Laplace approximation: low IMH acceptance
#'   means the posterior is far from Gaussian and Laplace is biased.
#'
#' @section Tier:
#' Tier 1 (Exact). The MH accept/reject step makes the chain
#' asymptotically correct under the standard MH conditions. Tier
#' status does not depend on Laplace's quality — only its quality
#' affects efficiency.
#'
#' @param log_posterior Function `function(theta) -> numeric` returning
#'   the *unnormalized* log posterior at a single parameter vector.
#' @param mode Numeric vector of length `d`: the Laplace mode (i.e.,
#'   `tulpa_laplace(...)$mode[seq_len(d)]` for the fixed-effects
#'   block, or any other mode you trust).
#' @param hessian Symmetric positive-definite `d x d` matrix: the
#'   negative Hessian of `log_posterior` at `mode` (or `H_beta`
#'   from `tulpa_laplace`).
#' @param n_iter Total iterations including warmup (default 2000).
#' @param warmup Warmup iterations to discard (default `n_iter / 2`).
#' @param scale Optional inflation factor on the proposal covariance
#'   (default 1.0). Values slightly > 1 (e.g., 1.5) give heavier-tailed
#'   proposals that improve mixing when Laplace underestimates posterior
#'   spread.
#' @param init Optional starting parameter vector (default = `mode`).
#' @param thin Keep every `thin`-th post-warmup sample (default 1).
#' @param verbose Print acceptance rate at end (default `FALSE`).
#'
#' @return A list with class `tulpa_fit` carrying:
#'   * `draws`: `(n_iter - warmup) / thin` × `d` matrix of draws.
#'   * `means`: posterior means.
#'   * `accept_prob`: per-iteration acceptance indicators.
#'   * `mean_accept`: overall post-warmup acceptance rate.
#'   * `log_prob`: per-draw log posterior values.
#'   * `inference_mode`: `"exact"`.
#'   * `inference_tier`: `1L`.
#'   * `backend`: `"imh_laplace"`.
#'
#' @seealso [tulpa_laplace()] for the mode + Hessian,
#'   [bridge_sampling()] for marginal-likelihood estimation on the
#'   resulting draws.
#'
#' @examples
#' \dontrun{
#'   # Toy: Bernoulli logistic with one covariate.
#'   set.seed(1)
#'   n <- 200
#'   x <- rnorm(n)
#'   eta <- 0.3 + 1.2 * x
#'   y <- rbinom(n, 1, plogis(eta))
#'   X <- cbind(1, x)
#'
#'   lap <- tulpa_laplace(y, n_trials = rep(1L, n), X = X,
#'                        family = "binomial")
#'
#'   log_post <- function(beta) {
#'     eta <- as.numeric(X %*% beta)
#'     sum(y * eta - log1p(exp(eta))) +
#'       sum(dnorm(beta, 0, 10, log = TRUE))
#'   }
#'
#'   fit <- imh_laplace(log_post, mode = lap$mode[1:2],
#'                      hessian = lap$H_beta, n_iter = 4000)
#'   fit$mean_accept
#'   colMeans(fit$draws)
#' }
#'
#' @export
imh_laplace <- function(log_posterior,
                        mode,
                        hessian,
                        n_iter = 2000L,
                        warmup = n_iter %/% 2L,
                        scale = 1.0,
                        init = NULL,
                        thin = 1L,
                        verbose = FALSE) {

  if (!is.function(log_posterior)) {
    stop("`log_posterior` must be a function(theta) -> numeric.",
         call. = FALSE)
  }
  d <- length(mode)
  if (!is.matrix(hessian) || any(dim(hessian) != d)) {
    stop(sprintf("`hessian` must be a %d x %d matrix.", d, d),
         call. = FALSE)
  }
  if (n_iter < 2L || warmup < 0L || warmup >= n_iter) {
    stop("Need 0 <= warmup < n_iter and n_iter >= 2.", call. = FALSE)
  }
  if (scale <= 0) stop("`scale` must be positive.", call. = FALSE)

  # Proposal: theta ~ N(mode, scale^2 * H^{-1}).
  prop_cov <- tryCatch(scale^2 * solve(hessian),
                       error = function(e) {
                         stop("Hessian is singular; cannot invert. ",
                              "Pass a positive-definite H_beta or ",
                              "regularise.", call. = FALSE)
                       })
  prop_cov <- 0.5 * (prop_cov + t(prop_cov))   # symmetrise
  R <- tryCatch(chol(prop_cov),
                error = function(e) {
                  stop("Proposal covariance not positive-definite. ",
                       "Try `scale = 1` or pre-condition the Hessian.",
                       call. = FALSE)
                })
  log_det_cov <- 2 * sum(log(diag(R)))

  # Proposal log-density at a vector.
  log_prop <- function(theta) {
    z <- forwardsolve(t(R), theta - mode)
    -0.5 * d * log(2 * pi) - 0.5 * log_det_cov -
      0.5 * sum(z^2)
  }

  # Initial state.
  if (is.null(init)) init <- mode
  theta <- as.numeric(init)
  log_p_curr <- log_posterior(theta)
  log_q_curr <- log_prop(theta)
  if (!is.finite(log_p_curr)) {
    stop("`log_posterior(init)` is not finite; pick a different init.",
         call. = FALSE)
  }

  # Storage.
  draws_all <- matrix(NA_real_, nrow = n_iter, ncol = d)
  log_prob_all <- numeric(n_iter)
  accept_all <- logical(n_iter)

  # Pre-draw all proposals (vectorised standard normals once).
  z_all <- matrix(rnorm(n_iter * d), nrow = n_iter)

  for (t in seq_len(n_iter)) {
    prop <- mode + as.numeric(z_all[t, ] %*% R)
    log_p_prop <- log_posterior(prop)
    log_q_prop <- log_prop(prop)

    # IMH acceptance: alpha = (p(prop)/q(prop)) / (p(curr)/q(curr))
    log_alpha <- (log_p_prop - log_q_prop) - (log_p_curr - log_q_curr)
    if (is.finite(log_p_prop) &&
        log(stats::runif(1L)) < log_alpha) {
      theta <- prop
      log_p_curr <- log_p_prop
      log_q_curr <- log_q_prop
      accept_all[t] <- TRUE
    }
    draws_all[t, ] <- theta
    log_prob_all[t] <- log_p_curr
  }

  keep_idx <- seq.int(warmup + 1L, n_iter, by = thin)
  draws <- draws_all[keep_idx, , drop = FALSE]
  log_prob <- log_prob_all[keep_idx]
  accept_kept <- accept_all[keep_idx]
  mean_accept <- mean(accept_kept)

  if (verbose) {
    cat(sprintf("imh_laplace: post-warmup acceptance = %.3f (%d / %d)\n",
                mean_accept, sum(accept_kept), length(accept_kept)))
    if (mean_accept < 0.10) {
      cat("  WARNING: acceptance < 0.10. Posterior is far from the ",
          "Laplace Gaussian; consider HMC or `scale > 1`.\n", sep = "")
    }
  }

  fit <- list(
    draws = draws,
    means = colMeans(draws),
    n_params = d,
    n_samples = nrow(draws),
    log_prob = log_prob,
    accept_prob = accept_kept,
    mean_accept = mean_accept,
    inference_mode = "exact",
    inference_tier = 1L,
    backend = "imh_laplace",
    proposal = list(mode = mode, cov = prop_cov, scale = scale)
  )
  class(fit) <- c("tulpa_imh_fit", "tulpa_fit")
  fit
}
