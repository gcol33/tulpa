#' Independence Metropolis-Hastings with Laplace proposal
#'
#' @description
#' Sample from a posterior using independence Metropolis-Hastings with
#' a multivariate-t proposal centred at the Laplace mode, with the inverse
#' Hessian as its scale matrix. The t's polynomial tails dominate any
#' target whose tails are no heavier than its own, which is the condition
#' under which an independence sampler is uniformly ergodic; a Gaussian
#' proposal lighter-tailed than the target under-visits the target's tails
#' and the chain sticks there. The degrees of freedom are one package
#' setting. For posteriors that are well-
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
#' asymptotically correct for the supplied `log_posterior` under the
#' standard MH conditions. Posterior exactness requires that function to
#' evaluate the true density. Supplying a Laplace, EP or VI target leaves
#' that target's approximation error intact. The proposal affects efficiency.
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
#' @param scale Optional inflation factor on the proposal scale matrix
#'   (default 1.0); the scale matrix is `scale^2 * hessian^{-1}`.
#' @param init Optional starting parameter vector (default = `mode`).
#' @param thin Keep every `thin`-th post-warmup sample (default 1).
#' @param seed Optional integer RNG seed. Scoped to this call: the caller's
#'   `.Random.seed` is restored on exit, so two calls with the same `seed`
#'   give identical draws regardless of the surrounding RNG stream.
#' @param verbose Print acceptance rate at end (default `FALSE`).
#'
#' @return A list with class `tulpa_fit` carrying:
#'   * `draws`: `(n_iter - warmup) / thin` x `d` matrix of draws.
#'   * `means`: posterior means.
#'   * `accept_prob`: per-iteration acceptance indicators.
#'   * `mean_accept`: overall post-warmup acceptance rate.
#'   * `log_prob`: per-draw log posterior values.
#'   * `inference_mode`: `"exact"`.
#'   * `inference_tier`: `1L`.
#'   * `backend`: `"imh_laplace"`.
#'   * `proposal`: `list(mode, cov, scale, df)`, the proposal's location,
#'     scale matrix, inflation and degrees of freedom.
#'
#' @seealso [tulpa_laplace()] for the mode + Hessian,
#'   [bridge_sampling()] for marginal-likelihood estimation on the
#'   resulting draws.
#'
#' @examples
#' # Toy: Bernoulli logistic with one covariate.
#' set.seed(1)
#' n <- 100
#' x <- rnorm(n)
#' eta <- 0.3 + 1.2 * x
#' y <- rbinom(n, 1, plogis(eta))
#' X <- cbind(1, x)
#'
#' lap <- tulpa_laplace(y, n_trials = rep(1L, n), X = X,
#'                      family = "binomial")
#'
#' log_post <- function(beta) {
#'   eta <- as.numeric(X %*% beta)
#'   sum(y * eta - log1p(exp(eta))) +
#'     sum(dnorm(beta, 0, 10, log = TRUE))
#' }
#'
#' fit <- imh_laplace(log_post, mode = lap$mode[1:2],
#'                    hessian = lap$H_beta, n_iter = 1000)
#' fit$mean_accept
#' colMeans(fit$draws)
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
                        seed = NULL,
                        verbose = FALSE) {

  if (!is.function(log_posterior)) {
    stop("`log_posterior` must be a function(theta) -> numeric.",
         call. = FALSE)
  }
  .seed_scoped(seed)
  d <- length(mode)
  if (!is.matrix(hessian) || any(dim(hessian) != d)) {
    stop(sprintf("`hessian` must be a %d x %d matrix.", d, d),
         call. = FALSE)
  }
  .check_run_length(n_iter, warmup, "imh_laplace")
  if (scale <= 0) stop("`scale` must be positive.", call. = FALSE)

  fit <- .imh_laplace_chain(
    evaluate = function(theta) list(log_prob = log_posterior(theta)),
    mode = mode, hessian = hessian, n_iter = n_iter, warmup = warmup,
    scale = scale, init = init, thin = thin, verbose = verbose)
  fit$states <- NULL
  .finalize_fit(fit, param_names = names(mode))
}

# Multivariate-t independence proposal with the Laplace inverse precision as scale.
.imh_laplace_proposal <- function(mode, hessian, scale = 1) {
  d <- length(mode)
  if (length(scale) != 1L || !is.finite(scale) || scale <= 0) {
    stop("`scale` must be positive and finite.", call. = FALSE)
  }

  # Proposal: theta ~ t_df(mode, scale^2 * H^{-1}), a multivariate t with
  # location `mode` and scale matrix `scale^2 * H^{-1}`.
  df <- .imh_proposal("df")
  prop_cov <- tryCatch(scale^2 * .spd_inverse(hessian),
                       error = function(e) {
                         stop("Hessian is singular or not positive-definite; ",
                              "cannot invert. Pass a positive-definite ",
                              "H_beta or regularise.", call. = FALSE)
                       })
  R <- tryCatch(chol(prop_cov),
                error = function(e) {
                  stop("Proposal covariance not positive-definite. ",
                       "Try `scale = 1` or pre-condition the Hessian.",
                       call. = FALSE)
                })
  log_det_cov <- 2 * sum(log(diag(R)))
  log_norm <- lgamma((df + d) / 2) - lgamma(df / 2) -
    0.5 * d * log(df * pi) - 0.5 * log_det_cov

  # Proposal log-density at a vector.
  log_prop <- function(theta) {
    z <- forwardsolve(t(R), theta - mode)
    log_norm - 0.5 * (df + d) * log1p(sum(z^2) / df)
  }

  draw <- function(n) {
    z <- matrix(rnorm(n * d), nrow = n) /
      sqrt(stats::rchisq(n, df) / df)
    sweep(z %*% R, 2L, mode, `+`)
  }
  list(draw = draw, log_density = log_prop, mode = mode,
       cov = prop_cov, scale = scale, df = df)
}

# A rejection retains the parameter, density value and auxiliary state together.
.imh_laplace_transition <- function(theta, state, proposal, evaluate,
                                     proposed = drop(proposal$draw(1L))) {
  candidate <- evaluate(proposed)
  log_alpha <- (candidate$log_prob - proposal$log_density(proposed)) -
    (state$log_prob - proposal$log_density(theta))
  accepted <- is.finite(candidate$log_prob) &&
    log(stats::runif(1L)) < log_alpha
  if (accepted) {
    theta <- proposed
    state <- candidate
  }
  list(theta = theta, state = state, accepted = accepted, log_alpha = log_alpha)
}

# Independence MH on an extended state. The target value and auxiliary state
# are retained together on rejection, including for a random likelihood estimate.
.imh_laplace_chain <- function(evaluate, mode, hessian, n_iter, warmup,
                                scale = 1, init = NULL, thin = 1L,
                                verbose = FALSE) {
  d <- length(mode)
  .check_run_length(n_iter, warmup, "imh_laplace")
  if (length(thin) != 1L || !is.finite(thin) || thin < 1 || thin != floor(thin)) {
    stop("`thin` must be a positive whole number.", call. = FALSE)
  }
  proposal <- .imh_laplace_proposal(mode, hessian, scale)

  # Initial state.
  if (is.null(init)) init <- mode
  theta <- as.numeric(init)
  state_curr <- evaluate(theta)
  log_p_curr <- state_curr$log_prob
  if (!is.finite(log_p_curr)) {
    stop("`log_posterior(init)` is not finite; pick a different init.",
         call. = FALSE)
  }

  # Storage.
  draws_all <- matrix(NA_real_, nrow = n_iter, ncol = d)
  log_prob_all <- numeric(n_iter)
  accept_all <- logical(n_iter)
  keep_idx <- seq.int(warmup + 1L, n_iter, by = thin)
  states <- vector("list", length(keep_idx))
  kept <- 0L

  # Pre-draw all proposals: a t draw is a standard normal divided by
  # sqrt(W / df), W ~ chi^2_df, mapped through the scale matrix's factor.
  proposed <- proposal$draw(n_iter)

  for (t in seq_len(n_iter)) {
    step <- .imh_laplace_transition(theta, state_curr, proposal, evaluate,
                                     proposed[t, ])
    theta <- step$theta
    state_curr <- step$state
    log_p_curr <- state_curr$log_prob
    accept_all[t] <- step$accepted
    draws_all[t, ] <- theta
    log_prob_all[t] <- log_p_curr
    if (t > warmup && (t - warmup - 1L) %% thin == 0L) {
      kept <- kept + 1L
      states[[kept]] <- state_curr
    }
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
    proposal = proposal[c("mode", "cov", "scale", "df")]
  )
  fit$states <- states
  fit
}
