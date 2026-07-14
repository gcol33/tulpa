#' Metropolis-Adjusted Langevin Algorithm (MALA)
#'
#' @description
#' Sample from a posterior using MALA: a Langevin proposal driven by
#' the gradient of the log posterior, corrected by a Metropolis-
#' Hastings accept/reject step. Each iteration is one log-posterior +
#' one gradient evaluation. The natural stepping stone between
#' random-walk MH and HMC: cheaper per iteration than HMC (no leapfrog
#' integration), better-mixing than RWMH because the drift term moves
#' toward higher density.
#'
#' Step size `epsilon` is adapted via dual averaging during warmup to
#' target an acceptance rate of 0.574 (Roberts & Rosenthal 1998 optimal
#' for high-dimensional Gaussians).
#'
#' @section Tier:
#' Tier 1 (Exact). The MH step makes the chain asymptotically correct.
#' Mixing depends on whether the gradient gives useful local geometry --
#' poor for posteriors with very different scales across dimensions
#' (use a preconditioner or HMC instead).
#'
#' @param log_posterior Function `function(theta) -> numeric` returning
#'   the *unnormalized* log posterior.
#' @param grad_log_posterior Function `function(theta) -> numeric vector`
#'   returning the gradient at `theta`.
#' @param init Numeric vector: initial state (must give finite
#'   log_posterior).
#' @param n_iter Total iterations including warmup (default 2000).
#' @param warmup Warmup iterations (step-size adaptation + discarded;
#'   default `n_iter / 2`).
#' @param epsilon Initial step size (default 0.1). Adapted during warmup.
#' @param target_accept Target acceptance during warmup adaptation
#'   (default 0.574, the Roberts & Rosenthal 1998 optimum).
#' @param mass_diag Optional preconditioner: a length-`d` vector of
#'   per-dimension scales, equivalent to using `diag(mass_diag)` as
#'   a metric. Default `rep(1, d)`. For posteriors with very different
#'   scales across dimensions, set this to (an estimate of) the
#'   posterior SDs.
#' @param thin Keep every `thin`-th post-warmup sample (default 1).
#' @param verbose Print acceptance + step-size summary at end (default
#'   FALSE).
#'
#' @return A list with class `tulpa_fit` carrying:
#'   * `draws`: post-warmup draws.
#'   * `means`, `n_params`, `n_samples`, `log_prob`.
#'   * `accept_prob`: per-iteration accepted indicators (post-warmup).
#'   * `mean_accept`: post-warmup acceptance rate.
#'   * `epsilon`: final adapted step size.
#'   * `inference_mode`: `"exact"`.
#'   * `inference_tier`: `1L`.
#'   * `backend`: `"mala"`.
#'
#' @references
#' Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of
#' Langevin distributions and their discrete approximations.
#' *Bernoulli*, 2(4), 341-363.
#'
#' Roberts, G. O., & Rosenthal, J. S. (1998). Optimal scaling of
#' discrete approximations to Langevin diffusions. *JRSS B*, 60(1),
#' 255-268.
#'
#' @examples
#' log_post <- function(t) -0.5 * sum((t - c(1, 2))^2)
#' grad <- function(t) -(t - c(1, 2))
#' fit <- mala(log_post, grad, init = c(0, 0), n_iter = 1000)
#' colMeans(fit$draws)  # near c(1, 2)
#' fit$mean_accept      # should adapt toward 0.574
#'
#' @export
mala <- function(log_posterior,
                 grad_log_posterior,
                 init,
                 n_iter = 2000L,
                 warmup = n_iter %/% 2L,
                 epsilon = 0.1,
                 target_accept = 0.574,
                 mass_diag = NULL,
                 thin = 1L,
                 verbose = FALSE) {

  if (!is.function(log_posterior) || !is.function(grad_log_posterior)) {
    stop("`log_posterior` and `grad_log_posterior` must be functions.",
         call. = FALSE)
  }
  d <- length(init)
  if (n_iter < 2L || warmup < 0L || warmup >= n_iter) {
    stop("Need 0 <= warmup < n_iter and n_iter >= 2.", call. = FALSE)
  }
  if (epsilon <= 0) stop("`epsilon` must be positive.", call. = FALSE)

  if (is.null(mass_diag)) mass_diag <- rep(1, d)
  if (length(mass_diag) != d || any(mass_diag <= 0)) {
    stop("`mass_diag` must be a positive vector of length d.",
         call. = FALSE)
  }
  M_inv <- mass_diag        # we use sqrt(M_inv) for the noise scale

  theta <- as.numeric(init)
  log_p_curr <- log_posterior(theta)
  grad_curr <- grad_log_posterior(theta)
  if (!is.finite(log_p_curr)) {
    stop("`log_posterior(init)` is not finite.", call. = FALSE)
  }

  draws_all <- matrix(NA_real_, nrow = n_iter, ncol = d)
  log_prob_all <- numeric(n_iter)
  accept_all <- logical(n_iter)
  alpha_all <- numeric(n_iter)

  # Dual-averaging adaptation (Hoffman & Gelman 2014, eqn 6) on
  # log_eps. Same scheme as our NUTS warmup; here we target the MH
  # acceptance directly.
  log_eps <- log(epsilon)
  log_eps_bar <- 0
  H_bar <- 0
  mu <- log(10 * epsilon)
  gamma <- 0.05
  t0 <- 10
  kappa <- 0.75

  # MALA proposal density q(theta' | theta).
  # theta' ~ N(theta + (eps^2/2) * M_inv * grad, eps^2 * M_inv)
  log_q <- function(to, from, grad_from, eps) {
    drift <- from + 0.5 * eps^2 * M_inv * grad_from
    z <- (to - drift) / (eps * sqrt(M_inv))
    -0.5 * d * log(2 * pi) - sum(log(eps * sqrt(M_inv))) -
      0.5 * sum(z^2)
  }

  for (t in seq_len(n_iter)) {
    eps <- exp(log_eps)
    z <- rnorm(d)
    prop <- theta + 0.5 * eps^2 * M_inv * grad_curr +
      eps * sqrt(M_inv) * z

    log_p_prop <- log_posterior(prop)
    if (!is.finite(log_p_prop)) {
      log_alpha <- -Inf
      grad_prop <- grad_curr   # placeholder
    } else {
      grad_prop <- grad_log_posterior(prop)
      log_alpha <- (log_p_prop + log_q(theta, prop, grad_prop, eps)) -
        (log_p_curr + log_q(prop, theta, grad_curr, eps))
    }

    accepted <- log(stats::runif(1L)) < log_alpha
    if (accepted) {
      theta <- prop
      log_p_curr <- log_p_prop
      grad_curr <- grad_prop
    }
    draws_all[t, ] <- theta
    log_prob_all[t] <- log_p_curr
    accept_all[t] <- accepted
    alpha_all[t] <- min(1, exp(log_alpha))

    if (t <= warmup) {
      eta <- 1 / (t + t0)
      H_bar <- (1 - eta) * H_bar +
        eta * (target_accept - alpha_all[t])
      log_eps <- mu - sqrt(t) / gamma * H_bar
      eta_bar <- t^(-kappa)
      log_eps_bar <- eta_bar * log_eps + (1 - eta_bar) * log_eps_bar
    } else if (t == warmup + 1L) {
      log_eps <- log_eps_bar   # freeze adapted value
    }
  }

  keep_idx <- seq.int(warmup + 1L, n_iter, by = thin)
  draws <- draws_all[keep_idx, , drop = FALSE]
  log_prob <- log_prob_all[keep_idx]
  accept_kept <- accept_all[keep_idx]
  mean_accept <- mean(accept_kept)

  if (verbose) {
    cat(sprintf("mala: epsilon = %.4f (adapted), accept = %.3f\n",
                exp(log_eps), mean_accept))
  }

  fit <- list(
    draws = draws,
    means = colMeans(draws),
    n_params = d,
    n_samples = nrow(draws),
    log_prob = log_prob,
    accept_prob = accept_kept,
    mean_accept = mean_accept,
    epsilon = exp(log_eps),
    mass_diag = mass_diag,
    inference_mode = "exact",
    inference_tier = 1L,
    backend = "mala"
  )
  class(fit) <- c("tulpa_mala_fit", "tulpa_fit")
  fit
}
