#' Pathfinder: variational warm-start via L-BFGS + ELBO scoring
#'
#' @description
#' Single-path Pathfinder (Zhang, Carpenter, Gelman, Vehtari 2022):
#' run L-BFGS toward the posterior mode, fit a Gaussian at the optimum
#' using the inverse-Hessian estimate, and report draws plus the ELBO.
#' Cheap, derivative-only, embarrassingly parallel — meant as an HMC
#' warm-start, an initialiser for [imh_laplace()], or a quick
#' sanity check on the Laplace approximation.
#'
#' This implementation is **single-path only**. The full multi-path
#' Pathfinder (K parallel L-BFGS runs + mixture proposal + Pareto-
#' smoothed importance reweighting) is a follow-on. The single-path
#' version is what most users want as a Laplace-equivalent diagnostic.
#'
#' @section Tier:
#' Tier 2 (Structured). The output is a Gaussian approximation, not
#' samples from the exact posterior — same epistemic class as
#' [tulpa_laplace()]. Pair with [imh_laplace()] for an exact-tier
#' upgrade.
#'
#' @param log_posterior Function `function(theta) -> numeric` returning
#'   the *unnormalized* log posterior at `theta`.
#' @param init Numeric vector: initial point for L-BFGS. Should be in
#'   the support of the posterior (finite log_posterior).
#' @param grad_log_posterior Optional function returning the gradient
#'   of `log_posterior` at `theta`. If `NULL`, gradients are computed
#'   numerically via [stats::optim]'s built-in finite differences.
#' @param n_draws Number of draws from the fitted Gaussian (default 1000).
#' @param max_iter L-BFGS iteration cap (default 100).
#' @param tol Gradient-norm tolerance for L-BFGS convergence
#'   (default 1e-6).
#' @param verbose Print L-BFGS / ELBO summary at end (default FALSE).
#'
#' @return A list with class `tulpa_fit` carrying:
#'   * `draws`: `n_draws x d` matrix of Gaussian draws at the mode.
#'   * `means`: posterior means (= mode for a Gaussian fit).
#'   * `mode`: the L-BFGS mode.
#'   * `cov`: the proposal covariance (`solve(-hessian)`).
#'   * `elbo`: Monte-Carlo estimate of `E_q[log p - log q]`.
#'   * `n_iter`: L-BFGS iterations.
#'   * `converged`: logical.
#'   * `inference_mode`: `"structured"`.
#'   * `inference_tier`: `2L`.
#'   * `backend`: `"pathfinder"`.
#'
#' @references
#' Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022).
#' Pathfinder: parallel quasi-Newton variational inference.
#' *Journal of Machine Learning Research*, 23(306), 1–49.
#'
#' @seealso [imh_laplace()] for an exact-tier MH using the Pathfinder
#'   Gaussian as proposal; [bridge_sampling()] for marginal-likelihood
#'   estimation on the resulting draws.
#'
#' @examples
#' \dontrun{
#'   # Toy: 2-D conjugate normal.
#'   y <- c(0.5, -0.7)
#'   log_post <- function(t) {
#'     sum(dnorm(y, t, 1, log = TRUE)) +
#'       sum(dnorm(t, 0, sqrt(10), log = TRUE))
#'   }
#'   pf <- pathfinder(log_post, init = c(0, 0), n_draws = 2000)
#'   pf$mode      # near c(0.45, -0.64)
#'   pf$elbo
#' }
#'
#' @export
pathfinder <- function(log_posterior,
                       init,
                       grad_log_posterior = NULL,
                       n_draws = 1000L,
                       max_iter = 100L,
                       tol = 1e-6,
                       verbose = FALSE) {

  if (!is.function(log_posterior)) {
    stop("`log_posterior` must be a function(theta) -> numeric.",
         call. = FALSE)
  }
  d <- length(init)
  if (n_draws < 2L) stop("`n_draws` must be >= 2.", call. = FALSE)

  # optim minimises; we maximise log_posterior, so negate.
  fn <- function(theta) -log_posterior(theta)
  gr <- if (!is.null(grad_log_posterior)) {
    function(theta) -grad_log_posterior(theta)
  } else {
    NULL
  }

  if (!is.finite(log_posterior(init))) {
    stop("`log_posterior(init)` is not finite; pick a different init.",
         call. = FALSE)
  }

  opt <- stats::optim(
    par = init, fn = fn, gr = gr,
    method = "L-BFGS-B",
    hessian = TRUE,
    control = list(
      maxit = max_iter,
      pgtol = tol,
      factr = 1e7      # ~ standard optim default
    )
  )

  mode <- opt$par
  # opt$hessian is the Hessian of fn = -log_posterior, which equals
  # the negative Hessian of log_posterior — exactly the precision
  # matrix at the mode.
  H <- opt$hessian
  H <- 0.5 * (H + t(H))   # symmetrise

  prop_cov <- tryCatch(solve(H),
                       error = function(e) {
                         stop("Hessian is singular at the mode; cannot ",
                              "form Gaussian fit. ",
                              "Try a different `init`, regularise the ",
                              "log-posterior, or check identifiability.",
                              call. = FALSE)
                       })
  prop_cov <- 0.5 * (prop_cov + t(prop_cov))
  R <- tryCatch(chol(prop_cov),
                error = function(e) {
                  stop("Inverse Hessian is not positive-definite. ",
                       "L-BFGS may not be at a local maximum.",
                       call. = FALSE)
                })
  log_det_cov <- 2 * sum(log(diag(R)))

  # Draws from N(mode, H^{-1}).
  z <- matrix(rnorm(n_draws * d), nrow = n_draws)
  draws <- sweep(z %*% R, 2L, mode, "+")

  log_q_draws <- -0.5 * d * log(2 * pi) - 0.5 * log_det_cov -
    0.5 * rowSums(z^2)
  log_p_draws <- vapply(seq_len(n_draws),
                        function(i) log_posterior(draws[i, ]),
                        numeric(1L))
  ok <- is.finite(log_p_draws)
  if (sum(ok) < 2L) {
    stop("Too few finite log_posterior evaluations on Pathfinder draws. ",
         "Mode may be in a low-density region.", call. = FALSE)
  }
  elbo <- mean(log_p_draws[ok] - log_q_draws[ok])

  if (verbose) {
    cat(sprintf("pathfinder: L-BFGS converged = %s in %d iter; ELBO = %.3f\n",
                opt$convergence == 0L,
                opt$counts["function"],
                elbo))
  }

  fit <- list(
    draws = draws,
    means = colMeans(draws),
    n_params = d,
    n_samples = nrow(draws),
    mode = mode,
    cov = prop_cov,
    hessian = H,
    elbo = elbo,
    log_prob = log_p_draws,
    n_iter = as.integer(opt$counts["function"]),
    converged = (opt$convergence == 0L),
    inference_mode = "structured",
    inference_tier = 2L,
    backend = "pathfinder"
  )
  class(fit) <- c("tulpa_pathfinder_fit", "tulpa_fit")
  fit
}
