#' Fit a Gaussian linear model via tulpa's generic engine
#'
#' Proof-of-concept function demonstrating the tulpa generic interface: the
#' Gaussian likelihood is supplied as a `LikelihoodSpec` and fitted by the
#' production NUTS sampler every other ModelData backend runs (dual-averaged
#' step size, adapted diagonal mass matrix, reverse-mode AD gradient checked
#' against finite differences before sampling). Fits
#' `y ~ Normal(X * beta, sigma)`.
#'
#' @param formula A formula (e.g., y ~ x1 + x2)
#' @param data A data frame
#' @param beta_prior Fixed-effect prior as `list(mean, sd)`: a mean-zero
#'   (`mean = 0`) Gaussian on every coefficient with SD `sd`
#'   (default the engine default, `prior_normal(0, 2.5)`).
#' @param control List of sampler knobs: `iter` (total iterations, default
#'   2000), `warmup` (default 1000), `max_treedepth` (default 10),
#'   `adapt_delta` (target acceptance statistic, default 0.8), `seed` (`NULL`
#'   draws from the session RNG).
#'
#' @return A `tulpa_fit` list with `draws` (post-warmup draws of `beta[j]` and
#'   `log_sigma`), `means`, `sigma` (`exp()` of the posterior mean of
#'   `log_sigma`), `accept_rate` (the mean NUTS acceptance statistic over the
#'   post-warmup iterations), the per-iteration `accept_prob`, `divergent`,
#'   `treedepth` and `log_prob`, the adapted step size `epsilon`, and
#'   `formula`, `N`, `p`.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(x = rnorm(40))
#' d$y <- 1 + 2 * d$x + rnorm(40, 0, 0.6)
#' fit <- tulpa_gaussian(y ~ x, d, control = list(iter = 1000, warmup = 500,
#'                                               seed = 1))
#' fit$means
#' fit$accept_rate
#' }
#' @export
tulpa_gaussian <- function(formula, data,
                           beta_prior = .tulpa_default_beta_prior("gaussian"),
                           control = list()) {
  tulpa_check_control(control, .CONTROL_KEYS$gaussian, "tulpa_gaussian")
  sigma_beta <- .beta_prior_ridge_sd(beta_prior, .tulpa_prior_sd("gaussian"))
  iter       <- as.integer(control$iter %||% 2000L)
  warmup     <- as.integer(control$warmup %||% 1000L)
  .check_run_length(iter, warmup, "tulpa_gaussian", n_iter_name = "iter")
  max_treedepth <- .check_count(control$max_treedepth %||% 10L,
                                "control$max_treedepth")
  adapt_delta <- .check_unit_interval(control$adapt_delta %||% 0.8,
                                      "control$adapt_delta")
  seed       <- control$seed

  # Parse formula
  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- model.matrix(formula, data)

  if (!is.numeric(y)) {
    stop("`formula` must name a numeric response for tulpa_gaussian().",
         call. = FALSE)
  }
  if (length(y) != nrow(X)) {
    stop(sprintf(
      "length(y) (%d) must equal nrow(model.matrix) (%d).",
      length(y), nrow(X)), call. = FALSE)
  }

  # The generic LikelihoodSpec path through production NUTS. Its gradient is
  # the arena-AD one; the fixed-step HMC this used to run took a central-
  # difference gradient of the whole log-posterior, 2p + 1 evaluations per
  # leapfrog step, at about a second per iteration on n = 40 (#897).
  fit <- cpp_tulpa_fit_generic(
    y_r = y,
    X_r = X,
    sigma_beta = sigma_beta,
    n_iter = iter,
    n_warmup = warmup,
    max_treedepth = max_treedepth,
    adapt_delta = adapt_delta,
    seed = as.integer(seed %||% sample.int(.Machine$integer.max, 1L)),
    verbose = FALSE
  )

  fit$accept_rate <- mean(fit$accept_prob)
  # Add sigma (not log_sigma) to means
  fit$sigma <- exp(fit$means["log_sigma"])
  fit$formula <- formula
  fit$N <- length(y)
  fit$p <- ncol(X)

  class(fit) <- "tulpa_fit"
  fit
}
