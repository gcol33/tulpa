# multinomial.R
# ------------------------------------------------------------------------------
# Baseline-category multinomial (nominal, unordered K-class) logistic regression
# via a Laplace fit. The per-observation coupled multinomial log-likelihood,
# score and (K-1)x(K-1) Fisher information come from the validated native kernel
# (cpp_multinomial_logit_terms, src/multinomial_logit.h); this driver assembles
# them into the joint Newton solve over the (K-1) coefficient blocks and returns
# a Laplace-approximated posterior. No engine change -- it consumes the existing
# multi-process likelihood unit directly.
#
#   eta_i = (X beta)_i  in R^{K-1};  class c_i in 1..K (K = baseline)
#   grad wrt beta_{.j} = X' g_{.j},  g_{ij} = [c_i == j] - p_{ij}
#   Hessian block (j,l) = X' diag(nh_{i,jl}) X,  nh = p_j([j==l] - p_l) (PSD)
# ------------------------------------------------------------------------------

#' Multinomial (nominal K-class) logistic regression via Laplace
#'
#' @description
#' Fits a baseline-category multinomial logit model: for a K-level unordered
#' response, classes `1..K-1` each get their own linear predictor and class `K`
#' is the baseline. The coupled multinomial likelihood is solved by a Newton
#' step to the penalized mode (a Gaussian ridge prior on the coefficients) and
#' summarized by a Laplace approximation, reusing the native multinomial kernel.
#'
#' @param formula Model formula; the response must be a factor (or coercible to
#'   one) with >= 3 levels. The baseline is the last level.
#' @param data A data frame.
#' @param beta_prior_sd SD of the mean-zero Gaussian prior on every coefficient
#'   (ridge; default 10). A finite value keeps the mode finite under separation.
#' @param control List of numerical knobs: `max_iter` (default 100), `tol`
#'   (default 1e-8), `n_draws` (posterior draws, default 2000), `seed`.
#'
#' @return A `tulpa_fit` (subclass `tulpa_multinomial`) with `coef` (named
#'   `class:term`), `vcov`, `draws`, `log_marginal`, `classes`, `baseline`, and
#'   the standard generic-method support.
#'
#' @seealso [tulpa()] for single-process GLMMs.
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 300L; x <- rnorm(n)
#' eta <- cbind(0.5 + 1.0 * x, -0.3 - 0.8 * x)          # classes 1, 2 vs baseline 3
#' P <- cbind(exp(eta), 1); P <- P / rowSums(P)
#' y <- factor(apply(P, 1, function(pr) sample.int(3L, 1L, prob = pr)))
#' fit <- tulpa_multinomial(y ~ x, data = data.frame(y = y, x = x))
#' coef(fit)
#' }
#' @export
tulpa_multinomial <- function(formula, data, beta_prior_sd = 10,
                              control = list()) {
  .check_control(control, .CONTROL_KEYS$multinomial, "tulpa_multinomial")
  max_iter <- as.integer(control$max_iter %||% 100L)
  tol      <- control$tol %||% 1e-8
  n_draws  <- as.integer(control$n_draws %||% 2000L)

  mf <- stats::model.frame(formula, data)
  y  <- stats::model.response(mf)
  if (!is.factor(y)) y <- factor(y)
  K  <- nlevels(y)
  if (K < 3L) {
    stop("tulpa_multinomial() needs a response with >= 3 levels; for 2 use ",
         "family = \"binomial\".", call. = FALSE)
  }
  K1  <- K - 1L
  cls <- as.integer(y)                       # 1..K, K = baseline
  X   <- stats::model.matrix(stats::terms(mf), mf)
  n   <- nrow(X); p <- ncol(X)
  tau <- 1 / (beta_prior_sd^2)               # ridge precision

  beta <- matrix(0, p, K1)                   # [p x (K-1)]
  dim_b <- p * K1
  H <- matrix(0, dim_b, dim_b)
  loglik <- -Inf

  for (it in seq_len(max_iter)) {
    eta <- X %*% beta                        # [n x K1]
    G   <- matrix(0, p, K1)                  # score
    H[] <- 0
    ll  <- 0
    # Per-observation kernel terms, assembled by the chain rule into (beta) space.
    nh_store <- array(0, c(n, K1, K1))
    g_store  <- matrix(0, n, K1)
    for (i in seq_len(n)) {
      out <- cpp_multinomial_logit_terms(eta[i, ], cls[i])
      ll  <- ll + out$ll
      g_store[i, ]   <- out$grad
      nh_store[i, , ] <- out$neg_hess
    }
    for (j in seq_len(K1)) {
      G[, j] <- crossprod(X, g_store[, j]) - tau * beta[, j]
      for (l in seq_len(K1)) {
        Hjl <- crossprod(X, nh_store[, j, l] * X)   # X' diag(nh_jl) X
        if (j == l) Hjl <- Hjl + diag(tau, p)
        H[((j - 1) * p + 1):(j * p), ((l - 1) * p + 1):(l * p)] <- Hjl
      }
    }
    lp <- ll - 0.5 * tau * sum(beta^2)
    step <- tryCatch(solve(H, as.numeric(G)),
                     error = function(e) solve(H + diag(1e-6, dim_b), as.numeric(G)))
    beta <- beta + matrix(step, p, K1)
    if (max(abs(step)) < tol) { loglik <- ll; break }
    loglik <- ll
  }

  # Laplace marginal at the mode; H is the negative Hessian of the log-joint (PD).
  ch <- chol(H)
  logdetH <- 2 * sum(log(diag(ch)))
  log_prior <- -0.5 * dim_b * log(2 * pi * beta_prior_sd^2) -
    0.5 * tau * sum(beta^2)
  log_marginal <- loglik + log_prior + 0.5 * dim_b * log(2 * pi) - 0.5 * logdetH

  # Names: "class:term" in class-major, term-within order.
  lev <- levels(y)
  cn  <- colnames(X)
  pn  <- as.vector(vapply(seq_len(K1), function(j) paste0(lev[j], ":", cn),
                          character(p)))
  bhat <- setNames(as.numeric(beta), pn)
  V <- chol2inv(ch); dimnames(V) <- list(pn, pn)

  .seed_scoped(control$seed)
  draws <- .ps_rmvnorm(n_draws, bhat, V)     # reuse the Cholesky sampler

  fit <- list(
    coefficients = bhat, vcov = V, draws = draws,
    means = bhat, param_names = pn,
    log_marginal = log_marginal, converged = it < max_iter,
    classes = lev, baseline = lev[K], n_classes = K,
    family = "multinomial", formula = formula,
    model_matrix = X, backend = "multinomial_laplace",
    inference_tier = 2L, inference_mode = "structured",
    draws_kind = "iid"
  )
  class(fit) <- c("tulpa_multinomial", "tulpa_fit")
  fit
}

#' @export
vcov.tulpa_multinomial <- function(object, ...) object$vcov

#' @export
coef.tulpa_multinomial <- function(object, ...) object$coefficients

#' @export
print.tulpa_multinomial <- function(x, ...) {
  cat(sprintf("Multinomial logit (%d classes, baseline '%s'), Laplace fit\n",
              x$n_classes, x$baseline))
  cat(sprintf("log marginal: %.2f\n\n", x$log_marginal))
  print(round(x$coefficients, 4))
  invisible(x)
}
