# posterior_predict.R
# ------------------------------------------------------------------------------
# Posterior-predictive replicates for tulpa_fit objects: rebuild the in-sample
# linear predictor per posterior draw (fixed effects + formula random effects +
# offset), push it through the family's sampling function (.FAMILY_OPS$sample),
# and return a draws x n_obs matrix. simulate.tulpa_fit() is the base-R alias;
# pp_check() falls back to this when the fit carries no stored y_rep.
# ------------------------------------------------------------------------------

# Sparse map from the random-effect coefficient vector (group-major,
# coef-within-group -- the layout .tulpa_param_layout names and the draws tail
# uses) to the per-observation RE contribution: eta_re = b' M with M (n_re x
# n_obs). Built from the $re_design rows tulpa() attaches. NULL when the fit
# carries no formula RE terms or no design.
#' @keywords internal
.tulpa_re_map <- function(object) {
  des <- object$re_design
  if (is.null(des) || !length(des)) return(NULL)
  X <- object$model_matrix
  if (is.null(X)) return(NULL)
  n_obs <- nrow(X)

  i_idx <- integer(0); j_idx <- integer(0); x_val <- numeric(0)
  offset <- 0L
  for (rt in des) {
    nc <- rt$n_coefs %||% 1L
    Z  <- cbind(
      if (isTRUE(rt$has_intercept)) rep(1, n_obs),
      rt$slope_matrix
    )
    if (is.null(Z) || ncol(Z) != nc) return(NULL)
    for (cc in seq_len(nc)) {
      i_idx <- c(i_idx, offset + (rt$group_idx - 1L) * nc + cc)
      j_idx <- c(j_idx, seq_len(n_obs))
      x_val <- c(x_val, Z[, cc])
    }
    offset <- offset + (rt$n_groups %||% max(rt$group_idx)) * nc
  }
  Matrix::sparseMatrix(i = i_idx, j = j_idx, x = x_val,
                       dims = c(offset, n_obs))
}

# Point estimate of the RE coefficient vector (length n_re) for fits without
# RE draws: the tail of the Laplace $mode or the $means vector past the fixed
# block. NULL when neither carries the RE block.
#' @keywords internal
.tulpa_re_point <- function(object, n_re) {
  nf <- object$n_fixed %||% 0L
  for (v in list(object$mode, object$means)) {
    if (is.numeric(v) && length(v) >= nf + n_re) {
      return(as.numeric(v)[(nf + 1L):(nf + n_re)])
    }
  }
  NULL
}

# Linear-predictor posterior draws (S x n_obs). Fixed effects come from the
# fit's draws when it carries any (joint with the RE draws, so a subsample
# keeps rows aligned), otherwise from a Gaussian draw at coef()/vcov() (the
# Laplace / nested-moment approximation). At the training design (newdata =
# NULL) the offset, the formula random effects, and a posterior-mean SPDE
# field are added; at `newdata` the prediction is population-level (fixed
# effects only), matching predict().
#' @keywords internal
.tulpa_eta_draws <- function(object, newdata = NULL, ndraws = NULL) {
  X <- if (is.null(newdata)) object$model_matrix
       else .tulpa_fixed_design(object, newdata)
  if (is.null(X)) {
    stop("posterior_predict() needs the fixed-effect design ($model_matrix); ",
         "refit with tulpa().", call. = FALSE)
  }

  fd   <- .fixed_draws_mat(object)
  keep <- NULL
  if (!is.null(fd)) {
    S <- nrow(fd)
    keep <- if (!is.null(ndraws) && ndraws < S) sample.int(S, ndraws)
            else seq_len(S)
    beta <- fd[keep, , drop = FALSE]
  } else {
    S  <- as.integer(ndraws %||% 400L)
    mu <- coef(object)
    V  <- vcov(object)
    if (anyNA(V)) {
      stop("posterior_predict(): the fixed-effect covariance is unavailable ",
           "(vcov() returned NA).", call. = FALSE)
    }
    L <- tryCatch(chol(V), error = function(e) {
      chol(V + diag(1e-10 * max(diag(V), 1), nrow(V)))
    })
    beta <- matrix(rep(mu, each = S), S) +
      matrix(stats::rnorm(S * length(mu)), S) %*% L
    colnames(beta) <- names(mu)
  }

  nm <- object$fixed_names %||% colnames(beta)
  if (!is.null(nm) && all(nm %in% colnames(X))) {
    X <- X[, nm, drop = FALSE]
  } else if (ncol(X) != ncol(beta)) {
    stop("posterior_predict(): design columns do not match the fixed-effect ",
         "draws.", call. = FALSE)
  }
  eta <- beta %*% t(X)

  if (is.null(newdata)) {
    if (!is.null(object$offset)) {
      eta <- sweep(eta, 2, object$offset, "+")
    }

    M <- .tulpa_re_map(object)
    if (!is.null(M)) {
      rd <- .re_draws_mat(object)
      if (!is.null(rd) && ncol(rd) == nrow(M)) {
        if (!is.null(keep)) rd <- rd[keep, , drop = FALSE]
        eta <- eta + as.matrix(rd %*% M)
      } else {
        b <- .tulpa_re_point(object, nrow(M))
        if (!is.null(b)) {
          eta <- sweep(eta, 2, as.numeric(Matrix::crossprod(M, b)), "+")
        } else {
          message("posterior_predict(): no random-effect draws or point ",
                  "values available; replicates are population-level.")
        }
      }
    }

    if (identical(object$spatial$type, "spde") &&
        !is.null(object$spatial_effects) && !is.null(object$spatial$A)) {
      eta <- sweep(eta, 2,
                   as.numeric(object$spatial$A %*% object$spatial_effects), "+")
    }
  }

  eta
}

#' Posterior predictive replicates
#'
#' @description
#' Draw replicated responses from the posterior predictive distribution: the
#' linear predictor is rebuilt per posterior draw (fixed effects, formula
#' random effects, offset, and a posterior-mean SPDE field when present) and
#' pushed through the family's sampling distribution.
#'
#' Fits carrying posterior draws use them directly (fixed and random effects
#' jointly per draw). The Laplace tier samples the fixed effects from the
#' Gaussian approximation `N(coef(fit), vcov(fit))` and holds the random
#' effects at their posterior mode, so its replicates understate the RE
#' posterior uncertainty. At `newdata` the prediction is population level
#' (random effects at zero), matching [predict.tulpa_fit()].
#'
#' @param object A `tulpa_fit` object from [tulpa()].
#' @param ... Passed to methods.
#' @return A `ndraws x n_obs` numeric matrix of replicated responses.
#' @seealso [pp_check()], which uses these replicates; [simulate.tulpa_fit()].
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(y = rpois(100, 4), x = rnorm(100))
#' fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
#' yrep <- posterior_predict(fit, ndraws = 100)
#' dim(yrep)  # 100 x 100
#' }
#' @export
posterior_predict <- function(object, ...) {
  UseMethod("posterior_predict")
}

#' @param newdata Optional data frame of covariates to predict at. Population
#'   level (fixed effects only); `NULL` (default) replicates at the training
#'   data with random effects and offset included.
#' @param ndraws Number of posterior draws to use. Defaults to all stored
#'   draws, or 400 on the draw-free Laplace tier.
#' @param n_trials Binomial / beta-binomial trial counts for the replicates.
#'   Defaults to the training trials when `newdata` is `NULL`, else 1.
#' @param seed Optional integer seed (RNG state is restored on exit).
#' @rdname posterior_predict
#' @export
posterior_predict.tulpa_fit <- function(object, newdata = NULL, ndraws = NULL,
                                        n_trials = NULL, seed = NULL, ...) {
  if (!is.character(object$family) || length(object$family) != 1L) {
    stop("posterior_predict() supports fits with a builtin character family; ",
         "model packages provide their own methods.", call. = FALSE)
  }
  if (!is.null(seed)) {
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) {
      get(".Random.seed", envir = .GlobalEnv)
    } else NULL
    set.seed(seed)
    on.exit({
      if (is.null(old_seed)) rm(".Random.seed", envir = .GlobalEnv)
      else assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }, add = TRUE)
  }

  eta <- .tulpa_eta_draws(object, newdata = newdata, ndraws = ndraws)
  if (is.null(n_trials)) {
    n_trials <- if (is.null(newdata)) object$n_trials else NULL
  }
  phi <- object$phi %||% 1.0

  yrep <- matrix(NA_real_, nrow(eta), ncol(eta))
  for (s in seq_len(nrow(eta))) {
    yrep[s, ] <- family_sample(eta[s, ], object$family,
                               n_trials = n_trials, phi = phi,
                               phi2 = object$phi2)
  }
  yrep
}

#' Simulate responses from a fitted tulpa model
#'
#' @description
#' Base-R alias for [posterior_predict()]: each simulation is one posterior
#' predictive replicate at the training data.
#'
#' @param object A `tulpa_fit` object.
#' @param nsim Number of simulated datasets (default 1).
#' @param seed Optional integer seed (RNG state is restored on exit).
#' @param ... Ignored.
#' @return A data frame with `nsim` columns (`sim_1`, ...), one row per
#'   observation, following the [stats::simulate()] convention.
#' @export
simulate.tulpa_fit <- function(object, nsim = 1, seed = NULL, ...) {
  yrep <- posterior_predict(object, ndraws = nsim, seed = seed)
  if (nrow(yrep) > nsim) yrep <- yrep[seq_len(nsim), , drop = FALSE]
  out <- as.data.frame(t(yrep))
  names(out) <- paste0("sim_", seq_len(ncol(out)))
  rownames(out) <- NULL
  out
}
