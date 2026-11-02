# Long-format latent-factor door of tulpa(): `tulpa(y ~ x, data, family,
# factors = latent_factor(n_factors, unit =, response =))`.
#
# Each level of the `response` column is one arm of the joint nested-Laplace
# driver, with its own fixed effects from the formula, and the `n_factors`
# factors are the driver's `lf` blocks over the levels of the `unit` column:
#
#   eta_i = X_i beta_{s(i)} + sum_f lambda_{s(i), f} u_{l(i), f},
#   u_{l, f} ~ N(0, 1),
#
# with s(i) the response level and l(i) the unit of row i. The f-th factor
# does not load on the first f - 1 response levels and loads positively on the
# f-th, so the loading matrix is lower-triangular with a positive diagonal; the
# inner solve integrates the scores out unit by unit.

.LF_FIT_MODES <- c("auto", "laplace", "nested_laplace")

# The loading prior SD the lf block reads, from a latent_factor() spec. The
# block's loading prior is N(0, sigma_lambda^2), folded onto (0, Inf) on the
# diagonal, so a normal prior centred at zero is the one prior it can carry.
.lf_sigma_lambda <- function(prior) {
  if (is.null(prior)) return(1.0)
  if (identical(prior$distribution, "normal") && isTRUE(prior$mean == 0)) {
    return(as.numeric(prior$sd))
  }
  stop("`factors$prior` is the prior of the factor loadings; the latent-factor ",
       "block of tulpa() carries a normal prior centred at zero on every ",
       "loading (folded onto the positive half-line on the diagonal), so it ",
       "takes `prior_normal(0, sd)` or NULL (sd = 1). Got a '",
       prior$distribution, "' prior.", call. = FALSE)
}

# Arguments of tulpa() the latent-factor door does not read, each refused with
# the reason rather than dropped.
.lf_check_frontdoor_args <- function(parsed, bundle, mode, args) {
  if (!tolower(mode %||% "auto") %in% .LF_FIT_MODES) {
    stop("A `factors =` model is fit by nested Laplace with the factor scores ",
         "integrated out; `mode` must be one of ",
         paste0("'", .LF_FIT_MODES, "'", collapse = ", "),
         " (got '", mode, "').", call. = FALSE)
  }
  terms_used <- c(
    "random-effect terms" = parsed$n_re_terms > 0L,
    "latent() blocks" = (parsed$n_latent_blocks %||% 0L) > 0L,
    "smooth terms" = (parsed$n_smooth_terms %||% 0L) > 0L,
    "inline spatial() fields" = (parsed$n_spatial_field_blocks %||% 0L) > 0L,
    "inline temporal() fields" = (parsed$n_temporal_field_blocks %||% 0L) > 0L,
    "spatial() terms" = !is.null(parsed$spatial_var),
    "temporal() terms" = !is.null(parsed$temporal_var),
    "an offset" = !is.null(bundle$offset) && any(bundle$offset != 0))
  if (any(terms_used)) {
    stop("A `factors =` model takes fixed effects in `formula` (one set per ",
         "level of the response column); the formula carries ",
         paste(names(terms_used)[terms_used], collapse = ", "),
         ", which the latent-factor arms do not carry.", call. = FALSE)
  }
  set <- vapply(args, function(a) !is.null(a) && !isFALSE(a), logical(1))
  if (any(set)) {
    stop("A `factors =` model does not read ",
         paste0("`", names(args)[set], "`", collapse = ", "),
         ": its arms carry fixed effects, the factors and a dispersion held at ",
         "`phi`.", call. = FALSE)
  }
  invisible(TRUE)
}

# The column of `data` a latent_factor() spec names, as a factor with its
# unused levels dropped.
.lf_column <- function(data, col, what) {
  if (!is.character(col) || length(col) != 1L || is.na(col)) {
    stop("`factors$", what, "` must name one column of `data` (use ",
         "latent_factor(", what, " = \"<column>\")).", call. = FALSE)
  }
  if (!col %in% names(data)) {
    stop("`factors$", what, "` names column '", col, "', which `data` does ",
         "not have.", call. = FALSE)
  }
  v <- data[[col]]
  if (anyNA(v)) {
    stop("Column '", col, "' (`factors$", what, "`) has missing values.",
         call. = FALSE)
  }
  droplevels(as.factor(v))
}

# Block design of the stacked fixed effects: row i carries X_i in the columns
# of its response level's coefficients and zero elsewhere, so `X_block %*% beta`
# is each row's fixed-effect predictor.
.lf_block_design <- function(X, resp_idx, cols_per_level, n_beta) {
  B <- matrix(0, nrow(X), n_beta)
  for (k in seq_along(cols_per_level)) {
    rows <- which(resp_idx == k)
    cc <- cols_per_level[[k]]
    if (length(rows)) B[rows, cc$block] <- X[rows, cc$x, drop = FALSE]
  }
  B
}

# The stacked fixed-effect design of a latent-factor fit at `newdata`: the
# formula's design `X`, placed by each row's response level.
.lf_newdata_design <- function(object, X, newdata) {
  col <- object$factor_spec$response
  if (!col %in% names(newdata)) {
    stop("newdata needs the response-level column '", col, "' of the ",
         "latent-factor model.", call. = FALSE)
  }
  levs <- object$factor_levels$response
  resp_idx <- match(as.character(newdata[[col]]), levs)
  if (anyNA(resp_idx)) {
    bad <- unique(as.character(newdata[[col]])[is.na(resp_idx)])
    stop("newdata column '", col, "' holds level(s) the model was not fitted ",
         "to: ", paste(bad, collapse = ", "), ".", call. = FALSE)
  }
  miss <- setdiff(unlist(lapply(object$coef_cols, `[[`, "x")), colnames(X))
  if (length(miss)) {
    stop("newdata cannot reproduce fixed-effect column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  B <- .lf_block_design(X, resp_idx, object$coef_cols, length(object$fixed_names))
  colnames(B) <- object$fixed_names
  attr(B, "offset") <- attr(X, "offset")
  B
}

# Front-door fit for `tulpa(..., factors = latent_factor(...))`.
.tulpa_fit_factors <- function(factors, parsed, bundle, data, family, phi,
                               n_trials, control, formula, call) {
  if (!inherits(factors, "tulpa_latent")) {
    stop("`factors` must be a latent_factor() specification.", call. = FALSE)
  }
  if (!isTRUE(factors$shared)) {
    stop("`factors$shared = FALSE` loads the factors on the first process of ",
         "a two-process model; in a `factors =` model every response level ",
         "carries its own loading, so the factors reach every level.",
         call. = FALSE)
  }
  if (!is.null(factors$constraint)) {
    stop("`factors$constraint` pins the factor scores of a sampled model; the ",
         "scores of a `factors =` model carry a proper N(0, 1) prior and each ",
         "response level's intercept absorbs their level, so nothing is ",
         "pinned. Leave `constraint` NULL.", call. = FALSE)
  }
  resp_f <- .lf_column(data, factors$response, "response")
  unit_f <- .lf_column(data, factors$unit, "unit")
  levs   <- levels(resp_f)
  K      <- length(levs)
  F_     <- factors$n_factors
  S      <- nlevels(unit_f)
  if (K < 2L) {
    stop("A `factors =` model needs at least two response levels (column '",
         factors$response, "' has ", K, ").", call. = FALSE)
  }
  if (F_ > K) {
    stop(F_, " factors for ", K, " response levels: factor f does not load ",
         "on the first f - 1 levels, so there are at most as many factors as ",
         "levels.", call. = FALSE)
  }
  if (S < 2L) {
    stop("A `factors =` model needs at least two units (column '",
         factors$unit, "' has ", S, ").", call. = FALSE)
  }
  sigma_lambda <- .lf_sigma_lambda(factors$prior)

  N <- bundle$n_obs
  X <- bundle$X
  y <- bundle$y
  nt <- n_trials %||% rep(1L, N)
  resp_idx <- as.integer(resp_f)
  unit_idx <- as.integer(unit_f)
  rows <- lapply(seq_len(K), function(k) which(resp_idx == k))

  # Per-level design: a column constant or aliased within one level's rows
  # (a covariate that does not vary there) is not identified for that level.
  X_arm <- lapply(rows, function(r) {
    Xk <- X[r, , drop = FALSE]
    keep <- !vapply(seq_len(ncol(Xk)), function(j) {
      colnames(Xk)[j] != "(Intercept)" && length(unique(Xk[, j])) < 2L
    }, logical(1))
    .drop_aliased_fixed(Xk[, keep, drop = FALSE])
  })
  p_arm <- vapply(X_arm, ncol, integer(1))
  ends <- cumsum(p_arm)
  cols_per_level <- lapply(seq_len(K), function(k) list(
    x = colnames(X_arm[[k]]),
    block = (ends[k] - p_arm[k]) + seq_len(p_arm[k])))
  fixed_names <- unlist(lapply(seq_len(K), function(k)
    paste0(levs[k], ":", colnames(X_arm[[k]]))), use.names = FALSE)

  arms <- lapply(seq_len(K), function(k) list(
    y = y[rows[[k]]], n_trials = as.integer(nt[rows[[k]]]), X = X_arm[[k]],
    family = family, phi = phi))
  names(arms) <- make.names(levs, unique = TRUE)
  obs_idx <- lapply(rows, function(r) unit_idx[r])
  blk <- list(type = "lf", n_latent = S, obs_idx = obs_idx,
              sigma_lambda = sigma_lambda)

  ctrl_in <- control %||% list()
  n_draws <- as.integer(ctrl_in$n_draws %||% 1000L)
  ctrl <- .control_subset(ctrl_in, .CONTROL_KEYS$nested_laplace_joint,
                          "nested_laplace_joint", reads = "n_draws")
  ctrl$store_Q <- TRUE
  ctrl$progress <- ctrl$progress %||% FALSE

  jfit <- tulpa_nested_laplace_joint(responses = arms,
                                     prior = rep(list(blk), F_),
                                     control = ctrl)

  al <- jfit$arm_layout
  beta_cols <- unlist(lapply(seq_len(K), function(k)
    al$beta_start[k] + seq_len(al$p[k])), use.names = FALSE)
  score_cols <- lapply(seq_len(F_), function(f) al$block_start[f] + seq_len(S))
  load_cols <- lapply(seq_len(F_), function(f)
    al$block_start[f] + S + seq_len(K - f + 1L))

  dr <- tulpa_posterior_draws(jfit, n = n_draws)
  beta_draws <- dr[, beta_cols, drop = FALSE]
  colnames(beta_draws) <- fixed_names
  lam_draws <- array(0, c(nrow(dr), K, F_),
                     dimnames = list(NULL, levs, paste0("factor", seq_len(F_))))
  for (f in seq_len(F_)) lam_draws[, f:K, f] <- dr[, load_cols[[f]]]
  score_draws <- array(0, c(nrow(dr), S, F_),
                       dimnames = list(NULL, levels(unit_f),
                                       paste0("factor", seq_len(F_))))
  for (f in seq_len(F_)) score_draws[, , f] <- dr[, score_cols[[f]]]

  # Lambda Lambda' per draw, then averaged: the factors' covariance between
  # response levels on the link scale, invariant to a rotation of the factors.
  factor_cov <- Reduce(`+`, lapply(seq_len(nrow(dr)), function(s)
    tcrossprod(matrix(lam_draws[s, , ], K, F_)))) / nrow(dr)
  dimnames(factor_cov) <- list(levs, levs)

  # Linear predictor at each outer cell's mode, in the rows of `data`.
  X_block <- .lf_block_design(X, resp_idx, cols_per_level, ends[K])
  fitted_eta <- t(vapply(seq_len(nrow(jfit$modes)), function(g) {
    m <- jfit$modes[g, ]
    eta <- as.numeric(X_block %*% m[beta_cols])
    for (f in seq_len(F_)) {
      lam <- numeric(K)
      lam[f:K] <- m[load_cols[[f]]]
      eta <- eta + lam[resp_idx] * m[score_cols[[f]]][unit_idx]
    }
    eta
  }, numeric(N)))

  colnames(X_block) <- fixed_names
  fit <- jfit
  fit$y <- y
  fit$n_trials <- nt
  fit$phi <- phi
  fit$family <- family
  fit$formula <- formula
  fit$call <- call
  fit$N <- N
  fit$draws <- beta_draws
  fit$draws_kind <- "iid"
  fit$means <- colMeans(beta_draws)
  fit$param_names <- fixed_names
  fit$n_fixed <- length(fixed_names)
  fit$fixed_names <- fixed_names
  fit$model_matrix <- X_block
  fit$fitted_eta <- fitted_eta
  fit$fitted_eta_var <- NULL
  fit$inference_mode <- "laplace"
  fit$inference_tier <- 2L
  fit$backend <- "latent_factor_nested_laplace"
  fit$selection_reason <- paste(
    "factors = latent_factor(); joint nested Laplace with one arm per",
    "response level and the factor scores integrated out per unit")
  fit$loadings <- apply(lam_draws, c(2, 3), mean)
  fit$loadings_draws <- lam_draws
  fit$factor_cov <- factor_cov
  fit$factor_spec <- factors
  fit$factor_levels <- list(response = levs, unit = levels(unit_f))
  fit$factor_index <- list(response = resp_idx, unit = unit_idx)
  fit$coef_cols <- cols_per_level
  fit$.internal$latent_info <- list(type = "latent", n_factors = F_, n_obs = S)
  fit$.internal$latent_draws <- score_draws
  class(fit) <- c("tulpa_factor_fit", "tulpa_fit", oldClass(jfit))
  fit
}


#' @rdname latent_factors
#' @export
latent_factors.tulpa_factor_fit <- function(fit, summary = TRUE,
                                            probs = c(0.025, 0.5, 0.975), ...) {
  scores <- fit$.internal$latent_draws
  n_unit <- dim(scores)[2]
  n_fac  <- dim(scores)[3]
  draws <- matrix(scores, nrow = dim(scores)[1])
  colnames(draws) <- sprintf("factor[%d,%d]", rep(seq_len(n_unit), n_fac),
                             rep(seq_len(n_fac), each = n_unit))
  out <- latent_factors.default(
    list(draws = draws, .internal = list(latent_info = fit$.internal$latent_info)),
    summary = summary, probs = probs)
  if (summary) {
    out <- cbind(unit = fit$factor_levels$unit[out$observation], out,
                 stringsAsFactors = FALSE)
  }
  out
}


#' @rdname tulpa
#' @param x A `tulpa_factor_fit`, the fit `tulpa()` returns for a `factors =`
#'   model.
#' @export
print.tulpa_factor_fit <- function(x, ...) {
  F_ <- ncol(x$loadings)
  cat("tulpa fit: latent-factor model (", F_, " factor",
      if (F_ > 1L) "s", " over ", length(x$factor_levels$unit), " units, ",
      length(x$factor_levels$response), " response levels)\n", sep = "")
  cat("  N =", x$N, " family =", x$family, " fixed effects =",
      length(x$means), "\n\n")
  cat("Loadings (posterior mean):\n")
  print(round(x$loadings, 3))
  cat("\nFixed effects (posterior mean):\n")
  print(round(x$means, 4))
  invisible(x)
}
