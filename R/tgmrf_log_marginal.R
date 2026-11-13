# Laplace marginal, joint Gaussian proposal, and fixed-effect conditionals
# from the same per-theta inner solve. The marginal includes the theta prior.
.tgmrf_make_log_marginal <- function(y, n_trials, X, block, obs_idx,
                                     re_idx, n_re_groups, sigma_re,
                                     family, phi,
                                     max_iter, tol, n_threads) {
  n_solves <- 0L
  blk <- block
  blk$obs_idx <- obs_idx
  nt <- as.integer(n_trials %||% 1L)
  if (length(nt) == 1L) nt <- rep(nt, length(y))
  ri <- .validate_re_idx(re_idx %||% rep(0L, length(y)), n_re_groups,
                         length(y), "tulpa_tgmrf")
  cargs <- .nl_kernel_args(y, nt, X, NULL, ri, n_re_groups,
                           sigma_re, family, phi, max_iter, tol, n_threads,
                           fitted_var = FALSE)
  solve_grid <- function(theta_mat, keep_fixed) {
    theta_mat <- matrix(theta_mat,
                                   ncol = length(block$theta_names),
                                   dimnames = list(NULL, block$theta_names))
    args <- cargs
    args$store_Q <- keep_fixed
    n_solves <<- n_solves + nrow(theta_mat)
    .nl_dispatch_multi(args, list(blk), theta_grid_override = theta_mat)
  }

  raw <- function(theta_vec) {
    out <- solve_grid(matrix(theta_vec, nrow = 1L), keep_fixed = FALSE)
    as.numeric(out$log_marginal[1])
  }

  eval <- function(theta_vec) {
    lm <- tryCatch(suppressWarnings(raw(theta_vec)),
                   error = function(e) NA_real_,
                   warning = function(w) NA_real_)
    if (!is.finite(lm)) return(-Inf)
    lm
  }

  fixed <- function(theta_mat) {
    theta_mat <- matrix(theta_mat, ncol = length(block$theta_names))
    out <- solve_grid(theta_mat, keep_fixed = TRUE)
    p <- ncol(X)
    out <- .nl_attach_grid_hessians(out, p)
    grid <- matrix(out$theta_grid, ncol = ncol(theta_mat))
    if (nrow(grid) != nrow(theta_mat) ||
        !isTRUE(all.equal(unname(grid), unname(theta_mat))) ||
        length(out$grid_modes) != nrow(theta_mat) ||
        length(out$grid_hessians) != nrow(theta_mat)) {
      stop("The fixed-effect solve did not return one cell per theta draw.",
           call. = FALSE)
    }
    mean <- t(vapply(out$grid_modes, function(m) as.numeric(m)[seq_len(p)],
                     numeric(p)))
    if (p == 1L) mean <- matrix(mean, ncol = 1L)
    cov <- lapply(out$grid_hessians, .nested_cell_fixed_cov, p = p)
    list(mean = mean, cov = cov)
  }

  conditional <- function(theta) {
    out <- solve_grid(matrix(theta, nrow = 1L), keep_fixed = TRUE)
    H <- .nl_modal_joint_precision(out)
    if (is.null(H) || !is.finite(out$log_marginal[1L])) {
      stop("No finite joint Laplace proposal at theta.", call. = FALSE)
    }
    list(mean = as.numeric(out$modes[1L, ]), precision = H,
         log_marginal = as.numeric(out$log_marginal[1L]))
  }
  list(eval = eval, raw = raw, fixed = fixed, conditional = conditional,
       inner_solves = function() n_solves)
}


# Fixed-effect column names of a tgmrf fit, shared by every adapter.
.tgmrf_beta_names <- function(X) {
  nm <- colnames(X)
  if (is.null(nm)) nm <- paste0("beta_", seq_len(ncol(X)))
  nm
}


# Joint (beta, theta) draws for the adapters that sample theta alone. The
# fixed effects are Gaussian given theta in the Laplace body, so a draw of the
# marginal posterior of beta is one conditional draw beta_s ~ N(mean(theta_s),
# cov(theta_s)) per theta draw: the mixture over the sampled theta. Each
# distinct theta row is solved once (an IMH chain repeats its rejected states).
.tgmrf_joint_draws <- function(theta_draws, lm, X) {
  p <- ncol(X)
  key <- apply(theta_draws, 1L, function(r) paste(format(r, digits = 17L),
                                                   collapse = "|"))
  first <- !duplicated(key)
  cond <- lm$fixed(theta_draws[first, , drop = FALSE])
  row_of <- match(key, key[first])
  beta <- matrix(NA_real_, nrow(theta_draws), p)
  for (u in seq_len(sum(first))) {
    rows <- which(row_of == u)
    V <- cond$cov[[u]]
    R <- tryCatch(chol(V), error = function(e) NULL)
    if (is.null(R)) {
      stop("The conditional fixed-effect covariance at a theta draw is not ",
           "positive definite.", call. = FALSE)
    }
    Z <- matrix(stats::rnorm(length(rows) * p), length(rows), p)
    beta[rows, ] <- sweep(Z %*% R, 2L, cond$mean[u, ], `+`)
  }
  colnames(beta) <- .tgmrf_beta_names(X)
  beta
}


# Lays the joint draws onto an adapter's fit: fixed effects first, as every
# chain producer reports them to the generic accessors, with the per-block
# matrices and moments beside them (the shape mode = "nuts_joint" carries).
.tgmrf_with_fixed <- function(fit, theta_draws, lm, X) {
  beta_draws <- .tgmrf_joint_draws(theta_draws, lm, X)
  draws <- cbind(beta_draws, theta_draws)
  fit$draws       <- draws
  fit$means       <- colMeans(draws)
  fit$sds         <- apply(draws, 2L, stats::sd)
  fit$param_names <- colnames(draws)
  fit$n_fixed     <- ncol(beta_draws)
  fit$fixed_names <- colnames(beta_draws)
  fit$n_params    <- ncol(draws)
  fit$n_samples   <- nrow(draws)
  fit$draws_beta  <- beta_draws
  fit$draws_theta <- theta_draws
  fit$means_beta  <- colMeans(beta_draws)
  fit$means_theta <- colMeans(theta_draws)
  fit$sds_beta    <- apply(beta_draws, 2L, stats::sd)
  fit$sds_theta   <- apply(theta_draws, 2L, stats::sd)
  fit
}
