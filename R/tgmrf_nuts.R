#' NUTS over a tgmrf block's hyperparameters
#'
#' @description
#' No-U-Turn Sampler (Hoffman & Gelman 2014) over the user-defined
#' hyperparameter vector `theta` of a [tgmrf()] block, with a finite-
#' difference gradient on `log_marginal(theta)`. Same Tier 1
#' (MCMC-corrected) composition as tulpa_tgmrf_imh() but the proposal
#' uses gradient information through leapfrog integration, so mixing
#' degrades more gracefully with `theta_dim` than independence-MH does.
#'
#' Each leapfrog step evaluates the gradient via `2 * theta_dim` extra
#' inner Laplace solves (central FD on `log_marginal`); a NUTS tree of
#' depth `j` integrates `2^j` leapfrog steps. The implementation is the
#' textbook Hoffman & Gelman (2014) Algorithm 3 (NUTS *without* dual
#' averaging): pilot Laplace gives the initial step size and mass-matrix
#' diagonal; warmup performs a simple Robbins-Monro adaptation toward
#' the target acceptance.
#'
#' @inheritParams.tgmrf_fit_imh
#' @param epsilon Initial leapfrog step size. Default is `0.2 / sqrt(theta_dim)`.
#' @param max_depth Maximum NUTS tree depth (`2^max_depth` leapfrog steps
#'   per draw at the limit). Default 6.
#' @param target_accept Target acceptance for the warmup step-size
#'   adaptation. Default 0.65.
#' @param fd_gradient_step Finite-difference step for `d/dtheta_j
#'   log_marginal`. Default 0.02.
#' @param n_iter Total number of NUTS iterations (including warmup). Default
#'   `500L`.
#' @param warmup Number of warmup iterations used for step-size adaptation and
#'   then discarded. Must satisfy `1 <= warmup < n_iter`. Default `n_iter %/% 2L`.
#'
#' @return A list with class `c("tulpa_tgmrf_nuts", "tulpa_fit")`:
#'   * `draws`, `means`, `sds` -- posterior draws of `theta` and moments.
#'   * `mean_accept` -- mean Metropolis acceptance probability across the
#'     post-warmup tree expansion steps.
#'   * `tree_depth` -- depth of the NUTS tree at each iteration.
#'   * `epsilon` -- final step size after warmup adaptation.
#'   * `pilot`, `mode_theta`, `hessian_theta` -- pilot diagnostics (same as
#'     tulpa_tgmrf_imh()).
#'   * `inference_mode`, `inference_tier`, `backend` --
#'     `"exact"`, `1L`, `"tgmrf_nuts"`.
#'
#' @references Hoffman & Gelman (2014). The No-U-Turn Sampler. JMLR
#'   15:1593-1623.
#' @seealso tulpa_tgmrf_imh() for the independence-MH counterpart.
#' @noRd
.tgmrf_fit_nuts <- function(y, n_trials, X, block,
                             family = "binomial",
                             phi = 1.0,
                             re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                             n_iter = 500L, warmup = n_iter %/% 2L,
                             epsilon = NULL,
                             max_depth = 6L,
                             target_accept = 0.65,
                             pilot_axis_points = 5L,
                             fd_gradient_step = 0.02,
                             max_iter = 50L, tol = 1e-7,
                             n_threads = 1L,
                             verbose = FALSE) {

  if (!inherits(block, "tgmrf")) {
    stop("`block` must be a tgmrf object.", call. = FALSE)
  }
  if (max_depth < 1L || max_depth > 12L) {
    stop("`max_depth` must be in [1, 12].", call. = FALSE)
  }
  if (n_iter < 2L || warmup < 1L || warmup >= n_iter) {
    stop("Need 1 <= warmup < n_iter and n_iter >= 2.", call. = FALSE)
  }

  d <- block$theta_dim
  N <- length(y)
  obs_idx <- block$obs_idx %||% seq_len(N)

  # --- Pilot Laplace for init theta, mass matrix, eps ---------------------
  pilot_block <- block
  pilot_block$obs_idx <- obs_idx
  if (!is.null(pilot_axis_points) && pilot_axis_points != 5L) {
    axes <- vector("list", d)
    for (j in seq_len(d)) {
      lo <- if (!is.null(block$bounds)) block$bounds$lower[j] else block$init[j] - 2
      hi <- if (!is.null(block$bounds)) block$bounds$upper[j] else block$init[j] + 2
      axes[[j]] <- seq(lo, hi, length.out = pilot_axis_points)
    }
    names(axes) <- block$theta_names
    pilot_block$theta_grid_built <- as.matrix(do.call(expand.grid, axes))
  }

  pilot <- tulpa_nested_laplace(
    y = y, n_trials = n_trials, X = X,
    prior = pilot_block,
    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
    family = family, phi = phi,
    control = list(max_iter = max_iter, tol = tol, n_threads = n_threads)
  )
  k_star <- which.max(pilot$log_marginal)
  theta_init <- as.numeric(pilot$theta_grid[k_star, ])
  names(theta_init) <- block$theta_names

  # Mass-matrix diagonal: posterior SDs from the grid give the right scale
  # for each axis. Clip away from zero.
  mass_diag <- pmax(as.numeric(pilot$theta_sd)^2, 1e-3)
  M_inv <- mass_diag                       # diagonal inverse mass matrix
  sqrt_M <- 1 / sqrt(M_inv)                # for momentum draws

  if (is.null(epsilon)) epsilon <- 0.2 / sqrt(d)

  # --- Closures: log_post and FD gradient on log_marginal -----------------
  # Shared builder (see .tgmrf_make_log_marginal): hard-wall bounds when the
  # user supplied them -- a leapfrog step that overshoots into the infeasible
  # region returns -Inf and NUTS rejects via the slice variable. This stops
  # the chain from wandering to numerically-divergent theta where Q goes
  # singular and log|Q| has wide swings that the FD gradient can't track.
  bounds_lo <- if (!is.null(block$bounds)) block$bounds$lower else NULL
  bounds_hi <- if (!is.null(block$bounds)) block$bounds$upper else NULL
  .lm <- .tgmrf_make_log_marginal(
    y = y, n_trials = n_trials, X = X, block = block, obs_idx = obs_idx,
    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
    family = family, phi = phi,
    max_iter = max_iter, tol = tol, n_threads = n_threads,
    bounds_lo = bounds_lo, bounds_hi = bounds_hi
  )
  log_marginal_at <- .lm$eval

  grad_log_marginal_at <- function(theta_vec) {
    g <- numeric(d)
    for (j in seq_len(d)) {
      e_j <- numeric(d); e_j[j] <- fd_gradient_step
      lp <- log_marginal_at(theta_vec + e_j)
      lm <- log_marginal_at(theta_vec - e_j)
      g[j] <- (lp - lm) / (2 * fd_gradient_step)
    }
    g
  }

  # --- NUTS components ----------------------------------------------------
  # Leapfrog step. r is momentum, M_inv is diag inverse mass; the
  # Hamiltonian is H = -log_post + 0.5 * r' M_inv r.
  leapfrog <- function(theta, r, grad, eps, dir) {
    r_half <- r + 0.5 * eps * dir * grad
    theta_new <- theta + eps * dir * (M_inv * r_half)
    grad_new <- grad_log_marginal_at(theta_new)
    r_new <- r_half + 0.5 * eps * dir * grad_new
    list(theta = theta_new, r = r_new, grad = grad_new,
         log_post = log_marginal_at(theta_new))
  }

  # Hamiltonian.
  H_at <- function(log_post, r) -log_post + 0.5 * sum(M_inv * r * r)

  # U-turn condition (Hoffman-Gelman eqn 9).
  no_uturn <- function(theta_minus, theta_plus, r_minus, r_plus) {
    diff <- theta_plus - theta_minus
    (sum(diff * (M_inv * r_minus)) >= 0) &&
      (sum(diff * (M_inv * r_plus)) >= 0)
  }

  # Recursive tree-building (Hoffman-Gelman Algorithm 3).
  build_tree <- function(theta, r, grad, log_post, log_u, dir, depth, eps, H0) {
    if (depth == 0L) {
      step <- leapfrog(theta, r, grad, eps, dir)
      Hp <- H_at(step$log_post, step$r)
      # Divergent or non-finite leapfrog: treat as a dead branch. n_prime
      # = 0, s_prime = FALSE -- the outer loop swallows the contribution.
      if (!is.finite(step$log_post) || !is.finite(Hp) ||
          any(!is.finite(step$grad))) {
        return(list(
          theta_minus = theta, r_minus = r, grad_minus = grad,
          theta_plus  = theta, r_plus  = r, grad_plus  = grad,
          log_post_minus = log_post, log_post_plus = log_post,
          theta_prime = theta, log_post_prime = log_post,
          n_prime = 0L, s_prime = FALSE,
          alpha = 0, n_alpha = 1L
        ))
      }
      log_w <- -Hp                          # slice weight = exp(-H')
      n_prime <- as.integer(log_u <= log_w)
      s_prime <- log_u < (1000 - Hp)        # divergent if exp(-H'+H0) >> 1
      list(
        theta_minus = step$theta, r_minus = step$r, grad_minus = step$grad,
        theta_plus  = step$theta, r_plus  = step$r, grad_plus  = step$grad,
        log_post_minus = step$log_post, log_post_plus = step$log_post,
        theta_prime = step$theta, log_post_prime = step$log_post,
        n_prime = n_prime, s_prime = isTRUE(s_prime),
        alpha = min(1, exp(H0 - Hp)), n_alpha = 1L
      )
    } else {
      left <- build_tree(theta, r, grad, log_post, log_u, dir, depth - 1L, eps, H0)
      if (!left$s_prime) return(left)
      if (dir == -1L) {
        right <- build_tree(left$theta_minus, left$r_minus, left$grad_minus,
                            left$log_post_minus, log_u, dir, depth - 1L, eps, H0)
        theta_minus <- right$theta_minus; r_minus <- right$r_minus
        grad_minus  <- right$grad_minus;  log_post_minus <- right$log_post_minus
        theta_plus  <- left$theta_plus;   r_plus  <- left$r_plus
        grad_plus   <- left$grad_plus;    log_post_plus  <- left$log_post_plus
      } else {
        right <- build_tree(left$theta_plus, left$r_plus, left$grad_plus,
                            left$log_post_plus, log_u, dir, depth - 1L, eps, H0)
        theta_minus <- left$theta_minus;  r_minus <- left$r_minus
        grad_minus  <- left$grad_minus;   log_post_minus <- left$log_post_minus
        theta_plus  <- right$theta_plus;  r_plus  <- right$r_plus
        grad_plus   <- right$grad_plus;   log_post_plus  <- right$log_post_plus
      }
      n_combined <- left$n_prime + right$n_prime
      if (n_combined > 0 &&
          stats::runif(1L) < right$n_prime / n_combined) {
        theta_prime <- right$theta_prime
        log_post_prime <- right$log_post_prime
      } else {
        theta_prime <- left$theta_prime
        log_post_prime <- left$log_post_prime
      }
      s_combined <- right$s_prime &&
        no_uturn(theta_minus, theta_plus, r_minus, r_plus)
      alpha_combined <- left$alpha + right$alpha
      n_alpha_combined <- left$n_alpha + right$n_alpha
      list(
        theta_minus = theta_minus, r_minus = r_minus, grad_minus = grad_minus,
        theta_plus  = theta_plus,  r_plus  = r_plus,  grad_plus  = grad_plus,
        log_post_minus = log_post_minus, log_post_plus = log_post_plus,
        theta_prime = theta_prime, log_post_prime = log_post_prime,
        n_prime = n_combined, s_prime = s_combined,
        alpha = alpha_combined, n_alpha = n_alpha_combined
      )
    }
  }

  # --- Main loop ----------------------------------------------------------
  theta_curr <- theta_init
  # Eager structural check at the feasible pilot mode (non-swallowing): a
  # failure here is a bug, not numerical infeasibility.
  log_post_curr <- .lm$raw(theta_curr)
  if (!is.finite(log_post_curr)) {
    stop("Pilot log_marginal at the grid argmax is not finite.", call. = FALSE)
  }
  grad_curr  <- grad_log_marginal_at(theta_curr)

  draws_all  <- matrix(NA_real_, nrow = n_iter, ncol = d)
  tree_depth <- integer(n_iter)
  alpha_all  <- numeric(n_iter)

  log_eps <- log(epsilon)

  for (t in seq_len(n_iter)) {
    r0 <- rnorm(d) * sqrt_M
    H0 <- H_at(log_post_curr, r0)
    log_u <- -H0 + log(stats::runif(1L))

    theta_minus <- theta_plus <- theta_curr
    r_minus <- r_plus <- r0
    grad_minus <- grad_plus <- grad_curr
    log_post_minus <- log_post_plus <- log_post_curr
    theta_prime <- theta_curr
    log_post_prime <- log_post_curr

    j <- 0L
    n <- 1L
    s <- TRUE
    alpha_total <- 0
    n_alpha_total <- 0L

    while (s && j < max_depth) {
      dir <- if (stats::runif(1L) < 0.5) -1L else 1L
      if (dir == -1L) {
        bt <- build_tree(theta_minus, r_minus, grad_minus, log_post_minus,
                         log_u, dir, j, exp(log_eps), H0)
        theta_minus <- bt$theta_minus; r_minus <- bt$r_minus
        grad_minus <- bt$grad_minus;   log_post_minus <- bt$log_post_minus
      } else {
        bt <- build_tree(theta_plus, r_plus, grad_plus, log_post_plus,
                         log_u, dir, j, exp(log_eps), H0)
        theta_plus <- bt$theta_plus; r_plus <- bt$r_plus
        grad_plus <- bt$grad_plus;   log_post_plus <- bt$log_post_plus
      }
      if (isTRUE(bt$s_prime) && bt$n_prime > 0 &&
          stats::runif(1L) < bt$n_prime / max(1L, n)) {
        theta_prime <- bt$theta_prime
        log_post_prime <- bt$log_post_prime
      }
      n <- n + bt$n_prime
      alpha_total <- alpha_total + bt$alpha
      n_alpha_total <- n_alpha_total + bt$n_alpha
      s <- isTRUE(bt$s_prime) &&
        no_uturn(theta_minus, theta_plus, r_minus, r_plus)
      j <- j + 1L
    }

    theta_curr <- theta_prime
    log_post_curr <- log_post_prime
    grad_curr <- grad_log_marginal_at(theta_curr)

    draws_all[t, ] <- theta_curr
    tree_depth[t] <- j
    alpha_all[t]  <- if (n_alpha_total > 0L) alpha_total / n_alpha_total else 0

    # Simple Robbins-Monro step-size adaptation during warmup. Clamped
    # to [-3, 1] in log to keep the leapfrog from blowing up when the
    # adapter overshoots on a couple of high-acceptance early draws --
    # the FD-gradient + boundary-wall combination is much more sensitive
    # than a smooth-target NUTS and large eps gives meaningless steps.
    if (t <= warmup) {
      log_eps <- log_eps + (alpha_all[t] - target_accept) / sqrt(t + 10)
      log_eps <- min(max(log_eps, -3), 1)
    }
  }

  keep <- seq.int(warmup + 1L, n_iter)
  draws <- draws_all[keep, , drop = FALSE]
  colnames(draws) <- block$theta_names

  fit <- list(
    draws         = draws,
    means         = colMeans(draws),
    sds           = apply(draws, 2L, stats::sd),
    n_params      = d,
    n_samples     = nrow(draws),
    mean_accept   = mean(alpha_all[keep]),
    tree_depth    = tree_depth[keep],
    epsilon       = exp(log_eps),
    pilot         = pilot,
    mode_theta    = theta_init,
    mass_diag     = mass_diag,
    inference_mode = "exact",
    inference_tier = 1L,
    backend        = "tgmrf_nuts"
  )
  .finalize_fit(fit, draws_kind = "chain", extra_class = "tulpa_tgmrf")
}
