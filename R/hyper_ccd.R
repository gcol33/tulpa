#' Mode-centred central composite design over outer hyperparameters
#'
#' Places the nodes of a central composite design (CCD) at the mode of an outer
#' log-marginal and scales them by its curvature, the outer integration INLA
#' uses (Rue, Martino and Chopin 2009). `eval_logm` is the caller's inner solve:
#' the log marginal at one physical hyperparameter vector. Each axis is
#' integrated in an unconstrained coordinate with a flat prior there, the log
#' of a positive scale or the value of a bounded parameter, and its physical
#' box bounds the mode-find and clamps the nodes. The design reuses
#' [ccd_grid()], [ccd_to_theta()] and [ccd_weights()], and the outer PSIS
#' k-hat of the Gaussian proposal comes from [tulpa_batched_pareto_k()].
#'
#' The design declines (returning `ok = FALSE` with the reason) when there are
#' fewer than two axes (a one-dimensional grid is cheaper and as accurate), when
#' the mode-find fails or lands on the box, or when the outer curvature is not
#' positive definite or is ill-conditioned (a weakly identified axis); the
#' caller then integrates on its tensor grid.
#'
#' @param eval_logm `function(theta)` returning the log marginal (a scalar) at
#'   the physical hyperparameter vector `theta` (axes in `axes` order), or a
#'   non-finite value where it is not defined.
#' @param axes List of axes, each `list(name, log_scale, lower, upper, start)`:
#'   `log_scale = TRUE` integrates `log(theta)`; `lower` / `upper` is the
#'   physical box; `start` the physical starting value of the mode-find.
#' @param f0_mult Multiplier on the design radius `sqrt(k)`.
#' @param diagnose_k Compute the outer PSIS k-hat.
#' @param k_samples Proposal draws for the k-hat.
#' @param max_modefind Iteration cap of the bounded mode-find (each step is one
#'   inner solve per coordinate).
#' @return A list with `ok`; when `TRUE` also `nodes` (`n_node x k` physical
#'   values, columns named), `dnode` (design weights), `u_hat` and `L` (mode
#'   and Cholesky factor of the outer covariance in the unconstrained
#'   coordinates), `kind` (per-node label), `n_points`, `pareto_k`; when
#'   `FALSE`, `reason`.
#' @seealso [tulpa_hyper_grid()] (`design = "ccd"`), which builds the axes from
#'   its axis specs.
#' @examples
#' # A bivariate Gaussian log marginal in (log tau, rho).
#' f <- function(th) -0.5 * ((log(th[1]) - 1)^2 / 0.04 + (th[2] - 0.4)^2 / 0.01)
#' axes <- list(list(name = "tau", log_scale = TRUE, lower = 0.3, upper = 30,
#'                   start = 3),
#'              list(name = "rho", log_scale = FALSE, lower = 0.05,
#'                   upper = 0.95, start = 0.5))
#' cc <- tulpa_ccd_outer_grid(f, axes, diagnose_k = FALSE)
#' cc$nodes
#' @export
tulpa_ccd_outer_grid <- function(eval_logm, axes, f0_mult = 1.1,
                                 diagnose_k = TRUE, k_samples = .nl_diag("ccd_k_samples"),
                                 max_modefind = 40L) {
  decline <- function(why) list(ok = FALSE, reason = why)
  k <- length(axes)
  if (k < 2L) return(decline("single axis: the one-dimensional grid is cheaper"))
  fwd <- function(a, x) if (isTRUE(a$log_scale)) log(x) else x
  inv <- function(a, u) if (isTRUE(a$log_scale)) exp(u) else u
  u0    <- vapply(axes, function(a) fwd(a, a$start), 0.0)
  lower <- vapply(axes, function(a) fwd(a, a$lower), 0.0)
  upper <- vapply(axes, function(a) fwd(a, a$upper), 0.0)
  phys <- function(u) vapply(seq_len(k), function(j) inv(axes[[j]], u[j]), 0.0)
  # Negative outer log-marginal; an undefined node returns a large finite value
  # so L-BFGS-B keeps its box.
  neg <- function(u) {
    lm <- tryCatch(eval_logm(phys(u)), error = function(e) NA_real_)
    if (!is.finite(lm)) 1e10 else -lm
  }
  # Every evaluation is a full inner solve, so the mode-find is capped; a
  # capped interior point is kept, since the design weights correct the
  # Gaussian approximation and the node log marginals carry the integrand.
  op <- tryCatch(
    stats::optim(u0, neg, method = "L-BFGS-B", lower = lower, upper = upper,
                 control = list(factr = 1e7, maxit = as.integer(max_modefind),
                                ndeps = rep(5e-2, k))),
    error = function(e) NULL)
  if (is.null(op) || !op$convergence %in% c(0L, 1L) || any(!is.finite(op$par)))
    return(decline("mode-find failed"))
  if (any(abs(op$par - lower) < 1e-3) || any(abs(op$par - upper) < 1e-3))
    return(decline("mode on the box (weakly identified axis)"))
  u_hat <- op$par
  # The Hessian of the negative log marginal at the mode is the precision of
  # the Gaussian approximation to p(theta | y).
  H <- tryCatch(stats::optimHess(u_hat, neg), error = function(e) NULL)
  if (is.null(H) || any(!is.finite(H))) return(decline("non-finite outer Hessian"))
  H <- 0.5 * (H + t(H))
  ev <- tryCatch(eigen(H, symmetric = TRUE, only.values = TRUE)$values,
                 error = function(e) NA_real_)
  if (any(!is.finite(ev)) || max(abs(ev)) <= 0 || min(ev) <= 1e-6 * max(abs(ev)))
    return(decline("ill-conditioned outer curvature"))
  Sigma <- tryCatch(.spd_inverse(H), error = function(e) NULL)
  if (is.null(Sigma)) return(decline("outer Hessian not invertible"))
  L <- tryCatch(t(chol(Sigma)), error = function(e) NULL)
  if (is.null(L)) return(decline("outer covariance not positive definite"))

  ccd    <- ccd_grid(k, f_0 = sqrt(k) * f0_mult)
  dnode  <- ccd_weights(ccd)
  u_grid <- ccd_to_theta(ccd$z, u_hat, L, log_scale = FALSE)
  for (j in seq_len(k)) u_grid[, j] <- pmin(pmax(u_grid[, j], lower[j]), upper[j])
  nodes <- t(apply(u_grid, 1L, phys))
  if (k == 1L) nodes <- t(nodes)
  colnames(nodes) <- vapply(axes, function(a) a$name, "")

  pareto_k <- NA_real_
  if (isTRUE(diagnose_k)) {
    # Proposal N(u_hat, L L') in the unconstrained coordinates, against
    # exp(eval_logm) under the same flat prior the weights use.
    lt <- function(U) vapply(seq_len(nrow(U)), function(i) {
      lm <- tryCatch(eval_logm(phys(U[i, ])), error = function(e) NA_real_)
      if (is.finite(lm)) lm else -Inf
    }, 0.0)
    kd <- tryCatch(tulpa_batched_pareto_k(u_hat, L, lt, as.integer(k_samples)),
                   error = function(e) list(pareto_k = NA_real_))
    pareto_k <- kd$pareto_k
  }
  list(ok = TRUE, nodes = nodes, dnode = dnode, u_hat = u_hat, L = L,
       kind = ccd$kind, n_points = ccd$n_points, pareto_k = pareto_k)
}
