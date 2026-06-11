# Shared `log_marginal(theta)` builder for the tgmrf hyperparameter adapters
# (tulpa_tgmrf(mode = "imh" / "vi" / "nuts")).
#
# All three integrate the same Laplace body for (beta, z) | theta and differ
# only in how they explore theta-space (FD Hessian + IMH, Pathfinder L-BFGS,
# NUTS). The per-theta evaluation is identical: pack theta into a one-row
# grid, run one inner nested-Laplace solve, read its scalar log-marginal. It
# lives here once so the inner-call argument contract has a single source of
# truth. (It used to be copy-pasted into all three; the Phase-2 migration of
# the nested-Laplace knobs into `control = list()` then desynced two copies,
# which silently caught the resulting "unused argument" error and returned
# -Inf -- the init guards in vi/nuts fired with a cryptic "not finite".)
#
# Returns `list(eval, raw)`:
#   * `raw(theta)`  one inner solve, NO error swallowing. A failure here is a
#                   structural bug (e.g. an engine-API drift), not numerical
#                   infeasibility, so it surfaces with its real message. Use
#                   for the eager check at the feasible pilot mode.
#   * `eval(theta)` the optimizer/sampler-facing closure: optional hard-wall
#                   bounds, then `raw` with errors/warnings from an infeasible
#                   theta mapped to -Inf so the caller retreats from the
#                   infeasible region.
#
# `bounds_lo` / `bounds_hi` NULL  -> no wall (local FD use in mode = "imh",
# whose +/- fd_step steps stay near the interior mode).
.tgmrf_make_log_marginal <- function(y, n_trials, X, block, obs_idx,
                                     re_idx, n_re_groups, sigma_re,
                                     family, phi,
                                     max_iter, tol, n_threads,
                                     bounds_lo = NULL, bounds_hi = NULL) {
  raw <- function(theta_vec) {
    blk <- block
    blk$obs_idx <- obs_idx
    blk$theta_grid_built <- matrix(theta_vec, nrow = 1L,
                                   dimnames = list(NULL, block$theta_names))
    out <- tulpa_nested_laplace(
      y = y, n_trials = n_trials, X = X, prior = blk,
      re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
      family = family, phi = phi,
      control = list(max_iter = max_iter, tol = tol, n_threads = n_threads)
    )
    as.numeric(out$log_marginal[1])
  }

  eval <- function(theta_vec) {
    if (!is.null(bounds_lo) &&
        (any(theta_vec < bounds_lo) || any(theta_vec > bounds_hi))) {
      return(-Inf)
    }
    lm <- tryCatch(suppressWarnings(raw(theta_vec)),
                   error = function(e) NA_real_,
                   warning = function(w) NA_real_)
    if (!is.finite(lm)) return(-Inf)
    lm
  }

  list(eval = eval, raw = raw)
}
