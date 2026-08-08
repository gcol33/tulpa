# =============================================================================
# subspace_debias.R -- the reliability band as the debias SELECTOR
# (gcol33/tulpa#304).
#
# The engine's stated composition is that exact MCMC corrects only the residual
# directions the deterministic approximation is biased in. Before this the
# escalation was whole-fit: tulpa_re_cov_nested -> tulpa_re_cov_gibbs, every
# coordinate paying the sampler's price including the ones the Gaussian already
# fits. Meanwhile the inner-layer diagnostics are PER INDEX -- gamma_3
# (src/inner_laplace_skew.h) and the inner importance k-hat
# (src/inner_laplace_is.h) both score one probed latent coordinate at a time --
# so the map of which directions need exact treatment already exists on the fit.
#
# This file turns that map into a selector. `.subspace_select()` bands every
# probed index, takes S = the misfit set, optionally closes it under strong
# posterior coupling, and hands S to the compiled sampler
# (src/subspace_debias.h), which corrects x_S exactly with x_{-S} carried at its
# Gaussian conditional. `.subspace_node_draws()` puts the sampled coordinates
# back together with the Gaussian ones.
#
# WHAT THE APPROXIMATION IS. Conditioning x_{-S} on the Gaussian is not exact:
# the Gaussian conditional is not the true conditional, so error enters in
# exactly the directions being corrected unless the couplings across the S
# boundary are weak. That is what the closure rule addresses and what the
# recovery gate measures -- see tests/testthat/test-subspace-debias-recovery.R
# for the both-ways numbers, and NEWS.md for the measured coverage and cost
# against the full Gibbs debias.
# =============================================================================

# Rank of a band, for "at or above this floor" comparisons. NA (never assessed)
# ranks below `good`, so an unscored coordinate is never selected on the
# strength of not having been looked at.
.SUBSPACE_BAND_RANK <- c(good = 1L, ok = 2L, unreliable = 3L)

.subspace_band_rank <- function(band) {
  out <- .SUBSPACE_BAND_RANK[band]
  out[is.na(out)] <- 0L
  as.integer(out)
}

# Per-probed-index inner band of a Laplace solve, as the WORSE of the two inner
# scores it carries: the cubic term gamma_3 and the importance k-hat. Mirrors
# `.tulpa_inner_layer()`, which resolves the same pair into one band for the
# whole-fit verdict -- the difference is only that this keeps the resolution per
# index, which is what a selector needs.
#
# `fit` is a `tulpa_laplace()` result solved with `compute_skew = TRUE`. Returns
# a data frame with one row per probed latent index (`idx`, 1-based),
# `gamma3`, `pareto_k`, `rel_ess` and the resolved `band`.
.subspace_bands <- function(fit) {
  idx <- fit[["inner_skew_idx"]]
  if (is.null(idx) || !length(idx)) {
    return(data.frame(idx = integer(0), gamma3 = numeric(0),
                      pareto_k = numeric(0), rel_ess = numeric(0),
                      band = character(0), stringsAsFactors = FALSE))
  }
  idx <- as.integer(idx)
  n <- length(idx)
  g <- as.numeric(fit[["inner_skew"]] %||% rep(NA_real_, n))[seq_len(n)]

  # The per-index inner k-hat, from whichever representation the fit carries.
  # A nested-Laplace fit has it already: `.inner_k_attach()` fitted it at attach
  # time from the probe's importance curve and stored the per-index vectors, so
  # it is read rather than re-fitted. A bare `tulpa_laplace()` result carries
  # only the curve the kernel returned (the Pareto fit is R-side), so there it
  # is fitted here -- from the same raw material, through the same estimator.
  k <- rep(NA_real_, n); rel <- rep(NA_real_, n)
  k_att   <- fit[["inner_pareto_k"]]
  rel_att <- fit[["inner_pareto_k_rel_ess"]]
  if (length(k_att) == n && length(rel_att) == n) {
    k <- as.numeric(k_att); rel <- as.numeric(rel_att)
  } else {
    lj <- fit[["inner_is_log_joint"]]
    z  <- fit[["inner_is_z"]]
    if (!is.null(lj) && !is.null(z) && is.matrix(lj) && nrow(lj) == length(z) &&
        ncol(lj) == n) {
      got <- tryCatch(.inner_k_from_curve(as.numeric(z), lj),
                      error = function(e) NULL)
      if (!is.null(got)) { k <- got$pareto_k; rel <- got$rel_ess }
    }
  }

  band_g <- vapply(g, .tulpa_gamma3_band, character(1))
  # The k-hat is banded only where the importance correction is MATERIAL, the
  # same gate `.tulpa_inner_k_summary()` applies: a scale-free shape index
  # fitted to uniform weights describes residual wiggle, not misfit.
  material <- is.finite(k) & is.finite(rel) & rel < .nl_diag("inner_k_material_ess")
  band_k <- rep(NA_character_, n)
  band_k[material] <- vapply(k[material], .tulpa_khat_band, character(1))

  rank <- pmax(.subspace_band_rank(band_g), .subspace_band_rank(band_k))
  band <- rep(NA_character_, n)
  hit <- rank > 0L
  band[hit] <- names(.SUBSPACE_BAND_RANK)[rank[hit]]

  data.frame(idx = idx, gamma3 = g, pareto_k = k, rel_ess = rel,
             band = band, stringsAsFactors = FALSE)
}

# Grow an index set by the precision-graph neighbours it is strongly coupled to.
#
# A coordinate left OUT of S is carried at its Gaussian conditional mean, which
# is a LINEAR function of x_S. Where that coordinate is strongly coupled to a
# member of S, the linear carry is exactly the approximation the correction is
# trying to remove, and the bias it leaves comes back through the conditional.
# The coupling is read off the joint precision H directly: the partial
# correlation of x_i and x_j given everything else is -H_ij / sqrt(H_ii H_jj),
# so a large one is a strong conditional dependence and a candidate for
# inclusion.
#
# One pass over the neighbours of S (not a transitive closure): a second pass
# grows S toward the whole field on a dense H, which is the full debias wearing
# a different name. `max_add` caps the growth for the same reason; when the cap
# binds, the strongest couplings are the ones kept.
.subspace_closure <- function(H, idx, threshold = .nl_diag("debias_closure_pcor"),
                              max_add = .nl_diag("debias_closure_max")) {
  if (is.null(H) || !length(idx)) return(list(idx = idx, added = integer(0)))
  H <- tryCatch(as.matrix(H), error = function(e) NULL)
  if (is.null(H) || nrow(H) != ncol(H) || any(idx > nrow(H))) {
    return(list(idx = idx, added = integer(0)))
  }
  d <- sqrt(abs(diag(H)))
  d[!is.finite(d) | d <= 0] <- NA_real_
  sub <- H[idx, , drop = FALSE]
  pc <- abs(sub / outer(d[idx], d))
  pc[!is.finite(pc)] <- 0
  strength <- apply(pc, 2L, max)
  strength[idx] <- -Inf                # already in S, never a candidate
  cand <- which(strength >= threshold)
  if (!length(cand)) return(list(idx = idx, added = integer(0)))
  cand <- cand[order(strength[cand], decreasing = TRUE)]
  if (length(cand) > max_add) cand <- cand[seq_len(max_add)]
  list(idx = sort(c(idx, as.integer(cand))), added = sort(as.integer(cand)))
}

# Normalize the `control$subspace_debias` value into a settings list, or NULL
# when the correction is off. `TRUE` takes every default; a list overrides the
# entries it names. `idx` pins the set explicitly and skips the selector
# entirely, which is what makes the both-ways closure comparison possible
# without a second code path.
.subspace_debias_config <- function(spec) {
  if (is.null(spec) || identical(spec, FALSE)) return(NULL)
  if (isTRUE(spec)) spec <- list()
  if (!is.list(spec)) {
    stop("`control$subspace_debias` must be TRUE, FALSE, or a list of ",
         "settings.", call. = FALSE)
  }
  known <- c("idx", "probe", "band", "closure", "closure_max", "n_iter",
             "warmup", "thin", "n_draws")
  unknown <- setdiff(names(spec), known)
  if (length(unknown)) {
    stop(sprintf("Unknown `control$subspace_debias` setting(s): %s.\nAllowed: %s.",
                 paste(sQuote(unknown, q = FALSE), collapse = ", "),
                 paste(known, collapse = ", ")), call. = FALSE)
  }
  band <- spec$band %||% .nl_diag("debias_select_band")
  if (!band %in% names(.SUBSPACE_BAND_RANK)) {
    stop("`control$subspace_debias$band` must be one of ",
         paste(names(.SUBSPACE_BAND_RANK), collapse = ", "), ".", call. = FALSE)
  }
  list(
    idx         = if (is.null(spec$idx)) NULL else as.integer(spec$idx),
    probe       = if (is.null(spec$probe)) NULL else as.integer(spec$probe),
    band        = band,
    closure     = spec$closure %||% FALSE,
    closure_max = as.integer(spec$closure_max %||% .nl_diag("debias_closure_max")),
    n_iter      = as.integer(spec$n_iter %||% .nl_diag("debias_n_iter")),
    warmup      = as.integer(spec$warmup %||% .nl_diag("debias_warmup")),
    thin        = as.integer(spec$thin %||% .nl_diag("debias_thin")),
    n_draws     = as.integer(spec$n_draws %||% .nl_diag("debias_n_draws"))
  )
}

# Select the coordinates to correct from a probe solve.
#
# `probe` is a `tulpa_laplace()` result solved with `compute_skew = TRUE` (and,
# when the closure is requested, `return_joint_hessian = TRUE`). Returns the
# selected 1-based latent indices, the band table they were read from, and the
# closure's additions -- all recorded on the fit, so the escalation is auditable
# rather than implicit.
.subspace_select <- function(probe, cfg) {
  bands <- .subspace_bands(probe)
  if (!is.null(cfg$idx)) {
    return(list(idx = cfg$idx, bands = bands, added = integer(0),
                selected_by = "idx"))
  }
  floor_rank <- .SUBSPACE_BAND_RANK[[cfg$band]]
  sel <- bands$idx[.subspace_band_rank(bands$band) >= floor_rank]
  sel <- sort(as.integer(sel))
  added <- integer(0)
  closure <- cfg$closure
  if (length(sel) && !identical(closure, FALSE)) {
    thr <- if (isTRUE(closure)) .nl_diag("debias_closure_pcor") else as.numeric(closure)
    cl <- .subspace_closure(probe$H_joint, sel, threshold = thr,
                            max_add = cfg$closure_max)
    sel <- cl$idx
    added <- cl$added
  }
  list(idx = sel, bands = bands, added = added, selected_by = "band")
}

# Recombine one node's Metropolis draws with the Gaussian coordinates it did not
# sample.
#
# `mu` is the node's fixed-effect mode, `V` its marginal fixed-effect covariance
# (solve(H_beta)), `pos` the positions within the fixed-effect block that were
# sampled, and `d` the [n_kept x length(pos)] matrix of x_S - mode_S draws for
# those positions. The sampled coordinates come straight from `d`; the rest are
# drawn from the GAUSSIAN CONDITIONAL given them,
#
#   beta_{-S} | beta_S ~ N(mu_{-S} + V_{-S,S} V_SS^{-1} (beta_S - mu_S),
#                          V_{-S,-S} - V_{-S,S} V_SS^{-1} V_{S,-S}),
#
# so a coordinate outside S keeps exactly the Gaussian the plain path gave it,
# now conditioned on a corrected partner instead of an uncorrected one. Returns
# NULL when the conditional cannot be formed, and the caller falls back to the
# plain Gaussian draw for that node rather than reporting a mixture of two
# different constructions without saying so.
.subspace_node_draws <- function(mu, V, pos, d, n_draws) {
  p <- length(mu)
  q <- length(pos)
  if (q == 0L || is.null(d) || !nrow(d)) return(NULL)
  if (q < p && (is.null(V) || any(dim(V) != p))) return(NULL)
  rows <- sample.int(nrow(d), n_draws, replace = TRUE)
  out <- matrix(NA_real_, n_draws, p)
  out[, pos] <- d[rows, , drop = FALSE] +
    rep(mu[pos], each = n_draws)
  rest <- setdiff(seq_len(p), pos)
  if (!length(rest)) return(out)
  Vss <- V[pos, pos, drop = FALSE]
  A <- tryCatch(V[rest, pos, drop = FALSE] %*% solve(Vss), error = function(e) NULL)
  if (is.null(A)) return(NULL)
  Vc <- V[rest, rest, drop = FALSE] - A %*% V[pos, rest, drop = FALSE]
  Lc <- tryCatch(t(chol((Vc + t(Vc)) / 2)), error = function(e) NULL)
  if (is.null(Lc)) return(NULL)
  dm <- d[rows, , drop = FALSE]
  shift <- dm %*% t(A)
  noise <- matrix(stats::rnorm(n_draws * length(rest)), n_draws, length(rest))
  out[, rest] <- rep(mu[rest], each = n_draws) + shift + noise %*% t(Lc)
  out
}

# The kernel-facing form of a debias request: the 1-based index set plus the
# sweep budget, in ONE list. Every nested kernel entry takes it under that name
# (`unwrap_debias` in src/laplace_spec_fit.h), so a backend gains the correction
# by forwarding one argument rather than four. NULL for an absent or empty set,
# which is the C++ side's "never enter the sampler" contract.
.subspace_debias_request <- function(spec) {
  if (is.null(spec)) return(NULL)
  idx <- as.integer(spec$idx)
  if (!length(idx)) return(NULL)
  list(idx    = idx,
       n_iter = as.integer(spec$n_iter %||% .nl_diag("debias_n_iter")),
       warmup = as.integer(spec$warmup %||% .nl_diag("debias_warmup")),
       thin   = as.integer(spec$thin   %||% .nl_diag("debias_thin")))
}

# What a fit records about the escalation, whether or not anything was
# corrected. Written even for an empty selection, so a fit says it looked and
# found nothing rather than being silent about it -- and so a fit that WAS
# corrected can be told from one that was not without re-reading the bands.
.subspace_debias_record <- function(sel, cfg, accept = NA_real_,
                                    declined = NA_character_) {
  list(idx = sel$idx, bands = sel$bands, closure_added = sel$added,
       selected_by = sel$selected_by, band_floor = cfg$band,
       closure = cfg$closure, n_iter = cfg$n_iter, warmup = cfg$warmup,
       thin = cfg$thin, accept = accept, declined = declined)
}

# Rebuild a nested-Laplace grid's fixed-effect posterior as DRAWS, with the
# selected coordinates taken from each cell's Metropolis sample.
#
# The plain grid path reports the fixed effects through moments
# (`.nested_fixed_moments()`: the law of total variance over `grid_modes` /
# `grid_hessians`), which is a Gaussian mixture summary and cannot carry a
# corrected shape. Once a coordinate is sampled the posterior is no longer that
# mixture, so the fit reports draws instead and every coefficient-facing method
# reads them through the accessor it already uses.
#
# Recombination is `.re_cov_nested_beta_draws()` -- the same node mixture the
# RE-covariance backend draws from, given this grid's per-cell mode, covariance
# and weight. A cell whose sampler declined (or that a resume loaded without
# draws) falls back to its Gaussian there, so one unusable cell degrades that
# cell rather than the fit.
.nl_subspace_grid_draws <- function(res, cell_draws, idx, n_draws, beta_names) {
  H <- res$grid_hessians
  M <- res$grid_modes
  w <- res$weights
  if (is.null(H) || is.null(M) || is.null(w)) return(NULL)
  if (length(H) != length(w) || length(M) != length(w)) return(NULL)
  ok <- which(is.finite(w) & w > 0 &
                !vapply(M, is.null, logical(1)) &
                !vapply(H, is.null, logical(1)))
  if (!length(ok)) return(NULL)
  p <- length(M[[ok[1L]]])
  beta_nodes <- matrix(NA_real_, length(w), p)
  cov_nodes  <- vector("list", length(w))
  for (k in ok) {
    mu <- M[[k]]
    if (length(mu) != p) next
    V <- tryCatch(solve(H[[k]]), error = function(e) NULL)
    if (is.null(V) || any(dim(V) != p)) next
    beta_nodes[k, ] <- mu
    cov_nodes[[k]]  <- V
  }
  nodes <- if (is.null(cell_draws)) NULL else {
    out <- vector("list", length(w))
    m <- min(length(cell_draws), length(w))
    if (m > 0L) out[seq_len(m)] <- cell_draws[seq_len(m)]
    out
  }
  ds <- .re_cov_nested_beta_draws(beta_nodes, cov_nodes, w,
                                  as.integer(n_draws), beta_names,
                                  debias_nodes = nodes, debias_idx = idx)
  ds$draws
}

# The escalation itself, for a backend that integrates an outer hyperparameter
# grid (gcol33/tulpa#306). One selector, one recombination, one record.
#
# The selector's input is ALREADY on the fit: `control$diagnose_skew` (on by
# default) re-dispatches the kernel at the fitted MAP cell and attaches
# gamma_3 and the inner importance curve, which is exactly what
# `.subspace_bands()` reads. So selecting costs no extra solve here, unlike on
# `tulpa_re_cov_nested()`, where the correction is decided before the node grid
# is laid out and needs its own probe.
#
# `redispatch(request)` re-runs the fit's OWN grid with the correction on and
# returns the kernel result carrying `debias_draws_per_grid`. That second pass
# is the cost: the corrected shape is a property of each cell, and which cell is
# the MAP is only known once the first pass has run.
#
# The coupling closure needs the joint precision, which a grid fit does not
# retain; it is recorded as declined rather than silently reporting no
# neighbours found.
.nl_subspace_debias_attach <- function(res, cfg, redispatch, p_fixed,
                                       beta_names = NULL) {
  if (is.null(cfg)) return(res)
  if (!is.null(cfg$probe)) {
    stop("`control$subspace_debias$probe` is a `tulpa_re_cov_nested()` setting. ",
         "On a grid or joint nested fit the selector reads the bands the ",
         "inner-Laplace diagnostic already attached, so widen the probe with ",
         "`control$skew_idx` instead (or pin the set with ",
         "`control$subspace_debias$idx`).", call. = FALSE)
  }
  sel <- .subspace_select(res, cfg)
  cl_declined <- if (!identical(cfg$closure, FALSE) && is.null(res$H_joint))
    "closure_needs_joint_hessian" else NA_character_
  if (!is.na(cl_declined)) sel$added <- integer(0)
  if (!length(sel$idx)) {
    # No bands at all is a different answer from "every band came back good":
    # the selector was never given anything to read (control$diagnose_skew off,
    # or a backend that declined the inner diagnostic).
    if (is.null(cfg$idx) && !nrow(sel$bands)) cl_declined <- "selector_unavailable"
    res$subspace_debias <- .subspace_debias_record(sel, cfg,
                                                   declined = cl_declined)
    return(res)
  }
  out <- tryCatch(redispatch(.subspace_debias_request(
    list(idx = sel$idx, n_iter = cfg$n_iter, warmup = cfg$warmup,
         thin = cfg$thin))), error = function(e) NULL)
  if (is.null(out) || is.null(out$debias_draws_per_grid)) {
    res$subspace_debias <- .subspace_debias_record(sel, cfg,
                                                   declined = "redispatch_failed")
    return(res)
  }
  draws <- .nl_subspace_grid_draws(res, out$debias_draws_per_grid, sel$idx,
                                   cfg$n_draws,
                                   beta_names %||% res$fixed_names)
  if (is.null(draws)) {
    res$subspace_debias <- .subspace_debias_record(
      sel, cfg, accept = as.numeric(out$debias_accept %||% NA_real_),
      declined = "no_per_cell_fixed_moments")
    return(res)
  }
  res$draws <- draws
  res$n_fixed <- res$n_fixed %||% as.integer(p_fixed)
  res$subspace_debias <- .subspace_debias_record(
    sel, cfg, accept = as.numeric(out$debias_accept %||% NA_real_),
    declined = cl_declined)
  res
}
