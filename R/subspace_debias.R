# Select latent directions from the inner reliability bands, construct the
# Gaussian-conditioned fast correction or full-latent MH correction, and
# combine cells using their declared outer integration measure.

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

.subspace_integer <- function(value, name, minimum = 1L) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value < minimum || value != floor(value) || value > .Machine$integer.max) {
    stop("`", name, "` must be a whole number >= ", minimum, ".", call. = FALSE)
  }
  as.integer(value)
}

.subspace_indices <- function(value, name) {
  if (is.null(value)) return(NULL)
  if (!is.numeric(value) || anyNA(value) || any(!is.finite(value)) ||
      any(value < 1 | value > .Machine$integer.max | value != floor(value)) ||
      anyDuplicated(value)) {
    stop("`", name, "` must contain distinct positive whole numbers.", call. = FALSE)
  }
  as.integer(value)
}

.subspace_method <- function(value) {
  value <- value %||% "fast"
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !value %in% c("fast", "exact")) {
    stop("`subspace_debias$method` must be 'fast' or 'exact'.", call. = FALSE)
  }
  unname(value)
}

# Normalize the `control$subspace_debias` value into a settings list, or NULL
# when the correction is off. `TRUE` takes every default; a list overrides the
# entries it names. `idx` pins the set explicitly and skips the selector
# entirely, which is what makes the both-ways closure comparison possible
# without a second code path.
.subspace_debias_config <- function(spec, cila = NULL) {
  if (is.null(spec) || identical(spec, FALSE)) return(NULL)
  if (isTRUE(spec)) spec <- list()
  if (!is.list(spec)) {
    stop("`control$subspace_debias` must be TRUE, FALSE, or a list of ",
         "settings.", call. = FALSE)
  }
  known <- c("method", "idx", "probe", "band", "closure", "closure_max", "n_iter",
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
  method <- .subspace_method(spec$method)
  if (identical(method, "exact") && !is.null(cila) && !identical(cila, FALSE)) {
    stop("`subspace_debias$method = 'exact'` includes the outer IS correction; ",
         "choose one of `subspace_debias` and `cila`.", call. = FALSE)
  }
  out <- list(
    method      = method,
    idx         = .subspace_indices(spec$idx, "control$subspace_debias$idx"),
    probe       = .subspace_indices(spec$probe, "control$subspace_debias$probe"),
    band        = band,
    closure     = spec$closure %||% FALSE,
    closure_max = spec$closure_max %||% .nl_diag("debias_closure_max"),
    n_iter      = spec$n_iter %||% .nl_diag("debias_n_iter"),
    warmup      = spec$warmup %||% .nl_diag("debias_warmup"),
    thin        = spec$thin %||% .nl_diag("debias_thin"),
    n_draws     = spec$n_draws %||% .nl_diag("debias_n_draws")
  )
  for (nm in c("n_iter", "warmup", "thin", "n_draws", "closure_max")) {
    minimum <- if (nm %in% c("warmup", "closure_max")) 0 else 1
    out[[nm]] <- .subspace_integer(out[[nm]], paste0("control$subspace_debias$", nm), minimum)
  }
  out
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
  A <- tryCatch(t(solve(Vss, V[pos, rest, drop = FALSE])), error = function(e) NULL)
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
  if (!is.list(spec)) stop("`debias` must be NULL or a list.", call. = FALSE)
  idx <- as.integer(.subspace_indices(spec$idx, "debias$idx"))
  exact <- identical(.subspace_method(spec$method), "exact")
  if (!length(idx) && !exact) return(NULL)
  out <- list(idx    = idx,
       n_iter = .subspace_integer(spec$n_iter %||% .nl_diag("debias_n_iter"), "debias$n_iter"),
       warmup = .subspace_integer(spec$warmup %||% .nl_diag("debias_warmup"), "debias$warmup", 0L),
       thin   = .subspace_integer(spec$thin %||% .nl_diag("debias_thin"), "debias$thin"))
  if (as.double(out$n_iter) + out$warmup >= .Machine$integer.max) {
    stop("The total subspace sweep count exceeds the native integer range.", call. = FALSE)
  }
  if (exact) {
    out$method <- "exact"
    if (!is.null(spec$importance)) out$importance <- spec$importance
    else if (isTRUE(spec$correct_outer)) out$importance <- .subspace_importance
  }
  out
}

# What a fit records about the escalation, whether or not anything was
# corrected. Written even for an empty selection, so a fit says it looked and
# found nothing rather than being silent about it -- and so a fit that WAS
# corrected can be told from one that was not without re-reading the bands.
.subspace_debias_record <- function(sel, cfg, accept = NA_real_,
                                    declined = NA_character_) {
  list(method = cfg$method, idx = sel$idx, bands = sel$bands, closure_added = sel$added,
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
    V <- tryCatch(.spd_inverse(H[[k]]), error = function(e) NULL)
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
# grid. One selector, one recombination, one record.
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
                                       beta_names = NULL, remoments = identity) {
  if (is.null(cfg)) return(res)
  res$inference_mode <- "structured"
  res$inference_tier <- 2L
  res$target <- "Laplace outer-cell weights with Gaussian latent conditionals"
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
  if (!length(sel$idx) && !identical(cfg$method, "exact")) {
    # No bands at all is a different answer from "every band came back good":
    # the selector was never given anything to read (control$diagnose_skew off,
    # or a backend that declined the inner diagnostic).
    if (is.null(cfg$idx) && !nrow(sel$bands)) cl_declined <- "selector_unavailable"
    res$subspace_debias <- .subspace_debias_record(sel, cfg,
                                                   declined = cl_declined)
    return(res)
  }
  request <- .subspace_debias_request(c(cfg, list(correct_outer = TRUE)) |>
    utils::modifyList(list(idx = sel$idx)))
  out <- if (identical(cfg$method, "exact")) redispatch(request)
         else tryCatch(redispatch(request), error = function(e) NULL)
  if (identical(cfg$method, "exact")) {
    return(.nl_subspace_exact_attach(res, out, sel, cfg, p_fixed,
                                      beta_names, remoments))
  }
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
  res$subspace_debias$within_cell_target <- "joint density on the Gaussian conditional-mean surface"
  res$subspace_debias$cell_draws <- out$debias_draws_per_grid
  res$subspace_debias$cell_modes <- out$modes
  res$target <- "Gaussian-conditioned subspace surface with Gaussian nuisance carry and Laplace outer weights"
  res
}

# Independent final IS estimates replace the Laplace cell integrals. The
# normalized weights retain finite-particle and outer-quadrature uncertainty.
.subspace_outer_correction <- function(weights, estimates, log_ratios) {
  cols <- c("log_correction", "particle_ess", "n_particles", "log_estimate_sd",
            "pilot_evaluations", "pareto_k", "k_threshold")
  values <- matrix(NA_real_, length(weights), length(cols),
                    dimnames = list(NULL, cols))
  for (k in seq_along(weights)) {
    e <- estimates[[k]]
    if (is.numeric(e) && length(e) == length(cols)) values[k, ] <- e
  }
  active <- which(is.finite(weights) & weights > 0)
  reason <- if (!length(active)) "no_positive_cell_weight"
    else if (any(!is.finite(values[active, "log_correction"]))) "cell_estimator_unavailable"
    else if (any(vapply(active, function(k)
      length(log_ratios[[k]]) != values[k, "n_particles"], logical(1))))
      "cell_importance_weights_unavailable" else NA_character_
  ps <- NULL
  threshold <- NA_real_
  if (is.na(reason)) {
    total <- sum(values[active, "n_particles"])
    pooled <- unlist(lapply(active, function(k)
      log(weights[k]) + log_ratios[[k]] + log(total) - log(values[k, "n_particles"])),
      use.names = FALSE)
    ps <- tulpa_psis(pooled)
    threshold <- .ps_khat_threshold(length(pooled))
    reason <- if (length(pooled) < .PSIS_MIN_EVAL) "draws_too_few"
      else if (!is.finite(ps$pareto_k)) "nonfinite_pareto_k"
      else if (ps$pareto_k >= threshold) "pareto_k_above_threshold" else NA_character_
  }
  out <- list(applied = is.na(reason), declined = reason,
              diagnostics = as.data.frame(values), laplace_weights = weights,
              pareto_k = ps$pareto_k %||% NA_real_, k_threshold = threshold,
              is_ess = ps$is_ess %||% NA_real_,
              scope = "stratified latent importance correction on retained outer cells")
  if (is.na(reason)) {
    lw <- rep(-Inf, length(weights))
    lw[active] <- log(weights[active]) + values[active, "log_correction"]
    out$log_normalizer <- .tulpa_logsumexp(lw)
    out$weights <- exp(lw - out$log_normalizer)
    out$log_correction <- values[, "log_correction"]
  }
  out
}

# A node and one retained chain row identify the complete latent state.
.subspace_joint_draws <- function(nodes, modes, weights, n_draws) {
  active <- which(weights > 0 & is.finite(weights))
  if (any(vapply(active, function(k) {
    d <- nodes[[k]]
    !is.matrix(d) || !nrow(d) || ncol(d) != ncol(modes) || any(!is.finite(d))
  }, logical(1)))) {
    stop("Exact subspace correction requires a full latent chain at every weighted cell.",
         call. = FALSE)
  }
  picks <- sample.int(length(weights), n_draws, replace = TRUE, prob = weights)
  rows <- integer(n_draws)
  joint <- matrix(NA_real_, n_draws, ncol(modes))
  for (k in unique(picks)) {
    at <- which(picks == k)
    rows[at] <- sample.int(nrow(nodes[[k]]), length(at), replace = TRUE)
    joint[at, ] <- sweep(nodes[[k]][rows[at], , drop = FALSE], 2L, modes[k, ], `+`)
  }
  list(draws = joint, picks = picks, rows = rows)
}

.subspace_posterior_draws <- function(fit, idx, n) {
  record <- fit$subspace_debias
  if (!identical(record$method, "exact") || !isTRUE(record$outer$applied)) return(NULL)
  D <- ncol(record$cell_modes)
  idx <- idx %||% seq_len(D)
  if (!is.numeric(idx) || anyNA(idx) || any(idx < 1L | idx > D | idx != floor(idx))) {
    stop("`idx` must index the corrected latent vector.", call. = FALSE)
  }
  ds <- .subspace_joint_draws(record$cell_draws, record$cell_modes, fit$weights, n)
  out <- ds$draws[, idx, drop = FALSE]
  attr(out, "cells") <- ds$picks
  attr(out, "source_rows") <- ds$rows
  attr(out, "scope") <- "latent"
  attr(out, "draws_kind") <- "mcmc_mixture"
  attr(out, "theta") <- .nl_theta_matrix(fit)[ds$picks, , drop = FALSE]
  attr(out, "theta_within_cell") <- "node"
  out
}

.subspace_chain_diagnostics <- function(fit, measures, probs, pars = NULL) {
  nodes <- fit$subspace_debias$cell_draws
  active <- which(fit$weights > 0)
  do.call(rbind, lapply(active, function(k) {
    d <- nodes[[k]]
    colnames(d) <- paste0("latent_", seq_len(ncol(d)))
    fixed <- fit$fixed_names
    if (length(fixed)) colnames(d)[seq_along(fixed)] <- fixed
    out <- .tulpa_chain_diag_table(list(draws = d, param_names = colnames(d),
      n_chains = 1L), pars, measures, probs)
    if (is.null(out)) return(NULL)
    out$cell <- k
    out$weight <- fit$weights[k]
    out
  }))
}

# Express the native ratio against the retained fit's Laplace integral. The
# two mode solves may differ numerically; folded hyperpriors are separate.
.subspace_rebase_importance <- function(res, out) {
  n <- length(res$weights)
  base_prior <- .nl_log_hyperprior_folded(res, n)
  run_prior <- .nl_log_hyperprior_folded(out, n)
  if (length(out$log_marginal) != n || is.null(base_prior) || is.null(run_prior)) {
    stop("Exact subspace cell integrals must align with the retained grid.", call. = FALSE)
  }
  shift <- (out$log_marginal - run_prior) - (res$log_marginal - base_prior)
  active <- which(res$weights > 0)
  shift[setdiff(seq_len(n), active)] <- NA_real_
  for (k in seq_len(n)) {
    if (length(out$debias_importance_per_grid[[k]])) {
      out$debias_importance_per_grid[[k]][1L] <-
        out$debias_importance_per_grid[[k]][1L] + shift[k]
      out$debias_importance_log_weights_per_grid[[k]] <-
        out$debias_importance_log_weights_per_grid[[k]] + shift[k]
    }
  }
  out$debias_reference_shift <- shift
  out
}

.subspace_attach_diagnostics <- function(fit, outer) {
  fields <- grep("^pareto_k", names(fit), value = TRUE)
  fit$subspace_debias$laplace_diagnostics <- fit[fields]
  fit[fields] <- NULL
  fit$pareto_k <- outer$pareto_k
  fit$pareto_k_is_ess <- outer$is_ess
  fit$pareto_k_scope <- outer$scope
  fit$pareto_k_declined <- NA_character_
  fit$pareto_k_proposal_source <- "full-latent t at retained outer cells"
  fit$pareto_k_first_pass <- NA_real_
  fit
}

.nl_subspace_exact_attach <- function(res, out, sel, cfg, p_fixed,
                                       beta_names, remoments) {
  if (is.null(out$debias_draws_per_grid) || is.null(out$debias_importance_per_grid)) {
    stop("Exact subspace correction was not returned by the latent solver.", call. = FALSE)
  }
  out <- .subspace_rebase_importance(res, out)
  outer <- .subspace_outer_correction(res$weights, out$debias_importance_per_grid,
    out$debias_importance_log_weights_per_grid)
  record <- .subspace_debias_record(sel, cfg, out$debias_accept,
    declined = outer$declined)
  record$within_cell_target <- "p(x | theta_k, y)"
  record$within_cell_applied <- vapply(out$debias_draws_per_grid,
    function(d) is.matrix(d) && nrow(d) > 0L, logical(1))
  record$cell_declined <- out$debias_declined
  record$outer <- outer
  record$laplace_solver_shift <- out$debias_reference_shift
  res$subspace_debias <- record
  res$inference_mode <- "structured"
  res$inference_tier <- 2L
  if (!outer$applied) {
    res$target <- "Laplace outer-cell weights with Gaussian latent conditionals"
    return(res)
  }
  res$subspace_debias$laplace_log_marginal <- res$log_marginal
  res <- .subspace_attach_diagnostics(res, outer)
  res$weights <- outer$weights
  at <- which(res$weights > 0)
  res$log_marginal[at] <- res$log_marginal[at] + outer$log_correction[at]
  if (is.finite(res$log_evidence %||% NA_real_))
    res$log_evidence <- res$log_evidence + outer$log_normalizer
  res <- remoments(res)
  ds <- .subspace_joint_draws(out$debias_draws_per_grid, out$modes,
                               res$weights, cfg$n_draws)
  res$joint_latent_draws <- ds$draws
  res$draws <- ds$draws[, seq_len(p_fixed), drop = FALSE]
  colnames(res$draws) <- beta_names %||% res$fixed_names
  res$means <- res$beta_mean <- colMeans(res$draws)
  res$sds <- res$beta_sd <- apply(res$draws, 2L, stats::sd)
  res$n_samples <- nrow(res$draws)
  res$draw_nodes <- ds$picks
  res$draws_theta <- .nl_theta_matrix(res)[ds$picks, , drop = FALSE]
  res$subspace_debias$source_rows <- ds$rows
  res$subspace_debias$cell_draws <- out$debias_draws_per_grid
  res$subspace_debias$cell_modes <- out$modes
  res$weights_source <- "subspace_is"
  res$target <- "finite-grid IS approximation to p(x, theta | y), with true conditional MH within cells"
  res$draws_kind <- "mcmc_mixture"
  res$n_fixed <- res$n_fixed %||% as.integer(p_fixed)
  res
}
