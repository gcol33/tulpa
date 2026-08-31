# Nested-Laplace integration over random-effect covariances Sigma.
#
# Bias-2 fix (the `Marginalize Derived Quantities` principle): rather than
# report each Sigma at its mode (the plug-in MAP, biased low for skewed
# variance-component marginals at small G), integrate the Laplace marginal
# likelihood over the joint Sigma-grid and report a marginalized median and
# interval for every Sigma and its derived scale / correlation parameters.
#
# Composition: the inner solve at each grid point is the multi-RE Laplace fit
# (tulpa_laplace, which returns the Laplace log-marginal at SUPPLIED covariances);
# the grid is the nested_laplace + CCD recipe over the stacked covariance
# parameters; the summary reuses the same weighted-quantile machinery as the
# spatial/temporal nested-Laplace surface.
#
# A model may carry several random-effect terms (e.g. `(1 + x | g) + (1 | h)`),
# each with its own covariance block, and a block may be CORRELATED (a full
# `Sigma`, `(1 + x | g)`) or UNCORRELATED (a diagonal `Sigma`, `(1 + x || g)`).
# Each block contributes its own parameters to the joint integration grid; a
# single-term model is the length-1 case of the same path.

# Node checkpoint/resume for the CCD / grid integration.
# `checkpoint = list(path =, resume =)`. Nodes are deterministic given the
# fingerprint (which folds the data, layout and the node grid), so each node is
# keyed by its integer index. The store is a single RDS rewritten atomically
# (temp + rename) after each completed node, so a kill leaves either the prior
# complete state or the new one -- never a torn file. A present file whose
# fingerprint disagrees errors rather than resuming onto a stale result; an
# unreadable file (corrupt) re-solves from scratch. Returns NULL when no path is
# given, so the caller wires it unconditionally.
.re_cov_node_checkpoint <- function(checkpoint, fingerprint) {
  spec <- .nl_checkpoint_args(list(checkpoint = checkpoint))
  path <- spec$path
  if (!nzchar(path)) return(NULL)
  resume <- isTRUE(spec$resume)
  store  <- list()
  if (file.exists(path)) {
    if (!resume) {
      file.remove(path)
    } else {
      obj <- tryCatch(readRDS(path), error = function(e) NULL)
      if (!is.null(obj)) {
        if (!identical(obj$fingerprint, fingerprint)) {
          stop("`checkpoint`: '", path, "' was written for different data, ",
               "layout, or integration grid (fingerprint mismatch). Use a ",
               "fresh path or set checkpoint$resume = FALSE to start over.",
               call. = FALSE)
        }
        store <- obj$nodes
      }
    }
  }
  e <- new.env(parent = emptyenv())
  e$store <- store
  list(
    has  = function(key) !is.null(e$store[[key]]),
    get  = function(key) e$store[[key]],
    save = function(key, value) {
      e$store[[key]] <- value
      tmp <- paste0(path, ".tmp")
      saveRDS(list(fingerprint = fingerprint, nodes = e$store), tmp)
      file.rename(tmp, path)
    }
  )
}

# log-Cholesky <-> matrix helpers ---------------------------------------------
# theta packs the lower Cholesky factor L (Sigma = L L') in column-major
# lower-triangular order: diagonal entries as log(L_ii) (positivity), strictly
# lower entries as raw L_ij. Length c(c+1)/2. Keeps Sigma positive definite for
# every theta in R^{c(c+1)/2}.
.re_logchol_to_L <- function(theta, c) {
  L <- matrix(0, c, c)
  idx <- 1L
  for (j in seq_len(c)) {
    for (i in j:c) {
      L[i, j] <- if (i == j) exp(theta[idx]) else theta[idx]
      idx <- idx + 1L
    }
  }
  L
}

.re_L_to_logchol <- function(L, c) {
  theta <- numeric(c * (c + 1L) / 2L)
  idx <- 1L
  for (j in seq_len(c)) {
    for (i in j:c) {
      theta[idx] <- if (i == j) log(max(L[i, j], 1e-8)) else L[i, j]
      idx <- idx + 1L
    }
  }
  theta
}

# Multi-block covariance layout ------------------------------------------------
# Several RE terms share one nested integration. Each term becomes a covariance
# `block`, described once and reused by the prior, the parameter packing, and the
# derived-quantity summary. A block is `full` (correlated, `Sigma = L L'`,
# c(c+1)/2 log-Cholesky params, LKJ-style prior + correlations reported) or not
# (uncorrelated / diagonal `Sigma`, c log-SD params, no off-diagonal). A scalar
# `(1 | g)` term is the degenerate c = 1 block.

# Normalize the `re_terms` argument: accept either a single term (a list with an
# `idx` field) or a list of such terms. Returns a list of terms.
.as_re_terms_list <- function(re_terms) {
  if (is.null(re_terms)) stop("`re_terms` must be supplied.", call. = FALSE)
  if (!is.null(re_terms$idx)) list(re_terms) else re_terms
}

# Build the per-block layout from normalized re_terms. `n_obs` is needed to
# default the scalar-block design Z to the intercept column.
#
# A block without `idx` is a *covariance-only* block: it carries the dimension /
# correlation structure (nc, full, k) for the Sigma packing but no
# per-observation design (idx = Z = NULL). This is the form the structure-
# agnostic `tulpa_re_aghq(make_group = )` path uses, where the per-group
# likelihood oracle -- not the engine -- owns the design and observation
# granularity. The per-row callers (single-arm AGHQ, re_cov_nested / gibbs)
# always pass `idx` and get the full design path.
.re_cov_block_layout <- function(re_terms, n_obs) {
  lapply(seq_along(re_terms), function(m) {
    rt <- re_terms[[m]]
    nc <- as.integer(rt$n_coefs %||% 1L)
    if (nc < 1L) stop("RE block has n_coefs < 1.", call. = FALSE)
    if (is.null(rt$n_groups)) {
      stop(sprintf("RE block %d must supply `n_groups`.", m), call. = FALSE)
    }
    # Default to a full (correlated) covariance when c > 1; an explicit
    # `correlated = FALSE` selects a diagonal Sigma. For c = 1 the two coincide.
    full <- (rt$correlated %||% TRUE) && nc > 1L
    base <- list(nc = nc, full = full,
                 k = as.integer(if (full) nc * (nc + 1L) / 2L else nc),
                 label = rt$label %||% rt$group_var %||% NA_character_,
                 n_groups = as.integer(rt$n_groups))
    if (is.null(rt$idx)) {
      # Covariance-only block (make_group path): no per-observation design.
      return(c(base, list(idx = NULL, Z = NULL)))
    }
    if (nc > 1L && is.null(rt$Z)) {
      stop(sprintf("RE block %d (n_coefs = %d) requires `Z` (the n_obs x ",
                   "n_coefs RE design).", m, nc), call. = FALSE)
    }
    # Design: a supplied Z (slopes, incl. a single `(0 + x | g)`) or the
    # intercept indicator when absent (a `(1 | g)` block).
    Z <- if (is.null(rt$Z)) matrix(1, n_obs, nc) else as.matrix(rt$Z)
    c(base, list(idx = as.integer(rt$idx), Z = Z))
  })
}

.re_cov_block_label <- function(bl, m) {
  if (!is.null(bl$label) && !is.na(bl$label)) bl$label else paste0("re", m)
}

# Stacked theta -> list of Cholesky factors L_m, one per block. A full block
# unpacks log-Cholesky coords; a diagonal block exponentiates its log-SDs onto
# the diagonal (L_m = diag(sigma)).
.re_cov_theta_to_L_list <- function(theta, layout) {
  out <- vector("list", length(layout)); pos <- 0L
  for (m in seq_along(layout)) {
    bl <- layout[[m]]; th <- theta[pos + seq_len(bl$k)]; pos <- pos + bl$k
    out[[m]] <- if (bl$full) {
      .re_logchol_to_L(th, bl$nc)
    } else {
      Lm <- matrix(0, bl$nc, bl$nc); diag(Lm) <- exp(th); Lm
    }
  }
  out
}

# Inverse of .re_cov_theta_to_L_list: stack the per-block log-Cholesky / log-SD
# coordinates of supplied factors into one theta vector (used for grid centring).
.re_cov_L_list_to_theta <- function(L_list, layout) {
  unlist(lapply(seq_along(layout), function(m) {
    bl <- layout[[m]]; L <- L_list[[m]]
    if (bl$full) .re_L_to_logchol(L, bl$nc)
    else log(pmax(diag(L), 1e-8))
  }), use.names = FALSE)
}

# Build the inner-solve `re_list` for tulpa_laplace from per-block factors. A
# scalar block (c = 1) is passed as a marginal SD (the diagonal Laplace path);
# a c > 1 block is passed as the Cholesky factor `L` (correlated path) -- a
# diagonal `L` reproduces the uncorrelated covariance exactly.
.re_cov_build_re_list <- function(L_list, layout) {
  lapply(seq_along(layout), function(m) {
    bl <- layout[[m]]; L <- L_list[[m]]
    base <- list(idx = bl$idx, n_groups = bl$n_groups, n_coefs = bl$nc, Z = bl$Z)
    if (bl$nc == 1L) c(base, list(sigma = as.numeric(L[1L, 1L])))
    else c(base, list(L = L))
  })
}

# Per-block log-prior in the block's integration coordinates. Shared by the
# exported single-block builder (re_cov_pc_lkj_prior) and the joint multi-block
# prior (.re_cov_joint_prior) -- one source of truth for the PC + LKJ + Jacobian
# algebra. `full` selects the log-Cholesky (correlated) coordinates with the LKJ
# term; otherwise log-SD (diagonal) coordinates with no correlation.
.re_cov_block_logprior <- function(nc, full, prior_sigma, eta) {
  U <- prior_sigma[1L]; alpha <- prior_sigma[2L]
  if (U <= 0 || alpha <= 0 || alpha >= 1) {
    stop("`prior_sigma = c(U, alpha)` needs U > 0 and 0 < alpha < 1.",
         call. = FALSE)
  }
  lambda     <- -log(alpha) / U
  log_lambda <- log(lambda)

  if (full) {
    jac_coef <- nc + 2L - seq_len(nc)        # (c + 2 - i) on each log L_ii
    function(th) {
      L      <- .re_logchol_to_L(th, nc)
      logLii <- log(diag(L))                 # = th diagonal entries
      sig    <- sqrt(rowSums(L^2))           # sigma_i = sqrt(Sigma_ii)
      logsig <- log(sig)
      lp <- sum(log_lambda - lambda * sig)   # PC on each marginal SD
      # LKJ: (eta - 1) log det(R), log det(R) = 2 sum log L_ii - 2 sum log sigma_i.
      if (nc > 1L && eta != 1) {
        lp <- lp + (eta - 1) * (2 * sum(logLii) - 2 * sum(logsig))
      }
      # Change of variables (sigma, R) -> theta.
      lp + sum(jac_coef * logLii) - nc * sum(logsig)
    }
  } else {
    # Diagonal: th = log sigma_i, so d sigma_i / d th_i = sigma_i and the
    # log-Jacobian is sum_i log sigma_i = sum_i th_i. No correlation term.
    function(th) {
      sig <- exp(th)
      sum(log_lambda - lambda * sig) + sum(th)
    }
  }
}

#' PC + LKJ hyperprior for a random-effect covariance
#'
#' @description
#' Construct the default weakly-informative hyperprior used by
#' [tulpa_re_cov_nested()] for one covariance block: independent
#' Penalized-Complexity (PC) priors on the marginal standard deviations
#' `sigma_i` together with an LKJ prior on the correlation matrix `R`
#' (correlated block) or no correlation (diagonal block), returned as a
#' `log_prior_theta` function in the block's integration coordinates.
#'
#' @details
#' For a correlated block the prior is specified on the natural scale,
#' `p(sigma, R) = LKJ(R | eta) * prod_i PC(sigma_i)`, then pushed to the
#' log-Cholesky coordinates `theta` of `Sigma = L L'` by the exact
#' change-of-variables Jacobian. For a diagonal (uncorrelated) block the LKJ
#' factor drops and `theta_i = log sigma_i` with Jacobian `sum_i theta_i`.
#'
#' PC prior (Simpson et al. 2017) on each marginal SD: exponential with rate
#' `lambda = -log(alpha) / U`, so `P(sigma_i > U) = alpha` -- the
#' `prior_sigma = c(U, alpha)` convention also used by the SPDE prior in tulpa.
#'
#' LKJ prior (Lewandowski et al. 2009) on the correlation matrix:
#' `p(R)` proportional to `det(R)^(eta - 1)`. `eta = 1` is uniform over
#' correlation matrices; `eta > 1` concentrates toward the identity. The
#' normalizing constant is dropped (constant across the grid, so it cancels when
#' the integration weights are renormalized).
#'
#' Jacobian (correlated block): with `theta` packing `log L_ii` on the diagonal
#' and the raw strict-lower entries of `L`, the change of variables from
#' `(sigma, R)` to `theta` adds `sum_i (c + 2 - i) * log L_ii  -  c * sum_i log
#' sigma_i` to `log p(sigma, R)`. (Composition of the log-diagonal map, the
#' standard Cholesky-to-covariance Jacobian `2^c prod_i L_ii^(c+1-i)`, and the
#' covariance-to-`(sigma, R)` Jacobian; verified against numerical
#' differentiation in `test-re-cov-prior.R`.)
#'
#' @param n_coefs Number of coefficients `c` in the RE block.
#' @param prior_sigma `c(U, alpha)` giving `P(sigma_i > U) = alpha` (default
#'   `c(3, 0.05)`), applied independently to every marginal SD.
#' @param eta LKJ shape (default 2). `eta = 1` is uniform on correlation
#'   matrices; larger values favour weaker correlations. Ignored for a diagonal
#'   block.
#' @param correlated `TRUE` (default) for a full covariance block (log-Cholesky
#'   coordinates, LKJ prior); `FALSE` for a diagonal / uncorrelated block
#'   (log-SD coordinates, no correlation). For `n_coefs = 1` the two coincide.
#'
#' @return A `function(theta)` returning the scalar log prior density in the
#'   block's integration coordinates, suitable for one block of the
#'   `log_prior_theta` argument of [tulpa_re_cov_nested()].
#' @seealso [tulpa_re_cov_nested()]
#' @export
re_cov_pc_lkj_prior <- function(n_coefs, prior_sigma = c(3, 0.05), eta = 2,
                                correlated = TRUE) {
  c_re <- as.integer(n_coefs)
  if (c_re < 1L) stop("`n_coefs` must be >= 1.", call. = FALSE)
  .re_cov_block_logprior(c_re, isTRUE(correlated) && c_re > 1L, prior_sigma, eta)
}

# Joint default prior over all blocks: blocks are a priori independent, so the
# joint log-prior is the sum of per-block log-priors evaluated on each block's
# slice of the stacked theta.
.re_cov_joint_prior <- function(layout, prior_sigma, eta) {
  blocks <- lapply(layout, function(bl)
    .re_cov_block_logprior(bl$nc, bl$full, prior_sigma, eta))
  ks <- vapply(layout, `[[`, integer(1), "k")
  function(theta) {
    pos <- 0L; lp <- 0
    for (m in seq_along(blocks)) {
      lp <- lp + blocks[[m]](theta[pos + seq_len(ks[m])])
      pos <- pos + ks[m]
    }
    lp
  }
}

# Hyperprior convention on the scale of a Gaussian latent block.
#
# icar / rw1 / rw2 / ar1(tau) / iid, the nested-Laplace path's own scale axes
# (build_blocks_from_spec in src/nested_laplace_multi.cpp; the single-block
# driver in R/nested_laplace.R), carry NO hyperprior on tau / sigma: the C++
# kernels there compute only log p(x | tau), the grid is uniform in log(theta),
# and the outer weights are plain softmax(log_marginal) with no added term --
# flat in log(theta). The one existing exception, ar1's rho, follows the same
# rule: `.nl_apply_ar1_rho_prior()` defaults to Beta(1, 1), a no-op, and only
# a caller-supplied `rho_prior` moves it off flat.
#
# .re_cov_theta_fit() is shared by tulpa_re_cov_nested() and tulpa_eb(), whose
# test suites assert the two compute the SAME theta_hat from the SAME objective
# (test-eb.R, "tulpa_eb() and tulpa_re_cov_nested() find the same theta_hat") --
# so the two must always resolve to the same hyperprior, never one per function.
# `hyperprior = "flat"` (the default for both) makes that objective match the
# nested-Laplace convention above: the zero function, unless the caller already
# supplied a `log_prior_theta` of their own. `hyperprior = "pc_lkj"` opts into
# the weakly-informative PC + LKJ prior built from `prior_sigma` / `eta`
# (re_cov_pc_lkj_prior()) -- the regularizer tulpa_eb() documents as what keeps
# a block off the `sigma = 0` boundary at small G, still available on request.
.re_cov_resolve_hyperprior <- function(hyperprior, log_prior_theta) {
  if (is.null(log_prior_theta) && identical(hyperprior, "flat")) {
    return(function(theta) 0)
  }
  log_prior_theta
}

# Derived quantities of one block's Sigma draws as a matrix: one row per Sigma,
# named columns sigma_i (= sqrt(Sigma_ii)); for a `full` block also rho_ij
# (i<j) and the upper-triangular Sigma_ij; for a diagonal block only the
# diagonal Sigma_ii (off-diagonals are structurally zero, so no rho). Each value
# is a transform of a SINGLE Sigma (never of summarized components), so this is
# the per-cell / per-draw input the weighted-quantile summary and the
# posterior-draw synthesis consume (`Marginalize Derived Quantities`).
#
# Every column also carries its DOMAIN, in the matrix's `domain` attribute: a
# scale and a variance are `positive`, a correlation is `correlation`, an
# off-diagonal covariance is `unbounded`. The moment-matched interval
# (`.nl_moment_quantile`) forms each interval on that domain's own coordinate,
# so the domain is registered here beside the quantity it belongs to rather than
# recovered from the column name downstream.
.re_cov_derived_matrix <- function(Sig_list, nc, full = TRUE) {
  cols <- list(); dom <- character(0)
  put <- function(name, domain, fn) {
    cols[[name]] <<- vapply(Sig_list, fn, numeric(1))
    dom[[name]]  <<- domain
  }
  for (i in seq_len(nc)) {
    put(sprintf("sigma_%d", i), "positive", function(S) sqrt(S[i, i]))
  }
  if (full && nc > 1L) {
    for (i in seq_len(nc - 1L)) for (j in (i + 1L):nc) {
      put(sprintf("rho_%d%d", i, j), "correlation",
          function(S) S[i, j] / sqrt(S[i, i] * S[j, j]))
    }
  }
  if (full) {
    for (i in seq_len(nc)) for (j in i:nc) {
      put(sprintf("Sigma_%d%d", i, j),
          if (i == j) "positive" else "unbounded",
          function(S) S[i, j])
    }
  } else {
    for (i in seq_len(nc)) {
      put(sprintf("Sigma_%d%d", i, i), "positive", function(S) S[i, i])
    }
  }
  D <- do.call(cbind, cols)   # length(Sig_list) x n_derived, named columns
  attr(D, "domain") <- unname(dom[colnames(D)])
  D
}

# Combine the per-block derived matrices for a set of joint Sigma draws. Each
# element of `Sig_node_list` is the list of per-block Sigma matrices for one
# cell / draw. With a single block the column names are bare (`sigma_1`,
# `rho_12`, ...); with several blocks each is prefixed by the block label
# (`g.sigma_1`, `h.sigma_1`, ...) so terms stay distinguishable.
.re_cov_derived_matrix_multi <- function(Sig_node_list, layout) {
  M <- length(layout)
  parts <- lapply(seq_len(M), function(m) {
    bl <- layout[[m]]
    Sig_m <- lapply(Sig_node_list, `[[`, m)
    D <- .re_cov_derived_matrix(Sig_m, bl$nc, full = bl$full)
    if (M > 1L) {
      colnames(D) <- paste0(.re_cov_block_label(bl, m), ".", colnames(D))
    }
    D
  })
  D <- do.call(cbind, parts)
  attr(D, "domain") <- unlist(lapply(parts, attr, "domain"), use.names = FALSE)
  D
}

# Marginalized summary of one or more random-effect covariances. Shared by the
# grid integrator (tulpa_re_cov_nested, weighted grid cells) and the Gibbs
# sampler (tulpa_re_cov_gibbs, equal-weight posterior draws). Each element of
# `Sig_node_list` is the per-block list of Sigma matrices for one cell / draw.
# `Marginalize Derived Quantities`: each derived value is computed PER matrix
# then summarized, never assembled from summarized components.
#
# `support` names the KIND of node set, which decides how the median and the
# interval are read off it; the kinds and their outer-edge policies are the one
# `.NL_SUPPORT` table. The grid integrator's tensor cells are `"density"`, its
# central-composite design is `"moment_rule"` (the node positions carry no mass,
# so the interval comes from the moments via `.nl_moment_quantile()` on each
# quantity's own domain), and the Gibbs sweep's equal-weight
# posterior draws are `"sample"` -- a CDF like the grid, but over order
# statistics rather than cell representatives, so it clamps at the extremes
# instead of mirroring a half-cell it does not have. The mean
# and SD columns are the same weighted moments in every case.
.re_cov_derived_summary <- function(Sig_node_list, w, layout,
                                    support = .NL_SUPPORT_KINDS) {
  support <- match.arg(support)
  D   <- .re_cov_derived_matrix_multi(Sig_node_list, layout)
  dom <- attr(D, "domain")
  probs <- c(0.025, 0.5, 0.975)

  # The WITHIN-CELL read is pinned to `chord` here, which is NOT the engine
  # default. `x` is a DERIVED quantity evaluated at the
  # nodes -- `sigma_i`, `rho_ij`, `Sigma_ij` -- so its values are function
  # values, not the integration design's own cell coordinates on the axis being
  # reported. A box read needs a partition that tiles the reported axis, and
  # half the gap between two derived values is not a cell width, which is the
  # same objection that makes `sample` decline. The measurement that moved the
  # default was taken on the outer hyperparameter axes, where the grid's own
  # cells ARE that partition; extending it here would be asserting a property
  # nothing measured.
  summarize <- function(x, domain) {
    ms <- .nl_wtd_mean_sd(x, w)
    q  <- .nl_summary_quantile(x, w, probs, domain, support, "chord")
    c(mean = ms$mean, sd = ms$sd, median = q[2L], ci_lo = q[1L], ci_hi = q[3L])
  }
  post <- t(vapply(seq_len(ncol(D)),
                   function(j) summarize(D[, j], dom[[j]]), numeric(5)))
  rownames(post) <- colnames(D)
  posterior <- data.frame(parameter = rownames(post), post,
                          row.names = NULL, check.names = FALSE)

  M <- length(layout)
  Sig_mean <- lapply(seq_len(M), function(m) {
    Sig_m <- lapply(Sig_node_list, `[[`, m)
    Reduce(`+`, Map(function(S, wi) S * wi, Sig_m, w))
  })
  Sigma_mean <- if (M == 1L) Sig_mean[[1L]] else
    stats::setNames(Sig_mean,
                    vapply(seq_len(M), function(m)
                      .re_cov_block_label(layout[[m]], m), character(1)))
  list(posterior = posterior, Sigma_mean = Sigma_mean)
}

# Posterior beta draws for the nested-Laplace mixture. Each grid node k carries
# a Gaussian fixed-effect block N(beta_k, Vb_k) from its inner Laplace solve;
# the node weights w summarize the joint Sigma posterior. Drawing node ~
# Categorical(w) then beta ~ N(beta_k, Vb_k) yields an equal-weight sample of
# the marginal fixed-effect posterior (the Sigma uncertainty is propagated
# through the node mixture). Nodes with zero weight or a failed solve are
# dropped. Returns the draws plus the per-draw node index (`picks`), so
# node-level quantities (e.g. the hyperparameter log-prior for power-scaling)
# can be aligned draw-by-draw.
#
# `debias_nodes` / `debias_idx` replace the Gaussian block of
# the SELECTED fixed-effect coordinates with the node's Metropolis draws, the
# rest of the block following from the Gaussian conditional
# (`.subspace_node_draws`). Everything else about the mixture -- which nodes are
# usable, how the weights are renormalized, how a node is picked -- is
# unchanged, so the correction is scoped to the coordinates the selector named.
.re_cov_nested_beta_draws <- function(beta_nodes, beta_cov_nodes, w,
                                      n_draws, beta_names,
                                      debias_nodes = NULL, debias_idx = NULL) {
  p  <- ncol(beta_nodes)
  ok <- is.finite(w) & w > 0 & is.finite(rowSums(beta_nodes))
  # Cholesky each usable node's fixed-effect covariance. A node whose covariance
  # is missing or not factorizable (a weakly identified CCD corner with a
  # near-singular H_beta) is DROPPED and its weight redistributed -- keeping it
  # with a zero factor would make its full-weight draws a point mass at beta_k
  # and understate the fixed-effect spread in confint()/vcov()/summary().
  Lb <- vector("list", length(ok))
  for (k in seq_along(ok)) {
    if (!isTRUE(ok[k])) next
    V <- beta_cov_nodes[[k]]
    if (is.null(V) || any(!is.finite(V))) { ok[k] <- FALSE; next }
    L <- tryCatch(t(chol((V + t(V)) / 2)), error = function(e) NULL)
    if (is.null(L)) { ok[k] <- FALSE; next }
    Lb[[k]] <- L
  }
  if (!any(ok)) return(NULL)
  w2 <- w; w2[!ok] <- 0; w2 <- w2 / sum(w2)
  picks <- sample.int(length(w2), n_draws, replace = TRUE, prob = w2)
  out <- matrix(NA_real_, n_draws, p)

  # Positions of the corrected coordinates within the fixed-effect block. A
  # selected index beyond `p` is a random effect the closure pulled in: it moves
  # under the sampler, which is the point, but it is not a reported coefficient
  # and so has no position here. With none of them the Gaussian mixture below is
  # the untouched plain path, right down to its RNG consumption.
  pos <- integer(0); take <- integer(0)
  if (!is.null(debias_idx) && length(debias_idx)) {
    take <- which(debias_idx <= p)
    pos  <- as.integer(debias_idx[take])
  }
  if (length(pos) == 0L || is.null(debias_nodes)) {
    for (d in seq_len(n_draws)) {
      k <- picks[d]
      out[d, ] <- beta_nodes[k, ] + as.numeric(Lb[[k]] %*% stats::rnorm(p))
    }
    colnames(out) <- beta_names %||% paste0("beta", seq_len(p))
    return(list(draws = out, picks = picks))
  }

  for (k in unique(picks)) {
    rows <- which(picks == k)
    dk <- debias_nodes[[k]]
    got <- if (!is.null(dk) && is.matrix(dk) && ncol(dk) >= max(take))
      .subspace_node_draws(beta_nodes[k, ], beta_cov_nodes[[k]], pos,
                           dk[, take, drop = FALSE], length(rows))
      else NULL
    # A node whose sampler declined keeps the Gaussian draw it would have had,
    # so one unusable node degrades that node rather than the fit.
    if (is.null(got)) {
      got <- t(vapply(rows, function(ignored)
        beta_nodes[k, ] + as.numeric(Lb[[k]] %*% stats::rnorm(p)),
        numeric(p)))
    }
    out[rows, ] <- got
  }
  colnames(out) <- beta_names %||% paste0("beta", seq_len(p))
  list(draws = out, picks = picks)
}

# Draws of the NON-fixed latent coordinates the subspace sampler moved -- the
# random effects a closure or an explicit probe pulled into S, which
# `.re_cov_nested_beta_draws()` drops because they are not reported
# coefficients.
#
# The node mixture is REUSED rather than redrawn: `picks` is the very vector
# `.re_cov_nested_beta_draws()` returned, one node index per draw sampled from
# the integration weights, so the coordinates summarized here and the fixed
# effects are marginalized over the same weighted node set. What is left to do
# is the per-node reconstruction, which the fixed-effect path does not perform
# for these coordinates: a node's Metropolis output holds x_S - mode_S, so the
# value is the node's own mode plus a sampled offset.
#
# `centers` is one row per node holding the modes of the requested coordinates,
# `cols` their columns within each node's offset matrix. A node whose sampler
# declined contributes its mode, so one unusable node degrades that node. NULL
# when no picked node sampled, or when any assembled value is non-finite -- the
# caller then reports the Gaussian mixture, never a half-sampled column.
.re_cov_debias_coord_draws <- function(picks, debias_nodes, centers, cols) {
  q <- length(cols)
  if (q == 0L || is.null(picks) || !length(picks) || is.null(debias_nodes)) {
    return(NULL)
  }
  centers <- as.matrix(centers)
  if (ncol(centers) != q) return(NULL)
  out <- matrix(NA_real_, length(picks), q)
  sampled <- FALSE
  for (k in unique(picks)) {
    rows <- which(picks == k)
    mu <- centers[k, ]
    dk <- debias_nodes[[k]]
    if (is.matrix(dk) && nrow(dk) > 0L && ncol(dk) >= max(cols)) {
      take <- sample.int(nrow(dk), length(rows), replace = TRUE)
      out[rows, ] <- dk[take, cols, drop = FALSE] + rep(mu, each = length(rows))
      sampled <- TRUE
    } else {
      out[rows, ] <- rep(mu, each = length(rows))
    }
  }
  if (!sampled || any(!is.finite(out))) return(NULL)
  out
}

# Marginal variance of every random-effect coefficient from an inner solve's
# per-(term, group) covariance blocks (`cov_blocks` from tulpa_laplace(
# return_re_cov = TRUE), term-major then group order). Returns one value per
# (term, group, coefficient) -- the order the latent tail of the mode uses -- or
# NULL when the blocks are absent or do not describe this layout, so a caller
# reports no within-node spread rather than a mislabelled one.
.re_cov_block_variances <- function(cov_blocks, layout) {
  if (!is.list(cov_blocks)) return(NULL)
  n_blocks <- sum(vapply(layout, function(b) as.integer(b$n_groups), integer(1)))
  if (length(cov_blocks) != n_blocks) return(NULL)
  out <- numeric(.re_cov_n_re(layout))
  pos <- 0L; at <- 0L
  for (m in seq_along(layout)) {
    bl <- layout[[m]]
    for (g in seq_len(bl$n_groups)) {
      B <- cov_blocks[[pos + g]]
      if (!is.matrix(B) || nrow(B) != bl$nc) return(NULL)
      out[at + seq_len(bl$nc)] <- diag(B)
      at <- at + bl$nc
    }
    pos <- pos + bl$n_groups
  }
  if (!all(is.finite(out))) return(NULL)
  out
}

# Total number of random-effect coefficients across the covariance blocks --
# the width of the latent tail of an inner solve's mode, and of every per-node
# random-effect matrix keyed by it.
.re_cov_n_re <- function(layout) {
  sum(vapply(layout, function(b) as.integer(b$n_groups) * as.integer(b$nc),
             integer(1)))
}

# Per-block plug-in MAP summary (Sigma, sigma, rho) from the mode theta_hat.
.re_cov_map_summary <- function(theta_hat, layout) {
  L_list <- .re_cov_theta_to_L_list(theta_hat, layout)
  M <- length(layout)
  per <- lapply(seq_len(M), function(m) {
    bl <- layout[[m]]
    S  <- L_list[[m]] %*% t(L_list[[m]])
    sig <- sqrt(diag(S))
    rho <- if (bl$full && bl$nc > 1L) {
      R <- S / outer(sig, sig); R[upper.tri(R)]
    } else numeric(0)
    list(Sigma = S, sigma = sig, rho = rho)
  })
  if (M == 1L) per[[1L]] else
    stats::setNames(per, vapply(seq_len(M), function(m)
      .re_cov_block_label(layout[[m]], m), character(1)))
}

# Shared outer-hyperparameter core: everything up to and including the mode of
# g(theta) = log_marginal(Sigma(theta)) + log_prior_theta(theta).
#
# Both outer drivers need exactly this. `tulpa_re_cov_nested()` goes on to place
# integration nodes around the mode and marginalize; `tulpa_eb()` stops here and
# reports the plug-in. Keeping the objective, its inner solve and its optimizer
# in one place is what makes "EB is the nested integrator's mode" a fact about
# the code rather than a claim about two implementations that happen to agree --
# and it is the reason .re_cov_theta_fit() returns the live `inner_fit` closure
# instead of a value: the EB path re-solves at theta_hat through the SAME closure
# the optimizer drove, so no second inner-solve configuration can drift from it.
#
# Returns the closures and the geometry (post_cov / L_scale) the nested path
# needs for node placement, so nothing downstream has to rebuild them.
.re_cov_theta_fit <- function(y, n_trials, X, re_terms,
                              family, phi, phi2 = NULL,
                              prior_sigma, eta, log_prior_theta,
                              beta_prior, n_quad,
                              max_iter, tol, n_threads,
                              caller,
                              need_scale = TRUE,
                              outer_maxit = 500L,
                              offset = NULL,
                              estimate_phi = FALSE,
                              outer_reltol = NULL,
                              sigma_init = NULL,
                              X_zi = NULL, zi_prior_sd = 2.5) {
  re_terms <- .as_re_terms_list(re_terms)
  if (!is.matrix(X)) X <- as.matrix(X)
  vd <- .validate_glm_design(y, X, n_trials, caller)
  n_trials <- vd$n_trials
  # Zero inflation adds a SECOND linear predictor, so the inner solve's mode is
  # [beta | beta_zi | RE] rather than [beta | RE]. `p_fix` stays the count
  # block -- the AGHQ profile below optimizes over it alone -- and `p_fixed` is
  # the whole fixed prefix anything reading the mode has to skip.
  if (!is.null(X_zi)) {
    X_zi <- as.matrix(X_zi)
    if (nrow(X_zi) != length(y)) {
      stop(caller, "(): `X_zi` has ", nrow(X_zi), " rows but y has length ",
           length(y), ".", call. = FALSE)
    }
  }
  p_zi <- if (is.null(X_zi)) 0L else ncol(X_zi)
  has_zi <- p_zi > 0L
  layout <- .re_cov_block_layout(re_terms, length(y))
  if (length(layout) == 0L) {
    stop(caller, "(): no random-effect terms. The outer objective is over the ",
         "random-effect covariances, so there is nothing to fit; use ",
         "tulpa_laplace() for a fixed-effect-only model.", call. = FALSE)
  }
  k <- sum(vapply(layout, `[[`, integer(1), "k"))
  if (is.null(log_prior_theta)) {
    log_prior_theta <- .re_cov_joint_prior(layout, prior_sigma, eta)
  }

  # The second dispersion reaches the inner solve rather than being defaulted
  # there. Left unthreaded, `t` fits at the compiled kernel's built-in degrees of
  # freedom whatever the caller asked for, and `tweedie` fails inside the inner
  # solve's tryCatch as a generic convergence failure. Both are resolved here:
  # a phi2 supplied for a family that takes none errors (matching
  # tulpa_laplace()), and tweedie's variance power is required by name rather
  # than defaulted, since a silently chosen power is a statistical decision the
  # caller never made.
  phi2 <- .phi2_or_stop(family, phi2)
  if (identical(.family_base(family), "tweedie")) .tweedie_power(phi2)

  # The dispersion rides as one extra coordinate, log phi, appended after the
  # covariance coordinates. Refused rather than approximated wherever the exact
  # gradient for it is unavailable: the whole point of estimating phi here is
  # that the outer objective is differentiated exactly, and a derivative-free
  # search over a k+1 dimensional simplex is a different (and much worse)
  # method wearing the same argument.
  if (isTRUE(estimate_phi)) {
    if (is.null(.family_dphi(family))) {
      stop(caller, "(): `estimate_phi = TRUE` is not available for family '",
           family, "'. The dispersion derivative of the Laplace log-marginal ",
           "is registered for: ",
           paste(.dispersion_families(), collapse = ", "),
           ". Families without a free dispersion (poisson, binomial, ",
           "truncated_poisson) have nothing to estimate.", call. = FALSE)
    }
    if (n_quad > 1L) {
      stop(caller, "(): `estimate_phi = TRUE` needs the joint-field Laplace ",
           "inner solve (`n_quad = 1`). The adaptive Gauss-Hermite path runs ",
           "through a compiled per-group oracle that the dispersion gradient ",
           "does not reach.", call. = FALSE)
    }
    if (!is.finite(phi) || phi <= 0) {
      stop(caller, "(): `phi` is the starting value for the dispersion when ",
           "`estimate_phi = TRUE`, so it must be finite and positive.",
           call. = FALSE)
    }
  }
  # Index of the dispersion coordinate within the stacked theta, or NA.
  phi_idx <- if (isTRUE(estimate_phi)) k + 1L else NA_integer_
  # Split a stacked theta into its covariance half and the dispersion it
  # implies. One definition, so the objective, the gradient and the reported
  # estimate cannot disagree about which coordinate is which.
  theta_split <- function(theta) {
    if (is.na(phi_idx)) return(list(re = theta, phi = phi))
    list(re = theta[seq_len(k)], phi = exp(theta[[phi_idx]]))
  }

  # AGHQ refinement (n_quad > 1) replaces the joint-field Laplace inner marginal
  # with the per-group adaptive Gauss-Hermite marginal (the same debias as
  # tulpa_re_aghq, applied INSIDE the Sigma integration), profiling the fixed
  # effects out by an inner optimization + a fixed-effect Laplace term. The
  # per-group integral only factorizes over ONE shared grouping factor, so AGHQ
  # is available only then; crossed RE terms keep the joint-field Laplace inner
  # solve (n_quad = 1).
  idx1 <- layout[[1L]]$idx
  ng1  <- layout[[1L]]$n_groups
  single_factor <- !is.null(idx1) && all(vapply(layout, function(b)
    identical(b$idx, idx1) && identical(b$n_groups, ng1), logical(1)))
  if (n_quad > 1L && !single_factor) {
    stop("`n_quad > 1` (adaptive Gauss-Hermite refinement of the inner solve) ",
         "requires a single shared grouping factor; this model has crossed RE ",
         "terms. Use `n_quad = 1` (the joint-field Laplace inner solve).",
         call. = FALSE)
  }
  use_core <- single_factor && n_quad > 1L

  # The compiled per-group oracle behind the AGHQ inner marginal carries one
  # linear predictor, so it cannot express the zero-inflation mixture. Refused
  # rather than run without the mixture, which would fit a different model.
  if (use_core && has_zi) {
    stop(caller, "(): `n_quad > 1` (the adaptive Gauss-Hermite inner marginal) ",
         "does not carry a zero-inflation process -- its compiled per-group ",
         "oracle has a single linear predictor. Use `n_quad = 1` (the ",
         "joint-field Laplace inner solve), which does.", call. = FALSE)
  }

  # cpp_glmm_oracle_make() takes no offset, so the AGHQ inner marginal cannot
  # carry one. Silently dropping it would fit a different model; say so instead.
  if (use_core && !is.null(offset) && any(offset != 0)) {
    stop(caller, "(): `n_quad > 1` (the adaptive Gauss-Hermite inner marginal) ",
         "does not support an offset -- the compiled per-group oracle carries ",
         "no offset term. Use `n_quad = 1` (the joint-field Laplace inner ",
         "solve), which does.", call. = FALSE)
  }

  # Inner solve: Laplace log-marginal at the supplied per-block covariances.
  # Failures at extreme grid edges (non-finite / non-convergent) return -Inf so
  # the cell gets zero weight rather than aborting the integration.
  # `phi_` defaults to the fixed dispersion, so every caller that conditions on
  # it is unchanged; the dispersion-estimating path passes exp(log phi) from the
  # theta vector instead.
  inner_logmarg <- function(L_list, phi_ = phi) {
    val <- tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X,
        re_list = .re_cov_build_re_list(L_list, layout),
        family = family, phi = phi_, phi2 = phi2, return_hessian = FALSE,
        beta_prior = beta_prior, offset = offset,
        X_zi = X_zi, zi_prior_sd = zi_prior_sd,
        max_iter = max_iter, tol = tol, n_threads = n_threads
      )$log_marginal,
      error = function(e) -Inf
    )
    if (length(val) != 1L || !is.finite(val)) -Inf else val
  }

  # Inner solve that also keeps the joint posterior precision, which the exact
  # gradient differentiates through. Separate from inner_logmarg because the
  # extra factor costs memory the objective-only path has no use for.
  # `return_hessian = FALSE`: the exact gradient reads H_joint and the mode, not
  # the marginal fixed-effect block, and this runs at every trial theta the
  # optimizer visits -- including extremes where the Schur legitimately declines
  # and would warn about a fit nobody reports.
  inner_fit_grad <- function(L_list, phi_ = phi) {
    tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X,
        re_list = .re_cov_build_re_list(L_list, layout),
        family = family, phi = phi_, phi2 = phi2, return_hessian = FALSE,
        return_joint_hessian = TRUE,
        beta_prior = beta_prior, offset = offset,
        X_zi = X_zi, zi_prior_sd = zi_prior_sd,
        max_iter = max_iter, tol = tol, n_threads = n_threads
      ),
      error = function(e) NULL
    )
  }

  # Full inner solve at the integration nodes: the Laplace log-marginal plus the
  # fixed-effect mode and its MARGINAL covariance (solve(H_beta)). Paid O(n_grid)
  # times, not per optim step.
  #
  # `re_cov = TRUE` additionally asks for the per-(term, group) marginal
  # covariance blocks Cov(b_g | y, Sigma). The nested driver requests them at its
  # integration nodes so the per-group posterior can be marginalized over the
  # Sigma grid with its within-node curvature (ranef()); it costs one triangular
  # solve per random-effect coefficient against the factor the log-determinant
  # already built, so it is off wherever nobody reads the blocks (tulpa_eb()'s
  # re-solve at theta_hat, which reports the mode).
  #
  # `compute_skew` / `debias` are the two inner-layer extras the subspace debias
  # needs: the first turns the node solve into the reliability
  # probe the selector reads, the second hands it the selected index set so the
  # flagged coordinates come back as Metropolis draws instead of a Gaussian.
  # Both default off, so every existing caller solves exactly what it did.
  inner_fit <- function(L_list, phi_ = phi, re_cov = FALSE,
                        compute_skew = FALSE, skew_idx = NULL, debias = NULL,
                        joint_hessian = FALSE) {
    tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X,
        re_list = .re_cov_build_re_list(L_list, layout),
        family = family, phi = phi_, phi2 = phi2, return_hessian = TRUE,
        return_re_cov = isTRUE(re_cov),
        return_joint_hessian = isTRUE(joint_hessian),
        beta_prior = beta_prior, offset = offset,
        X_zi = X_zi, zi_prior_sd = zi_prior_sd,
        max_iter = max_iter, tol = tol, n_threads = n_threads,
        compute_skew = isTRUE(compute_skew), skew_idx = skew_idx,
        debias = debias
      ),
      error = function(e) NULL
    )
  }

  # --- pilot init: method-of-moments per block from a Sigma = I fit ----------
  p_fix   <- ncol(X)
  p_fixed <- p_fix + p_zi
  L0_list <- lapply(layout, function(bl) diag(bl$nc))
  pilot <- tryCatch(
    tulpa_laplace(
      y = y, n_trials = n_trials, X = X,
      re_list = .re_cov_build_re_list(L0_list, layout),
      family = family, phi = phi, phi2 = phi2, return_hessian = FALSE,
      beta_prior = beta_prior, offset = offset,
      X_zi = X_zi, zi_prior_sd = zi_prior_sd,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    ),
    error = function(e) NULL
  )
  L_init_list <- L0_list
  if (!is.null(pilot) && !is.null(pilot$mode)) {
    # Skip the WHOLE fixed prefix: with zero inflation the mode carries
    # beta_zi between the count coefficients and the random effects, and
    # reading it as random-effect values would seed the search from garbage.
    re_vals <- pilot$mode[-seq_len(p_fixed)]
    pos <- 0L
    for (m in seq_along(layout)) {
      bl  <- layout[[m]]
      len <- bl$n_groups * bl$nc
      if (pos + len <= length(re_vals) && bl$n_groups > bl$nc) {
        U <- matrix(re_vals[pos + seq_len(len)], ncol = bl$nc, byrow = TRUE)
        S <- stats::cov(U)
        if (all(is.finite(S))) {
          if (!bl$full) S <- diag(diag(S), bl$nc)     # diagonal block: drop covs
          diag(S) <- pmax(diag(S), 1e-3)
          ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
          if (min(ev) > 1e-8) {
            L_init_list[[m]] <- if (bl$full) t(chol(S)) else {
              Lm <- matrix(0, bl$nc, bl$nc); diag(Lm) <- sqrt(diag(S)); Lm
            }
          }
        }
      }
      pos <- pos + len
    }
  }
  # A caller-supplied starting SD replaces the pilot's method-of-moments guess.
  # Worth having because the pilot fits at Sigma = I, which on a design whose
  # true scale is far from 1 starts the search in a flat region -- and because
  # it makes a run reproducible from its inputs rather than from a pilot fit.
  # Diagonal only: it sets each coefficient's scale and leaves any correlation
  # to be fitted, since a starting correlation is far harder to guess than a
  # starting scale.
  if (!is.null(sigma_init)) {
    s0 <- as.numeric(sigma_init)
    if (any(!is.finite(s0)) || any(s0 <= 0)) {
      stop(caller, "(): `control$sigma_init` must be finite and positive.",
           call. = FALSE)
    }
    nc_all <- vapply(layout, `[[`, integer(1), "nc")
    if (length(s0) == 1L) s0 <- rep(s0, sum(nc_all))
    if (length(s0) != sum(nc_all)) {
      stop(sprintf(paste0(
        "%s(): `control$sigma_init` must have length 1 or %d (one per ",
        "random-effect coefficient across all blocks); got %d."),
        caller, sum(nc_all), length(s0)), call. = FALSE)
    }
    pos <- 0L
    for (m in seq_along(layout)) {
      nc <- layout[[m]]$nc
      Lm <- matrix(0, nc, nc)
      diag(Lm) <- s0[pos + seq_len(nc)]
      L_init_list[[m]] <- Lm
      pos <- pos + nc
    }
  }

  theta0 <- .re_cov_L_list_to_theta(L_init_list, layout)
  # The supplied `phi` becomes the starting value rather than the fixed value.
  if (isTRUE(estimate_phi)) theta0 <- c(theta0, log(phi))

  # --- AGHQ inner solve (single shared grouping factor, n_quad > 1) ----------
  # Replace the joint-field Laplace inner_logmarg / inner_fit with the per-group
  # adaptive Gauss-Hermite marginal from the shared compiled engine: at a fixed
  # Sigma, profile the fixed effects beta out by an inner optimization of
  # sum_g log M_g(beta, Sigma) and add the fixed-effect Laplace correction
  # (0.5 p log 2pi - 0.5 logdet H_beta + log prior(beta_hat)), so log_marginal
  # integrates beta exactly as the joint-field path does -- but with each
  # per-group integral debiased by n_quad-point quadrature. beta is integrated
  # (not just profiled), so the reported fixed-effect posterior is the marginal
  # (ML-II) one, not the joint-mode (PQL) estimate.
  if (use_core) {
    nc_terms <- vapply(layout, function(b) b$nc, integer(1))
    full_vec <- vapply(layout, function(b) isTRUE(b$full), logical(1))
    Zc  <- do.call(cbind, lapply(layout, function(b) b$Z))   # n x sum(nc), stacked
    orc <- cpp_glmm_oracle_make(family, phi, as.numeric(y), as.numeric(n_trials),
                                as.matrix(X), as.matrix(Zc), as.integer(idx1), ng1)
    bp  <- .normalize_beta_prior(beta_prior, p_fix)
    log_prior_beta_at <- if (is.null(bp)) function(b) 0 else
      function(b) sum(stats::dnorm(b, bp$mean, bp$sd, log = TRUE))
    # Gradient of the fixed-effect Gaussian prior (0 when absent), for the
    # analytic beta-gradient of the inner profile.
    dlog_prior_beta <- if (is.null(bp)) function(b) rep(0, p_fix) else
      function(b) -(b - bp$mean) / bp$sd^2
    beta_warm <- if (!is.null(pilot) && !is.null(pilot$mode))
      pilot$mode[seq_len(p_fix)] else rep(0, p_fix)

    # n_quad > 1 here (the AGHQ inner path), so the analytic Fisher-identity
    # gradient is consistent with the objective. Profiling beta at a fixed Sigma
    # only needs the theta-block (first p_fix entries) of the joint gradient plus
    # the beta-prior gradient; the cached eval serves fn and gr in one sweep.
    core_solve <- function(L_list) {
      sc   <- .re_cov_L_list_to_theta(L_list, layout)
      eval_at <- .aghq_grad_cache(orc, nc_terms, full_vec, n_quad, 1.0)
      negf <- function(b) {
        v <- eval_at(c(b, sc))$f
        if (!is.finite(v) || v <= -1e9) return(.Machine$double.xmax)
        -(v + log_prior_beta_at(b))
      }
      negg <- function(b) {
        r <- eval_at(c(b, sc))
        if (!isTRUE(r$ok)) return(rep(0, p_fix))
        -(r$grad[seq_len(p_fix)] + dlog_prior_beta(b))
      }
      opt <- tryCatch(stats::optim(beta_warm, negf, negg, method = "BFGS",
                                   hessian = TRUE,
                                   control = list(reltol = 1e-10, maxit = 300L)),
                      error = function(e) NULL)
      if (is.null(opt) || !all(is.finite(opt$par))) return(NULL)
      logMb <- cpp_aghq_objective(c(opt$par, sc), orc, nc_terms, full_vec,
                                  n_quad, 1.0)
      ld <- tryCatch(as.numeric(determinant(opt$hessian,
                                            logarithm = TRUE)$modulus),
                     error = function(e) NA_real_)
      if (!is.finite(logMb) || !is.finite(ld)) return(NULL)
      lm <- logMb + log_prior_beta_at(opt$par) +
        0.5 * p_fix * log(2 * pi) - 0.5 * ld
      list(log_marginal = lm, mode = opt$par, H_beta = opt$hessian)
    }
    # Same signature as the joint-field pair above: the outer objective always
    # passes the dispersion, so both inner solves have to take it. `phi_` is
    # deliberately unused here -- `estimate_phi = TRUE` is refused for
    # `n_quad > 1` further up, so on this path phi never leaves its starting
    # value and is already baked into the compiled oracle by
    # cpp_glmm_oracle_make(). It is NOT a value that needs threading through:
    # rebuilding the oracle per solve would copy X, Z and y for nothing.
    inner_logmarg <- function(L_list, phi_ = phi) {
      r <- core_solve(L_list); if (is.null(r)) -Inf else r$log_marginal
    }
    # `re_cov` is accepted and ignored: this inner solve integrates each group's
    # random effects out by quadrature rather than conditioning at their mode, so
    # it produces no per-group mean or covariance to hand back. The driver reads
    # `re_conditional` (FALSE here) and says so rather than reporting an empty
    # random-effect table.
    # The inner-layer extras the joint-field solve carries are refused here
    # rather than ignored: this solve integrates each group out, so there is no
    # conditional latent field to score a cubic term on, no joint precision to
    # read a coupling from, and nothing for a subspace sampler to move. The
    # front door refuses `control$subspace_debias` at `n_quad > 1` for the same
    # reason; this is the guard for anything reaching the closure directly.
    inner_fit <- function(L_list, phi_ = phi, re_cov = FALSE,
                          compute_skew = FALSE, skew_idx = NULL, debias = NULL,
                          joint_hessian = FALSE) {
      if (isTRUE(compute_skew) || !is.null(debias) || isTRUE(joint_hessian)) {
        stop("the adaptive Gauss-Hermite inner marginal (n_quad > 1) exposes ",
             "no conditional latent field, so it carries neither the ",
             "inner-Laplace diagnostics nor the subspace debias. Use ",
             "n_quad = 1 (the joint-field Laplace inner solve).", call. = FALSE)
      }
      core_solve(L_list)
    }
  }

  # --- mode of g(theta) = log_marginal(Sigma(theta)) + log_prior ------------
  # The hyperprior is over the COVARIANCE coordinates; the dispersion enters
  # unpenalized, which is what makes this the ML-II estimate of phi rather than
  # a MAP under an undeclared prior. So the prior only ever sees theta's
  # covariance half.
  negg <- function(theta) {
    sp <- theta_split(theta)
    L_list <- .re_cov_theta_to_L_list(sp$re, layout)
    -(inner_logmarg(L_list, sp$phi) + log_prior_theta(sp$re))
  }

  # Analytic outer gradient. Available when the inner solve is the joint-field
  # Laplace (the AGHQ inner marginal at n_quad > 1 is a different objective, and
  # its Fisher-identity gradient is not this one) and the family has an exact
  # curvature derivative. `use_exact_grad` is resolved once here rather than per
  # optim step so a family without one falls back cleanly instead of alternating
  # between two objectives.
  use_exact_grad <- !use_core &&
    isTRUE(tryCatch(cpp_family_has_curvature_derivative(family),
                    error = function(e) FALSE)) &&
    (!has_zi || .laplace_exact_supports_zi(family))

  # Estimating the dispersion is defined by the exact gradient carrying a
  # log-phi coordinate. Without that gradient the fallbacks would search the
  # extra dimension derivative-free -- a different and much weaker method under
  # the same argument -- so refuse instead of silently substituting it.
  if (isTRUE(estimate_phi) && !use_exact_grad) {
    stop(caller, "(): `estimate_phi = TRUE` needs the exact outer gradient, ",
         "which is unavailable for family '", family, "' on this path. ",
         "The dispersion would otherwise be searched derivative-free, which ",
         "is a different method than the one this argument selects.",
         call. = FALSE)
  }
  # The gradient's dispersion coordinate covers a mixture through two sources:
  # the base family's registered phi derivatives on the rows the model leaves
  # additively separable, and the mixture engine on a genuine mixture's y = 0
  # rows, where log(pi + (1 - pi) P(Y = 0, phi)) carries phi through P(Y = 0) and
  # couples it to both predictors. A hurdle needs only the first -- its zero
  # branch is log(pi), phi-free. Refused where the second is unregistered, so a
  # family never maximizes along the base derivative under a model it does not
  # describe.
  if (isTRUE(estimate_phi) && has_zi &&
      !isTRUE(tryCatch(cpp_family_has_zi_phi_deriv(family),
                       error = function(e) FALSE))) {
    stop(caller, "(): `estimate_phi = TRUE` is not available alongside a ",
         "zero-inflation process on the family '", family,
         "'. The mixture's zero branch depends on phi through P(Y = 0), and ",
         "no dispersion derivative for that branch is registered here, so ",
         "maximizing would optimize a different objective than the one ",
         "reported. Condition on `phi`, profile it by refitting over a grid, ",
         "or use a family whose mixture dispersion IS registered ",
         "(neg_binomial_2, or any zero-truncated base as a hurdle).",
         call. = FALSE)
  }

  # The hyperprior is a closed form in theta with no inner solve behind it, so
  # its gradient is a central difference costing 2k prior evaluations -- no
  # Laplace solves. Only the expensive half of the gradient needs to be exact.
  # Differentiated over the covariance coordinates only, then padded with a zero
  # for the unpenalized dispersion so it lines up with the stacked gradient.
  d_log_prior_theta <- function(theta_re, h = 1e-6) {
    g <- vapply(seq_along(theta_re), function(j) {
      tp <- theta_re; tp[j] <- tp[j] + h
      tm <- theta_re; tm[j] <- tm[j] - h
      (log_prior_theta(tp) - log_prior_theta(tm)) / (2 * h)
    }, numeric(1))
    if (is.na(phi_idx)) g else c(g, 0)
  }

  # Curvature of the same hyperprior, over the covariance coordinates only. The
  # closed-form outer Hessian is exact for the Laplace marginal but the prior is
  # user-supplied with no closed form, so its d2/dtheta2 is a second central
  # difference -- costing only prior evaluations, no inner Laplace solves. h is
  # larger than the gradient's because a second difference divides by h^2.
  d2_log_prior_theta <- function(theta_re, h = 1e-4) {
    kk <- length(theta_re)
    Hp <- matrix(0, kk, kk)
    lp0 <- log_prior_theta(theta_re)
    for (j in seq_len(kk)) {
      tp <- theta_re; tp[j] <- tp[j] + h
      tm <- theta_re; tm[j] <- tm[j] - h
      Hp[j, j] <- (log_prior_theta(tp) - 2 * lp0 + log_prior_theta(tm)) / h^2
    }
    if (kk > 1L) for (j in seq_len(kk - 1L)) for (l in (j + 1L):kk) {
      tpp <- theta_re; tpp[j] <- tpp[j] + h; tpp[l] <- tpp[l] + h
      tpm <- theta_re; tpm[j] <- tpm[j] + h; tpm[l] <- tpm[l] - h
      tmp <- theta_re; tmp[j] <- tmp[j] - h; tmp[l] <- tmp[l] + h
      tmm <- theta_re; tmm[j] <- tmm[j] - h; tmm[l] <- tmm[l] - h
      Hp[j, l] <- Hp[l, j] <-
        (log_prior_theta(tpp) - log_prior_theta(tpm) -
         log_prior_theta(tmp) + log_prior_theta(tmm)) / (4 * h^2)
    }
    Hp
  }

  # Value and gradient of the FULL outer objective, log_marginal + log_prior, at
  # one theta. Everything downstream -- the optimizer, the marginal correction's
  # curvature -- reads this, so there is one definition of what is being
  # differentiated and the prior term cannot be attached twice or dropped once.
  # Cached because BFGS asks for value and gradient at the same theta and the
  # inner solve is the entire cost.
  grad_cache <- new.env(parent = emptyenv())
  eval_grad_at <- function(theta, want_jacobian = FALSE, want_hessian = FALSE) {
    want_list <- want_jacobian || want_hessian
    key <- paste(c(signif(theta, 12), if (want_jacobian) "J", if (want_hessian) "H"),
                 collapse = ",")
    if (!is.null(grad_cache[[key]])) return(grad_cache[[key]])
    sp <- theta_split(theta)
    L_list <- .re_cov_theta_to_L_list(sp$re, layout)
    fit <- inner_fit_grad(L_list, sp$phi)
    val <- if (is.null(fit) || !is.finite(fit$log_marginal %||% NA_real_)) NULL else {
      r <- .laplace_exact_re_grad(
        fit = fit, y = y, X = X, n_trials = n_trials, offset = offset,
        weights = NULL,
        re_list = .re_cov_build_re_list(L_list, layout),
        layout = layout, L_list = L_list, family = family,
        # The same phi2 the inner solves ran at: the gradient has to describe
        # the model the inner solve fitted, not a differently-parameterized one.
        # NA_real_ is the "no second dispersion" spelling the family kernels
        # take, so an absent phi2 reaches them as it always did.
        phi = sp$phi, phi2 = phi2 %||% NA_real_, want_jacobian = want_jacobian,
        want_hessian = want_hessian, X_zi = X_zi,
        # Appends the log-phi coordinate, so the returned gradient is stacked in
        # the same order as theta.
        want_phi = !is.na(phi_idx)
      )
      if (is.null(r)) NULL else {
        # The prior shifts the objective but not the mode, so it enters the
        # gradient and leaves J alone. H below is the Laplace marginal's exact
        # curvature over the covariance coordinates; the prior's own curvature is
        # added where the negative outer Hessian is formed (exact_grad_at).
        g <- (if (want_list) r$grad else r) + d_log_prior_theta(sp$re)
        list(f = fit$log_marginal + log_prior_theta(sp$re), g = g,
             J = if (want_list) r$J else NULL,
             H = if (want_hessian) r$H else NULL)
      }
    }
    grad_cache[[key]] <- val
    val
  }

  negg_exact <- function(theta) {
    r <- eval_grad_at(theta)
    if (is.null(r) || !is.finite(r$f)) return(.Machine$double.xmax)
    -r$f
  }
  negg_gr <- function(theta) {
    r <- eval_grad_at(theta)
    if (is.null(r)) return(rep(0, length(theta)))
    -r$g
  }

  # Nelder-Mead degenerates in one dimension (R warns as much), and k == 1 is the
  # common case: a single scalar `(1 | g)` block, whose only coordinate is
  # log(sigma). Brent bracketing over a wide log-SD interval is both the reliable
  # and the cheaper option there; the simplex takes over from k >= 2. Brent is
  # bracketed rather than iteration-limited, so `outer_maxit` applies to the
  # simplex only. With the analytic gradient available, BFGS replaces both: it
  # converges in far fewer inner solves and does not degrade with k.
  brent_lo <- log(1e-4)
  brent_hi <- log(1e3)
  # Brent is a ONE-dimensional method, so it applies only when the whole stacked
  # theta is scalar -- which the appended dispersion coordinate makes false even
  # for a single scalar covariance block.
  n_theta <- length(theta0)

  # The dispersion coordinate is bracketed, for the same reason sigma is. As phi
  # grows the objective FLATTENS rather than turning over -- a negative binomial
  # becomes a Poisson, a gamma becomes tight around its mean -- so an oversized
  # early step lands in a region where the gradient is ~0 and unbounded BFGS
  # reports convergence at a value with no support in the data. Measured on a
  # 480-row negative binomial whose profile peaks cleanly at phi = 3, unbounded
  # BFGS stopped at 3.5e15. Only this coordinate is bounded; the covariance
  # coordinates keep the full line.
  phi_lo <- log(1e-6)
  phi_hi <- log(1e6)

  # Convergence tolerance on the outer objective. One value resolved here and
  # handed to whichever method runs, rather than a literal repeated per branch
  # where the four could drift apart. L-BFGS-B measures convergence as
  # `factr * .Machine$double.eps` rather than as a relative tolerance, so the
  # same request is converted for it instead of being passed through as a
  # different number under the same name.
  reltol_gr <- outer_reltol %||% 1e-10
  reltol_nm <- outer_reltol %||% 1e-8
  factr_lbfgs <- max(1, reltol_gr / .Machine$double.eps)
  # One warning per fit rather than one per trial theta if the exact gradient is
  # declined because an inner mode did not settle (.laplace_mode_settled).
  # `unsettled$n` is live inside the block below, which is what lets the
  # gradient-driven branch notice the refusal and hand over.
  unsettled <- .new_unsettled_state()
  opt <- .with_unsettled_report(if (use_exact_grad) {
    o <- tryCatch(
      if (is.na(phi_idx)) {
        stats::optim(theta0, negg_exact, negg_gr, method = "BFGS",
                     hessian = need_scale,
                     control = list(maxit = as.integer(outer_maxit),
                                    reltol = reltol_gr))
      } else {
        stats::optim(theta0, negg_exact, negg_gr, method = "L-BFGS-B",
                     lower = c(rep(-Inf, k), phi_lo),
                     upper = c(rep(Inf, k), phi_hi),
                     hessian = need_scale,
                     control = list(maxit = as.integer(outer_maxit),
                                    factr = factr_lbfgs))
      },
      error = function(e) NULL
    )
    # A gradient-driven run that fails outright (a singular H at some trial
    # theta, say) must not take the fit down with it; fall back to the
    # derivative-free path rather than reporting a stop point as an estimate.
    # A refusal from the settled-mode gate lands here for the same reason and is
    # the more dangerous case: optim() receives a declined gradient as a vector
    # of zeros, reads that as a stationary point, and can stop at theta0 and
    # report the STARTING value as the estimate. Whatever it returned after
    # walking blind is discarded.
    if (is.null(o) || !all(is.finite(o$par)) || unsettled$n > 0L) {
      if (n_theta == 1L) {
        stats::optim(theta0, negg, method = "Brent",
                     lower = brent_lo, upper = brent_hi, hessian = need_scale,
                     control = list(reltol = reltol_gr))
      } else {
        stats::optim(theta0, negg, method = "Nelder-Mead",
                     hessian = need_scale,
                     control = list(maxit = as.integer(outer_maxit),
                                    reltol = reltol_nm))
      }
    } else o
  } else if (n_theta == 1L) {
    stats::optim(theta0, negg, method = "Brent",
                 lower = brent_lo, upper = brent_hi, hessian = need_scale,
                 control = list(reltol = reltol_gr))
  } else {
    stats::optim(theta0, negg, method = "Nelder-Mead",
                 hessian = need_scale,
                 control = list(maxit = as.integer(outer_maxit), reltol = reltol_nm))
  }, caller, unsettled)
  theta_hat <- opt$par

  # A non-zero optim code means the reported theta_hat is wherever the optimizer
  # stopped, not a maximizer -- for EB that IS the estimate, and for the nested
  # path it is the centre the integration nodes are placed around. Either way it
  # was previously returned without a word.
  if (!identical(as.integer(opt$convergence), 0L)) {
    warning(caller, "(): the outer optimization over the random-effect ",
            "covariance(s) did not converge (optim code ", opt$convergence,
            "). The reported estimate is where it stopped. Consider a longer ",
            "run, a different `re_prior` / `prior_sigma`, or more data per ",
            "group.", call. = FALSE)
  }

  # Brent reports success at a bracket endpoint, so a variance component that
  # wanted to run past the bracket is indistinguishable from a converged one by
  # the code alone. The low end is the one that happens in practice -- the
  # classic empirical-Bayes collapse to sigma = 0 -- and reporting it as an
  # estimate rather than a boundary hit would misrepresent it as a fitted value.
  # A dispersion pinned at its bracket is a finding, not an estimate: the data
  # want it outside the box, which for the upper end means the extra-Poisson
  # variation the parameter exists to carry is not there. Reported rather than
  # returned silently, since the number itself looks like any other fit.
  if (!is.na(phi_idx) && is.finite(theta_hat[[phi_idx]])) {
    lp <- theta_hat[[phi_idx]]
    if (lp <= phi_lo + 1e-6 || lp >= phi_hi - 1e-6) {
      warning(sprintf(paste0(
        "%s(): the estimated dispersion hit the %s end of its search bracket ",
        "(phi = %.3g). %s Treat it as a boundary hit, not a fitted value."),
        caller, if (lp <= phi_lo + 1e-6) "lower" else "upper", exp(lp),
        if (lp >= phi_hi - 1e-6) {
          paste("The data show no dispersion beyond what the mean-variance",
                "relation and the random effects already explain; a family",
                "without a free dispersion may be the better model.")
        } else {
          paste("The dispersion is collapsing, which usually means the",
                "response is far more variable than the family allows.")
        }), call. = FALSE)
    }
  }

  # Keyed on the stacked length, not on k: the bracket only exists when Brent
  # ran, and an appended dispersion coordinate takes that path away.
  if (n_theta == 1L && is.finite(theta_hat[1L])) {
    at_lo <- theta_hat[1L] <= brent_lo + 1e-6
    at_hi <- theta_hat[1L] >= brent_hi - 1e-6
    if (at_lo || at_hi) {
      detail <- if (at_lo) {
        paste("The variance component has collapsed: the data carry no",
              "detectable between-group variation, or the hyperprior is not",
              "holding it off the boundary.")
      } else {
        paste("The between-group variation exceeds the bracket, so the value",
              "is capped rather than fitted.")
      }
      warning(sprintf(paste0(
        "%s(): the random-effect standard deviation hit the %s end of the ",
        "search bracket (sigma = %.3g). %s Treat it as a boundary hit, not a ",
        "fitted value."),
        caller, if (at_lo) "lower" else "upper", exp(theta_hat[1L]), detail),
        call. = FALSE)
    }
  }

  # Posterior covariance of theta ~ solve(Hessian of negg). Regularize to PD;
  # fall back to a diagonal scale if the numerical Hessian is unusable. Only the
  # nested path needs this (it places nodes with it); EB reports the plug-in and
  # skips the numerical Hessian entirely.
  post_cov <- NULL; L_scale <- NULL
  if (need_scale) {
    Hn <- opt$hessian
    post_cov <- tryCatch({
      Hs <- (Hn + t(Hn)) / 2
      ev <- eigen(Hs, symmetric = TRUE, only.values = TRUE)$values
      if (min(ev) <= 1e-8) stop("non-PD Hessian")
      solve(Hs)
    }, error = function(e) {
      warning(caller, "(): outer theta-Hessian not usable (",
              conditionMessage(e), "); falling back to a diagonal proposal ",
              "scale. Integration nodes may be mis-placed -- check the outer ",
              "Pareto-k (fit$pareto_k).", call. = FALSE)
      diag(0.5^2, k)
    })
    L_scale <- t(chol(post_cov))
  }

  # No `log_marginal` here on purpose: backing it out of opt$value would be a
  # second, differently-derived copy of a number both callers already read off an
  # actual fit -- and one that goes NaN whenever the optimizer's last step hit
  # the failure sentinel.
  # `exact_grad_at` is the analytic outer gradient and, on request, the exact
  # mode Jacobian at a given theta -- NULL when this path has no exact gradient
  # (the AGHQ inner marginal, or a family whose working weight has no closed-form
  # eta-derivative). tulpa_eb(marginal = TRUE) uses it to replace the
  # finite-difference stencil; a NULL sends it back to the stencil.
  exact_grad_at <- if (!use_exact_grad) NULL else
    function(theta, want_jacobian = FALSE, want_hessian = FALSE) {
      r <- eval_grad_at(theta, want_jacobian = want_jacobian,
                        want_hessian = want_hessian)
      if (is.null(r)) return(NULL)
      if (isTRUE(want_hessian)) {
        # The negative outer Hessian the marginal correction consumes:
        # -(d2 log_marginal + d2 log_prior). The marginal half is closed-form
        # (r$H); the prior half is the cheap second difference above. NULL r$H
        # (a family without a second curvature derivative) hands the Hessian back
        # to the differencing stencil.
        if (is.null(r$H)) return(NULL)
        sp <- theta_split(theta)
        Hp <- tryCatch(d2_log_prior_theta(sp$re), error = function(e) NULL)
        if (is.null(Hp)) return(NULL)
        # The dispersion enters the objective unpenalized, so the prior adds a
        # zero row/column for log phi -- pad it to match the Laplace Hessian's
        # phi border (r$H is (k+1)x(k+1) when phi is estimated).
        if (!is.na(phi_idx) && nrow(r$H) == nrow(Hp) + 1L) {
          Hp <- rbind(cbind(Hp, rep(0, nrow(Hp))), rep(0, ncol(Hp) + 1L))
        }
        if (!all(dim(Hp) == dim(r$H))) return(NULL)
        Ht <- -(r$H + Hp)
        return(list(grad = r$g, J = r$J, H = (Ht + t(Ht)) / 2))
      }
      if (isTRUE(want_jacobian)) list(grad = r$g, J = r$J) else r$g
    }

  # The dispersion is reported separately from theta_hat's covariance half, and
  # `theta_hat` is trimmed to the covariance coordinates so every downstream
  # consumer (.re_cov_theta_to_L_list, the map summary, the marginal correction)
  # keeps receiving the vector length it was written for. `phi_hat` is the
  # estimate; `phi` stays the value the caller supplied, which is the start.
  sp_hat <- theta_split(theta_hat)
  list(layout = layout, k = k, n_trials = n_trials, X = X, p_fix = p_fix,
       # The count block and the whole fixed prefix. They differ only under
       # zero inflation, where the mode is [beta | beta_zi | RE]; callers that
       # slice the mode or name the fixed effects want `p_fixed`, while the
       # AGHQ profile optimizes over the count block alone.
       X_zi = X_zi, p_zi = p_zi, p_fixed = p_fixed,
       log_prior_theta = log_prior_theta,
       inner_logmarg = inner_logmarg, inner_fit = inner_fit,
       # Does `inner_fit` condition on the random effects (returning their mode
       # in the latent tail of `$mode`, and their covariance blocks on request)?
       # TRUE for the joint-field Laplace inner solve, FALSE for the AGHQ one,
       # which integrates them out per group. The per-group posterior is
       # reportable exactly when this is TRUE.
       re_conditional = !use_core,
       exact_grad_at = exact_grad_at,
       theta0 = theta0, theta_hat = sp_hat$re, opt = opt,
       theta_hat_full = theta_hat,
       phi_hat = if (is.na(phi_idx)) NULL else sp_hat$phi,
       estimate_phi = isTRUE(estimate_phi),
       post_cov = post_cov, L_scale = L_scale)
}


#' Nested-Laplace integration over random-effect covariances
#'
#' @description
#' For one or more random-effects terms (e.g. `(1 + x | g)`, `(1 + x || g)`, or
#' several terms together), integrate the Laplace marginal likelihood over the
#' random-effect covariances `Sigma` instead of fixing them at point estimates.
#' Reports weighted posterior summaries (mean, SD, median, 2.5\%/97.5\%) of every
#' `Sigma` and its derived scale (`sigma_i`) and correlation (`rho_ij`)
#' parameters, marginalizing the joint posterior over a `Sigma`-grid.
#'
#' This corrects the plug-in-MAP ("summary") bias: the mode of a skewed
#' variance-component marginal is biased low relative to its median, so the
#' headline summary should be the marginalized median, not the mode.
#'
#' @details
#' Each term is one covariance **block**. A correlated block (`(1 + x | g)`) is
#' a full `Sigma = L L'` parameterized by its lower Cholesky factor in
#' log-Cholesky coordinates (the log-diagonal and the strictly-lower entries of
#' `L`, `c(c+1)/2` values for a `c`-coefficient block), which keeps `Sigma`
#' positive definite for every coordinate. An uncorrelated block
#' (`(1 + x || g)`) is a diagonal `Sigma` parameterized by its `c` log standard
#' deviations. A scalar `(1 | g)` term is the degenerate `c = 1` block. Several
#' blocks stack their parameters into one integration vector; a single-term
#' model is the length-1 case.
#'
#' Integration nodes live in the whitened stacked-parameter space, centred at
#' the joint marginal-likelihood mode and rotated/scaled by the Cholesky of the
#' mode's posterior covariance (`solve(Hessian)`), so points track the posterior
#' ridge. Two node layouts are available via `integration`:
#' \itemize{
#'   \item `"ccd"` (default): a central-composite design ([ccd_grid()]) of
#'     `1 + 2k + 2^(k-q)` points for the total `k = sum_blocks` parameters, with
#'     the corrected R-INLA design weights ([ccd_weights()]). Scales polynomially
#'     in `k`, where the tensor grid is exponential.
#'   \item `"grid"`: the full `n_per_axis^k` tensor product with uniform cell
#'     weights -- denser and more robust to a non-Gaussian whitened posterior,
#'     but only tractable for small `k`.
#' }
#' Each node `k` contributes integration weight proportional to
#' `Delta_k * exp(log_marginal(Sigma_k) + log_prior_theta(theta_k))`, following
#' the INLA convention `int ~ sum_k Delta_k pi(theta_k)`.
#'
#' The two layouts also decide how the reported median and 2.5\%/97.5\% interval
#' of each derived quantity are read off the nodes. A tensor grid's uniform
#' cells discretize the posterior density, so the cumulative node weights are a
#' CDF and the summary is the weighted quantile. A CCD is a moment rule: its
#' nodes sit where they reproduce the integrand's first two moments and carry no
#' probability mass of their own, so the summary is moment-matched instead --
#' the first two weighted moments on each quantity's own coordinate (`log` for a
#' scale or a variance, `atanh` for a correlation, the identity for a covariance)
#' define a Gaussian there whose quantiles are mapped back. Scale intervals are
#' therefore positive and asymmetric, and correlation intervals stay inside
#' `(-1, 1)`. The `mean` and `sd` columns are the weighted moments under either
#' layout.
#'
#' By default (`hyperprior = "flat"`) `log_prior_theta` is the zero function:
#' flat in log(theta), the same convention the nested-Laplace spatial /
#' temporal / RE-scale axes use (icar / rw1 / rw2 / ar1's tau / iid, none of
#' which carry a hyperprior on their scale either -- see `vignette("priors")`).
#' Set `hyperprior = "pc_lkj"` to use the weakly-informative PC + LKJ hyperprior
#' instead, built per block by [re_cov_pc_lkj_prior()] and summed over blocks
#' (PC prior on each marginal SD via `prior_sigma`, LKJ prior on each correlated
#' block's correlation matrix via `eta`), expressed in the same parameterization
#' with the exact change-of-variables Jacobian. Supply a custom `log_prior_theta`
#' function to override either default (then `prior_sigma` / `eta` /
#' `hyperprior` are ignored); it must act on the full stacked parameter vector.
#' [tulpa_eb()] shares this same objective and the same default, so
#' `tulpa_eb()$theta_hat` and `tulpa_re_cov_nested()$theta_hat` stay the same
#' estimate on the same data under either setting.
#'
#' @param y,n_trials,X,family,phi Passed to [tulpa_laplace()] for the inner
#'   solve. `n_trials = NULL` defaults to 1 (binary / single-trial).
#' @param phi2 Optional second dispersion, threaded into every inner
#'   [tulpa_laplace()] solve: the Student-t degrees of freedom (`family = "t"`,
#'   default 4 when `NULL`) or the Tweedie variance power (`family = "tweedie"`,
#'   required -- a defaulted power would be a statistical decision the caller
#'   never made). A `phi2` supplied for any other family errors rather than being
#'   ignored. It is conditioned on: the integration is over the random-effect
#'   covariances, not over `phi2`.
#' @param re_terms Either a single random-effect term or a list of them. Each
#'   term is a list with `idx` (1-based group index per observation),
#'   `n_groups`, `n_coefs` (`c`), `Z` (the `n_obs x c` RE design, e.g.
#'   `cbind(1, x)` for `(1 + x | g)`; only required when `c > 1`), and
#'   `correlated` (`TRUE` for a full `Sigma`, `FALSE` for a diagonal one;
#'   defaults to `TRUE`). An optional `label` / `group_var` names the block in
#'   the output. Any `L` / `cov` / `sigma` field is ignored -- `Sigma` is what
#'   this function integrates over.
#' @param prior_sigma,eta Hyperparameters of the PC + LKJ prior used when
#'   `hyperprior = "pc_lkj"` (see [re_cov_pc_lkj_prior()]):
#'   `prior_sigma = c(U, alpha)` with `P(sigma_i > U) = alpha` (default
#'   `c(3, 0.05)`) and LKJ shape `eta` (default 2). Ignored when
#'   `hyperprior = "flat"` or `log_prior_theta` is supplied.
#' @param hyperprior `"flat"` (default) or `"pc_lkj"`. `"flat"` integrates with
#'   `log_prior_theta` the zero function (flat in log(theta)), matching the
#'   nested-Laplace convention on every other scale axis in the engine.
#'   `"pc_lkj"` builds the PC + LKJ prior from `prior_sigma` / `eta` (the
#'   regularizer that keeps a variance component off the `sigma = 0` boundary
#'   at small G). Ignored when `log_prior_theta` is supplied.
#' @param log_prior_theta Optional `function(theta)` returning a scalar log
#'   prior density on the full stacked parameter vector, overriding
#'   `hyperprior` entirely. Default `NULL`, which defers to `hyperprior`.
#' @param beta_prior Optional Gaussian prior on the fixed effects, threaded into
#'   every inner [tulpa_laplace()] solve (`list(mean, sd)`).
#'   `NULL` (default) keeps the weak built-in prior.
#' @param offset Optional observation-level offset on the linear predictor
#'   (length `length(y)`), e.g. `log(exposure)` for a rate model. Not supported
#'   with `n_quad > 1`, which errors rather than dropping it.
#' @param n_quad Quadrature order for the inner marginal. `1` (default) uses the
#'   joint-field Laplace inner solve ([tulpa_laplace()]). `> 1` refines the inner
#'   marginal with `n_quad`-point adaptive Gauss-Hermite quadrature (the
#'   [tulpa_re_aghq()] debias applied inside the `Sigma` integration), reducing
#'   the small-cluster variance attenuation for binary / low-count data. AGHQ
#'   requires a single shared grouping factor (the per-group integral must
#'   factorize); with crossed RE terms `n_quad > 1` errors. When AGHQ is used the
#'   fixed effects are integrated, so the reported fixed-effect posterior is the
#'   marginal (ML-II) one rather than the joint-mode (PQL) estimate.
#' @param X_zi Optional zero-inflation design matrix (`length(y)` rows), making
#'   the model a two-process mixture: each observation is a structural zero with
#'   probability `plogis(X_zi beta_zi)` and otherwise follows `family`. Paired
#'   with a zero-truncated family it is the hurdle model. The random effects
#'   enter the count predictor only, and the integration runs over the same
#'   covariance coordinates -- the mixture changes the inner solve, not the
#'   parameters being integrated over. The ZI coefficients are reported
#'   alongside the count ones in `coef()` / `vcov()`, so the fixed block is
#'   `ncol(X) + ncol(X_zi)` wide. Needs `n_quad = 1`: the adaptive
#'   Gauss-Hermite inner marginal runs through a single-predictor oracle.
#' @param zi_prior_sd Prior SD on `beta_zi`, keeping the logit identified where
#'   a level carries no zeros (the likelihood alone would send it to `-Inf`).
#'   Ignored when `X_zi` is `NULL`.
#' @param control A named list of numerical / tuning knobs (statistical
#'   arguments stay in the signature above). Recognized entries:
#'   \itemize{
#'     \item `integration`: node layout, `"ccd"` (default, central-composite
#'       design, scales to larger total parameter count) or `"grid"` (full
#'       tensor product).
#'     \item `n_per_axis`: points per parameter axis in the tensor grid
#'       (default 5); used only when `integration = "grid"`.
#'     \item `span`: half-width of the tensor grid in posterior standard
#'       deviations per whitened axis (default 3); grid only.
#'     \item `n_draws`: posterior draws of the fixed effects synthesized from the
#'       node mixture (default 2000), exposed as `draws` for the generic
#'       `tulpa_fit` methods. The `Sigma` posterior is summarized directly from
#'       the integration nodes in `posterior`, independent of `n_draws`.
#'     \item `seed`: optional integer seed for the fixed-effect draw synthesis.
#'     \item `diagnose_k`: if `TRUE` (default), compute the outer Pareto k-hat
#'       accuracy diagnostic for the Gaussian proposal over the hyperparameters,
#'       returned as `pareto_k`. Several proposal candidates are scored and the
#'       best is kept: `pareto_k_proposal_source` names which one produced the
#'       reported number and `pareto_k_first_pass` is the k-hat of the proposal
#'       exactly as the mode-find placed it, before refinement. A large gap
#'       between the two says the placement is poor even where the verdict is
#'       fine -- on a small-group binary fit the first pass runs 15 to 49 where
#'       the reported k-hat is 0.3 to 0.8.
#'     \item `k_samples`: importance draws for the `diagnose_k` estimate
#'       (default 500). It is a precision knob: the GPD tail size is held at
#'       the fraction that default budget implies, so raising it supplies more
#'       tail ratios for the SAME estimand rather than moving the fit to a
#'       deeper quantile of the weight distribution (gcol33/tulpa#631).
#'     \item `k_tail_points`: expert override for that tail size, in upper-tail
#'       order statistics. Silently capped at 20% of the draws, beyond which
#'       body ratios enter the tail and bias the shape.
#'     \item `max_iter`, `tol`, `n_threads`: inner-solve controls (see
#'       [tulpa_laplace()]).
#'     \item `outer_maxit`: iteration budget for the mode-finding step that
#'       centres the integration grid (default 500). Applies to the Nelder-Mead
#'       simplex used from two parameters up; the one-parameter case is bracketed
#'       by Brent. Exhausting the budget warns, since the nodes are then centred
#'       on wherever the optimizer stopped.
#'     \item `checkpoint`: node checkpoint/resume spec `list(path = , resume = )`.
#'       Each completed CCD / grid node (one inner Laplace solve) is cached to
#'       `path`; a `resume = TRUE` run loads the finished nodes and re-solves
#'       only the rest. `resume = FALSE` starts fresh. A file written for
#'       different data, layout, or grid is rejected (fingerprint mismatch).
#'       Default `NULL` (off).
#'     \item `subspace_debias`: subspace debias, `FALSE` by
#'       default. `TRUE` takes every default; a list overrides `band` (the
#'       inner-reliability floor a coordinate is selected at, default `"ok"`),
#'       `idx` (pin the corrected set explicitly, skipping the selector),
#'       `probe` (the latent indices scored, default the fixed effects),
#'       `closure` (`FALSE`, `TRUE`, or a partial-correlation threshold: grow
#'       the set by the precision-graph neighbours it is strongly coupled to),
#'       `closure_max`, and the sampler budget `n_iter` / `warmup` / `thin`.
#'       When the selected set is non-empty, each integration node reports the
#'       selected fixed-effect coordinates from a Metropolis sample of the exact
#'       conditional along the Gaussian-conditional-mean surface, and the rest
#'       from the Gaussian conditional given them; an EMPTY set leaves the fit
#'       bit-for-bit identical to the plain path. What was selected is recorded
#'       in `subspace_debias` on the returned fit.
#'   }
#'
#' @return A list with:
#'   - `posterior`: data frame with one row per parameter and columns `mean`,
#'     `sd`, `median`, `ci_lo`, `ci_hi`. Parameter names are `sigma_i`, `rho_ij`,
#'     `Sigma_ij` for a single block, prefixed by the block label
#'     (`g.sigma_1`, ...) when there are several blocks. Diagonal blocks report
#'     no `rho`.
#'   - `map`: the plug-in-mode summary at `theta_hat` (a single `list(Sigma,
#'     sigma, rho)` for one block, or a named list of them).
#'   - `Sigma_mean`: the weighted posterior mean of `Sigma` (a matrix for one
#'     block, or a named list of matrices).
#'   - `beta`, `draws`, `means`, `param_names`, `process_info`: the fixed-effect
#'     posterior from the node mixture (drives `coef`/`confint`/`vcov`/`summary`).
#'   - `re_nodes`, `re_var_nodes`: the per-group random-effect posterior at each
#'     integration node -- conditional mean and marginal variance of every
#'     (block, group, coefficient), one row per node. [ranef()] reports the
#'     `weights`-mixture of them. `NULL` at `n_quad > 1`, whose inner marginal
#'     integrates each group out instead of conditioning on it; that fit carries
#'     `ranef_unavailable` (the reason) in their place.
#'   - `re_debias_draws`, `re_debias_idx`: present when the subspace debias
#'     selected a random-effect coordinate. The sampled draws of those
#'     coordinates on the node mixture the fixed-effect draws use, and their
#'     positions within the random-effect block. [ranef()] reports those rows
#'     empirically and the rest from the Gaussian mixture, recording which in
#'     its `source` column.
#'   - `theta_hat`, `theta_grid`, `weights`, `log_marginal`, `n_grid`, `layout`,
#'     `n_blocks`, `n_coefs` (vector of per-block `c`).
#'   - `subspace_debias`: present only when `control$subspace_debias` was set.
#'     `idx` are the corrected latent coordinates, `bands` the per-probed-index
#'     reliability table they were read from, `closure_added` what the coupling
#'     closure added, and `accept` the per-node Metropolis acceptance rate.
#'
#' @seealso [tulpa_laplace()] for the inner solve; [tulpa_nested_laplace()] for
#'   the analogous outer integration over spatial / temporal prior
#'   hyperparameters.
#'
#' @references
#' Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
#' Gaussian models by using integrated nested Laplace approximations.
#' \emph{JRSS-B} 71(2):319-392.
#' Lewandowski, Kurowicka & Joe (2009). Generating random correlation matrices
#' based on vines and extended onion method. \emph{Journal of Multivariate
#' Analysis} 100(9):1989-2001.
#' @examples
#' \donttest{
#' set.seed(1)
#' G <- 20L; per <- 12L; n <- G * per
#' grp <- rep(seq_len(G), each = per); x <- rnorm(n)
#' b <- cbind(rnorm(G, 0, 0.7), rnorm(G, 0, 0.5))     # random intercept + slope
#' eta <- -0.2 + 0.5 * x + b[grp, 1] + b[grp, 2] * x
#' y <- rbinom(n, 1L, plogis(eta))
#' re_term <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
#'                 correlated = TRUE)
#' fit <- tulpa_re_cov_nested(y, rep(1L, n), cbind(1, x), re_term,
#'                            family = "binomial")
#' fit$Sigma_mean        # marginalized RE covariance
#' }
#' @export
tulpa_re_cov_nested <- function(y, n_trials = NULL, X, re_terms,
                                family = "binomial", phi = 1.0, phi2 = NULL,
                                prior_sigma = c(3, 0.05), eta = 2,
                                hyperprior = c("flat", "pc_lkj"),
                                log_prior_theta = NULL,
                                beta_prior = NULL, offset = NULL, n_quad = 1L,
                                X_zi = NULL, zi_prior_sd = 2.5,
                                control = list()) {
  # Perf/numerical knobs live in `control = list()` (matching tulpa() /
  # tulpa_nested_laplace()); the signature carries only statistical arguments.
  tulpa_check_control(control, .CONTROL_KEYS$re_cov_nested, "tulpa_re_cov_nested")
  hyperprior <- match.arg(hyperprior)
  log_prior_theta <- .re_cov_resolve_hyperprior(hyperprior, log_prior_theta)
  integration <- match.arg(control$integration %||% "ccd", c("ccd", "grid"))
  n_per_axis  <- as.integer(control$n_per_axis %||% 5L)
  span        <- control$span %||% 3
  n_draws     <- as.integer(control$n_draws %||% 2000L)
  seed        <- control$seed
  diagnose_k  <- isTRUE(control$diagnose_k %||% TRUE)
  k_samples   <- as.integer(control$k_samples %||% .nl_diag("k_samples"))
  k_tail_points <- control$k_tail_points
  max_iter    <- as.integer(control$max_iter %||% 100L)
  tol         <- control$tol %||% 1e-8
  n_threads   <- as.integer(control$n_threads %||% 1L)
  checkpoint  <- control$checkpoint
  sd_cfg      <- .subspace_debias_config(control$subspace_debias)
  n_quad <- as.integer(n_quad)
  if (n_quad < 1L) stop("`n_quad` must be >= 1.", call. = FALSE)
  .seed_scoped(seed)

  core <- .re_cov_theta_fit(
    y = y, n_trials = n_trials, X = X, re_terms = re_terms,
    family = family, phi = phi, phi2 = phi2,
    prior_sigma = prior_sigma, eta = eta, log_prior_theta = log_prior_theta,
    beta_prior = beta_prior, n_quad = n_quad,
    max_iter = max_iter, tol = tol, n_threads = n_threads,
    caller = "tulpa_re_cov_nested", need_scale = TRUE,
    outer_maxit = as.integer(control$outer_maxit %||% 500L),
    offset = offset, X_zi = X_zi, zi_prior_sd = zi_prior_sd)

  layout          <- core$layout
  k               <- core$k
  n_trials        <- core$n_trials
  X               <- core$X
  # The whole fixed prefix of each node's mode: [beta | beta_zi] under zero
  # inflation, [beta] otherwise. Every node's H_beta is that size too, so the
  # per-node covariance and the draw synthesis follow from this one number.
  p_fix           <- core$p_fixed
  log_prior_theta <- core$log_prior_theta
  inner_logmarg   <- core$inner_logmarg
  inner_fit       <- core$inner_fit
  theta_hat       <- core$theta_hat
  L_scale         <- core$L_scale

  # --- subspace debias: which latent directions need the exact sampler? -----
  # One probe solve at the fitted MAP covariance, scored by the inner-layer
  # diagnostics, decides S once for the whole fit. Selecting
  # per node would make the correction's scope a function of the integration
  # design, and would leave nothing auditable to record; the MAP cell is where
  # the weight is and is the cell every other inner-layer probe in the engine
  # re-dispatches at.
  subspace <- NULL
  if (!is.null(sd_cfg) && n_quad > 1L) {
    stop("tulpa_re_cov_nested(): `control$subspace_debias` needs the ",
         "joint-field Laplace inner solve (`n_quad = 1`). The adaptive ",
         "Gauss-Hermite inner marginal integrates each group's random effects ",
         "out, so it exposes no conditional latent field to correct.",
         call. = FALSE)
  }
  if (!is.null(sd_cfg)) {
    probe_idx <- sd_cfg$probe %||% seq_len(p_fix)
    probe <- inner_fit(.re_cov_theta_to_L_list(theta_hat, layout),
                       compute_skew = is.null(sd_cfg$idx),
                       skew_idx = probe_idx,
                       joint_hessian = !identical(sd_cfg$closure, FALSE))
    if (is.null(probe)) {
      stop("tulpa_re_cov_nested(): the subspace-debias probe solve failed at ",
           "the fitted covariance, so no reliability band could be read. ",
           "Refit with control$subspace_debias = FALSE, or pin the set with ",
           "control$subspace_debias = list(idx = ...).", call. = FALSE)
    }
    subspace <- .subspace_select(probe, sd_cfg)
  }
  debias_arg <- if (!is.null(subspace) && length(subspace$idx))
    list(idx = subspace$idx, n_iter = sd_cfg$n_iter, warmup = sd_cfg$warmup,
         thin = sd_cfg$thin) else NULL

  # --- integration nodes in whitened theta-space ----------------------------
  if (integration == "ccd") {
    ccd   <- ccd_grid(k, f_0 = sqrt(k) * 1.1)
    z     <- ccd$z
    dnode <- ccd_weights(ccd)
  } else {
    ax    <- seq(-span, span, length.out = as.integer(n_per_axis))
    z     <- as.matrix(expand.grid(rep(list(ax), k)))
    dimnames(z) <- NULL
    dnode <- rep(1, nrow(z))                # uniform tensor-cell weight
  }
  theta_grid <- ccd_to_theta(z, theta_hat, L_scale)   # n_grid x k
  ng <- nrow(theta_grid)

  # --- node checkpoint/resume -----------------------------
  # Each CCD / grid node is one full inner Laplace solve. `checkpoint =
  # list(path =, resume =)` caches each completed node so a killed run resumes.
  # The node grid is deterministic given the fingerprint (which includes
  # theta_grid), so nodes are keyed by their integer index; the store is an
  # atomically-rewritten RDS, fingerprint-guarded against resuming onto a file
  # written for different data / layout / grid.
  # The joint-field inner solve also carries the per-group random-effect
  # posterior, which is what the per-group summary is marginalized from; it is a
  # node payload, so the fingerprint records it alongside the rest.
  re_cond <- isTRUE(core$re_conditional)
  ckpt <- .re_cov_node_checkpoint(checkpoint, fingerprint = list(
    y = as.numeric(y), n_trials = as.integer(n_trials), X = X,
    family = family, phi = phi,
    layout = lapply(layout, function(b) b[c("k", "nc", "full")]),
    theta_grid = theta_grid, max_iter = max_iter, tol = tol,
    beta_prior = beta_prior, n_quad = n_quad, re_cov = re_cond,
    debias = debias_arg))

  # --- evaluate inner marginal + derived quantities per cell ----------------
  # One FULL Laplace solve per node: the log-marginal feeds the integration
  # weight; the fixed-effect mode + marginal covariance feed the posterior-draw
  # synthesis. A failed / non-finite node keeps logm = -Inf (weight 0).
  #
  # The latent tail of each node's mode is E(b | y, Sigma_i) and its covariance
  # blocks are Var(b | y, Sigma_i), so the two together are the node's Gaussian
  # random-effect posterior. Retained per node (not collapsed here) because the
  # marginal per-group posterior is the WEIGHTED MIXTURE of them -- summarizing
  # each node first and averaging the summaries is the plug-in error the nested
  # path exists to avoid.
  n_re <- .re_cov_n_re(layout)
  logm           <- rep(-Inf, ng)
  lp_theta_nodes <- rep(NA_real_, ng)   # hyperparameter log-prior per node
  Sig_node_list  <- vector("list", ng)
  beta_nodes     <- matrix(NA_real_, ng, p_fix)
  beta_cov_nodes <- vector("list", ng)
  debias_nodes   <- vector("list", ng)
  debias_accept  <- rep(NA_real_, ng)
  re_nodes     <- if (re_cond) matrix(NA_real_, ng, n_re) else NULL
  re_var_nodes <- if (re_cond) matrix(NA_real_, ng, n_re) else NULL
  for (i in seq_len(ng)) {
    th     <- theta_grid[i, ]
    L_list <- .re_cov_theta_to_L_list(th, layout)
    Sig_node_list[[i]] <- lapply(L_list, function(L) L %*% t(L))
    key <- as.character(i)
    if (!is.null(ckpt) && ckpt$has(key)) {
      fit_i <- ckpt$get(key)
    } else {
      fit_i <- inner_fit(L_list, re_cov = re_cond, debias = debias_arg)
      if (!is.null(ckpt) && !is.null(fit_i) && !is.null(fit_i$mode) &&
          length(fit_i$log_marginal) == 1L && is.finite(fit_i$log_marginal)) {
        ckpt$save(key, fit_i)
      }
    }
    if (is.null(fit_i) || is.null(fit_i$mode) ||
        length(fit_i$log_marginal) != 1L || !is.finite(fit_i$log_marginal)) next
    lp_theta_nodes[i]  <- log_prior_theta(th)
    logm[i]            <- fit_i$log_marginal + lp_theta_nodes[i]
    beta_nodes[i, ]    <- fit_i$mode[seq_len(p_fix)]
    beta_cov_nodes[[i]] <-
      if (is.null(fit_i$H_beta)) NULL
      else tryCatch(solve(fit_i$H_beta), error = function(e) NULL)
    if (!is.null(debias_arg)) {
      debias_nodes[[i]]  <- fit_i$debias_draws
      debias_accept[i]   <- fit_i$debias_accept %||% NA_real_
    }
    if (re_cond && length(fit_i$mode) == p_fix + n_re) {
      re_nodes[i, ] <- fit_i$mode[p_fix + seq_len(n_re)]
      # Diagonals of the per-(term, group) covariance blocks, concatenated in the
      # block layout's own order -- the marginal variance of each random-effect
      # coefficient at this node.
      v <- .re_cov_block_variances(fit_i$cov_blocks, layout)
      if (!is.null(v)) re_var_nodes[i, ] <- v
    }
  }
  # Cell weight = design weight (CCD: corrected INLA; grid: uniform) times the
  # evaluated joint exp(logm); log-sum-exp shift for stability. Every failed /
  # non-finite node carries logm = -Inf; if all nodes fail, max(logm) = -Inf
  # would give NaN weights, so guard on the finite set and error loudly instead.
  lw <- log(dnode) + logm
  finite_lw <- lw[is.finite(lw)]
  if (length(finite_lw) == 0L) {
    stop("tulpa_re_cov_nested(): every integration node returned a non-finite ",
         "log-marginal (all inner Laplace solves failed). Check the data, the ",
         "hyperprior, and the Sigma initialisation.", call. = FALSE)
  }
  lw <- lw - max(finite_lw)
  w  <- exp(lw); w[!is.finite(w)] <- 0
  w  <- w / sum(w)

  # A weighted node whose latent tail was the wrong width, or whose covariance
  # blocks did not describe the layout, leaves NA in that row and the mixture
  # drops it. If that took out every weighted node the per-group posterior is
  # unreportable, and it says so under the same field the AGHQ path uses -- the
  # Sigma posterior is unaffected, so the fit still stands.
  ranef_note <- NULL
  if (re_cond && !any(is.finite(rowSums(re_nodes)) & w > 0)) {
    re_nodes <- NULL; re_var_nodes <- NULL
    ranef_note <- paste0(
      "no integration node returned a per-group random-effect block of the ",
      "expected width (", n_re, "), so the per-group posterior could not be ",
      "marginalized. The Sigma posterior is unaffected.")
  }

  # --- derived quantities, marginalized over the grid -----------------------
  # The CCD is a moment rule, so its median and interval come from the moments
  # it reproduces; the tensor grid's uniform cells discretize the density, so
  # its cumulative weights are a CDF and the weighted quantile is the summary.
  # `.nl_node_support()` is the one place that reads a producer name into a
  # support kind, here and on every grid consumer.
  summ <- .re_cov_derived_summary(Sig_node_list, w, layout,
                                  support = .nl_node_support(integration))
  posterior <- summ$posterior

  # --- plug-in MAP summary (for comparison) ---------------------------------
  map <- .re_cov_map_summary(theta_hat, layout)

  # --- fixed-effect posterior from the node mixture -------------------------
  beta_names <- c(colnames(X) %||% paste0("beta", seq_len(core$p_fix)),
                  if (core$p_zi > 0L)
                    colnames(core$X_zi) %||% paste0("zi_", seq_len(core$p_zi)))
  ds <- .re_cov_nested_beta_draws(beta_nodes, beta_cov_nodes, w,
                                  as.integer(n_draws), beta_names,
                                  debias_nodes = if (is.null(debias_arg)) NULL
                                                 else debias_nodes,
                                  debias_idx = subspace$idx)
  draws <- ds$draws
  # Random-effect coordinates the selector pulled into S are moved by the
  # Metropolis sampler at every node, so their posterior is no longer the
  # Gaussian mixture ranef() otherwise reports -- it is the sampled one, and
  # reporting the mixture for them would describe a distribution the fit did not
  # use. Recombined on the SAME node mixture the fixed effects were synthesized
  # on, so the two tables marginalize one weighted node set.
  re_debias_draws <- NULL
  re_debias_idx   <- NULL
  if (!is.null(debias_arg) && !is.null(ds) && re_cond && !is.null(re_nodes)) {
    cols <- which(subspace$idx > p_fix & subspace$idx <= p_fix + n_re)
    if (length(cols)) {
      pos <- as.integer(subspace$idx[cols]) - p_fix
      got <- .re_cov_debias_coord_draws(
        ds$picks, debias_nodes, re_nodes[, pos, drop = FALSE], cols)
      if (!is.null(got)) { re_debias_draws <- got; re_debias_idx <- pos }
    }
  }
  # Per-draw hyperparameter log-prior (the node's log_prior_theta), aligned
  # with the draw rows: the input power-scaling needs to reweight the
  # hyperparameter prior (tulpa_powerscale_sensitivity).
  hyper_lp_draws <- if (is.null(ds)) NULL else lp_theta_nodes[ds$picks]
  beta_mean <- if (is.null(draws)) {
    ok <- is.finite(rowSums(beta_nodes)) & w > 0
    if (any(ok)) colSums(w[ok] / sum(w[ok]) * beta_nodes[ok, , drop = FALSE])
    else rep(NA_real_, p_fix)
  } else colMeans(draws)
  names(beta_mean) <- beta_names

  # --- outer Pareto-k-hat: is the Gaussian grid proposal correctable? --------
  # Importance-sample the hyperparameter posterior with the same Gaussian
  # proposal (theta_hat, L_scale) the grid is placed with; k-hat gauges whether
  # the nested integration is trustworthy (< 0.7) or the hyperparameter
  # posterior is too skewed / heavy-tailed for the grid (>= 0.7). Run after the
  # draw synthesis and with the RNG state restored, so existing draws are
  # bit-for-bit unchanged whether or not the diagnostic is requested.
  # A decline says which one it was rather than a bare NA.
  pareto_k <- NA_real_; k_is_ess <- NA_real_; k_source <- NA_character_
  k_first  <- NA_real_
  k_declined <- if (!isTRUE(diagnose_k)) .k_decline_label(.k_decline("not_requested"))
                else .k_decline_label(.k_decline("no_varying_axis",
                                                 "no free covariance coordinate"))
  if (isTRUE(diagnose_k) && k > 0L) {
    kd <- .with_preserved_seed(tryCatch(
      .nested_outer_pareto_k(
        log_target = function(th) inner_logmarg(.re_cov_theta_to_L_list(th, layout)) +
          log_prior_theta(th),
        theta_hat = theta_hat, L_scale = L_scale, n_samples = k_samples,
        tail_points = k_tail_points),
      error = function(e) NULL))
    if (is.null(kd)) {
      k_declined <- .k_decline_label(.k_decline("degenerate_proposal",
                                                "the scorer errored"))
    } else {
      pareto_k <- kd$pareto_k; k_is_ess <- kd$is_ess
      k_source   <- kd$proposal_source %||% NA_character_
      k_first    <- kd$first_pass_k %||% NA_real_
      k_declined <- .k_reason_of(kd)
    }
  }

  .finalize_fit(list(
    posterior   = posterior,
    map         = map,
    Sigma_mean  = summ$Sigma_mean,
    beta        = beta_mean,
    draws       = draws,
    hyper_log_prior_draws = hyper_lp_draws,
    pareto_k    = pareto_k,
    pareto_k_is_ess = k_is_ess,
    pareto_k_declined = k_declined,
    pareto_k_proposal_source = k_source,
    pareto_k_first_pass = k_first,
    pareto_k_scope  = "outer (hyperparameter) Gaussian proposal",
    # Per-node random-effect posterior: mode and marginal variance of every
    # (term, group, coefficient), one row per integration node. ranef() mixes
    # them under `weights`. `ranef_unavailable` is the AGHQ inner marginal's
    # answer instead -- a stated reason, so the accessor reports why rather than
    # an empty table indistinguishable from a model with no random effects.
    re_nodes     = re_nodes,
    re_var_nodes = re_var_nodes,
    # The sampled per-group coordinates: one column per
    # random effect the subspace debias selected, `re_debias_idx` giving its
    # position within the random-effect block. ranef() reports these rows
    # empirically and the rest from the Gaussian mixture above, saying per row
    # which it used. Absent when the selected set contains no random effect.
    re_debias_draws = re_debias_draws,
    re_debias_idx   = re_debias_idx,
    ranef_unavailable = if (!re_cond) paste0(
      "the adaptive Gauss-Hermite inner marginal (n_quad > 1) integrates each ",
      "group's random effects out by quadrature instead of conditioning at ",
      "their mode, so this fit carries no per-group posterior. Refit with ",
      "n_quad = 1 (the joint-field Laplace inner solve) or with ",
      "control$re_cov = \"gibbs\", both of which report random effects.")
      else ranef_note,
    means       = beta_mean,
    param_names = beta_names,
    process_info = list(list(name = "fixed_effects", p = p_fix,
                             coef_names = beta_names)),
    n_samples   = if (is.null(draws)) 0L else nrow(draws),
    n_params    = p_fix,
    N           = length(y),
    theta_hat   = theta_hat,
    theta_grid  = theta_grid,
    weights     = w,
    log_marginal = logm,
    n_grid      = ng,
    layout      = layout,
    n_blocks    = length(layout),
    n_coefs     = vapply(layout, `[[`, integer(1), "nc"),
    # The escalation, recorded rather than implicit: which latent coordinates
    # were sampled exactly, which band each probed coordinate was read at, and
    # what the coupling closure added on top.
    subspace_debias = if (is.null(subspace)) NULL else list(
      idx = subspace$idx, bands = subspace$bands, closure_added = subspace$added,
      selected_by = subspace$selected_by, band_floor = sd_cfg$band,
      closure = sd_cfg$closure,
      n_iter = sd_cfg$n_iter, warmup = sd_cfg$warmup, thin = sd_cfg$thin,
      accept = debias_accept)
  ), backend = "re_cov_nested", n_fixed = p_fix, fixed_names = beta_names)
}
