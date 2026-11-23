# Nested-Laplace integration over random-effect covariances Sigma.
#
# Bias-2 fix (the `Marginalize Derived Quantities` principle): rather than
# report each Sigma at its mode (the plug-in MAP, biased low for skewed
# variance-component marginals at small G), integrate the Laplace marginal
# likelihood over the covariance coordinates and report a marginalized median
# and interval for every Sigma and its derived scale / correlation parameters.
#
# The integration is the joint nested-Laplace door: a covariance fit is a
# one-arm fit there, each covariance block its latent blocks (see
# `.re_cov_joint_blocks()`). This file holds the covariance layout, its
# coordinates and prior, the derived-quantity summaries shared with the Gibbs
# sampler and the joint door, and the map from the joint fit's cells back onto
# the layout.
#
# A model may carry several random-effect terms (e.g. `(1 + x | g) + (1 | h)`),
# each with its own covariance block, and a block may be CORRELATED (a full
# `Sigma`, `(1 + x | g)`) or UNCORRELATED (a diagonal `Sigma`, `(1 + x || g)`).
# A single-term model is the length-1 case of the same path.

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
      stop(sprintf(paste0("RE block %d (n_coefs = %d) requires `Z` (the ",
                          "n_obs x n_coefs RE design)."), m, nc),
           call. = FALSE)
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
.re_cov_block_logprior <- function(nc, full, prior_sigma = NULL, eta = NULL) {
  prior_sigma <- prior_sigma %||% .nl_scale_anchor()
  eta <- eta %||% .nl_hyperprior("lkj_eta")
  .check_pc_anchor_pair(prior_sigma, "prior_sigma", "RE covariance prior")
  .check_lkj_eta(eta, "eta", "RE covariance prior")
  U <- prior_sigma[1L]; alpha <- prior_sigma[2L]
  lambda     <- -log(alpha) / U
  log_lambda <- log(lambda)

  if (full) {
    jac_coef <- nc + 2L - seq_len(nc)        # (c + 2 - i) on each log L_ii
    lkj_norm <- if (nc > 1L) .lkj_log_normaliser(nc, eta) else 0
    function(th) {
      L      <- .re_logchol_to_L(th, nc)
      logLii <- log(diag(L))                 # = th diagonal entries
      sig    <- sqrt(rowSums(L^2))           # sigma_i = sqrt(Sigma_ii)
      logsig <- log(sig)
      lp <- sum(log_lambda - lambda * sig)   # PC on each marginal SD
      # LKJ: (eta - 1) log det(R), log det(R) = 2 sum log L_ii - 2 sum log sigma_i.
      if (nc > 1L) {
        lp <- lp + (eta - 1) * (2 * sum(logLii) - 2 * sum(logsig)) - lkj_norm
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
#' `p(R) = det(R)^(eta - 1) / c_d(eta)`. `eta = 1` is uniform over correlation
#' matrices; `eta > 1` concentrates toward the identity. The normalizing
#' constant is kept, so the prior is a density and a grid's evidence can be read
#' under it.
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
#' @param prior_sigma `c(U, alpha)` giving `P(sigma_i > U) = alpha`, applied
#'   independently to every marginal SD. `NULL` (the default) is `c(3, 0.01)`,
#'   the engine's prior on every field scale.
#' @param eta LKJ shape. `NULL` (the default) is 2. `eta = 1` is uniform on
#'   correlation matrices; larger values favour weaker correlations. Ignored for
#'   a diagonal block.
#' @param correlated `TRUE` (default) for a full covariance block (log-Cholesky
#'   coordinates, LKJ prior); `FALSE` for a diagonal / uncorrelated block
#'   (log-SD coordinates, no correlation). For `n_coefs = 1` the two coincide.
#'
#' @return A `function(theta)` returning the scalar log prior density in the
#'   block's integration coordinates, suitable for one block of the
#'   `log_prior_theta` argument of [tulpa_re_cov_nested()].
#' @seealso [tulpa_re_cov_nested()]
#' @examples
#' # Log prior of a correlated 2 x 2 block at the origin of its log-Cholesky
#' # coordinates (log sd_1, log sd_2, L_21):
#' lp <- re_cov_pc_lkj_prior(2)
#' lp(c(0, 0, 0))
#' # A diagonal block takes one log-SD coordinate per coefficient:
#' re_cov_pc_lkj_prior(2, correlated = FALSE)(c(0, 0))
#' @export
re_cov_pc_lkj_prior <- function(n_coefs, prior_sigma = NULL, eta = NULL,
                                correlated = TRUE) {
  c_re <- as.integer(n_coefs)
  if (c_re < 1L) stop("`n_coefs` must be >= 1.", call. = FALSE)
  .re_cov_block_logprior(c_re, isTRUE(correlated) && c_re > 1L, prior_sigma, eta)
}

# Log normalizing constant of the LKJ(eta) density on d x d correlation
# matrices (Lewandowski, Kurowicka & Joe 2009, section 3.2), the c_d in
# p(R) = det(R)^(eta - 1) / c_d:
#
#   c_d = 2^(sum_{k=1}^{d-1} (2 eta - 2 + d - k) (d - k))
#         * prod_{k=1}^{d-1} B(eta + (d - k - 1) / 2, eta + (d - k - 1) / 2)^(d - k)
#
# At d = 2 it is 2^(2 eta - 1) B(eta, eta), the integral of (1 - r^2)^(eta - 1)
# over (-1, 1). test-re-cov-prior.R integrates the density to one at d = 2, 3.
.lkj_log_normaliser <- function(d, eta) {
  k <- seq_len(d - 1L)
  a <- eta + (d - k - 1) / 2
  sum((2 * eta - 2 + d - k) * (d - k)) * log(2) +
    sum((d - k) * lbeta(a, a))
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
# Every nested-Laplace outer axis carries a proper default prior
# (`R/hyperprior_default.R`), the PC prior on a scale with the anchor
# `.NL_HYPERPRIOR` holds, and a random-effect covariance takes the same anchor
# through the PC + LKJ prior below (gcol33/tulpa#730).
#
# tulpa_eb() and tulpa_re_cov_nested() read one mode from `.eb_theta_fit()`
# (test-eb.R, "tulpa_eb() and tulpa_re_cov_nested() find the same theta_hat"),
# so the two must always resolve to the same hyperprior, never one per function.
# `hyperprior = "proper"` (the default for both) builds the PC + LKJ prior from
# `prior_sigma` / `eta` (re_cov_pc_lkj_prior()). `hyperprior = "flat"` is the
# zero function, unless the caller already supplied a `log_prior_theta` of their
# own: an improper prior, under which the nested integrator reads no evidence
# and the empirical-Bayes estimate is the unpenalized maximum marginal
# likelihood.
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
  .re_cov_derived_columns(nc, full, function(i, j)
    vapply(Sig_list, function(S) S[i, j], numeric(1)))
}

.re_cov_derived_columns <- function(nc, full, entry) {
  cols <- list(); dom <- character(0)
  put <- function(name, domain, values) {
    cols[[name]] <<- values
    dom[[name]]  <<- domain
  }
  for (i in seq_len(nc)) {
    put(sprintf("sigma_%d", i), "positive", sqrt(entry(i, i)))
  }
  if (full && nc > 1L) {
    for (i in seq_len(nc - 1L)) for (j in (i + 1L):nc) {
        denom <- cols[[sprintf("sigma_%d", i)]] * cols[[sprintf("sigma_%d", j)]]
        rho <- pmax(-1, pmin(1, entry(i, j) / denom))
      put(sprintf("rho_%d%d", i, j), "correlation",
          rho)
    }
  }
  if (full) {
    for (i in seq_len(nc)) for (j in i:nc) {
      put(sprintf("Sigma_%d%d", i, j),
          if (i == j) "positive" else "unbounded",
          entry(i, j))
    }
  } else {
    for (i in seq_len(nc)) {
      put(sprintf("Sigma_%d%d", i, i), "positive", entry(i, i))
    }
  }
  D <- do.call(cbind, cols)
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
  list(posterior = .re_cov_summarize_columns(D, w, support),
       Sigma_mean = .re_cov_sigma_mean(Sig_node_list, w, layout))
}

# The `w`-weighted mean of each block's Sigma over the cells / draws of
# `Sig_node_list`: a matrix for one block, a list named by block label for
# several.
.re_cov_sigma_mean <- function(Sig_node_list, w, layout) {
  M <- length(layout)
  Sig_mean <- lapply(seq_len(M), function(m) {
    Sig_m <- lapply(Sig_node_list, `[[`, m)
    Reduce(`+`, Map(function(S, wi) S * wi, Sig_m, w))
  })
  if (M == 1L) return(Sig_mean[[1L]])
  stats::setNames(Sig_mean, vapply(seq_len(M), function(m)
    .re_cov_block_label(layout[[m]], m), character(1)))
}

.re_cov_summarize_columns <- function(D, w, support) {
  support <- match.arg(support, .NL_SUPPORT_KINDS)
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
  data.frame(parameter = rownames(post), post,
                          row.names = NULL, check.names = FALSE)
}

# The derived covariance quantities of every free-covariance block an outer grid
# integrates (`mcar` / `miid`, whose Sigma = L L' is laid on the log-Cholesky
# columns `.hp_logchol_block_cols()` finds), summarized by
# `.re_cov_summarize_columns()` over the grid's cells and weights: covariance
# entries are formed from the Cholesky columns, and `sigma_i`, `rho_ij` and
# `Sigma_ij` are read from those entries before any weighting. Rows
# carry the block's column prefix (`b1.sigma_1`). NULL when the grid holds no
# complete block.
#
# `within` is the grid's within-cell construction (`.nl_within_cell_setup()`).
# Given it, each quantity is read ACROSS the cells rather than at their
# centres: the cells are expanded into the stratified points of
# `.nl_within_cell_points()`, the quantity is evaluated at every point and the
# points' weights are read as a sample of the distribution the axes' own
# intervals are read off. A nonlinear map of several axes laid at 1.25 SDs
# takes only a handful of distinct values at the cell centres, and its interval
# ends then sit where those values fall: on the two-arm MCAR fixture the
# correlation's lower end read 0.098 off a 5^3 grid against -0.025 on a dense
# one. `within` is built on the grid in the coordinates of `designs` (a
# `.hp_logchol_designs()` record, `.hp_grid_design_coords()`), the coordinates
# whose boxes the cells are; its points are mapped back to the log-Cholesky
# columns before any Sigma is formed.
.logchol_grid_derived_summary <- function(theta_grid, w,
                                          support = .NL_SUPPORT_KINDS,
                                          within = NULL, designs = list()) {
  if (is.null(theta_grid) || !length(w)) return(NULL)
  tg <- as.matrix(theta_grid)
  if (!is.null(within)) {
    pts <- .nl_within_cell_points(within, .hp_grid_design_coords(tg, designs), w)
    tg <- .hp_grid_design_coords(pts$theta, designs, inverse = TRUE)
    w <- pts$w
    support <- "sample"
  }
  cn <- colnames(tg) %||% character(0)
  lc <- cn[.hp_is_logchol_col(.axis_bare(cn))]
  parts <- list()
  for (pre in unique(.axis_label_prefix(lc))) {
    cols <- .hp_logchol_block_cols(cn, pre)
    if (is.null(cols)) next
    p <- as.integer(round((sqrt(8 * length(cols) + 1) - 1) / 2))
    idx <- .re_logchol_indices(p)
    L <- tg[, cols, drop = FALSE]
    for (i in seq_len(p)) L[, idx[i, i]] <- exp(L[, idx[i, i]])
    entry <- function(i, j) {
      v <- numeric(nrow(L))
      for (k in seq_len(min(i, j))) v <- v + L[, idx[i, k]] * L[, idx[j, k]]
      v
    }
    D <- .re_cov_derived_columns(p, TRUE, entry)
    post <- .re_cov_summarize_columns(D, w, support)
    post$parameter <- paste0(pre, post$parameter)
    parts[[length(parts) + 1L]] <- post
  }
  if (!length(parts)) return(NULL)
  do.call(rbind, parts)
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
    # The plain path samples exactly this node mixture, so its moments travel
    # with the draws (`.grid_mixture_moments()`); the debiased path below is a
    # different posterior and is not stamped.
    vars <- matrix(0, nrow(beta_nodes), p)
    for (k in which(ok)) vars[k, ] <- diag(beta_cov_nodes[[k]])
    modes <- beta_nodes
    modes[!ok, ] <- 0
    attr(out, "grid_mixture") <- list(weights = w2, modes = modes, vars = vars)
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

# Marginal-SD / partial-correlation coordinates ------------------------------
#
# A full block's log-Cholesky coordinates carry a funnel: where the data hold a
# slope SD weakly the density along L21 = 0 does not decay as log L22 -> -Inf.
# The marginal SDs and the canonical partial correlations (Lewandowski,
# Kurowicka & Joe 2009) parameterize the same Sigma without the neck: on
# `log sigma_i` the change of variables contributes `sigma_i^(i - 1)`, so the
# density decays as a scale shrinks, and on `atanh` of each partial correlation
# the LKJ factor holds it off +/-1. The joint door's free-covariance placement
# searches in these coordinates (`R/hyperprior_default.R`).
#
# Per full block the coordinates are `log sigma_1..c`, then `atanh z_ij` for the
# strictly-lower entries in the column-major order of the log-Cholesky packing.
# With W the Cholesky factor of the correlation matrix,
#   W_ij = z_ij * sqrt(r_ij),  r_ij = 1 - sum_{k<j} W_ik^2,  W_ii = sqrt(r_ii),
# and L = diag(sigma) W. Row i maps (log sigma_i, atanh z_i.) to
# (L_i., log L_ii) with log-Jacobian (i - 1) log sigma_i + (1/2) sum_{j<i} log
# r_ij; row 1 is the identity on log sigma_1.

# log(1 - tanh(y)^2) without cancellation at large |y|.
.log1m_tanh2 <- function(y) {
  a <- abs(y)
  2 * (log(2) - a - log1p(exp(-2 * a)))
}

.re_logchol_indices <- function(nc) {
  idx <- matrix(0L, nc, nc)
  idx[lower.tri(idx, diag = TRUE)] <- seq_len(nc * (nc + 1L) / 2L)
  idx
}

# Full-block placement coordinates (one row per point) -> log-Cholesky
# coordinates and the log-Jacobian of that map, per row.
.re_cpc_batch_to_logchol <- function(V, nc) {
  V <- as.matrix(V)
  n <- nrow(V)
  idx <- .re_logchol_indices(nc)
  yidx <- matrix(0L, nc, nc)
  pos <- 0L
  for (j in seq_len(nc - 1L)) for (i in (j + 1L):nc) {
    pos <- pos + 1L
    yidx[i, j] <- nc + pos
  }
  theta <- matrix(0, n, ncol(V))
  log_jac <- numeric(n)
  for (i in seq_len(nc)) {
    log_r <- numeric(n)
    for (j in seq_len(i - 1L)) {
      yy <- V[, yidx[i, j]]
      theta[, idx[i, j]] <- exp(V[, i] + 0.5 * log_r) * tanh(yy)
      log_jac <- log_jac + 0.5 * log_r + V[, i]
      log_r <- log_r + .log1m_tanh2(yy)
    }
    theta[, idx[i, i]] <- V[, i] + 0.5 * log_r
  }
  list(theta = theta, log_jac = log_jac)
}

# Inverse of .re_cpc_batch_to_logchol() for one point.
.re_logchol_block_to_cpc <- function(theta, nc) {
  as.numeric(.re_logchol_batch_to_cpc(matrix(theta, 1L), nc))
}

.re_logchol_batch_to_cpc <- function(theta, nc) {
  theta <- as.matrix(theta)
  n <- nrow(theta)
  idx <- .re_logchol_indices(nc)
  out <- matrix(0, n, ncol(theta))
  W <- theta
  for (i in seq_len(nc)) {
    cols <- idx[i, seq_len(i)]
    W[, idx[i, i]] <- exp(W[, idx[i, i]])
    out[, i] <- .5 * log(rowSums(W[, cols, drop = FALSE]^2))
    W[, cols] <- W[, cols, drop = FALSE] / exp(out[, i])
  }
  pos <- nc
  for (j in seq_len(nc - 1L)) for (i in (j + 1L):nc) {
    r <- 1 - rowSums(W[, idx[i, seq_len(j - 1L)], drop = FALSE]^2)
    z <- W[, idx[i, j]] / sqrt(pmax(r, .Machine$double.xmin))
    z <- pmax(pmin(z, 1 - .Machine$double.eps), -1 + .Machine$double.eps)
    pos <- pos + 1L
    out[, pos] <- atanh(z)
  }
  out
}

# The joint nested-Laplace door ---------------------------------------------
#
# A random-effect covariance fit is a one-arm fit on the joint door
# (tulpa_nested_laplace_joint()): the response is the arm, and every covariance
# block of the layout becomes latent blocks of it. A correlated block is one
# `miid` block over its coefficients, whose outer axes are the log-Cholesky
# coordinates of its Sigma; every coefficient of an uncorrelated block, and a
# scalar block, is a one-field `miid` block, whose one axis is its log SD. Each
# carries this block's PC + LKJ density as its `logchol_prior`, in those same
# coordinates. The joint door lays, places, screens and refines the outer grid
# and solves every cell; this file maps the cells back onto the covariance
# layout.

# The joint blocks of a covariance layout, and for each layout block the joint
# blocks it became (`owner`). The prior of each joint block is its layout
# block's PC + LKJ density (`hyperprior = "proper"`), the zero function
# (`"flat"`), or the caller's `log_prior_theta`, which states one density over
# the stacked coordinates and so can only be one joint block's prior.
.re_cov_joint_blocks <- function(layout, prior_sigma, eta, hyperprior,
                                 log_prior_theta, caller) {
  n_joint <- sum(vapply(layout, function(bl) if (bl$full) 1L else bl$nc,
                        integer(1)))
  if (!is.null(log_prior_theta) && n_joint > 1L) {
    stop(caller, "(): a `log_prior_theta` over the stacked coordinates of ",
         "several covariance blocks (or of an uncorrelated block's several ",
         "coefficients) couples outer axes the joint door integrates as ",
         "separate blocks. State each block's prior on its own coordinates: ",
         "fit with tulpa_nested_laplace_joint(), giving every `miid` block its ",
         "`logchol_prior`, or use `prior_sigma` / `eta` here.", call. = FALSE)
  }
  prior_of <- function(nc, full) {
    if (!is.null(log_prior_theta)) return(log_prior_theta)
    if (identical(hyperprior, "flat")) return(NULL)
    .re_cov_block_logprior(nc, full, prior_sigma, eta)
  }
  blocks <- list()
  owner <- vector("list", length(layout))
  add <- function(blk) {
    blocks[[length(blocks) + 1L]] <<- blk
    length(blocks)
  }
  for (m in seq_along(layout)) {
    bl <- layout[[m]]
    cols <- lapply(seq_len(bl$nc), function(j) list(as.numeric(bl$Z[, j])))
    one <- function(fields, nc, full) list(
      type = "miid", obs_idx = bl$idx, n_groups = bl$n_groups,
      n_fields = nc, field_weight = fields,
      logchol_prior = prior_of(nc, full))
    owner[[m]] <- if (bl$full) {
      add(one(cols, bl$nc, TRUE))
    } else {
      vapply(seq_len(bl$nc), function(j) add(one(cols[j], 1L, FALSE)),
             integer(1))
    }
  }
  list(blocks = blocks, owner = owner)
}

# The joint fit's outer cells in the layout's stacked coordinates: a full
# block's log-Cholesky packing, a diagonal block's log SDs, block after block.
.re_cov_joint_theta <- function(fit, layout, owner) {
  tg <- as.matrix(fit$theta_grid)
  cn <- colnames(tg)
  cols <- unlist(lapply(seq_along(layout), function(m) {
    unlist(lapply(owner[[m]], function(b) {
      cc <- .hp_logchol_block_cols(cn, .axis_prefix(b))
      if (is.null(cc)) {
        stop("the joint fit carries no log-Cholesky axes for block ", b,
             call. = FALSE)
      }
      cc
    }))
  }))
  out <- tg[, cols, drop = FALSE]
  colnames(out) <- NULL
  out
}

# The posterior table of the covariance layout, read from the joint fit's
# derived quantities (`hyper_derived`, each read across the cells of its own
# grid): a full block's rows are its joint block's, a diagonal block's are the
# scale and variance rows of its coefficients' one-field blocks, renumbered.
# With several layout blocks each row carries its block's label.
.re_cov_joint_posterior <- function(fit, layout, owner) {
  hd <- fit$hyper_derived
  row <- function(b, p) {
    r <- hd[hd$parameter == paste0(.axis_prefix(b), p), , drop = FALSE]
    if (nrow(r) != 1L) {
      stop("the joint fit reports no `", p, "` for block ", b, call. = FALSE)
    }
    r
  }
  parts <- lapply(seq_along(layout), function(m) {
    bl <- layout[[m]]
    b <- owner[[m]]
    rows <- if (bl$full) {
      p <- .re_cov_derived_names(bl$nc, TRUE)
      out <- do.call(rbind, lapply(p, function(pp) row(b, pp)))
      out$parameter <- p
      out
    } else {
      sig <- lapply(seq_len(bl$nc), function(j) row(b[j], "sigma_1"))
      var <- lapply(seq_len(bl$nc), function(j) row(b[j], "Sigma_11"))
      out <- do.call(rbind, c(sig, var))
      out$parameter <- .re_cov_derived_names(bl$nc, FALSE)
      out
    }
    if (length(layout) > 1L) {
      rows$parameter <- paste0(.re_cov_block_label(bl, m), ".", rows$parameter)
    }
    rows
  })
  out <- do.call(rbind, parts)
  rownames(out) <- NULL
  out[, c("parameter", "mean", "sd", "median", "ci_lo", "ci_hi")]
}

# The names of one block's derived quantities, in the order
# `.re_cov_derived_columns()` forms them.
.re_cov_derived_names <- function(nc, full) {
  colnames(.re_cov_derived_columns(nc, full, function(i, j) 1))
}

# Each cell's Gaussian over [fixed effects | random effects], the random effects
# ordered as the layout names them (block by block, group by group,
# coefficient by coefficient): `mode` (cells x latent) and, where the fit kept
# each cell's precision, `H` (one sparse matrix per cell). `cols` locates the
# fixed effects of every arm and the random effects in the joint latent vector.
.re_cov_joint_latent <- function(fit, cols, modes) {
  Qp <- fit$Q_csc_p_per_grid
  n_x <- ncol(fit$modes)
  H <- if (!is.null(Qp)) lapply(seq_along(Qp), function(k) {
    if (is.null(Qp[[k]])) return(NULL)
    Q <- Matrix::sparseMatrix(i = fit$Q_csc_i_per_grid[[k]] + 1L,
                              p = as.integer(Qp[[k]]),
                              x = fit$Q_csc_x_per_grid[[k]],
                              dims = c(n_x, n_x), symmetric = TRUE)
    Q[cols, cols]
  })
  list(mode = modes[, cols, drop = FALSE], H = H)
}

# The joint fit restricted to the cells it solved: a cell the cheap-pass screen
# left out carries no weight, no mode and no precision, so it leaves the grid,
# and the debias draws' cell index is carried over to the cells that remain.
.re_cov_joint_solved_cells <- function(fit) {
  ng <- length(fit$weights)
  drop <- fit$prune_mask
  if (length(drop) != ng || !any(drop)) return(fit)
  keep <- which(!drop)
  for (nm in c("theta_grid", "modes", "grid_mean_shift")) {
    if (is.matrix(fit[[nm]]) && nrow(fit[[nm]]) == ng) {
      fit[[nm]] <- fit[[nm]][keep, , drop = FALSE]
    }
  }
  for (nm in c("weights", "log_marginal", "log_hyperprior", "Q_csc_p_per_grid",
               "Q_csc_i_per_grid", "Q_csc_x_per_grid", "grid_hessians",
               "grid_modes")) {
    if (length(fit[[nm]]) == ng) fit[nm] <- list(fit[[nm]][keep])
  }
  if (!is.null(fit$draw_nodes)) fit$draw_nodes <- match(fit$draw_nodes, keep)
  for (nm in c("cell_draws", "cell_modes")) {
    v <- fit$subspace_debias[[nm]]
    if (is.list(v) && length(v) == ng) fit$subspace_debias[[nm]] <- v[keep]
    if (is.matrix(v) && nrow(v) == ng) {
      fit$subspace_debias[[nm]] <- v[keep, , drop = FALSE]
    }
  }
  fit
}

# A `control$subspace_debias` request with its latent positions stated in this
# fit's order, restated in the joint latent vector's (`perm`); the probe is not a
# joint-door setting and is passed as its `skew_idx` instead.
.re_cov_debias_to_joint <- function(cfg, perm) {
  if (!is.list(cfg)) return(cfg)
  if (!is.null(cfg$idx)) cfg$idx <- perm[as.integer(cfg$idx)]
  cfg$probe <- NULL
  cfg
}

# The joint fit's subspace-debias record with every latent position read back
# into this fit's order.
.re_cov_debias_from_joint <- function(rec, perm) {
  if (is.null(rec)) return(rec)
  back <- function(j) match(as.integer(j), perm)
  rec$idx <- back(rec$idx)
  if (length(rec$closure_added)) rec$closure_added <- back(rec$closure_added)
  if (is.data.frame(rec$bands) && nrow(rec$bands)) {
    rec$bands$idx <- back(rec$bands$idx)
  }
  o <- order(rec$idx)
  rec$idx <- rec$idx[o]
  if (is.list(rec$cell_draws)) {
    rec$cell_draws <- lapply(rec$cell_draws, function(D)
      if (is.matrix(D) && ncol(D) == length(o)) D[, o, drop = FALSE] else D)
  }
  rec
}

# Which cells the joint fit's mean shift centres (`.nested_grid_shift()`): per
# cell `"applied"`, else the reason the cell stays at its mode.
.re_cov_joint_mean_shift_record <- function(fit) {
  S <- .nested_grid_shift(fit)
  why <- fit$grid_mean_shift_declined
  ok <- if (is.matrix(S)) is.finite(rowSums(S)) & rowSums(abs(S)) > 0
        else rep(FALSE, nrow(fit$modes))
  ifelse(ok, "applied",
         if (length(why) == 1L && !is.na(why)) why else "not_formed")
}

#' Nested-Laplace integration over random-effect covariances
#'
#' @description
#' For one or more random-effects terms (e.g. `(1 + x | g)`, `(1 + x || g)`, or
#' several terms together), integrate the Laplace marginal likelihood over the
#' random-effect covariances `Sigma` instead of fixing them at point estimates.
#' Reports weighted posterior summaries (mean, SD, median, 2.5\%/97.5\%) of every
#' `Sigma` and its derived scale (`sigma_i`) and correlation (`rho_ij`)
#' parameters, marginalizing the joint posterior over an outer grid.
#'
#' This corrects the plug-in-MAP ("summary") bias: the mode of a skewed
#' variance-component marginal is biased low relative to its median, so the
#' headline summary should be the marginalized median, not the mode.
#'
#' @details
#' The fit is a one-arm fit on the joint nested-Laplace door
#' ([tulpa_nested_laplace_joint()]): the response is the arm and each covariance
#' block is latent blocks of it. A correlated block (`(1 + x | g)`) is one
#' `miid` block, a full `Sigma = L L'` whose outer axes are the log-Cholesky
#' coordinates of its lower Cholesky factor (the log-diagonal and the
#' strictly-lower entries of `L`, `c(c+1)/2` values for a `c`-coefficient
#' block), which keep `Sigma` positive definite for every coordinate. Each
#' coefficient of an uncorrelated block (`(1 + x || g)`), and a scalar
#' `(1 | g)` term, is a one-field `miid` block whose one axis is its log
#' standard deviation. The joint door lays the outer grid at the posterior,
#' screens and refines it, and solves the inner Laplace approximation at every
#' cell; its settings reach it through `control`.
#'
#' Each cell carries weight proportional to its measure times
#' `exp(log_marginal(Sigma_k) + log_prior_theta(theta_k))`. The reported median
#' and 2.5\%/97.5\% interval of each derived quantity are read across the
#' cells of the grid (the joint door's within-cell read); the `mean` and `sd`
#' columns are the weighted moments.
#'
#' By default (`hyperprior = "proper"`) each block carries the
#' weakly-informative PC + LKJ hyperprior of [re_cov_pc_lkj_prior()] (PC prior
#' on each marginal SD via `prior_sigma`, LKJ prior on each correlated block's
#' correlation matrix via `eta`), in the block's own coordinates with the exact
#' change-of-variables Jacobian. `hyperprior = "flat"` makes it the zero
#' function, flat in log(theta); that prior is improper, so the fit then reports
#' no evidence (`log_evidence_declined_axes` names the outer axes without a
#' proper density). Its support is a prior choice: each block's coordinates are
#' integrated flat over the span the placed nodes tile (`axis_support`), and
#' refinement resolves that span without following the posterior past it,
#' since a marginal likelihood that stays level as a standard deviation goes to
#' zero leaves the flat posterior's tail undecayed. A custom `log_prior_theta`
#' replaces either default when the model has exactly one outer block (one correlated term, or one scalar term);
#' a density over the stacked coordinates of several blocks couples axes the
#' joint door integrates as separate blocks and is refused, and each block's
#' prior is then stated as its `logchol_prior` on
#' [tulpa_nested_laplace_joint()].
#'
#' `theta_hat` and `map` are the empirical-Bayes estimate: the mode of the same
#' log marginal plus log prior in the stacked coordinates, the estimate
#' [tulpa_eb()] reports on the same data. A mode is not invariant under a change
#' of coordinates, so `theta_hat` is the mode in log-Cholesky coordinates (log
#' standard deviations and the strictly lower Cholesky entries), and `map` is
#' the covariance read off at that point, not a mode in `(sigma, rho)`. Where
#' `theta_hat` sits on the boundary of the covariance (a standard deviation at
#' the lower end of its search bracket, the case [tulpa_eb()] warns about as a
#' singular fit), the hit is recorded in `theta_hat_boundary` without a
#' warning: the posterior summaries integrate over the covariance and do not
#' rest on that one point.
#'
#' @param y,n_trials,X,family,phi The response, trials, fixed-effect design,
#'   family and dispersion of the one arm. `n_trials = NULL` defaults to 1
#'   (binary / single-trial).
#' @param phi2 Optional second dispersion: the Student-t degrees of freedom
#'   (`family = "t"`, default 4 when `NULL`) or the Tweedie variance power
#'   (`family = "tweedie"`, required). A `phi2` supplied for any other family
#'   errors rather than being ignored. It is conditioned on: the integration is
#'   over the random-effect covariances, not over `phi2`.
#' @param re_terms Either a single random-effect term or a list of them. Each
#'   term is a list with `idx` (1-based group index per observation),
#'   `n_groups`, `n_coefs` (`c`), `Z` (the `n_obs x c` RE design, e.g.
#'   `cbind(1, x)` for `(1 + x | g)`; only required when `c > 1`), and
#'   `correlated` (`TRUE` for a full `Sigma`, `FALSE` for a diagonal one;
#'   defaults to `TRUE`). An optional `label` / `group_var` names the block in
#'   the output. Any `L` / `cov` / `sigma` field is ignored -- `Sigma` is what
#'   this function integrates over.
#' @param prior_sigma,eta Hyperparameters of the PC + LKJ prior used when
#'   `hyperprior = "proper"` (see [re_cov_pc_lkj_prior()]):
#'   `prior_sigma = c(U, alpha)` with `P(sigma_i > U) = alpha` (`NULL`, the
#'   default, is `c(3, 0.01)`) and LKJ shape `eta` (`NULL` is 2). Ignored when
#'   `hyperprior = "flat"` or `log_prior_theta` is supplied.
#' @param hyperprior `"proper"` (default) or `"flat"`, see Details. Ignored
#'   when `log_prior_theta` is supplied.
#' @param log_prior_theta Optional `function(theta)` returning a scalar log
#'   prior density on the coordinates of a model with one outer block,
#'   overriding `hyperprior`. Default `NULL`, which defers to `hyperprior`.
#' @param beta_prior Optional Gaussian prior on the fixed effects
#'   (`list(mean, sd)`). `NULL` (default) takes the engine default
#'   `N(0, 2.5^2)`.
#' @param offset Optional observation-level offset on the linear predictor
#'   (length `length(y)`), e.g. `log(exposure)` for a rate model.
#' @param weights Optional non-negative observation weights (length
#'   `length(y)`), each scaling its row's log-likelihood, so the covariance is
#'   integrated under the weighted likelihood.
#' @param n_quad Quadrature order for the inner marginal. `1` (default) is the
#'   Laplace approximation at the joint mode. `> 1` integrates each group's
#'   random effects by `n_quad`-point adaptive Gauss-Hermite quadrature (the
#'   inner marginal of [tulpa_re_aghq()]) and the fixed effects by a Laplace
#'   step over the result, reducing the small-cluster variance attenuation for
#'   binary / low-count data. It needs a single shared grouping factor; with
#'   crossed RE terms it errors. The groups are integrated out, so such a fit
#'   reports `ranef_unavailable` in place of per-group effects.
#' @param X_zi Optional zero-inflation design matrix (`length(y)` rows), making
#'   the model a two-process mixture: each observation is a structural zero with
#'   probability `plogis(X_zi beta_zi)` and otherwise follows `family`. Paired
#'   with a zero-truncated family it is the hurdle model. The random effects
#'   enter the count predictor only. The ZI coefficients are reported alongside
#'   the count ones in `coef()` / `vcov()`, so the fixed block is
#'   `ncol(X) + ncol(X_zi)` wide.
#' @param zi_prior_sd Prior SD on `beta_zi`, keeping the logit identified where
#'   a level carries no zeros (the likelihood alone would send it to `-Inf`).
#'   Ignored when `X_zi` is `NULL`.
#' @param control A named list of numerical / tuning knobs (statistical
#'   arguments stay in the signature above). Recognized entries:
#'   \itemize{
#'     \item Every control of [tulpa_nested_laplace_joint()] except `n_quad`
#'       (an argument here) and its retention switches (`store_Q`,
#'       `keep_grid_hessians`, `hessian`), passed to the joint door:
#'       `integration`, the screen (`prune`, `prune_tol`, ...), the
#'       refinement passes, `diagnose_k` / `k_samples`, `skew_correct`,
#'       `subspace_debias`, `checkpoint`, `max_iter`, `tol`, `n_threads`, ...
#'       `skew_correct` (`TRUE`) centres each
#'       cell's joint Gaussian over the fixed and random effects at the
#'       leading-order posterior mean, `mode + Sigma A' (l''' * s) / 2` (Rue,
#'       Martino & Chopin 2009, Sec 3.2.3), instead of at the mode; the fixed
#'       effects, [ranef()], [posterior_predict()] and the WAIC / LOO criteria
#'       read the cell centres. Which cells were centred is recorded in
#'       `mean_shift`. `checkpoint = list(path =, resume =)` caches each solved
#'       cell so a killed run resumes.
#'     \item `n_draws`: posterior draws of the fixed effects synthesized from the
#'       cell mixture (default 2000), exposed as `draws` for the generic
#'       `tulpa_fit` methods.
#'     \item `seed`: optional integer seed for the draw synthesis.
#'     \item `outer_maxit`: round budget of the mode-find behind `theta_hat`
#'       (default 500). Exhausting it warns.
#'   }
#'
#' @return A list with:
#'   - `posterior`: data frame with one row per parameter and columns `mean`,
#'     `sd`, `median`, `ci_lo`, `ci_hi`. Parameter names are `sigma_i`, `rho_ij`,
#'     `Sigma_ij` for a single block, prefixed by the block label
#'     (`g.sigma_1`, ...) when there are several blocks. Diagonal blocks report
#'     no `rho`.
#'   - `map`: the plug-in summary at `theta_hat` (a single `list(Sigma, sigma,
#'     rho)` for one block, or a named list of them).
#'   - `Sigma_mean`: the weighted posterior mean of `Sigma` (a matrix for one
#'     block, or a named list of matrices).
#'   - `beta`, `draws`, `means`, `param_names`, `process_info`: the fixed-effect
#'     posterior from the cell mixture (drives `coef`/`confint`/`vcov`/`summary`).
#'   - `re_nodes`, `re_var_nodes`: the per-group random-effect posterior at each
#'     cell -- mean and marginal variance of every (block, group, coefficient),
#'     one row per cell. [ranef()] reports the `weights`-mixture of them. `NULL`
#'     at `n_quad > 1`, whose fit carries `ranef_unavailable` (the reason)
#'     instead.
#'   - `re_debias_draws`, `re_debias_idx`: present when the subspace debias
#'     sampled a random-effect coordinate: its draws on the cell mixture the
#'     fixed-effect draws use, and its position within the random-effect block.
#'     [ranef()] reports those rows empirically and the rest from the Gaussian
#'     mixture.
#'   - `latent_nodes`, `draw_nodes`: each cell's joint Gaussian over the fixed
#'     and random effects (`mode`, precision `H`), and the cell each row of
#'     `draws` was taken from. [posterior_predict()] and the WAIC / LOO criteria
#'     draw each row's random effects from that cell's conditional given the
#'     row's fixed effects. `NULL` at `n_quad > 1`.
#'   - `mean_shift`: per cell, `"applied"` where `control$skew_correct` centred
#'     the cell at the leading-order mean, otherwise the reason it stayed at the
#'     mode. `NULL` when the centring was off.
#'   - `theta_hat_boundary`: one row per log-scale coordinate of `theta_hat`
#'     at either end of its search bracket (`block`, `coef`, `end`, `sigma`, as
#'     [tulpa_eb()]'s `boundary`); no rows when none is.
#'   - `theta_hat`, `theta_grid` (the cells in the stacked coordinates),
#'     `weights`, `log_marginal` (log marginal plus log prior per cell),
#'     `log_hyperprior`, `log_evidence`, `log_evidence_declined_axes`,
#'     `axis_support` (the span integrated on each outer axis, keyed by the
#'     joint door's axis label), `n_grid`, `layout`, `n_blocks`,
#'     `n_coefs` (vector of per-block `c`), and the outer Pareto k-hat fields of
#'     the joint fit (`pareto_k`, ...).
#'   - `subspace_debias`: present only when `control$subspace_debias` was set.
#'
#' @seealso [tulpa_nested_laplace_joint()] for the door the fit runs on;
#'   [tulpa_eb()] for the plug-in estimate; [tulpa_re_cov_gibbs()] for the exact
#'   sampler.
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
                                prior_sigma = NULL, eta = NULL,
                                hyperprior = c("proper", "flat"),
                                log_prior_theta = NULL,
                                beta_prior = NULL, offset = NULL,
                                weights = NULL, n_quad = 1L,
                                X_zi = NULL, zi_prior_sd = 2.5,
                                control = list()) {
  caller <- "tulpa_re_cov_nested"
  tulpa_check_control(control, .CONTROL_KEYS$re_cov_nested, caller)
  family <- .canonical_family(family)
  .family_or_stop(family)
  hyperprior <- .hp_choice(match.arg(hyperprior))
  prior_proper <- !is.null(log_prior_theta) || identical(hyperprior, "proper")
  n_draws <- as.integer(control$n_draws %||% 2000L)
  n_quad  <- .check_n_quad(n_quad)
  if (!is.matrix(X)) X <- as.matrix(X)
  .require_fixed_effects(X, caller)
  n_trials <- .validate_glm_design(y, X, n_trials, caller)$n_trials
  re_terms <- .as_re_terms_list(re_terms)
  layout <- .re_cov_block_layout(re_terms, length(y))
  jb <- .re_cov_joint_blocks(layout, prior_sigma, eta, hyperprior,
                             log_prior_theta, caller)
  .seed_scoped(control$seed)

  # theta_hat, the empirical-Bayes mode of the same objective; it also
  # validates the dispersions, the zero-inflation design and the AGHQ
  # requirements before the grid is solved.
  eb <- .eb_theta_fit(
    y = y, n_trials = n_trials, X = X, re_terms = re_terms,
    family = family, phi = phi, phi2 = phi2,
    prior_sigma = prior_sigma, eta = eta,
    log_prior_theta = .re_cov_resolve_hyperprior(hyperprior, log_prior_theta),
    beta_prior = beta_prior, n_quad = n_quad,
    max_iter = as.integer(control$max_iter %||% 100L),
    tol = control$tol %||% 1e-8,
    n_threads = as.integer(control$n_threads %||% 1L),
    caller = caller, outer_maxit = as.integer(control$outer_maxit %||% 500L),
    offset = offset, X_zi = X_zi, zi_prior_sd = zi_prior_sd,
    weights = weights, warn_boundary = FALSE)
  theta_hat <- eb$theta_hat
  phi2 <- .phi2_or_stop(family, phi2)

  p <- ncol(X)
  bp <- .normalize_beta_prior(beta_prior, p)
  responses <- list(y = Filter(Negate(is.null), list(
    y = y, X = X, family = family, phi = phi, n_trials = n_trials,
    offset = offset, beta_prior_mean = bp$mean,
    beta_prior_prec = if (!is.null(bp)) 1 / bp$sd^2,
    weights = weights, phi2 = phi2)))
  blocks <- jb$blocks
  if (!is.null(X_zi)) {
    responses <- .joint_arm_zi(responses, as.matrix(X_zi), zi_prior_sd)
    blocks <- lapply(blocks, .joint_zi_block, n_obs = length(y))
  }
  jctl <- control[intersect(names(control), .CONTROL_KEYS$nested_laplace_joint)]
  jctl$n_quad <- n_quad
  jctl$progress <- jctl$progress %||% FALSE
  # Latent positions are stated in this fit's order, [fixed | random effects
  # block by block, group by group, coefficient by coefficient]; `perm` takes
  # each to its position in the joint latent vector.
  p_fixed <- p + if (is.null(X_zi)) 0L else ncol(as.matrix(X_zi))
  starts <- p_fixed + cumsum(c(0L, vapply(blocks, function(b)
    as.integer(b$n_groups) * as.integer(b$n_fields), integer(1))))
  re_map <- .joint_arm_re_map(
    list(arm_layout = list(block_start = starts[seq_along(blocks)])), blocks,
    jb$owner, lapply(layout, function(bl)
      list(n_groups = bl$n_groups, n_coefs = bl$nc)))
  perm <- c(seq_len(p_fixed), re_map$cols)
  jctl$subspace_debias <- .re_cov_debias_to_joint(jctl$subspace_debias, perm)
  if (is.list(control$subspace_debias) && !is.null(control$subspace_debias$probe)) {
    jctl$skew_idx <- perm[as.integer(control$subspace_debias$probe)]
  }
  fit <- tulpa_nested_laplace_joint(
    responses, prior = blocks, hyperprior = hyperprior,
    cell_coupling = if (is.null(X_zi)) "separable" else .JOINT_ZI_COUPLING,
    control = jctl)
  if (!identical(as.integer(fit$arm_layout$block_start),
                 as.integer(starts[seq_along(blocks)]))) {
    stop(caller, "(): the joint fit laid its latent blocks at other positions ",
         "than the covariance layout implies.", call. = FALSE)
  }
  fit <- .re_cov_joint_solved_cells(fit)
  fit$subspace_debias <- .re_cov_debias_from_joint(fit$subspace_debias, perm)

  theta_grid <- .re_cov_joint_theta(fit, layout, jb$owner)
  w <- as.numeric(fit$weights)
  ng <- nrow(theta_grid)
  lp_theta <- as.numeric(fit$log_hyperprior)
  logm <- as.numeric(fit$log_marginal)
  p_fix <- as.integer(fit$n_fixed)
  beta_names <- c(colnames(X) %||% paste0("beta", seq_len(p)),
                  if (!is.null(X_zi))
                    colnames(X_zi) %||% paste0("zi_", seq_len(ncol(X_zi))))

  # A Laplace inner solve conditions on the random effects, so every cell
  # carries their Gaussian; the AGHQ inner marginal integrates them out.
  re_cond <- !is.character(fit$ranef_unavailable) && is.matrix(fit$modes) &&
    ncol(fit$modes) > p_fix
  # Each cell is centred at its mode plus the leading-order mean shift the
  # joint fit admits; a selected subspace is drawn about the mode.
  mean_shift_on <- re_cond && !length(fit$subspace_debias$idx) &&
    isTRUE(control$skew_correct %||% .nl_diag("skew_correct"))
  if (!mean_shift_on) fit$grid_mean_shift <- NULL
  modes <- if (re_cond) .nested_grid_latent_means(fit)
  n_re <- .re_cov_n_re(layout)
  rc <- latent <- NULL
  if (re_cond) {
    rc <- .joint_re_cells(fit, re_map)
    latent <- .re_cov_joint_latent(fit, c(seq_len(p_fix), re_map$cols), modes)
  }

  # The fixed effects: the draws the subspace debias took where it ran, else
  # the cell mixture of each cell's Gaussian at its centre.
  beta_nodes <- matrix(NA_real_, ng, p_fix)
  beta_cov_nodes <- vector("list", ng)
  for (k in seq_len(ng)) {
    Hk <- fit$grid_hessians[[k]]
    if (is.null(Hk)) next
    beta_nodes[k, ] <- if (re_cond) modes[k, seq_len(p_fix)] else fit$grid_modes[[k]]
    beta_cov_nodes[k] <- list(tryCatch(.spd_inverse(Hk), error = function(e) NULL))
  }
  joint_draws <- fit$joint_latent_draws
  exact_applied <- identical(fit$weights_source, "subspace_is") &&
    is.matrix(joint_draws)
  # The fast debias moves its selected coordinates by a per-cell Metropolis
  # sample on the conditional-mean surface (`cell_draws`, offsets from the
  # cell's mode); the rest of each cell's Gaussian follows them.
  sdb <- fit$subspace_debias
  fast <- !exact_applied && is.list(sdb$cell_draws) && length(sdb$idx) > 0L
  ds <- if (exact_applied) {
    list(draws = `colnames<-`(fit$draws, beta_names), picks = fit$draw_nodes)
  } else {
    .re_cov_nested_beta_draws(beta_nodes, beta_cov_nodes, w, n_draws, beta_names,
                              debias_nodes = if (fast) sdb$cell_draws,
                              debias_idx = if (fast) sdb$idx)

  }
  draws <- ds$draws
  beta_mean <- if (is.null(draws)) {
    ok <- is.finite(rowSums(beta_nodes)) & w > 0
    colSums(w[ok] / sum(w[ok]) * beta_nodes[ok, , drop = FALSE])
  } else (.grid_mixture_moments(draws)$mean %||% colMeans(draws))
  names(beta_mean) <- beta_names
  re_debias_draws <- re_debias_idx <- NULL
  if (exact_applied && re_cond) {
    re_debias_draws <- joint_draws[, re_map$cols, drop = FALSE]
    re_debias_idx <- seq_len(n_re)
  } else if (fast && re_cond && !is.null(rc) && !is.null(ds)) {
    cols <- which(sdb$idx > p_fix)
    pos <- sdb$idx[cols] - p_fix
    if (length(cols)) {
      got <- .re_cov_debias_coord_draws(ds$picks, sdb$cell_draws,
                                        rc$mean[, pos, drop = FALSE], cols)
      if (!is.null(got)) {
        o <- order(pos)
        re_debias_draws <- got[, o, drop = FALSE]
        re_debias_idx <- pos[o]
      }
    }
  }

  Sig_nodes <- lapply(seq_len(ng), function(i)
    lapply(.re_cov_theta_to_L_list(theta_grid[i, ], layout), tcrossprod))

  .finalize_fit(list(
    inference_mode = "structured",
    inference_tier = 2L,
    posterior   = .re_cov_joint_posterior(fit, layout, jb$owner),
    map         = .re_cov_map_summary(theta_hat, layout),
    Sigma_mean  = .re_cov_sigma_mean(Sig_nodes, w, layout),
    beta        = beta_mean,
    draws       = draws,
    joint_latent_draws = if (exact_applied)
      joint_draws[, c(seq_len(p_fix), re_map$cols), drop = FALSE],
    re          = if (exact_applied) re_debias_draws,
    draws_theta = if (exact_applied) theta_grid[ds$picks, , drop = FALSE],
    weights_source = fit$weights_source %||% "laplace_grid",
    target      = fit$target %||%
      "Laplace outer-cell weights with Gaussian latent conditionals",
    draws_kind  = if (exact_applied) "mcmc_mixture" else "iid",
    hyper_log_prior_draws = if (!is.null(ds)) lp_theta[ds$picks],
    pareto_k    = fit$pareto_k,
    pareto_k_is_ess = fit$pareto_k_is_ess,
    pareto_k_declined = fit$pareto_k_declined,
    pareto_k_proposal_source = fit$pareto_k_proposal_source,
    pareto_k_first_pass = fit$pareto_k_first_pass,
    pareto_k_tail_points = fit$pareto_k_tail_points,
    pareto_k_scope = fit$pareto_k_scope,
    re_nodes     = rc$mean,
    re_var_nodes = rc$var,
    re_debias_draws = re_debias_draws,
    re_debias_idx   = re_debias_idx,
    latent_nodes = if (re_cond) lapply(seq_len(ng), function(k)
      list(mode = latent$mode[k, ], H = latent$H[[k]])),
    draw_nodes   = if (!is.null(ds)) ds$picks,
    mean_shift   = if (mean_shift_on) .re_cov_joint_mean_shift_record(fit),
    ranef_unavailable = if (!re_cond) .AGHQ_RANEF_UNAVAILABLE,
    means        = beta_mean,
    param_names  = beta_names,
    process_info = list(list(name = "fixed_effects", p = p_fix,
                             coef_names = beta_names)),
    n_samples   = if (is.null(draws)) 0L else nrow(draws),
    n_params    = p_fix,
    N           = length(y),
    theta_hat   = theta_hat,
    theta_hat_boundary = eb$boundary,
    theta_grid  = theta_grid,
    weights     = w,
    log_marginal = logm,
    log_hyperprior = lp_theta,
    log_evidence = if (prior_proper) fit$log_evidence else NA_real_,
    log_evidence_declined = if (prior_proper) NA_character_
                            else "improper_hyperprior",
    log_evidence_declined_axes = fit$log_evidence_declined_axes,
    axis_support = fit$axis_support,
    n_grid      = ng,
    layout      = layout,
    n_blocks    = length(layout),
    n_coefs     = vapply(layout, `[[`, integer(1), "nc"),
    subspace_debias = fit$subspace_debias
  ), backend = "re_cov_nested", n_fixed = p_fix, fixed_names = beta_names,
     data = list(y = y, n_trials = n_trials, model_matrix = X,
                 family = family, offset = offset, phi = phi, phi2 = phi2))
}
