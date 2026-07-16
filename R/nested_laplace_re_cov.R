# Nested-Laplace integration over random-effect covariances Sigma.
#
# Bias-2 fix (the `Marginalize Derived Quantities` principle): rather than
# report each Sigma at its mode (the plug-in MAP, biased low for skewed
# variance-component marginals at small G), integrate the Laplace marginal
# likelihood over the joint Sigma-grid and report weighted quantiles of every
# Sigma and its derived scale / correlation parameters.
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

# Node checkpoint/resume for the CCD / grid integration (gcol33/tulpa#50).
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

# Derived quantities of one block's Sigma draws as a matrix: one row per Sigma,
# named columns sigma_i (= sqrt(Sigma_ii)); for a `full` block also rho_ij
# (i<j) and the upper-triangular Sigma_ij; for a diagonal block only the
# diagonal Sigma_ii (off-diagonals are structurally zero, so no rho). Each value
# is a transform of a SINGLE Sigma (never of summarized components), so this is
# the per-cell / per-draw input the weighted-quantile summary and the
# posterior-draw synthesis consume (`Marginalize Derived Quantities`).
.re_cov_derived_matrix <- function(Sig_list, nc, full = TRUE) {
  cols <- list()
  for (i in seq_len(nc)) {
    cols[[sprintf("sigma_%d", i)]] <-
      vapply(Sig_list, function(S) sqrt(S[i, i]), numeric(1))
  }
  if (full && nc > 1L) {
    for (i in seq_len(nc - 1L)) for (j in (i + 1L):nc) {
      cols[[sprintf("rho_%d%d", i, j)]] <-
        vapply(Sig_list, function(S) S[i, j] / sqrt(S[i, i] * S[j, j]),
               numeric(1))
    }
  }
  if (full) {
    for (i in seq_len(nc)) for (j in i:nc) {
      cols[[sprintf("Sigma_%d%d", i, j)]] <-
        vapply(Sig_list, function(S) S[i, j], numeric(1))
    }
  } else {
    for (i in seq_len(nc)) {
      cols[[sprintf("Sigma_%d%d", i, i)]] <-
        vapply(Sig_list, function(S) S[i, i], numeric(1))
    }
  }
  do.call(cbind, cols)   # length(Sig_list) x n_derived, named columns
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
  do.call(cbind, parts)
}

# Marginalized summary of one or more random-effect covariances. Shared by the
# grid integrator (tulpa_re_cov_nested, weighted grid cells) and the Gibbs
# sampler (tulpa_re_cov_gibbs, equal-weight posterior draws). Each element of
# `Sig_node_list` is the per-block list of Sigma matrices for one cell / draw.
# `Marginalize Derived Quantities`: each derived value is computed PER matrix
# then weighted-quantiled. With equal weights `.nl_wtd_quantile` reduces to the
# sample quantile, so the same code summarizes both paths.
.re_cov_derived_summary <- function(Sig_node_list, w, layout) {
  D <- .re_cov_derived_matrix_multi(Sig_node_list, layout)

  summarize <- function(x) {
    ms <- .nl_wtd_mean_sd(x, w)
    q  <- .nl_wtd_quantile(x, w, c(0.025, 0.5, 0.975))
    c(mean = ms$mean, sd = ms$sd, median = q[2L], ci_lo = q[1L], ci_hi = q[3L])
  }
  post <- t(vapply(seq_len(ncol(D)), function(j) summarize(D[, j]), numeric(5)))
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
.re_cov_nested_beta_draws <- function(beta_nodes, beta_cov_nodes, w,
                                      n_draws, beta_names) {
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
  for (d in seq_len(n_draws)) {
    k <- picks[d]
    out[d, ] <- beta_nodes[k, ] + as.numeric(Lb[[k]] %*% stats::rnorm(p))
  }
  colnames(out) <- beta_names %||% paste0("beta", seq_len(p))
  list(draws = out, picks = picks)
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
#' By default `log_prior_theta` is the weakly-informative PC + LKJ hyperprior
#' built per block by [re_cov_pc_lkj_prior()] and summed over blocks (PC prior on
#' each marginal SD via `prior_sigma`, LKJ prior on each correlated block's
#' correlation matrix via `eta`), expressed in the same parameterization with the
#' exact change-of-variables Jacobian. Supply a custom `log_prior_theta` function
#' to override it (then `prior_sigma` / `eta` are ignored); it must act on the
#' full stacked parameter vector.
#'
#' @param y,n_trials,X,family,phi Passed to [tulpa_laplace()] for the inner
#'   solve. `n_trials = NULL` defaults to 1 (binary / single-trial).
#' @param re_terms Either a single random-effect term or a list of them. Each
#'   term is a list with `idx` (1-based group index per observation),
#'   `n_groups`, `n_coefs` (`c`), `Z` (the `n_obs x c` RE design, e.g.
#'   `cbind(1, x)` for `(1 + x | g)`; only required when `c > 1`), and
#'   `correlated` (`TRUE` for a full `Sigma`, `FALSE` for a diagonal one;
#'   defaults to `TRUE`). An optional `label` / `group_var` names the block in
#'   the output. Any `L` / `cov` / `sigma` field is ignored -- `Sigma` is what
#'   this function integrates over.
#' @param prior_sigma,eta Hyperparameters of the default PC + LKJ prior (see
#'   [re_cov_pc_lkj_prior()]): `prior_sigma = c(U, alpha)` with
#'   `P(sigma_i > U) = alpha` (default `c(3, 0.05)`) and LKJ shape `eta`
#'   (default 2). Ignored when `log_prior_theta` is supplied.
#' @param log_prior_theta Optional `function(theta)` returning a scalar log
#'   prior density on the full stacked parameter vector. Default `NULL`, which
#'   builds the PC + LKJ prior from `prior_sigma` / `eta`.
#' @param beta_prior Optional Gaussian prior on the fixed effects, threaded into
#'   every inner [tulpa_laplace()] solve (`list(mean, sd)`).
#'   `NULL` (default) keeps the weak built-in prior.
#' @param n_quad Quadrature order for the inner marginal. `1` (default) uses the
#'   joint-field Laplace inner solve ([tulpa_laplace()]). `> 1` refines the inner
#'   marginal with `n_quad`-point adaptive Gauss-Hermite quadrature (the
#'   [tulpa_re_aghq()] debias applied inside the `Sigma` integration), reducing
#'   the small-cluster variance attenuation for binary / low-count data. AGHQ
#'   requires a single shared grouping factor (the per-group integral must
#'   factorize); with crossed RE terms `n_quad > 1` errors. When AGHQ is used the
#'   fixed effects are integrated, so the reported fixed-effect posterior is the
#'   marginal (ML-II) one rather than the joint-mode (PQL) estimate.
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
#'       `tulpa_fit` methods. The `Sigma` posterior is summarized exactly
#'       (weighted node quantiles) in `posterior`, independent of `n_draws`.
#'     \item `seed`: optional integer seed for the fixed-effect draw synthesis.
#'     \item `diagnose_k`: if `TRUE` (default), compute the outer Pareto k-hat
#'       accuracy diagnostic for the Gaussian proposal over the hyperparameters,
#'       returned as `pareto_k`.
#'     \item `k_samples`: importance draws for the `diagnose_k` estimate
#'       (default 200).
#'     \item `max_iter`, `tol`, `n_threads`: inner-solve controls (see
#'       [tulpa_laplace()]).
#'     \item `checkpoint`: node checkpoint/resume spec `list(path = , resume = )`.
#'       Each completed CCD / grid node (one inner Laplace solve) is cached to
#'       `path`; a `resume = TRUE` run loads the finished nodes and re-solves
#'       only the rest. `resume = FALSE` starts fresh. A file written for
#'       different data, layout, or grid is rejected (fingerprint mismatch).
#'       Default `NULL` (off).
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
#'   - `theta_hat`, `theta_grid`, `weights`, `log_marginal`, `n_grid`, `layout`,
#'     `n_blocks`, `n_coefs` (vector of per-block `c`).
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
                                family = "binomial", phi = 1.0,
                                prior_sigma = c(3, 0.05), eta = 2,
                                log_prior_theta = NULL,
                                beta_prior = NULL, n_quad = 1L,
                                control = list()) {
  # Perf/numerical knobs live in `control = list()` (matching tulpa() /
  # tulpa_nested_laplace()); the signature carries only statistical arguments.
  .check_control(control, .CONTROL_KEYS$re_cov_nested, "tulpa_re_cov_nested")
  integration <- match.arg(control$integration %||% "ccd", c("ccd", "grid"))
  n_per_axis  <- as.integer(control$n_per_axis %||% 5L)
  span        <- control$span %||% 3
  n_draws     <- as.integer(control$n_draws %||% 2000L)
  seed        <- control$seed
  diagnose_k  <- isTRUE(control$diagnose_k %||% TRUE)
  k_samples   <- as.integer(control$k_samples %||% 200L)
  max_iter    <- as.integer(control$max_iter %||% 100L)
  tol         <- control$tol %||% 1e-8
  n_threads   <- as.integer(control$n_threads %||% 1L)
  checkpoint  <- control$checkpoint
  n_quad <- as.integer(n_quad)
  if (n_quad < 1L) stop("`n_quad` must be >= 1.", call. = FALSE)
  .seed_scoped(seed)

  re_terms <- .as_re_terms_list(re_terms)
  if (is.null(n_trials)) n_trials <- rep(1L, length(y))
  layout <- .re_cov_block_layout(re_terms, length(y))
  k <- sum(vapply(layout, `[[`, integer(1), "k"))
  if (is.null(log_prior_theta)) {
    log_prior_theta <- .re_cov_joint_prior(layout, prior_sigma, eta)
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

  # Inner solve: Laplace log-marginal at the supplied per-block covariances.
  # Failures at extreme grid edges (non-finite / non-convergent) return -Inf so
  # the cell gets zero weight rather than aborting the integration.
  inner_logmarg <- function(L_list) {
    val <- tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X,
        re_list = .re_cov_build_re_list(L_list, layout),
        family = family, phi = phi, return_hessian = FALSE,
        beta_prior = beta_prior,
        max_iter = max_iter, tol = tol, n_threads = n_threads
      )$log_marginal,
      error = function(e) -Inf
    )
    if (length(val) != 1L || !is.finite(val)) -Inf else val
  }

  # Full inner solve at the integration nodes: the Laplace log-marginal plus the
  # fixed-effect mode and its MARGINAL covariance (solve(H_beta)). Paid O(n_grid)
  # times, not per optim step.
  inner_fit <- function(L_list) {
    tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X,
        re_list = .re_cov_build_re_list(L_list, layout),
        family = family, phi = phi, return_hessian = TRUE,
        beta_prior = beta_prior,
        max_iter = max_iter, tol = tol, n_threads = n_threads
      ),
      error = function(e) NULL
    )
  }

  # --- pilot init: method-of-moments per block from a Sigma = I fit ----------
  p_fix  <- ncol(X)
  L0_list <- lapply(layout, function(bl) diag(bl$nc))
  pilot <- tryCatch(
    tulpa_laplace(
      y = y, n_trials = n_trials, X = X,
      re_list = .re_cov_build_re_list(L0_list, layout),
      family = family, phi = phi, return_hessian = FALSE,
      beta_prior = beta_prior,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    ),
    error = function(e) NULL
  )
  L_init_list <- L0_list
  if (!is.null(pilot) && !is.null(pilot$mode)) {
    re_vals <- pilot$mode[-seq_len(p_fix)]
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
  theta0 <- .re_cov_L_list_to_theta(L_init_list, layout)

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
    inner_logmarg <- function(L_list) {
      r <- core_solve(L_list); if (is.null(r)) -Inf else r$log_marginal
    }
    inner_fit <- function(L_list) core_solve(L_list)
  }

  # --- mode of g(theta) = log_marginal(Sigma(theta)) + log_prior ------------
  negg <- function(theta) {
    L_list <- .re_cov_theta_to_L_list(theta, layout)
    -(inner_logmarg(L_list) + log_prior_theta(theta))
  }
  opt <- stats::optim(theta0, negg, method = "Nelder-Mead", hessian = TRUE,
                      control = list(maxit = 500L, reltol = 1e-8))
  theta_hat <- opt$par

  # Posterior covariance of theta ~ solve(Hessian of negg). Regularize to PD;
  # fall back to a diagonal scale if the numerical Hessian is unusable.
  Hn <- opt$hessian
  post_cov <- tryCatch({
    Hs <- (Hn + t(Hn)) / 2
    ev <- eigen(Hs, symmetric = TRUE, only.values = TRUE)$values
    if (min(ev) <= 1e-8) stop("non-PD Hessian")
    solve(Hs)
  }, error = function(e) {
    warning("tulpa_re_cov_nested(): outer theta-Hessian not usable (",
            conditionMessage(e), "); falling back to a diagonal proposal ",
            "scale. Integration nodes may be mis-placed -- check the outer ",
            "Pareto-k (fit$pareto_k).", call. = FALSE)
    diag(0.5^2, k)
  })
  L_scale <- t(chol(post_cov))

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

  # --- node checkpoint/resume (gcol33/tulpa#50) -----------------------------
  # Each CCD / grid node is one full inner Laplace solve. `checkpoint =
  # list(path =, resume =)` caches each completed node so a killed run resumes.
  # The node grid is deterministic given the fingerprint (which includes
  # theta_grid), so nodes are keyed by their integer index; the store is an
  # atomically-rewritten RDS, fingerprint-guarded against resuming onto a file
  # written for different data / layout / grid.
  ckpt <- .re_cov_node_checkpoint(checkpoint, fingerprint = list(
    y = as.numeric(y), n_trials = as.integer(n_trials), X = X,
    family = family, phi = phi,
    layout = lapply(layout, function(b) b[c("k", "nc", "full")]),
    theta_grid = theta_grid, max_iter = max_iter, tol = tol,
    beta_prior = beta_prior, n_quad = n_quad))

  # --- evaluate inner marginal + derived quantities per cell ----------------
  # One FULL Laplace solve per node: the log-marginal feeds the integration
  # weight; the fixed-effect mode + marginal covariance feed the posterior-draw
  # synthesis. A failed / non-finite node keeps logm = -Inf (weight 0).
  logm           <- rep(-Inf, ng)
  lp_theta_nodes <- rep(NA_real_, ng)   # hyperparameter log-prior per node
  Sig_node_list  <- vector("list", ng)
  beta_nodes     <- matrix(NA_real_, ng, p_fix)
  beta_cov_nodes <- vector("list", ng)
  for (i in seq_len(ng)) {
    th     <- theta_grid[i, ]
    L_list <- .re_cov_theta_to_L_list(th, layout)
    Sig_node_list[[i]] <- lapply(L_list, function(L) L %*% t(L))
    key <- as.character(i)
    if (!is.null(ckpt) && ckpt$has(key)) {
      fit_i <- ckpt$get(key)
    } else {
      fit_i <- inner_fit(L_list)
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

  # --- derived quantities, marginalized over the grid -----------------------
  summ <- .re_cov_derived_summary(Sig_node_list, w, layout)
  posterior <- summ$posterior

  # --- plug-in MAP summary (for comparison) ---------------------------------
  map <- .re_cov_map_summary(theta_hat, layout)

  # --- fixed-effect posterior from the node mixture -------------------------
  beta_names <- colnames(X) %||% paste0("beta", seq_len(p_fix))
  ds <- .re_cov_nested_beta_draws(beta_nodes, beta_cov_nodes, w,
                                  as.integer(n_draws), beta_names)
  draws <- ds$draws
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
  pareto_k <- NA_real_; k_is_ess <- NA_real_
  if (isTRUE(diagnose_k) && k > 0L) {
    kd <- .with_preserved_seed(tryCatch(
      .nested_outer_pareto_k(
        log_target = function(th) inner_logmarg(.re_cov_theta_to_L_list(th, layout)) +
          log_prior_theta(th),
        theta_hat = theta_hat, L_scale = L_scale, n_samples = k_samples),
      error = function(e) NULL))
    if (!is.null(kd)) { pareto_k <- kd$pareto_k; k_is_ess <- kd$is_ess }
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
    pareto_k_scope  = "outer (hyperparameter) Gaussian proposal",
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
    n_coefs     = vapply(layout, `[[`, integer(1), "nc")
  ), backend = "re_cov_nested", n_fixed = p_fix, fixed_names = beta_names)
}
