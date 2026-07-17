#' Adaptive Gauss-Hermite refinement of a grouped random-effect covariance
#'
#' @description
#' Refines a generalized linear mixed model's fixed effects and random-effect
#' covariance by replacing the per-group Laplace integral with `n_quad`-point
#' adaptive Gauss-Hermite quadrature (AGHQ). At `n_quad = 1` this is the joint
#' Laplace (glmer `nAGQ = 1`); higher `n_quad` reduces the small-cluster
#' attenuation of the variance components for binary / count data. Unlike
#' [agq_fit()] (intercept-only RE, built-in `binomial`/`poisson`/`gaussian`
#' likelihoods), this engine is **callback-driven**: the caller supplies the
#' per-group conditional likelihood, so a custom marginal (e.g. a
#' latent-state-integrated occupancy / detection likelihood, or the
#' latent-abundance-integrated N-mixture marginal) refines through the same
#' quadrature.
#'
#' This is an engine block, not a front door: model packages call it
#' programmatically, so its tuning knobs (`max_iter`, `n_quad`, `keep`)
#' sit in the signature rather than in a `control` list.
#'
#' The engine is **structure-agnostic**. It integrates the per-group marginal
#' \deqn{M_g = \int \exp\{\ell_g(b_g)\}\, N(b_g; 0, \Sigma)\, db_g,}
#' where \eqn{b_g} is the group's random-effect vector (dimension
#' \eqn{\sum_m c_m} over the RE terms) and \eqn{\ell_g(b_g)} is the group's
#' conditional log-likelihood when its linear predictors are perturbed by
#' \eqn{b_g}. How \eqn{b_g} enters the likelihood -- through one linear
#' predictor, or through several coupled arms at different observation
#' granularities (e.g. a per-site abundance arm and a per-visit detection arm
#' sharing a species grouping) -- lives entirely in the callback. The engine
#' only needs, per group, the value / gradient / Hessian of \eqn{\ell_g} in
#' \eqn{b_g} (for the mode) and \eqn{\ell_g} at the quadrature nodes (for the
#' sum). The fixed parameters `theta` and the log-Cholesky coordinates of
#' \eqn{\Sigma} are optimized jointly on \eqn{\sum_g \log M_g}; standard errors
#' come from the exact-marginal Hessian.
#'
#' Two callback forms select the structure (supply exactly one):
#'
#' * **`make_site`** -- the common **single-arm, per-row-separable** case
#'   (\eqn{\ell_g(b_g) = \sum_{i \in g} \log f_i(\eta_i + Z_i b_g)} for one
#'   linear predictor \eqn{\eta}). The engine builds the oracle from the
#'   per-row marginal and the RE design `Z` itself.
#' * **`make_group`** -- the **general / multi-arm** case. The caller supplies
#'   the per-group `b`-space oracle directly, so non-separable units (e.g. the
#'   visits of an N-mixture site coupled through the shared latent count) and
#'   random effects on several arms at once are handled with no engine change.
#'
#' Scope: one shared grouping factor across all RE terms (the per-group integral
#' factorizes). The total RE dimension per group should be small (the quadrature
#' grid is `n_quad^dim`).
#'
#' @param theta0 Initial fixed-parameter vector. The engine optimizes these
#'   jointly with the RE covariance; the callback interprets them.
#' @param re_terms A list of RE term specs (or one spec), each defining a
#'   covariance block: `n_coefs` (block dimension \eqn{c_m}), optional
#'   `correlated` (default `TRUE` for `c_m > 1`; `FALSE` gives a diagonal block),
#'   and `n_groups` (shared across terms). For the `make_site` path each term
#'   also carries `idx` (1-based group index, length `n_obs`) and, for a slope
#'   block, `Z` (the `n_obs x n_coefs` design). For the `make_group` path the
#'   per-observation `idx` / `Z` are optional -- the callback owns them -- and
#'   the term needs only `n_coefs` / `correlated` / `n_groups`.
#' @param Sigma0 List of initial per-term covariance matrices (the EM estimate).
#' @param make_site `function(theta)` for the single-arm separable case,
#'   returning a list with: `eta_re` (length `n_obs`, the RE-arm fixed
#'   predictor), `deriv = function(rows, eta)` returning `list(logL, d1, d2)`
#'   (per-row marginal log-likelihood and its first/second derivatives w.r.t.
#'   the RE-arm predictor `eta`, used for the per-group mode), and `lmat =
#'   function(rows, ETA)` returning a `length(rows) x ncol(ETA)` matrix of
#'   per-observation log-likelihoods over the quadrature node columns. Supply
#'   this or `make_group`, not both.
#' @param make_group `function(theta)` for the general / multi-arm case,
#'   returning a list with two per-group closures (let `d = sum(n_coefs)` be the
#'   group RE dimension):
#'   * `grad_hess(g, b)` -- for group `g` at RE value `b` (length `d`), the list
#'     `list(logL, grad, negH)`: the group conditional log-likelihood
#'     \eqn{\ell_g(b)}, its gradient \eqn{\partial \ell_g/\partial b} (length
#'     `d`), and the **data-only** observed information
#'     \eqn{-\partial^2 \ell_g/\partial b^2} (`d x d`; the engine adds the
#'     \eqn{\Sigma^{-1}} prior curvature).
#'   * `node_ll(g, B)` -- for group `g`, a numeric vector of length `nrow(B)`
#'     giving \eqn{\ell_g} at each quadrature node (rows of the `nrow x d`
#'     matrix `B` are candidate `b` vectors).
#'   The callback owns all arm / design / clamping bookkeeping. Supply this or
#'   `make_site`, not both.
#' @param oracle Optional prebuilt native (compiled) oracle, an external pointer
#'   to a `REGroupOracle` (constructed in a consumer package's src/ via
#'   `LinkingTo: tulpa` against `<tulpa/aghq_oracle.h>`). When supplied the
#'   engine drives it directly, with no per-group / per-node round trip into R,
#'   and neither `make_site` nor `make_group` is needed; `re_terms`, `theta0`
#'   and `Sigma0` must still describe the same layout the oracle exposes. The
#'   integration core is identical to the R-closure path.
#' @param n_obs Number of observations (length of each term's `idx`). Required
#'   for the `make_site` path; ignored for `make_group`.
#' @param keep Optional logical/integer mask of observations to include
#'   (default all; `make_site` path only). Rows outside `keep` are dropped from
#'   every group.
#' @param n_quad Quadrature nodes per RE dimension. Either a single integer
#'   (default 9; `1` = Laplace) broadcast to every covariance block, or an
#'   integer vector of length `length(re_terms)` giving a per-block node count.
#'   The tensor grid then uses `n_quad[b]` nodes along every dimension of block
#'   `b`, for `prod_b n_quad[b]^(dim_b)` total nodes; a scalar reproduces the
#'   uniform grid exactly. Per-block orders let a heterogeneous stack spend fewer
#'   nodes on cheap scalar nuisance blocks than on the correlated coefficient
#'   blocks (e.g. `c(3, 3, 2, 2)` on blocks of dimension `2, 2, 1, 1` gives
#'   `3^2 * 3^2 * 2 * 2 = 324` nodes rather than `3^6 = 729`).
#' @param lkj_eta LKJ shape for an optional correlation penalty on each
#'   *correlated* block (log-density `(eta - 1) log det R`, maximized at
#'   independence). `1` disables it; `> 1` regularizes a weakly-identified
#'   correlation off the boundary without touching the marginal SDs. The
#'   marginal SDs are otherwise unpenalized (pure ML), so the refinement debiases
#'   them rather than shrinking them.
#' @param theta_prior_sd Optional Gaussian ridge SD on the fixed parameters
#'   `theta` (a mean-zero `N(0, theta_prior_sd^2)` prior, added to the optimized
#'   objective and hence the marginal Hessian). `Inf` (default) is pure ML on
#'   `theta`; a large finite value (e.g. 100) is a weak ridge that stabilizes a
#'   weakly-identified fixed effect without materially shifting the estimate.
#' @param gradient How `stats::optim` gets the gradient of the AGHQ objective.
#'   `"fd"` (default) lets `optim` finite-difference the objective -- correct at
#'   every `n_quad` and the only option for the R-closure (`make_site` /
#'   `make_group`) paths. `"analytic"` supplies the Fisher-identity gradient
#'   (posterior-weighted theta-score plus the `Sigma` moment-matching residual),
#'   which avoids the per-coordinate objective re-solve and so is far cheaper for
#'   the quadrature debias. It requires a prebuilt native `oracle` (the only one
#'   exposing the theta-score) and `n_quad > 1`: being the gradient of the true
#'   marginal it omits the node-placement terms (`O` of the AGHQ truncation), so
#'   it agrees with the objective only as `n_quad` grows.
#' @param max_iter Optimizer iteration cap (default 200).
#'
#' @return A list with: `theta` (refined fixed parameters), `Sigma_list`
#'   (refined per-term covariance), `blup` / `blup_var` (per-term `n_groups x
#'   n_coefs` posterior mean / variance of the RE), `theta_cov` / `theta_se`
#'   (fixed-parameter covariance / SE from the marginal Hessian), `log_marginal`
#'   (the AGHQ marginal log-likelihood at the optimum, excluding any ridge),
#'   `n_quad`, `lkj_eta`, and `converged`. RE terms that do not share one
#'   grouping factor are an input error and stop; a singular / non-finite
#'   optimum warns and returns `NULL` (caller keeps its prior fit).
#' @references
#' Pinheiro & Bates (1995). Approximations to the log-likelihood function in
#' the nonlinear mixed-effects model. \emph{Journal of Computational and
#' Graphical Statistics} 4(1):12-35.
#' Lewandowski, Kurowicka & Joe (2009). Generating random correlation matrices
#' based on vines and extended onion method. \emph{Journal of Multivariate
#' Analysis} 100(9):1989-2001.
#' @examples
#' \donttest{
#' # A per-row-separable binomial GLMM marginal supplied through `make_site`.
#' l1pe <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
#' make_binom_site <- function(X, y, nt) function(theta) {
#'   eta_fixed <- as.numeric(X %*% theta)
#'   list(eta_re = eta_fixed,
#'        deriv = function(rows, eta) {
#'          p <- plogis(eta)
#'          list(logL = y[rows] * eta - nt[rows] * l1pe(eta),
#'               d1 = y[rows] - nt[rows] * p, d2 = -nt[rows] * p * (1 - p))
#'        },
#'        lmat = function(rows, ETA) y[rows] * ETA - nt[rows] * l1pe(ETA))
#' }
#' set.seed(1)
#' ng <- 30L; npg <- 8L; n <- ng * npg
#' g <- rep(seq_len(ng), each = npg); x <- rnorm(n)
#' X <- cbind(1, x); nt <- rep(3L, n); u <- rnorm(ng, 0, 0.9)
#' y <- rbinom(n, nt, plogis(0.3 + 0.7 * x + u[g]))
#' fit <- tulpa_re_aghq(theta0 = c(0, 0),
#'                      re_terms = list(list(idx = g, n_groups = ng, n_coefs = 1L)),
#'                      Sigma0 = list(matrix(0.25, 1, 1)),
#'                      make_site = make_binom_site(X, y, nt), n_obs = n, n_quad = 5L)
#' sqrt(fit$Sigma_list[[1]][1, 1])     # adaptive-GHQ RE standard deviation
#' }
#' @export
tulpa_re_aghq <- function(theta0, re_terms, Sigma0,
                          make_site = NULL, make_group = NULL, oracle = NULL,
                          n_obs = NULL,
                          keep = NULL, n_quad = 9L, lkj_eta = 1,
                          theta_prior_sd = Inf, gradient = c("fd", "analytic"),
                          max_iter = 200L) {
  gradient <- match.arg(gradient)
  native <- !is.null(oracle)
  if (!native && (is.null(make_site) == is.null(make_group))) {
    stop("Supply exactly one of `make_site` (single-arm), `make_group` ",
         "(general / multi-arm), or a prebuilt native `oracle`.", call. = FALSE)
  }
  # The analytic gradient is the posterior-weighted theta-score, which only the
  # native compiled oracles supply (the R-closure bridge's theta_score is a
  # no-op). So gradient = "analytic" requires a prebuilt native `oracle`.
  if (gradient == "analytic" && !native) {
    stop("gradient = \"analytic\" requires a prebuilt native `oracle`; the ",
         "R-closure bridge (make_site / make_group) does not supply the ",
         "theta-score. Use gradient = \"fd\".", call. = FALSE)
  }
  single_arm <- !is.null(make_site)
  if (single_arm && is.null(n_obs)) {
    stop("`n_obs` is required for the `make_site` path.", call. = FALSE)
  }

  layout <- .re_cov_block_layout(.as_re_terms_list(re_terms), n_obs)

  # One shared grouping factor (the per-group integral factorizes only then).
  # The make_group path may omit the per-observation `idx`; the layout then
  # carries `idx = NULL`, and the shared-factor check reduces to matching
  # `n_groups` (identical(NULL, NULL) is TRUE).
  idx1 <- layout[[1L]]$idx
  ng   <- layout[[1L]]$n_groups
  same <- all(vapply(layout, function(b)
    identical(b$idx, idx1) && identical(b$n_groups, ng), logical(1)))
  if (!same) {
    stop("`re_terms` must share one grouping factor (identical `idx` / ",
         "`n_groups` across terms): the per-group AGHQ integral factorizes ",
         "only then. For multiple grouping factors use tulpa_re_cov_nested() ",
         "or tulpa_re_cov_gibbs().", call. = FALSE)
  }

  nc_terms <- vapply(layout, function(b) b$nc, integer(1))
  dtot     <- sum(nc_terms)
  coef_off <- cumsum(c(0L, nc_terms))

  # Quadrature order: one scalar broadcast to every block, or one node count per
  # covariance block (same order as `re_terms`). The compiled grid builder
  # expands this to a per-dimension count, so a scalar reproduces the uniform
  # tensor grid byte-for-byte.
  n_quad <- as.integer(n_quad)
  if (length(n_quad) != 1L && length(n_quad) != length(layout)) {
    stop(sprintf(paste0("`n_quad` must be a single integer or one per RE block ",
                        "(length 1 or %d); got length %d."),
                 length(layout), length(n_quad)), call. = FALSE)
  }
  if (anyNA(n_quad) || any(n_quad < 1L)) {
    stop("`n_quad` entries must be positive integers (>= 1).", call. = FALSE)
  }

  # Fixed-parameter / RE-covariance parameter split. The RE block reuses tulpa's
  # log-Cholesky packing (.re_cov_theta_to_L_list); `theta` is everything else.
  re_par0 <- .re_cov_L_list_to_theta(lapply(Sigma0, .re_chol_spd), layout)
  n_theta <- length(theta0)

  # Block dimensions / correlated flags for the compiled engine. `cl` is shared
  # with the single-arm oracle assembly below; the quadrature grid, log-Cholesky
  # packing and LKJ penalty all live in C++ (src/aghq_re*.{h,cpp}).
  full_vec <- vapply(layout, function(b) isTRUE(b$full), logical(1))
  cl <- function(e) pmin(pmax(e, -30), 30)

  # -------------------------------------------------------------------------
  # Per-group conditional-likelihood oracle. `build_oracle(theta)` returns
  #   grad_hess(g, b) -> list(logL, grad, negH)   data-only value/score/info
  #   node_ll(g, B)   -> numeric over node rows    data log-lik at nodes
  # The compiled core is identical for both callback forms; only the oracle
  # differs. For make_site the engine assembles it from the per-row marginal and
  # the RE design; for make_group the caller supplies it directly. The R-closure
  # bridge (cpp_aghq_make_rclosure_oracle) marshals it into the C++ engine.
  # -------------------------------------------------------------------------
  if (single_arm) {
    Zc <- do.call(cbind, lapply(layout, function(b) b$Z))   # n_obs x dtot
    if (is.null(keep)) keep <- rep(TRUE, n_obs)
    if (is.logical(keep)) keep <- which(keep)
    rows_by_g <- lapply(seq_len(ng), function(g) {
      r <- which(idx1 == g); r[r %in% keep]
    })

    build_oracle <- function(theta) {
      site <- make_site(theta)
      aRE  <- cl(site$eta_re)
      list(
        grad_hess = function(g, b) {
          rows <- rows_by_g[[g]]
          Zg <- Zc[rows, , drop = FALSE]
          gv <- site$deriv(rows, cl(aRE[rows] + as.numeric(Zg %*% b)))
          list(logL = sum(gv$logL),
               grad = as.numeric(crossprod(Zg, gv$d1)),
               negH = -crossprod(Zg, gv$d2 * Zg))
        },
        node_ll = function(g, B) {
          rows <- rows_by_g[[g]]
          Zg <- Zc[rows, , drop = FALSE]
          ETA <- cl(matrix(aRE[rows], length(rows), nrow(B)) + Zg %*% t(B))
          colSums(site$lmat(rows, ETA))
        })
    }
  } else {
    build_oracle <- function(theta) make_group(theta)
  }

  # -------------------------------------------------------------------------
  # Optimize sum_g log M_g (+ optional LKJ) over [theta ; log-Cholesky Sigma]
  # with the compiled AGHQ engine. cpp_aghq_objective returns a large finite
  # penalty on a failed solve so stats::optim rejects it.
  #   gradient = "fd"        -- optim finite-differences the objective
  #                             (consistent at every n_quad, incl. n_quad = 1).
  #   gradient = "analytic"  -- supply the Fisher-identity gradient
  #                             (cpp_aghq_objective_grad). It is the gradient of
  #                             the true marginal: it omits the node-placement
  #                             terms (O(AGHQ truncation)), so it is consistent
  #                             with the objective only for n_quad > 1 and is
  #                             cheap (one group sweep, no per-coordinate
  #                             re-solve). The objective + gradient share one
  #                             evaluation per par via a small cache.
  # -------------------------------------------------------------------------
  orc   <- if (native) oracle
           else cpp_aghq_make_rclosure_oracle(build_oracle, ng, dtot, n_theta)
  ridge <- if (is.finite(theta_prior_sd)) 0.5 / theta_prior_sd^2 else 0

  if (gradient == "analytic") {
    fns <- .aghq_analytic_optim_fns(orc, nc_terms, full_vec, n_quad, lkj_eta,
                                    ridge = ridge, n_theta = n_theta)
    opt <- stats::optim(c(theta0, re_par0), fns$fn, fns$gr, method = "BFGS",
                        hessian = TRUE, control = list(maxit = max_iter, reltol = 1e-9))
  } else {
    negf <- function(par)
      -cpp_aghq_objective(par, orc, nc_terms, full_vec, n_quad, lkj_eta) +
        ridge * sum(par[seq_len(n_theta)]^2)
    opt <- stats::optim(c(theta0, re_par0), negf, method = "BFGS", hessian = TRUE,
                        control = list(maxit = max_iter, reltol = 1e-9))
  }
  V <- tryCatch(solve(opt$hessian), error = function(e) NULL)
  if (is.null(V) || any(!is.finite(opt$par))) {
    warning("tulpa_re_aghq: the joint optimum is singular or non-finite ",
            "(no usable exact-marginal Hessian); returning NULL.",
            call. = FALSE)
    return(NULL)
  }

  theta_ref  <- opt$par[seq_len(n_theta)]
  L_list     <- .re_cov_theta_to_L_list(opt$par[-seq_len(n_theta)], layout)
  Sigma_list <- lapply(L_list, tcrossprod)
  # Pure AGHQ marginal at the optimum, for callers reporting log-lik. Evaluate
  # at lkj_eta = 1 (uniform LKJ) so the reported value excludes the (eta - 1)
  # log|R| penalty as well as the ridge -- otherwise a fit with lkj_eta > 1
  # carries a penalty term that differs across models and biases LRT / AIC
  # comparisons. The optimization still used the caller's lkj_eta above.
  log_marginal <- cpp_aghq_objective(opt$par, orc, nc_terms, full_vec,
                                     n_quad, 1.0)

  # Per-group BLUPs + marginal variances at the optimum. The engine returns the
  # prior fallback for empty groups (mode 0, variance diag(Sigma)).
  bl   <- cpp_aghq_blups(opt$par, orc, nc_terms, full_vec)
  BHAT <- bl$bhat; BVAR <- bl$bvar
  blup     <- lapply(seq_along(layout), function(m) BHAT[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])
  blup_var <- lapply(seq_along(layout), function(m) BVAR[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])

  list(
    theta      = theta_ref,
    Sigma_list = Sigma_list,
    blup       = blup,
    blup_var   = blup_var,
    theta_cov  = V[seq_len(n_theta), seq_len(n_theta), drop = FALSE],
    theta_se   = sqrt(pmax(diag(V)[seq_len(n_theta)], 0)),
    log_marginal = log_marginal,
    n_quad     = n_quad,
    lkj_eta    = lkj_eta,
    converged  = isTRUE(opt$convergence == 0L)
  )
}

# Memoize cpp_aghq_objective_grad over `par`: returns an `eval_at(par)` that
# recomputes only when `par` changes (stats::optim queries fn and gr separately,
# usually at the same point, so one C++ group sweep serves both). The single
# source of the analytic objective+gradient call -- shared by the full-par ML-II
# optimizer (.aghq_analytic_optim_fns) and the fixed-Sigma beta profile in
# tulpa_re_cov_nested(n_quad > 1).
.aghq_grad_cache <- function(orc, nc_terms, full_vec, n_quad, lkj_eta) {
  cache <- new.env(parent = emptyenv()); cache$par <- NULL; cache$val <- NULL
  function(par) {
    if (is.null(cache$par) || !identical(par, cache$par)) {
      cache$par <- par
      cache$val <- cpp_aghq_objective_grad(par, orc, nc_terms, full_vec,
                                            as.integer(n_quad), lkj_eta)
    }
    cache$val
  }
}

# Shared optim closures for the full-par analytic-gradient AGHQ path. The
# returned functions are NEGATED for minimization. `ridge` adds a mean-zero
# Gaussian ridge `ridge * sum(par[1:n_theta]^2)` (ridge = 0.5 / sd^2) on the
# first `n_theta` parameters and its gradient `2 * ridge * par`; `ridge = 0`
# disables it. A failed solve returns the objective sentinel and a zero gradient
# so optim backtracks. Consumed by tulpa_re_aghq(gradient = "analytic") and
# agq_fit(n_quad > 1).
.aghq_analytic_optim_fns <- function(orc, nc_terms, full_vec, n_quad, lkj_eta,
                                     ridge = 0, n_theta = 0L) {
  eval_at <- .aghq_grad_cache(orc, nc_terms, full_vec, n_quad, lkj_eta)
  list(
    fn = function(par) {
      r <- eval_at(par)
      f <- if (isTRUE(r$ok)) r$f else -1e10
      -f + (if (ridge > 0) ridge * sum(par[seq_len(n_theta)]^2) else 0)
    },
    gr = function(par) {
      r <- eval_at(par)
      g <- if (isTRUE(r$ok)) -r$grad else rep(0, length(par))
      if (ridge > 0)
        g[seq_len(n_theta)] <- g[seq_len(n_theta)] + 2 * ridge * par[seq_len(n_theta)]
      g
    }
  )
}
