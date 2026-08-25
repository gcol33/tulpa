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
#' @param sigma_prior Optional Penalized-Complexity prior on the marginal
#'   standard deviations of one or more RE covariance blocks, added to the
#'   objective (and hence the marginal Hessian). `NULL` (default) is pure ML on
#'   the covariances -- the refinement debiases the SDs rather than shrinking
#'   them. Otherwise a `c(U, alpha)` pair (`P(sigma_i > U) = alpha`, the same
#'   convention as [re_cov_pc_lkj_prior()]) applied to every block, or a list
#'   `list(blocks = <integer indices>, prior_sigma = c(U, alpha))` applied to
#'   the named blocks only. Reuses the exact PC log-prior + Jacobian of
#'   [re_cov_pc_lkj_prior()]. A weakly-identified variance component (e.g. a
#'   scalar dispersion / zero-inflation random effect at few groups) can drift to
#'   the boundary and flatten the marginal Hessian; a weak PC prior adds
#'   curvature there (the `+ log sigma` Jacobian repels `sigma -> 0`, the
#'   `- lambda sigma` term caps inflation), keeping the joint optimum non-singular
#'   without materially shifting an identified fit.
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
#'   n_coefs` posterior mean / variance of the RE), `group_ok` (logical, length
#'   `n_groups`: `FALSE` where that group's mode search or precision
#'   factorization failed, which is what the `NA` rows of `blup`, `blup_var`,
#'   `blup_cov_g` and `blup_cross_g` mean -- a caller conditions its per-group
#'   reads on this rather than on trapping the accompanying warning),
#'   `blup_cross` (per-term
#'   `n_groups x n_theta x n_coefs` array: the mode/theta cross-Hessian block
#'   `Bf`, `-d^2 ell_g / d theta db` at each group's mode, in the same
#'   negative-Hessian sign convention as the posterior precision underlying
#'   `blup_var` -- so a joint draw of `theta` and a group's RE `b_g` uses
#'   `b_g | theta ~ N(blup_g - Cinv_g %*% t(Bf_g) %*% (theta_draw - theta),
#'   Cinv_g)` with `Cinv_g` the group's `n_coefs x n_coefs` posterior
#'   covariance block. `NA` throughout when `blup_cross_available` is `FALSE`:
#'   the cross-Hessian needs the oracle's analytic `theta_score`, which the
#'   R-closure bridge (`make_site` / `make_group`) does not supply -- only a
#'   prebuilt native `oracle` carries it), `blup_cov_g` / `blup_cross_g`
#'   (per-group lists, length `n_groups`, of the FULL joint posterior
#'   covariance (`d x d`, `d` = every RE term's width combined) and mode/theta
#'   cross-Hessian (`n_theta x d`) across ALL RE terms sharing that group --
#'   the superset `blup_var`/`blup_cross` reduce to a per-term diagonal block
#'   of when a group carries more than one term, since a group's terms are
#'   found jointly and can carry real posterior covariance BETWEEN terms
#'   (e.g. an abundance-arm and a detection-arm term sharing one grouping
#'   factor); same `NA`-when-unavailable rule as `blup_cross`), `theta_cov` /
#'   `theta_se`
#'   (fixed-parameter covariance / SE from the marginal Hessian), `re_par` /
#'   `re_par_cov` / `re_par_se` (the RE-covariance coordinates the optimizer
#'   carried -- log-Cholesky for a full block, log-SD for a diagonal one -- with
#'   their block of the same inverse Hessian, so `SE(log sigma)` is available
#'   for a boundary test on a weakly-identified variance component),
#'   `re_par_layout` (per block: `label`, `nc`, `full`, the `index` range into
#'   `re_par` and the `coord` names, so a caller does not reconstruct the
#'   packing), `joint_cov` (the whole `(n_theta + n_chol)` inverse Hessian),
#'   `log_marginal`
#'   (the AGHQ marginal log-likelihood at the optimum, excluding any ridge),
#'   `n_quad`, `lkj_eta`, `converged`, and `counts` (`stats::optim`'s own
#'   `function` / `gradient` evaluation counts, so a caller reporting how much
#'   work the fit took has a number to report rather than `NA`). RE terms that
#'   do not share one
#'   grouping factor are an input error and stop. Three conditions warn and
#'   return `NULL` (caller keeps its prior fit): a singular / non-finite
#'   optimum, an objective that is already undefined at the starting parameters
#'   (some group's solve fails there, so there is nothing to descend), and an
#'   optimum whose objective is the failure sentinel rather than an attained
#'   marginal likelihood -- the last two report which groups failed.
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
                          theta_prior_sd = Inf, sigma_prior = NULL,
                          gradient = c("fd", "analytic"),
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

  # A group whose solve fails takes the whole objective to the failure sentinel,
  # and optim only ever moves to a point it has accepted as an IMPROVEMENT -- so
  # a run can finish on the sentinel only by starting on it, and then whatever it
  # reports is the shape of the R-side ridge / PC prior alone. Worse, `reltol`
  # is relative: at |f| = 1e10 a tolerance of 1e-9 is 10 nats, so such a run
  # declares convergence after one hair of a step rather than searching
  # (gcol33/tulpa#606). Refuse it here, where the groups behind it can still be
  # named, instead of returning a fit whose numbers are the start point.
  par0 <- c(theta0, re_par0)
  if (.aghq_is_fail(cpp_aghq_objective(par0, orc, nc_terms, full_vec,
                                       n_quad, lkj_eta))) {
    gok <- .aghq_group_status(par0, orc, nc_terms, full_vec)
    warning("tulpa_re_aghq: the AGHQ objective is not defined at the starting ",
            "parameters -- the per-group solve failed for ",
            if (is.null(gok)) "at least one group"
            else .aghq_failed_group_phrase(gok),
            ", so there is nothing for the optimizer to descend. Returning ",
            "NULL (a different warm start, covariance start, or group ",
            "configuration is needed).", call. = FALSE)
    return(NULL)
  }

  # Optional PC prior on the RE-block marginal SDs. It touches only the RE-
  # covariance coordinates (the second `re_par` slice of `par`), so it enters the
  # objective as `+ log p_sigma(re_par)` (we minimize the negative). The per-block
  # log-prior + Jacobian is the shared .re_cov_block_logprior (single source of
  # truth with re_cov_pc_lkj_prior / tulpa_re_cov_nested); its gradient is a
  # central difference over the touched coordinates only (no oracle solve, so it
  # is cheap). sigma_pen = NULL keeps the objective byte-identical.
  sigma_pen <- .aghq_sigma_penalty(sigma_prior, layout, n_theta)

  if (gradient == "analytic") {
    fns <- .aghq_analytic_optim_fns(orc, nc_terms, full_vec, n_quad, lkj_eta,
                                    ridge = ridge, n_theta = n_theta)
    if (!is.null(sigma_pen)) {
      fn0 <- fns$fn; gr0 <- fns$gr
      fns$fn <- function(par) fn0(par) - sigma_pen$val(par)
      fns$gr <- function(par) gr0(par) - sigma_pen$grad(par)
    }
    opt <- stats::optim(c(theta0, re_par0), fns$fn, fns$gr, method = "BFGS",
                        hessian = TRUE, control = list(maxit = max_iter, reltol = 1e-9))
  } else {
    negf <- function(par)
      -cpp_aghq_objective(par, orc, nc_terms, full_vec, n_quad, lkj_eta) +
        ridge * sum(par[seq_len(n_theta)]^2) -
        (if (is.null(sigma_pen)) 0 else sigma_pen$val(par))
    opt <- stats::optim(c(theta0, re_par0), negf, method = "BFGS", hessian = TRUE,
                        control = list(maxit = max_iter, reltol = 1e-9))
  }
  # Pure AGHQ marginal at the optimum, for callers reporting log-lik. Evaluate
  # at lkj_eta = 1 (uniform LKJ) so the reported value excludes the (eta - 1)
  # log|R| penalty as well as the ridge -- otherwise a fit with lkj_eta > 1
  # carries a penalty term that differs across models and biases LRT / AIC
  # comparisons. The optimization still used the caller's lkj_eta above.
  log_marginal <- cpp_aghq_objective(opt$par, orc, nc_terms, full_vec,
                                     n_quad, 1.0)
  # It is read BEFORE the Hessian, because the sentinel is finite: an optimum
  # carrying it passes every downstream check while `log_marginal` is a sentinel
  # a consumer adds to other terms, and `theta_cov` is the finite-difference
  # curvature of the penalty rather than of a likelihood (#606).
  if (.aghq_is_fail(log_marginal)) {
    gok <- .aghq_group_status(opt$par, orc, nc_terms, full_vec)
    warning("tulpa_re_aghq: the optimizer stopped at a parameter where the ",
            "AGHQ objective is undefined -- the per-group solve failed for ",
            if (is.null(gok)) "at least one group"
            else .aghq_failed_group_phrase(gok),
            ". Its value is the failure sentinel, not a marginal likelihood, ",
            "so no fit is reported; returning NULL.", call. = FALSE)
    return(NULL)
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
  # Per-group BLUPs + marginal variances at the optimum. The engine returns the
  # prior fallback for empty groups (mode 0, variance diag(Sigma)). A group whose
  # mode search or precision factorization failed comes back NA with
  # `group_ok[g]` FALSE, rather than the numbers a failed decomposition's solve
  # would otherwise return.
  bl   <- cpp_aghq_blups(opt$par, orc, nc_terms, full_vec)
  # The status travels on the fit as `group_ok`, not only in the warning below:
  # a caller taking `blup[[m]][g, ]` needs to know which rows are trustworthy,
  # and a warning is the one channel it cannot read without parsing indices out
  # of the message text (gcol33/tulpa#605).
  group_ok <- as.logical(bl$group_ok)
  if (!all(group_ok)) {
    warning("tulpa_re_aghq: the per-group posterior solve failed for ",
            .aghq_failed_group_phrase(group_ok),
            "; their blup, blup_var, blup_cov_g and blup_cross_g entries are ",
            "NA. The full status is `group_ok` on the returned fit.",
            call. = FALSE)
  }
  # A coordinate whose assembled variance was not positive is carried by the
  # absolute PD backstop on Sigma, so the covariance reported for it came from
  # that constant and not from the fitted parameter. It is surfaced rather than
  # inherited: nothing else on the fit distinguishes "the variance really is
  # this small" from "the jitter is what you are reading" (gcol33/tulpa#595).
  sigma_jitter_floored <- as.logical(bl$sigma_jitter_floored)
  if (any(sigma_jitter_floored)) {
    warning(sprintf(
      "tulpa_re_aghq: the assembled RE covariance had %d of %d diagonal entries at or below zero (coordinate%s %s); their variance is the PD backstop rather than a fitted value, so the BLUP variances on those coordinates are not a function of the parameter.",
      sum(sigma_jitter_floored), length(sigma_jitter_floored),
      if (sum(sigma_jitter_floored) > 1L) "s" else "",
      paste(which(sigma_jitter_floored), collapse = ", ")), call. = FALSE)
  }
  BHAT <- bl$bhat; BVAR <- bl$bvar; BCROSS <- bl$bcross; BCOV <- bl$bcov
  blup     <- lapply(seq_along(layout), function(m) BHAT[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])
  blup_var <- lapply(seq_along(layout), function(m) BVAR[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])
  # blup_cross[[m]] is n_groups x n_theta x nc_terms[m]: the mode/theta
  # cross-Hessian block (Bf, see cpp_aghq_blups) for RE term m, one slice per
  # group. NA throughout when bl$bcross_available is FALSE (the R-closure
  # bridge -- make_site / make_group -- has no analytic theta_score; only a
  # prebuilt native `oracle` supplies this).
  blup_cross <- lapply(seq_along(layout), function(m)
    BCROSS[, , coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])

  # blup_cov_g[[g]] / blup_cross_g[[g]]: the group's FULL joint posterior
  # covariance (d x d, d = every RE term's width combined) and mode/theta
  # cross-Hessian (n_theta x d), unsliced by term -- the superset
  # `blup_var`/`blup_cross` reduce to a per-term diagonal block of. A group
  # carrying more than one RE term (e.g. one term per formula arm sharing the
  # same grouping factor) has real posterior covariance BETWEEN those terms
  # (`cpp_aghq_blups`'s mode-finding solves every term's coefficients for a
  # group jointly, one Newton step over the combined vector) -- `blup_var`
  # alone cannot express it, and drawing the terms independently repeats the
  # cross-term bug one level deeper (inside a group instead of
  # between the community mean and a group). NA throughout when
  # `blup_cross_available` is FALSE, matching `blup_cross`.
  n_groups <- dim(BCOV)[1L]
  blup_cov_g <- lapply(seq_len(n_groups), function(g)
    array(BCOV[g, , ], dim = c(dtot, dtot)))
  blup_cross_g <- lapply(seq_len(n_groups), function(g)
    array(BCROSS[g, , ], dim = c(n_theta, dtot)))

  # The joint inverse Hessian covers c(theta, log-Cholesky Sigma). The RE half
  # is what says how well determined a variance component is -- a scalar block
  # collapsing toward its boundary at few groups converges cleanly and leaves no
  # other trace on the fit -- so it is reported alongside theta's rather than
  # discarded.
  re_par_idx <- seq_len(nrow(V))[-seq_len(n_theta)]
  re_par_layout <- .aghq_re_par_layout(layout)
  re_par_names <- unlist(lapply(re_par_layout, `[[`, "coord"), use.names = FALSE)
  re_par_cov <- V[re_par_idx, re_par_idx, drop = FALSE]
  dimnames(re_par_cov) <- list(re_par_names, re_par_names)
  re_par_se <- stats::setNames(sqrt(pmax(diag(V)[re_par_idx], 0)), re_par_names)

  list(
    theta      = theta_ref,
    Sigma_list = Sigma_list,
    blup       = blup,
    blup_var   = blup_var,
    # Per group: TRUE where the posterior solve succeeded. FALSE is what the NA
    # rows of blup / blup_var / blup_cov_g / blup_cross_g mean (#605).
    group_ok   = group_ok,
    blup_cross = blup_cross,
    blup_cross_available = bl$bcross_available,
    # Per-coordinate: TRUE where the reported variance came from the PD
    # backstop on Sigma rather than from the fitted parameter (#595).
    sigma_jitter_floored = sigma_jitter_floored,
    blup_cov_g   = blup_cov_g,
    blup_cross_g = blup_cross_g,
    theta_cov  = V[seq_len(n_theta), seq_len(n_theta), drop = FALSE],
    theta_se   = sqrt(pmax(diag(V)[seq_len(n_theta)], 0)),
    re_par     = stats::setNames(opt$par[re_par_idx], re_par_names),
    re_par_cov = re_par_cov,
    re_par_se  = re_par_se,
    re_par_layout = re_par_layout,
    joint_cov  = V,
    log_marginal = log_marginal,
    n_quad     = n_quad,
    lkj_eta    = lkj_eta,
    converged  = isTRUE(opt$convergence == 0L),
    # stats::optim's own evaluation counts, verbatim (BFGS reports evaluations,
    # not iterations). A caller reporting the effort behind a fit has nothing
    # else to read: the joint driver is one optim call, so without this its
    # report is NA on a fit it declares converged (gcol33/tulpaObs#281).
    counts     = opt$counts
  )
}

# Coordinate map for the RE-covariance half of the joint optimizer's parameter
# vector. `.re_cov_theta_to_L_list()` unpacks each block's `k` coordinates in
# turn -- a full block in log-Cholesky order (column-major lower triangle, the
# diagonal on the log scale), a diagonal block as `nc` log-SDs -- so the packing
# is what a caller reading `re_par_se` needs in order to attach an SE to a named
# variance component. Names follow `.re_cov_derived_summary()`: bare with one
# block, block-label-prefixed with several.
.aghq_re_par_layout <- function(layout) {
  M <- length(layout)
  pos <- 0L
  lapply(seq_along(layout), function(m) {
    bl <- layout[[m]]
    nm <- if (bl$full) {
      unlist(lapply(seq_len(bl$nc), function(j)
        vapply(j:bl$nc, function(i)
          if (i == j) sprintf("log_L%d%d", i, j) else sprintf("L%d%d", i, j),
          character(1))), use.names = FALSE)
    } else {
      sprintf("log_sd_%d", seq_len(bl$nc))
    }
    label <- .re_cov_block_label(bl, m)
    if (M > 1L) nm <- paste0(label, ".", nm)
    idx <- pos + seq_len(bl$k)
    pos <<- pos + bl$k
    list(label = label, nc = bl$nc, full = bl$full,
         index = idx, coord = nm)
  })
}


# Optional PC prior on the RE-block marginal SDs for the AGHQ joint optimizer.
# `sigma_prior` is NULL (off), a `c(U, alpha)` pair (all blocks), or a list
# `list(blocks = <int>, prior_sigma = c(U, alpha), eta = <lkj, optional>)` (named
# blocks). Returns NULL when off, else a `list(val, grad)` acting on the full
# `par = c(theta, re_par)` vector: `val(par)` is the joint PC log-prior on the
# targeted blocks' `re_par` slice (via the shared .re_cov_block_logprior, so the
# PC + Jacobian algebra is the single source of truth), `grad(par)` its central-
# difference gradient over the touched RE coordinates only (zero on theta and on
# untargeted blocks). The prior depends on no oracle solve, so the FD gradient is
# cheap and exact to O(h^2); mixing it with the analytic data-gradient is valid
# (the total is data-score + prior-score).
.aghq_sigma_penalty <- function(sigma_prior, layout, n_theta) {
  if (is.null(sigma_prior)) return(NULL)
  if (is.list(sigma_prior)) {
    blks <- as.integer(sigma_prior$blocks %||% seq_along(layout))
    psig <- sigma_prior$prior_sigma
    eta_pc <- sigma_prior$eta %||% 1
  } else {
    blks <- seq_along(layout); psig <- sigma_prior; eta_pc <- 1
  }
  if (!is.numeric(psig) || length(psig) != 2L) {
    stop("`sigma_prior` must be `c(U, alpha)` or a list with `prior_sigma = ",
         "c(U, alpha)`.", call. = FALSE)
  }
  if (anyNA(blks) || any(blks < 1L) || any(blks > length(layout))) {
    stop("`sigma_prior$blocks` must index the RE blocks (1..", length(layout),
         ").", call. = FALSE)
  }
  ks  <- vapply(layout, `[[`, integer(1), "k")
  off <- cumsum(c(0L, ks))
  blk_fns <- lapply(seq_along(layout), function(m)
    if (m %in% blks)
      .re_cov_block_logprior(layout[[m]]$nc, layout[[m]]$full, psig, eta_pc)
    else NULL)
  touched <- unlist(lapply(which(!vapply(blk_fns, is.null, logical(1))),
                           function(m) off[m] + seq_len(ks[m])), use.names = FALSE)
  logprior_re <- function(rp) {
    lp <- 0
    for (m in seq_along(blk_fns)) if (!is.null(blk_fns[[m]]))
      lp <- lp + blk_fns[[m]](rp[off[m] + seq_len(ks[m])])
    lp
  }
  list(
    val = function(par) logprior_re(par[-seq_len(n_theta)]),
    grad = function(par) {
      g  <- numeric(length(par))
      rp <- par[-seq_len(n_theta)]; h <- 1e-5
      for (j in touched) {
        rpp <- rp; rpp[j] <- rpp[j] + h
        rpm <- rp; rpm[j] <- rpm[j] - h
        g[n_theta + j] <- (logprior_re(rpp) - logprior_re(rpm)) / (2 * h)
      }
      g
    })
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

# TRUE when `f` is the AGHQ objective's failure sentinel rather than an attained
# objective. The number itself is the compiled producer's
# (`kAghqFailPenalty`, src/aghq_re_core.h); nothing here writes it. The test is
# `<=` because the sentinel is what a failed solve reports EXACTLY, and no
# marginal log-likelihood reaches -1e10 -- a value at or below it is the
# sentinel, whatever arithmetic an R-side ridge or PC prior did to it after.
.aghq_is_fail <- function(f) !is.finite(f) || f <= cpp_aghq_fail_penalty()

# "%d of %d groups (1, 2, ...)" -- the phrase naming which per-group solves
# failed, shared by the post-fit warning and the refusal at the start point so
# the two cannot describe the same status differently.
.aghq_failed_group_phrase <- function(group_ok) {
  bad <- which(!group_ok)
  sprintf("%d of %d groups (%s%s)", length(bad), length(group_ok),
          paste(utils::head(bad, 5L), collapse = ", "),
          if (length(bad) > 5L) ", ..." else "")
}

# The per-group solve status at `par`, or NULL when even that could not be read.
# Used to name the groups behind a failed objective; a failure this reports is
# the same event the objective's sentinel is (both read aghq_group_solve, the
# extractor asking for strictly less), so it explains the sentinel rather than
# being a second opinion on it.
.aghq_group_status <- function(par, orc, nc_terms, full_vec) {
  ok <- tryCatch(as.logical(cpp_aghq_blups(par, orc, nc_terms, full_vec)$group_ok),
                 error = function(e) NULL)
  # All-TRUE does not explain a failed objective (the extractor asks for the
  # weaker of the two solves, so it can succeed where the objective's Cholesky
  # of C did not). Report nothing rather than a "0 of N groups ()" phrase.
  if (is.null(ok) || all(ok)) NULL else ok
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
      f <- if (isTRUE(r$ok)) r$f else cpp_aghq_fail_penalty()
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
