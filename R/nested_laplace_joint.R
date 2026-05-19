#' Joint multi-likelihood nested Laplace approximation
#'
#' @description
#' Outer-grid nested Laplace driver for *joint* models — multiple response
#' arms sharing one latent prior block, parameterized as a per-arm field
#' amplitude (sigma) on a unit-precision latent.
#'
#' Supported priors (Phase 1c):
#'  * `"bym2"`       — outer grid over `(sigma_spatial, rho [, sigma_pos])`.
#'                     Latent: `phi (n_s) | theta (n_s)` with unit-precision
#'                     ICAR + iid components.
#'  * `"icar"`       — outer grid over `(sigma_spatial [, sigma_pos])`.
#'                     Latent: `phi (n_s)` with unit-precision ICAR.
#'  * `"car_proper"` — outer grid over `(sigma_spatial, rho_car [, sigma_pos])`.
#'                     Latent: `phi (n_s)` with `Q = D - rho_car * W`.
#'
#' Other backends (NNGP, HSGP, RW1/2, AR1) follow the same interface and
#' land under Phase 3.
#'
#' @section (sigma, alpha) parameterization (gcol33/tulpa#22):
#' Each arm's linear predictor reads
#' \deqn{\eta_{arm} = X_{arm} \beta_{arm} + \sigma_{arm} \cdot z_s,}
#' where \eqn{z_s} is a unit-precision latent (ICAR(tau=1) for ICAR/BYM2,
#' or the BYM2 mix; CAR_proper uses the structure of
#' \eqn{D - \rho_{car} W}). All arms share the donor amplitude
#' \eqn{\sigma} from the outer-grid `sigma_grid` axis. The copy arm's
#' amplitude is \eqn{\sigma_{arm} = \alpha \cdot \sigma}, where \eqn{\alpha}
#' is a direct outer-grid axis taken from `copy$alpha_grid`. The Cartesian
#' product is over `(sigma, [rho/rho_car,] alpha[, phi])`. Direct \eqn{\alpha}
#' as an outer-grid axis (rather than a post-hoc ratio
#' \eqn{\alpha = \sigma_{pos} / \sigma_{occ}}) avoids plug-in bias on the
#' weakly-identified ratio at small `n_pos` and lets a regularizing
#' hyperprior land on \eqn{\alpha} directly.
#'
#' @param responses A named list of arm specs (length >= 1). Each arm:
#'   * `y`           — numeric `[N_arm]` response.
#'   * `n_trials`    — integer `[N_arm]` (use `rep(1L, N_arm)` for non-binomial).
#'   * `X`           — numeric matrix `[N_arm x p_arm]` fixed-effects design.
#'   * `spatial_idx` — integer `[N_arm]`, 1-based map obs -> spatial unit.
#'   * `re_idx`      — optional numeric `[N_arm]` 1-based RE group index;
#'                     defaults to `rep(0, N_arm)` (no RE).
#'   * `n_re_groups` — optional integer (default `0L`).
#'   * `sigma_re`    — optional numeric (default `1`); ignored when
#'                     `n_re_groups == 0`.
#'   * `family`      — one of `"binomial"`, `"gaussian"`, `"poisson"`,
#'                     `"neg_binomial_2"`, `"beta"`, `"lognormal"`,
#'                     `"gamma"`, `"inverse_gaussian"`. For `"lognormal"`,
#'                     `y` is on the natural scale and the linear
#'                     predictor `eta = E[log y]` (identity link on the
#'                     log scale); the `-log(y)` Jacobian is included in
#'                     the kernel's `log_lik`.
#'   * `phi`         — numeric dispersion (gaussian/lognormal residual
#'                     SD, negbin size, beta precision); default `1`.
#'
#' @param prior A list describing the shared latent prior block. Required
#'   field `type`. Backend-specific fields:
#'   * **bym2**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'     `n_neighbors`, `scale_factor` (default `1`); optional `sigma_grid`
#'     (donor-arm field amplitude, default 5 log-spaced values in
#'     `[0.1, 3]`), `rho_grid` (default `c(0.2, 0.5, 0.8, 0.95)`).
#'   * **icar**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'     `n_neighbors`; optional `sigma_grid` (default 5 log-spaced values
#'     in `[0.1, 3]`).
#'   * **car_proper**: same as icar plus `rho_car_grid`
#'     (default `c(0.5, 0.8, 0.95, 0.99)`).
#'
#' @param copy Optional list controlling the copy arm:
#'   * `arm`        — name (or 1-based index) of the copy arm.
#'   * `alpha_grid` — numeric grid for the copy coefficient \eqn{\alpha};
#'                    the copy arm's field amplitude at each cell is
#'                    \eqn{\alpha \cdot \sigma}. Default
#'                    `c(0, exp(seq(log(0.1), log(3), length.out = 5)))`.
#'   When `NULL` (default), no copy scaling is applied — all arms share
#'   the donor `sigma_grid` axis and \eqn{\alpha = 1} implicitly.
#'
#' @param phi_grid Optional list specifying per-arm dispersion axes on the
#'   outer grid. Accepts either a named list (keys = arm names) or a
#'   positional list of length `n_arms`. Each entry is one of:
#'   * `NULL` or scalar — no axis for that arm; the kernel uses the
#'     parse-time scalar `responses[[k]]$phi`.
#'   * numeric vector of length > 1 — adds a new outer-grid axis
#'     `phi_<arm>` taking those values; the kernel rewrites
#'     `arms[k].phi` at each grid point before the inner Newton solve.
#'
#'   Family-specific interpretation of `arm$phi` (the parse-time scalar
#'   and the grid values):
#'   * `gaussian` — residual SD (variance is `phi^2`). Use `phi_grid` to
#'     estimate the residual SD as a hyperparameter instead of pinning
#'     it pre-fit.
#'   * `lognormal` — residual SD on the log scale; identical kernel
#'     parameterization as `gaussian` plus the `-log(y)` Jacobian.
#'   * `neg_binomial_2` — dispersion (variance is `mu + mu^2/phi`).
#'   * `beta` — precision (variance is `mu(1-mu)/(1+phi)`).
#'   * `gamma`, `inverse_gaussian` — shape / dispersion.
#'   * `binomial`, `poisson` — ignored.
#'
#'   Each `phi_<arm>` axis is appended to the Cartesian product and
#'   varies slowest (within-spatial warm starts hold). The axis appears
#'   as a regular hyperparameter in `theta_grid`, `theta_mean`, and
#'   `theta_sd`, and participates in adaptive-grid refinement.
#'
#' @param prior_sigma,prior_alpha Optional regularizing hyperpriors on the
#'   donor field amplitude \eqn{\sigma} and on the copy coefficient
#'   \eqn{\alpha}. Each is `NULL` (flat, default) or a list of the form
#'   `list(family, params)`:
#'   * `list("pc.prec", c(U, alpha))` -- Penalized Complexity prior,
#'     calibrated by `P(theta > U) = alpha`. Closed-form density
#'     `lambda * exp(-lambda * theta)` with `lambda = -log(alpha)/U`.
#'     Drop-in for the weakly-identified small-`n_pos` regime
#'     (gcol33/tulpa#22): default-friendly choice on \eqn{\alpha} is
#'     `c(U = 1.0, alpha = 0.01)`, which shrinks the upper tail of
#'     \eqn{\alpha} without biasing the modal cell when the data
#'     identifies it.
#'   * `list("half_normal", scale)` -- half-normal with scale `scale > 0`.
#'     Sharper tail decay than PC; use when stronger regularization is
#'     desired and the truth is well inside the prior.
#'   The contribution is added to `log_marginal` cell-by-cell at the
#'   kernel-call boundary, so refinement passes (adaptive grid,
#'   var-of-means consistency) see the regularized posterior. When the
#'   data identifies the parameter (e.g. `n_pos >= ~200`) the prior is
#'   essentially harmless -- the lever is tail-shrinkage at small
#'   `n_pos`. `prior_alpha` only applies when `copy` is active;
#'   `prior_sigma` applies on any `sigma`-named axis.
#'
#' @param max_iter,tol Inner Newton iteration budget and tolerance.
#' @param n_threads Inner-loop OpenMP threads (per-observation scatter,
#'   compute_eta, log-likelihood reduction). Defaults to `1L` — for typical
#'   tulpa joint workloads (`N` in the hundreds to a few thousand) inner
#'   parallelism is overhead-dominated, see
#'   `dev_notes/issue_body_adaptive_grid_cost.md`. Coarse-grained
#'   parallelism over the outer hyperparameter grid is exposed via
#'   `n_threads_outer` and stacks better on hardware with many cores.
#' @param n_threads_outer Outer-grid OpenMP threads. When `> 1`, the driver
#'   runs a pilot Laplace at the centre cell, then dispatches the remaining
#'   cells across `n_threads_outer` OpenMP threads with each cell warm-started
#'   from the pilot mode. Each outer thread owns its own CHOLMOD solver and
#'   NewtonScratch; inner-loop OpenMP is auto-disabled to avoid nested
#'   parallelism overhead. Default `1L` (serial, chained warm-starts —
#'   bitwise identical to the pre-speedup driver). Recommended on multi-core
#'   workstations: `parallel::detectCores() - 1L`.
#' @param tile_warm Logical (default `TRUE`). When `n_threads_outer > 1`
#'   and a copy block is present, the driver groups outer-grid cells into
#'   *tiles* sharing every hyperparameter axis except the copy coefficient
#'   `alpha`, runs one Tier-2 warm solve per tile from the centre-cell
#'   pilot, and warm-starts the remaining cells from their tile pilot
#'   instead of the global pilot. Cuts inner-Newton iterations by a tile-
#'   sized factor on the slowest cells (boundary alpha values). Falls back
#'   silently to the single-tier Phase 1 path when no copy block is
#'   present, the grid has a single tile, or `n_threads_outer <= 1L`. Set
#'   to `FALSE` to recover the Phase 1 behaviour (e.g. for regression
#'   testing).
#' @param prune Logical (default `FALSE`). When `TRUE`, the driver runs a
#'   cheap-pass screening Laplace at the centre-cell pilot mode for every
#'   outer-grid cell (no inner Newton, just one `log_lik + log_prior`
#'   evaluation per cell at the pilot mode), softmax-normalises the
#'   screening log-marginals, and skips the full inner Newton on cells
#'   whose normalised weight falls below `prune_tol`. Pruned cells are
#'   marked with `log_marginal = -Inf`, `n_iter = 0`, and inherit the
#'   pilot mode as their `mode` row. The pilot cell itself is never pruned.
#'   Stacks with `n_threads_outer` (serial and parallel paths both honour
#'   the prune mask). Approximate: the screen uses `log|H_pilot|` for every
#'   cell rather than recomputing the Hessian, so cells where the data
#'   posterior shape differs sharply from the pilot can be misranked at
#'   the edges. Keep `prune_tol` conservative (`<= 1e-3`) when the cover
#'   posterior is data-rich; pruning is most useful when the outer grid
#'   has many low-mass tail cells.
#' @param prune_tol Numeric (default `1e-3`). Screening weight threshold
#'   below which cells are pruned. Cells whose softmax-normalised cheap
#'   log-marginal weight is `< prune_tol` skip the full inner Newton. Has
#'   no effect when `prune = FALSE`. Default keeps cells holding ~99.9%
#'   of cheap-pass mass.
#' @param x_init Optional warm-start for the first grid point's inner solve.
#' @param verbose Currently a no-op; reserved.
#' @param store_Q If `TRUE`, also return the per-grid joint precision Q
#'   (lower triangle, CSC) as `Q_csc_p_per_grid`, `Q_csc_i_per_grid`,
#'   `Q_csc_x_per_grid`, `Q_csc_n`. Lets callers compute INLA-style
#'   total-variance posterior moments (`Var-of-means + Mean-of-Var`) on
#'   inner latent coordinates such as fixed-effect betas. Default `FALSE`
#'   to keep the result lightweight.
#' @param adaptive_grid Logical (default `FALSE`). When `TRUE`, a mode-
#'   tracked 1D refinement pass is triggered on any hyperparameter axis
#'   whose marginal posterior weight on the boundary point(s) exceeds
#'   `adaptive_grid_edge_thresh`. New points are appended on that axis
#'   (one densification between the boundary and its neighbour, two
#'   outward extensions beyond the boundary on a log-spaced axis) and
#'   the kernel is evaluated at each new point paired with the modal
#'   other-axis values from the boundary cell — *not* the full
#'   cartesian product. Each slice cell carries a calibration term
#'   `log S_b` so it contributes to the joint softmax on the
#'   *marginal* scale (where `S_b` is the integrated relative weight
#'   of the other axes at the boundary). Cost per refinement is
#'   `O(n_new_points)` kernel solves, not
#'   `O(n_new_points * prod(other_axis_sizes))`. Fixes posterior CI
#'   under-coverage when truth sits near or at the user's grid edge.
#'   Opt-in for now; defaults to `FALSE` to preserve legacy fixed-grid
#'   behaviour for existing callers.
#' @param adaptive_grid_edge_thresh Numeric (default `0.02`). Refinement
#'   triggers when the per-axis edge score on the boundary of the
#'   refinable axis (currently `sigma_pos` only) exceeds this value. The
#'   score is
#'   `max(marginal_weight_at_boundary,
#'        exp(max_log_marginal_at_boundary - max_log_marginal_overall))`,
#'   i.e. the larger of the integrated weight on the boundary level and
#'   the relative integrand density at the boundary level. Catches both
#'   boundary pile-up (truth lies *at* a grid endpoint, weight is heavy
#'   there) and integrand truncation (integrand still has appreciable
#'   density at the boundary, but the cell width is so narrow it gets
#'   little integrated weight). The default `0.02` corresponds to ~4 log
#'   units of decay before refinement stops. Lower the threshold to
#'   refine more aggressively at the cost of extra kernel passes.
#' @param adaptive_grid_max_passes Integer (default `1L`). Maximum number
#'   of refinement passes. One pass typically suffices; two is rarely
#'   useful and inflates runtime.
#'
#' @return A list of class `c("tulpa_nested_laplace_joint",
#'   "tulpa_nested_laplace", "list")` with:
#'   * `theta_grid`, `theta_names` — outer-grid hyperparameter values
#'     (includes the derived `alpha` column when `copy` is set).
#'   * `log_marginal`, `weights` — per-grid-point log-marginal and integration
#'      weights (sum to 1).
#'   * `theta_mean`, `theta_sd` — posterior moments per hyperparameter,
#'      including the derived `alpha` moment computed from
#'      `sigma_pos / sigma_occ` over the joint grid.
#'   * `modes` — `[n_grid x n_x]` matrix of inner modes.
#'   * `n_iter` — inner Newton iterations per grid point.
#'   * `arm_layout` — list with per-arm `beta_start`, `re_start`,
#'      spatial offset(s) and `n_x` for decoding modes.
#'   * `prior`, `responses`, `copy` — echoed inputs.
#'   * `adaptive_grid_info` — when `adaptive_grid = TRUE`, a list with
#'      `triggered_axes` (character) and `n_points_added` (integer)
#'      describing the refinement passes. NULL otherwise.
#'   * `prune_cheap_log_marginal`, `prune_mask`, `prune_n_pruned`,
#'      `prune_tol` — present only when `prune = TRUE`. Cheap-pass log-
#'      marginals at every cell, a logical mask of pruned cells, the
#'      pruned-cell count, and the threshold actually applied. Pruned
#'      cells have `log_marginal = -Inf` so they get zero weight under
#'      `.nl_normalise_weights`.
#'
#' @seealso [tulpa_nested_laplace()] for the single-arm engine.
#' @export
tulpa_nested_laplace_joint <- function(responses,
                                       prior,
                                       copy = NULL,
                                       phi_grid = NULL,
                                       prior_sigma = NULL,
                                       prior_alpha = NULL,
                                       max_iter = 50L, tol = 1e-6,
                                       n_threads = 1L,
                                       n_threads_outer = 1L,
                                       tile_warm = TRUE,
                                       prune = FALSE,
                                       prune_tol = 1e-3,
                                       x_init = NULL, verbose = FALSE,
                                       store_Q = FALSE,
                                       adaptive_grid = FALSE,
                                       adaptive_grid_edge_thresh = 0.02,
                                       adaptive_grid_max_passes = 1L,
                                       var_of_means_consistency = TRUE,
                                       prior_sigma_occ = NULL,
                                       prior_sigma_pos = NULL) {
    if (!is.list(responses) || length(responses) < 1L) {
        stop("`responses` must be a non-empty list of arm specs.", call. = FALSE)
    }
    if (!is.list(prior)) {
        stop("`prior` must be a list with a `type` field, or a list of block specs.",
             call. = FALSE)
    }
    # Hard deprecation: the engine reparameterized from (sigma_occ, sigma_pos)
    # to (sigma, alpha). Legacy args raise an error rather than silently
    # translating, because the grid coverage differs (different posterior
    # values for the same prior intent). See accuracy.md / accuracy_phase1.md.
    if (!is.null(prior_sigma_occ)) {
        stop("`prior_sigma_occ` was renamed to `prior_sigma` after the ",
             "(sigma, alpha) reparameterization. Pass `prior_sigma` instead. ",
             "The PC / half-normal prior families are unchanged.", call. = FALSE)
    }
    if (!is.null(prior_sigma_pos)) {
        stop("`prior_sigma_pos` was replaced by `prior_alpha` after the ",
             "(sigma, alpha) reparameterization. Pass `prior_alpha` directly ",
             "(same PC / half-normal families). The translation ",
             "alpha = sigma_pos / sigma_occ depends on the grid, so silent ",
             "translation would change posterior values.", call. = FALSE)
    }
    # Parse user-specified regularizing hyperpriors on (sigma, alpha) once,
    # at the entry point. Validation errors raised here surface to the user
    # without going through the multi-block / single-block fork.
    fn_sigma <- .joint_parse_sigma_prior(prior_sigma, "prior_sigma")
    fn_alpha <- .joint_parse_sigma_prior(prior_alpha, "prior_alpha")

    # Detect multi-block prior (list-of-blocks). Routed to a separate
    # dispatch path that builds vector<LatentBlock> on the joint side.
    # Pruning toggle: the kernel takes prune_tol > 0 to mean "screen and
    # prune"; gate it here with the user-facing `prune` boolean so the toggle
    # composes cleanly (prune=FALSE always disables, prune=TRUE uses
    # prune_tol). prune_tol must be in [0, 1) — clamp out-of-range to 0
    # rather than erroring, because a zero tolerance is the safe no-op.
    if (!isTRUE(prune)) {
        prune_tol_eff <- 0.0
    } else {
        prune_tol_eff <- as.numeric(prune_tol)
        if (!is.finite(prune_tol_eff) || prune_tol_eff < 0 ||
            prune_tol_eff >= 1) {
            stop("`prune_tol` must be a finite numeric in [0, 1).",
                 call. = FALSE)
        }
    }

    if (.is_multi_block_prior_joint(prior)) {
        return(.joint_dispatch_multi(
            responses = responses, prior_list = prior, copy = copy,
            phi_grid = phi_grid,
            fn_sigma = fn_sigma, fn_alpha = fn_alpha,
            max_iter = max_iter, tol = tol, n_threads = n_threads,
            n_threads_outer = n_threads_outer,
            tile_warm = tile_warm,
            prune_tol = prune_tol_eff,
            x_init = x_init, verbose = verbose, store_Q = store_Q
        ))
    }
    if (is.null(prior$type)) {
        stop("`prior` must be a list with a `type` field, or a list of block specs.",
             call. = FALSE)
    }
    type <- tolower(prior$type)
    backend <- .joint_backends[[type]]
    if (is.null(backend)) {
        stop("tulpa_nested_laplace_joint(): unsupported prior$type '", type,
             "'. Supported: ",
             paste(shQuote(names(.joint_backends)), collapse = ", "),
             ".", call. = FALSE)
    }

    arms <- lapply(seq_along(responses), function(k) {
        a <- responses[[k]]
        .normalise_joint_arm(a, k)
    })

    cp <- .resolve_copy(copy, responses, prior, type)
    arm_names <- names(responses) %||% paste0("arm", seq_along(responses))
    phi_axes <- .normalise_phi_grid(phi_grid, arm_names)
    grids <- backend$build_grids(prior, cp$has_copy, cp$alpha_grid, phi_axes)

    res <- backend$call_kernel(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, isTRUE(store_Q),
                                arm_names = arm_names,
                                n_threads_outer = n_threads_outer,
                                tile_warm = tile_warm,
                                prune_tol = prune_tol_eff)

    # Bake the regularizing hyperprior on (sigma, alpha) into log_marginal
    # at the kernel-call boundary. Every cell carries the same prior
    # contribution ratio so refinement decisions (edge scores, modal cell
    # selection, var-of-means thresholds) all read the *regularized*
    # posterior. New cells appended in refinement passes get the prior
    # baked in via the same `hp_fn` closure threaded through
    # `.apply_axis_refinement`.
    hp_fn <- if (is.null(fn_sigma) && is.null(fn_alpha)) NULL else {
        function(grids_obj) {
            tg <- backend$theta_grid(grids_obj, cp$has_copy)
            .joint_hp_vec_for_grids(tg, fn_sigma, fn_alpha)
        }
    }
    if (!is.null(hp_fn)) {
        hp_init <- hp_fn(grids)
        if (!is.null(hp_init)) res$log_marginal <- res$log_marginal + hp_init
    }

    # Adaptive grid refinement. Detect heavy boundary mass on any axis
    # and append cartesian-product points covering an interior bisection
    # plus an outward extension on that axis. The merged grid is fed to
    # the same `backend$call_kernel`, and the C++ kernel is shape-agnostic
    # (it just consumes paired hyperparameter vectors of equal length),
    # so the legacy fixed-grid path and the refined path share a single
    # implementation — no primary-and-fallback branching.
    refine_info <- NULL
    if (isTRUE(adaptive_grid)) {
        refined <- .adaptive_refine_pass(
            grids       = grids,
            res         = res,
            backend     = backend,
            arms        = arms, prior = prior, cp = cp,
            max_iter = max_iter, tol = tol, n_threads = n_threads,
            x_init = x_init, store_Q = store_Q,
            edge_thresh = adaptive_grid_edge_thresh,
            max_passes  = adaptive_grid_max_passes,
            arm_names   = arm_names,
            hp_fn       = hp_fn
        )
        grids       <- refined$grids
        res         <- refined$res
        refine_info <- refined$info
    }

    res$theta_grid  <- backend$theta_grid(grids, cp$has_copy)
    res$theta_names <- colnames(res$theta_grid)
    res$weights     <- .nl_normalise_weights(res$log_marginal)
    res             <- .nl_posterior_moments(res, paste0("joint_", type))
    res             <- .joint_recalibrate_axis_moments(res)
    # Replace per-axis var-of-means SDs with Laplace-at-mode SDs where the
    # 3-point parabolic fit at the modal cell succeeds (gcol33/tulpa#20).
    res             <- .nl_refit_axis_sd_laplace(res)

    # Var-of-means consistency pass. Sharply peaked axes (gaussian
    # noise SD, beta phi at high n_pos) collapse joint weight onto a
    # single grid cell, so downstream `sum(w*x^2) - mean^2` underestimates
    # SD even when `theta_sd` (Laplace-at-mode) is correct. Add slice
    # points at `theta_mean +/- k*theta_sd` to repopulate the Laplace
    # support; var-of-means on the merged grid converges to Laplace SD,
    # which lets downstream packages use the legacy `weights * theta_grid`
    # pattern without reaching into `theta_sd` directly (gcol33/tulpa#21).
    if (isTRUE(var_of_means_consistency)) {
        consistency <- .nl_var_of_means_consistency_pass(
            grids = grids, res = res, backend = backend,
            arms = arms, prior = prior, cp = cp,
            max_iter = max_iter, tol = tol, n_threads = n_threads,
            x_init = x_init, store_Q = store_Q, arm_names = arm_names,
            hp_fn = hp_fn
        )
        if (consistency$n_added > 0L) {
            grids           <- consistency$grids
            res             <- consistency$res
            res$theta_grid  <- backend$theta_grid(grids, cp$has_copy)
            res$theta_names <- colnames(res$theta_grid)
            res$weights     <- .nl_normalise_weights(res$log_marginal)
            res             <- .nl_posterior_moments(res, paste0("joint_", type))
            res             <- .joint_recalibrate_axis_moments(res)
            res             <- .nl_refit_axis_sd_laplace(res)
        }
        res$var_of_means_consistency_info <- consistency$info
    }
    res             <- .joint_attach_alpha_moments(res, cp$has_copy)
    res$arm_layout  <- backend$layout(arms, prior)
    res$prior       <- prior
    res$responses   <- responses
    res$copy        <- copy
    res$adaptive_grid_info <- refine_info
    class(res) <- c("tulpa_nested_laplace_joint", "tulpa_nested_laplace", "list")
    res
}


# --- regularizing hyperpriors on sigma -------------------------------------
#
# Joint nested-Laplace defaults to a flat hyperprior on (sigma_occ,
# sigma_pos) -- the kernel returns `log_marginal[i]` proportional to
# `p(y | theta_i)` with theta uniform over the grid. At small `n_pos`
# the cover-arm likelihood is weakly identifying on sigma_pos and its
# marginal is right-skewed (inverse-gamma-ish tail). The derived ratio
# `alpha = sigma_pos / sigma_occ` inherits that skew: even after
# marginalizing the joint grid (so no plug-in MAP) the posterior median
# of alpha overshoots truth at small `n_pos` (gcol33/tulpa#22; D7 Cell B
# fixture, n_pos ~ 45, +8.5% geom bias). The well-identified sigma_occ
# axis doesn't compensate -- the bias is intrinsic to the sigma_pos
# marginal.
#
# Mitigation: a regularizing hyperprior on sigma_pos (and symmetrically
# on sigma_occ) that gently shrinks the upper tail without biasing the
# modal cell when the data identifies sigma_pos. Two families:
#
#  * "pc.prec" (Penalized Complexity, Simpson et al. 2017):
#       pi(sigma) = lambda exp(-lambda sigma),  lambda = -log(alpha)/U
#    Calibrated by `P(sigma > U) = alpha`. Default-friendly: pick U at
#    the upper end of plausible field amplitudes and alpha small.
#    Exponential tail decay; concentrates mass near zero only as much
#    as needed.
#
#  * "half_normal":
#       pi(sigma) = (2/(scale*sqrt(2*pi))) exp(-sigma^2/(2*scale^2))
#    Calibrated by `scale`. Sharper decay than PC -- penalizes large
#    sigma more aggressively, but also more aggressive on the mode if
#    `scale` is small relative to the truth.
#
# Both are normalized log-densities (the additive constant matters
# across the grid because grid cells share it, but it doesn't bias
# moments -- kept for downstream log-evidence consumers).
#
# When the data identifies sigma_pos (n_pos >= ~200, D5/D6 regimes),
# the likelihood concentrates well inside the prior's bulk and either
# family is essentially harmless. The lever is tail-shrinkage at
# small `n_pos`.
#
# Returns NULL for spec = NULL (flat prior, no contribution). Returns
# a function(sigma) -> log_density (vectorised, sigma > 0) otherwise.
# Validation errors are raised on malformed specs.
.joint_parse_sigma_prior <- function(spec, axis_label) {
    if (is.null(spec)) return(NULL)
    if (!is.list(spec) || length(spec) < 2L) {
        stop(sprintf(
            "`%s` must be a list of the form list(<family>, <params>), e.g. list(\"pc.prec\", c(U = 1.0, alpha = 0.01)).",
            axis_label), call. = FALSE)
    }
    fam <- as.character(spec[[1L]])
    pp  <- as.numeric(spec[[2L]])
    switch(fam,
        "pc.prec" = {
            if (length(pp) != 2L || !all(is.finite(pp)) ||
                pp[1L] <= 0 || pp[2L] <= 0 || pp[2L] >= 1) {
                stop(sprintf(
                    "`%s = list(\"pc.prec\", c(U, alpha))` requires U > 0 and 0 < alpha < 1; got U=%s, alpha=%s.",
                    axis_label, format(pp[1L]), format(pp[2L])),
                    call. = FALSE)
            }
            U <- pp[1L]; a <- pp[2L]
            lambda <- -log(a) / U
            # PC prior density at sigma=0 is the finite limit `lambda`,
            # not zero. The strict `s > 0` would zero the boundary grid
            # cell (sigma_pos = 0) and bias derived quantities like
            # alpha = sigma_pos / sigma_occ. Accept the closed half-line
            # [0, Inf).
            function(s) {
                ok <- is.finite(s) & s >= 0
                out <- rep(-Inf, length(s))
                out[ok] <- log(lambda) - lambda * s[ok]
                out
            }
        },
        "half_normal" = {
            if (length(pp) != 1L || !is.finite(pp) || pp <= 0) {
                stop(sprintf(
                    "`%s = list(\"half_normal\", scale)` requires scale > 0; got scale=%s.",
                    axis_label, format(pp)), call. = FALSE)
            }
            sc <- pp
            const <- log(2) - 0.5 * log(2 * pi) - log(sc)
            # Half-normal density at sigma=0 is the finite peak
            # 2/(scale*sqrt(2*pi)), not zero. Same closed-half-line
            # rationale as the PC branch above.
            function(s) {
                ok <- is.finite(s) & s >= 0
                out <- rep(-Inf, length(s))
                out[ok] <- const - s[ok]^2 / (2 * sc^2)
                out
            }
        },
        stop(sprintf(
            "Unknown hyperprior family for `%s`: '%s'. Supported families: 'pc.prec', 'half_normal'.",
            axis_label, fam), call. = FALSE)
    )
}

# Per-cell log-hyperprior contribution over a joint theta-grid. NULL fns
# contribute zero. Sums independent priors on sigma (donor amplitude)
# and alpha (copy coefficient). Cells missing either column (no-copy
# paths skip the alpha term) get zero contribution from that prior.
.joint_hp_vec_for_grids <- function(theta_grid, fn_sigma, fn_alpha) {
    if (is.null(fn_sigma) && is.null(fn_alpha)) return(NULL)
    if (is.null(theta_grid) || !is.matrix(theta_grid)) return(NULL)
    n <- nrow(theta_grid)
    out <- numeric(n)
    cn <- colnames(theta_grid)
    if (!is.null(fn_sigma) && "sigma" %in% cn) {
        out <- out + fn_sigma(as.numeric(theta_grid[, "sigma"]))
    }
    if (!is.null(fn_alpha) && "alpha" %in% cn) {
        out <- out + fn_alpha(as.numeric(theta_grid[, "alpha"]))
    }
    out
}


# --- backend dispatch table --------------------------------------------------
#
# Adding a new joint backend means appending one entry here. Each entry
# bundles the four backend-specific concerns: grid construction, kernel call,
# theta-grid materialisation for the result, and latent-layout metadata.

# Each entry now owns three concerns: grid construction, theta-grid
# materialisation, and latent-layout metadata. The kernel call is shared
# across all single-block backends -- they all route through
# `.joint_call_kernel_via_multi()` which packs the single-block prior into
# a length-1 multi-block spec and dispatches via
# `cpp_nested_laplace_joint_multi`. This is the J-E unification: one
# inner C++ entry, one R-side post-processing path. The legacy
# per-backend `cpp_nested_laplace_joint_{bym2,icar,car_proper}` shims and
# their bespoke LatentBlock construction are gone.
.joint_backends <- list(
    bym2 = list(
        build_grids = function(prior, has_copy, alpha_axis, phi_axes = NULL) {
            sigma_axis <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            rho_axis <- prior$rho_grid %||% c(0.2, 0.5, 0.8, 0.95)
            .joint_cartesian(list(sigma = sigma_axis, rho = rho_axis),
                              has_copy, alpha_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL, n_threads_outer = 1L,
                                tile_warm = TRUE, prune_tol = 0.0) {
            .joint_call_kernel_via_multi("bym2", arms, prior, cp, grids,
                                          max_iter, tol, n_threads,
                                          x_init, store_Q, arm_names,
                                          n_threads_outer = n_threads_outer,
                                          tile_warm = tile_warm,
                                          prune_tol = prune_tol)
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma = grids$sigma, rho = grids$rho,
                      alpha = grids$alpha)
            } else {
                cbind(sigma = grids$sigma, rho = grids$rho)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 2L,
                          spatial_block_names = c("phi_start", "theta_start"))
        }
    ),

    icar = list(
        build_grids = function(prior, has_copy, alpha_axis, phi_axes = NULL) {
            sigma_axis <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            .joint_cartesian(list(sigma = sigma_axis), has_copy,
                              alpha_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL, n_threads_outer = 1L,
                                tile_warm = TRUE, prune_tol = 0.0) {
            .joint_call_kernel_via_multi("icar", arms, prior, cp, grids,
                                          max_iter, tol, n_threads,
                                          x_init, store_Q, arm_names,
                                          n_threads_outer = n_threads_outer,
                                          tile_warm = tile_warm,
                                          prune_tol = prune_tol)
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma = grids$sigma, alpha = grids$alpha)
            } else {
                cbind(sigma = grids$sigma)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 1L,
                          spatial_block_names = "phi_start")
        }
    ),

    car_proper = list(
        build_grids = function(prior, has_copy, alpha_axis, phi_axes = NULL) {
            sigma_axis   <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            rho_car_axis <- prior$rho_car_grid %||% c(0.5, 0.8, 0.95, 0.99)
            .joint_cartesian(list(sigma = sigma_axis, rho_car = rho_car_axis),
                              has_copy, alpha_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL, n_threads_outer = 1L,
                                tile_warm = TRUE, prune_tol = 0.0) {
            .joint_call_kernel_via_multi("car_proper", arms, prior, cp, grids,
                                          max_iter, tol, n_threads,
                                          x_init, store_Q, arm_names,
                                          n_threads_outer = n_threads_outer,
                                          tile_warm = tile_warm,
                                          prune_tol = prune_tol)
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma = grids$sigma, rho_car = grids$rho_car,
                      alpha = grids$alpha)
            } else {
                cbind(sigma = grids$sigma, rho_car = grids$rho_car)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 1L,
                          spatial_block_names = "phi_start")
        }
    )
)

# Single-block call_kernel: route through cpp_nested_laplace_joint_multi by
# packing the legacy prior + per-arm spatial_idx into a length-1
# blocks_spec list and a theta_grid matrix matching the C++ side's axis
# conventions:
#
#   bym2  +copy: axes = (sigma_occ, sigma_pos, rho)   -- unit-precision
#                latent. R side feeds sigma_occ = sigma, sigma_pos =
#                alpha * sigma from the (sigma, alpha) outer grid; the
#                C++ reads two named columns and uses them as per-arm
#                scaling factors. Mathematically identical to the legacy
#                (sigma_occ, sigma_pos) layout cell-by-cell, but the SET
#                of cells covers alpha evenly instead of forming a
#                Cartesian product in sigma_pos.
#   bym2  -copy: axes = (sigma, rho)                  -- sigma rolled into
#                d_fac directly (no copy block).
#   icar  +copy: axes = (sigma_occ, sigma_pos)        -- as above.
#   icar  -copy: axes = (tau,)                        -- tau on prior;
#                grid$sigma is translated to tau = 1 / sigma^2.
#   car_proper +copy: (sigma_occ, sigma_pos, rho_car) -- as above.
#   car_proper -copy: (sigma, rho_car)                -- sigma in d_fac.
#
# The legacy backends used `grid$sigma` for both the donor amplitude
# (copy case) and the prior scale (no-copy case). The C++ kernel is
# unchanged after the reparameterization; only the physical cells it
# visits change.
.joint_call_kernel_via_multi <- function(type, arms, prior, cp, grids,
                                          max_iter, tol, n_threads,
                                          x_init, store_Q, arm_names,
                                          n_threads_outer = 1L,
                                          tile_warm = TRUE,
                                          prune_tol = 0.0) {
    n_arms <- length(arms)
    spi <- lapply(arms, function(a) as.integer(a$spatial_idx))

    block_spec <- list(
        type            = type,
        n_spatial_units = as.integer(prior$n_spatial_units),
        adj_row_ptr     = as.integer(prior$adj_row_ptr),
        adj_col_idx     = as.integer(prior$adj_col_idx),
        n_neighbors     = as.integer(prior$n_neighbors),
        spatial_idx     = spi
    )
    if (type == "bym2") {
        block_spec$scale_factor <- as.numeric(prior$scale_factor %||% 1.0)
    }

    # Construct theta_grid columns in the order the C++ kernel expects.
    # Names get a `b1.` prefix to match cpp_nested_laplace_joint_multi's
    # naming convention. The C++ axis names `sigma_occ` / `sigma_pos` are
    # preserved as a contract with the kernel; the R outer grid lives in
    # (sigma, alpha) space and materializes sigma_pos = alpha * sigma at
    # the kernel-call boundary.
    if (cp$has_copy) {
        sigma_occ_col <- as.numeric(grids$sigma)
        sigma_pos_col <- as.numeric(grids$sigma) * as.numeric(grids$alpha)
        cols <- switch(
            type,
            bym2       = list(b1.sigma_occ = sigma_occ_col,
                              b1.sigma_pos = sigma_pos_col,
                              b1.rho       = grids$rho),
            icar       = list(b1.sigma_occ = sigma_occ_col,
                              b1.sigma_pos = sigma_pos_col),
            car_proper = list(b1.sigma_occ = sigma_occ_col,
                              b1.sigma_pos = sigma_pos_col,
                              b1.rho_car   = grids$rho_car)
        )
        copy_block <- 0L
    } else {
        cols <- switch(
            type,
            bym2       = list(b1.sigma = grids$sigma, b1.rho = grids$rho),
            icar       = list(b1.tau   = 1.0 / (as.numeric(grids$sigma)^2)),
            car_proper = list(b1.sigma = grids$sigma, b1.rho_car = grids$rho_car)
        )
        copy_block <- -1L
    }
    theta_grid <- do.call(cbind, lapply(cols, as.numeric))
    colnames(theta_grid) <- names(cols)
    axis_offsets <- as.integer(c(0L, length(cols)))

    phi_grid_per_arm <- .joint_phi_grid_per_arm(grids, arm_names)

    # Tile partition for the three-tier warm-start (dev_notes/speedup.md
    # Opt 2). Tile = unique (sigma, [rho/rho_car,] phi_<arm>...) per cell,
    # i.e. every axis except the copy coefficient alpha. The joint mode is
    # tile-constant on the donor arm (Q + design depend on (sigma, rho)),
    # so any non-alpha cell in the same tile gives a much better warm-
    # start than the global pilot for the inner Newton at boundary alpha
    # cells. Only active when n_threads_outer > 1 and the user opts in
    # (tile_warm = TRUE, the default).
    tile_partition <- NULL
    if (isTRUE(tile_warm) && cp$has_copy &&
        as.integer(n_threads_outer) > 1L &&
        !is.null(grids$alpha) && length(grids$alpha) > 0L) {
        sigma_vec <- as.numeric(grids$sigma)
        n_grid <- length(sigma_vec)
        non_alpha_cols <- list(sigma = sigma_vec)
        if (!is.null(grids$rho)) {
            non_alpha_cols$rho <- as.numeric(grids$rho)
        }
        if (!is.null(grids$rho_car)) {
            non_alpha_cols$rho_car <- as.numeric(grids$rho_car)
        }
        phi_col_names <- grep("^phi_", names(grids), value = TRUE)
        for (nm in phi_col_names) {
            non_alpha_cols[[nm]] <- as.numeric(grids[[nm]])
        }
        non_alpha_mat <- do.call(cbind, non_alpha_cols)
        tile_partition <- .joint_compute_tile_partition(
            non_alpha_mat, as.numeric(grids$alpha), n_grid)
    }

    res <- cpp_nested_laplace_joint_multi(
        arms_list    = arms,
        copy_arm     = as.integer(cp$copy_arm_zero),
        copy_block   = copy_block,
        blocks_spec  = list(block_spec),
        theta_grid   = theta_grid,
        axis_offsets = axis_offsets,
        max_iter     = as.integer(max_iter),
        tol          = as.numeric(tol),
        n_threads    = as.integer(n_threads),
        x_init_nullable = x_init,
        store_Q      = isTRUE(store_Q),
        phi_grid_per_arm = phi_grid_per_arm,
        n_threads_outer = as.integer(n_threads_outer),
        tile_ids        = tile_partition$tile_ids,
        tile_pilot_cells = tile_partition$tile_pilot_cells,
        prune_tol       = as.numeric(prune_tol)
    )
    # Strip the C++-side theta_grid / axis_offsets — the backend's
    # `theta_grid()` callback rebuilds them with the user-facing bare
    # names in tulpa_nested_laplace_joint().
    res$theta_grid   <- NULL
    res$axis_offsets <- NULL
    res
}


# --- helpers -----------------------------------------------------------------

# Cartesian product over a named list of spatial axes plus an optional
# alpha axis (copy coefficient) and optional per-arm phi axes.
# `phi_axes` is a list keyed by arm name; entries are either NULL/empty (no
# axis for that arm) or numeric vectors that become a new outer-grid axis
# named `phi_<arm>`. Returns a named list of paired vectors of identical
# length, ready to feed the C++ kernel. Phi axes vary slowest (added last)
# so within-spatial-block warm-starts stay good.
.joint_cartesian <- function(axes, has_copy, alpha_axis, phi_axes = NULL) {
    full <- if (has_copy) c(axes, list(alpha = alpha_axis)) else axes
    if (!is.null(phi_axes)) {
        active <- phi_axes[vapply(phi_axes, length, integer(1)) > 0L]
        if (length(active) > 0L) {
            names(active) <- paste0("phi_", names(active))
            full <- c(full, active)
        }
    }
    gr <- do.call(expand.grid,
                  c(full, list(KEEP.OUT.ATTRS = FALSE,
                                stringsAsFactors = FALSE)))
    out <- as.list(gr)
    if (!has_copy) out$alpha <- numeric(0)
    out
}

# Append any `phi_<arm>` columns from `grids` onto a backend's spatial
# theta_grid matrix so downstream posterior-moment helpers see phi as a
# regular hyperparameter axis.
.append_phi_columns <- function(base, grids) {
    phi_cols <- grep("^phi_", names(grids), value = TRUE)
    if (length(phi_cols) == 0L) return(base)
    extra <- do.call(cbind, lapply(phi_cols, function(c) {
        out <- as.numeric(grids[[c]]); attr(out, "name") <- c; out
    }))
    colnames(extra) <- phi_cols
    cbind(base, extra)
}

# Build the `phi_grid_per_arm` argument for the C++ kernels from a
# Cartesian-product `grids` list and arm names. Returns a list of length
# `n_arms`: entry k is either `NULL` (no phi axis for that arm — kernel
# uses the parse-time scalar phi) or a NumericVector of length n_grid
# matching the flat outer-grid size. Phi columns in `grids` follow the
# `phi_<arm_name>` convention produced by `.joint_cartesian`.
.joint_phi_grid_per_arm <- function(grids, arm_names) {
    out <- vector("list", length(arm_names))
    any_active <- FALSE
    for (k in seq_along(arm_names)) {
        col <- paste0("phi_", arm_names[k])
        if (!is.null(grids[[col]])) {
            out[[k]] <- as.numeric(grids[[col]])
            any_active <- TRUE
        }
    }
    if (!any_active) NULL else out
}

# Compute a tile partition for the outer-grid loop's three-tier warm-start
# (Phase 2 of the speedup plan, dev_notes/speedup.md). A tile groups all
# outer-grid cells that share every hyperparameter coordinate except the
# copy coefficient alpha — for the joint copy block under the (sigma, alpha)
# reparam, the shared latent prior Q and the donor-arm linear predictor are
# tile-constant, so the joint mode varies smoothly across the alpha axis
# within a tile. Using the tile's median-alpha cell as a warm-start for the
# remaining alpha cells saves 1-2 Newton iters each.
#
# Arguments:
#   non_alpha_matrix - numeric matrix [n_grid x n_axes_excl_alpha]; row k
#                      is the non-alpha coordinates of cell k.
#   alpha_vec        - numeric [n_grid]; alpha value for each cell.
#   n_grid           - integer; number of outer-grid cells (= nrow).
#
# Returns NULL when the partition has no useful structure (every cell its
# own tile, or only one tile total); otherwise a list with 0-based
# integer fields:
#   tile_ids[k]        - tile membership of cell k.
#   tile_pilot_cells[t]- cell index used as tile t's representative pilot.
#                        When the global pilot cell (n_grid %/% 2) falls
#                        into tile t, it is reused so the Tier-2 pass
#                        does not re-solve that tile.
.joint_compute_tile_partition <- function(non_alpha_matrix, alpha_vec, n_grid) {
    if (is.null(non_alpha_matrix) || ncol(non_alpha_matrix) == 0L) return(NULL)
    if (nrow(non_alpha_matrix) != n_grid || length(alpha_vec) != n_grid) {
        return(NULL)
    }
    keys <- do.call(paste, c(
        lapply(seq_len(ncol(non_alpha_matrix)), function(j) {
            formatC(non_alpha_matrix[, j], digits = 15L, format = "g")
        }),
        list(sep = "\r")
    ))
    uniq_keys <- unique(keys)
    tile_ids <- match(keys, uniq_keys) - 1L  # 0-based
    n_tiles <- length(uniq_keys)
    if (n_tiles <= 1L || n_tiles >= n_grid) return(NULL)

    # Tier-1 global pilot cell matches the C++ driver's `n_grid / 2`.
    k_global_pilot <- as.integer(n_grid %/% 2L)
    tile_of_global <- tile_ids[k_global_pilot + 1L]

    tile_pilot_cells <- integer(n_tiles)
    for (t in seq_len(n_tiles) - 1L) {
        if (t == tile_of_global) {
            # Global pilot doubles as this tile's pilot — no Tier-2 solve.
            tile_pilot_cells[t + 1L] <- k_global_pilot
            next
        }
        cells_1based <- which(tile_ids == t)
        med <- stats::median(alpha_vec[cells_1based])
        best <- cells_1based[which.min(abs(alpha_vec[cells_1based] - med))][1L]
        tile_pilot_cells[t + 1L] <- as.integer(best - 1L)  # 0-based
    }
    list(tile_ids        = as.integer(tile_ids),
         tile_pilot_cells = as.integer(tile_pilot_cells))
}

# Normalise the user-facing `phi_grid` argument into a list keyed by arm
# name, with NULL entries for arms without a phi axis. Accepts either a
# named list (subset of arm names) or a positional list of length n_arms.
# Single-element entries are treated as no-axis (the parse-time scalar phi
# already serves as that arm's dispersion).
.normalise_phi_grid <- function(phi_grid, arm_names) {
    if (is.null(phi_grid)) return(NULL)
    if (!is.list(phi_grid)) {
        stop("`phi_grid` must be a list (named by arm or positional).",
             call. = FALSE)
    }
    out <- vector("list", length(arm_names))
    names(out) <- arm_names
    if (!is.null(names(phi_grid))) {
        unknown <- setdiff(names(phi_grid), arm_names)
        if (length(unknown) > 0L) {
            stop("`phi_grid` names not in `responses`: ",
                 paste(shQuote(unknown), collapse = ", "), ".", call. = FALSE)
        }
        for (nm in names(phi_grid)) {
            v <- phi_grid[[nm]]
            if (!is.null(v) && length(v) > 1L) out[[nm]] <- as.numeric(v)
        }
    } else {
        if (length(phi_grid) != length(arm_names)) {
            stop("positional `phi_grid` must have length n_arms (",
                 length(arm_names), ").", call. = FALSE)
        }
        for (k in seq_along(phi_grid)) {
            v <- phi_grid[[k]]
            if (!is.null(v) && length(v) > 1L) out[[k]] <- as.numeric(v)
        }
    }
    out
}

# Validate one arm spec and fill in defaults.
.normalise_joint_arm <- function(a, k) {
    if (!is.list(a)) {
        stop("Arm ", k, ": expected a list of arm spec fields.", call. = FALSE)
    }
    must_have <- c("y", "X", "spatial_idx", "family")
    missing <- setdiff(must_have, names(a))
    if (length(missing)) {
        stop("Arm ", k, ": missing fields ", paste(shQuote(missing), collapse = ", "),
             ".", call. = FALSE)
    }
    N <- length(a$y)
    a$y <- as.numeric(a$y)
    if (!is.matrix(a$X)) {
        stop("Arm ", k, ": `X` must be a numeric matrix.", call. = FALSE)
    }
    if (nrow(a$X) != N) {
        stop("Arm ", k, ": nrow(X) (", nrow(a$X), ") must equal length(y) (",
             N, ").", call. = FALSE)
    }
    if (length(a$spatial_idx) != N) {
        stop("Arm ", k, ": length(spatial_idx) (", length(a$spatial_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$spatial_idx <- as.integer(a$spatial_idx)
    a$n_trials <- if (is.null(a$n_trials)) rep(1L, N) else as.integer(a$n_trials)
    if (length(a$n_trials) != N) {
        stop("Arm ", k, ": length(n_trials) (", length(a$n_trials),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$re_idx <- if (is.null(a$re_idx)) rep(0, N) else as.numeric(a$re_idx)
    if (length(a$re_idx) != N) {
        stop("Arm ", k, ": length(re_idx) (", length(a$re_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$n_re_groups <- as.integer(a$n_re_groups %||% 0L)
    a$sigma_re    <- as.numeric(a$sigma_re %||% 1.0)
    a$family      <- as.character(a$family)
    a$phi         <- as.numeric(a$phi %||% 1.0)
    a
}

# Decide which arm (if any) is the copy arm and what the alpha grid is.
# After the (sigma, alpha) reparameterization, `copy$alpha_grid` is the
# only accepted spec. `copy$sigma_pos_grid` raises an error (the silent
# translation would change posterior values because the grid coverage
# differs).
.resolve_copy <- function(copy, responses, prior, type) {
    if (is.null(copy)) {
        return(list(has_copy = FALSE, copy_arm_zero = -1L,
                    alpha_grid = numeric(0)))
    }
    if (!is.null(copy$sigma_pos_grid)) {
        stop("`copy$sigma_pos_grid` was replaced by `copy$alpha_grid` after ",
             "the (sigma, alpha) reparameterization. Pass `alpha_grid` ",
             "directly. The translation alpha = sigma_pos / sigma depends ",
             "on the grid, so silent translation would change posterior ",
             "values.", call. = FALSE)
    }
    arm_id <- copy$arm
    if (is.null(arm_id)) {
        stop("`copy$arm` must be a name or 1-based index.", call. = FALSE)
    }
    if (is.character(arm_id)) {
        nm <- names(responses)
        if (is.null(nm) || !arm_id %in% nm) {
            stop("`copy$arm` = '", arm_id, "' not found in names(responses).",
                 call. = FALSE)
        }
        arm_zero <- match(arm_id, nm) - 1L
    } else {
        arm_zero <- as.integer(arm_id) - 1L
        if (arm_zero < 0L || arm_zero >= length(responses)) {
            stop("`copy$arm` index out of range.", call. = FALSE)
        }
    }
    if (!is.null(copy$alpha_grid)) {
        alpha_axis <- as.numeric(copy$alpha_grid)
    } else {
        # Default: a small log-spaced grid in [~0.1, ~3], with 0 included so
        # the alpha=0 base-model atom carries posterior mass when the data
        # supports "no copy".
        alpha_axis <- c(0, exp(seq(log(0.1), log(3), length.out = 5)))
    }
    if (length(alpha_axis) == 0L) {
        stop("`copy$alpha_grid` must have at least one non-negative value.",
             call. = FALSE)
    }
    if (any(alpha_axis < 0)) {
        stop("`copy$alpha_grid` values must be non-negative.",
             call. = FALSE)
    }
    list(has_copy = TRUE, copy_arm_zero = arm_zero,
         alpha_grid = alpha_axis)
}

# Recalibrate per-axis posterior moments after the joint pass. Slice
# cells from mode-tracked refinement on axis Y are pinned at modal
# (non-Y) values; including them in axis X's marginal (X != Y) collapses
# X to a point and shrinks Sd(X). Recompute mean/Sd for each column of
# `theta_grid` using only cells that vary that column — cartesian cells
# (`refining_axis == ""`) plus same-axis slice cells.
#
# Joint theta_mean / theta_sd come from `.nl_posterior_moments` and are
# left in place for axes with no foreign slice cells (the recompute is a
# no-op there). The original `theta_mean` / `theta_sd` are overwritten
# in place rather than augmented — downstream callers should read the
# axis marginal, not the cartesian-only joint moment.
.joint_recalibrate_axis_moments <- function(res) {
    refining <- res$refining_axis
    if (is.null(refining) || all(refining == "")) return(res)
    if (is.null(res$theta_grid) || !is.matrix(res$theta_grid)) return(res)
    cols  <- colnames(res$theta_grid)
    lm    <- res$log_marginal
    for (col in cols) {
        # Include `consistency_<col>` cells too: they vary on `col` only
        # (other axes pinned at mode) and ARE part of col's own slice. The
        # only cells we exclude are foreign-axis slices that fix `col` at
        # a single non-varying value.
        keep <- refining == "" | refining == col |
                refining == paste0("consistency_", col)
        if (all(keep)) next
        lm_k <- lm[keep]
        m    <- max(lm_k)
        w    <- exp(lm_k - m)
        w    <- w / sum(w)
        vals <- res$theta_grid[keep, col]
        mu   <- sum(w * vals)
        sd_v <- sqrt(max(0, sum(w * vals^2) - mu^2))
        res$theta_mean[[col]] <- mu
        res$theta_sd[[col]]   <- sd_v
    }
    res
}

# Weighted-quantile median + 2.5/97.5 empirical CI for alpha on the joint
# nested-Laplace grid. After the (sigma, alpha) reparameterization alpha
# is a primary grid axis; mean and SD are produced generically by
# `.nl_posterior_moments` + `.joint_recalibrate_axis_moments`, so this
# helper only adds the quantile-based summaries.
#
# `tg` is a matrix with named column `alpha`. `res` is a list with
# `log_marginal`. `refining` is the per-cell refining-axis mask: foreign-
# axis slice cells (which pin alpha at a single value while varying some
# other axis) are excluded; Cartesian and same-axis slice cells stay.
#
# Returns list(median, ci_lo, ci_hi); each scalar is NA when unavailable.
# Shared by the single-block and multi-block joint-fit code paths.
.alpha_grid_moments <- function(tg, res, refining) {
    out <- list(median = NA_real_, ci_lo = NA_real_, ci_hi = NA_real_)
    alpha_vec <- as.numeric(tg[, "alpha"])
    keep <- refining == "" | refining == "alpha" |
            refining == "consistency_alpha"
    use <- keep & is.finite(alpha_vec)
    if (sum(use) == 0L) return(out)

    lm_use <- res$log_marginal[use]
    ws     <- exp(lm_use - max(lm_use)); ws <- ws / sum(ws)
    qs     <- .nl_wtd_quantile(alpha_vec[use], ws, c(0.025, 0.5, 0.975))
    out$ci_lo  <- qs[1L]
    out$median <- qs[2L]
    out$ci_hi  <- qs[3L]
    out
}

# Attach weighted-quantile alpha summaries to the joint result. After
# the reparameterization, alpha is already a column of theta_grid via
# the backend's `theta_grid()` callback -- this helper only computes
# the median + 2.5/97.5 empirical CI and merges them with the standard
# theta_mean / theta_sd that `.nl_posterior_moments` already produced
# for every grid axis (alpha included).
.joint_attach_alpha_moments <- function(res, has_copy) {
    if (!isTRUE(has_copy)) return(res)
    tg <- res$theta_grid
    if (is.null(tg) || !("alpha" %in% colnames(tg))) return(res)
    refining <- res$refining_axis %||% rep("", nrow(tg))
    m <- .alpha_grid_moments(tg, res, refining)
    # theta_mean / theta_sd for alpha are already populated by
    # .nl_posterior_moments + .joint_recalibrate_axis_moments (alpha is
    # a regular axis). Append the quantile-based summaries here.
    res$theta_median <- c(res$theta_median, alpha = m$median)
    res$theta_ci_lo  <- c(res$theta_ci_lo,  alpha = m$ci_lo)
    res$theta_ci_hi  <- c(res$theta_ci_hi,  alpha = m$ci_hi)
    res
}

# Note (2026-05-19): the dedicated `.nl_alpha_consistency_step` is gone
# after the (sigma, alpha) reparameterization. With alpha as a primary
# grid axis, the generic `.nl_var_of_means_consistency_pass` handles it
# automatically -- alpha appears in `.refinable_axes` and the standard
# var-of-means trigger fires whenever the joint-grid alpha SD falls
# short of the Laplace-at-mode alpha SD. No bespoke delta-method-on-log
# composition, no anchoring through `sigma_pos = alpha * sigma_occ`.


# Compute per-arm latent offsets so callers can decode `modes` back into
# per-arm (beta, re) blocks plus the shared spatial block(s). For BYM2 the
# spatial block is two sub-blocks (phi, theta); for ICAR/CAR_proper it's
# just phi.
.joint_layout <- function(arms, n_spatial_units, n_spatial_blocks,
                          spatial_block_names) {
    n_arms <- length(arms)
    p_arm  <- vapply(arms, function(a) ncol(a$X),     integer(1))
    n_re   <- vapply(arms, function(a) a$n_re_groups, integer(1))

    beta_start <- integer(n_arms)
    cur <- 0L
    for (k in seq_len(n_arms)) {
        beta_start[k] <- cur
        cur <- cur + p_arm[k]
    }
    re_start <- integer(n_arms)
    for (k in seq_len(n_arms)) {
        re_start[k] <- cur
        cur <- cur + n_re[k]
    }
    spatial_starts <- integer(n_spatial_blocks)
    for (b in seq_len(n_spatial_blocks)) {
        spatial_starts[b] <- cur
        cur <- cur + n_spatial_units
    }
    n_x <- cur

    out <- list(
        n_arms     = n_arms,
        p          = p_arm,
        n_re       = n_re,
        beta_start = beta_start,
        re_start   = re_start,
        n_x        = n_x
    )
    for (b in seq_len(n_spatial_blocks)) {
        out[[spatial_block_names[b]]] <- spatial_starts[b]
    }
    out
}

# --- adaptive grid refinement -----------------------------------------------
#
# Math ground.
#
# The joint marginal is approximated by
#   p(y) ~ sum_k w_k * exp(L_k),  L_k = log p(y, x_hat_k | theta_k),
# with normalised weights w_k. Posterior moments for any hyperparameter
# theta_j read
#   E[theta_j]   = sum_k w_k * theta_jk
#   Var[theta_j] = sum_k w_k * theta_jk^2 - E[theta_j]^2.
# When the user's grid for axis j is {t_1, ..., t_m} and the true posterior
# concentrates beyond the endpoints, the discretisation truncates the
# integrand. The resulting moments are biased toward the grid centre and
# their variance is bounded by the span of the grid points carrying weight.
#
# Refinement strategy: mode-tracked 1D slice (gcol33/tulpa#19).
#   1. Normalise log_marginal -> weights, project onto each refinable axis
#      by summing weight over the other axes (marginal weight per level).
#   2. For each refinable axis, check whether the lowest or highest level
#      carries marginal weight (or peak integrand density) above
#      `edge_thresh`. If so, propose:
#        a) one densification point at the (geometric, for log-spaced)
#           midpoint between the boundary and its inward neighbour,
#        b) two outward extension points past the boundary, mirroring the
#           spacing from the boundary to its neighbour.
#   3. Locate the boundary MAP cell: argmax of `log_marginal` over rows
#      whose refining-axis value equals the boundary level. Read off the
#      modal values of every *other* axis (sigma_occ, rho/rho_car/tau,
#      phi_<arm>, ...) from that single cell.
#   4. Evaluate the kernel at each new axis point paired with the modal
#      other-axis values — one Newton-Laplace solve per new point, *not*
#      one per `(new_point) x (other_cartesian)`. For an outer grid with
#      M other-axis cells this is an M-fold cost cut vs. the legacy
#      cartesian refinement (often M = 100-400 in production).
#   5. Calibrate the slice cells onto the *marginal* scale before merging:
#      add `log S_b = logsumexp_{k: axis == boundary} (L_k - L_b_modal)`
#      to each slice cell's log_marginal. Without this, slice cells sit
#      at the conditional-MAP density and are systematically under-weighted
#      by a factor of S_b in the joint softmax. With it, slice cells
#      contribute the integrated marginal mass the cartesian refinement
#      would have produced (under the assumption that the conditional
#      posterior in the other axes is locally stable across the refining
#      axis — true at the tail because the other axes are anchored by the
#      rest of the grid, not the refining axis).
#   6. Append the slice cells to the joint grid. Modes and per-grid Q come
#      from the real kernel evaluation, so downstream total-variance and
#      field-decoding code reads correct values at the new cells.
#
# Single helper handles every refinable axis (currently `sigma_pos`); the
# legacy cartesian path is gone — one code path, no fixed-vs-refined
# branching.

# Decide which axes are eligible for refinement.
#
# `alpha` (copy coefficient): at small effective sample size the
#   cover-arm likelihood barely identifies alpha and the posterior tail
#   can extend past the user's grid endpoint. Boundary refinement
#   catches the truncation; interior densification handles the case
#   where the posterior peak sits between grid levels.
#
# `phi_<arm>` (per-arm dispersion, e.g. beta concentration): the joint
#   engine integrates dispersion over an outer log-spaced grid; coarsening
#   that grid is the main lever on baseline wall time, but the posterior
#   peak can sit between grid levels and a discretisation bias creeps in.
#   Boundary + interior densification (mode-tracked) lets callers ship a
#   smaller default grid without regressing phi-recovery.
#
# Donor sigma is the spatial prior amplitude and is treated as a
# deliberate prior choice; extending it requires explicit user opt-in
# (separate feature). Spatial mixing axes (rho, rho_car, tau) are
# similarly fixed for now — they have small grids and the integrand
# tends to span the user range.
.refinable_axes <- function(grids, cp_has_copy) {
    out <- character(0)
    if (cp_has_copy) {
        lev <- sort(unique(as.numeric(grids$alpha)))
        if (length(lev) >= 2L) out <- c(out, "alpha")
    }
    phi_cols <- grep("^phi_", names(grids), value = TRUE)
    for (col in phi_cols) {
        lev <- sort(unique(as.numeric(grids[[col]])))
        if (length(lev) >= 2L) out <- c(out, col)
    }
    out
}

# Per-axis edge diagnostics. Marginal weight gives the integrated posterior
# share on the boundary level — useful when the integrand has clearly
# decayed but the user's grid is just wide enough that the boundary
# *cell* still contributes. Integrand density at the boundary
# (`max_lm[boundary] - max_lm[overall]`, exponentiated) catches the
# complementary failure: when the boundary level is the heaviest *point-
# wise* mass but the user's grid is so narrow that very little is left
# inside it, so the boundary's *integrated* weight stays small while the
# truncated tail beyond the grid still carries posterior mass.
#
# We trigger refinement when EITHER criterion exceeds the threshold,
# which catches both failure modes with a single helper.
#
# Returns a named list keyed by axis with sorted unique `levels` and the
# scalar `min_score`, `max_score` to compare against `edge_thresh`.
.axis_edge_scores <- function(grids, log_marginal, axes) {
    if (length(log_marginal) == 0L) return(list())
    lm_max_total <- max(log_marginal)
    weights      <- .nl_normalise_weights(log_marginal)
    out <- list()
    for (a in axes) {
        v   <- as.numeric(grids[[a]])
        lev <- sort(unique(v))
        if (length(lev) < 2L) next
        # Integrated marginal weight at each boundary.
        w_lo <- sum(weights[v == lev[1L]])
        w_hi <- sum(weights[v == lev[length(lev)]])
        # Peak integrand density at each boundary (normalised by overall
        # peak): exp(max_lm@boundary - max_lm@overall).
        d_lo <- exp(max(log_marginal[v == lev[1L]])               - lm_max_total)
        d_hi <- exp(max(log_marginal[v == lev[length(lev)]])      - lm_max_total)
        out[[a]] <- list(
            levels    = lev,
            min_frac  = w_lo, max_frac = w_hi,
            min_dens  = d_lo, max_dens = d_hi,
            min_score = max(w_lo, d_lo),
            max_score = max(w_hi, d_hi)
        )
    }
    out
}

# Axis-aware spacing: sigma / alpha / tau live on a log scale; rho and
# rho_car live on a linear scale. Per-arm phi axes (`phi_<arm>`) follow
# the GP-lengthscale convention — strictly positive, log-spaced.
.axis_is_log_scale <- function(axis_name) {
    axis_name %in% c("sigma", "alpha", "tau", "sigma2",
                      "phi_gp", "lengthscale") ||
        startsWith(axis_name, "phi_")
}

# Natural domain clamp for bounded axes. NULL means unbounded (so just
# log-scale-positive for sigma/tau axes, which the log midpoint handles).
.axis_bounds <- function(axis_name) {
    if (axis_name == "sigma")     return(c(0, Inf))   # field amplitude >= 0
    if (axis_name == "alpha")     return(c(0, Inf))   # copy coefficient >= 0
    if (axis_name == "rho")       return(c(0, 1))     # BYM2/AR1 mixing fraction
    if (axis_name == "rho_car")   return(c(-Inf, 1))  # proper-CAR (eigenvalue gated upstream)
    if (startsWith(axis_name, "phi_")) return(c(0, Inf))  # dispersion > 0
    NULL
}

# Propose new levels on one axis. Always adds one densification point
# between the boundary and its inward neighbour. Adds an outward extension
# only when the boundary is genuinely in the integrand's *tail*. When the
# boundary is the local mode, extension is suppressed because the data has
# not signalled that the true peak lies further out — densification alone
# safely tightens the integration in the heavy region without expanding
# the prior support beyond what the user requested.
.propose_axis_extension <- function(axis_name, lev, side,
                                     extend_ok = TRUE) {
    log_scale <- .axis_is_log_scale(axis_name)
    bounds    <- .axis_bounds(axis_name)
    mid <- if (log_scale) {
        function(a, b) exp(0.5 * (log(a) + log(b)))
    } else {
        function(a, b) 0.5 * (a + b)
    }
    if (side == "max") {
        edge      <- lev[length(lev)]
        neighbour <- lev[length(lev) - 1L]
        densify   <- mid(neighbour, edge)
        extend1   <- if (log_scale) edge * (edge / neighbour)
                     else            edge + (edge - neighbour)
        # Second extension point catches seeds where the integrand is
        # still appreciable one step past the boundary; without it we
        # under-cover when the true peak sits two steps beyond the user
        # grid.
        extend2   <- if (log_scale) extend1 * (edge / neighbour)
                     else            extend1 + (edge - neighbour)
    } else {
        edge      <- lev[1L]
        neighbour <- lev[2L]
        densify   <- mid(edge, neighbour)
        extend1   <- if (log_scale) edge * (edge / neighbour)
                     else            edge - (neighbour - edge)
        extend2   <- if (log_scale) extend1 * (edge / neighbour)
                     else            extend1 - (neighbour - edge)
    }
    pts <- if (isTRUE(extend_ok)) c(densify, extend1, extend2) else densify
    if (!is.null(bounds)) {
        pts <- pts[pts >= bounds[1L] & pts <= bounds[2L]]
    }
    # De-duplicate vs the existing levels at numerical tolerance.
    keep <- vapply(pts, function(p) {
        all(abs(lev - p) > 1e-8 * max(1, abs(p)))
    }, logical(1))
    pts[keep]
}

# Propose midpoint(s) on one axis to densify around an interior peak.
# `mode_idx` is the level index of the peak (1 < mode_idx < length(lev));
# `do_left` adds the midpoint between (mode_idx - 1, mode_idx),
# `do_right` adds the midpoint between (mode_idx, mode_idx + 1).
# Spacing follows the axis scale (geometric mean on log-spaced axes,
# arithmetic on linear axes). De-duplicated against the existing grid.
.propose_interior_densification <- function(axis_name, lev, mode_idx,
                                             do_left = FALSE, do_right = FALSE) {
    log_scale <- .axis_is_log_scale(axis_name)
    mid <- if (log_scale) {
        function(a, b) exp(0.5 * (log(a) + log(b)))
    } else {
        function(a, b) 0.5 * (a + b)
    }
    pts <- numeric(0)
    if (do_left  && mode_idx > 1L)              pts <- c(pts, mid(lev[mode_idx - 1L], lev[mode_idx]))
    if (do_right && mode_idx < length(lev))     pts <- c(pts, mid(lev[mode_idx], lev[mode_idx + 1L]))
    if (length(pts) == 0L) return(pts)
    bounds <- .axis_bounds(axis_name)
    if (!is.null(bounds)) {
        pts <- pts[pts >= bounds[1L] & pts <= bounds[2L]]
    }
    keep <- vapply(pts, function(p) {
        all(abs(lev - p) > 1e-8 * max(1, abs(p)))
    }, logical(1))
    pts[keep]
}

# Build mode-tracked slice triples for one (axis, anchor-level) pair plus
# the matching marginal-scale calibration. For each new axis point we
# produce *one* triple at (axis = p, others = modal_at_anchor), not a
# full cartesian — this is the M-fold cost cut.
#
# `anchor_lev` is the existing axis level whose modal cell we use as the
# anchor for the conditional posterior on the other axes:
#   * boundary refinement (peak at/near edge): anchor is the boundary
#     level (outer for max-side, inner for min-side).
#   * interior densification (peak between grid levels): anchor is the
#     peak level; new points sit on either side. The conditional posterior
#     is locally stable across one grid step in either direction.
#
# Returns NULL when no new points were given. Otherwise:
#   triples       : named list of equal-length vectors, kernel-input shaped
#                   (alpha is present iff `cp_has_copy`).
#   calibration   : numeric, one entry per new point, added to that cell's
#                   raw log_marginal before it is merged into the joint
#                   grid. `log S_b` defined in the file-level math note.
#   warm_start_idx: index into the joint grid of the anchor MAP cell.
.new_mode_tracked_triples <- function(grids, log_marginal, axis_name,
                                       new_pts, anchor_lev, cp_has_copy) {
    if (length(new_pts) == 0L) return(NULL)
    v <- as.numeric(grids[[axis_name]])
    mask <- abs(v - anchor_lev) < 1e-12 * max(1, abs(anchor_lev))
    if (!any(mask)) return(NULL)
    anchor_lm   <- log_marginal[mask]
    k_map_local <- which.max(anchor_lm)
    idx_global  <- which(mask)[k_map_local]
    L_map       <- anchor_lm[k_map_local]
    # log S_a = log sum_{k @ anchor} exp(L_k - L_anchor_modal). >= 0.
    # Equals 0 when the conditional posterior collapses to a single cell,
    # log(M) at most (M = number of other-axis cells) when it is flat.
    calibration <- log(sum(exp(anchor_lm - L_map)))

    # Modal values of every *other* axis at the boundary MAP cell. Empty
    # axes (alpha when has_copy = FALSE) stay empty.
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "alpha")
    other_axes <- setdiff(full_axes, axis_name)
    triples <- vector("list", length(names(grids)))
    names(triples) <- names(grids)
    for (a in names(grids)) {
        if (a == axis_name) {
            triples[[a]] <- as.numeric(new_pts)
        } else if (a %in% other_axes && length(grids[[a]]) > 0L) {
            triples[[a]] <- rep(grids[[a]][idx_global], length(new_pts))
        } else {
            triples[[a]] <- numeric(0)
        }
    }
    list(triples       = triples,
         calibration   = rep(calibration, length(new_pts)),
         # Index into the original joint grid of the boundary MAP cell
         # whose modes are the best available warm-start for the slice
         # kernel call (first slice triple reuses it; later triples
         # chain prev_mode within the kernel).
         warm_start_idx = idx_global)
}

# Stitch the slice triples from multiple (axis, side) pairs into a single
# kernel-input shape, alongside one calibration vector. Drops any rows
# whose all-axis key already appears in `grids` (numerical tolerance via
# the same `%.10g` key format the legacy path used).
.concat_slice_triples <- function(triple_packs, grids, cp_has_copy) {
    if (length(triple_packs) == 0L) return(NULL)
    axes <- names(grids)
    combined <- vector("list", length(axes))
    names(combined) <- axes
    for (a in axes) {
        parts <- lapply(triple_packs, function(p) p$triples[[a]] %||% numeric(0))
        combined[[a]] <- as.numeric(unlist(parts, use.names = FALSE))
    }
    calib <- as.numeric(unlist(lapply(triple_packs, `[[`, "calibration"),
                                use.names = FALSE))
    n_new <- length(combined[[axes[1L]]])
    if (n_new == 0L) return(NULL)

    # De-duplicate against the existing grid on the active (non-empty)
    # axes. alpha when has_copy = FALSE has length 0 and is ignored.
    active <- vapply(axes, function(a) length(grids[[a]]) > 0L, logical(1))
    active_axes <- axes[active]
    fmt <- function(lst) {
        if (length(lst[[active_axes[1L]]]) == 0L) return(character(0))
        cols <- lapply(active_axes, function(a) sprintf("%.10g", lst[[a]]))
        do.call(paste, c(cols, sep = ":"))
    }
    new_keys <- fmt(combined)
    old_keys <- fmt(grids)
    keep <- !new_keys %in% old_keys
    if (!any(keep)) return(NULL)
    for (a in axes) {
        if (length(combined[[a]]) > 0L) combined[[a]] <- combined[[a]][keep]
    }
    calib <- calib[keep]
    if (!cp_has_copy) combined$alpha <- numeric(0)
    list(triples = combined, calibration = calib)
}

# Concatenate two kernel results row-wise, matching the run_nested_laplace_grid
# output layout (log_marginal, modes, n_iter, optional Q_csc_* lists).
# Also carries the per-cell `refining_axis` tag so per-axis moment
# recalibration downstream can drop slice cells that don't vary the axis
# whose marginal is being computed. Cells without an explicit tag (e.g.
# the initial cartesian pass) default to "" (= varies on every axis).
.concat_kernel_results <- function(a, b) {
    out <- a
    out$log_marginal <- c(a$log_marginal, b$log_marginal)
    out$n_iter       <- c(a$n_iter,       b$n_iter)
    out$n_grid       <- a$n_grid + b$n_grid
    if (!is.null(a$modes) && !is.null(b$modes)) {
        out$modes <- rbind(a$modes, b$modes)
    }
    tag_a <- a$refining_axis %||% rep("", length(a$log_marginal))
    tag_b <- b$refining_axis %||% rep("", length(b$log_marginal))
    out$refining_axis <- c(tag_a, tag_b)
    if (!is.null(a$Q_csc_p_per_grid) || !is.null(b$Q_csc_p_per_grid)) {
        out$Q_csc_p_per_grid <- c(a$Q_csc_p_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_p_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_i_per_grid <- c(a$Q_csc_i_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_i_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_x_per_grid <- c(a$Q_csc_x_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_x_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_n <- a$Q_csc_n %||% b$Q_csc_n
    }
    out
}

# Merge new axis triples into the kernel-input `grids` representation
# (paired vectors, alpha empty when has_copy = FALSE).
.merge_grids <- function(grids, new_triples, cp_has_copy) {
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "alpha")
    out <- grids
    for (a in full_axes) {
        out[[a]] <- c(grids[[a]], new_triples[[a]])
    }
    if (!cp_has_copy) out$alpha <- numeric(0)
    out
}

# Detect refinement triggers on one axis and return the slice triple_pack
# (or NULL if nothing triggers). Pulled out so the outer pass can apply
# axes sequentially and let later axes see the updated grid/posterior.
.detect_axis_refinement <- function(grids, res, edge_info_a, axis_name,
                                     edge_thresh, cp_has_copy) {
    ei  <- edge_info_a
    lev <- ei$levels
    v   <- as.numeric(grids[[axis_name]])
    lm_max_at_lev <- vapply(lev, function(lv) {
        max(res$log_marginal[v == lv])
    }, numeric(1))
    mode_idx <- which.max(lm_max_at_lev)
    n_lev    <- length(lev)

    # Boundary triggers (peak-at-edge or tail-mass-at-edge).
    tr_min <- ei$min_score >= edge_thresh &&
              (mode_idx == 1L || ei$min_dens >= edge_thresh)
    tr_max <- ei$max_score >= edge_thresh &&
              (mode_idx == n_lev || ei$max_dens >= edge_thresh)

    # Interior densification trigger. When the posterior on the refining
    # axis is narrower than the grid spacing, weights collapse to a
    # single grid point and the posterior mean snaps to that grid value
    # — no continuous interpolation. Adjacent-level density rounds to
    # zero in that regime, so a density-based trigger cannot catch the
    # discretisation bias. Trigger on *grid coarseness* instead: when
    # the log-axis neighbour ratio exceeds `interior_log_step`, an
    # interior peak is suspected of sub-grid mismatch — densify between
    # peak and neighbour. Thresholds tuned to "always fire on a 7-point
    # log grid over [2, 300] (ratio ~2.4), never fire on a 25-point
    # grid (ratio ~1.25)".
    interior_log_step <- 0.55
    interior_lin_step <- 0.25
    wide_left <- wide_right <- FALSE
    if (mode_idx > 1L && mode_idx < n_lev) {
        if (.axis_is_log_scale(axis_name)) {
            wide_left  <- (log(lev[mode_idx])     - log(lev[mode_idx - 1L])) >= interior_log_step
            wide_right <- (log(lev[mode_idx + 1L]) - log(lev[mode_idx]))     >= interior_log_step
        } else {
            span <- diff(range(lev))
            if (span > 0) {
                wide_left  <- (lev[mode_idx]     - lev[mode_idx - 1L]) / span >= interior_lin_step
                wide_right <- (lev[mode_idx + 1L] - lev[mode_idx])     / span >= interior_lin_step
            }
        }
    }

    if (!tr_min && !tr_max && !wide_left && !wide_right) return(NULL)
    packs <- list()
    if (tr_max) {
        pts <- .propose_axis_extension(axis_name, lev, "max", extend_ok = TRUE)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[n_lev],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (tr_min) {
        pts <- .propose_axis_extension(axis_name, lev, "min", extend_ok = TRUE)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[1L],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (wide_left || wide_right) {
        pts <- .propose_interior_densification(axis_name, lev, mode_idx,
                                                wide_left, wide_right)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[mode_idx],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (length(packs) == 0L) return(NULL)
    packs
}

# Run the slice kernel for one axis's triple_packs and merge the result
# into (res, grids). Warm-start from the first pack's anchor MAP cell.
# Tags each new cell with `axis_name` so per-axis moment recalibration
# downstream can exclude cells that fix this axis at a non-varying value
# from OTHER axes' marginals.
.apply_axis_refinement <- function(grids, res, triple_packs, axis_name,
                                    backend, arms, prior, cp,
                                    max_iter, tol, n_threads, x_init,
                                    store_Q, arm_names, hp_fn = NULL) {
    merged <- .concat_slice_triples(triple_packs, grids, cp$has_copy)
    if (is.null(merged)) return(list(grids = grids, res = res, n_new = 0L))
    new_triples <- merged$triples
    calibration <- merged$calibration

    # Warm-start the slice kernel call from the first pack's anchor MAP
    # mode. The kernel chains prev_mode forward within a call, so this
    # skips a cold Newton on triple 1; subsequent triples benefit
    # transitively. Slice cells live close in theta-space to the
    # anchor, so the converged anchor mode is the best initial guess.
    slice_x_init <- x_init
    if (!is.null(res$modes) && is.matrix(res$modes)) {
        idx0 <- triple_packs[[1L]]$warm_start_idx
        if (length(idx0) == 1L && idx0 >= 1L && idx0 <= nrow(res$modes)) {
            slice_x_init <- as.numeric(res$modes[idx0, ])
        }
    }

    res_extra <- backend$call_kernel(arms, prior, cp, new_triples,
                                      max_iter, tol, n_threads,
                                      slice_x_init, isTRUE(store_Q),
                                      arm_names = arm_names)
    # Lift slice cells onto the marginal scale before merging. Same
    # softmax invariant the cartesian path used to give us for free.
    res_extra$log_marginal  <- res_extra$log_marginal + calibration
    # Bake the regularizing hyperprior into the new cells' log_marginal
    # before merging, so the appended cells join an invariant where
    # every cell carries its prior contribution (gcol33/tulpa#22).
    if (!is.null(hp_fn)) {
        hp_new <- hp_fn(new_triples)
        if (!is.null(hp_new) && length(hp_new) == length(res_extra$log_marginal)) {
            res_extra$log_marginal <- res_extra$log_marginal + hp_new
        }
    }
    res_extra$refining_axis <- rep(axis_name, length(res_extra$log_marginal))

    res   <- .concat_kernel_results(res, res_extra)
    grids <- .merge_grids(grids, new_triples, cp$has_copy)
    n_new <- length(new_triples[[names(grids)[which(vapply(grids, length,
                                                            integer(1)) > 0L)[1L]]]])
    list(grids = grids, res = res, n_new = n_new)
}

# Priority order for refining axes. `alpha` (boundary truncation on the
# copy coefficient) is refined first; any `phi_<arm>` axes (interior
# densification — sharpens marginal recovery without shifting joint
# structure) are refined second so they read the post-`alpha`-extension
# modal. Other axes fall in declaration order.
.axis_refinement_order <- function(axes) {
    priority <- function(a) {
        if (a == "alpha") 1L
        else if (startsWith(a, "phi_")) 2L
        else 3L
    }
    axes[order(vapply(axes, priority, integer(1)), seq_along(axes))]
}

# Main entry: run up to `max_passes` of refinement. Each pass processes
# refinable axes *sequentially* (one kernel call per axis), so each axis
# sees the grid/posterior updates from any earlier axis in the same pass.
# Re-uses the backend kernel — one code path, no fixed-grid fallback.
# Stops early when no axis triggers.
.adaptive_refine_pass <- function(grids, res, backend, arms, prior, cp,
                                  max_iter, tol, n_threads, x_init, store_Q,
                                  edge_thresh, max_passes,
                                  arm_names = NULL, hp_fn = NULL) {
    info <- list(triggered_axes = character(0),
                 n_points_added = integer(0))
    if (max_passes < 1L) {
        return(list(grids = grids, res = res, info = NULL))
    }
    for (pass in seq_len(max_passes)) {
        axes <- .axis_refinement_order(.refinable_axes(grids, cp$has_copy))
        if (length(axes) == 0L) break
        any_triggered <- FALSE
        triggered_this_pass <- character(0)
        for (a in axes) {
            edge_info <- .axis_edge_scores(grids, res$log_marginal, a)
            ei <- edge_info[[a]]
            if (is.null(ei)) next
            packs <- .detect_axis_refinement(grids, res, ei, a,
                                              edge_thresh, cp$has_copy)
            if (is.null(packs)) next
            step <- .apply_axis_refinement(grids, res, packs, a, backend,
                                            arms, prior, cp, max_iter, tol,
                                            n_threads, x_init, store_Q,
                                            arm_names, hp_fn = hp_fn)
            if (step$n_new == 0L) next
            grids <- step$grids
            res   <- step$res
            triggered_this_pass <- c(triggered_this_pass, a)
            info$n_points_added <- c(info$n_points_added, step$n_new)
            any_triggered <- TRUE
        }
        if (!any_triggered) break
        info$triggered_axes <- c(info$triggered_axes,
                                  paste(triggered_this_pass, collapse = ","))
    }
    if (length(info$triggered_axes) == 0L) info <- NULL
    list(grids = grids, res = res, info = info)
}


# Propose new slice points on `axis_name` covering the Laplace-implied
# support `mu +/- k*sd` (k = 0.7, 1.5; symmetric, four points). Log-scale
# axes get points on the log scale via delta-method (log_sd = sd / mu).
# Points falling outside the axis's natural bounds, or within ~5% spacing
# of an existing level, are dropped. Returns numeric() if no usable points
# remain.
.propose_consistency_points <- function(axis_name, mu, sd, lev) {
    if (!is.finite(mu) || !is.finite(sd) || sd <= 0) return(numeric(0))
    is_log <- .axis_is_log_scale(axis_name)
    if (is_log) {
        if (mu <= 0) return(numeric(0))
        log_mu <- log(mu)
        log_sd <- sd / mu
        if (!is.finite(log_sd) || log_sd <= 0) return(numeric(0))
        pts <- exp(log_mu + c(-1.5, -0.7, 0.7, 1.5) * log_sd)
    } else {
        pts <- mu + c(-1.5, -0.7, 0.7, 1.5) * sd
    }
    bounds <- .axis_bounds(axis_name)
    if (!is.null(bounds)) {
        pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
    }
    if (length(pts) == 0L) return(numeric(0))
    keep <- vapply(pts, function(p) {
        if (is_log) {
            !any(abs(log(lev) - log(p)) < 0.05)
        } else {
            !any(abs(lev - p) < 0.05 * max(abs(lev), 1))
        }
    }, logical(1))
    pts[keep]
}

# Var-of-means consistency pass. For each refinable axis with a finite
# Laplace SD whose joint-grid var-of-means falls short of that SD by more
# than `tolerance`, add four Laplace-guided slice points anchored at the
# overall modal cell (other axes pinned). One kernel call per axis;
# re-uses `.apply_axis_refinement` for the merge.
#
# Triggers ONLY when the discrepancy exceeds `tolerance` (default 0.7),
# so axes whose default grid already resolves the marginal pay no
# extra kernel time. See gcol33/tulpa#21.
.nl_var_of_means_consistency_pass <- function(grids, res, backend, arms, prior,
                                               cp, max_iter, tol, n_threads,
                                               x_init, store_Q,
                                               arm_names = NULL,
                                               tolerance = 0.7,
                                               hp_fn = NULL) {
    refinable <- .refinable_axes(grids, cp$has_copy)
    info <- list(axes = character(0), n_added = integer(0),
                 vom_before = numeric(0), sd_laplace = numeric(0))
    n_added_total <- 0L
    if (length(refinable) == 0L) {
        return(list(grids = grids, res = res, info = NULL, n_added = 0L))
    }
    overall_mode_idx <- which.max(res$log_marginal)
    for (axis in refinable) {
        sd_lap <- res$theta_sd[[axis]] %||% NA_real_
        mu     <- res$theta_mean[[axis]] %||% NA_real_
        if (!is.finite(sd_lap) || !is.finite(mu) || sd_lap <= 0) next

        axis_vals <- as.numeric(res$theta_grid[, axis])
        vom_mean  <- sum(res$weights * axis_vals)
        vom_sd    <- sqrt(max(0, sum(res$weights * axis_vals^2) - vom_mean^2))
        if (vom_sd >= tolerance * sd_lap) next

        lev <- sort(unique(as.numeric(grids[[axis]])))
        new_pts <- .propose_consistency_points(axis, mu, sd_lap, lev)
        if (length(new_pts) == 0L) next

        # Anchor at the overall modal cell's value on `axis`. The slice
        # builder pins other axes at the cells where `axis == anchor_lev`
        # has the maximum log_marginal, which is the overall mode here.
        anchor_lev <- as.numeric(grids[[axis]])[overall_mode_idx]
        pack <- .new_mode_tracked_triples(grids, res$log_marginal, axis,
                                           new_pts, anchor_lev, cp$has_copy)
        if (is.null(pack)) next

        n_before <- length(res$log_marginal)
        step <- .apply_axis_refinement(grids, res, list(pack), axis, backend,
                                        arms, prior, cp, max_iter, tol,
                                        n_threads, x_init, store_Q,
                                        arm_names, hp_fn = hp_fn)
        if (step$n_new == 0L) next
        grids <- step$grids
        res   <- step$res
        # Re-tag the appended cells with a `consistency_<axis>` prefix.
        # `.apply_axis_refinement` tags them as `<axis>` (its standard slice
        # tag), but the consistency pass pins other axes at the modal cell
        # and these slices would distort posterior moments of *derived*
        # quantities that read multiple axes (e.g. alpha = sigma_pos /
        # sigma_occ in `.joint_attach_alpha_moments`). The prefix lets
        # `.joint_recalibrate_axis_moments` still pick them up for the
        # refined axis's own marginal (via the `consistency_<axis>` keep
        # check) while derived-quantity moments naturally skip them.
        n_after <- length(res$log_marginal)
        if (n_after > n_before && !is.null(res$refining_axis)) {
            new_idx <- (n_before + 1L):n_after
            res$refining_axis[new_idx] <- paste0("consistency_", axis)
        }
        info$axes        <- c(info$axes, axis)
        info$n_added     <- c(info$n_added, step$n_new)
        info$vom_before  <- c(info$vom_before, vom_sd)
        info$sd_laplace  <- c(info$sd_laplace, sd_lap)
        n_added_total    <- n_added_total + step$n_new
    }
    # Note (2026-05-19): after the (sigma, alpha) reparameterization,
    # alpha is a regular refinable axis -- the loop above already picks
    # it up through `.refinable_axes`. The previous bespoke alpha-axis
    # extension (via the delta-method-on-log-sigma composition) is gone.
    if (length(info$axes) == 0L) info <- NULL
    list(grids = grids, res = res, info = info, n_added = n_added_total)
}


# =============================================================================
# Multi-block joint dispatch (Phase J-B)
# =============================================================================
#
# Activated when `prior` is a list-of-blocks (each element has a `type`
# field). Routes to cpp_nested_laplace_joint_multi, which builds a
# std::vector<LatentBlock> on the joint side. At most one block can be
# the copy block; first ship restricts the copy block to spatial types
# (icar / bym2 / car_proper). See dev_notes/plan_multi_block_joint.md.

# Multi-block detection. Same shape as the single-arm side: a list whose
# elements are themselves named lists carrying a `type` field, and whose
# top-level entry does NOT carry a `type` (which would mark a single-block
# prior).
.is_multi_block_prior_joint <- function(p) {
    is.list(p) && is.null(p$type) && length(p) > 0 &&
        all(vapply(p, function(x) is.list(x) && !is.null(x$type), logical(1)))
}

# Validate one arm spec for the multi-block path. Same shape as
# `.normalise_joint_arm` *except* `spatial_idx` is no longer required at
# the arm level — per-arm idx vectors live inside each block spec instead.
# A trailing `spatial_idx = integer(0)` placeholder is set so the existing
# JointArm packaging path doesn't complain.
.normalise_joint_arm_multi <- function(a, k) {
    if (!is.list(a)) {
        stop("Arm ", k, ": expected a list of arm spec fields.", call. = FALSE)
    }
    must_have <- c("y", "X", "family")
    missing <- setdiff(must_have, names(a))
    if (length(missing)) {
        stop("Arm ", k, ": missing fields ",
             paste(shQuote(missing), collapse = ", "), ".", call. = FALSE)
    }
    N <- length(a$y)
    a$y <- as.numeric(a$y)
    if (!is.matrix(a$X)) {
        stop("Arm ", k, ": `X` must be a numeric matrix.", call. = FALSE)
    }
    if (nrow(a$X) != N) {
        stop("Arm ", k, ": nrow(X) (", nrow(a$X), ") must equal length(y) (",
             N, ").", call. = FALSE)
    }
    # spatial_idx is OPTIONAL in multi-block (per-arm idx lives on the block).
    # The legacy joint kernels still reach into parsed[k_arm].spatial_idx, but
    # the multi-block kernel does NOT, so a length-N placeholder of zeros is
    # safe -- it shuts up parse_joint_arms' length check without contributing
    # to eta.
    a$spatial_idx <- if (is.null(a$spatial_idx)) rep(0L, N)
                     else as.integer(a$spatial_idx)
    if (length(a$spatial_idx) != N) {
        stop("Arm ", k, ": length(spatial_idx) (", length(a$spatial_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$n_trials <- if (is.null(a$n_trials)) rep(1L, N) else as.integer(a$n_trials)
    if (length(a$n_trials) != N) {
        stop("Arm ", k, ": length(n_trials) (", length(a$n_trials),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$re_idx <- if (is.null(a$re_idx)) rep(0, N) else as.numeric(a$re_idx)
    if (length(a$re_idx) != N) {
        stop("Arm ", k, ": length(re_idx) (", length(a$re_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$n_re_groups <- as.integer(a$n_re_groups %||% 0L)
    a$sigma_re    <- as.numeric(a$sigma_re %||% 1.0)
    a$family      <- as.character(a$family)
    a$phi         <- as.numeric(a$phi %||% 1.0)
    a
}

# Resolve which block is the copy block (first-ship restriction: spatial
# types only). Accepts `copy$block` as a 1-based index. Returns a list:
#   has_copy, copy_arm_zero (0-based), copy_block_zero (0-based),
#   alpha_grid (numeric).
.resolve_copy_multi <- function(copy, responses, prior_list) {
    if (is.null(copy)) {
        return(list(has_copy = FALSE, copy_arm_zero = -1L,
                    copy_block_zero = -1L, alpha_grid = numeric(0)))
    }
    if (!is.null(copy$sigma_pos_grid)) {
        stop("`copy$sigma_pos_grid` was replaced by `copy$alpha_grid` after ",
             "the (sigma, alpha) reparameterization. Pass `alpha_grid` ",
             "directly.", call. = FALSE)
    }
    arm_id <- copy$arm
    if (is.null(arm_id)) {
        stop("`copy$arm` must be a name or 1-based index.", call. = FALSE)
    }
    if (is.character(arm_id)) {
        nm <- names(responses)
        if (is.null(nm) || !arm_id %in% nm) {
            stop("`copy$arm` = '", arm_id, "' not found in names(responses).",
                 call. = FALSE)
        }
        arm_zero <- match(arm_id, nm) - 1L
    } else {
        arm_zero <- as.integer(arm_id) - 1L
        if (arm_zero < 0L || arm_zero >= length(responses)) {
            stop("`copy$arm` index out of range.", call. = FALSE)
        }
    }
    block_id <- copy$block
    if (is.null(block_id)) {
        stop("`copy$block` must be a 1-based index into `prior` for the ",
             "multi-block joint driver. (First ship: only spatial blocks ",
             "can be copy blocks; see dev_notes/plan_multi_block_joint.md.)",
             call. = FALSE)
    }
    block_zero <- as.integer(block_id) - 1L
    if (block_zero < 0L || block_zero >= length(prior_list)) {
        stop("`copy$block` index (", block_id, ") out of range for length(prior) (",
             length(prior_list), ").", call. = FALSE)
    }
    block_type <- tolower(prior_list[[block_zero + 1L]]$type)
    spatial_types <- c("icar", "bym2", "car_proper")
    if (!block_type %in% spatial_types) {
        stop("`copy$block` points at type '", block_type, "'. First ship ",
             "restricts copy semantics to spatial types (",
             paste(shQuote(spatial_types), collapse = ", "), "). ",
             "See dev_notes/plan_multi_block_joint.md.", call. = FALSE)
    }
    if (!is.null(copy$alpha_grid)) {
        alpha_axis <- as.numeric(copy$alpha_grid)
    } else {
        # Default: a small log-spaced alpha grid with 0 included so the
        # "no copy" base model carries posterior mass when supported.
        alpha_axis <- c(0, exp(seq(log(0.1), log(3), length.out = 5)))
    }
    if (length(alpha_axis) == 0L) {
        stop("`copy$alpha_grid` must have at least one non-negative value.",
             call. = FALSE)
    }
    if (any(alpha_axis < 0)) {
        stop("`copy$alpha_grid` values must be non-negative.",
             call. = FALSE)
    }
    list(has_copy = TRUE, copy_arm_zero = arm_zero,
         copy_block_zero = block_zero, alpha_grid = alpha_axis)
}

# Per-block axis grid for the multi-block joint driver. When the block is
# the copy block (spatial only for first ship), the parameterisation
# uses (sigma, alpha[, rho/rho_car]) directly. Non-copy blocks use the
# standard single-arm conventions and reuse the `.NL_REGISTRY` defaults.
#
# Returns:
#   $grid    : matrix [n_block_cells x n_axes_for_block]
#   $names   : axis names
#   $prepared: block spec with defaults filled in (so downstream code can
#              read prior$adj_row_ptr etc. without re-checking presence)
.joint_block_axis_grid <- function(p, is_copy, alpha_grid,
                                    block_index) {
    type <- tolower(p$type)
    if (is_copy) {
        sigma_axis <- p$sigma_grid
        if (is.null(sigma_axis)) {
            sigma_axis <- exp(seq(log(0.1), log(3), length.out = 5))
        }
        sigma_axis <- as.numeric(sigma_axis)
        if (type == "icar") {
            gr <- expand.grid(sigma = sigma_axis,
                              alpha = alpha_grid,
                              KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)
        } else if (type == "bym2") {
            rho <- p$rho_grid %||% c(0.2, 0.5, 0.8, 0.95)
            gr <- expand.grid(sigma = sigma_axis,
                              alpha = alpha_grid,
                              rho   = as.numeric(rho),
                              KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)
        } else if (type == "car_proper") {
            rho_car <- p$rho_car_grid %||% c(0.5, 0.8, 0.95, 0.99)
            gr <- expand.grid(sigma   = sigma_axis,
                              alpha   = alpha_grid,
                              rho_car = as.numeric(rho_car),
                              KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)
        } else {
            stop("Block ", block_index, ": copy semantics not supported for ",
                 "type '", type, "' in first ship.", call. = FALSE)
        }
        return(list(grid     = as.matrix(gr),
                    names    = colnames(gr),
                    prepared = p))
    }
    # Non-copy: reuse the single-arm registry's grid construction. The
    # registry returns the standard (tau, [rho]) / (sigma, [rho]) axes.
    .nl_block_axis_grid(p)
}

# Convert one R-side block to the C++ `cpp_nested_laplace_joint_multi`
# format. Per-arm idx vectors live in the BLOCK spec (not the arm spec);
# the user supplies `spatial_idx` / `temporal_idx` / `obs_idx` as a list
# of length n_arms.
.joint_block_spec_for_cpp <- function(p, n_arms, block_index) {
    type <- tolower(p$type)
    if (type %in% c("icar", "bym2", "car_proper")) {
        if (is.null(p$spatial_idx)) {
            stop("Block ", block_index, " (type '", type, "'): ",
                 "`spatial_idx` is required as a list of length n_arms.",
                 call. = FALSE)
        }
        spatial_idx <- .multi_block_per_arm_idx(p$spatial_idx, n_arms,
                                                  block_index, "spatial_idx")
        out <- list(
            type            = type,
            spatial_idx     = spatial_idx,
            n_spatial_units = as.integer(p$n_spatial_units),
            adj_row_ptr     = as.integer(p$adj_row_ptr),
            adj_col_idx     = as.integer(p$adj_col_idx),
            n_neighbors     = as.integer(p$n_neighbors)
        )
        if (type == "bym2") {
            out$scale_factor <- as.numeric(p$scale_factor %||% 1.0)
        }
        out
    } else if (type %in% c("rw1", "rw2", "ar1")) {
        if (is.null(p$temporal_idx)) {
            stop("Block ", block_index, " (type '", type, "'): ",
                 "`temporal_idx` is required as a list of length n_arms.",
                 call. = FALSE)
        }
        temporal_idx <- .multi_block_per_arm_idx(p$temporal_idx, n_arms,
                                                   block_index, "temporal_idx")
        out <- list(
            type         = type,
            temporal_idx = temporal_idx,
            n_times      = as.integer(p$n_times)
        )
        if (type == "rw1") out$cyclic <- isTRUE(p$cyclic)
        out
    } else if (type == "iid") {
        if (is.null(p$obs_idx)) {
            stop("Block ", block_index, " (type 'iid'): ",
                 "`obs_idx` is required as a list of length n_arms.",
                 call. = FALSE)
        }
        obs_idx <- .multi_block_per_arm_idx(p$obs_idx, n_arms,
                                              block_index, "obs_idx")
        list(
            type    = "iid",
            obs_idx = obs_idx,
            n_units = as.integer(p$n_units)
        )
    } else {
        stop("Block type '", type, "' is not supported in multi-block joint priors.",
             call. = FALSE)
    }
}

# Coerce a per-arm idx field into a length-n_arms list of integer vectors.
# Accepts either a list (already per-arm) or a single vector (replicated
# across arms; useful when every arm shares the same indexing — e.g. years
# 1..T mapped one-to-one).
.multi_block_per_arm_idx <- function(idx, n_arms, block_index, field_name) {
    if (is.list(idx)) {
        if (length(idx) != n_arms) {
            stop("Block ", block_index, ": `", field_name, "` is a list of ",
                 length(idx), " entries but n_arms = ", n_arms, ".",
                 call. = FALSE)
        }
        lapply(idx, as.integer)
    } else {
        rep(list(as.integer(idx)), n_arms)
    }
}

# Main multi-block joint dispatch. Mirrors the structure of the
# single-block path (build grid, call C++, post-process) but accepts a
# list-of-blocks `prior_list` and a `copy` spec that points at a
# specific block.
.joint_dispatch_multi <- function(responses, prior_list, copy,
                                  phi_grid,
                                  fn_sigma = NULL,
                                  fn_alpha = NULL,
                                  max_iter, tol, n_threads,
                                  x_init, verbose, store_Q,
                                  n_threads_outer = 1L,
                                  tile_warm = TRUE,
                                  prune_tol = 0.0) {
    n_arms <- length(responses)
    arms <- lapply(seq_along(responses), function(k) {
        a <- responses[[k]]
        .normalise_joint_arm_multi(a, k)
    })
    arm_names <- names(responses) %||% paste0("arm", seq_along(responses))

    cp <- .resolve_copy_multi(copy, responses, prior_list)

    # Per-block axis grids (with copy-block parameterisation if applicable).
    B <- length(prior_list)
    per_block <- lapply(seq_len(B), function(b) {
        is_copy <- cp$has_copy && (b - 1L) == cp$copy_block_zero
        .joint_block_axis_grid(prior_list[[b]], is_copy, cp$alpha_grid, b)
    })
    block_grids <- lapply(per_block, function(x) x$grid)
    prepared    <- lapply(per_block, function(x) x$prepared)

    # Cartesian product of per-block axis grids.
    row_counts <- vapply(block_grids, nrow, integer(1))
    idx <- do.call(expand.grid, lapply(row_counts, seq_len))
    n_cells <- nrow(idx)
    if (n_cells > .NL_MULTI_GRID_HARD_CAP) {
        stop(sprintf(
            "Joint multi-block grid has %d cells (hard cap %d). Reduce per-block grid sizes or wait for CCD integration support.",
            n_cells, .NL_MULTI_GRID_HARD_CAP
        ), call. = FALSE)
    }
    if (n_cells > .NL_MULTI_GRID_WARN) {
        warning(sprintf(
            "Joint multi-block grid has %d cells (>%d). Each cell costs one inner Newton solve; reduce per-block grid sizes if this is slow. CCD integration is a follow-up.",
            n_cells, .NL_MULTI_GRID_WARN
        ), call. = FALSE)
    }

    joint_grid <- do.call(cbind, lapply(seq_along(block_grids), function(b) {
        block_grids[[b]][idx[[b]], , drop = FALSE]
    }))
    axis_counts  <- vapply(block_grids, ncol, integer(1))
    axis_offsets <- as.integer(c(0L, cumsum(axis_counts)))
    axis_names <- unlist(lapply(seq_along(block_grids), function(b) {
        paste0("b", b, ".", colnames(block_grids[[b]]))
    }))
    colnames(joint_grid) <- axis_names

    blocks_spec <- lapply(seq_along(prepared), function(b) {
        .joint_block_spec_for_cpp(prepared[[b]], n_arms, b)
    })

    # phi_grid (per-arm dispersion overrides on the outer grid): only
    # populated when the user supplied one. Same convention as the
    # single-block joint path.
    phi_axes <- .normalise_phi_grid(phi_grid, arm_names)
    phi_grid_per_arm_list <- NULL
    if (!is.null(phi_axes)) {
        # Embed phi axes into the joint grid as new columns. For now we
        # support the single-block convention: phi varies independently
        # of the latent-block grid, so phi axes multiply n_cells.
        active <- phi_axes[vapply(phi_axes, length, integer(1)) > 0L]
        if (length(active) > 0L) {
            phi_extra <- do.call(expand.grid,
                                  c(active, list(KEEP.OUT.ATTRS = FALSE,
                                                  stringsAsFactors = FALSE)))
            joint_idx <- expand.grid(joint = seq_len(nrow(joint_grid)),
                                       phi   = seq_len(nrow(phi_extra)),
                                       KEEP.OUT.ATTRS = FALSE)
            joint_grid <- cbind(joint_grid[joint_idx$joint, , drop = FALSE],
                                  as.matrix(phi_extra[joint_idx$phi, , drop = FALSE]))
            phi_cols <- paste0("phi_", names(active))
            colnames(joint_grid) <- c(axis_names, phi_cols)
            # phi columns don't belong to any latent block; the C++ entry
            # consumes them via phi_grid_per_arm, not the block axes.
            phi_grid_per_arm_list <- vector("list", length(arm_names))
            for (k in seq_along(arm_names)) {
                col <- paste0("phi_", arm_names[k])
                if (col %in% colnames(joint_grid)) {
                    phi_grid_per_arm_list[[k]] <- as.numeric(joint_grid[, col])
                }
            }
        }
    }

    # Build the C++-facing theta_grid. The C++ kernel reads `sigma_occ` /
    # `sigma_pos` column names on the copy block; the R-side outer grid
    # lives in (sigma, alpha) and materializes sigma_pos = alpha * sigma
    # here. Columns for non-copy blocks pass through unchanged.
    cpp_grid <- joint_grid[, seq_len(axis_offsets[B + 1L]), drop = FALSE]
    if (cp$has_copy) {
        b_copy   <- cp$copy_block_zero + 1L
        cols_b   <- (axis_offsets[b_copy] + 1L):axis_offsets[b_copy + 1L]
        bare_b   <- sub("^b[0-9]+\\.", "", colnames(cpp_grid)[cols_b])
        i_sigma  <- match("sigma", bare_b)
        i_alpha  <- match("alpha", bare_b)
        if (is.na(i_sigma) || is.na(i_alpha)) {
            stop(".joint_dispatch_multi: copy block missing 'sigma' or 'alpha' axis.",
                 call. = FALSE)
        }
        sigma_col <- as.numeric(cpp_grid[, cols_b[i_sigma]])
        alpha_col <- as.numeric(cpp_grid[, cols_b[i_alpha]])
        cpp_grid[, cols_b[i_sigma]] <- sigma_col            # -> sigma_occ
        cpp_grid[, cols_b[i_alpha]] <- alpha_col * sigma_col # -> sigma_pos
        new_names <- colnames(cpp_grid)
        new_names[cols_b[i_sigma]] <- paste0("b", b_copy, ".sigma_occ")
        new_names[cols_b[i_alpha]] <- paste0("b", b_copy, ".sigma_pos")
        colnames(cpp_grid) <- new_names
    }

    # Tile partition for the three-tier warm-start (Phase 2 of
    # dev_notes/speedup.md). Tile axis = every joint_grid column EXCEPT the
    # copy block's alpha column. Built from the *user-facing* (sigma,
    # alpha) grid (before sigma_pos materialisation) so the partition
    # reflects what is constant across alpha at fixed (sigma, rho, ...
    # other-block axes).
    tile_partition <- NULL
    if (isTRUE(tile_warm) && cp$has_copy &&
        as.integer(n_threads_outer) > 1L) {
        b_copy_R <- cp$copy_block_zero + 1L
        cols_bc  <- (axis_offsets[b_copy_R] + 1L):axis_offsets[b_copy_R + 1L]
        bare_bc  <- sub("^b[0-9]+\\.", "", colnames(joint_grid)[cols_bc])
        i_alpha  <- match("alpha", bare_bc)
        if (!is.na(i_alpha)) {
            alpha_col_idx <- cols_bc[i_alpha]
            non_alpha_idx <- setdiff(seq_len(ncol(joint_grid)), alpha_col_idx)
            tile_partition <- .joint_compute_tile_partition(
                joint_grid[, non_alpha_idx, drop = FALSE],
                as.numeric(joint_grid[, alpha_col_idx]),
                nrow(joint_grid)
            )
        }
    }

    res <- cpp_nested_laplace_joint_multi(
        arms_list    = arms,
        copy_arm     = as.integer(cp$copy_arm_zero),
        copy_block   = as.integer(cp$copy_block_zero),
        blocks_spec  = blocks_spec,
        theta_grid   = cpp_grid,
        axis_offsets = axis_offsets,
        max_iter     = as.integer(max_iter),
        tol          = as.numeric(tol),
        n_threads    = as.integer(n_threads),
        x_init_nullable = x_init,
        store_Q      = isTRUE(store_Q),
        phi_grid_per_arm = phi_grid_per_arm_list,
        n_threads_outer = as.integer(n_threads_outer),
        tile_ids        = tile_partition$tile_ids,
        tile_pilot_cells = tile_partition$tile_pilot_cells,
        prune_tol       = as.numeric(prune_tol)
    )

    # Bake the regularizing hyperprior on (sigma, alpha) into log_marginal
    # (gcol33/tulpa#22). Multi-block has no in-package refinement passes,
    # so one apply at the kernel-call boundary suffices. Columns in
    # joint_grid are prefixed `b<N>.` -- build a bare-named view over the
    # copy block's (sigma, alpha) columns to feed the shared helper.
    if (!is.null(fn_sigma) || !is.null(fn_alpha)) {
        view_map <- integer(0)
        for (b_idx in seq_len(B)) {
            cols_b   <- (axis_offsets[b_idx] + 1L):axis_offsets[b_idx + 1L]
            bare_b   <- sub("^b[0-9]+\\.", "", colnames(joint_grid)[cols_b])
            i_sigma  <- match("sigma", bare_b)
            i_alpha  <- match("alpha", bare_b)
            if (!is.na(i_sigma) && is.na(view_map["sigma"])) {
                view_map["sigma"] <- cols_b[i_sigma]
            }
            if (!is.na(i_alpha) && is.na(view_map["alpha"])) {
                view_map["alpha"] <- cols_b[i_alpha]
            }
        }
        if (length(view_map) > 0L) {
            view <- joint_grid[, view_map, drop = FALSE]
            colnames(view) <- names(view_map)
            hp <- .joint_hp_vec_for_grids(view, fn_sigma, fn_alpha)
            if (!is.null(hp) && length(hp) == length(res$log_marginal)) {
                res$log_marginal <- res$log_marginal + hp
            }
        }
    }

    res$theta_grid   <- joint_grid
    res$theta_names  <- colnames(joint_grid)
    res$axis_offsets <- axis_offsets
    res$weights      <- .nl_normalise_weights(res$log_marginal)
    res <- .joint_posterior_moments_multi(res, prepared, axis_offsets,
                                           joint_grid, cp)
    # Replace per-axis var-of-means SDs with Laplace-at-mode SDs at the
    # modal cell (gcol33/tulpa#20).
    res <- .nl_refit_axis_sd_laplace(res)
    res$arm_layout  <- .joint_multi_layout(arms, prepared)
    res$blocks      <- prepared
    res$prior       <- prior_list
    res$responses   <- responses
    res$copy        <- copy
    class(res) <- c("tulpa_nested_laplace_joint_multi",
                    "tulpa_nested_laplace_joint",
                    "tulpa_nested_laplace", "list")
    res
}

# Latent-vector layout for the multi-block joint result. Mirrors the
# single-block `.joint_layout()` but generalised over `prepared` blocks:
# per-arm beta, per-arm RE, then each prepared block in order. For
# back-compat with single-block consumers (e.g. tulpaObs cover-hurdle's
# `.joint_field_at_obs_copy_multi`), the first spatial-like block also
# emits `phi_start` / `theta_start` aliases:
#
#   * BYM2  -> phi_start, theta_start (length-2 sub-block).
#   * ICAR / CAR_proper -> phi_start (length-1).
#
# Non-spatial blocks expose `block_start[b]` so callers can index into
# any block by ordinal position.
.joint_multi_layout <- function(arms, prepared) {
    n_arms <- length(arms)
    p_arm  <- vapply(arms, function(a) ncol(a$X),     integer(1))
    n_re   <- vapply(arms, function(a) a$n_re_groups, integer(1))

    beta_start <- integer(n_arms)
    cur <- 0L
    for (k in seq_len(n_arms)) {
        beta_start[k] <- cur
        cur <- cur + p_arm[k]
    }
    re_start <- integer(n_arms)
    for (k in seq_len(n_arms)) {
        re_start[k] <- cur
        cur <- cur + n_re[k]
    }

    B <- length(prepared)
    block_start <- integer(B)
    block_size  <- integer(B)
    phi_start   <- NULL
    theta_start <- NULL
    for (b in seq_len(B)) {
        block_start[b] <- cur
        type <- tolower(prepared[[b]]$type %||% "")
        # Latent size per block. BYM2 is length-2 (phi ICAR + theta IID)
        # on n_spatial_units; all other blocks are length-1 on the
        # block's `n_spatial_units` / `n_times` / `n_units`.
        n_units <- prepared[[b]]$n_spatial_units %||%
                   prepared[[b]]$n_times         %||%
                   prepared[[b]]$n_units         %||% 0L
        n_units <- as.integer(n_units)
        if (type == "bym2") {
            block_size[b] <- 2L * n_units
            if (is.null(phi_start)) {
                phi_start   <- cur
                theta_start <- cur + n_units
            }
        } else {
            block_size[b] <- n_units
            if (is.null(phi_start) && type %in% c("icar", "car_proper")) {
                phi_start <- cur
            }
        }
        cur <- cur + block_size[b]
    }
    n_x <- cur

    out <- list(
        n_arms      = n_arms,
        p           = p_arm,
        n_re        = n_re,
        beta_start  = beta_start,
        re_start    = re_start,
        block_start = block_start,
        block_size  = block_size,
        n_x         = n_x
    )
    if (!is.null(phi_start))   out$phi_start   <- phi_start
    if (!is.null(theta_start)) out$theta_start <- theta_start
    out
}

# Posterior moments for the multi-block joint grid. Mirrors the single-arm
# multi-block helper (`.nl_posterior_moments_multi`) and additionally
# attaches alpha = sigma_pos / sigma_occ when a copy block is active.
.joint_posterior_moments_multi <- function(res, prepared, axis_offsets,
                                            joint_grid, cp) {
    w <- res$weights
    # Joint moments across every column of joint_grid (including phi
    # columns appended for per-arm dispersion overrides).
    res$theta_mean <- as.numeric(crossprod(w, joint_grid))
    names(res$theta_mean) <- colnames(joint_grid)
    res$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, joint_grid^2)) -
                                  res$theta_mean^2))
    names(res$theta_sd) <- colnames(joint_grid)

    B <- length(prepared)
    per_block_moments <- vector("list", B)
    for (b in seq_len(B)) {
        cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
        sub <- joint_grid[, cols, drop = FALSE]
        block_mean <- as.numeric(crossprod(w, sub))
        block_sd   <- sqrt(pmax(0, as.numeric(crossprod(w, sub^2)) - block_mean^2))
        # Strip the "b<N>." prefix on per-block moment names — block index
        # is implicit in the list position.
        bare_names <- sub("^b[0-9]+\\.", "", colnames(sub))
        names(block_mean) <- bare_names
        names(block_sd)   <- bare_names
        per_block_moments[[b]] <- list(
            type      = tolower(prepared[[b]]$type),
            mean      = block_mean,
            sd        = block_sd,
            axis_cols = cols
        )
    }
    res$block_moments <- per_block_moments

    # alpha is a primary grid axis after the (sigma, alpha) reparameterization.
    # `theta_mean / theta_sd` on `alpha` are already populated by the
    # crossprod above (alpha is one of joint_grid's columns). Compute the
    # weighted-quantile median + empirical 2.5/97.5 CI here so the multi-
    # block path exposes the same `theta_median / theta_ci_lo / theta_ci_hi`
    # surface as the single-block path.
    if (cp$has_copy) {
        b <- cp$copy_block_zero + 1L
        cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
        col_names <- sub("^b[0-9]+\\.", "", colnames(joint_grid)[cols])
        i_alpha <- match("alpha", col_names)
        if (!is.na(i_alpha)) {
            alpha_vec <- as.numeric(joint_grid[, cols[i_alpha]])
            tg_local <- cbind(alpha = alpha_vec)
            res_local <- list(log_marginal = res$log_marginal)
            refining <- res$refining_axis %||% rep("", length(alpha_vec))
            m <- .alpha_grid_moments(tg_local, res_local, refining)
            if (is.finite(m$median)) {
                res$theta_median <- c(res$theta_median, alpha = m$median)
                res$theta_ci_lo  <- c(res$theta_ci_lo,  alpha = m$ci_lo)
                res$theta_ci_hi  <- c(res$theta_ci_hi,  alpha = m$ci_hi)
            }
        }
    }
    res
}
