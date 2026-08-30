# Which lever an outer Pareto-k miss actually takes (gcol33/tulpa#629).
#
# #629 proposed that a genuinely SKEWED hyperparameter posterior is unreachable
# by the k_quality ladder, and that the proposal-side rescues
# (.joint_pareto_score_mixture / _skew) are the untouched lever for it. Measured
# over 165 synthetic configurations (dev_notes/issue629/RESULTS629.md), neither
# half holds, and these are the facts the verdict rests on. They are pinned here
# because each is a property a future change to the dispatch could silently
# invert.
#
# The device is the one test-joint-pareto-k-proposal.R already uses: a `res`
# carrying only theta_grid / weights / axis_offsets / blocks plus an analytic
# `refit_log_marginal`. A `log` axis carries log_jac = 0, so the target the PSIS
# sees is exactly g(u) with u = log(theta) -- the shape is CONTROLLED and its
# true skewness / kurtosis are known, rather than inferred from a fit.

`%||%` <- function(a, b) if (is.null(a)) b else a

# A skew-normal in u: skewed, but Gaussian-tailed on both sides.
.lv_skew <- function(s, alpha) function(u)
    stats::dnorm(u / s, log = TRUE) + stats::pnorm(alpha * u / s, log.p = TRUE)

# A Student-t in u: symmetric, genuinely heavy-tailed.
.lv_heavy <- function(s, df) function(u) stats::dt(u / s, df, log = TRUE)

.lv_res <- function(lg, n_nodes, half_width, s) {
    u <- seq(-half_width * s, half_width * s, length.out = n_nodes)
    l <- lg(u); l <- l - max(l)
    w <- exp(l); w <- w / sum(w)
    list(theta_grid = matrix(exp(u), ncol = 1L,
                             dimnames = list(NULL, "b1.sigma")),
         weights = w, axis_offsets = c(0L, 1L),
         blocks = list(list(type = "icar")))
}

# Median over seeds of a per-candidate scoring, with the candidates the shipped
# dispatch skipped computed as counterfactuals on the continuing stream.
.lv_score <- function(lg, n_nodes, half_width, s = 0.4, n_samples = 500L,
                      seeds = 1:15) {
    res   <- .lv_res(lg, n_nodes, half_width, s)
    refit <- function(theta_mat) lg(log(theta_mat[, 1]))
    prep  <- tulpa:::.joint_pareto_prepare(res, refit, n_samples, NULL)
    vary  <- tulpa:::.joint_pareto_vary_axes(prep$Su)
    num <- function(x) if (is.null(x) || !length(x)) NA_real_
                       else as.numeric(x)[1]
    rows <- lapply(seeds, function(sd_i) {
        set.seed(sd_i)
        g   <- tulpa:::.joint_pareto_score(prep, vary, refit, n_samples)
        gm  <- g$gm %||% g
        mix <- tryCatch(tulpa:::.joint_pareto_score_mixture(prep, vary, refit,
                                                            n_samples),
                        error = function(e) NULL)
        mom <- if (is.null(g)) NULL
               else tulpa:::.joint_pareto_wtd_moments(g$U, g$log_weights,
                                                      g$prop_u, g$prop_L)
        sk <- if (is.null(mom)) NULL
              else tryCatch(tulpa:::.joint_pareto_score_skew(
                       prep, vary, refit, n_samples, g$prop_u, g$prop_L, mom),
                   error = function(e) NULL)
        set.seed(sd_i)
        disp <- tulpa:::.joint_pareto_score_dispatch(prep, vary, refit,
                                                     n_samples)
        c(k_gm    = if (is.list(gm)) num(gm$pareto_k) else NA_real_,
          k_gauss = num(g$pareto_k), k_mix = num(mix$pareto_k),
          k_skew  = num(sk$pareto_k),
          k_full  = if (tulpa:::.k_is_decline(disp)) NA_real_
                    else num(disp$best$pareto_k),
          skew_est = if (is.null(mom)) NA_real_ else max(abs(mom$skew)))
    })
    m <- do.call(rbind, rows)
    stats::setNames(apply(m, 2L, stats::median, na.rm = TRUE), colnames(m))
}

# --------------------------------------------------------------------------- #
# 1. Skewness alone does not produce a bad outer k-hat                         #
# --------------------------------------------------------------------------- #

test_that("a skewed but light-tailed hyperparameter posterior reads good", {
    # shape 12 is essentially a half-normal: true skewness 0.967, true excess
    # kurtosis 0.837. The whitened skewness the rescue gate reads is large, and
    # the single-Gaussian k-hat is nonetheless deep in the good band -- the
    # importance ratio against a Gaussian proposal stays bounded because the
    # target's tails are Gaussian on the right and LIGHTER on the left.
    sc <- .lv_score(.lv_skew(0.4, 12), n_nodes = 25L, half_width = 6)
    expect_gt(sc[["skew_est"]], tulpa:::.K_DIAG_SKEW_MIN)   # visibly skewed
    expect_lt(sc[["k_gm"]], tulpa:::.K_DIAG_GOOD)           # and still good
    # So there is nothing for a proposal-side rung to lift: the skew-normal
    # rescue only fires above the good band, and here it is never reached.
})

test_that("the skew-normal rescue cannot absorb a heavy tail", {
    # The regime that DOES defeat a Gaussian proposal. The skew-normal has
    # Gaussian tails on both sides, so it scores WORSE than the Gaussian it is
    # asked to rescue -- the property .joint_pareto_score_skew's own comment
    # relies on to call the rescue safe.
    sc <- .lv_score(.lv_heavy(0.4, 8), n_nodes = 25L, half_width = 3)
    expect_gt(sc[["k_gm"]], 1)                              # unreliable
    expect_gt(sc[["k_skew"]], sc[["k_gauss"]])              # rescue is worse
    expect_lt(sc[["k_mix"]], sc[["k_gm"]])                  # mixture is better
})

# --------------------------------------------------------------------------- #
# 2. The grid rung is live where #629 assumed it was exhausted                 #
# --------------------------------------------------------------------------- #

test_that("widening a heavy-tailed grid lowers the k-hat; densifying does not", {
    lg <- .lv_heavy(0.4, 2)
    narrow_coarse <- .lv_score(lg, n_nodes =  5L, half_width = 3)
    narrow_dense  <- .lv_score(lg, n_nodes = 41L, half_width = 3)
    wide_dense    <- .lv_score(lg, n_nodes = 41L, half_width = 12)

    # Densifying a grid that is too NARROW moves the k-hat the wrong way: the
    # proposal sharpens onto the covered region while the target's tail mass
    # stays outside it.
    expect_gt(narrow_dense[["k_gm"]], narrow_coarse[["k_gm"]])
    # Widening is the half that works, and it reaches the good band. The shipped
    # "grid" rung extends the boundary where integrand mass piles at an edge, so
    # it carries this half -- a rung that only densified would not.
    expect_lt(wide_dense[["k_gm"]], tulpa:::.K_DIAG_GOOD)
    expect_lt(wide_dense[["k_gm"]], narrow_dense[["k_gm"]] - 1)
})

# --------------------------------------------------------------------------- #
# 3. The layer decomposition the four backends differ by                       #
# --------------------------------------------------------------------------- #

test_that("the joint dispatch reads a strictly better k than its first pass", {
    # .nested_grid_pareto_k / .spde_pareto_k / .nested_outer_pareto_k each call
    # .nested_is_pareto_k ONCE, so what they report is the raw grid-moment
    # Gaussian -- `k_gm` here. The joint path adds moment matching and the two
    # rescues and keeps the minimum, so it can never read worse, and on a
    # heavy-tailed posterior it reads a different BAND.
    sc <- .lv_score(.lv_heavy(0.4, 8), n_nodes = 25L, half_width = 3)
    expect_lte(sc[["k_gauss"]], sc[["k_gm"]])
    expect_lte(sc[["k_full"]],  sc[["k_gauss"]] + 1e-8)
    expect_gt(sc[["k_gm"]],   tulpa:::.nl_diag("k_usable"))   # unreliable
    expect_lt(sc[["k_full"]], tulpa:::.nl_diag("k_usable"))   # usable
})
