# Outer Pareto-k mode-Hessian proposal (gcol33/tulpa#116).
#
# When the hyperparameter posterior is sharp the integration grid concentrates
# on too few cells to estimate the proposal covariance from grid weights; the
# residual far-cell weight then yields a degenerate covariance and a spurious
# k-hat. The fix builds the importance proposal from a mode Hessian instead:
# the CCD integrator's analytic curvature when present (spliced over its axes),
# else a finite-difference Hessian of the outer target at the modal cell (the
# tensor-grid path). These tests pin both routes on synthetic `.joint_pareto_k`
# inputs (fast) plus two end-to-end fits.

# A two-axis single block (bym2: sigma -> log, rho -> logit01) with the
# integration weight fully concentrated on one cell == a collapsed grid.
.pk_collapsed_res <- function() {
    tg <- as.matrix(expand.grid(`b1.sigma` = c(0.8, 1.0, 1.2),
                                `b1.rho`   = c(0.3, 0.5, 0.7)))
    colnames(tg) <- c("b1.sigma", "b1.rho")
    K <- nrow(tg)
    centre <- which(abs(tg[, 1] - 1.0) < 1e-9 & abs(tg[, 2] - 0.5) < 1e-9)
    w <- rep(0, K); w[centre] <- 1
    list(theta_grid = tg, weights = w,
         axis_offsets = c(0L, 2L), blocks = list(list(type = "bym2")))
}

# Smooth inner log-marginal: Gaussian in the unconstrained coordinate, so the
# FD-Hessian is well conditioned and the importance ratios are well behaved.
.pk_refit <- function(u_hat, s = c(0.2, 0.3)) {
    function(theta_mat) {
        u1 <- log(theta_mat[, 1]); u2 <- stats::qlogis(theta_mat[, 2])
        -0.5 * ((u1 - u_hat[1])^2 / s[1]^2 + (u2 - u_hat[2])^2 / s[2]^2)
    }
}

# --------------------------------------------------------------------------- #
# Tensor-path finite-difference Hessian fallback                              #
# --------------------------------------------------------------------------- #

test_that("a collapsed tensor grid recovers a finite k-hat from the FD Hessian", {
    res   <- .pk_collapsed_res()
    u_hat <- c(log(1.0), stats::qlogis(0.5))
    kd <- tulpa:::.joint_pareto_k(res, .pk_refit(u_hat), n_samples = 200L,
                                  proposal = NULL)
    expect_false(is.na(kd$pareto_k))
    expect_identical(kd$proposal_source, "mode_hessian")
    expect_true(is.finite(kd$is_ess) && kd$is_ess > 0)
    expect_lt(kd$pareto_k, 0.7)             # Gaussian target under Gaussian proposal
})

test_that("a collapsed grid with degenerate curvature declines to NA", {
    # A flat outer target gives a non-PD Hessian: the FD fallback declines, and
    # with all weight on one cell the grid covariance is exactly zero, so the
    # diagnostic declines to NA (-> quad-ESS downstream) rather than inventing a
    # proposal.
    res  <- .pk_collapsed_res()
    flat <- function(theta_mat) rep(0, nrow(theta_mat))
    kd <- tulpa:::.joint_pareto_k(res, flat, n_samples = 200L, proposal = NULL)
    expect_true(is.na(kd$pareto_k))
    expect_true(is.na(kd$proposal_source))
})

# A collapsed grid carrying pinned axes (gcol33/tulpa#117). Mirrors the Abies
# occu_cover repro: two icar blocks each (sigma -> log, alpha -> identity) plus a
# trailing phi_pos dispersion axis. The two `alpha` axes are pinned at 0 (no
# copy on the cover arm) and `phi_pos` collapses to one grid value, so 3 of 5
# axes carry zero weighted variance. Weight is sharp but not a delta, so the
# sigma axes keep nonzero variance while the pinned axes stay exactly zero;
# ess_grid <= d triggers the #116 fallback.
.pk_pinned_res <- function() {
    tg <- as.matrix(expand.grid(`b1.sigma` = c(0.8, 1.0, 1.2),
                                `b1.alpha` = 0,
                                `b2.sigma` = c(0.25, 0.35, 0.50),
                                `b2.alpha` = 0,
                                `phi_pos`  = 1.0))
    centre <- which(abs(tg[, "b1.sigma"] - 1.0)  < 1e-9 &
                    abs(tg[, "b2.sigma"] - 0.35) < 1e-9)
    sig1lo <- which(abs(tg[, "b1.sigma"] - 0.8)  < 1e-9 &
                    abs(tg[, "b2.sigma"] - 0.35) < 1e-9)
    sig1hi <- which(abs(tg[, "b1.sigma"] - 1.2)  < 1e-9 &
                    abs(tg[, "b2.sigma"] - 0.35) < 1e-9)
    sig2lo <- which(abs(tg[, "b1.sigma"] - 1.0)  < 1e-9 &
                    abs(tg[, "b2.sigma"] - 0.25) < 1e-9)
    sig2hi <- which(abs(tg[, "b1.sigma"] - 1.0)  < 1e-9 &
                    abs(tg[, "b2.sigma"] - 0.50) < 1e-9)
    w <- rep(0, nrow(tg))
    w[centre] <- 0.8
    w[c(sig1lo, sig1hi, sig2lo, sig2hi)] <- 0.05         # spread on sigma axes only
    list(theta_grid = tg, weights = w,
         axis_offsets = c(0L, 2L, 4L),
         blocks = list(list(type = "icar"), list(type = "icar")))
}

# Smooth Gaussian inner log-marginal in the two log-sigma coordinates (cols 1 &
# 3); independent of the pinned alpha / phi axes.
.pk_refit_pinned <- function(u_hat, s = c(0.20, 0.25)) {
    function(theta_mat) {
        u1 <- log(theta_mat[, "b1.sigma"]); u3 <- log(theta_mat[, "b2.sigma"])
        -0.5 * ((u1 - u_hat[1])^2 / s[1]^2 + (u3 - u_hat[2])^2 / s[2]^2)
    }
}

test_that("a partial collapse with pinned axes recovers a usable k-hat", {
    # ess_grid <= d but the two sigma axes still carry weighted spread (only the
    # alpha + phi axes are pinned). The grid is spread, so the scorer considers
    # both the single grid-moment Gaussian and the grid-mixture proposal and keeps
    # whichever covers best (gcol33/tulpa#121); the FD mode-Hessian is reserved
    # for a true delta collapse (gcol33/tulpa#119). The inner target here is
    # Gaussian on the sigma axes, so either proposal gives a usable k-hat.
    res   <- .pk_pinned_res()
    wn    <- res$weights / sum(res$weights)
    expect_lt(1 / sum(wn^2), ncol(res$theta_grid))       # ess_grid <= d
    u_hat <- c(log(1.0), log(0.35))
    set.seed(1)
    kd <- tulpa:::.joint_pareto_k(res, .pk_refit_pinned(u_hat), n_samples = 200L,
                                  proposal = NULL)
    expect_false(is.na(kd$pareto_k))
    expect_true(kd$proposal_source %in%
                c("grid_moment", "moment_matched", "grid_mixture"))
    expect_true(is.finite(kd$is_ess) && kd$is_ess > 0)
    expect_lt(kd$pareto_k, 0.7)
})

test_that("a delta-weight collapse with pinned axes still recovers k (grid-layout detection)", {
    # The extreme of gcol33/tulpa#117: weight on exactly one cell (ess_grid = 1),
    # so the weighted variance of EVERY axis is zero. Weighted-variance detection
    # (the literal #114 set) would then drop the varying sigma axes too and
    # decline to NA; grid-layout detection keeps the multi-valued sigma axes and
    # recovers their FD curvature while still pinning the single-valued alpha/phi.
    res <- .pk_pinned_res()
    res$weights <- rep(0, nrow(res$theta_grid))
    centre <- which(abs(res$theta_grid[, "b1.sigma"] - 1.0)  < 1e-9 &
                    abs(res$theta_grid[, "b2.sigma"] - 0.35) < 1e-9)
    res$weights[centre] <- 1
    expect_equal(1 / sum(res$weights^2), 1)              # exact delta collapse
    u_hat <- c(log(1.0), log(0.35))
    kd <- tulpa:::.joint_pareto_k(res, .pk_refit_pinned(u_hat), n_samples = 200L,
                                  proposal = NULL)
    expect_false(is.na(kd$pareto_k))
    expect_identical(kd$proposal_source, "mode_hessian")
    expect_lt(kd$pareto_k, 0.7)
})

test_that(".joint_pareto_grid_vary_axes reads the grid layout, not the weights", {
    # Single-valued columns are pinned; multi-valued ones vary, independent of any
    # weight concentration (gcol33/tulpa#117).
    tg <- as.matrix(expand.grid(`b1.sigma` = c(0.8, 1.0, 1.2), `b1.alpha` = 0,
                                `b2.sigma` = c(0.25, 0.35), `phi_pos` = 1.0))
    expect_equal(unname(tulpa:::.joint_pareto_grid_vary_axes(tg)), c(1L, 3L))
})

test_that(".joint_pareto_mode_cov restricts to the varying axes and zeros the pinned ones", {
    # Direct unit on the helper: a 3-axis target Gaussian in axes 1 & 3, axis 2
    # pinned. The returned covariance is the inverse curvature on {1, 3} embedded
    # block-diagonally, zero on the pinned row/col.
    cn   <- c("b1.sigma", "b1.alpha", "b2.sigma")
    tags <- c("log", "identity", "log")
    s    <- c(0.20, 0.30)
    refit <- function(theta_mat) {
        u1 <- log(theta_mat[, 1]); u3 <- log(theta_mat[, 3])
        -0.5 * ((u1 - log(1.0))^2 / s[1]^2 + (u3 - log(0.4))^2 / s[2]^2)
    }
    u_center <- c(log(1.0), 0, log(0.4))
    cov <- tulpa:::.joint_pareto_mode_cov(u_center, tags, cn, refit, d = 3L,
                                          vary = c(1L, 3L))
    expect_false(is.null(cov))
    expect_equal(dim(cov), c(3L, 3L))
    expect_equal(cov[2, ], rep(0, 3))                    # pinned row zero
    expect_equal(cov[, 2], rep(0, 3))                    # pinned col zero
    expect_equal(diag(cov)[c(1, 3)], s^2, tolerance = 1e-2)   # recovers curvature
    # Full-d stencil (no vary) is singular on the pinned axis -> NULL, the
    # pre-#117 behaviour the fix routes around.
    expect_null(tulpa:::.joint_pareto_mode_cov(u_center, tags, cn, refit, d = 3L))
})

# --------------------------------------------------------------------------- #
# CCD mode-Hessian proposal splice                                            #
# --------------------------------------------------------------------------- #

test_that("a supplied CCD mode-Hessian proposal yields a finite k-hat at collapse", {
    res   <- .pk_collapsed_res()
    u_hat <- c(log(1.0), stats::qlogis(0.5))
    prop  <- list(u_hat = u_hat, L_scale = diag(c(0.2, 0.3)),
                  tags = c("log", "logit01"), cols = 1:2)
    kd <- tulpa:::.joint_pareto_k(res, .pk_refit(u_hat), n_samples = 200L,
                                  proposal = prop)
    expect_false(is.na(kd$pareto_k))
    expect_identical(kd$proposal_source, "mode_hessian")
    expect_lt(kd$pareto_k, 0.7)
})

test_that("an inconsistent proposal is ignored (identical to no proposal)", {
    # Wrong per-axis transforms: the splice guard must reject the proposal, after
    # which the path is exactly the no-proposal path (here, the FD fallback).
    res   <- .pk_collapsed_res()
    u_hat <- c(log(1.0), stats::qlogis(0.5)); refit <- .pk_refit(u_hat)
    bad   <- list(u_hat = u_hat, L_scale = diag(c(0.2, 0.3)),
                  tags = c("identity", "identity"), cols = 1:2)
    set.seed(7); k_bad  <- tulpa:::.joint_pareto_k(res, refit, 200L, proposal = bad)
    set.seed(7); k_null <- tulpa:::.joint_pareto_k(res, refit, 200L, proposal = NULL)
    expect_equal(k_bad$pareto_k, k_null$pareto_k)
    expect_equal(k_bad$is_ess,   k_null$is_ess)
})

# --------------------------------------------------------------------------- #
# Well-resolved grid: a matched proposal is a no-op                           #
# --------------------------------------------------------------------------- #

test_that("a well-resolved grid scores deterministically; a supplied proposal routes to the single Gaussian", {
    # On a well-resolved spread grid (effective ESS > d) the scorer considers the
    # single grid-moment Gaussian and the grid mixture and keeps the better-
    # covering one (gcol33/tulpa#121). For a Gaussian inner target the single
    # Gaussian is near-exact, so it typically wins on ESS; either way the result
    # is deterministic under a fixed seed and usable. A SUPPLIED consistent
    # proposal (e.g. a CCD mode-Hessian) routes to the single-Gaussian scorer.
    tg <- as.matrix(expand.grid(`b1.sigma` = c(0.55, 0.75, 1.0, 1.35, 1.8),
                                `b1.rho`   = c(0.2, 0.35, 0.5, 0.65, 0.8)))
    colnames(tg) <- c("b1.sigma", "b1.rho")
    u_hat0 <- c(log(1.0), stats::qlogis(0.5)); s <- c(0.35, 0.40)
    u1 <- log(tg[, 1]); u2 <- stats::qlogis(tg[, 2])
    lm <- -0.5 * ((u1 - u_hat0[1])^2 / s[1]^2 + (u2 - u_hat0[2])^2 / s[2]^2)
    w  <- exp(lm - max(lm)); w <- w / sum(w)
    res <- list(theta_grid = tg, weights = w,
                axis_offsets = c(0L, 2L), blocks = list(list(type = "bym2")))
    expect_gt(1 / sum(w^2), ncol(tg))        # not collapsed
    refit <- .pk_refit(u_hat0, s)

    # NULL proposal -> ESS-selected proposal, deterministic under a fixed seed.
    set.seed(42); k1 <- tulpa:::.joint_pareto_k(res, refit, 400L, NULL)
    set.seed(42); k2 <- tulpa:::.joint_pareto_k(res, refit, 400L, NULL)
    expect_true(k1$proposal_source %in% c("grid_moment", "moment_matched", "grid_mixture"))
    expect_false(is.na(k1$pareto_k))
    expect_equal(k2$pareto_k, k1$pareto_k)
    expect_equal(k2$is_ess,   k1$is_ess)
    expect_lt(k1$pareto_k, 0.7)              # Gaussian target, well covered

    # A supplied consistent proposal routes to the single-Gaussian scorer.
    u_grid <- cbind(u1, u2)
    u_hat  <- as.numeric(crossprod(w, u_grid))
    cen    <- sweep(u_grid, 2L, u_hat)
    Su     <- crossprod(cen * w, cen); Su <- (Su + t(Su)) / 2
    prop   <- list(u_hat = u_hat, L_scale = t(chol(Su)),
                   tags = c("log", "logit01"), cols = 1:2)
    set.seed(42); k_sup <- tulpa:::.joint_pareto_k(res, refit, 400L, prop)
    expect_identical(k_sup$proposal_source, "mode_hessian")
    expect_false(is.na(k_sup$pareto_k))
    expect_lt(k_sup$pareto_k, 0.7)
})

test_that("the grid-mixture proposal beats the single Gaussian on a skewed grid", {
    # The fix's core invariant (gcol33/tulpa#121): when the hyperparameter
    # posterior is right-skewed in the unconstrained coordinate but the grid spans
    # its support, a single symmetric Gaussian proposal underweights the heavy
    # shoulder (high k-hat); the grid mixture covers the skew through its nodes, so
    # the same fit gets a usable k-hat. The grid here extends well past the skew
    # tail, so the deficiency is a WITHIN-grid shape mismatch, not a too-narrow
    # grid. The integration weights are the unnormalised posterior at the nodes, so
    # the mixture built from them tracks the skew by construction.
    sig <- exp(seq(log(0.25), log(35), length.out = 25))  # spans past the skew tail
    tg  <- matrix(sig, ncol = 1L, dimnames = list(NULL, "b1.sigma"))
    m <- log(1.1); sL <- 0.30; sR <- 0.95                 # split-normal in u = log sigma
    skew_u <- function(u) ifelse(u <= m, -0.5 * ((u - m) / sL)^2,
                                         -0.5 * ((u - m) / sR)^2)
    w   <- exp(skew_u(log(sig))); w <- w / sum(w)
    res <- list(theta_grid = tg, weights = w,
                axis_offsets = c(0L, 1L), blocks = list(list(type = "bym2")))
    refit <- function(theta_mat) skew_u(log(theta_mat[, 1]))

    prep <- tulpa:::.joint_pareto_prepare(res, refit, 600L, NULL)
    expect_identical(prep$proposal_source, "grid_moment")
    vary <- tulpa:::.joint_pareto_vary_axes(prep$Su)
    set.seed(1); g <- tulpa:::.joint_pareto_score(prep, vary, refit, 600L)
    set.seed(1); mx <- tulpa:::.joint_pareto_score_mixture(prep, vary, refit, 600L)
    expect_false(is.null(mx))
    expect_lt(mx$pareto_k, g$pareto_k)       # mixture improves the tail shape
    expect_lt(mx$pareto_k, 0.7)              # ... into the usable band
    expect_gt(mx$is_ess, g$is_ess)           # ... with a better effective sample

    set.seed(1); kd <- tulpa:::.joint_pareto_k(res, refit, 600L, NULL)
    expect_identical(kd$proposal_source, "grid_mixture")
})

test_that("a grid-width deficiency stays unreliable: the reported k is never the moment-matched Gaussian (gcol33/tulpa#130)", {
    # The dispatcher's load-bearing #130 invariant, pinned DIRECTLY rather than
    # only via expect_gte(pareto_k, 0.7). Construction: a spread grid (-3..3, 61
    # cells) whose integration weights are a NARROW Gaussian (sd 0.5 in log-sigma)
    # but whose true outer target is a much WIDER Gaussian (sd 3). The grid cannot
    # represent the target's spread -- a genuine grid-width deficiency.
    #
    # Moment matching the single-Gaussian proposal to the importance-weighted draws
    # WIDENS it toward the target, so the MM-refined single Gaussian reads in the
    # good band. If the dispatcher reported THAT it would mask the deficiency. The
    # invariant: the adopted / reported k must come from the faithful within-grid
    # representation (grid-moment / grid-mixture), stay unreliable, and never be the
    # moment-matched Gaussian's optimistic value -- regardless of how many MM passes
    # the cost backstop (.K_DIAG_MM_MAX) allows.
    sg  <- exp(seq(-3, 3, length.out = 61))
    w   <- { e <- exp(stats::dnorm(log(sg), 0, 0.5, log = TRUE)); e / sum(e) }
    res <- list(theta_grid = matrix(sg, ncol = 1, dimnames = list(NULL, "sigma")),
                weights = w, prior = list(type = "icar"))
    refit_wide <- function(tm) { u <- log(tm[, "sigma"]); -0.5 * (u / 3)^2 - u }

    prep <- tulpa:::.joint_pareto_prepare(res, refit_wide, 4000L, NULL)
    expect_identical(prep$proposal_source, "grid_moment")     # a spread grid
    vary <- tulpa:::.joint_pareto_vary_axes(prep$Su)

    set.seed(203)
    g <- tulpa:::.joint_pareto_score(prep, vary, refit_wide, 4000L)
    # The grid-moment proposal -- the engine's faithful single-Gaussian summary of
    # what it integrates over -- is unreliable: the grid is too narrow.
    expect_gte(g$gm$pareto_k, 0.7)
    # ... yet moment matching widens the Gaussian into the good band. This is the
    # trap: the optimistic value EXISTS and is materially below the grid-moment k.
    expect_true(isTRUE(g$refined))
    expect_lt(g$pareto_k, 0.6)
    expect_lt(g$pareto_k, g$gm$pareto_k - 0.2)

    # The dispatched verdict: adopt the faithful within-grid proposal, stay
    # unreliable, and NOT report the moment-matched Gaussian.
    set.seed(203)
    disp <- tulpa:::.joint_pareto_score_dispatch(prep, vary, refit_wide, 4000L)
    expect_identical(disp$source, "grid_mixture")
    expect_false(identical(disp$source, "moment_matched"))
    expect_gte(disp$best$pareto_k, 0.7)
    # The reported k is the faithful mixture's, well above the moment-matched
    # Gaussian's optimistic value -- the masking is refused.
    expect_gt(disp$best$pareto_k, g$pareto_k)

    # End to end through the public scorer: same verdict, unreliable band.
    set.seed(203)
    kd <- tulpa:::.joint_pareto_k(res, refit_wide, n_samples = 4000L)
    expect_identical(kd$proposal_source, "grid_mixture")
    expect_false(identical(kd$proposal_source, "moment_matched"))
    expect_gte(kd$pareto_k, 0.7)
    expect_identical(tulpa:::.tulpa_khat_band(kd$pareto_k), "unreliable")
})

# --------------------------------------------------------------------------- #
# End-to-end: CCD engages, and a forced tensor grid collapses                 #
# --------------------------------------------------------------------------- #

.pkp_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn  <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# --------------------------------------------------------------------------- #
# Opt-in per-arm outer Pareto-k (gcol33/tulpa#120)                            #
# --------------------------------------------------------------------------- #

# A separable two-block, two-arm result: block 1 (b1.sigma, b1.alpha) loads the
# occ arm, block 2 (b2.sigma, b2.alpha) loads the pos arm, and a trailing
# phi_pos dispersion axis belongs to pos. No copy (alpha pinned at 0), so each
# block's axes are attributed to the single arm it loads.
.pk_arm_res <- function() {
    res <- .pk_pinned_res()
    res$responses  <- list(occ = list(), pos = list())
    res$arm_layout <- list(n_arms = 2L)
    res$blocks <- list(
        list(type = "icar", spatial_idx = list(1:5, integer(0))),   # occ only
        list(type = "icar", spatial_idx = list(integer(0), 1:5)))    # pos only
    res$copy <- NULL
    res$prior <- res$blocks
    res
}

test_that(".joint_pareto_arm_axes maps block + phi columns to arms", {
    res <- .pk_arm_res()
    ax  <- tulpa:::.joint_pareto_arm_axes(res)
    expect_named(ax, c("occ", "pos"))
    expect_equal(ax$occ, c(1L, 2L))            # b1.sigma, b1.alpha
    expect_equal(ax$pos, c(3L, 4L, 5L))        # b2.sigma, b2.alpha, phi_pos
})

test_that(".joint_pareto_arm_axes declines on the single-block layout", {
    # No axis_offsets / blocks -> one shared field across arms -> decline rather
    # than mis-attribute (the single-block joint path).
    res <- list(theta_grid = matrix(1, 1, 2,
                                    dimnames = list(NULL, c("sigma", "alpha"))),
                weights = 1, responses = list(occ = list(), pos = list()))
    expect_null(tulpa:::.joint_pareto_arm_axes(res))
})

test_that("per-arm k is reported and leaves the joint k bit-identical", {
    res   <- .pk_arm_res()
    u_hat <- c(log(1.0), log(0.35))
    refit <- .pk_refit_pinned(u_hat)
    ax    <- tulpa:::.joint_pareto_arm_axes(res)

    set.seed(3)
    k_arm   <- tulpa:::.joint_pareto_k(res, refit, n_samples = 200L,
                                       proposal = NULL, arm_axes = ax)
    set.seed(3)
    k_joint <- tulpa:::.joint_pareto_k(res, refit, n_samples = 200L,
                                       proposal = NULL)

    # The joint k is scored first from the same seed -> unchanged by the opt-in.
    expect_equal(k_arm$pareto_k, k_joint$pareto_k)
    expect_equal(k_arm$is_ess,   k_joint$is_ess)
    expect_null(k_joint$by_arm_k)

    expect_named(k_arm$by_arm_k, c("occ", "pos"))
    expect_true(all(is.finite(k_arm$by_arm_k)))
    expect_true(all(k_arm$by_arm_k < 0.7))     # each arm: 1-axis Gaussian target
    expect_named(k_arm$by_arm_is_ess, c("occ", "pos"))
})

test_that("a multi-block CCD fit sources the k-hat from the mode Hessian", {
    skip_if_not_slow()
    set.seed(31)
    N <- 800L; nA <- 40L; nB <- 30L
    iA <- sample.int(nA, N, TRUE); iB <- sample.int(nB, N, TRUE)
    pA <- { r <- cumsum(rnorm(nA, 0, 1.1 / sqrt(nA))); r - mean(r) }
    pB <- { r <- cumsum(rnorm(nB, 0, 0.9 / sqrt(nB))); r - mean(r) }
    x <- rnorm(N); Xo <- cbind(1, x)
    oc <- rbinom(N, 1, plogis(as.numeric(Xo %*% c(-0.3, 0.5)) + pA[iA] + pB[iB]))
    ip <- oc == 1L; Xp <- Xo[ip, , drop = FALSE]; iAp <- iA[ip]; iBp <- iB[ip]
    yp <- rnorm(sum(ip),
                as.numeric(Xp %*% c(0.2, -0.4)) + 1.2 * pA[iAp] + 0.8 * pB[iBp], 0.5)
    aA <- .pkp_adj(nA); aB <- .pkp_adj(nB)
    arms <- list(
        occ = list(y = as.numeric(oc), n_trials = rep(1L, N), X = Xo,
                   spatial_idx = as.integer(iA), re_idx = rep(0, N),
                   n_re_groups = 0L, sigma_re = 1.0, family = "binomial", phi = 1.0),
        pos = list(y = yp, n_trials = rep(1L, length(yp)), X = Xp,
                   spatial_idx = as.integer(iAp), re_idx = rep(0, length(yp)),
                   n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 0.5))
    prior <- list(
        list(type = "icar", n_spatial_units = nA, adj_row_ptr = aA$adj_row_ptr,
             adj_col_idx = aA$adj_col_idx, n_neighbors = aA$n_neighbors,
             sigma_grid = c(0.4, 0.8, 1.2, 1.6),
             spatial_idx = list(as.integer(iA), as.integer(iAp))),
        list(type = "icar", n_spatial_units = nB, adj_row_ptr = aB$adj_row_ptr,
             adj_col_idx = aB$adj_col_idx, n_neighbors = aB$n_neighbors,
             sigma_grid = c(0.3, 0.7, 1.1),
             spatial_idx = list(as.integer(iB), as.integer(iBp))))
    cp <- list(list(block = 1, arm = "pos", alpha_grid = c(0, 0.6, 1.2, 1.8)),
               list(block = 2, arm = "pos", alpha_grid = c(0, 0.5, 1.0, 1.5)))
    fit <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior, copy = cp,
        control = list(integration = "ccd", diagnose_k = TRUE, k_samples = 200L,
                       var_of_means_consistency = FALSE))
    expect_identical(fit$integration, "ccd")
    expect_identical(fit$pareto_k_proposal_source, "mode_hessian")
    expect_true(is.finite(fit$pareto_k))
    expect_gt(fit$pareto_k_is_ess, 0)
})

test_that("a collapsed tensor fit recovers a usable k-hat (grid-moment + moment matching)", {
    skip_if_not_slow()
    set.seed(12)
    N <- 4000L; n_s <- 40L
    si <- sample.int(n_s, N, TRUE)
    phi <- { r <- cumsum(rnorm(n_s, 0, 1.4 / sqrt(n_s))); r - mean(r) }
    x <- rnorm(N); Xo <- cbind(1, x)
    oc <- rbinom(N, 1, plogis(as.numeric(Xo %*% c(-0.3, 0.5)) + phi[si]))
    ip <- oc == 1L; Xp <- Xo[ip, , drop = FALSE]; sp <- si[ip]
    yp <- rnorm(sum(ip), as.numeric(Xp %*% c(0.2, -0.4)) + 1.4 * phi[sp], 0.4)
    adj <- .pkp_adj(n_s)
    arms <- list(
        occ = list(y = as.numeric(oc), n_trials = rep(1L, N), X = Xo,
                   spatial_idx = as.integer(si), re_idx = rep(0, N),
                   n_re_groups = 0L, sigma_re = 1.0, family = "binomial", phi = 1.0),
        pos = list(y = yp, n_trials = rep(1L, length(yp)), X = Xp,
                   spatial_idx = as.integer(sp), re_idx = rep(0, length(yp)),
                   n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 0.4))
    prior <- list(list(
        type = "icar", n_spatial_units = n_s, adj_row_ptr = adj$adj_row_ptr,
        adj_col_idx = adj$adj_col_idx, n_neighbors = adj$n_neighbors,
        sigma_grid = c(0.4, 0.7, 1.0, 1.4, 1.8),
        spatial_idx = list(as.integer(si), as.integer(sp))))
    cp <- list(block = 1, arm = "pos", alpha_grid = c(0, 0.5, 1.0, 1.5, 2.0))
    fit <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior, copy = cp,
        control = list(integration = "grid", diagnose_k = TRUE, k_samples = 200L,
                       var_of_means_consistency = FALSE))
    wn <- fit$weights / sum(fit$weights)
    expect_identical(fit$integration, "grid")
    expect_lt(1 / sum(wn^2), ncol(fit$theta_grid))   # grid genuinely collapsed
    # Partial collapse (the sigma / alpha axes still carry weighted spread): the
    # grid is spread, so the faithful proposal is the grid mixture
    # (gcol33/tulpa#121). The single-Gaussian sources (grid_moment, its
    # moment-matching refinement) remain valid where the mixture cannot form;
    # the FD mode-Hessian is reserved for a true delta collapse. Any of these
    # yields a usable k-hat.
    expect_true(fit$pareto_k_proposal_source %in%
                c("grid_mixture", "grid_moment", "moment_matched"))
    expect_true(is.finite(fit$pareto_k))
    expect_lt(fit$pareto_k, 0.7)
})

test_that("diagnose_k = 'by_arm' adds per-arm k and leaves the joint k unchanged", {
    skip_if_not_slow()
    set.seed(31)
    N <- 800L; nA <- 40L; nB <- 30L
    iA <- sample.int(nA, N, TRUE); iB <- sample.int(nB, N, TRUE)
    pA <- { r <- cumsum(rnorm(nA, 0, 1.1 / sqrt(nA))); r - mean(r) }
    pB <- { r <- cumsum(rnorm(nB, 0, 0.9 / sqrt(nB))); r - mean(r) }
    x <- rnorm(N); Xo <- cbind(1, x)
    oc <- rbinom(N, 1, plogis(as.numeric(Xo %*% c(-0.3, 0.5)) + pA[iA] + pB[iB]))
    ip <- oc == 1L; Xp <- Xo[ip, , drop = FALSE]; iAp <- iA[ip]; iBp <- iB[ip]
    yp <- rnorm(sum(ip),
                as.numeric(Xp %*% c(0.2, -0.4)) + 1.2 * pA[iAp] + 0.8 * pB[iBp], 0.5)
    aA <- .pkp_adj(nA); aB <- .pkp_adj(nB)
    arms <- list(
        occ = list(y = as.numeric(oc), n_trials = rep(1L, N), X = Xo,
                   spatial_idx = as.integer(iA), re_idx = rep(0, N),
                   n_re_groups = 0L, sigma_re = 1.0, family = "binomial", phi = 1.0),
        pos = list(y = yp, n_trials = rep(1L, length(yp)), X = Xp,
                   spatial_idx = as.integer(iAp), re_idx = rep(0, length(yp)),
                   n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 0.5))
    prior <- list(
        list(type = "icar", n_spatial_units = nA, adj_row_ptr = aA$adj_row_ptr,
             adj_col_idx = aA$adj_col_idx, n_neighbors = aA$n_neighbors,
             sigma_grid = c(0.4, 0.8, 1.2, 1.6),
             spatial_idx = list(as.integer(iA), as.integer(iAp))),
        list(type = "icar", n_spatial_units = nB, adj_row_ptr = aB$adj_row_ptr,
             adj_col_idx = aB$adj_col_idx, n_neighbors = aB$n_neighbors,
             sigma_grid = c(0.3, 0.7, 1.1),
             spatial_idx = list(as.integer(iB), as.integer(iBp))))
    cp <- list(list(block = 1, arm = "pos", alpha_grid = c(0, 0.6, 1.2, 1.8)),
               list(block = 2, arm = "pos", alpha_grid = c(0, 0.5, 1.0, 1.5)))
    ctrl <- list(integration = "ccd", k_samples = 200L,
                 var_of_means_consistency = FALSE)

    fit_def <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior, copy = cp,
        control = modifyList(ctrl, list(diagnose_k = TRUE)))
    fit_arm <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior, copy = cp,
        control = modifyList(ctrl, list(diagnose_k = "by_arm")))

    # The default fit reports no per-arm k; the opt-in adds it without moving the
    # joint k (the diagnostic is RNG-restored and the joint k is scored first).
    expect_null(fit_def$pareto_k_by_arm)
    expect_equal(fit_arm$pareto_k, fit_def$pareto_k)
    expect_named(fit_arm$pareto_k_by_arm, c("occ", "pos"))
    expect_true(all(is.finite(fit_arm$pareto_k_by_arm)))
    expect_match(fit_arm$pareto_k_by_arm_scope, "per-arm")
})
