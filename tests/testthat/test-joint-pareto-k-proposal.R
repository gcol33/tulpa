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

test_that("a collapsed grid with pinned axes recovers k from the varying-axis FD Hessian", {
    # Before gcol33/tulpa#117 the full-d FD Hessian is singular on the 3 pinned
    # axes, so the fallback bailed to grid_moment. Restricting the stencil to the
    # varying sigma axes makes it well conditioned -> mode_hessian with finite k.
    res   <- .pk_pinned_res()
    wn    <- res$weights / sum(res$weights)
    expect_lt(1 / sum(wn^2), ncol(res$theta_grid))       # ess_grid <= d
    u_hat <- c(log(1.0), log(0.35))
    kd <- tulpa:::.joint_pareto_k(res, .pk_refit_pinned(u_hat), n_samples = 200L,
                                  proposal = NULL)
    expect_false(is.na(kd$pareto_k))
    expect_identical(kd$proposal_source, "mode_hessian")
    expect_true(is.finite(kd$is_ess) && kd$is_ess > 0)
    expect_lt(kd$pareto_k, 0.7)             # Gaussian target under Gaussian proposal
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

test_that("a proposal matching the grid covariance is a byte-exact no-op", {
    # On a well-resolved grid (effective ESS > d, so no FD fallback) a proposal
    # whose mean/covariance equal the grid-weighted estimate must not move the
    # k-hat under a fixed seed.
    tg <- as.matrix(expand.grid(`b1.sigma` = c(0.55, 0.75, 1.0, 1.35, 1.8),
                                `b1.rho`   = c(0.2, 0.35, 0.5, 0.65, 0.8)))
    colnames(tg) <- c("b1.sigma", "b1.rho")
    u_hat0 <- c(log(1.0), stats::qlogis(0.5)); s <- c(0.35, 0.40)
    u1 <- log(tg[, 1]); u2 <- stats::qlogis(tg[, 2])
    lm <- -0.5 * ((u1 - u_hat0[1])^2 / s[1]^2 + (u2 - u_hat0[2])^2 / s[2]^2)
    w  <- exp(lm - max(lm)); w <- w / sum(w)
    res <- list(theta_grid = tg, weights = w,
                axis_offsets = c(0L, 2L), blocks = list(list(type = "bym2")))
    expect_gt(1 / sum(w^2), ncol(tg))        # not collapsed -> grid path, no FD
    refit <- .pk_refit(u_hat0, s)

    u_grid <- cbind(u1, u2)
    u_hat  <- as.numeric(crossprod(w, u_grid))
    cen    <- sweep(u_grid, 2L, u_hat)
    Su     <- crossprod(cen * w, cen); Su <- (Su + t(Su)) / 2
    prop   <- list(u_hat = u_hat, L_scale = t(chol(Su)),
                   tags = c("log", "logit01"), cols = 1:2)

    set.seed(42); k_grid <- tulpa:::.joint_pareto_k(res, refit, 400L, NULL)
    set.seed(42); k_hess <- tulpa:::.joint_pareto_k(res, refit, 400L, prop)
    expect_false(is.na(k_grid$pareto_k))
    expect_identical(k_grid$proposal_source, "grid_moment")
    expect_equal(k_hess$pareto_k, k_grid$pareto_k)
    expect_equal(k_hess$is_ess,   k_grid$is_ess)
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

test_that("a collapsed tensor fit recovers the k-hat via the FD Hessian", {
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
    expect_identical(fit$pareto_k_proposal_source, "mode_hessian")
    expect_true(is.finite(fit$pareto_k))
})
