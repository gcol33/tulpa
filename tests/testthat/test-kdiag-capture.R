# The `tulpa.kdiag.capture` validation aperture (gcol33/tulpa#356).
#
# The aperture exists so an external check (loo::psis, posterior::pareto_khat)
# can refit the GPD shape on a fit's OWN importance log-ratios and land on the
# number the fit reports. That only holds if the ratios it publishes are the
# ratios the reported k-hat was fitted on, which is the invariant every test
# here pins:
#
#     tulpa_psis(cap$lr, tail_points = cap$tail_points)$pareto_k == fit$pareto_k
#
# It used to fail on the joint path. The aperture was written by
# `.nested_is_pareto_k()`, the scoring primitive, which runs once per CANDIDATE
# proposal: the joint dispatch scores each moment-matching pass, then the grid
# mixture, then the skew-normal rescue, and keeps whichever gives the lowest
# k-hat. A losing candidate scored last therefore left its ratios in the
# aperture. On a spatial occu_cover fit that read 0.77 reported against 1.20
# recomputed (loo::psis agreeing with 1.20, which is a statement about the PSIS
# arithmetic on the discarded proposal and not about which ratios belong to the
# fit). The synthetic Student-t fixture below reproduces the same shape --
# 0.960131 reported, 1.198978 left in the aperture -- and is the reason it is
# the fixture used.

# --- capture harness --------------------------------------------------------
#
# `lr` is an active binding, so every write is counted rather than only the
# last one surviving. A backend must publish exactly once per k-hat it reports.
.kcap_env <- function() {
    st <- new.env(parent = emptyenv())
    st$n <- 0L
    st$lr <- NULL
    cap <- new.env(parent = emptyenv())
    makeActiveBinding("lr", function(v) {
        if (missing(v)) return(st$lr)
        st$n  <- st$n + 1L
        st$lr <- v
        invisible(v)
    }, cap)
    list(cap = cap, state = st)
}

.kcap_run <- function(fn) {
    kc  <- .kcap_env()
    old <- options(tulpa.kdiag.capture = kc$cap)
    on.exit(options(old), add = TRUE)
    value <- fn()
    list(value = value, n_writes = kc$state$n, lr = kc$state$lr,
         tail_points = kc$cap$tail_points, scope = kc$cap$scope)
}

# The invariant itself, shared by every backend below.
.kcap_expect_reproduces <- function(k, k_reported) {
    expect_false(is.null(k$lr))
    expect_gte(length(k$lr), 25L)
    expect_true(is.character(k$scope) && nzchar(k$scope))
    expect_equal(tulpa_psis(k$lr, tail_points = k$tail_points)$pareto_k,
                 k_reported, tolerance = 1e-12)
}

# --- fixtures ---------------------------------------------------------------

.kcap_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# Two-arm fixture: binomial occupancy + gaussian positive over one spatial field.
.kcap_sim <- function(N = 220, n_s = 20, sigma = 0.6, seed = 7) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    rw    <- cumsum(rnorm(n_s, 0, sigma / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    x <- rnorm(N)
    Xocc <- cbind(1, x)
    occur <- rbinom(N, 1, plogis(as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]))
    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]
    spi_pos <- spatial_idx[is_pos]
    y_pos <- rnorm(sum(is_pos),
                   as.numeric(Xpos %*% c(0.2, -0.4)) + phi_s[spi_pos], 0.5)
    list(N = N, n_s = n_s, spatial_idx = as.integer(spatial_idx), Xocc = Xocc,
         occur = occur, Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos))
}

.kcap_arms <- function(sim) {
    list(occ = list(y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
                    X = sim$Xocc, spatial_idx = sim$spatial_idx,
                    re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
                    family = "binomial", phi = 1.0),
         pos = list(y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
                    X = sim$Xpos, spatial_idx = sim$spi_pos,
                    re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L,
                    sigma_re = 1.0, family = "gaussian", phi = 0.5))
}

.kcap_rook <- function(nr, nc) {
    n <- nr * nc; W <- matrix(0, n, n); id <- function(r, c) (c - 1) * nr + r
    for (r in seq_len(nr)) for (c in seq_len(nc)) {
        if (r < nr) { W[id(r, c), id(r + 1, c)] <- 1; W[id(r + 1, c), id(r, c)] <- 1 }
        if (c < nc) { W[id(r, c), id(r, c + 1)] <- 1; W[id(r, c + 1), id(r, c)] <- 1 }
    }
    W
}

# A one-axis (sigma, log scale) joint result whose outer target is analytic, so
# the whole transform / proposal-fit / IS-PSIS composition runs with no inner
# solve. `sd_w` sets the grid weighting, `refit` the target.
.kcap_joint_res <- function(sd_w = 1.0) {
    sg <- exp(seq(-3, 3, length.out = 61))
    lw <- stats::dnorm(log(sg), 0, sd_w, log = TRUE)
    w  <- exp(lw - max(lw)); w <- w / sum(w)
    list(theta_grid = matrix(sg, ncol = 1, dimnames = list(NULL, "sigma")),
         weights = w, prior = list(type = "icar"))
}

# --- the scoring primitive publishes nothing --------------------------------

test_that(".nested_is_pareto_k hands its ratios back and writes no aperture", {
    # The primitive scores ONE candidate proposal and is called once per
    # candidate, so it is not in a position to know whether its ratios are the
    # ones the fit will report. It returns them; the backend publishes.
    set.seed(4)
    k <- .kcap_run(function()
        tulpa:::.nested_is_pareto_k(0, matrix(1, 1L, 1L),
                                    function(U) stats::dt(U[, 1L], df = 2, log = TRUE),
                                    n_samples = 300L))
    expect_identical(k$n_writes, 0L)
    expect_null(k$lr)
    expect_length(k$value$lr, 300L)
    # The ratios it returns are exactly the ones its own k-hat was fitted on.
    expect_equal(tulpa_psis(k$value$lr)$pareto_k, k$value$pareto_k,
                 tolerance = 1e-12)
})

test_that("repeated scoring passes leave the aperture untouched", {
    set.seed(5)
    kc  <- .kcap_env()
    old <- options(tulpa.kdiag.capture = kc$cap)
    on.exit(options(old), add = TRUE)
    for (df in c(2, 5, 30)) {
        tulpa:::.nested_is_pareto_k(0, matrix(1, 1L, 1L),
                                    function(U) stats::dt(U[, 1L], df = df, log = TRUE),
                                    n_samples = 200L)
    }
    expect_identical(kc$state$n, 0L)
})

# --- joint backend ----------------------------------------------------------

test_that("the joint driver publishes the SELECTED proposal's ratios", {
    # Heavy-tailed target: the grid-moment pass reads above the usable band, so
    # the moment-matching loop runs a second pass which scores WORSE and loses.
    # Pre-#356 that losing pass was the last thing written to the aperture.
    res   <- .kcap_joint_res(sd_w = 1.0)
    refit <- function(tm) {
        u <- log(tm[, "sigma"]); stats::dt(u, df = 2, log = TRUE) - u
    }
    set.seed(102)
    k <- .kcap_run(function()
        tulpa:::.joint_pareto_k(res, refit, n_samples = 4000L))

    expect_true(is.finite(k$value$pareto_k))
    expect_gt(k$value$pareto_k, 0.7)                 # above the usable band
    expect_identical(k$n_writes, 1L)                 # was 2 (the losing pass won the write)
    .kcap_expect_reproduces(k, k$value$pareto_k)
    expect_identical(k$scope, paste0("joint nested (", k$value$proposal_source, ")"))
})

test_that("the joint driver publishes a moment-matched proposal's ratios", {
    # A skewed target the refinement loop DOES improve on: the reported k-hat
    # comes from a later pass than the first. Same aperture contract.
    res   <- .kcap_joint_res(sd_w = 1.0)
    refit <- function(tm) {
        u <- log(tm[, "sigma"])
        stats::dgamma(u + 6, shape = 4, scale = 0.7, log = TRUE) - u
    }
    set.seed(21)
    k <- .kcap_run(function()
        tulpa:::.joint_pareto_k(res, refit, n_samples = 2000L))

    expect_identical(k$value$proposal_source, "moment_matched")
    expect_identical(k$n_writes, 1L)                 # was 3
    .kcap_expect_reproduces(k, k$value$pareto_k)
})

test_that("a declining joint fit publishes nothing rather than stale ratios", {
    # An unguessable axis is a permanent decline: with no reported k-hat there
    # are no ratios to publish, and an empty aperture is the honest state.
    res <- list(theta_grid = matrix(c(0.5, 1.0, 0.8, 0.9), ncol = 2,
                                    dimnames = list(NULL, c("sigma", "rho_car"))),
                weights = c(0.5, 0.5), prior = list(type = "car_proper"))
    k <- .kcap_run(function()
        tulpa:::.joint_pareto_k(res, function(tm) rep(0, nrow(tm)), n_samples = 200L))
    expect_true(is.na(k$value$pareto_k))
    expect_identical(k$n_writes, 0L)
    expect_null(k$lr)
})

test_that("per-arm scoring does not overwrite the joint fit's own ratios", {
    # The per-arm k-hats are scored AFTER the joint k and are reported on their
    # own fields; the aperture belongs to `pareto_k`, so those passes publish
    # nothing.
    res   <- .kcap_joint_res(sd_w = 1.0)
    refit <- function(tm) {
        u <- log(tm[, "sigma"]); -0.5 * (u / 0.8)^2 - u
    }
    set.seed(103)
    k <- .kcap_run(function()
        tulpa:::.joint_pareto_k(res, refit, n_samples = 800L,
                                arm_axes = list(a = 1L, b = 1L)))
    expect_identical(k$n_writes, 1L)
    .kcap_expect_reproduces(k, k$value$pareto_k)
})

test_that("joint single-block fit: aperture reproduces the reported k-hat", {
    skip_on_cran()
    sim <- .kcap_sim(seed = 11)
    adj <- .kcap_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.7, 1.1))
    arms <- .kcap_arms(sim)
    arms$pos$field_coef <- list(name = "alpha", grid = c(0, 0.5, 1.0, 1.5))
    k <- .kcap_run(function()
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
                                   control = list(k_samples = 150L)))
    skip_if(is.na(k$value$pareto_k), "joint k-hat declined on this fixture")
    .kcap_expect_reproduces(k, k$value$pareto_k)
})

test_that("joint multi-block fit: aperture reproduces the reported k-hat", {
    skip_on_cran()
    sim <- .kcap_sim(seed = 41)
    adj <- .kcap_chain_adj(sim$n_s)
    prior_multi <- list(list(
        type = "icar", n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.8),
        spatial_idx = list(sim$spatial_idx, sim$spi_pos)))
    k <- .kcap_run(function()
        tulpa_nested_laplace_joint(
            responses = .kcap_arms(sim), prior = prior_multi,
            copy = list(block = 1, arm = "pos", alpha_grid = c(0.5, 1.0, 1.5)),
            control = list(k_samples = 150L)))
    expect_s3_class(k$value, "tulpa_nested_laplace_joint_multi")
    skip_if(is.na(k$value$pareto_k), "multi-block k-hat declined on this fixture")
    .kcap_expect_reproduces(k, k$value$pareto_k)
})

# --- single-block grid backend ---------------------------------------------

test_that("nested_laplace grid fit: aperture reproduces the reported k-hat", {
    skip_on_cran()
    set.seed(5)
    nr <- nc <- 5L; S <- nr * nc; reps <- 4L
    W <- .kcap_rook(nr, nc)
    unit <- rep(seq_len(S), each = reps); N <- length(unit)
    x <- rnorm(N); ntr <- rep(3L, N)
    y <- rbinom(N, ntr, plogis(-0.3 + 0.6 * x))
    idx <- tulpa:::.resolve_unit_index(factor(unit), "region", S)
    csr <- tulpa:::adjacency_to_csr_tulpa(W)
    prior <- list(type = "icar", spatial_idx = idx, n_spatial_units = S,
                  adj_row_ptr = csr$row_ptr, adj_col_idx = csr$col_idx,
                  n_neighbors = csr$n_neighbors)
    k <- .kcap_run(function()
        tulpa_nested_laplace(y = y, n_trials = ntr, X = cbind(1, x),
                             prior = prior, family = "binomial"))
    skip_if(is.na(k$value$pareto_k), "grid k-hat declined on this fixture")
    .kcap_expect_reproduces(k, k$value$pareto_k)
})

# --- RE-covariance backend --------------------------------------------------

test_that("re_cov nested fit: aperture reproduces the reported k-hat", {
    skip_on_cran()
    set.seed(1L)
    G <- 30L; npg <- 8L; N <- G * npg
    grp <- rep(seq_len(G), each = npg); x <- rnorm(N)
    X <- cbind(1, x); Z <- cbind(1, x)
    Sigma <- matrix(c(0.64, 0.24, 0.24, 0.36), 2)
    u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), 2))
    eta <- as.numeric(X %*% c(-0.3, 0.7)) + rowSums(Z * u[grp, ])
    y <- rbinom(N, 1L, plogis(eta))
    re_term <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = Z)
    k <- .kcap_run(function()
        tulpa_re_cov_nested(y, rep(1L, N), X, re_term, family = "binomial",
                            control = list(k_samples = 120L)))
    skip_if(is.na(k$value$pareto_k), "re_cov k-hat declined on this fixture")
    expect_identical(k$n_writes, 1L)
    .kcap_expect_reproduces(k, k$value$pareto_k)
})

# --- SPDE backend -----------------------------------------------------------

test_that("fit_spde: aperture reproduces the reported k-hat", {
    skip_on_cran()
    skip_if_not_installed("fmesher")
    set.seed(42)
    n_obs <- 150
    coords <- cbind(runif(n_obs), runif(n_obs))
    spec <- spatial_spde(coords)
    y <- rbinom(n_obs, 1, 0.4)
    X <- matrix(1, nrow = n_obs, ncol = 1)
    k <- .kcap_run(function()
        suppressWarnings(fit_spde(y, X, spec, family = "binomial",
                                  n_trials = rep(1L, n_obs),
                                  control = list(method = "grid", n_grid = 5L))))
    skip_if(is.na(k$value$pareto_k), "SPDE k-hat declined on this fixture")
    expect_identical(k$n_writes, 1L)
    .kcap_expect_reproduces(k, k$value$pareto_k)
})

# --- cross-check against the reference implementation -----------------------

test_that("loo::psis on the published ratios lands on the reported k-hat", {
    skip_on_cran()
    skip_if_not_installed("loo")
    res   <- .kcap_joint_res(sd_w = 1.0)
    refit <- function(tm) {
        u <- log(tm[, "sigma"]); stats::dt(u, df = 2, log = TRUE) - u
    }
    set.seed(102)
    k <- .kcap_run(function()
        tulpa:::.joint_pareto_k(res, refit, n_samples = 4000L))
    k_loo <- suppressWarnings(
        loo::psis(matrix(k$lr, ncol = 1), r_eff = NA)$diagnostics$pareto_k)
    # The external check is the whole point of the aperture: an independent GPD
    # fit on the published ratios must land on the number the fit reports.
    expect_equal(k_loo, k$value$pareto_k, tolerance = 1e-6)
})
