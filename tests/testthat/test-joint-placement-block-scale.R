# Placement of every multi-block scale axis, and of an axis the fit's kept
# cells hold at one node.
#
# The placement mode-find searches each axis over its DECLARED nodes, not only
# the nodes the kept cells carry (an adaptive flood, a pilot thinning or a prune
# screen keeps a subset), and the placement pass moves a non-copy block's own
# scale -- an iid random effect's SD, an arm-specific field's precision -- the
# way it moves a copied field's SD. On data that identify those scales sharply
# against their node spacing they were otherwise integrated at one node with
# posterior SD 0.

test_that("the mode-find varies an axis its kept cells hold at one node", {
    tg <- matrix(c(0.5, 4), nrow = 1L,
                 dimnames = list(NULL, c("b1.sigma", "b2.tau")))
    lp <- function(theta) {
        -((log(theta[, 1L]) - log(0.45))^2 / (2 * 0.1^2) +
              (log(theta[, 2L]) - log(3.5))^2 / (2 * 0.2^2))
    }
    held <- tulpa:::.nl_placement_mode(tg, 1, c("log", "log"), lp)
    expect_identical(held$declined, "no_usable_curvature")

    nodes <- list(b1.sigma = exp(seq(log(0.05), log(5), length.out = 9)),
                  b2.tau   = exp(seq(log(0.04), log(400), length.out = 9)))
    pm <- tulpa:::.nl_placement_mode(tg, 1, c("log", "log"), lp,
                                     axis_nodes = nodes)
    expect_null(pm$declined)
    expect_equal(pm$mode_u, log(c(0.45, 3.5)), tolerance = 1e-3)
    expect_equal(sqrt(diag(pm$cov_u)), c(0.1, 0.2), tolerance = 0.02)
})

.bs_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# Two arms sharing a copied ICAR field (block 1), an arm-specific ICAR field on
# the positive arm alone (block 2, precision axis) and an iid random intercept
# on the positive arm (block 3, SD axis). The positive part is gaussian with a
# small residual SD, so both arm-specific scales are sharply identified. Their
# posterior width is set by the 40 units and 12 groups that carry them; N only
# has to determine each unit's and each group's effect, and at N = 2000 the
# positive arm holds on average 23 observations per unit and 78 per group at
# residual SD 0.3.
.bs_fixture <- function(seed = 951L, N = 2000L, n_s = 40L, n_g = 12L) {
    set.seed(seed)
    sp <- sample.int(n_s, N, replace = TRUE)
    w_s  <- 0.9 * as.numeric(scale(cumsum(rnorm(n_s, sd = 0.6))))
    w_pf <- 0.7 * as.numeric(scale(cumsum(rnorm(n_s, sd = 0.6))))
    g <- sample.int(n_g, N, replace = TRUE)
    u <- rnorm(n_g, sd = 0.8)
    X <- cbind(1, rnorm(N))
    occ <- rbinom(N, 1, plogis(as.numeric(X %*% c(-0.2, 0.5)) + w_s[sp]))
    k <- occ == 1L
    y <- rnorm(sum(k), as.numeric(X[k, ] %*% c(0.2, -0.4)) +
                   0.6 * w_s[sp[k]] + w_pf[sp[k]] + u[g[k]], 0.3)
    a <- .bs_chain_adj(n_s)
    icar <- function(extra) c(list(type = "icar",
                                   n_spatial_units = a$n_spatial_units,
                                   adj_row_ptr = a$adj_row_ptr,
                                   adj_col_idx = a$adj_col_idx,
                                   n_neighbors = a$n_neighbors), extra)
    list(
        responses = list(
            occ = list(y = as.numeric(occ), n_trials = rep(1L, N), X = X,
                       spatial_idx = sp, re_idx = rep(0, N), n_re_groups = 0L,
                       sigma_re = 1, family = "binomial", phi = 1),
            pos = list(y = y, n_trials = rep(1L, sum(k)), X = X[k, ],
                       spatial_idx = sp[k], re_idx = rep(0, sum(k)),
                       n_re_groups = 0L, sigma_re = 1, family = "gaussian",
                       phi = 0.09)),
        prior = list(
            icar(list(sigma_grid = auto_grid(c(0.25, 0.5, 1, 2, 4)),
                      spatial_idx = list(sp, sp[k]))),
            icar(list(tau_grid = auto_grid(
                          1 / exp(seq(log(0.05), log(5), length.out = 9))^2),
                      spatial_idx = list(rep(0L, N), sp[k]))),
            list(type = "iid", n_units = n_g,
                 sigma_grid = auto_grid(c(0.2, 0.6, 1.5)),
                 obs_idx = list(rep(0L, N), g[k]))),
        copy = list(arm = "pos", block = 1L,
                    alpha_grid = auto_grid(c(0, 0.3, 0.6, 1.2))))
}

.bs_node_mass <- function(fit, axis) {
    max(tapply(fit$weights, fit$theta_grid[, axis], sum))
}

test_that("a non-copy block's own scale axes are placed and integrated", {
    skip_on_cran()
    fx <- .bs_fixture()
    fit <- tulpa_nested_laplace_joint(
        fx$responses, fx$prior, copy = fx$copy,
        control = list(diagnose_k = FALSE, integration = "grid_adaptive"))
    expect_true(all(c("b2.tau", "b3.sigma") %in% fit$outer_grid_recenter_axes))
    for (a in c("b2.tau", "b3.sigma")) {
        expect_gt(length(unique(fit$theta_grid[, a])), 2L)
        expect_lt(.bs_node_mass(fit, a), 0.9)
        expect_gt(fit$theta_sd[[a]], 0)
    }
    expect_length(fit$outer_grid_single_node_axes, 0L)
    # The RE SD the fixture simulates is 0.8.
    expect_lt(abs(fit$theta_mean[["b3.sigma"]] - 0.8), 0.4)
})

test_that("a pilot-screened fit places the same axes", {
    skip_on_cran()
    fx <- .bs_fixture()
    fit <- tulpa_nested_laplace_joint(
        fx$responses, fx$prior, copy = fx$copy,
        control = list(diagnose_k = FALSE, integration = "grid_adaptive",
                       recenter_pilot = TRUE, prune = TRUE))
    for (a in c("b2.tau", "b3.sigma")) {
        expect_lt(.bs_node_mass(fit, a), 0.9)
    }
})

test_that("an axis left on one node is recorded and, when unheld, warned", {
    res <- list(
        theta_grid = cbind(b1.sigma = c(0.5, 0.5, 0.5), b2.tau = c(1, 2, 4)),
        weights = c(0.2, 0.5, 0.3),
        outer_axis_nodes = list(b1.sigma = c(0.25, 0.5, 1), b2.tau = c(1, 2, 4)))
    expect_identical(tulpa:::.nl_single_node_axes(res), "b1.sigma")
    res$outer_grid_single_node_axes <- "b1.sigma"
    expect_warning(tulpa:::.nl_warn_single_node_axes(res, "f()"),
                   "`b1.sigma`.*single grid node")
    res$outer_grid_axis_declined <- c(b1.sigma = "axis_pinned")
    expect_silent(tulpa:::.nl_warn_single_node_axes(res, "f()"))
    # A declared one-node axis is a pin of a value, not a collapse.
    res1 <- res
    res1$outer_axis_nodes$b1.sigma <- 0.5
    expect_length(tulpa:::.nl_single_node_axes(res1), 0L)
    # The copy scale's "no coupling" point mass is a model, not a collapse.
    res_a <- list(theta_grid = cbind(b1.alpha = c(0, 0.5, 1)),
                  weights = c(0.995, 0.004, 0.001))
    expect_length(tulpa:::.nl_single_node_axes(res_a), 0L)
})
