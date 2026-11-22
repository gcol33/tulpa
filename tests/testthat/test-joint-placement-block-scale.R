
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

skip_on_cran_budget()
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
    skip_on_cran_budget()
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

test_that("a block's scale is written as a tensor over its paired second axis", {
    skip_on_cran_budget()
    # car_proper carries (tau, rho) as paired cells. A placed precision is
    # re-crossed with the block's rho levels, never written beside them.
    blk <- list(type = "car_proper", n_spatial_units = 3L, spatial_idx = 1:3,
                adj_row_ptr = c(0L, 1L, 3L, 4L), adj_col_idx = c(1L, 0L, 2L, 1L),
                n_neighbors = c(1L, 2L, 1L))
    gr <- expand.grid(tau = c(0.5, 1, 2), rho = c(0.3, 0.6, 0.9))
    blk$tau_grid <- gr$tau
    blk$rho_grid <- gr$rho
    st <- list(prior = list(blk, list(type = "rw1", tau_grid = c(1, 2))))
    tg <- cbind(b1.tau = gr$tau, b1.rho = gr$rho, b2.tau = 1)
    fam <- tulpa:::.joint_placement_families()
    slots <- tulpa:::.joint_placement_slots(list(theta_grid = tg), st, fam)
    s <- Filter(function(x) identical(x$label, "b1.tau"), slots)[[1L]]
    new_tau <- c(0.4, 0.8, 1.6, 3.2, 6.4)
    out <- fam[[s$family]]$write(st, s, new_tau)$prior[[1L]]
    filled <- tulpa:::.NL_REGISTRY$car_proper$defaults(out, NULL)
    cells <- unique(cbind(filled$tau_grid, filled$rho_grid))
    expect_identical(nrow(cells), 15L)
    expect_setequal(unique(filled$tau_grid), new_tau)
    expect_setequal(unique(filled$rho_grid), c(0.3, 0.6, 0.9))

    # A non-copy BYM2 block's SD and mixing weight are both its registry axes,
    # so whichever one an attempt moves, the pair is re-crossed.
    bym <- list(type = "bym2", sigma_grid = gr$tau, rho_grid = gr$rho)
    stb <- list(prior = list(bym, list(type = "rw1", tau_grid = c(1, 2))))
    tgb <- cbind(b1.sigma = gr$tau, b1.rho = gr$rho, b2.tau = 1)
    sl <- tulpa:::.joint_placement_slots(list(theta_grid = tgb), stb, fam)
    fam_of <- stats::setNames(vapply(sl, `[[`, "", "family"),
                              vapply(sl, `[[`, "", "label"))
    expect_identical(unname(fam_of[c("b1.sigma", "b1.rho")]), c("field", "field"))
    s_rho <- sl[[match("b1.rho", names(fam_of))]]
    outb <- fam[["field"]]$write(stb, s_rho, c(0.2, 0.5, 0.8, 0.95))$prior[[1L]]
    expect_identical(nrow(unique(cbind(outb$sigma_grid, outb$rho_grid))), 12L)
})

skip_on_cran()

test_that("a placed precision beside a stated proper-CAR rho is crossed with it", {
    skip_on_cran()
    S <- 50L
    set.seed(707L + S + 5000L)
    eff <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4)), scale = FALSE))
    set.seed(707L + S)
    idx <- rep(seq_len(S), each = 10L)
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] + rnorm(length(idx), 0, sqrt(0.5))
    nbr <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
    nn <- lengths(nbr)
    base <- list(type = "car_proper", n_spatial_units = S, spatial_idx = idx,
                 adj_row_ptr = as.integer(c(0L, cumsum(nn))),
                 adj_col_idx = as.integer(unlist(nbr)) - 1L,
                 n_neighbors = as.integer(nn))
    ctl <- list(n_threads = 1L, diagnose_k = FALSE, progress = FALSE)
    for (rg in list(c(0.3, 0.6, 0.85, 0.95), c(0.3, 0.5, 0.7, 0.85, 0.95))) {
        fit <- suppressWarnings(tulpa_nested_laplace(
            y = y, n_trials = rep(1L, length(y)), X = X,
            prior = c(base, list(rho_grid = rg)), family = "gaussian",
            phi = 0.5, control = c(ctl, list(axis_refine = "none"))))
        expect_true("b1.tau" %in% fit$outer_grid_recenter_axes)
        tg <- fit$theta_grid
        n_tau <- length(unique(tg[, "b1.tau"]))
        expect_setequal(unique(tg[, "b1.rho"]), rg)
        expect_identical(nrow(unique(tg)), n_tau * length(rg))
    }
    # Dense reference: tau on 120 log-spaced nodes crossed with the five rho
    # nodes, tau's 95% interval 2.18 to 7.09, median 4.06.
    ci <- c(fit$theta_ci_lo[["b1.tau"]], fit$theta_median[["b1.tau"]],
            fit$theta_ci_hi[["b1.tau"]])
    expect_lt(max(abs(log(ci) - log(c(2.18, 4.06, 7.09)))), 0.1 * 0.300)
})

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

test_that("a placed precision is followed out along a heavy upper tail", {
    skip_on_cran()
    # A binomial ICAR the data barely identify: the precision's posterior runs
    # out along its PC prior's tau^(-3/2) tail. The placed axis is the engine's
    # own, so refinement may lay levels past its top node. A dense 300-node
    # log-spaced reference over (1, 1e7) reads tau's median 264 and 97.5% point
    # 1.13e5 (log-sd 2.28), and x's posterior sd 0.1952.
    set.seed(20260716)
    S <- 30L
    W <- matrix(0, S, S)
    for (i in 1:(S - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
    df <- data.frame(region = factor(seq_len(S)))
    df$x <- as.integer(df$region) / 10 + rnorm(S, 0, 0.3)
    df$y <- rbinom(S, 20, plogis(-0.4 + 0.5 * df$x))
    fit <- suppressMessages(tulpa(
        y ~ x + spatial(region), data = df, family = "binomial",
        n_trials = rep(20L, S), spatial = spatial_car(W, level = "obs"),
        mode = "structured", control = list(progress = FALSE)))
    expect_identical(fit$axis_span$b1.tau$refine, "extend")
    expect_gt(max(fit$theta_grid[, "b1.tau"]),
              max(fit$theta_grid[!nzchar(fit$refining_axis), "b1.tau"]))
    hp <- summary(fit)$hyperparameters
    hp <- hp[hp$parameter == "b1.tau", ]
    expect_lt(abs(log(hp$ci_hi / 1.134e5)), 0.1 * 2.28)
    expect_lt(abs(log(hp$median / 263.8)), 0.1 * 2.28)
    expect_lt(abs(sqrt(vcov(fit)[2L, 2L]) / 0.1952 - 1), 0.01)
})
