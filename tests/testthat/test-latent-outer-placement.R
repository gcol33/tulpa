# A `latent()` block's hyperparameter axes are placed at the outer mode and
# refined by the same pass and the same level refinement as every other
# registry axis (gcol33/tulpa#1059). A tgmrf block's theta is the coordinate its
# own prior is a density on, so its axes are laid on the identity; the fixed
# +/- 2 unit-spaced box the block starts from is only where the pass looks first.

skip_on_cran()

# Marginal mean / SD / 95% interval of each outer axis of a fit laid on a dense
# tensor (spacing well below one posterior SD), each node owning the cell of
# width h centred on it.
.dense_axis_summary <- function(fit) {
    g <- tulpa_outer_grid(fit)
    wt <- g$weights / sum(g$weights)
    t(vapply(colnames(g$nodes), function(a) {
        m <- tapply(wt, g$nodes[, a], sum)
        u <- as.numeric(names(m)); m <- as.numeric(m)
        mu <- sum(u * m); h <- u[2L] - u[1L]
        cdf <- c(0, cumsum(m)); edges <- c(u - h / 2, u[length(u)] + h / 2)
        q <- function(p) stats::approx(cdf, edges, xout = p, ties = "ordered")$y
        c(mean = mu, sd = sqrt(sum((u - mu)^2 * m)), ci_lo = q(0.025),
          ci_hi = q(0.975), edge = m[1L] + m[length(m)])
    }, numeric(5)))
}

.hp_table <- function(fit) {
    hp <- summary(fit)$hyperparameters
    out <- as.matrix(hp[, c("mean", "sd", "ci_lo", "ci_hi")])
    rownames(out) <- hp$parameter
    out
}

test_that("a tgmrf block's axis is read off the posterior, not the start box", {
    skip_on_cran()
    set.seed(5)
    G <- 40L; m <- 6L
    g <- rep(seq_len(G), each = m)
    u <- stats::rnorm(G, 0, 0.6)
    x <- stats::rnorm(G * m)
    d <- data.frame(x = x, y = stats::rpois(G * m, exp(0.3 + 0.4 * x + u[g])))
    blk <- function() tgmrf(
        Q = function(theta) Matrix::Diagonal(G, exp(-2 * theta[1])),
        prior = function(theta) stats::dnorm(theta[1], 0, 1, log = TRUE),
        init = c(log_sigma = 0), obs_idx = g, name = "iid")
    fit_with <- function(b, control = list()) suppressWarnings(tulpa(
        y ~ x + latent(b), data = d, family = "poisson",
        mode = "nested_laplace", control = control))

    fit <- fit_with(blk())
    # The start box is five unit-spaced nodes; the posterior SD is ~0.15, so
    # the box alone puts the whole posterior on one node.
    expect_gt(nrow(tulpa_outer_grid(fit)$nodes), 5L)

    dense_blk <- blk()
    dense_blk$theta_grid_built <- matrix(seq(-1.4, 0.7, by = 0.01), ncol = 1L,
                                         dimnames = list(NULL, "log_sigma"))
    ref <- .dense_axis_summary(fit_with(dense_blk, list(
        axis_refine = "none", prune = FALSE, diagnose_k = FALSE)))
    expect_lt(ref[1L, "edge"], 1e-4)
    got <- .hp_table(fit)
    # Each read within a quarter of the reference posterior SD.
    expect_lt(max(abs(got[1L, ] - ref[1L, 1:4])), 0.25 * ref[1L, "sd"])
})

test_that("latent(temporal_ar2()) is placed at the outer mode and integrates its spread", {
    skip_on_cran()
    # The gcol33/tulpa#1059 fixture: AR(2) phi = (1.5, -0.75), innovation sd
    # 0.5, noise sd 0.3. On the start box 99.5% of the weight sat on one cell
    # (ESS 1.01 of 125) and every interval was the cell's own width.
    set.seed(11)
    Tt <- 150L
    w <- numeric(Tt + 100)
    for (t in 3:(Tt + 100)) {
        w[t] <- 1.5 * w[t - 1] - 0.75 * w[t - 2] + stats::rnorm(1, 0, 0.5)
    }
    w <- w[101:(Tt + 100)]
    x <- stats::rnorm(Tt)
    y <- 1 + 0.5 * x + w + stats::rnorm(Tt, 0, 0.3)
    d <- data.frame(t = seq_len(Tt), x = x, y = y)
    fit <- suppressWarnings(tulpa(y ~ x + latent(temporal_ar2(d$t)), data = d,
                                  family = "gaussian", phi = 0.3^2,
                                  mode = "nested_laplace"))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_setequal(fit$outer_grid_recenter_axes,
                    c("b1.log_tau", "b1.atanh_psi1", "b1.atanh_psi2"))
    wt <- tulpa_outer_grid(fit)$weights
    expect_gt(1 / sum((wt / sum(wt))^2), 20)

    # Reference: the same model on a 35^3 tensor at 0.29 posterior SDs per
    # node over +/- 5 SDs per axis (42875 cells, mass on the outer nodes at most
    # 3e-4), read with `.dense_axis_summary()`.
    ref <- rbind(
        b1.log_tau    = c( 1.3792, 0.2097,  0.9646,  1.7901),
        b1.atanh_psi1 = c( 1.3845, 0.0804,  1.2363,  1.5540),
        b1.atanh_psi2 = c(-0.9082, 0.1580, -1.2509, -0.6280))
    got <- .hp_table(fit)[rownames(ref), ]
    tol <- 0.25 * ref[, 2L]
    expect_true(all(abs(got[, "mean"] - ref[, 1L]) < tol))
    expect_true(all(abs(got[, "sd"] - ref[, 2L]) < tol))
    expect_true(all(abs(got[, "ci_lo"] - ref[, 3L]) < tol))
    expect_true(all(abs(got[, "ci_hi"] - ref[, 4L]) < tol))
})

test_that("a tgmrf block with a bounds box keeps its declared span", {
    skip_on_cran()
    set.seed(7)
    G <- 30L
    g <- rep(seq_len(G), each = 4L)
    u <- stats::rnorm(G, 0, 0.5)
    d <- data.frame(y = 0.2 + u[g] + stats::rnorm(length(g), 0, 0.4))
    blk <- tgmrf(
        Q = function(theta) Matrix::Diagonal(G, exp(-2 * theta[1])),
        prior = function(theta) stats::dnorm(theta[1], 0, 1, log = TRUE),
        init = c(log_sigma = 0), obs_idx = g, n_grid = 4L,
        bounds = list(lower = -2, upper = 1))
    fit <- suppressWarnings(tulpa(y ~ 1 + latent(blk), data = d,
                                  family = "gaussian", phi = 0.16,
                                  mode = "nested_laplace"))
    expect_identical(fit$outer_grid_placement, "fixed")
    expect_match(fit$outer_grid_recenter_declined, "^unguessable_axis")
    # Refinement may densify inside the box and never leaves it.
    nodes <- sort(unique(as.numeric(tulpa_outer_grid(fit)$nodes)))
    expect_true(all(seq(-2, 1, length.out = 4L) %in% nodes))
    expect_true(all(nodes >= -2 & nodes <= 1))
})
