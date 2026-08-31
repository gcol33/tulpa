# The copy coefficient's outer axis, and the resolution knob it lacked
# (gcol33/tulpa#633).
#
# Every other outer axis a copy fit carries can be raised by the caller. The
# alpha axis could not: `alpha_grid` REPLACES it, and the axis carries prior
# structure -- the atom at 0, which is what gives the "no copy" base model
# posterior mass, plus a log-spaced slab -- so a caller who only wants a finer
# integration cannot supply nodes without also restating the prior. Consumers
# close `alpha.grid` off entirely under `copy()` for exactly that reason, which
# left no way to raise the axis at all.
#
# Measured engine-side before the fix (dev_notes/issue633/probe_alpha_engine.R):
# raising the donor `sigma_grid` 13 -> 21 -> 29 leaves the alpha axis at its
# declared 6 nodes at every setting, and grid ESS at 1.7 / 3.1 / 4.3 while the
# cell count more than doubles. With the resolution raised alongside, ESS runs
# 2.3 / 6.8 / 12.5. The saturation is in the PLACEMENT, not in the prune:
# `prune = TRUE` reproduces the same node counts and the same ESS to the digit.

test_that("a resolution override keeps the axis's declared shape", {
    base <- tulpa:::.nl_grid_axis("copy_alpha")
    expect_identical(base[1L], 0)                 # the atom
    expect_length(base, 6L)                       # atom + 5 slab nodes

    for (n in c(5L, 12L, 28L)) {
        ax <- tulpa:::.nl_grid_axis("copy_alpha", n = n)
        expect_length(ax, n + 1L)                 # `n` counts the SLAB nodes
        expect_identical(ax[1L], 0)               # atom preserved
        expect_equal(range(ax[-1L]), range(base[-1L]))   # slab bounds preserved
        expect_identical(ax, sort(ax))
        expect_false(anyDuplicated(ax) > 0L)
    }

    # The declared resolution reproduces the declared axis exactly, so the
    # knob's absence and its identity value are the same fit.
    expect_identical(tulpa:::.nl_grid_axis("copy_alpha",
                                           n = tulpa:::.nl_grid_par("copy_alpha", "n")),
                     base)
})

test_that("a resolution override is refused where there is no resolution", {
    # An axis declared as explicit nodes has no `lo`/`hi` to redistribute
    # between, so raising it would have to invent a rule.
    expect_error(tulpa:::.nl_grid_axis("bym2_rho", n = 9L), "explicit nodes")
    expect_error(tulpa:::.nl_grid_axis("copy_alpha", n = 0L), "integer >= 1")
    expect_error(tulpa:::.nl_grid_axis("copy_alpha", n = c(3L, 4L)), "integer >= 1")
})

test_that("alpha_grid and alpha_n answer different questions and cannot be mixed", {
    # Not ranked silently: one states the nodes, the other states how many.
    expect_error(tulpa:::.nl_copy_alpha_axis(c(0, 1, 2), 9L),
                 "not both")
    expect_identical(tulpa:::.nl_copy_alpha_axis(c(0, 1, 2), NULL), c(0, 1, 2))
    expect_identical(tulpa:::.nl_copy_alpha_axis(NULL, NULL),
                     tulpa:::.nl_grid_axis("copy_alpha"))
    expect_identical(tulpa:::.nl_copy_alpha_axis(NULL, 11L),
                     tulpa:::.nl_grid_axis("copy_alpha", n = 11L))
    # An empty grid is not a request, so it falls through to the default rather
    # than colliding with a resolution.
    expect_identical(tulpa:::.nl_copy_alpha_axis(numeric(0), 7L),
                     tulpa:::.nl_grid_axis("copy_alpha", n = 7L))
})

test_that("the multi-block copy resolver reads alpha_n", {
    spec <- list(arm = 2L, block = 1L, alpha_n = 9L)
    got <- tulpa:::.nl_copy_alpha_axis(spec$alpha_grid, spec$alpha_n)
    expect_length(got, 10L)
    expect_identical(got[1L], 0)
})

test_that("alpha_n reaches the fitted grid and raises its resolution", {
    skip_on_cran()
    # End to end: the axis the fit actually integrates over carries the
    # requested resolution, and the atom survives into it.
    n_s <- 12L
    adj <- local({
        rp <- integer(n_s + 1L); ci <- integer(0); nb <- integer(n_s)
        for (i in seq_len(n_s)) {
            nbr <- c(if (i > 1L) i - 1L, if (i < n_s) i + 1L)
            ci <- c(ci, nbr - 1L); nb[i] <- length(nbr); rp[i + 1L] <- length(ci)
        }
        list(n_spatial_units = n_s, adj_row_ptr = rp,
             adj_col_idx = ci, n_neighbors = nb)
    })
    set.seed(4)
    f <- as.numeric(scale(cumsum(rnorm(n_s)))) * 0.8
    N <- n_s * 4L; s <- rep(seq_len(n_s), each = 4L)
    X <- cbind(1, rnorm(N))
    arm1 <- list(y = rbinom(N, 1L, 1 / (1 + exp(-(X %*% c(0.2, 0.5) + f[s])))),
                 n_trials = rep(1L, N), X = X, re_idx = rep(0, N),
                 n_re_groups = 0L, sigma_re = 1.0, family = "binomial", phi = 1.0)
    arm2 <- list(y = as.numeric(rnorm(N, X %*% c(0.1, 0.2) + f[s], 0.3)),
                 n_trials = rep(1L, N), X = X, re_idx = rep(0, N),
                 n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 1.0)
    block <- c(adj, list(type = "icar", spatial_idx = list(s, s),
                         sigma_grid = exp(seq(log(0.1), log(3), length.out = 4L))))

    fit_of <- function(alpha_n) suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = arm1, pos = arm2), prior = list(block),
        copy = list(list(arm = "pos", block = 1L, alpha_n = alpha_n)),
        control = list(diagnose_k = FALSE)))

    n_alpha <- function(fit) length(unique(fit$theta_grid[, "b1.alpha"]))
    expect_identical(n_alpha(fit_of(NULL)),
                     length(tulpa:::.nl_grid_axis("copy_alpha")))
    expect_identical(n_alpha(fit_of(11L)), 12L)
    expect_true(0 %in% fit_of(11L)$theta_grid[, "b1.alpha"])
    # and it is a resolution, so it only adds cells
    expect_gt(nrow(fit_of(11L)$theta_grid), nrow(fit_of(NULL)$theta_grid))
})

test_that("a named field_coef is not read as a resolution request", {
    # `$` PARTIAL-matches on a list, so `fc$n` resolves to `fc$name` on every
    # spec that names its coefficient -- which fed a character into the integer
    # check and errored every existing `field_coef = list(name = , grid = )`
    # fixture. The reads are `[[`.
    a <- list(field_coef = list(name = "alpha", grid = c(0, 0.5, 1)))
    got <- tulpa:::.normalise_arm_field_coef(a, 1L)
    expect_identical(got$field_coef_axis$name, "alpha")
    expect_identical(got$field_coef_axis$grid, c(0, 0.5, 1))
    expect_null(got$field_coef_axis$alpha_n)

    # and a genuine resolution request still lands
    b <- list(field_coef = list(name = "alpha", n = 9L))
    expect_identical(tulpa:::.normalise_arm_field_coef(b, 1L)$field_coef_axis$alpha_n,
                     9L)
    expect_error(tulpa:::.normalise_arm_field_coef(
        list(field_coef = list(name = "alpha", n = 0L)), 1L), "integer >= 1")
})
