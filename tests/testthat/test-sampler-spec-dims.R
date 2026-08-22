# Dimension agreement at the R boundary of the sampler and tgmrf block builders
# (gcol33/tulpa#472).
#
# Both builders read a dimension from one field of a spec and then index other
# fields of the same spec against it. Rcpp's matrix and vector subscripts carry
# no bounds test, so a spec whose fields disagree used to read past the SEXP
# data block -- and on the tgmrf dense prior path, WRITE past the end of the
# Hessian. Each test below hands in one field with the wrong shape and asserts
# the error names that field.

# A minimal NNGP spatial spec for the sampler path, at the shapes the builder
# requires. Each test perturbs exactly one field.
.gp_spec <- function(n_loc = 6L, nn = 2L) {
    co <- cbind(seq_len(n_loc), rev(seq_len(n_loc)))
    list(
        type       = "gp",
        coords     = co,
        nn         = nn,
        nn_idx     = matrix(0L, n_loc, nn),
        nn_dist    = matrix(0, n_loc, nn),
        nn_neighbor_dist = numeric(n_loc * nn * nn),
        nn_order   = seq_len(n_loc) - 1L,
        nn_order_inv = seq_len(n_loc) - 1L,
        obs_to_loc = seq_len(n_loc) - 1L,
        cov_type   = 0L,
        nu         = 0.5,
        phi_prior_U = 1, phi_prior_alpha = 0.5,
        sigma2_prior_U = 1, sigma2_prior_alpha = 0.5
    )
}

.layout_with <- function(sp, n_loc = 6L) {
    set.seed(1L)
    y <- rpois(n_loc, 3)
    X <- cbind(1, seq_len(n_loc) / n_loc)
    cpp_tulpa_glmm_layout(
        y = as.numeric(y), n_trials = rep(1L, n_loc), X = X,
        family = "poisson", spatial_spec = sp
    )
}

test_that("the GP sampler spec is accepted at the shapes it declares", {
    expect_type(.layout_with(.gp_spec()), "list")
})

test_that("a neighbour table whose shape disagrees with nn is named", {
    sp <- .gp_spec()
    sp$nn_idx <- matrix(0L, 6L, 1L)          # nn says 2
    expect_error(.layout_with(sp), "nn_idx is 6 x 1; must be 6 x 2")

    sp <- .gp_spec()
    sp$nn_dist <- matrix(0, 3L, 2L)          # n_loc says 6
    expect_error(.layout_with(sp), "nn_dist is 3 x 2; must be 6 x 2")
})

test_that("the neighbour-pair distance block is checked against n_loc * nn * nn", {
    sp <- .gp_spec()
    sp$nn_neighbor_dist <- numeric(6L * 2L)   # missing the second nn factor
    expect_error(.layout_with(sp), "length\\(nn_neighbor_dist\\)")
})

test_that("the ordering is checked for length and for range", {
    sp <- .gp_spec()
    sp$nn_order <- seq_len(3L) - 1L
    expect_error(.layout_with(sp), "length\\(nn_order\\)")

    sp <- .gp_spec()
    sp$nn_order_inv[2L] <- 99L
    expect_error(.layout_with(sp), "nn_order_inv\\[2\\]")
})

test_that("the random-effect term arrays must describe the same number of terms", {
    n <- 12L
    set.seed(2L)
    y <- rpois(n, 3)
    X <- cbind(1, seq_len(n) / n)
    g <- rep(0:2, each = 4L)
    re_ok <- list(idx = list(g), ngroups = 3L, ncoefs = 1L, correlated = FALSE)
    expect_type(cpp_tulpa_glmm_layout(as.numeric(y), rep(1L, n), X,
                                      "poisson", re_spec = re_ok), "list")

    re_bad <- re_ok
    re_bad$ngroups <- c(3L, 3L)              # two terms declared, one supplied
    expect_error(
        cpp_tulpa_glmm_layout(as.numeric(y), rep(1L, n), X, "poisson",
                              re_spec = re_bad),
        "length\\(ncoefs\\)"
    )
})

# --- tgmrf block factory ---------------------------------------------------
#
# Driven through the joint multi-block front door, which passes the R block
# spec straight through to make_tgmrf_block.

.tgmrf_block <- function(size = 4L, n_grid = 2L, mutate = identity) {
    # An AR1-ish tridiagonal Q, stored full in CSC.
    Q <- diag(2, size)
    for (i in seq_len(size - 1L)) { Q[i, i + 1L] <- -1; Q[i + 1L, i] <- -1 }
    csc <- function(M) {
        p <- integer(ncol(M) + 1L); i <- integer(0); x <- numeric(0)
        for (j in seq_len(ncol(M))) {
            nz <- which(M[, j] != 0)
            i <- c(i, nz - 1L); x <- c(x, M[nz, j]); p[j + 1L] <- length(i)
        }
        list(p = as.integer(p), i = as.integer(i), x = as.numeric(x))
    }
    one <- csc(Q)
    mutate(list(
        n_latent            = size,
        Q_csc_p_per_grid    = rep(list(one$p), n_grid),
        Q_csc_i_per_grid    = rep(list(one$i), n_grid),
        Q_csc_x_per_grid    = rep(list(one$x), n_grid),
        logdet_Q_per_grid   = rep(0, n_grid),
        log_prior_theta_per_grid = rep(0, n_grid)
    ))
}

test_that("a well-formed tgmrf block spec builds", {
    expect_identical(cpp_test_tgmrf_block_spec(.tgmrf_block()), 4L)
})

test_that("the tgmrf CSC frame is checked against n_latent", {
    expect_error(
        cpp_test_tgmrf_block_spec(.tgmrf_block(mutate = function(s) {
            s$Q_csc_i_per_grid[[1L]] <- s$Q_csc_i_per_grid[[1L]][-1L]
            s
        })),
        "Q_csc_i_per_grid"
    )
    expect_error(
        cpp_test_tgmrf_block_spec(.tgmrf_block(mutate = function(s) {
            s$Q_csc_i_per_grid[[1L]][1L] <- s$n_latent
            s
        })),
        "0-based row index"
    )
})

test_that("an empty per-grid list is refused rather than read at grid point 0", {
    expect_error(
        cpp_test_tgmrf_block_spec(.tgmrf_block(mutate = function(s) {
            s$Q_csc_p_per_grid <- list()
            s$Q_csc_i_per_grid <- list()
            s$Q_csc_x_per_grid <- list()
            s$logdet_Q_per_grid <- numeric(0)
            s$log_prior_theta_per_grid <- numeric(0)
            s
        })),
        "at least one outer-grid point"
    )
})

test_that("the symbolic frame is the union over grid points", {
    # SparseHessianBuilder::add discards an out-of-frame write, so a frame taken
    # from grid point 1 alone would silently drop what a later grid point holds
    # beyond it -- and grid point 1 can be the sparser one (an AR1 Q at rho = 0
    # is diagonal). The frame the block reports must cover every grid point.
    spec <- .tgmrf_block(mutate = function(s) {
        # grid point 1 diagonal only, grid point 2 tridiagonal
        n <- s$n_latent
        s$Q_csc_p_per_grid[[1L]] <- as.integer(0:n)
        s$Q_csc_i_per_grid[[1L]] <- as.integer(seq_len(n) - 1L)
        s$Q_csc_x_per_grid[[1L]] <- rep(2, n)
        s
    })
    expect_identical(cpp_test_tgmrf_block_spec(spec), 4L)
    pat <- cpp_test_tgmrf_block_pattern(spec)
    # The off-diagonal pairs of grid point 2 are in the frame even though grid
    # point 1 does not carry them.
    expect_true(all(paste(2:4, 1:3) %in% paste(pat$row, pat$col)))
})
