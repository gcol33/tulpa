# What the joint multi-block spec builder verifies before a solve runs.
#
# Both guards here are checked at block-build time, so cpp_test_joint_pattern()
# reaches them without any Newton iteration: it builds the LatentBlocks and
# enumerates their sparsity, which is all either guard needs.

skip_on_cran()

.bsg_adj <- function(n_s = 3L) {
    nbr <- vector("list", n_s)
    nbr[[1]] <- 2L
    nbr[[n_s]] <- n_s - 1L
    if (n_s >= 3L) for (s in seq_len(n_s - 2L) + 1L) nbr[[s]] <- c(s - 1L, s + 1L)
    list(adj_row_ptr = as.integer(c(0L, cumsum(lengths(nbr)))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(lengths(nbr)),
         n_spatial_units = n_s)
}

.bsg_arm <- function(X, N = 3L) {
    list(y = rep(0.0, N), n_trials = rep(1L, N), X = X,
         re_idx = rep(0L, N), n_re_groups = 0L, sigma_re = 1.0,
         spatial_idx = seq_len(N), temporal_idx = rep(0L, N),
         obs_idx = rep(0L, N), family = "gaussian", phi = 1.0)
}

.bsg_pattern <- function(arms, bs, n_axes = 1L) {
    cpp_test_joint_pattern(
        arms_list = arms, copy_arms = -1L, copy_blocks = -1L,
        blocks_spec = list(bs), theta_grid = matrix(0.5, 1, n_axes),
        axis_offsets = as.integer(c(0, n_axes)))
}

# --------------------------------------------------------------------------- #
# (gcol33/tulpa#450) The uniform centering fold lands on an intercept          #
#                                                                              #
# center_joint removes the field's constant and compensates it by adding into  #
# coefficient beta_offset of each arm. eta survives that only if the column at #
# that offset is constant 1 on every arm: the centerer removes amount * d_fac  #
# uniformly while the fold shifts eta by amount * d_fac * X_k(i, offset). A    #
# weighted (SVC / TVC) field verified its alias column; the uniform path took  #
# column 0 on faith, and the documented joint API takes X directly, so nothing #
# upstream supplied one. The failure is silent and lives in the RETURNED       #
# LATENT: log_marginal and the SEs are read before centering, so only fitted   #
# values and predictions computed from the mode are wrong.                     #
# --------------------------------------------------------------------------- #

test_that("an all-ones column 0 is what an unweighted intrinsic block folds into", {
    adj <- .bsg_adj()
    bs  <- c(list(type = "icar", spatial_idx = list(1:3)), adj)
    pat <- .bsg_pattern(list(.bsg_arm(matrix(1.0, 3L, 1L))), bs)
    expect_equal(pat$n_x, 1L + 3L)
    # Intercept plus further covariates is still an intercept at offset 0.
    pat2 <- .bsg_pattern(list(.bsg_arm(cbind(1, c(-0.4, 0.2, 1.1)))), bs)
    expect_equal(pat2$n_x, 2L + 3L)
})

test_that("a column 0 that is not the intercept is refused, naming arm and row", {
    adj <- .bsg_adj()
    bs  <- c(list(type = "icar", spatial_idx = list(1:3)), adj)
    expect_error(
        .bsg_pattern(list(.bsg_arm(matrix(c(1, 2, 3), 3L, 1L))), bs),
        "arm 1 obs 2 has 2 against 1")
    # The offending arm is named even when it is not the first.
    two <- list(.bsg_arm(matrix(1.0, 3L, 1L)),
                .bsg_arm(cbind(c(1, 1, 0.5), c(0.1, -0.2, 0.3))))
    bs2 <- c(list(type = "icar", spatial_idx = list(1:3, 1:3)), adj)
    expect_error(.bsg_pattern(two, bs2), "arm 2 obs 3")
})

test_that("every unweighted intrinsic block type is checked from the one place", {
    # install_field_center is where either centerer is installed, so a block
    # type inherits the verification rather than repeating it. Only the
    # INTRINSIC blocks reach it: AR1 and proper CAR have full-rank precisions,
    # so their field level is identified by the prior and nothing is folded.
    bad <- .bsg_arm(matrix(c(1, 2, 3), 3L, 1L))
    adj <- .bsg_adj()
    specs <- list(
        list(bs = c(list(type = "icar", spatial_idx = list(1:3)), adj), k = 1L),
        list(bs = c(list(type = "bym2", spatial_idx = list(1:3)), adj), k = 2L),
        list(bs = list(type = "rw1", temporal_idx = list(1:3), n_times = 3L),
             k = 1L),
        list(bs = list(type = "rw2", temporal_idx = list(1:3), n_times = 3L),
             k = 1L))
    for (sp in specs) {
        expect_error(.bsg_pattern(list(bad), sp$bs, sp$k),
                     "design column must carry the field weight",
                     label = paste("block type", sp$bs$type))
    }
    # AR1 on the same non-intercept design builds without complaint.
    expect_silent(.bsg_pattern(
        list(bad), list(type = "ar1", temporal_idx = list(1:3), n_times = 3L),
        2L))
})

# --------------------------------------------------------------------------- #
# (gcol33/tulpa#456) The per-arm index / weight closures own their bound       #
#                                                                              #
# The cached closures read cache[k_arm][i] for every row i of arm k with no    #
# bounds check, so a per-arm vector shorter than that arm is a read past the   #
# allocation the first time a grid cell scatters those rows. R validates the   #
# length where the spec is assembled; the factory checks it again where the    #
# vector is taken, so the guarantee is local to the consumer rather than two   #
# files away.                                                                  #
# --------------------------------------------------------------------------- #

test_that("a short per-arm index vector is an error, not a read past the end", {
    adj <- .bsg_adj()
    arm <- .bsg_arm(matrix(1.0, 3L, 1L))
    expect_error(
        .bsg_pattern(list(arm),
                     c(list(type = "icar", spatial_idx = list(1:2)), adj)),
        "holds 2 entries but arm 1 has 3 observations")
    # A LONGER vector is not an error: only the arm's own rows are ever read.
    expect_silent(
        .bsg_pattern(list(arm),
                     c(list(type = "icar", spatial_idx = list(c(1L, 2L, 3L, 1L))),
                       adj)))
})

test_that("a short per-arm svc_weight is refused on the same footing", {
    adj <- .bsg_adj()
    arm <- .bsg_arm(cbind(1, c(0.4, -0.2, 0.9)))
    bs  <- c(list(type = "icar", spatial_idx = list(1:3),
                  svc_weight = list(c(0.4, -0.2)), svc_beta_offset = 1L), adj)
    err <- tryCatch(.bsg_pattern(list(arm), bs), error = conditionMessage)
    expect_match(err, "svc_weight", fixed = TRUE)
    expect_match(err, "holds 2 entries but arm 1 has 3", fixed = TRUE)
})
