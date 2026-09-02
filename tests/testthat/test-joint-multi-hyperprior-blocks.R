# Regularizing hyperpriors on a multi-block joint grid (gcol33/tulpa#655).
#
# `prior_sigma` / `prior_alpha` name a ROLE -- the donor field amplitude and
# the copy coefficient -- and a multi-block grid can carry that role on several
# blocks at once: two copied fields carry `b1.alpha` and `b2.alpha`. The
# contract held here:
#
#   * one spec on a two-copy-block grid reaches BOTH blocks, where it used to
#     be folded onto the first block carrying the axis and no other;
#   * a per-block spec reaches each named block with that block's own density,
#     and no unnamed block;
#   * an entry naming a block that carries no axis of that role errors, and the
#     message names the blocks that do, so the caller can see what to key by;
#   * a grid carrying ONE axis of a role folds exactly what the whole-view fold
#     produced, so the single-copy-block fit -- the common case -- is unmoved.

# --- fixtures --------------------------------------------------------------

# A two-copy-block outer grid. Each block carries the (sigma, alpha) pair
# `.joint_block_axis_grid()` lays for a copy block, so both blocks carry both
# roles. Each alpha axis carries its "no copy" zero.
hpb_grid_two <- function() {
    g <- as.matrix(expand.grid(b1.sigma = c(0.5, 1.0),
                               b1.alpha = c(0, 0.8, 1.6),
                               b2.sigma = c(0.7, 1.3),
                               b2.alpha = c(0, 1.1),
                               KEEP.OUT.ATTRS = FALSE))
    list(grid = g, axis_offsets = c(0L, 2L, 4L), B = 2L,
         axis_names = colnames(g))
}

# A grid carrying (sigma, alpha) on block 1 only; block 2 is a non-copy
# temporal block whose axis is a precision, named by neither role.
hpb_grid_one <- function() {
    g <- as.matrix(expand.grid(b1.sigma = c(0.5, 1.0),
                               b1.alpha = c(0, 0.8, 1.6),
                               b2.tau   = c(1, 4),
                               KEEP.OUT.ATTRS = FALSE))
    list(grid = g, axis_offsets = c(0L, 2L, 3L), B = 2L,
         axis_names = colnames(g))
}

# The per-cell contribution one PC(U, alpha) density makes to `log_marginal`
# on a log-integration axis, written out independently of the engine: the
# density on the axis's natural scale plus the log-Jacobian carrying it to the
# integration coordinate, and zero at a declared point mass.
hpb_pc_contrib <- function(x, U, a, atom = FALSE) {
    lambda <- -log(a) / U
    out <- log(lambda) - lambda * x + log(x)
    out[!is.finite(out)] <- -Inf
    if (atom) out[x == 0] <- 0
    out
}

hpb_pc <- function(U, a) list("pc.prec", c(U, a))

# --- tier 1: shapes, parsing, refusals -------------------------------------

test_that("the per-block shape is told from the family spec", {
    expect_false(tulpa:::.joint_is_block_hyperprior(hpb_pc(4, 0.01)))
    expect_false(tulpa:::.joint_is_block_hyperprior(list("half_normal", 2)))
    expect_false(tulpa:::.joint_is_block_hyperprior(NULL))
    expect_false(tulpa:::.joint_is_block_hyperprior(list()))
    expect_true(tulpa:::.joint_is_block_hyperprior(
        list(list(block = 1, prior = hpb_pc(4, 0.01)))))
    expect_true(tulpa:::.joint_is_block_hyperprior(
        list(list(block = 1, prior = hpb_pc(4, 0.01)),
             list(block = 2, prior = list("half_normal", 2)))))
    # An entry without a block is not the per-block shape, so it stays a
    # family spec and is refused as one.
    expect_false(tulpa:::.joint_is_block_hyperprior(
        list(list(prior = hpb_pc(4, 0.01)))))
})

test_that("one spec parses to one density, a per-block list to one per block", {
    fn <- tulpa:::.joint_parse_hyperprior(hpb_pc(4, 0.01), "prior_alpha",
                                          multi_block = TRUE)
    expect_true(is.function(fn))
    expect_equal(fn(1.0), log(-log(0.01) / 4) - (-log(0.01) / 4))

    by_b <- tulpa:::.joint_parse_hyperprior(
        list(list(block = 1, prior = hpb_pc(4, 0.01)),
             list(block = 3, prior = hpb_pc(2, 0.05))),
        "prior_alpha", multi_block = TRUE)
    expect_s3_class(by_b, "tulpa_block_hyperprior")
    expect_equal(sort(names(by_b)), c("1", "3"))
    expect_true(all(vapply(by_b, is.function, logical(1))))
    expect_equal(tulpa:::.joint_hp_named_blocks(by_b), c(1L, 3L))
    # A single spec names no block: it reaches every block carrying the axis.
    expect_equal(tulpa:::.joint_hp_named_blocks(fn), integer(0))
    expect_equal(tulpa:::.joint_hp_named_blocks(NULL), integer(0))

    # The density each block gets is that block's own.
    expect_identical(tulpa:::.joint_hp_fn_for_block(by_b, 1L)(1.0),
                     by_b[["1"]](1.0))
    expect_identical(tulpa:::.joint_hp_fn_for_block(by_b, 3L)(1.0),
                     by_b[["3"]](1.0))
    expect_null(tulpa:::.joint_hp_fn_for_block(by_b, 2L))
    # A single spec answers for any block.
    expect_identical(tulpa:::.joint_hp_fn_for_block(fn, 7L)(1.0), fn(1.0))
    expect_null(tulpa:::.joint_hp_fn_for_block(NULL, 1L))
})

test_that("a malformed per-block hyperprior is refused where it is written", {
    expect_error(
        tulpa:::.joint_parse_hyperprior(
            list(list(block = 0, prior = hpb_pc(4, 0.01))),
            "prior_alpha", multi_block = TRUE),
        "1-based block index")
    expect_error(
        tulpa:::.joint_parse_hyperprior(
            list(list(block = c(1, 2), prior = hpb_pc(4, 0.01))),
            "prior_alpha", multi_block = TRUE),
        "1-based block index")
    expect_error(
        tulpa:::.joint_parse_hyperprior(
            list(list(block = 1, prior = hpb_pc(4, 0.01)),
                 list(block = 1, prior = hpb_pc(2, 0.01))),
            "prior_alpha", multi_block = TRUE),
        "names block 1 twice")
    expect_error(
        tulpa:::.joint_parse_hyperprior(
            list(list(block = 1)), "prior_alpha", multi_block = TRUE),
        "no `prior` field")
    # The family spec inside an entry goes through the same family parser.
    expect_error(
        tulpa:::.joint_parse_hyperprior(
            list(list(block = 1, prior = list("nope", 1))),
            "prior_alpha", multi_block = TRUE),
        "Unknown hyperprior family")
    # A per-block list names blocks of a list-of-blocks prior.
    expect_error(
        tulpa:::.joint_parse_hyperprior(
            list(list(block = 1, prior = hpb_pc(4, 0.01))),
            "prior_alpha", multi_block = FALSE),
        "single-block joint")
    # The single spec is unchanged on either path.
    expect_true(is.function(tulpa:::.joint_parse_hyperprior(
        hpb_pc(4, 0.01), "prior_alpha", multi_block = FALSE)))
    expect_null(tulpa:::.joint_parse_hyperprior(NULL, "prior_alpha",
                                                multi_block = TRUE))
})

test_that("a block with no axes of its own contributes no column", {
    # A latent-factor block carries a 1 x 0 grid, so its column span is empty:
    # block 2 here owns no column and must not be read as owning block 1's or
    # block 3's.
    axis_names   <- c("b1.sigma", "b1.alpha", "b3.sigma", "b3.alpha")
    axis_offsets <- c(0L, 2L, 2L, 4L)
    cols <- tulpa:::.joint_block_axis_cols(axis_names, axis_offsets, 3L, "sigma")
    expect_equal(names(cols), c("1", "3"))
    expect_equal(unname(cols), c(1L, 3L))
    cols_a <- tulpa:::.joint_block_axis_cols(axis_names, axis_offsets, 3L,
                                             "alpha")
    expect_equal(names(cols_a), c("1", "3"))
    expect_equal(unname(cols_a), c(2L, 4L))
    # A role no block carries maps to nothing.
    expect_length(tulpa:::.joint_block_axis_cols(axis_names, axis_offsets, 3L,
                                                 "tau"), 0L)
})

# --- tier 1: which axes one spec reaches -----------------------------------

test_that("one spec reaches every block carrying the axis, not the first", {
    fx <- hpb_grid_two()
    fn <- tulpa:::.joint_parse_hyperprior(hpb_pc(4, 0.01), "prior_alpha",
                                          multi_block = TRUE)
    ent <- tulpa:::.joint_multi_hp_cols(fx$grid, fx$axis_offsets, fx$B,
                                        fn_sigma = NULL, fn_alpha = fn)
    expect_length(ent, 2L)
    expect_equal(vapply(ent, function(e) e[["block"]], integer(1)), c(1L, 2L))
    expect_equal(colnames(fx$grid)[vapply(ent, function(e) e[["col"]],
                                          integer(1))],
                 c("b1.alpha", "b2.alpha"))
    expect_true(all(vapply(ent, function(e) identical(e[["role"]], "alpha"),
                           logical(1))))

    # Both roles at once: sigma entries first (block order), then alpha.
    fs <- tulpa:::.joint_parse_hyperprior(hpb_pc(3, 0.01), "prior_sigma",
                                          multi_block = TRUE)
    ent2 <- tulpa:::.joint_multi_hp_cols(fx$grid, fx$axis_offsets, fx$B,
                                         fn_sigma = fs, fn_alpha = fn)
    expect_equal(colnames(fx$grid)[vapply(ent2, function(e) e[["col"]],
                                          integer(1))],
                 c("b1.sigma", "b2.sigma", "b1.alpha", "b2.alpha"))
})

test_that("a per-block spec reaches only the blocks it names", {
    fx <- hpb_grid_two()
    fn <- tulpa:::.joint_parse_hyperprior(
        list(list(block = 2, prior = hpb_pc(4, 0.01))),
        "prior_alpha", multi_block = TRUE)
    ent <- tulpa:::.joint_multi_hp_cols(fx$grid, fx$axis_offsets, fx$B,
                                        fn_sigma = NULL, fn_alpha = fn)
    expect_length(ent, 1L)
    expect_equal(ent[[1L]][["block"]], 2L)
    expect_equal(colnames(fx$grid)[ent[[1L]][["col"]]], "b2.alpha")
})

# --- tier 1: what the fold adds --------------------------------------------

test_that("one spec on two copy blocks folds BOTH alpha axes", {
    fx <- hpb_grid_two()
    fn <- tulpa:::.joint_parse_hyperprior(hpb_pc(4, 0.01), "prior_alpha",
                                          multi_block = TRUE)
    lm0 <- numeric(nrow(fx$grid))
    got <- tulpa:::.joint_multi_add_hp(lm0, fx$grid, fx$axis_offsets, fx$B,
                                       fn_sigma = NULL, fn_alpha = fn)

    c1 <- hpb_pc_contrib(fx$grid[, "b1.alpha"], 4, 0.01, atom = TRUE)
    c2 <- hpb_pc_contrib(fx$grid[, "b2.alpha"], 4, 0.01, atom = TRUE)
    expect_equal(got, unname(c1 + c2))

    # The defect: block 1 alone. The fold must no longer equal it, and the
    # difference is exactly block 2's own contribution.
    expect_false(isTRUE(all.equal(got, unname(c1))))
    expect_equal(got - unname(c1), unname(c2))
})

test_that("a per-block spec folds each named block's own density", {
    fx <- hpb_grid_two()
    fn <- tulpa:::.joint_parse_hyperprior(
        list(list(block = 1, prior = hpb_pc(4, 0.01)),
             list(block = 2, prior = hpb_pc(1, 0.05))),
        "prior_alpha", multi_block = TRUE)
    lm0 <- numeric(nrow(fx$grid))
    got <- tulpa:::.joint_multi_add_hp(lm0, fx$grid, fx$axis_offsets, fx$B,
                                       fn_sigma = NULL, fn_alpha = fn)
    c1 <- hpb_pc_contrib(fx$grid[, "b1.alpha"], 4, 0.01, atom = TRUE)
    c2 <- hpb_pc_contrib(fx$grid[, "b2.alpha"], 1, 0.05, atom = TRUE)
    expect_equal(got, unname(c1 + c2))

    # Naming one block leaves the other flat.
    fn1 <- tulpa:::.joint_parse_hyperprior(
        list(list(block = 2, prior = hpb_pc(1, 0.05))),
        "prior_alpha", multi_block = TRUE)
    got1 <- tulpa:::.joint_multi_add_hp(lm0, fx$grid, fx$axis_offsets, fx$B,
                                        fn_sigma = NULL, fn_alpha = fn1)
    expect_equal(got1, unname(c2))
})

test_that("each block's copy-scale atom is read on its own axis", {
    # The zero of an alpha axis is that axis's declared "no coupling" point
    # mass, so the density is not read there -- on EVERY block, not only the
    # first.
    fx <- hpb_grid_two()
    fn <- tulpa:::.joint_parse_hyperprior(hpb_pc(4, 0.01), "prior_alpha",
                                          multi_block = TRUE)
    got <- tulpa:::.joint_multi_add_hp(numeric(nrow(fx$grid)), fx$grid,
                                       fx$axis_offsets, fx$B,
                                       fn_sigma = NULL, fn_alpha = fn)
    both_zero <- fx$grid[, "b1.alpha"] == 0 & fx$grid[, "b2.alpha"] == 0
    expect_true(any(both_zero))
    expect_true(all(got[both_zero] == 0))
    expect_true(all(is.finite(got)))
})

test_that("a grid carrying one axis of a role folds what the whole view did", {
    # The single-copy-block case, the common one: the per-axis fold must
    # reproduce the whole-view call to the bit, so an existing fit is unmoved.
    fx <- hpb_grid_one()
    fs <- tulpa:::.joint_parse_hyperprior(hpb_pc(3, 0.01), "prior_sigma",
                                          multi_block = TRUE)
    fa <- tulpa:::.joint_parse_hyperprior(hpb_pc(4, 0.01), "prior_alpha",
                                          multi_block = TRUE)
    lm0 <- as.numeric(seq_len(nrow(fx$grid)))

    view <- fx$grid[, c("b1.sigma", "b1.alpha"), drop = FALSE]
    colnames(view) <- c("sigma", "alpha")
    ref <- lm0 + tulpa:::.joint_hp_vec_for_grids(view, fs, fa, NULL)

    got <- tulpa:::.joint_multi_add_hp(lm0, fx$grid, fx$axis_offsets, fx$B,
                                       fn_sigma = fs, fn_alpha = fa)
    expect_identical(got, ref)

    # The block-2 tau axis is named by neither role and stays flat.
    fx2 <- fx
    fx2$grid[, "b2.tau"] <- fx$grid[, "b2.tau"] * 10
    got2 <- tulpa:::.joint_multi_add_hp(lm0, fx2$grid, fx$axis_offsets, fx$B,
                                        fn_sigma = fs, fn_alpha = fa)
    expect_identical(got2, got)
})

test_that("no hyperprior leaves log_marginal untouched", {
    fx <- hpb_grid_two()
    lm0 <- as.numeric(seq_len(nrow(fx$grid)))
    expect_identical(
        tulpa:::.joint_multi_add_hp(lm0, fx$grid, fx$axis_offsets, fx$B,
                                    fn_sigma = NULL, fn_alpha = NULL),
        lm0)
    # A prior whose role no block carries reaches nothing.
    fs <- tulpa:::.joint_parse_hyperprior(hpb_pc(3, 0.01), "prior_sigma",
                                          multi_block = TRUE)
    g <- fx$grid[, c("b1.alpha", "b2.alpha"), drop = FALSE]
    expect_identical(
        tulpa:::.joint_multi_add_hp(lm0, g, c(0L, 1L, 2L), 2L,
                                    fn_sigma = fs, fn_alpha = NULL),
        lm0)
})

# --- tier 1: the refusal and the record ------------------------------------

test_that("a per-block entry naming a block with no such axis is refused", {
    fx <- hpb_grid_two()
    fn <- tulpa:::.joint_parse_hyperprior(
        list(list(block = 3, prior = hpb_pc(4, 0.01))),
        "prior_alpha", multi_block = TRUE)
    err <- expect_error(
        tulpa:::.joint_check_multi_hyperpriors(fx$axis_names, fx$axis_offsets,
                                               fx$B, NULL, fn),
        "prior_alpha")
    msg <- conditionMessage(err)
    # The message names the blocks that DO carry the axis, and the columns
    # they are, so the caller can see what to key a list by.
    expect_match(msg, "names block 3")
    expect_match(msg, "blocks carrying one are 1, 2")
    expect_match(msg, "b1.alpha, b2.alpha", fixed = TRUE)

    # Same refusal on the sigma side; the two roles are not handled apart.
    fs <- tulpa:::.joint_parse_hyperprior(
        list(list(block = 5, prior = hpb_pc(3, 0.01))),
        "prior_sigma", multi_block = TRUE)
    expect_error(
        tulpa:::.joint_check_multi_hyperpriors(fx$axis_names, fx$axis_offsets,
                                               fx$B, fs, NULL),
        "b1.sigma, b2.sigma", fixed = TRUE)
})

test_that("the record says which axes each hyperprior reached", {
    fx <- hpb_grid_two()
    fs <- tulpa:::.joint_parse_hyperprior(hpb_pc(3, 0.01), "prior_sigma",
                                          multi_block = TRUE)
    fa <- tulpa:::.joint_parse_hyperprior(
        list(list(block = 2, prior = hpb_pc(4, 0.01))),
        "prior_alpha", multi_block = TRUE)
    rec <- tulpa:::.joint_check_multi_hyperpriors(fx$axis_names,
                                                  fx$axis_offsets, fx$B, fs, fa)
    expect_equal(sort(names(rec)), c("alpha", "sigma"))
    expect_equal(rec$sigma$scope, "every_block")
    expect_equal(rec$sigma$blocks, c(1L, 2L))
    expect_equal(rec$sigma$axes, c("b1.sigma", "b2.sigma"))
    expect_equal(rec$alpha$scope, "per_block")
    expect_equal(rec$alpha$blocks, 2L)
    expect_equal(rec$alpha$axes, "b2.alpha")

    # No hyperprior, no record.
    expect_null(tulpa:::.joint_check_multi_hyperpriors(
        fx$axis_names, fx$axis_offsets, fx$B, NULL, NULL))
    # A role no block carries is recorded as reaching nothing, not refused:
    # `prior_alpha` on a fit with no copy is documented as inert.
    rec1 <- tulpa:::.joint_check_multi_hyperpriors(
        c("b1.tau", "b2.tau"), c(0L, 1L, 2L), 2L, fs, NULL)
    expect_equal(rec1$sigma$blocks, integer(0))
    expect_equal(rec1$sigma$axes, character(0))
})

# --- tier 2: the front door ------------------------------------------------

# Two independent shared ICAR fields on a chain graph, each copied onto arm 2
# with its own alpha axis -- the shape a cover model with a weighted trend
# builds, and the configuration the first-block fold silently narrowed.
hpb_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn),
         n_spatial_units = n_s)
}

hpb_two_copy_fixture <- function(seed = 1L) {
    set.seed(seed)
    n_s <- 30L
    adj <- hpb_chain_adj(n_s)
    f1 <- cumsum(rnorm(n_s)); f1 <- (f1 - mean(f1)) / sd(f1)
    f2 <- cumsum(rnorm(n_s)); f2 <- (f2 - mean(f2)) / sd(f2)
    N <- 200L
    s1 <- sample.int(n_s, N, replace = TRUE)
    s2 <- sample.int(n_s, N, replace = TRUE)
    X1 <- cbind(1, rnorm(N))
    X2 <- cbind(1, rnorm(N))
    eta1 <- X1 %*% c(0.2, 0.5) + f1[s1] + f2[s1]
    eta2 <- X2 %*% c(-0.1, 0.3) + 1.2 * f1[s2] + 0.6 * f2[s2]
    arm <- function(y, X) list(y = as.numeric(y), n_trials = rep(1L, N), X = X,
                               re_idx = rep(0, N), n_re_groups = 0L,
                               sigma_re = 1.0, family = "gaussian", phi = 1.0)
    blk <- function(idx1, idx2) c(adj, list(
        type = "icar", sigma_grid = c(0.7, 1.2),
        spatial_idx = list(idx1, idx2)))
    list(responses = list(occ = arm(rnorm(N, eta1, 0.3), X1),
                          pos = arm(rnorm(N, eta2, 0.3), X2)),
         prior = list(blk(s1, s2), blk(s1, s2)),
         copy = list(list(arm = "pos", block = 1L, alpha_grid = c(0.8, 1.2)),
                     list(arm = "pos", block = 2L, alpha_grid = c(0.4, 0.8))))
}

test_that("one prior_alpha reaches both copy blocks of a fitted grid", {
    skip_on_cran()
    fx <- hpb_two_copy_fixture()
    fit <- suppressWarnings(suppressMessages(tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior, copy = fx$copy,
        prior_alpha = list("pc.prec", c(4.0, 0.01)),
        control = list(max_iter = 30L, tol = 1e-6, diagnose_k = FALSE,
                       diagnose_skew = FALSE, auto_recenter = FALSE,
                       integration = "grid"))))
    rec <- fit$hyperprior_axes
    expect_false(is.null(rec))
    expect_equal(rec$alpha$scope, "every_block")
    expect_equal(rec$alpha$blocks, c(1L, 2L))
    expect_equal(rec$alpha$axes, c("b1.alpha", "b2.alpha"))
    expect_null(rec$sigma)
    expect_true(all(is.finite(fit$log_marginal)))
})

test_that("a per-block prior_alpha reaches the block it names", {
    skip_on_cran()
    fx <- hpb_two_copy_fixture()
    fit <- suppressWarnings(suppressMessages(tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior, copy = fx$copy,
        prior_alpha = list(
            list(block = 2, prior = list("pc.prec", c(1.0, 0.05)))),
        control = list(max_iter = 30L, tol = 1e-6, diagnose_k = FALSE,
                       diagnose_skew = FALSE, auto_recenter = FALSE,
                       integration = "grid"))))
    rec <- fit$hyperprior_axes
    expect_equal(rec$alpha$scope, "per_block")
    expect_equal(rec$alpha$blocks, 2L)
    expect_equal(rec$alpha$axes, "b2.alpha")
    expect_true(all(is.finite(fit$log_marginal)))
})

test_that("the two shapes do not produce the same posterior", {
    skip_on_cran()
    fx <- hpb_two_copy_fixture()
    ctrl <- list(max_iter = 30L, tol = 1e-6, diagnose_k = FALSE,
                 diagnose_skew = FALSE, auto_recenter = FALSE,
                 integration = "grid")
    every <- suppressWarnings(suppressMessages(tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior, copy = fx$copy,
        prior_alpha = list("pc.prec", c(4.0, 0.01)), control = ctrl)))
    first <- suppressWarnings(suppressMessages(tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior, copy = fx$copy,
        prior_alpha = list(
            list(block = 1, prior = list("pc.prec", c(4.0, 0.01)))),
        control = ctrl)))
    # Regularizing both copy coefficients is not what regularizing the first
    # one does; the silent case was reporting the second as the first.
    expect_false(isTRUE(all.equal(every$weights, first$weights)))
    # Both read the same inner solves, so the difference is the fold alone.
    expect_equal(every$theta_grid, first$theta_grid)
})

test_that("a per-block prior naming an axis-free block is refused at the door", {
    skip_on_cran()
    fx <- hpb_two_copy_fixture()
    expect_error(
        suppressWarnings(suppressMessages(tulpa_nested_laplace_joint(
            responses = fx$responses, prior = fx$prior, copy = fx$copy,
            prior_alpha = list(
                list(block = 3, prior = list("pc.prec", c(4.0, 0.01)))),
            control = list(max_iter = 30L, tol = 1e-6,
                           integration = "grid")))),
        "blocks carrying one are 1, 2")
})
