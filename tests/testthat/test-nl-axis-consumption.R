# Axis consumption (gcol33/tulpa#352): a grid field the resolved path cannot
# read is refused when it is a pin and recorded when it is an engine default,
# never dropped in silence.
#
# Four things are tested, in order of what they protect:
#   1. THE TABLE -- `.NL_PATH_AXES` is well-formed and every token resolves, so
#      a family whose drivers disagree cannot be half-registered.
#   2. THE CONVERSION -- the one relation `.NL_AXIS_EQUIV` offers (icar's
#      `tau = 1 / sigma^2`) is the relation the engine itself applies, measured
#      against a fit rather than asserted from the source.
#   3. THE REFUSAL -- a pinned unread axis errors at the front door, naming the
#      field, the block, the path and the axis that path integrates.
#   4. THE RECORD -- a defaulted unread axis is dropped and the fit says so.

.axc_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

.axc_sim <- function(seed = 4L, n_s = 14L, N = 120L) {
    set.seed(seed)
    adj <- .axc_chain_adj(n_s)
    sidx <- sample.int(n_s, N, replace = TRUE)
    phi <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.5))))
    X <- cbind(1, rnorm(N))
    y <- rbinom(N, 1L, plogis(as.numeric(X %*% c(-0.2, 0.5)) + phi[sidx]))
    list(adj = adj, sidx = sidx, X = X, y = y, N = N, n_s = n_s)
}

.axc_icar_block <- function(sim, ...) {
    c(list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
           adj_row_ptr = sim$adj$adj_row_ptr,
           adj_col_idx = sim$adj$adj_col_idx,
           n_neighbors = sim$adj$n_neighbors, ...),
      list())
}

.axc_arm <- function(sim) {
    list(y = as.numeric(sim$y), n_trials = rep(1L, sim$N), X = sim$X,
         spatial_idx = as.integer(sim$sidx), re_idx = rep(0, sim$N),
         n_re_groups = 0L, sigma_re = 1.0, family = "binomial", phi = 1.0)
}

# --------------------------------------------------------------------------- #
# (1) The table                                                                #
# --------------------------------------------------------------------------- #

test_that("every path binding resolves to real fields of real families", {
    known <- tulpa:::.nl_known_axis_fields()
    for (path in names(tulpa:::.NL_PATH_AXES)) {
        expect_true(path %in% names(tulpa:::.NL_AXIS_PATH_LABEL), info = path)
        entries <- tulpa:::.NL_PATH_AXES[[path]]
        for (fam in names(entries)) {
            expect_true(fam %in% names(tulpa:::.NL_REGISTRY),
                        info = paste(path, fam))
            for (tok in entries[[fam]]) {
                if (startsWith(tok, ".")) {
                    expect_true(identical(tok, ".registry") ||
                                    tok %in% names(tulpa:::.NL_FAMILY_AXES),
                                info = paste(path, fam, tok))
                } else {
                    expect_true(tok %in% known, info = paste(path, fam, tok))
                }
            }
            # The expansion is non-empty and made of known field names.
            got <- tulpa:::.nl_path_axis_fields(fam, path)
            expect_gt(length(got), 0L)
            expect_true(all(got %in% known), info = paste(path, fam))
        }
    }
})

test_that("a family with no path entry reads exactly its registry binding", {
    # nngp's drivers agree, so it needs no entry and must not acquire one by
    # accident.
    for (path in names(tulpa:::.NL_PATH_AXES)) {
        expect_identical(tulpa:::.nl_path_axis_fields("nngp", path),
                         names(tulpa:::.NL_FAMILY_AXES$nngp), info = path)
    }
    # And the deviating families deviate in the documented direction.
    expect_identical(tulpa:::.nl_path_axis_fields("icar", "registry"), "tau_grid")
    expect_identical(tulpa:::.nl_path_axis_fields("icar", "joint_single"),
                     "sigma_grid")
    expect_setequal(tulpa:::.nl_path_axis_fields("icar", "copy"),
                    c("sigma_grid", "alpha_grid"))
})

test_that("every declared conversion is mutual and lands in the other axis", {
    for (fam in names(tulpa:::.NL_AXIS_EQUIV)) {
        eq <- tulpa:::.NL_AXIS_EQUIV[[fam]]
        for (from in names(eq)) {
            for (to in names(eq[[from]])) {
                # The pair is named the other way round too, so a caller
                # arriving from either path gets the sentence.
                expect_true(from %in% names(eq[[to]]),
                            info = paste(fam, from, to))
                # And the target IS a field some path of that family reads.
                reads <- unlist(lapply(names(tulpa:::.NL_AXIS_PATH_LABEL),
                                       function(p)
                                           tulpa:::.nl_path_axis_fields(fam, p)))
                expect_true(to %in% reads, info = paste(fam, to))
            }
        }
    }
})

# --------------------------------------------------------------------------- #
# (2) The conversion, measured                                                 #
# --------------------------------------------------------------------------- #

test_that("icar tau = 1 / sigma^2 is the engine's own conversion", {
    skip_on_cran()
    # The single-block joint areal backend hands its kernel `tau = 1 / sigma^2`
    # from the `sigma` axis it reads; the multi-block driver hands the SAME
    # kernel the registry's `tau` axis directly. So a one-block multi fit at
    # `tau_grid = 1 / s^2` must reproduce the single-block fit at
    # `sigma_grid = s` cell by cell. This is what makes rewriting a fixture's
    # `sigma_grid` into `tau_grid` preserve the grid it meant to pin.
    sim <- .axc_sim(seed = 21L)
    s <- c(0.4, 0.7, 1.3)
    arm <- .axc_arm(sim)

    fit_sd <- tulpa_nested_laplace_joint(
        responses = list(occ = arm),
        prior = .axc_icar_block(sim, sigma_grid = s),
        control = list(diagnose_k = FALSE))

    fit_prec <- tulpa_nested_laplace_joint(
        responses = list(occ = arm),
        prior = list(.axc_icar_block(sim, tau_grid = 1 / s^2,
                                     spatial_idx = list(sim$sidx))),
        control = list(diagnose_k = FALSE))

    expect_length(fit_prec$log_marginal, length(s))
    expect_lt(max(abs(as.numeric(fit_sd$log_marginal) -
                          as.numeric(fit_prec$log_marginal))), 1e-8)
    expect_lt(max(abs(as.numeric(fit_sd$weights) -
                          as.numeric(fit_prec$weights))), 1e-10)
})

# --------------------------------------------------------------------------- #
# (3) The refusal                                                              #
# --------------------------------------------------------------------------- #

test_that("a pinned axis the multi-block path cannot read is refused", {
    sim <- .axc_sim(seed = 22L)
    arm <- .axc_arm(sim)
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm),
            prior = list(.axc_icar_block(sim, sigma_grid = c(0.4, 0.9),
                                         spatial_idx = list(sim$sidx)))),
        "sigma_grid.*is not an axis.*registry path.*integrates `tau_grid`")
    # The message carries the conversion, the block index, and the family.
    msg <- tryCatch(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm),
            prior = list(.axc_icar_block(sim, spatial_idx = list(sim$sidx)),
                         .axc_icar_block(sim, sigma_grid = c(0.4, 0.9),
                                         spatial_idx = list(sim$sidx)))),
        error = conditionMessage)
    expect_match(msg, "prior block 2 'icar'", fixed = TRUE)
    expect_match(msg, "tau_grid = 1 / sigma_grid^2", fixed = TRUE)
})

test_that("the registry front door refuses the same axis", {
    sim <- .axc_sim(seed = 23L)
    expect_error(
        tulpa_nested_laplace(
            y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
            prior = .axc_icar_block(sim, spatial_idx = sim$sidx,
                                    sigma_grid = c(0.4, 0.9)),
            family = "binomial"),
        "`sigma_grid` is not an axis", fixed = TRUE)
})

test_that("the refusal is symmetric: a precision axis on the joint areal path", {
    sim <- .axc_sim(seed = 24L)
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(occ = .axc_arm(sim)),
            prior = .axc_icar_block(sim, tau_grid = c(1, 4, 9))),
        "tau_grid.*single-block joint areal backend.*sigma_grid = 1 / sqrt")
})

test_that("an axis the path DOES read passes untouched", {
    sim <- .axc_sim(seed = 25L)
    expect_null(tulpa:::.nl_check_axis_fields(
        .axc_icar_block(sim, tau_grid = c(1, 4)), "registry"))
    expect_null(tulpa:::.nl_check_axis_fields(
        .axc_icar_block(sim, sigma_grid = c(0.4, 0.9)), "joint"))
    # A copy block leads with (sigma, alpha), so sigma is read there.
    expect_length(tulpa:::.nl_check_block_axis_fields(
        .axc_icar_block(sim, sigma_grid = c(0.4, 0.9)), "copy", 1L), 0L)
})

test_that("a field no binding names is not this check's business", {
    sim <- .axc_sim(seed = 26L)
    blk <- .axc_icar_block(sim, tau_grid = c(1, 4))
    blk$some_other_grid <- c(1, 2, 3)
    expect_null(tulpa:::.nl_check_axis_fields(blk, "registry"))
})

# --------------------------------------------------------------------------- #
# (4) The record                                                               #
# --------------------------------------------------------------------------- #

test_that("an unread ENGINE DEFAULT is dropped with a reason, not refused", {
    skip_on_cran()
    sim <- .axc_sim(seed = 27L)
    fsd <- tulpa:::.nl_grid_axis("field_sd")
    fit <- tulpa_nested_laplace(
        y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
        prior = .axc_icar_block(sim, spatial_idx = sim$sidx, sigma_grid = fsd),
        family = "binomial", control = list(diagnose_k = FALSE))
    rec <- fit$axis_fields_dropped
    expect_s3_class(rec, "data.frame")
    expect_identical(nrow(rec), 1L)
    expect_identical(rec$field, "sigma_grid")
    expect_identical(rec$path, "registry")
    expect_identical(rec$integrates, "tau_grid")
    expect_true(nzchar(rec$reason))
    # The fit integrated the axis the path reads, at its own default size.
    expect_length(as.numeric(fit$theta_grid),
                  length(tulpa:::.nl_grid_axis("gmrf_tau")))
})

test_that("an auto_grid()-marked axis is dropped with a reason, not refused", {
    skip_on_cran()
    sim <- .axc_sim(seed = 28L)
    fit <- tulpa_nested_laplace(
        y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
        prior = .axc_icar_block(sim, spatial_idx = sim$sidx,
                                sigma_grid = auto_grid(c(0.4, 0.9))),
        family = "binomial", control = list(diagnose_k = FALSE))
    expect_identical(fit$axis_fields_dropped$field, "sigma_grid")
})

test_that("a fit with nothing dropped carries no record", {
    skip_on_cran()
    sim <- .axc_sim(seed = 29L)
    fit <- tulpa_nested_laplace(
        y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
        prior = .axc_icar_block(sim, spatial_idx = sim$sidx,
                                tau_grid = c(1, 4, 9)),
        family = "binomial", control = list(diagnose_k = FALSE))
    expect_null(fit$axis_fields_dropped)
    # And the record does not leak past the fit that produced it.
    expect_null(getOption("tulpa.nl_axis_dropped"))
})
