# Which families the outer-grid placement pass can move, and why each of the
# rest cannot (gcol33/tulpa#361).
#
# `.NL_REGISTRY_AXIS_FIELD` decides scope. Membership is not a hand list: a
# family belongs when the transform registry names a coordinate for EVERY axis
# of its grid and its grid fields bind to those axes one for one. These blocks
# hold the table to that rule from both sides, so a family added to the engine
# with a placeable grid and no entry fails here rather than being left out in
# silence -- which is the defect #361 was reopened on, one layer up.

# The axis names each registry family's `theta()` builder emits. mcar / miid are
# shown at p = 2 (the log-Cholesky coordinates of a 2 x 2 Sigma) and tgmrf at a
# two-axis user block; both are families whose grid is ONE matrix field, which
# is the point the scope check reads off them.
.reg_axis_names <- function() list(
    icar       = "tau",
    rw1        = "tau",
    rw2        = "tau",
    ar1        = c("tau", "rho"),
    iid        = "sigma",
    bym2       = c("sigma", "rho"),
    car_proper = c("tau", "rho"),
    nngp       = c("sigma2", "phi_gp"),
    hsgp       = c("sigma2", "lengthscale"),
    hsgp_mo    = c("sigma_1", "sigma_2", "rho", "ell"),
    spde       = c("range", "sigma"),
    mcar       = c("L11", "L21", "L22"),
    miid       = c("L11", "L21", "L22"),
    tgmrf      = c("a", "b"),
    lf         = character(0))

test_that("the name table matches what each registry theta() builder emits", {
    ax <- .reg_axis_names()
    expect_setequal(names(ax), names(.NL_REGISTRY))
    emit <- list(
        icar       = list(tau_grid = 1),
        rw1        = list(tau_grid = 1),
        rw2        = list(tau_grid = 1),
        ar1        = list(tau_grid = 1, rho_grid = 0.5),
        iid        = list(sigma_grid = 1),
        bym2       = list(sigma_grid = 1, rho_grid = 0.5),
        car_proper = list(tau_grid = 1, rho_grid = 0.5),
        nngp       = list(sigma2_grid = 1, phi_gp_grid = 1),
        hsgp       = list(sigma2_grid = 1, lengthscale_grid = 1),
        hsgp_mo    = list(sigma_1_grid = 1, sigma_2_grid = 1, rho_grid = 0.5,
                          lengthscale_grid = 1),
        spde       = list(range_grid = 1, sigma_grid = 1),
        lf         = list())
    for (fam in names(emit)) {
        nm <- .NL_REGISTRY[[fam]]$theta(emit[[fam]])$names
        expect_identical(as.character(nm), ax[[fam]], info = fam)
    }
    # The three matrix-field families name their axes from the matrix itself,
    # so the table above stands in for a shape rather than a fixed vector.
    g <- .mcar_default_logchol_grid(2L)
    expect_identical(colnames(g), ax$mcar)
    expect_identical(.NL_REGISTRY$mcar$theta(list(logchol_grid = g))$names,
                     ax$mcar)
    expect_identical(
        .NL_REGISTRY$tgmrf$theta(list(
            theta_grid_built = matrix(0, 1L, 2L,
                                      dimnames = list(NULL, c("a", "b"))),
            theta_names = c("a", "b")))$names,
        ax$tgmrf)
})

test_that("the movable-axis table names every axis of every family it lists", {
    ax <- .reg_axis_names()
    for (fam in names(.NL_REGISTRY_AXIS_FIELD)) {
        fields <- .NL_REGISTRY_AXIS_FIELD[[fam]]
        # Every axis of the family is listed, and nothing else: a passenger
        # axis re-crossed unchanged is what #361 was filed about.
        expect_setequal(unname(fields), ax[[fam]])
        # Every field is one `.NL_FAMILY_AXES` binds for that family, so the
        # provenance layer recognises the engine's own default coming back in
        # through a caller's prior (gcol33/tulpa#293).
        expect_true(all(names(fields) %in% names(.NL_FAMILY_AXES[[fam]])),
                    info = paste(fam, "lists a field the family never defaults"))
        # And the transform registry names a coordinate for each, which is what
        # makes the axis placeable at all.
        expect_false(anyNA(.joint_pareto_block_tags(fam, ax[[fam]])),
                     info = paste(fam, "has an unguessable axis"))
    }
})

test_that("every family absent from the table is absent for a stated reason", {
    ax <- .reg_axis_names()
    absent <- setdiff(names(.NL_REGISTRY), names(.NL_REGISTRY_AXIS_FIELD))
    expect_setequal(absent, c("car_proper", "ar1", "hsgp_mo", "mcar", "miid",
                              "tgmrf", "lf"))
    for (fam in absent) {
        a <- ax[[fam]]
        n_field <- length(.NL_FAMILY_AXES[[fam]] %||% list())
        tags <- if (length(a)) .joint_pareto_block_tags(fam, a) else character(0)
        # One of the three documented blocks must hold: an axis the transform
        # registry declines, a single grid field carrying several axes (so the
        # field-to-axis binding this table is built on does not describe the
        # family), or no outer axis at all.
        expect_true((length(a) > 0L && anyNA(tags)) ||
                        (n_field > 0L && n_field < length(a)) ||
                        length(a) == 0L,
                    info = paste(fam, "is out of scope with no stated reason"))
    }
    # The specific block each one carries, so a family moving between them is
    # visible rather than silently re-classified.
    expect_true(anyNA(.joint_pareto_block_tags("car_proper", ax$car_proper)))
    expect_true(anyNA(.joint_pareto_block_tags("ar1", ax$ar1)))
    expect_true(anyNA(.joint_pareto_block_tags("hsgp_mo", ax$hsgp_mo)))
    expect_false(anyNA(.joint_pareto_block_tags("mcar", ax$mcar)))
    expect_length(.NL_FAMILY_AXES$mcar, 1L)
    expect_length(ax$lf, 0L)
})

test_that(".nl_registry_write_theta writes each family's fields by column", {
    tm <- cbind(sigma2 = c(0.5, 1.5), phi_gp = c(0.2, 0.9))
    blk <- .nl_registry_write_theta(list(list(type = "nngp")), tm,
                                    colnames(tm))[[1L]]
    expect_identical(blk$sigma2_grid, c(0.5, 1.5))
    expect_identical(blk$phi_gp_grid, c(0.2, 0.9))

    # Block-prefixed spellings on a multi-block grid resolve to the same
    # fields, on the right block.
    tm2 <- cbind(b1.tau = c(1, 2), b2.sigma = c(0.3, 0.4))
    bl2 <- .nl_registry_write_theta(
        list(list(type = "rw1"), list(type = "iid")), tm2, colnames(tm2),
        multi = TRUE)
    expect_identical(bl2[[1L]]$tau_grid, c(1, 2))
    expect_identical(bl2[[2L]]$sigma_grid, c(0.3, 0.4))
    expect_null(bl2[[1L]]$sigma_grid)

    # A family with no entry is left untouched.
    tm3 <- cbind(tau = c(1, 2), rho = c(0.2, 0.4))
    bl3 <- .nl_registry_write_theta(list(list(type = "car_proper")), tm3,
                                    colnames(tm3))[[1L]]
    expect_null(bl3$tau_grid)
})

test_that("a newly-covered family recentres end to end and honours a pin", {
    skip_on_cran()
    set.seed(23)
    Tn <- 60L; per <- 8L
    idx <- rep(seq_len(Tn), each = per)
    eff <- cumsum(rnorm(Tn, 0, 0.35))
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] + rnorm(length(idx), 0, 0.7)
    fit_rw1 <- function(extra = list(), ctrl = list()) {
        suppressWarnings(tulpa_nested_laplace(
            y = y, n_trials = rep(1L, length(y)), X = X,
            prior = utils::modifyList(
                list(type = "rw1", temporal_idx = idx, n_times = Tn), extra),
            family = "gaussian", phi = 0.49,
            control = utils::modifyList(
                list(max_iter = 200L, tol = 1e-8, n_threads = 1L,
                     diagnose_k = FALSE, diagnose_skew = FALSE), ctrl)))
    }
    moved <- fit_rw1(ctrl = list(auto_recenter = "always"))
    expect_identical(moved$outer_grid_placement, "auto_recentered")
    expect_identical(moved$outer_grid_recenter_axes, "tau")
    expect_length(sort(unique(as.numeric(moved$theta_grid))),
                  .nl_recenter("n_pts"))

    # A caller's own nodes are a PIN: reported, never moved (gcol33/tulpa#293).
    pinned <- fit_rw1(list(tau_grid = c(0.3, 1, 3)),
                      list(auto_recenter = "always"))
    expect_identical(pinned$outer_grid_placement, "fixed")
    expect_identical(pinned$outer_grid_recenter_declined, "axis_pinned")
    expect_identical(sort(unique(as.numeric(pinned$theta_grid))), c(0.3, 1, 3))
})

test_that("an out-of-scope family names the axis that blocked it", {
    skip_on_cran()
    set.seed(24)
    Tn <- 60L; per <- 8L
    idx <- rep(seq_len(Tn), each = per)
    eff <- as.numeric(stats::arima.sim(list(ar = 0.8), Tn)) * 0.4
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] + rnorm(length(idx), 0, 0.7)
    fit <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X,
        prior = list(type = "ar1", temporal_idx = idx, n_times = Tn),
        family = "gaussian", phi = 0.49,
        control = list(max_iter = 200L, tol = 1e-8, n_threads = 1L,
                       diagnose_k = FALSE, diagnose_skew = FALSE,
                       auto_recenter = "always")))
    expect_identical(fit$outer_grid_placement, "fixed")
    expect_match(fit$outer_grid_recenter_declined, "^unguessable_axis: ")
    expect_match(fit$outer_grid_recenter_declined, "rho")
    # The rail REPORT is taken whether or not any rescue covers the family.
    expect_false(is.null(fit$outer_grid_railed_axes))
})

test_that("a multi-block registry fit always records a placement", {
    skip_on_cran()
    set.seed(25)
    G <- 40L; per <- 8L
    idx <- rep(seq_len(G), each = per)
    u <- rnorm(G, 0, 0.7)
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + u[idx] + rnorm(length(idx), 0, 0.7)
    fit_iid <- function(ctrl = list()) suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X,
        prior = list(list(type = "iid", obs_idx = idx, n_units = G)),
        family = "gaussian", phi = 0.49,
        control = utils::modifyList(
            list(max_iter = 200L, tol = 1e-8, n_threads = 1L, progress = FALSE,
                 diagnose_k = FALSE, diagnose_skew = FALSE), ctrl)))

    base <- fit_iid()
    expect_true(base$outer_grid_placement %in% c("fixed", "auto_recentered"))
    expect_false(is.null(base$outer_grid_railed_axes))

    always <- fit_iid(list(auto_recenter = "always"))
    expect_identical(always$outer_grid_placement, "auto_recentered")
    expect_identical(always$outer_grid_recenter_axes, "b1.sigma")
    expect_length(sort(unique(as.numeric(always$theta_grid))),
                  .nl_recenter("n_pts"))

    off <- fit_iid(list(auto_recenter = FALSE))
    expect_identical(off$outer_grid_placement, "fixed")
    expect_identical(off$outer_grid_recenter_declined, "auto_recenter_disabled")
    # FALSE holds the engine's own default axis exactly where it is.
    expect_identical(sort(unique(as.numeric(off$theta_grid))),
                     sort(.nl_grid_axis("field_sd")))
})
