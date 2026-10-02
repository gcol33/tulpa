# A caller-stated outer-grid axis is a bound (gcol33/tulpa#658).
#
# Refinability used to be decided by axis NAME -- `alpha` and every `phi_*` axis
# opted into the adaptive-grid and consistency passes whether or not the caller
# had written their nodes down -- so a fit could integrate a range the caller
# never asked for and report it nowhere. It is decided by PROVENANCE now: an
# axis whose nodes the caller stated is densified inside that range and never
# extended past its ends; one the engine placed may still be followed out.
#
# The contract asserted here:
#   * the per-axis mode reaches the spec field `.hyper_refinable_axes()` reads;
#   * a non-extendable axis's proposals stay inside its declared node span,
#     on both the boundary and the consistency path;
#   * provenance -- not the axis name -- is what picks the default;
#   * an engine-placed axis still extends exactly as it did;
#   * the reported span separates the half-node-step term from refinement.


# --------------------------------------------------------------------------- #
# 1. The spec field                                                            #
# --------------------------------------------------------------------------- #

test_that("hyper_axis_spec carries `extend`, defaulting to the historical answer", {
    s <- hyper_axis_spec("alpha", grid = c(0.2, 0.5), log_scale = TRUE,
                         refinable = TRUE)
    expect_true(s$extend)
    expect_true(tulpa:::.hyper_axis_may_extend(s))

    b <- hyper_axis_spec("alpha", grid = c(0.2, 0.5), log_scale = TRUE,
                         refinable = TRUE, extend = FALSE)
    expect_false(b$extend)
    expect_false(tulpa:::.hyper_axis_may_extend(b))

    # A spec built before the field existed says nothing about it and keeps the
    # historical answer, so no caller of the generic passes is silently bounded.
    legacy <- s
    legacy$extend <- NULL
    expect_true(tulpa:::.hyper_axis_may_extend(legacy))

    expect_error(hyper_axis_spec("alpha", grid = c(0.2, 0.5), extend = NA),
                 "extend")
    expect_error(hyper_axis_spec("alpha", grid = c(0.2, 0.5),
                                 extend = c(TRUE, FALSE)),
                 "extend")
})


test_that("the node limit is the declared span, and skips a log axis's atom", {
    b <- hyper_axis_spec("alpha", grid = c(0, 0.2, 0.35, 0.5), log_scale = TRUE,
                         bounds = c(0, Inf), refinable = TRUE, extend = FALSE,
                         atom_mass = 0.5)
    expect_equal(tulpa:::.hyper_axis_node_limit(b), c(0.2, 0.5))

    e <- hyper_axis_spec("alpha", grid = c(0, 0.2, 0.5), log_scale = TRUE,
                         bounds = c(0, Inf), refinable = TRUE, extend = TRUE,
                         atom_mass = 0.5)
    expect_null(tulpa:::.hyper_axis_node_limit(e))

    # One continuum node leaves no interior, so nothing may be proposed.
    one <- hyper_axis_spec("alpha", grid = c(0, 0.3), log_scale = TRUE,
                           bounds = c(0, Inf), refinable = TRUE, extend = FALSE,
                           atom_mass = 0.5)
    expect_length(tulpa:::.hyper_clip_to_node_limit(c(0.1, 0.3, 0.9), one), 0L)
})


# --------------------------------------------------------------------------- #
# 2. The per-axis mode reaches the field the refinement passes read            #
# --------------------------------------------------------------------------- #

test_that(".joint_axis_specs reads the resolved mode, not the axis name", {
    grids <- list(sigma = c(0.5, 1, 2),
                  alpha = c(0, 0.2, 0.5),
                  phi_y = c(0.5, 1, 2))
    cp <- list(has_copy = TRUE)

    specs <- tulpa:::.joint_axis_specs(
        grids, cp, axis_refine = c(alpha = "densify", phi_y = "extend"))
    by <- stats::setNames(specs, vapply(specs, `[[`, character(1), "name"))

    expect_true(by$alpha$refinable)
    expect_false(by$alpha$extend)
    expect_true(by$phi_y$refinable)
    expect_true(by$phi_y$extend)
    # An axis the override does not name keeps the driver's own eligibility.
    expect_true(by$sigma$refinable)

    # This is the field `.hyper_refinable_axes()` reads, which is what the issue
    # says was already in place and unreachable.
    expect_equal(tulpa:::.hyper_refinable_names(specs),
                 c("sigma", "alpha", "phi_y"))

    none <- tulpa:::.joint_axis_specs(
        grids, cp,
        axis_refine = c(sigma = "none", alpha = "none", phi_y = "none"))
    expect_length(tulpa:::.hyper_refinable_names(none), 0L)

    # No provenance supplied -- a spec list rebuilt from an assembled grid, which
    # no refinement pass reads -- keeps the driver's own eligibility.
    bare <- tulpa:::.joint_axis_specs(grids, cp)
    expect_equal(tulpa:::.hyper_refinable_names(bare),
                 c("sigma", "alpha", "phi_y"))
})


# --------------------------------------------------------------------------- #
# 3. Proposals on a bounded axis                                               #
# --------------------------------------------------------------------------- #

test_that("a bounded axis is densified inside its span and never past it", {
    lev <- c(0.2, 0.35, 0.5)
    mk <- function(extend) {
        hyper_axis_spec("alpha", grid = lev, log_scale = TRUE,
                        bounds = c(0, Inf), refinable = TRUE, extend = extend)
    }
    e <- mk(TRUE)
    b <- mk(FALSE)

    pts_e <- tulpa:::.hyper_propose_axis_extension(e, lev, "max")
    expect_gt(max(pts_e), 0.5)

    # The midpoint between the two outermost levels, and nothing else: on a log
    # axis that is the geometric mean.
    pts_b <- tulpa:::.hyper_propose_axis_extension(b, lev, "max")
    expect_equal(pts_b, sqrt(0.35 * 0.5))
    expect_true(all(pts_b > 0.35 & pts_b < 0.5))

    pts_lo <- tulpa:::.hyper_propose_axis_extension(b, lev, "min")
    expect_equal(pts_lo, sqrt(0.2 * 0.35))

    # The clip is on the proposal, not only on the caller's request, so an
    # explicit `extend_ok` cannot walk a bounded axis outward.
    forced <- tulpa:::.hyper_propose_axis_extension(b, lev, "max",
                                                    extend_ok = TRUE)
    expect_true(all(forced >= 0.2 & forced <= 0.5))

    # Interior densification is inside by construction, so the two agree there.
    di_e <- tulpa:::.hyper_propose_interior_densification(e, lev, 2L,
                                                          do_left = TRUE,
                                                          do_right = TRUE)
    di_b <- tulpa:::.hyper_propose_interior_densification(b, lev, 2L,
                                                          do_left = TRUE,
                                                          do_right = TRUE)
    expect_equal(di_b, di_e)
})


test_that("consistency points stay inside a bounded axis's span", {
    lev <- c(0.5, 1, 2)
    mk <- function(extend) {
        hyper_axis_spec("phi_y", grid = lev, log_scale = TRUE,
                        bounds = c(0, Inf), refinable = TRUE, extend = extend)
    }

    # The points bisect gaps between existing levels, so a bounded axis is
    # refined exactly as an extendable one is, and neither is widened.
    for (lm in list(c(-3, 0, -3), c(-1, 0, -0.5), c(-0.5, 0, -2))) {
        e <- tulpa:::.hyper_propose_mass_bisection(mk(TRUE), lev, lm)
        b <- tulpa:::.hyper_propose_mass_bisection(mk(FALSE), lev, lm)
        expect_gt(length(b), 0L)
        expect_equal(b, e)
        expect_true(all(b > 0.5 & b < 2))
    }
})


# --------------------------------------------------------------------------- #
# 4. Provenance, not the axis name, picks the default                          #
# --------------------------------------------------------------------------- #

test_that("a stated alpha axis is a bound and a placed one is not", {
    arm <- function(fc) {
        tulpa:::.normalise_arm_field_coef(list(field_coef = fc), 1L)
    }
    stated <- arm(list(name = "alpha", grid = c(0.2, 0.5)))
    marked <- arm(list(name = "alpha", grid = auto_grid(c(0.2, 0.5))))
    by_n   <- arm(list(name = "alpha", n = 9L))
    named  <- arm("alpha")
    deflt  <- arm(list(name = "alpha",
                       grid = tulpa:::.nl_grid_axis("copy_alpha")))

    expect_true(tulpa:::.joint_axis_is_stated("alpha", list(stated)))
    # `as.numeric()` drops the marker, so the normaliser has to record it.
    expect_true(marked$field_coef_axis$grid_auto)
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(marked)))
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(by_n)))
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(named)))
    # Nodes that ARE the engine's own axis carry nothing a statement would add.
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(deflt)))
    # No copy arm at all.
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(list())))
})


test_that("a phi axis is stated when the caller gave its nodes", {
    st <- tulpa:::.joint_axis_is_stated("phi_y", list(),
                                        phi_grid = list(y = c(0.5, 1, 2)),
                                        arm_names = c("y", "z"))
    expect_true(st)

    au <- tulpa:::.joint_axis_is_stated("phi_y", list(),
                                        phi_grid = list(y = auto_grid(c(0.5, 1, 2))),
                                        arm_names = c("y", "z"))
    expect_false(au)

    # Positional `phi_grid`, the other shape `.normalise_phi_grid()` accepts.
    pos <- tulpa:::.joint_axis_is_stated("phi_z", list(),
                                         phi_grid = list(NULL, c(0.5, 1, 2)),
                                         arm_names = c("y", "z"))
    expect_true(pos)

    # An arm with no axis of its own.
    expect_false(tulpa:::.joint_axis_is_stated("phi_z", list(),
                                               phi_grid = list(y = c(0.5, 1)),
                                               arm_names = c("y", "z")))
})


test_that(".joint_axis_refine_modes resolves every axis of one grid", {
    grids <- list(sigma = c(0.5, 1), alpha = c(0, 0.2, 0.5),
                  phi_y = c(0.5, 1, 2))
    cp <- list(has_copy = TRUE)
    arm_stated <- tulpa:::.normalise_arm_field_coef(
        list(field_coef = list(name = "alpha", grid = c(0.2, 0.5))), 1L)
    arm_placed <- tulpa:::.normalise_arm_field_coef(
        list(field_coef = list(name = "alpha", n = 9L)), 1L)

    m <- tulpa:::.joint_axis_refine_modes(grids, cp, list(arm_stated),
                                          phi_grid = list(y = c(0.5, 1, 2)),
                                          arm_names = "y")
    expect_equal(m[["alpha"]], "densify")
    expect_equal(m[["phi_y"]], "densify")
    # A field SD with no recorded provenance is the engine's own axis.
    expect_equal(m[["sigma"]], "extend")

    p <- tulpa:::.joint_axis_refine_modes(grids, cp, list(arm_placed),
                                          phi_grid = NULL, arm_names = "y")
    expect_equal(p[["alpha"]], "extend")
    # With no caller nodes recorded there is no statement to honour, so the
    # axis falls to the engine-placed default.
    expect_equal(p[["phi_y"]], "extend")

    u <- tulpa:::.joint_axis_refine_modes(grids, cp, list(arm_stated),
                                          phi_grid = list(y = c(0.5, 1, 2)),
                                          arm_names = "y",
                                          user = c(alpha = "extend",
                                                   phi_y = "none"))
    expect_equal(u[["alpha"]], "extend")
    expect_equal(u[["phi_y"]], "none")

    # A fit with no copy arm has no alpha axis to resolve.
    nc <- tulpa:::.joint_axis_refine_modes(grids, list(has_copy = FALSE),
                                           list(list()))
    expect_false("alpha" %in% names(nc))
})


test_that("a field SD's refinement mode follows who wrote its nodes", {
    grids <- list(sigma = c(0.5, 1, 2))
    cp <- list(has_copy = FALSE)
    mode <- function(prior, grid_auto = logical(0))
        tulpa:::.joint_axis_refine_modes(grids, cp, list(list()), prior = prior,
                                         grid_auto = grid_auto)[["sigma"]]
    pinned <- list(type = "icar", sigma_grid = c(0.3, 0.9, 2.7))
    expect_equal(mode(pinned), "densify")
    # Marked as a wrapper's default, the nodes carry no statement.
    expect_equal(mode(pinned, c(sigma_grid = TRUE)), "extend")
    # `place = FALSE` asks for the nodes as written, whoever made them.
    expect_equal(mode(pinned, c(sigma_grid = FALSE)), "densify")
    # The engine's own default axis is not a statement either.
    expect_equal(mode(list(type = "icar")), "extend")
})


# --------------------------------------------------------------------------- #
# 5. The control knob                                                          #
# --------------------------------------------------------------------------- #

test_that("control$axis_refine is validated against the fit's own axes", {
    axes <- c("sigma", "alpha", "phi_y", "rho_car")

    expect_null(tulpa:::.joint_check_axis_refine(NULL, axes))
    expect_equal(tulpa:::.joint_check_axis_refine(c(alpha = "none"), axes),
                 c(alpha = "none"))
    expect_equal(tulpa:::.joint_check_axis_refine(list(alpha = "densify"), axes),
                 c(alpha = "densify"))

    # One unnamed value applies to every refinable axis, and only to those.
    blanket <- tulpa:::.joint_check_axis_refine("none", axes)
    expect_equal(sort(names(blanket)), c("alpha", "phi_y", "sigma"))
    expect_true(all(blanket == "none"))

    expect_error(tulpa:::.joint_check_axis_refine(c(alpha = "widen"), axes),
                 "unknown mode")
    expect_error(tulpa:::.joint_check_axis_refine(c(alfa = "none"), axes),
                 "does not have")
    # Asking for nodes on an axis this driver never places any on is an error,
    # not a silent no-op that leaves the caller believing it took effect.
    expect_error(tulpa:::.joint_check_axis_refine(c(rho_car = "extend"), axes),
                 "only \"none\"")
    expect_equal(tulpa:::.joint_check_axis_refine(c(rho_car = "none"), axes),
                 c(rho_car = "none"))
    expect_equal(tulpa:::.joint_check_axis_refine(c(sigma = "densify"), axes),
                 c(sigma = "densify"))
    expect_error(tulpa:::.joint_check_axis_refine(c("none", "extend"), axes),
                 "named by axis")
})


# --------------------------------------------------------------------------- #
# 6. The reported span, and the two widening terms                             #
# --------------------------------------------------------------------------- #

test_that("axis_span separates the half-node-step term from refinement", {
    two <- c(0.2, 0.5)
    spec <- hyper_axis_spec("alpha", grid = two, log_scale = TRUE,
                            bounds = c(0, Inf), refinable = TRUE,
                            extend = FALSE)
    tg <- matrix(two, ncol = 1L, dimnames = list(NULL, "alpha"))
    sp <- tulpa:::.joint_axis_span(tg, tg, list(spec))[["alpha"]]

    expect_equal(sp$nodes, two)
    expect_equal(sp$refine, "densify")
    expect_equal(unname(sp$n_nodes), c(2L, 2L))
    expect_equal(sp$integrated, sp$declared)
    # For k equally log-spaced nodes the declared support is k / (k - 1) times
    # the node range on the axis's own coordinate: 2x at two nodes.
    expect_equal(diff(log(sp$declared)) / diff(log(sp$nodes)), 2)

    nine <- exp(seq(log(0.2), log(0.5), length.out = 9L))
    spec9 <- hyper_axis_spec("alpha", grid = nine, log_scale = TRUE,
                             bounds = c(0, Inf), refinable = TRUE,
                             extend = FALSE)
    tg9 <- matrix(nine, ncol = 1L, dimnames = list(NULL, "alpha"))
    sp9 <- tulpa:::.joint_axis_span(tg9, tg9, list(spec9))[["alpha"]]
    expect_equal(diff(log(sp9$declared)) / diff(log(sp9$nodes)), 9 / 8)

    # A grid refinement moved past the declared end: the second term is then
    # visible as `integrated` reaching past `declared`.
    ext <- hyper_axis_spec("alpha", grid = two, log_scale = TRUE,
                           bounds = c(0, Inf), refinable = TRUE, extend = TRUE)
    tgf <- matrix(c(0.08, 0.2, 0.5, 1.25), ncol = 1L,
                  dimnames = list(NULL, "alpha"))
    spe <- tulpa:::.joint_axis_span(tg, tgf, list(ext))[["alpha"]]
    expect_equal(spe$refine, "extend")
    expect_equal(spe$nodes, two)
    expect_gt(spe$integrated[2L], spe$declared[2L])
    expect_lt(spe$integrated[1L], spe$declared[1L])
    expect_equal(unname(spe$n_nodes), c(2L, 4L))

    # An axis with a single continuum level has no span to report, matching what
    # `.hyper_grid_supports()` leaves out.
    one <- hyper_axis_spec("alpha", grid = c(0, 0.3), log_scale = TRUE,
                           bounds = c(0, Inf), atom_mass = 0.5)
    tg1 <- matrix(c(0, 0.3), ncol = 1L, dimnames = list(NULL, "alpha"))
    expect_null(tulpa:::.joint_axis_span(tg1, tg1, list(one)))
})


# --------------------------------------------------------------------------- #
# 7. End to end: the fit integrates where the caller said                      #
# --------------------------------------------------------------------------- #

# The refinement fixture of test-nested-laplace-joint-adaptive-grid.R: a copy
# axis whose maximum (0.6) sits below the truth (2.0), so the boundary carries
# mass and the pass fires on every arm below.
.axr_sim <- function(seed = 6L, N = 600L, n_s = 50L) {
    set.seed(seed)
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_nb <- vapply(nbr, length, integer(1))
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    rw    <- cumsum(stats::rnorm(n_s, 0, 1 / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    Xocc  <- cbind(1, stats::rnorm(N))
    occur <- stats::rbinom(N, 1, stats::plogis(
        as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]))
    is_pos <- occur == 1L
    Xpos <- Xocc[is_pos, , drop = FALSE]
    spi  <- spatial_idx[is_pos]
    y_pos <- stats::rnorm(sum(is_pos),
                          as.numeric(Xpos %*% c(0.2, -0.4)) + 2.0 * phi_s[spi],
                          0.3)
    list(N = N, n_s = n_s, n_nb = n_nb, nbr = nbr,
         spatial_idx = as.integer(spatial_idx), Xocc = Xocc, occur = occur,
         Xpos = Xpos, y_pos = y_pos, spi = as.integer(spi))
}

.axr_fit <- function(sim, alpha_grid, control = list()) {
    ctrl <- utils::modifyList(
        list(adaptive_grid = TRUE, diagnose_k = FALSE,
             var_of_means_consistency = FALSE), control)
    tulpa_nested_laplace_joint(
        responses = list(
            occ = list(y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
                       X = sim$Xocc, spatial_idx = sim$spatial_idx,
                       re_idx = rep(0, sim$N), n_re_groups = 0L,
                       sigma_re = 1.0, family = "binomial", phi = 1.0),
            pos = list(y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
                       X = sim$Xpos, spatial_idx = sim$spi,
                       re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L,
                       sigma_re = 1.0, family = "gaussian", phi = 0.09,
                       field_coef = list(name = "alpha", grid = alpha_grid))),
        prior = list(type = "icar", n_spatial_units = sim$n_s,
                     adj_row_ptr = as.integer(c(0L, cumsum(sim$n_nb))),
                     adj_col_idx = as.integer(unlist(sim$nbr)) - 1L,
                     n_neighbors = as.integer(sim$n_nb),
                     sigma_grid = c(0.6, 1.0, 1.5)),
        control = ctrl)
}

test_that("a stated copy axis is densified, never extended, under adaptive_grid", {
    skip_on_cran()
    sim <- .axr_sim()
    stated <- c(0.2, 0.4, 0.6)

    fit <- .axr_fit(sim, stated)
    lev <- sort(unique(as.numeric(fit$theta_grid[, "alpha"])))

    # The pass still fires -- this is densify, not refusal -- and every node it
    # placed sits inside the range the caller wrote down.
    expect_false(is.null(fit$adaptive_grid_info))
    expect_gt(sum(fit$adaptive_grid_info$n_points_added), 0L)
    expect_gt(length(lev), length(stated))
    expect_equal(range(lev), range(stated))

    # The same fixture with the axis declared a default extends past 0.6, so the
    # difference is provenance and nothing else.
    fit_auto <- .axr_fit(sim, auto_grid(stated))
    expect_gt(max(fit_auto$theta_grid[, "alpha"]), max(stated) + 1e-6)

    # And the caller can restore that per axis without touching adaptive_grid.
    fit_ext <- .axr_fit(sim, stated,
                        control = list(axis_refine = c(alpha = "extend")))
    expect_gt(max(fit_ext$theta_grid[, "alpha"]), max(stated) + 1e-6)
})


test_that("axis_refine = 'none' keeps the copy axis out of refinement entirely", {
    skip_on_cran()
    sim <- .axr_sim()
    stated <- c(0.2, 0.4, 0.6)

    fit <- .axr_fit(sim, stated,
                    control = list(axis_refine = c(alpha = "none",
                                                   sigma = "none")))
    expect_null(fit$adaptive_grid_info)
    expect_equal(length(fit$log_marginal), 9L)   # 3 sigma x 3 alpha, untouched
    expect_equal(sort(unique(as.numeric(fit$theta_grid[, "alpha"]))), stated)
})


test_that("a fit reports the span it worked over on a stated axis", {
    skip_on_cran()
    sim <- .axr_sim()
    stated <- c(0.2, 0.4, 0.6)
    fit <- .axr_fit(sim, stated)

    sp <- fit$axis_span[["alpha"]]
    expect_false(is.null(sp))
    expect_equal(sp$refine, "densify")
    expect_equal(sp$nodes, range(stated))
    # The declared support is the stated nodes widened by half a node step at
    # each end -- the only widening term left once the axis cannot be extended.
    expect_lt(sp$declared[1L], min(stated))
    expect_gt(sp$declared[2L], max(stated))
    # Densification adds nodes inside the stated range, each carved out of the
    # base cells of its own row, so the measure integrates the declared span and
    # nothing more.
    expect_gt(sp$n_nodes[["final"]], sp$n_nodes[["initial"]])
    expect_identical(sp$integrated, sp$declared)
    # `axis_support` is the same interval, read off the same final grid.
    expect_equal(sp$integrated, fit$axis_support[["alpha"]])
})


test_that("a refinement slice carries the log marginal a tensor cell does", {
    # A slice cell is a point evaluation like any other, so its log marginal --
    # the kernel's marginal plus the hyperprior -- is the value a tensor holding
    # the same coordinates reads there. The slice batch holds `sigma` constant,
    # and reading the prior's axes off the batch dropped `sigma`'s density from
    # every slice (gcol33/tulpa#760).
    skip_on_cran()
    sim <- .axr_sim()
    fit <- .axr_fit(sim, c(0.2, 0.4, 0.6),
                    control = list(axis_refine = c(sigma = "none")))
    tag <- fit$refining_axis
    expect_true(any(nzchar(tag)))

    # The tensor is the reference, so it solves every cell; the fit's own
    # screen drops tensor cells it may, and never a slice cell.
    tensor <- .axr_fit(sim, sort(unique(as.numeric(fit$theta_grid[, "alpha"]))),
                       control = list(adaptive_grid = FALSE, prune = FALSE,
                                      axis_refine = c(alpha = "none",
                                                      sigma = "none")))
    expect_false(any(nzchar(tensor$refining_axis %||% "")))
    key <- function(g) sprintf("%.12g|%.12g", g[, "sigma"], g[, "alpha"])
    m <- match(key(fit$theta_grid), key(tensor$theta_grid))
    expect_false(anyNA(m))
    solved <- !cells_dropped(fit)
    expect_true(all(solved[nzchar(tag)]))
    expect_equal(fit$log_marginal[nzchar(tag)],
                 tensor$log_marginal[m][nzchar(tag)], tolerance = 1e-6)
    expect_equal(fit$log_marginal[!nzchar(tag) & solved],
                 tensor$log_marginal[m][!nzchar(tag) & solved], tolerance = 1e-6)
})


test_that("a field SD collapsed onto one node is resolved by the consistency pass", {
    # Informative data make the field SD's posterior narrower than the placed
    # grid's cell, so the whole axis sits on one node and its spread is a cell
    # box. The nodes are stated, so no placement moves them (an engine-placed
    # axis is laid at its measured SD instead). `axis_refine = "none"` is the
    # grid as stated; the default lets the consistency pass lay levels on the
    # axis until its marginal is resolved.
    skip_on_cran()
    set.seed(1)
    nr <- 12L
    n  <- nr * nr
    adj <- lapply(grid_neighbours(nr, nr), sort)
    nn  <- vapply(adj, length, integer(1))
    W <- matrix(0, n, n)
    for (i in seq_len(n)) W[i, adj[[i]]] <- 1
    ev <- eigen(diag(nn) - W, symmetric = TRUE)
    k  <- seq_len(n - 1L)
    u  <- as.numeric(ev$vectors[, k] %*% (rnorm(n - 1L) / sqrt(ev$values[k])))
    X1 <- cbind(1, rnorm(n))
    X2 <- cbind(1, rnorm(n))
    eta1 <- as.numeric(X1 %*% c(0.2, 0.5)) + 2 * u
    eta2 <- as.numeric(X2 %*% c(-0.3, 0.8)) + 2 * u
    responses <- list(
        occ = list(y = rbinom(n, 20L, plogis(eta1)), n_trials = rep(20L, n),
                   X = X1, spatial_idx = seq_len(n), family = "binomial"),
        cover = list(y = rnorm(n, eta2, 0.3), n_trials = rep(1L, n), X = X2,
                     spatial_idx = seq_len(n), family = "gaussian",
                     field_coef = list(name = "alpha",
                                       grid = seq(0.3, 1.7, length.out = 7))))
    prior <- list(type = "icar", n_spatial_units = n,
                  adj_row_ptr = c(0L, cumsum(nn)),
                  adj_col_idx = unlist(adj) - 1L, n_neighbors = nn,
                  sigma_grid = c(1, 2, 4))
    fit <- function(control)
        suppressWarnings(tulpa_nested_laplace_joint(
            responses, prior, phi_grid = list(cover = c(0.05, 0.09, 0.15)),
            control = c(list(diagnose_k = FALSE), control)))
    min_ess <- tulpa:::.nl_diag("axis_sd_ess")

    held <- fit(list(axis_refine = c(sigma = "none")))
    expect_lt(held$theta_sd_ess[["sigma"]], min_ess)

    resolved <- fit(list())
    expect_gte(resolved$theta_sd_ess[["sigma"]], min_ess)
    expect_gt(length(resolved$log_marginal), length(held$log_marginal))
})


# --------------------------------------------------------------------------- #
# 7. The multi-block driver runs the same passes                               #
# --------------------------------------------------------------------------- #

test_that("a multi-block axis is eligible by its bare name, a dispersion by its prefix", {
    elig <- tulpa:::.joint_axis_refine_eligible
    expect_true(all(vapply(c("b1.sigma", "b2.tau", "b2.alpha", "sigma", "tau",
                             "alpha", "phi_y"), elig, logical(1))))
    # `phi_gp` is a block's own lengthscale; only a bare `phi_<arm>` is a
    # dispersion.
    expect_false(elig("b1.phi_gp"))
    expect_false(any(vapply(c("b1.rho", "b2.rho_car", "rho"), elig, logical(1))))
})

test_that("a multi-block axis's refinement mode follows who wrote its nodes", {
    stated <- tulpa:::.joint_multi_axis_is_stated
    icar <- function(...) list(type = "icar", ...)
    copy <- list(arm = "y", block = 2L, alpha_grid = c(0.2, 0.5, 0.9))

    # A non-copy block lays its field SD on the registry's precision axis.
    expect_true(stated("b1.tau", list(icar(tau_grid = c(0.3, 1.1, 4))),
                       list(logical(0))))
    expect_false(stated("b1.tau", list(icar()), list(logical(0))))
    expect_false(stated("b1.tau", list(icar(tau_grid = c(0.3, 1.1, 4))),
                        list(c(tau_grid = TRUE))))
    # `place = FALSE` asks for the nodes as written.
    expect_true(stated("b1.tau", list(icar(tau_grid = c(0.3, 1.1, 4))),
                       list(c(tau_grid = FALSE))))

    # A copy block's field SD and copy scale follow the copy convention.
    blocks <- list(icar(), icar(sigma_grid = c(0.3, 0.9, 2.7)))
    expect_true(stated("b2.sigma", blocks, list(logical(0), logical(0)), copy,
                       copy_blocks = 2L))
    expect_false(stated("b2.sigma", blocks, list(logical(0), c(sigma_grid = TRUE)),
                        copy, copy_blocks = 2L))
    expect_true(stated("b2.alpha", blocks, list(), copy, copy_blocks = 2L))
    copy$alpha_grid <- auto_grid(copy$alpha_grid)
    expect_false(stated("b2.alpha", blocks, list(), copy, copy_blocks = 2L))
    expect_false(stated("b2.alpha", blocks, list(), list(arm = "y", block = 2L),
                        copy_blocks = 2L))

    # A dispersion reads the front door's record, as it does single-block.
    expect_true(stated("phi_y", blocks, list(), phi_grid = list(y = c(0.1, 0.2))))
    expect_false(stated("phi_y", blocks, list(), phi_grid = list(y = 0.1)))
})

.axr_multi_sim <- function(seed = 1L, nr = 12L) {
    set.seed(seed)
    n  <- nr * nr
    adj <- lapply(grid_neighbours(nr, nr), sort)
    nn  <- vapply(adj, length, integer(1))
    W <- matrix(0, n, n)
    for (i in seq_len(n)) W[i, adj[[i]]] <- 1
    ev <- eigen(diag(nn) - W, symmetric = TRUE)
    k  <- seq_len(n - 1L)
    u  <- as.numeric(ev$vectors[, k] %*% (rnorm(n - 1L) / sqrt(ev$values[k])))
    X1 <- cbind(1, rnorm(n))
    X2 <- cbind(1, rnorm(n))
    list(
        responses = list(
            occ = list(y = rbinom(n, 20L, plogis(as.numeric(X1 %*% c(0.2, 0.5)) +
                                                     2 * u)),
                       n_trials = rep(20L, n), X = X1, family = "binomial"),
            cover = list(y = rnorm(n, as.numeric(X2 %*% c(-0.3, 0.8)) + 2 * u, 0.3),
                         n_trials = rep(1L, n), X = X2, family = "gaussian")),
        block = list(type = "icar", n_spatial_units = n,
                     adj_row_ptr = c(0L, cumsum(nn)),
                     adj_col_idx = unlist(adj) - 1L, n_neighbors = nn,
                     spatial_idx = list(seq_len(n), seq_len(n))),
        copy = list(arm = "cover", block = 1L,
                    alpha_grid = seq(0.3, 1.7, length.out = 7)))
}

.axr_multi_fit <- function(sim, sigma_grid, control = list()) {
    prior <- list(c(sim$block, list(sigma_grid = sigma_grid)))
    suppressWarnings(tulpa_nested_laplace_joint(
        sim$responses, prior, copy = sim$copy,
        phi_grid = list(cover = c(0.05, 0.09, 0.15)),
        control = c(list(diagnose_k = FALSE), control)))
}

test_that("a multi-block field SD collapsed onto one node is resolved by the consistency pass", {
    skip_on_cran()
    sim <- .axr_multi_sim()
    min_ess <- tulpa:::.nl_diag("axis_sd_ess")

    # Stated nodes, which no placement moves.
    held <- .axr_multi_fit(sim, c(1, 2, 4),
                           list(axis_refine = c(b1.sigma = "none")))
    expect_lt(held$theta_sd_ess[["b1.sigma"]], min_ess)
    expect_false(any(grepl("b1.sigma", held$refining_axis, fixed = TRUE)))

    resolved <- .axr_multi_fit(sim, c(1, 2, 4))
    expect_gte(resolved$theta_sd_ess[["b1.sigma"]], min_ess)
    expect_gt(length(resolved$log_marginal), length(held$log_marginal))
    expect_true(any(grepl("b1.sigma", resolved$refining_axis, fixed = TRUE)))
    expect_false(is.null(resolved$var_of_means_consistency_info))
    # The measure and the weights carry the appended nodes: one entry per cell.
    n <- length(resolved$log_marginal)
    expect_equal(nrow(resolved$theta_grid), n)
    expect_length(resolved$weights, n)
    expect_length(resolved$log_quad, n)
    expect_equal(sum(resolved$weights, na.rm = TRUE), 1, tolerance = 1e-8)
    expect_length(resolved$modes[, 1L], n)
})

test_that("a multi-block axis_refine naming an axis the fit lacks is refused", {
    skip_on_cran()
    sim <- .axr_multi_sim()
    expect_error(
        .axr_multi_fit(sim, c(0.5, 1, 2), list(axis_refine = c(sigma = "none"))),
        "does not have")
    expect_error(
        .axr_multi_fit(sim, c(0.5, 1, 2), list(axis_refine = c(b1.rho = "none"))),
        "does not have")
})
