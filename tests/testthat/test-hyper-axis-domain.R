# The declared domain of a bounded outer axis, and the support rule that is
# closed inside it (gcol33/tulpa#657).
#
# An outer axis's cells are closed at their outermost node by an extrapolation
# half a node step further out. That step is a property of the node SPACING, so
# on a grid graded toward a boundary it steps past the boundary: the default
# proper-CAR nodes c(0.5, 0.8, 0.95, 0.99) closed at rho = 1.01, which is not a
# correlation, and `.hyper_axis_support()` is what a sampler asked to target the
# same measure takes as the flat prior's support.
#
# What is asserted here is the CONTRACT rather than the one arithmetic case: a
# bounded axis never reports a support outside its declared domain, for every
# entry of the domain registry and for node sets of 2, 3, 4 and many nodes.
#
# Tier 1 throughout: no fit, no sampler, pure table and arithmetic inspection.

test_that("every declared axis domain is a well-formed interval", {
    for (key in names(tulpa:::.NL_AXIS_DOMAIN)) {
        d <- tulpa:::.NL_AXIS_DOMAIN[[key]]
        expect_true(is.list(d), info = key)
        expect_identical(sort(names(d)), c("bounds", "open"), info = key)
        expect_true(is.numeric(d$bounds), info = key)
        expect_identical(length(d$bounds), 2L, info = key)
        expect_false(anyNA(d$bounds), info = key)
        expect_lt(d$bounds[1L], d$bounds[2L])
        expect_true(is.logical(d$open), info = key)
        expect_identical(length(d$open), 2L, info = key)
        expect_false(anyNA(d$open), info = key)
        # Every endpoint the engine declares today is the value at which the
        # parameterisation degenerates, so none of them is a member.
        expect_true(all(d$open), info = key)
    }
    expect_error(tulpa:::.nl_axis_domain_entry("no_such_axis"),
                 "Unknown axis domain")

    # The values, pinned. `rho` names four different parameters across the
    # families and 1 is an open upper bound of all four, while their lower ends
    # disagree, so only the upper end is claimed by the name; a block that
    # knows which one it holds declares the tighter interval on its own spec.
    expect_equal(tulpa:::.nl_axis_domain_entry("rho")$bounds, c(-Inf, 1))
    expect_equal(tulpa:::.nl_axis_domain_entry("rho_car")$bounds, c(-Inf, 1))
    expect_equal(tulpa:::.nl_axis_domain_entry(".positive")$bounds, c(0, Inf))

    # The positive family is read off the coordinate declaration rather than
    # from a second list of names.
    for (a in c("sigma", "sigma2", "tau", "range", "lengthscale", "phi_occ",
                "alpha")) {
        expect_true(isTRUE(tulpa:::.hyper_axis_scale(a)), info = a)
        dom <- tulpa:::.hyper_axis_domain(list(name = a))
        expect_equal(dom$bounds, c(0, Inf), info = a)
    }
    # And an axis nobody classifies carries no domain at all.
    expect_null(tulpa:::.hyper_axis_domain(list(name = "mcar_logchol_off")))
})

test_that("a domain is keyed on the axis name without its block prefix", {
    expect_identical(tulpa:::.hyper_axis_bare("b1.rho_car"), "rho_car")
    expect_identical(tulpa:::.hyper_axis_bare("b12.rho"), "rho")
    expect_identical(tulpa:::.hyper_axis_bare("rho_car"), "rho_car")
    expect_identical(tulpa:::.hyper_axis_bare("phi_occ"), "phi_occ")
    expect_identical(tulpa:::.hyper_axis_bare(NULL), "")

    g <- c(0.5, 0.8, 0.95, 0.99)
    plain  <- tulpa:::.hyper_axis_support(g, hyper_axis_spec("rho_car", g))
    prefixed <- tulpa:::.hyper_axis_support(g, hyper_axis_spec("b1.rho_car", g))
    expect_identical(plain, prefixed)
})

test_that("the proper-CAR default grid no longer closes outside its domain", {
    g <- tulpa:::.nl_grid_axis("joint_car_rho")
    expect_equal(g, c(0.5, 0.8, 0.95, 0.99))

    # The unclamped half-node-step rule is what put the upper edge at 1.01.
    expect_equal(tulpa:::.hyper_default_coord_bounds(g), c(0.35, 1.01))

    sup <- tulpa:::.hyper_axis_support(g, hyper_axis_spec("rho_car", g))
    # The boundary stands in for the next node, so the edge is the midpoint of
    # the outermost node and 1 -- inside the domain, and no further out than
    # the naive edge would have been.
    expect_equal(sup, c(0.35, 0.995))
    expect_lt(sup[2L], 1)
    expect_lt(sup[2L], 1.01)
    expect_gt(sup[2L], max(g))
})

test_that("no bounded axis reports a support outside its domain", {
    node_sets <- list(
        two    = c(0.90, 0.99),
        three  = c(0.80, 0.95, 0.99),
        four   = c(0.50, 0.80, 0.95, 0.99),
        many   = seq(0.10, 0.99, length.out = 20),
        signed = c(-0.9, -0.4, 0, 0.4, 0.9),
        bym2   = c(0.2, 0.5, 0.8, 0.95, 0.99, 0.999)
    )
    axes <- c("rho", "rho_car", "b1.rho", "b3.rho_car")
    for (axis in axes) {
        dom <- tulpa:::.nl_axis_domain_entry(tulpa:::.hyper_axis_bare(axis))
        for (nm in names(node_sets)) {
            g <- node_sets[[nm]]
            lab <- paste(axis, nm)
            sup <- tulpa:::.hyper_axis_support(g, hyper_axis_spec(axis, g))
            expect_false(is.null(sup), info = lab)
            expect_identical(length(sup), 2L, info = lab)
            expect_true(all(is.finite(sup)), info = lab)
            expect_lt(sup[1L], sup[2L])
            # Inside the domain, and still bracketing every node it closes.
            expect_lt(sup[2L], dom$bounds[2L])
            expect_true(sup[1L] <= min(g), info = lab)
            expect_true(sup[2L] >= max(g), info = lab)
        }
    }
})

test_that("a node set well inside the domain keeps the edges it had", {
    g <- c(0.1, 0.3, 0.5, 0.7)
    expect_identical(tulpa:::.hyper_axis_support(g, hyper_axis_spec("rho", g)),
                     tulpa:::.hyper_default_coord_bounds(g))

    # An axis the registry does not classify carries no domain at all.
    h <- c(-1.2, 0, 1.2)
    expect_identical(
        tulpa:::.hyper_axis_support(h, hyper_axis_spec("mcar_logchol_off", h)),
        tulpa:::.hyper_default_coord_bounds(h))
})

test_that("a block's own bounds tighten the domain the name declares", {
    g <- c(0.2, 0.5, 0.8, 0.95, 0.99, 0.999)
    bare <- tulpa:::.hyper_axis_support(g, hyper_axis_spec("rho", g))
    bym2 <- tulpa:::.hyper_axis_support(
        g, hyper_axis_spec("rho", g, bounds = c(0, 1)))
    # The name alone fixes the upper end, which is what the default grid needs.
    expect_equal(bare, c(0.05, 0.9995))
    expect_identical(bare, bym2)

    # The lower end is what only the block knows: `rho` names a mixing weight
    # on (0, 1) in one family and an autocorrelation reaching below zero in
    # another, so an undeclared axis is left alone there and a declared one is
    # closed inside its own interval.
    g2 <- c(0.02, 0.2, 0.5)
    open_below <- tulpa:::.hyper_axis_support(g2, hyper_axis_spec("rho", g2))
    unit <- tulpa:::.hyper_axis_support(
        g2, hyper_axis_spec("rho", g2, bounds = c(0, 1)))
    expect_lt(open_below[1L], 0)
    expect_gt(unit[1L], 0)
    expect_equal(unit[1L], 0.01)
    expect_true(unit[1L] < min(g2))
})

test_that("a node outside the declared domain sets the declaration aside", {
    # The convention `.nl_cell_partition()` takes on the reporting side: a
    # declaration the data contradicts is set aside rather than the data moved.
    g <- c(0.8, 0.95, 1.2)
    expect_identical(
        tulpa:::.hyper_axis_support(g, hyper_axis_spec("rho_car", g)),
        tulpa:::.hyper_default_coord_bounds(g))
})

test_that("a contradicted bound is set aside on its own side only", {
    # `hyper_axis_spec()` refuses a grid outside a spec's own `bounds`, so the
    # side-by-side decision is reached through the REGISTRY domain, which is a
    # claim about the axis name rather than a validated field. Driven directly
    # on the clamp, since a two-sided declared domain is what tells the two
    # readings apart and no engine axis declares one today.
    two_sided <- list(name = "rho", bounds = c(0, 1))
    u <- c(-0.4, 0, 0.4)
    bd <- tulpa:::.hyper_default_coord_bounds(u)
    expect_equal(bd, c(-0.6, 0.6))
    cl <- tulpa:::.hyper_domain_clamp(bd, u, two_sided)
    # The nodes contradict the lower bound, so it is set aside and the lower
    # edge is the naive one; the upper bound is not contradicted and holds.
    expect_equal(cl[1L], -0.6)
    expect_equal(cl[2L], 0.6)

    # And where the upper edge WOULD leave the uncontradicted side, it is the
    # one that moves while the contradicted side is still left alone.
    u2 <- c(-0.4, 0.9)
    bd2 <- tulpa:::.hyper_default_coord_bounds(u2)
    cl2 <- tulpa:::.hyper_domain_clamp(bd2, u2, two_sided)
    expect_equal(cl2[1L], bd2[1L])
    expect_lt(cl2[2L], 1)
    expect_equal(cl2[2L], 0.95)
})

test_that("an engine axis named rho accepts the nodes its families lay", {
    # `rho` names a BYM2 mixing weight on (0, 1) in one family and an AR1 or
    # multi-output cross-field correlation reaching below zero in two others,
    # so the support the NAME fixes is the upper end only. A spec builder
    # declaring (0, 1) by name alone refuses the multi-output default grid
    # `c(-0.4, 0, 0.4)` outright.
    expect_equal(tulpa:::.hyper_spec_bounds("rho"), c(-Inf, 1))
    expect_equal(tulpa:::.hyper_spec_bounds("rho_car"), c(-Inf, 1))
    expect_equal(tulpa:::.hyper_spec_bounds("sigma"), c(0, Inf))
    expect_equal(tulpa:::.hyper_spec_bounds("phi_occ"), c(0, Inf))
    expect_null(tulpa:::.hyper_spec_bounds("mcar_logchol_off"))

    for (nm in c("rho", "b1.rho")) {
        tg <- cbind(tau = rep(c(1, 2), each = 3), z = rep(c(-0.4, 0, 0.4), 2))
        colnames(tg) <- c(if (startsWith(nm, "b1.")) "b1.tau" else "tau", nm)
        specs <- tulpa:::.joint_axis_specs_from_grid(tg)
        expect_false(is.null(specs), info = nm)
        sup <- tulpa:::.hyper_grid_supports(tg, specs)
        expect_equal(sup[[nm]], c(-0.6, 0.6), info = nm)
        expect_lt(sup[[nm]][2L], 1)
        # And the quadrature weights are computable on the same grid.
        expect_true(all(is.finite(tulpa:::.nl_grid_log_quad(tg))), info = nm)
    }
})

test_that("a positive-scale axis keeps the support it had", {
    g <- tulpa:::.nl_grid_axis("field_sd")
    spec <- hyper_axis_spec("sigma", g, log_scale = TRUE, bounds = c(0, Inf))
    sup <- tulpa:::.hyper_axis_support(g, spec)
    # The log coordinate cannot reach 0, so the clamp is an exact no-op there.
    expect_identical(sup, exp(tulpa:::.hyper_default_coord_bounds(log(g))))
    expect_gt(sup[1L], 0)

    # A declared span already inside the domain is returned as declared.
    spec$slab_bounds <- exp(tulpa:::.hyper_default_coord_bounds(log(g)))
    expect_identical(tulpa:::.hyper_axis_support(g, spec), spec$slab_bounds)
})

test_that("the level weights are normalised over the same closed interval", {
    g <- c(0.5, 0.8, 0.95, 0.99)
    spec <- hyper_axis_spec("rho_car", g)
    w <- unname(tulpa:::.hyper_axis_level_weights(g, spec))
    sup <- tulpa:::.hyper_axis_support(g, spec)

    K <- length(g)
    edges <- c(sup[1L], (g[-K] + g[-1L]) / 2, sup[2L])
    expect_equal(w, diff(edges) / sum(diff(edges)))
    expect_equal(sum(w), 1)
    expect_true(all(w > 0))
    # The node next to the boundary owns only the part of its cell that fits
    # inside the domain: 0.025 wide, not the 0.04 the naive edge gave it.
    expect_equal(w[K], 0.025 / 0.645)
})

test_that("rho_car is declared on the coordinate its cells are measured on", {
    # Both correlation axes are integrated on the natural coordinate: their
    # domain is bounded, so the uniform measure on it is proper, and its
    # endpoints are models rather than an arbitrary origin.
    expect_false(tulpa:::.hyper_axis_scale("rho_car"))
    expect_false(tulpa:::.hyper_axis_scale("rho"))

    g <- tulpa:::.nl_grid_axis("joint_car_rho")
    spec <- hyper_axis_spec("rho_car", g)
    expect_identical(tulpa:::.hyper_axis_coord(g, spec), g)

    # Node placement is a quadrature choice, not the measure: the default nodes
    # are graded toward 1, and the cell widths carry that back so the weights
    # are those of the flat measure on the natural coordinate rather than the
    # near-equal weights an even reading of the same nodes would give.
    w <- unname(tulpa:::.hyper_axis_level_weights(g, spec))
    expect_gt(w[1L] / w[length(w)], 5)
    expect_true(all(diff(w) < 0))
})

test_that("the stored per-axis support is inside every axis's domain", {
    tg <- as.matrix(expand.grid(sigma = c(0.5, 1, 2),
                                rho_car = c(0.5, 0.8, 0.95, 0.99)))
    specs <- list(
        hyper_axis_spec("sigma", c(0.5, 1, 2), log_scale = TRUE,
                        bounds = c(0, Inf)),
        hyper_axis_spec("rho_car", c(0.5, 0.8, 0.95, 0.99))
    )
    sup <- tulpa:::.hyper_grid_supports(tg, specs)
    expect_named(sup, c("sigma", "rho_car"))
    expect_equal(sup$rho_car, c(0.35, 0.995))
    expect_lt(sup$rho_car[2L], 1)
    expect_gt(sup$sigma[1L], 0)
})

test_that("an axis with fewer than two continuum levels reports no support", {
    expect_null(tulpa:::.hyper_axis_support(0.9, hyper_axis_spec("rho", 0.9)))
})
