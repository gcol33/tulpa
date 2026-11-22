
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

test_that("a node outside the declared domain sets the declaration aside", {
    # The convention `.nl_cell_partition()` takes on the reporting side: a
    # declaration the data contradicts is set aside rather than the data moved.
    g <- c(0.8, 0.95, 1.2)
    expect_identical(
        tulpa:::.hyper_axis_support(g, hyper_axis_spec("rho_car", g)),
        tulpa:::.hyper_default_coord_bounds(g))
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

skip_on_cran_budget()

test_that("a domain is keyed on the axis name without its block prefix", {
    skip_on_cran_budget()
    expect_identical(tulpa:::.axis_bare("b1.rho_car"), "rho_car")
    expect_identical(tulpa:::.axis_bare("b12.rho"), "rho")
    expect_identical(tulpa:::.axis_bare("rho_car"), "rho_car")
    expect_identical(tulpa:::.axis_bare("phi_occ"), "phi_occ")
    expect_identical(tulpa:::.axis_bare(NULL), "")

    g <- c(0.5, 0.8, 0.95, 0.99)
    plain  <- tulpa:::.hyper_axis_support(g, hyper_axis_spec("rho_car", g))
    prefixed <- tulpa:::.hyper_axis_support(g, hyper_axis_spec("b1.rho_car", g))
    expect_identical(plain, prefixed)
})

test_that("the proper-CAR default grid no longer closes outside its domain", {
    skip_on_cran_budget()
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

test_that("a block's own bounds tighten the domain the name declares", {
    skip_on_cran_budget()
    g <- c(0.2, 0.5, 0.8, 0.95, 0.99, 0.999)
    bare <- tulpa:::.hyper_axis_support(g, hyper_axis_spec("rho", g))
    bym2 <- tulpa:::.hyper_axis_support(
        g, hyper_axis_spec("rho", g, bounds = c(0, 1)))
    # The name alone fixes the upper end, which is what the default grid needs.
    expect_equal(bare, c(0.05, 0.9995))
    # Declared on (0, 1) the axis is a mixing weight, whose cells are laid on
    # its logit, as the within-cell read lays its boxes (gcol33/tulpa#1279):
    # half a logit step past each outer level, inside (0, 1) by construction.
    lg <- stats::qlogis(g); n <- length(g)
    expect_equal(bym2, stats::plogis(c(lg[1L] - (lg[2L] - lg[1L]) / 2,
                                       lg[n] + (lg[n] - lg[n - 1L]) / 2)))
    expect_true(all(bym2 > 0 & bym2 < 1))

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
    expect_equal(unit[1L], stats::plogis(1.5 * stats::qlogis(0.02) -
                                         0.5 * stats::qlogis(0.2)))
    expect_true(unit[1L] < min(g2))
})

test_that("a contradicted bound is set aside on its own side only", {
    skip_on_cran_budget()
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
    skip_on_cran_budget()
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

test_that("a block's own support closes the axis its name leaves open", {
    skip_on_cran_budget()
    # The name `rho` fixes the upper end only; the block that owns the axis
    # says which parameter it is (gcol33/tulpa#1188).
    expect_equal(tulpa:::.hp_axis_support("rho", list(type = "bym2")), c(0, 1))
    expect_equal(tulpa:::.hp_axis_support("rho", list(type = "ar1")), c(-1, 1))
    expect_equal(tulpa:::.hp_axis_support("rho", list(type = "hsgp_mo")),
                 c(-1, 1))
    expect_equal(tulpa:::.hp_axis_support(
        "rho", list(type = "car_proper", rho_bounds = c(-0.3, 1))), c(-0.3, 1))
    expect_null(tulpa:::.hp_axis_support("sigma", list(type = "bym2")))
    expect_null(tulpa:::.hp_axis_support("rho", list()))

    # The record every door folds carries it, under either hyperprior choice.
    for (hp in c("proper", "flat")) {
        r <- tulpa:::.hp_axis_prior("rho", list(type = "bym2"), hyperprior = hp)
        expect_equal(r$support, c(0, 1), info = hp)
    }

    # Nodes whose half-step mirror crosses zero: by name alone the lower outer
    # cell closes below zero, and stays there however many levels are laid
    # toward zero, so the outermost level's width never shrinks.
    lv <- c(0.0666, 0.247, 0.425, 0.603, 0.739, 0.875, 0.97)
    grid_of <- function(rho) {
        tg <- as.matrix(expand.grid(sigma = c(0.3, 0.6), rho = rho))
        tg
    }
    by_name <- tulpa:::.joint_axis_specs_from_grid(grid_of(lv))
    by_block <- tulpa:::.joint_axis_specs_from_grid(
        grid_of(lv), axis_support = list(rho = c(0, 1)))
    rho_spec <- function(specs) Filter(function(s) s$name == "rho", specs)[[1L]]
    expect_equal(rho_spec(by_name)$bounds, c(-Inf, 1))
    expect_equal(rho_spec(by_block)$bounds, c(0, 1))
    expect_lt(tulpa:::.hyper_axis_measure(lv, rho_spec(by_name))$edges[1L], 0)
    # Declared on (0, 1), the cells are laid on the logit: the outer edge is
    # half a logit step below the lowest level.
    mirror_lo <- function(x) {
        lx <- stats::qlogis(sort(x))
        stats::plogis(lx[1L] - (lx[2L] - lx[1L]) / 2)
    }
    expect_equal(tulpa:::.hyper_axis_measure(lv, rho_spec(by_block))$edges[1L],
                 mirror_lo(lv))

    # With the support declared, the outer cell follows a ladder of levels
    # in: it runs from half a logit step below the lowest level to the logit
    # midpoint with the next one, so its width falls as the ladder descends.
    ladders <- list(0.02, c(0.002, 0.02), c(0.0002, 0.002, 0.02))
    widths <- vapply(ladders, function(lo) {
        m <- tulpa:::.hyper_axis_measure(c(lo, lv), rho_spec(by_block),
                                         declared = lv)
        expect_lt(tulpa:::.hyper_axis_measure(
            c(lo, lv), rho_spec(by_name), declared = lv)$edges[1L], 0)
        expect_gt(m$edges[1L], 0)
        diff(m$edges[1:2])
    }, numeric(1))
    expect_true(all(diff(widths) < 0))
    expect_equal(widths, vapply(ladders, function(lo) {
        x <- sort(c(lo, lv))
        stats::plogis(mean(stats::qlogis(x[1:2]))) - mirror_lo(x)
    }, numeric(1)))

    # A prefixed column is keyed by its full name.
    tg <- grid_of(lv); colnames(tg) <- c("b1.sigma", "b1.rho")
    sp <- tulpa:::.joint_axis_specs_from_grid(
        tg, axis_support = list(b1.rho = c(0, 1)))
    expect_equal(Filter(function(s) s$name == "b1.rho", sp)[[1L]]$bounds,
                 c(0, 1))

    # A side the declared nodes contradict is set aside, the other one holds.
    expect_equal(tulpa:::.hyper_bounds_meet(c(-Inf, 1), c(0, 1), c(-0.4, 0.4)),
                 c(-Inf, 1))
    expect_equal(tulpa:::.hyper_bounds_meet(NULL, c(0, 1), c(0.2, 0.8)),
                 c(0, 1))
    expect_equal(tulpa:::.hyper_bounds_meet(c(0, Inf), NULL, 1), c(0, Inf))
})

test_that("a BYM2 fit whose rho posterior reaches zero closes its lower edge", {
    skip_on_cran()
    skip_if_fast()
    # The fit of gcol33/tulpa#1188: its placed rho nodes start at 0.067, the
    # posterior is flat toward zero, and the edge-closing rounds laid levels
    # down to rho = 0 because the outer cell kept its width.
    k <- 7L
    W <- matrix(0, k * k, k * k)
    id <- function(i, j) (i - 1L) * k + j
    for (i in seq_len(k)) for (j in seq_len(k)) {
        if (i < k) W[id(i, j), id(i + 1L, j)] <- W[id(i + 1L, j), id(i, j)] <- 1
        if (j < k) W[id(i, j), id(i, j + 1L)] <- W[id(i, j + 1L), id(i, j)] <- 1
    }
    set.seed(4)
    e <- eigen(diag(rowSums(W)) - W, symmetric = TRUE)
    keep <- e$values > 1e-8
    V <- e$vectors[, keep]
    scale_u <- exp(mean(log(diag(V %*% (t(V) / e$values[keep])))))
    u <- as.numeric(V %*% (rnorm(sum(keep)) / sqrt(e$values[keep])))
    phi <- 0.6 * (sqrt(0.5) * u / sqrt(scale_u) + sqrt(0.5) * rnorm(k * k))
    r <- rep(seq_len(k * k), each = 10L)
    x <- rnorm(length(r))
    d <- data.frame(y = rpois(length(r), exp(1 + 0.3 * x + phi[r])), x = x,
                    region = factor(r))
    fit <- tulpa(y ~ x + spatial(region), data = d, family = "poisson",
                 spatial = spatial_bym2(W, group_var = "region"),
                 mode = "nested_laplace",
                 control = list(n_threads = 2, progress = FALSE))
    rho <- sort(unique(tulpa_theta_matrix(fit)[, "b1.rho"]))
    expect_equal(fit$log_hyperprior_support[["b1.rho"]], c(0, 1))
    expect_gt(fit$axis_support[["b1.rho"]][1L], 0)
    expect_lt(fit$axis_support[["b1.rho"]][2L], 1)
    expect_gt(min(rho), 1e-4)
    expect_true(all(is.finite(coef(fit))))
})

test_that("the level weights are normalised over the same closed interval", {
    skip_on_cran_budget()
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
    skip_on_cran_budget()
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

skip_on_cran()
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
    skip_on_cran()
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
    # The values, pinned. `rho` names four different parameters across the
    # families and 1 is an open upper bound of all four, while their lower ends
    # disagree, so only the upper end is claimed by the name; a block that
    # knows which one it holds declares the tighter interval on its own spec.
    expect_equal(tulpa:::.hyper_axis_domain(list(name = "rho"))$bounds, c(-Inf, 1))
    expect_equal(tulpa:::.hyper_axis_domain(list(name = "rho_car"))$bounds, c(-Inf, 1))

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

test_that("no bounded axis reports a support outside its domain", {
    skip_on_cran()
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
        dom <- tulpa:::.hyper_axis_domain(list(name = axis))
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
