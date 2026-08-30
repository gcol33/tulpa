# Engine settings (R/settings.R): the single source of truth for every default
# the nested-Laplace machinery lays down on its own.
#
# Three things are tested, in order of what they protect:
#   1. INTEGRITY -- the table is well-formed and every binding resolves, so a
#      new family cannot be half-registered.
#   2. THE VALUES -- each default axis is pinned to its exact nodes. These are
#      the numbers the engine's behaviour rests on; a change to one is a change
#      to every fit that does not pin that axis, so it has to be deliberate
#      enough to update a test.
#   3. NO RE-SCATTERING -- a source-level check that a default axis is not
#      written inline again somewhere else, which is the condition the table
#      exists to remove (gcol33/tulpa#293 turned on the engine being able to
#      recognise its OWN default, which is impossible while five files each
#      carry a copy of it).
# Runs at every tier: no fit, pure table and source inspection.

test_that("every grid axis materialises to a usable vector", {
    for (key in names(tulpa:::.NL_GRID)) {
        spec <- tulpa:::.NL_GRID[[key]]
        if (isTRUE(spec$data_dependent)) {
            # Refuses materialisation rather than silently returning something
            # shaped like an axis.
            expect_error(tulpa:::.nl_grid_axis(key), "data-dependent",
                         info = key)
            expect_true(!is.null(spec$n) || !is.null(spec$lo_mult), info = key)
            next
        }
        ax <- tulpa:::.nl_grid_axis(key)
        expect_true(is.numeric(ax), info = key)
        expect_gt(length(ax), 1L)
        expect_true(all(is.finite(ax)), info = key)
        expect_false(anyDuplicated(ax) > 0L, info = key)
        expect_identical(ax, sort(ax), info = key)
    }
    expect_error(tulpa:::.nl_grid_axis("no_such_axis"), "Unknown default grid axis")
})

test_that("every family binding resolves to a declared axis", {
    for (fam in names(tulpa:::.NL_FAMILY_AXES)) {
        bind <- tulpa:::.NL_FAMILY_AXES[[fam]]
        expect_true(length(bind) > 0L, info = fam)
        expect_true(all(nzchar(names(bind))), info = fam)
        for (field in names(bind)) {
            key <- bind[[field]]
            expect_true(key %in% names(tulpa:::.NL_GRID),
                        info = paste(fam, field, key))
        }
    }
    # Every registry family is bound, except the ones that genuinely default no
    # axis (lf carries no outer axes at all).
    unbound <- setdiff(names(tulpa:::.NL_REGISTRY),
                       names(tulpa:::.NL_FAMILY_AXES))
    expect_identical(unbound, "lf")
    # And every binding names a real family, or is a documented path
    # pseudo-type (leading dot).
    extra <- setdiff(names(tulpa:::.NL_FAMILY_AXES),
                     names(tulpa:::.NL_REGISTRY))
    expect_true(all(startsWith(extra, ".")), info = paste(extra, collapse = ", "))
})

test_that("the default axes are exactly these values", {
    # Pinned deliberately: these numbers ARE the engine's behaviour on every
    # fit that does not name its own grid.
    expect_equal(tulpa:::.nl_grid_axis("field_sd"),
                 exp(seq(log(0.1), log(3), length.out = 5)))
    expect_equal(tulpa:::.nl_grid_axis("gmrf_tau"),
                 exp(seq(log(0.3), log(30), length.out = 9)))
    expect_equal(tulpa:::.nl_grid_axis("car_tau"),
                 exp(seq(log(0.3), log(30), length.out = 5)))
    expect_equal(tulpa:::.nl_grid_axis("ar1_tau"),
                 exp(seq(log(0.5), log(20), length.out = 5)))
    expect_equal(tulpa:::.nl_grid_axis("gp_var"),
                 exp(seq(log(0.05), log(2), length.out = 5)))
    expect_equal(tulpa:::.nl_grid_axis("gp_lengthscale"),
                 exp(seq(log(0.05), log(1.5), length.out = 5)))
    expect_equal(tulpa:::.nl_grid_axis("mo_field_sd"),
                 exp(seq(log(0.3), log(1.5), length.out = 3)))
    expect_equal(tulpa:::.nl_grid_axis("mo_lengthscale"),
                 exp(seq(log(0.1), log(1.0), length.out = 3)))
    expect_equal(tulpa:::.nl_grid_axis("bym2_rho"),
                 c(0.2, 0.5, 0.8, 0.95, 0.99, 0.999))
    expect_equal(tulpa:::.nl_grid_axis("joint_car_rho"),
                 c(0.5, 0.8, 0.95, 0.99))
    expect_equal(tulpa:::.nl_grid_axis("ar1_rho"), c(0, 0.4, 0.7, 0.9, 0.97))
    expect_equal(tulpa:::.nl_grid_axis("mo_rho"), c(-0.4, 0, 0.4))
    expect_equal(tulpa:::.nl_grid_axis("mcar_sd"), c(0.4, 0.7, 1.1, 1.7))
    expect_equal(tulpa:::.nl_grid_axis("mcar_rho"),
                 c(-0.8, -0.4, 0, 0.4, 0.7, 0.9))

    # `prepend` puts the exact no-transfer node in the copy axis, at the front.
    alpha <- tulpa:::.nl_grid_axis("copy_alpha")
    expect_identical(alpha[1L], 0)
    expect_equal(alpha, c(0, exp(seq(log(0.1), log(3), length.out = 5))))

    # Shape parameters of the data-dependent axes.
    expect_identical(tulpa:::.nl_grid_par("car_rho", "n"), 5L)
    expect_equal(tulpa:::.nl_grid_par("car_rho", "margin"), 0.05)
    expect_equal(tulpa:::.nl_grid_par("car_rho", "bounds"), c(0, 1))
    expect_equal(tulpa:::.nl_grid_par("spde_registry", "span"), 1.4)
    expect_equal(tulpa:::.nl_grid_par("spde_direct", "lo_mult"), 0.3)
    expect_equal(tulpa:::.nl_grid_par("spde_direct", "hi_mult"), 3)
    expect_equal(tulpa:::.nl_grid_par("tgmrf_axis", "half_width"), 2)

    # Diagnostic / recenter / spatiotemporal defaults.
    expect_equal(tulpa:::.nl_diag("k_usable"), 0.7)
    expect_identical(tulpa:::.nl_diag("k_samples"), 200L)
    expect_equal(tulpa:::.nl_diag("gamma3_ok"), 0.5)
    expect_equal(tulpa:::.nl_diag("gamma3_unreliable"), 1.0)
    # The two skew-correction defaults, both decided on measurement and both
    # recorded here so a change to either is deliberate: the CENTRE band is off
    # (gcol33/tulpa#376) and the correction itself is ON (gcol33/tulpa#364).
    expect_identical(tulpa:::.nl_diag("centre_unreliable"), Inf)
    expect_true(tulpa:::.nl_diag("skew_correct"))
    # The within-cell default, decided on fixed-truth coverage at the placement
    # the engine ships (gcol33/tulpa#357), and the three places that name it
    # agreeing -- the setting, the `match.arg` vocabulary and the kind that
    # admits it.
    expect_identical(tulpa:::.nl_diag("within_cell"), "box_uniform")
    expect_identical(tulpa:::.NL_WITHIN_CELL[1L], "box_uniform")
    expect_true("box_uniform" %in% tulpa:::.NL_SUPPORT[["density"]]$within)
    expect_equal(tulpa:::.nl_diag("grid_resolved"), 1)
    expect_identical(tulpa:::.nl_recenter("n_pts"), 5L)
    expect_equal(tulpa:::.nl_recenter("span"), 2.5)
    expect_identical(tulpa:::.nl_recenter("max_attempts_joint"), 2L)
    expect_identical(tulpa:::.nl_recenter("max_attempts_registry"), 1L)
    # A multiple of what a flat marginal puts on one node, not a share of the
    # axis's weight (gcol33/tulpa#375). `2` is the retired 0.5 share at the four
    # nodes it was tuned on.
    expect_equal(tulpa:::.nl_recenter("edge_mass_mult"), 2)
    expect_error(tulpa:::.nl_recenter("edge_mass"), "Unknown recenter setting")
    expect_identical(tulpa:::.nl_st_default("n_spatial"), 4L)
    expect_equal(tulpa:::.nl_st_default("tau_upper"), 16)

    expect_error(tulpa:::.nl_diag("nope"), "Unknown diagnostic setting")
    expect_error(tulpa:::.nl_recenter("nope"), "Unknown recenter setting")
    expect_error(tulpa:::.nl_st_default("nope"), "Unknown spatiotemporal")
})

test_that("registry defaults() fill exactly the bound fields, crossed", {
    fill <- function(type, p = list()) {
        tulpa:::.NL_REGISTRY[[type]]$defaults(p, list())
    }

    # Single-axis family: the axis, unexpanded.
    expect_equal(fill("icar")$tau_grid, tulpa:::.nl_grid_axis("gmrf_tau"))
    expect_equal(fill("iid")$sigma_grid, tulpa:::.nl_grid_axis("field_sd"))

    # Two-axis family: the Cartesian product, in binding order, stored
    # pre-paired (one row of theta_grid per tuple).
    b <- fill("bym2")
    gr <- expand.grid(sigma = tulpa:::.nl_grid_axis("field_sd"),
                      rho   = tulpa:::.nl_grid_axis("bym2_rho"))
    expect_equal(b$sigma_grid, gr$sigma)
    expect_equal(b$rho_grid, gr$rho)
    expect_identical(length(b$sigma_grid), length(b$rho_grid))

    # Four-axis family: 3^4 cells.
    mo <- fill("hsgp_mo")
    expect_identical(length(mo$sigma_1_grid), 81L)
    expect_equal(sort(unique(mo$lengthscale_grid)),
                 tulpa:::.nl_grid_axis("mo_lengthscale"))

    # A supplied axis is honoured; the family's OTHER axes are then also taken
    # as supplied (a family's axes are pre-paired, so a partial fill would pair
    # unequal lengths).
    pinned <- fill("bym2", list(sigma_grid = c(1, 2), rho_grid = c(0.5, 0.5)))
    expect_equal(pinned$sigma_grid, c(1, 2))
    expect_equal(pinned$rho_grid, c(0.5, 0.5))

    # nngp / hsgp share the GP axes, so they default identically.
    expect_equal(fill("nngp")$sigma2_grid, fill("hsgp")$sigma2_grid)
    expect_equal(fill("nngp")$phi_gp_grid, fill("hsgp")$lengthscale_grid)

    # An unbound type is returned untouched rather than erroring.
    expect_identical(tulpa:::.nl_fill_family_axes(list(a = 1), "not_a_family"),
                     list(a = 1))
})

test_that("provenance narrows to the family that laid the axis", {
    pinned <- tulpa:::.nl_axis_is_pinned

    # Each family's OWN default is recognised on its own field.
    expect_false(pinned(list(type = "icar",
                             tau_grid = tulpa:::.nl_grid_axis("gmrf_tau")),
                        "tau_grid", type = "icar"))
    expect_false(pinned(list(type = "nngp",
                             sigma2_grid = tulpa:::.nl_grid_axis("gp_var")),
                        "sigma2_grid", type = "nngp"))
    expect_false(pinned(list(type = "ar1",
                             rho_grid = tulpa:::.nl_grid_axis("ar1_rho")),
                        "rho_grid", type = "ar1"))

    # The two PATH pseudo-types: the joint areal backends and every copy block
    # default the field-SD axis on `sigma_grid`, where the icar REGISTRY entry
    # defaults a precision axis instead. Narrowing to the wrong one here is what
    # would re-break gcol33/tulpa#293 for the occu_cover shape.
    fsd <- tulpa:::.nl_grid_axis("field_sd")
    expect_false(pinned(list(type = "icar", sigma_grid = fsd), "sigma_grid",
                        type = ".joint_areal"))
    expect_false(pinned(list(type = "rw1", sigma_grid = fsd), "sigma_grid",
                        type = ".copy"))

    # A grid that is some OTHER family's default axis is a pin, not a default.
    expect_true(pinned(list(type = "icar",
                            tau_grid = tulpa:::.nl_grid_axis("ar1_tau")),
                       "tau_grid", type = "icar"))
    # A real user grid is a pin.
    expect_true(pinned(list(type = "icar", tau_grid = c(1, 2, 3)), "tau_grid",
                       type = "icar"))
    # Absent is a default; marked is a default whatever the values.
    expect_false(pinned(list(type = "icar"), "tau_grid", type = "icar"))
    expect_false(pinned(list(type = "icar", tau_grid = auto_grid(c(1, 2, 3))),
                        "tau_grid", type = "icar"))

    # `type` is never inferred from the block. A joint areal block carries
    # `type = "icar"` while its `sigma_grid` default comes from `.joint_areal`;
    # inferring would read the engine's own default as a pin, which is the #293
    # failure one layer down.
    expect_false(pinned(list(type = "icar", sigma_grid = fsd), "sigma_grid"))

    # Without a type, every family's binding for that field is a candidate.
    expect_false(tulpa:::.nl_axis_matches_default(
        tulpa:::.nl_grid_axis("ar1_tau"), "tau_grid", type = "icar"))
    expect_true(tulpa:::.nl_axis_matches_default(
        tulpa:::.nl_grid_axis("ar1_tau"), "tau_grid"))

    # A data-dependent axis never matches, so such a grid is left alone.
    expect_false(tulpa:::.nl_axis_matches_default(seq(0.05, 0.95, length.out = 5),
                                                  "rho_grid", type = "car_proper"))
    # A field no family binds has no candidates.
    expect_false(tulpa:::.nl_axis_matches_default(c(1, 2), "not_a_grid_field"))
})

test_that("default grid axes are not written inline outside the settings file", {
    # The regression this file exists to prevent. A geometric axis over LITERAL
    # bounds is a default written where it is consumed; parameterised bounds
    # (`log(lo)`, `log(mode * span)`) are a caller building a data-dependent
    # axis from shape parameters the table holds, which is the sanctioned form.
    r_dir <- test_path("..", "..", "R")
    skip_if_not(dir.exists(r_dir), "package sources not available")
    files <- setdiff(list.files(r_dir, pattern = "\\.R$", full.names = TRUE),
                     file.path(r_dir, "settings.R"))

    offenders <- character(0)
    for (f in files) {
        src <- readLines(f, warn = FALSE)
        code <- src[!grepl("^\\s*#", src)]
        hits <- grep("exp\\(seq\\(log\\(\\s*[0-9]", code, value = TRUE)
        if (length(hits)) {
            offenders <- c(offenders, paste0(basename(f), ": ", trimws(hits)))
        }
    }
    expect_identical(offenders, character(0))
})

test_that("the reported Pareto-k threshold is read, never restated", {
    r_dir <- test_path("..", "..", "R")
    skip_if_not(dir.exists(r_dir), "package sources not available")
    files <- setdiff(list.files(r_dir, pattern = "\\.R$", full.names = TRUE),
                     file.path(r_dir, "settings.R"))

    # The literal 0.7 next to a Pareto-k / k-hat / k_threshold comparison. Bare
    # 0.7 elsewhere (a plotting alpha, a cex) is unrelated and not matched.
    pat <- paste0("(pareto_k|k_hat|khat|k_threshold|pk_i)\\s*(<|<=|>|>=|=)+\\s*0\\.7",
                  "|0\\.7\\s*(<|<=|>|>=)\\s*(pareto_k|k_hat|khat)")
    offenders <- character(0)
    for (f in files) {
        code <- readLines(f, warn = FALSE)
        code <- code[!grepl("^\\s*#", code)]
        hits <- grep(pat, code, value = TRUE)
        if (length(hits)) {
            offenders <- c(offenders, paste0(basename(f), ": ", trimws(hits)))
        }
    }
    expect_identical(offenders, character(0))

    # And the k_samples default likewise.
    offenders <- character(0)
    for (f in files) {
        code <- readLines(f, warn = FALSE)
        code <- code[!grepl("^\\s*#|^#'", code)]
        hits <- grep("k_samples\\s*(=|<-|%\\|\\|%)\\s*200", code, value = TRUE)
        if (length(hits)) {
            offenders <- c(offenders, paste0(basename(f), ": ", trimws(hits)))
        }
    }
    expect_identical(offenders, character(0))

    # The gamma_3 bands: a comparison of a skewness magnitude against the
    # literal band edges. `.tulpa_inner_skew_summary()` restated both after
    # `.tulpa_gamma3_band()` was already reading them.
    offenders <- character(0)
    for (f in files) {
        code <- readLines(f, warn = FALSE)
        code <- code[!grepl("^\\s*#|^#'", code)]
        hits <- grep("\\b(ag|abs_gamma3|gamma3|g)\\s*(<|<=|>|>=)\\s*(0\\.5|1\\.0)\\b",
                     code, value = TRUE)
        if (length(hits)) {
            offenders <- c(offenders, paste0(basename(f), ": ", trimws(hits)))
        }
    }
    expect_identical(offenders, character(0))
})

test_that("no user-facing text tells the caller to set a knob that hard-errors", {
    r_dir <- test_path("..", "..", "R")
    skip_if_not(dir.exists(r_dir), "package sources not available")

    # `control$diagnose_draws` was renamed to `control$k_samples` and the joint
    # front door hard-errors on the old name, so a message or a doc line telling
    # the reader to raise it sends them into that error. Two shipped for a while
    # -- the PSIS decline vocabulary and the tail-cap warning -- because the
    # rename swept the code and not the prose.
    #
    # The pattern is an IMPERATIVE aimed at the knob. Naming `diagnose_draws` as
    # a returned FIELD (`fit$diagnose_draws` is real) or quoting the old name in
    # the rename error itself is legitimate and must keep passing.
    dead <- c("diagnose_draws", "diagnose_cost")
    verbs <- "(raise|increase|lower|bump|set|use|pass|supply)"
    files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)

    offenders <- character(0)
    for (f in files) {
        code <- readLines(f, warn = FALSE)
        for (k in dead) {
            pat <- paste0(verbs, "\\s+\\W{0,3}(control\\$)?", k, "\\b")
            hits <- grep(pat, code, value = TRUE, ignore.case = TRUE)
            if (length(hits)) {
                offenders <- c(offenders, paste0(basename(f), ": ", trimws(hits)))
            }
        }
    }
    expect_identical(offenders, character(0))

    # And the knob the reader IS meant to reach exists on every front door that
    # reports the diagnostic, under one name, reached the same way.
    for (fn in c("tulpa_nested_laplace", "tulpa_nested_laplace_joint",
                 "tulpa_re_cov_nested", "fit_spde")) {
        expect_true("control" %in% names(formals(getExportedValue("tulpa", fn))),
                    info = fn)
    }
})
