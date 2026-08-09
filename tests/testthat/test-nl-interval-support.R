# A reported interval stays inside the quantity's own support
# (gcol33/tulpa#369), and a fit the placement rescue cannot help says so
# (gcol33/tulpa#370).
#
# gcol33/tulpa#353 gave the density read `outside = "extend"`: the extreme grid
# cell's mass reaches half a spacing past its coordinate, so the interval
# extends to a mirrored outer edge instead of clamping at the extreme node.
# gcol33/tulpa#358 made `sample` its own support kind, which clamps because half
# the gap between two draws is not a cell width. Neither gave the edge the
# quantity's DOMAIN, and the coordinate was guessed from the values -- log
# whenever they were all positive. A proportion is all positive, so a BYM2
# mixing weight got an edge above 1.

.supp_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn))
}

test_that("a cell edge is mirrored in the quantity's own coordinate", {
    v <- .nl_grid_axis("bym2_rho")           # 0.2 0.5 0.8 0.95 0.99 0.999

    # `unit`: both edges inside (0, 1), where the guessed log coordinate puts
    # the upper one above 1 for a quantity that cannot exceed it.
    e_unit <- .nl_cell_edges(v, "unit")
    expect_true(all(e_unit > 0 & e_unit < 1))
    expect_gt(.nl_cell_edges(v)[2L], 1)

    # `correlation`: the same statement on (-1, 1).
    r <- c(-0.8, -0.4, 0, 0.4, 0.9)
    expect_true(all(abs(.nl_cell_edges(r, "correlation")) < 1))
    expect_gt(.nl_cell_edges(r)[2L], 1)

    # `positive` IS the log mirroring the rule guessed for an all-positive
    # axis, so naming it changes nothing on a scale axis.
    s <- .nl_grid_axis("field_sd")
    expect_identical(.nl_cell_edges(s, "positive"), .nl_cell_edges(s))
    expect_gt(.nl_cell_edges(s, "positive")[1L], 0)

    # `unbounded` mirrors in the value itself: the support is the whole line, so
    # there is nothing for a log coordinate to protect.
    expect_identical(.nl_cell_edges(s, "unbounded"),
                     c(s[1L] - 0.5 * (s[2L] - s[1L]),
                       s[5L] + 0.5 * (s[5L] - s[4L])))

    # An axis whose support the registry will not name keeps the guess.
    expect_identical(.nl_cell_edges(v, NA_character_), .nl_cell_edges(v))

    # A node set that does not live where the caller says it does falls back
    # rather than returning a non-finite edge.
    expect_identical(.nl_cell_edges(c(0.5, 1.5, 2.5), "unit"),
                     .nl_cell_edges(c(0.5, 1.5, 2.5)))
})

test_that("the domain reaches the CDF read, not only the moment rule", {
    v <- c(0.2, 0.5, 0.8, 0.95)
    w <- c(0.001, 0.02, 0.35, 0.629)
    probs <- c(0.025, 0.5, 0.975)

    q_dom <- .nl_summary_quantile(v, w, probs, "unit", "density")
    q_na  <- .nl_summary_quantile(v, w, probs, NA_character_, "density")
    expect_lt(q_dom[3L], 1)
    expect_gt(q_na[3L], 1)
    # Only the outer half-cells move: the median sits between interior knots.
    expect_equal(q_dom[2L], q_na[2L], tolerance = 1e-12)

    # A `sample` support forms no edge at all, so the domain cannot change it.
    expect_identical(.nl_summary_quantile(v, w, probs, "unit", "sample"),
                     .nl_summary_quantile(v, w, probs, NA_character_, "sample"))
})

test_that("the domain reaches a box-uniform read's INTERIOR edges too", {
    # "Only the outer half-cells move" above is a property of the CHORD
    # construction, whose interior knots are the coordinates themselves and so
    # carry no coordinate choice. Box-uniform's knots are the cell EDGES, every
    # one of which is bisected in the domain's own coordinate, so the domain
    # reaches the whole read and not just its ends (gcol33/tulpa#357, #369).
    v <- c(0.2, 0.5, 0.8, 0.95)
    w <- c(0.001, 0.02, 0.35, 0.629)
    probs <- c(0.025, 0.5, 0.975)

    q_dom <- .nl_summary_quantile(v, w, probs, "unit", "density", "box_uniform")
    q_na  <- .nl_summary_quantile(v, w, probs, NA_character_, "density",
                                  "box_uniform")
    expect_lt(q_dom[3L], 1)
    expect_gt(q_na[3L], 1)
    expect_false(isTRUE(all.equal(q_dom[2L], q_na[2L])))
    # A logit-bisected interior edge is the one an inverse-logit-uniform cell
    # would have; the log guess places it elsewhere, which is the difference.
    e_dom <- .nl_box_edges(v, "unit")
    e_na  <- .nl_box_edges(v, NA_character_)
    expect_equal(e_dom[3L], stats::plogis(mean(stats::qlogis(v[2:3]))))
    expect_equal(e_na[3L], exp(mean(log(v[2:3]))))
    expect_true(all(e_dom > 0 & e_dom < 1))
})

test_that("declared bounds name the domain on the generic hyper-grid door", {
    expect_identical(.nl_domain_of_bounds(c(0, 1)), "unit")
    expect_identical(.nl_domain_of_bounds(c(-1, 1)), "correlation")
    expect_identical(.nl_domain_of_bounds(c(0, Inf)), "positive")
    expect_identical(.nl_domain_of_bounds(NULL, log_scale = TRUE), "positive")
    # An arbitrary finite interval is not one of the four domains, and inventing
    # a transform for it would be worse than the edge rule's own guess.
    expect_true(is.na(.nl_domain_of_bounds(c(0.3, 30))))
    expect_true(is.na(.nl_domain_of_bounds(NULL)))
})

test_that("a BYM2 fit reports a mixing weight inside (0, 1)", {
    skip_on_cran()
    S <- 100L
    set.seed(505L + S + 5000L)
    eff <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4)), scale = FALSE)) +
           rnorm(S, 0, 0.2)
    set.seed(505L + S)
    idx <- rep(seq_len(S), each = 10L)
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] +
         rnorm(length(idx), 0, sqrt(0.5))
    prior <- c(list(type = "bym2", n_spatial_units = S, spatial_idx = idx,
                    scale_factor = 1), .supp_chain_adj(S))
    ctrl <- list(max_iter = 200L, tol = 1e-9, n_threads = 1L,
                 diagnose_k = FALSE, diagnose_skew = FALSE)

    for (recenter in c(FALSE, TRUE)) {
        fit <- suppressWarnings(tulpa_nested_laplace(
            y = y, n_trials = rep(1L, length(y)), X = X, prior = prior,
            family = "gaussian", phi = sqrt(0.5),
            control = c(ctrl, list(auto_recenter = recenter))))
        rho_hi <- as.numeric(fit$theta_ci_hi[["rho"]])
        rho_lo <- as.numeric(fit$theta_ci_lo[["rho"]])
        expect_true(rho_hi < 1 && rho_lo > 0,
                    info = paste("auto_recenter =", recenter))
        # `sigma` is a positive scale: its interval is unchanged by naming the
        # domain, and stays positive.
        expect_gt(as.numeric(fit$theta_ci_lo[["sigma"]]), 0)
    }
})

test_that("a family the placement rescue covers no axis of says which axis blocks it", {
    skip_on_cran()
    # proper CAR: `rho_car`'s support is the adjacency eigenvalue interval, which
    # the transform registry will not guess, so no axis of the fit can be placed
    # -- including `tau`, an ordinary positive scale. Before gcol33/tulpa#370 the
    # rescue returned before stamping anything and the fit was indistinguishable
    # from one the rescue never applied to.
    S <- 50L
    set.seed(707L + S + 5000L)
    eff <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4)), scale = FALSE))
    set.seed(707L + S)
    idx <- rep(seq_len(S), each = 10L)
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] +
         rnorm(length(idx), 0, sqrt(0.5))
    fit <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X,
        prior = c(list(type = "car_proper", n_spatial_units = S,
                       spatial_idx = idx), .supp_chain_adj(S)),
        family = "gaussian", phi = sqrt(0.5),
        control = list(max_iter = 200L, tol = 1e-9, n_threads = 1L,
                       diagnose_k = FALSE, diagnose_skew = FALSE)))

    expect_identical(fit$outer_grid_placement, "fixed")
    expect_match(fit$outer_grid_recenter_declined, "^unguessable_axis: ")
    expect_match(fit$outer_grid_recenter_declined, "rho")
    # The rail REPORT needs neither curvature nor a scope entry, so it exists
    # even here.
    expect_false(is.null(fit$outer_grid_railed_axes))
})

test_that("the placement record is read back, not merely stored", {
    skip_on_cran()
    # gcol33/tulpa#346's lesson: a record nothing consumes is not a report. The
    # railed axes and the decline reason ride the same diagnostics surface the
    # #276 regime does.
    pl <- .tulpa_grid_placement(list(
        outer_grid_placement = "fixed",
        outer_grid_railed_axes = "rho:upper",
        outer_grid_recenter_declined = "axis_pinned"))
    expect_identical(pl$railed, "rho:upper")
    note <- .tulpa_grid_placement_note(pl)
    expect_match(note, "rho:upper")
    expect_match(note, "axis_pinned")

    moved <- .tulpa_grid_placement(list(
        outer_grid_placement = "auto_recentered",
        outer_grid_railed_axes = character(0),
        outer_grid_recenter_axes = "rho"))
    expect_match(.tulpa_grid_placement_note(moved), "re-centred on rho")

    # A grid whose every axis brackets its own mode needs no sentence.
    quiet <- .tulpa_grid_placement(list(outer_grid_placement = "fixed",
                                        outer_grid_railed_axes = character(0)))
    expect_null(.tulpa_grid_placement_note(quiet))
    expect_null(.tulpa_grid_placement(list()))
})
