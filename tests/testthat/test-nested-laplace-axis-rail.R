# A default outer axis that does not contain its own posterior mode
# (gcol33/tulpa#361).
#
# `.NL_GRID` fixes every default span as a constant, so a fit whose posterior
# sits past one end integrates a tail at any spacing. The engine's rescue
# (gcol33/tulpa#290) is what moves such an axis, and #361 is the two places it
# could not reach: it recentred ONE axis per family and only in the `log`
# coordinate, so a BYM2 `rho` railed against its 0.95 ceiling was detected
# (`pareto_k_grid_edge_axes` names it) and then left there.
#
# Detection is per axis here, off the fit's own marginal -- the same marginal
# `.nl_axis_quantiles()` reports that axis's median and interval from -- rather
# than off the whole tensor's `ess_grid`.

.rail_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn))
}

# A one-axis stub result carrying `w` as its marginal over `vals`.
.rail_stub <- function(vals, w, name = "tau") {
    tg <- matrix(vals, ncol = 1L, dimnames = list(NULL, name))
    list(theta_grid = tg, theta_names = name, weights = w / sum(w),
         log_marginal = log(w / sum(w)))
}

test_that("an axis is railed exactly when its own marginal peaks at an endpoint", {
    up   <- .rail_stub(c(0.2, 0.5, 0.8, 0.95), c(0.001, 0.02, 0.35, 0.63))
    down <- .rail_stub(c(0.2, 0.5, 0.8, 0.95), c(0.70, 0.25, 0.04, 0.01))
    mid  <- .rail_stub(c(0.2, 0.5, 0.8, 0.95), c(0.05, 0.60, 0.30, 0.05))
    flat <- .rail_stub(c(0.2, 0.5, 0.8, 0.95), c(0.26, 0.25, 0.245, 0.245))

    expect_identical(.nl_axis_rail(up,   "tau")$side, "upper")
    expect_identical(.nl_axis_rail(down, "tau")$side, "lower")
    expect_null(.nl_axis_rail(mid,  "tau"))
    # Maximal at an endpoint by 0.015 of the weight: the argmax of a marginal
    # this flat is arbitrary, and a recenter would lay a coarser grid.
    expect_null(.nl_axis_rail(flat, "tau"))

    expect_identical(.nl_railed_axes(up),  "tau:upper")
    expect_identical(.nl_railed_axes(mid), character(0))
})

test_that("the rail verdict does not move with the axis's node count", {
    # gcol33/tulpa#375. The materiality guard reads a per-NODE quantity, so it
    # has to be compared against a per-node reference. Against a fixed SHARE the
    # same posterior read at more nodes carries less on any one of them, and a
    # longer axis becomes a weaker detector -- which is the opposite of what
    # #361's longer `bym2_rho` span was meant to compose with.
    #
    # One marginal shape, read at six node counts over ONE fixed span in the
    # axis's own coordinate. The span is what fixes the posterior geometry; the
    # node count is the only thing that moves.
    span <- stats::qlogis(c(0.2, 0.95))
    ms   <- c(4L, 5L, 6L, 8L, 12L, 20L)
    read_at <- function(m, shape) {
        u <- seq(span[1L], span[2L], length.out = m)
        w <- shape(u); w <- w / sum(w)
        .rail_stub(stats::plogis(u), w, "rho")
    }

    # Piled against the ceiling: a mode 1.5 logit units PAST the top node, so
    # every read has its argmax at that node whatever the resolution.
    piled <- function(u) exp(-0.5 * ((u - (span[2L] + 1.5)) / 2.1)^2)
    lift  <- vapply(ms, function(m) .nl_axis_rail(read_at(m, piled), "rho")$lift,
                    numeric(1))
    expect_true(all(lift >= 2))
    # The share it used to be read against falls by more than 3x over the same
    # reads, and drops through 0.5 -- the defect, in one line.
    share <- vapply(ms, function(m) .nl_axis_rail(read_at(m, piled), "rho")$mass,
                    numeric(1))
    expect_gt(share[1L] / share[length(share)], 3)
    expect_gt(share[1L], 0.5)
    expect_lt(share[length(share)], 0.5)

    # Flat to within the weights' own noise: the argmax is arbitrary and lands
    # on the top node by construction. Declines at every node count -- the
    # guard's own job, which the rescale must not cost.
    for (m in ms) {
        set.seed(400L + m)
        w <- rep(1, m) + stats::runif(m, -0.02, 0.02)
        w[m] <- max(w) + 0.01
        expect_null(.nl_axis_rail(.rail_stub(
            stats::plogis(seq(span[1L], span[2L], length.out = m)),
            w / sum(w), "rho"), "rho"))
    }

    # And a marginal that merely tilts: 1.6x from the bottom node to the top
    # over the whole span, no curvature to recentre onto.
    gentle <- function(u) exp(0.47 * (u - u[1L]) / diff(range(u)))
    for (m in ms) expect_null(.nl_axis_rail(read_at(m, gentle), "rho"))
})

test_that("a recentred axis is laid in its own coordinate and stays in support", {
    # `log`: unchanged from the positive-scale generator every existing caller
    # uses, so the two names are the same nodes.
    expect_identical(.nl_recenter_axis("log", log(2), 0.4),
                     .nl_recenter_log_axis(log(2), 0.4))

    # `logit01`: a proportion axis maps back into the OPEN interval, which is
    # what makes a railed BYM2 `rho` movable at all.
    rho <- .nl_recenter_axis("logit01", stats::qlogis(0.95), 2.1)
    expect_length(rho, .nl_recenter("n_pts"))
    expect_true(all(rho > 0 & rho < 1))
    expect_false(is.unsorted(rho))
    expect_gt(max(rho), 0.95)

    # A mode far enough out that the upper nodes saturate to 1 in double
    # precision declines rather than laying a node ON the singular boundary.
    expect_null(.nl_recenter_axis("logit01", 40, 3))
    # An axis whose support the engine does not guess (car_proper's `rho_car`)
    # is never placed on a guessed transform.
    expect_null(.nl_recenter_axis(NA_character_, 0, 1))
})

test_that("the registry rescue moves a railed rho, not only a railed sigma", {
    # Direct unit test of `.nl_registry_grid_rescue()` on bym2's (sigma, rho)
    # grid with a synthetic target: sharp and interior in log-sigma, monotone
    # rising in rho all the way to the 0.95 ceiling. That is the shape the
    # gcol33/tulpa#357 census measures on a 100-region BYM2 -- the axis that
    # rails is the BOUNDED one -- and before #361 the rescue recentred
    # `sigma_grid` only and re-crossed rho's four default nodes unchanged.
    sg <- .nl_grid_axis("field_sd")
    rg <- .nl_grid_axis("bym2_rho")
    gr <- expand.grid(sigma = sg, rho = rg, KEEP.OUT.ATTRS = FALSE)
    theta_grid <- cbind(sigma = gr$sigma, rho = gr$rho)

    # Peaked at sigma = 0.55 (interior), and rising in logit(rho) with a mode at
    # logit(rho) = 8.5 (rho = 0.99980), past the axis's top node whatever
    # `.NL_GRID` places there -- the default reaches 0.999 (logit 6.91) since
    # gcol33/tulpa#361 extended the span.
    synthetic_lm <- function(tm)
        -0.5 * ((log(tm[, 1L]) - log(0.55)) / 0.10)^2 +
        -0.5 * ((stats::qlogis(tm[, 2L]) - 8.5) / 2.1)^2
    log_marg <- synthetic_lm(theta_grid)
    w <- exp(log_marg - max(log_marg)); w <- w / sum(w)
    res <- list(theta_grid = theta_grid, theta_names = c("sigma", "rho"),
                weights = w, log_marginal = log_marg)
    res <- .joint_attach_pareto_k_regime(res)

    # The rail is on rho and the sigma axis brackets its own mode.
    expect_identical(.nl_axis_rail(res, "rho")$side, "upper")
    expect_null(.nl_axis_rail(res, "sigma"))

    refit <- function(prior_i) {
        new_grid <- cbind(sigma = prior_i$sigma_grid, rho = prior_i$rho_grid)
        lm_new <- synthetic_lm(new_grid)
        w_new  <- exp(lm_new - max(lm_new)); w_new <- w_new / sum(w_new)
        .joint_attach_pareto_k_regime(
            list(theta_grid = new_grid, theta_names = c("sigma", "rho"),
                 weights = w_new, log_marginal = lm_new))
    }
    rescue <- .nl_registry_grid_rescue(res, "bym2", list(type = "bym2"),
                                       refit,
                                       function(prior_i, tm) synthetic_lm(tm))

    expect_identical(rescue$res$outer_grid_placement, "auto_recentered")
    expect_identical(rescue$res$outer_grid_recenter_axes, "rho")
    new_rho <- sort(unique(rescue$prior$rho_grid))
    # The rho axis now brackets a mode the default's ceiling excluded.
    expect_gt(max(new_rho), max(rg))
    expect_lt(min(new_rho), stats::plogis(8.5))
    expect_true(all(new_rho > 0 & new_rho < 1))
    # sigma was not railed, so its four default nodes are re-crossed unchanged.
    expect_identical(sort(unique(rescue$prior$sigma_grid)), sort(sg))
    expect_null(.nl_axis_rail(rescue$res, "rho"))
})

test_that("a pinned rho axis rails without being moved, and says which", {
    sg <- .nl_grid_axis("field_sd")
    rg <- c(0.1, 0.4, 0.7, 0.9)          # a caller's own nodes, not the default
    gr <- expand.grid(sigma = sg, rho = rg, KEEP.OUT.ATTRS = FALSE)
    theta_grid <- cbind(sigma = gr$sigma, rho = gr$rho)
    synthetic_lm <- function(tm)
        -0.5 * ((log(tm[, 1L]) - log(0.55)) / 0.10)^2 +
        -0.5 * ((stats::qlogis(tm[, 2L]) - 5.3) / 2.1)^2
    log_marg <- synthetic_lm(theta_grid)
    w <- exp(log_marg - max(log_marg)); w <- w / sum(w)
    res <- .joint_attach_pareto_k_regime(
        list(theta_grid = theta_grid, theta_names = c("sigma", "rho"),
             weights = w, log_marginal = log_marg))

    calls <- new.env(); calls$n <- 0L
    refit <- function(prior_i) { calls$n <- calls$n + 1L; res }
    prior <- list(type = "bym2", sigma_grid = gr$sigma, rho_grid = gr$rho)
    rescue <- .nl_registry_grid_rescue(res, "bym2", prior, refit,
                                       function(prior_i, tm) synthetic_lm(tm))

    expect_identical(calls$n, 0L)
    expect_identical(rescue$res$outer_grid_recenter_declined, "axis_pinned")
    expect_identical(rescue$prior$rho_grid, gr$rho)
    # The rail is still REPORTED -- a span the engine is not allowed to move is
    # not a span that contains its mode.
    expect_true("rho:upper" %in% rescue$res$outer_grid_railed_axes)
})

test_that("a BYM2 fit whose mixing weight rails is moved off its ceiling", {
    skip_on_cran()
    # gcol33/tulpa#357's census configuration: 100 chain-adjacent regions, whose
    # `bym2_rho` posterior is maximal at a 0.95 top node. Pre-#361 the fit
    # reported a rho median of 0.89 against a span stopping at 0.95, with the
    # upper interval bound extrapolated PAST the support, and recorded the
    # decline reason `grid_not_collapsed` on a grid that had collapsed.
    #
    # The 0.95 ceiling is supplied here through `auto_grid()` rather than taken
    # from `.NL_GRID`: #361's span change carried the default to 0.999, which
    # CONTAINS this fixture's mode, so the engine's own axis no longer rails on
    # it. `auto_grid()` is the marker for "a default the engine may move", so
    # the rescue path exercised is identical and the statement no longer moves
    # with the default's value. The block below asserts what the longer default
    # does on the same data.
    S <- 100L
    set.seed(505L + S + 5000L)
    eff <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4)), scale = FALSE)) +
           rnorm(S, 0, 0.2)
    set.seed(505L + S)
    idx <- rep(seq_len(S), each = 10L)
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] +
         rnorm(length(idx), 0, sqrt(0.5))

    gr <- expand.grid(sigma = .nl_grid_axis("field_sd"),
                      rho = c(0.2, 0.5, 0.8, 0.95), KEEP.OUT.ATTRS = FALSE)
    prior <- c(list(type = "bym2", n_spatial_units = S, spatial_idx = idx,
                    scale_factor = 1,
                    sigma_grid = auto_grid(gr$sigma),
                    rho_grid   = auto_grid(gr$rho)), .rail_chain_adj(S))
    ctrl <- list(max_iter = 200L, tol = 1e-9, n_threads = 1L,
                 diagnose_k = FALSE, diagnose_skew = FALSE)
    fit_held <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = prior,
        family = "gaussian", phi = sqrt(0.5),
        control = c(ctrl, list(auto_recenter = FALSE))))
    fit <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = prior,
        family = "gaussian", phi = sqrt(0.5), control = ctrl))

    # Held where it is, the axis stops at 0.95 with its marginal still climbing.
    expect_identical(max(fit_held$theta_grid[, "rho"]), 0.95)
    expect_identical(.nl_axis_rail(fit_held, "rho")$side, "upper")
    expect_true("rho:upper" %in% fit_held$outer_grid_railed_axes)

    # Moved, the axis brackets its own mode and the reported interval is inside
    # the support the mixing weight lives on.
    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$outer_grid_recenter_axes, "rho")
    expect_gt(max(fit$theta_grid[, "rho"]), 0.95)
    expect_null(.nl_axis_rail(fit, "rho"))
    expect_gt(as.numeric(fit$theta_median[["rho"]]),
              as.numeric(fit_held$theta_median[["rho"]]))
    # Both reads report a mixing weight inside its own support, which is
    # gcol33/tulpa#369 and not placement -- before that the held axis overshot
    # to 1.0028 and the moved one to 1.0012.
    expect_lt(as.numeric(fit$theta_ci_hi[["rho"]]), 1)
    expect_lt(as.numeric(fit_held$theta_ci_hi[["rho"]]), 1)

    # The engine's own axis, on the same data. gcol33/tulpa#361's span change
    # is what makes this a different statement from the block above: a bounded
    # axis's SPAN and its RESOLUTION are not interchangeable, so a span topping
    # out at 0.95 could not report a bound above 0.97642 at any node count.
    fit_default <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X,
        prior = c(list(type = "bym2", n_spatial_units = S, spatial_idx = idx,
                       scale_factor = 1), .rail_chain_adj(S)),
        family = "gaussian", phi = sqrt(0.5), control = ctrl))
    expect_identical(max(fit_default$theta_grid[, "rho"]), 0.999)
    # It contains its own mode, so nothing rails and no refit is spent.
    expect_null(.nl_axis_rail(fit_default, "rho"))
    expect_identical(fit_default$outer_grid_placement, "fixed")
    expect_identical(fit_default$outer_grid_railed_axes, character(0))
    # And it reports past the retired cap, inside the support.
    expect_gt(as.numeric(fit_default$theta_ci_hi[["rho"]]), 0.97642)
    expect_lt(as.numeric(fit_default$theta_ci_hi[["rho"]]), 1)
    expect_gt(as.numeric(fit_default$theta_median[["rho"]]),
              as.numeric(fit_held$theta_median[["rho"]]))
})

test_that("auto_recenter = \"always\" recentres an axis that did not rail", {
    skip_on_cran()
    # gcol33/tulpa#361 checklist items 1-2: the placement policy knob. The
    # unconditional arm drops the rail test and nothing else, so an axis that
    # WOULD have railed lands on the same nodes either way, and one that would
    # not is moved onto its own posterior mode at `h / sd = 1.25`.
    expect_identical(.nl_recenter_mode(NULL), "rail")
    expect_identical(.nl_recenter_mode(TRUE), "rail")
    expect_identical(.nl_recenter_mode(FALSE), "off")
    expect_identical(.nl_recenter_mode("always"), "always")
    expect_error(.nl_recenter_mode("sometimes"), "TRUE, FALSE")

    S <- 100L
    set.seed(811L)
    Q <- matrix(0, S, S)
    for (i in seq_len(S)) {
        nb <- setdiff(c(i - 1L, i + 1L), c(0L, S + 1L))
        Q[i, i] <- length(nb); Q[i, nb] <- -1
    }
    e <- eigen(Q, symmetric = TRUE); pos <- which(e$values > 1e-8)
    eff <- as.numeric(e$vectors[, pos, drop = FALSE] %*%
                      (rnorm(length(pos)) / sqrt(4 * e$values[pos])))
    idx <- rep(seq_len(S), each = 10L)
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] +
         rnorm(length(idx), 0, sqrt(0.5))
    prior <- c(list(type = "icar", n_spatial_units = S, spatial_idx = idx),
               .rail_chain_adj(S))
    ctrl <- list(max_iter = 200L, tol = 1e-9, n_threads = 1L,
                 diagnose_k = FALSE, diagnose_skew = FALSE)

    fit <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = prior,
        family = "gaussian", phi = sqrt(0.5), control = ctrl))
    always <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = prior,
        family = "gaussian", phi = sqrt(0.5),
        control = c(ctrl, list(auto_recenter = "always"))))

    # The default axis contains its own mode, so the rail-gated policy leaves
    # it alone and the unconditional one still moves it.
    expect_null(.nl_axis_rail(fit, "tau"))
    expect_identical(fit$outer_grid_placement, "fixed")
    expect_identical(always$outer_grid_placement, "auto_recentered")
    expect_identical(always$outer_grid_recenter_axes, "tau")
    expect_length(unique(as.numeric(always$theta_grid)),
                  .nl_recenter("n_pts"))
    # The recentred span is the tighter one: it brackets the fixed grid's own
    # posterior median rather than the prior's two decades.
    med <- as.numeric(fit$theta_median[[1L]])
    expect_lt(diff(range(as.numeric(always$theta_grid))),
              diff(range(.nl_grid_axis("gmrf_tau"))))
    expect_gt(max(as.numeric(always$theta_grid)), med)
    expect_lt(min(as.numeric(always$theta_grid)), med)

    # A pinned axis is still never moved, whatever the policy asks for.
    pinned <- prior
    pinned$tau_grid <- c(1, 2, 4, 8)
    held <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = pinned,
        family = "gaussian", phi = sqrt(0.5),
        control = c(ctrl, list(auto_recenter = "always"))))
    expect_identical(held$outer_grid_placement, "fixed")
    expect_identical(held$outer_grid_recenter_declined, "axis_pinned")

    # The paths that do not implement it refuse it rather than ignore it.
    expect_error(
        suppressWarnings(tulpa_nested_laplace_joint(
            responses = list(list(y = y, n_trials = rep(1L, length(y)), X = X,
                                  family = "gaussian", phi = sqrt(0.5))),
            prior = c(list(type = "icar", n_spatial_units = S,
                           spatial_idx = idx), .rail_chain_adj(S)),
            control = c(ctrl, list(auto_recenter = "always")))),
        "standalone")
})
