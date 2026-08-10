# The mode-SD clamp on a recentred outer axis (gcol33/tulpa#387).
#
# The clamp substitutes a number for a curvature the FD stencil could not read.
# Before #387 it was an anonymous `min(max(...))` inside the node generator, so
# a fit laid from a substituted spread and one laid from a measured spread were
# indistinguishable afterwards -- the gcol33/tulpa#293 rule, one layer down.
# These pin (a) that the clamp is ONE function, (b) that what it did is on the
# fit, and (c) that the shipped default is byte-identical to the formula it
# replaced.

test_that("the clamp reports which bound it hit, and the SD it measured", {
    mn <- .nl_recenter("min_sd_u")
    mx <- .nl_recenter("max_sd_u")

    inside <- .nl_recenter_sd_clamp((mn + mx) / 2)
    expect_identical(inside$clamp, "none")
    expect_identical(inside$sd, (mn + mx) / 2)
    expect_identical(inside$sd_raw, inside$sd)
    expect_null(inside$reason)

    # The ceiling declines by default, so there is no SD to lay an axis from --
    # but the MEASURED spread is reported either way, since it is the reading
    # that says whether declining was the right call.
    hi <- .nl_recenter_sd_clamp(mx * 20)
    expect_identical(hi$clamp, "ceiling")
    expect_identical(hi$reason, "sd_ceiling_unresolved")
    expect_true(is.na(hi$sd))
    expect_identical(hi$sd_raw, mx * 20)

    hic <- .nl_recenter_sd_clamp(mx * 20, ceiling = "clamp")
    expect_identical(hic$clamp, "ceiling")
    expect_identical(hic$sd, mx)
    expect_identical(hic$sd_raw, mx * 20)
    expect_null(hic$reason)

    lo <- .nl_recenter_sd_clamp(mn / 20)
    expect_identical(lo$clamp, "floor")
    expect_identical(lo$sd, mn)
    expect_identical(lo$sd_raw, mn / 20)

    # A curvature that is not a curvature at all is not a clamp -- it declines.
    for (bad in list(0, -1, NA_real_, Inf, numeric(0))) {
        z <- .nl_recenter_sd_clamp(bad)
        expect_identical(z$reason, "no_usable_curvature")
        expect_true(is.na(z$clamp))
    }
})

test_that("the clamp policy lays exactly the axis the pre-#387 formula did", {
    # `"clamp"` IS `min(max(sd, lo), hi)`, so the arm the measurement compared
    # against is the pre-#387 engine exactly. (It is no longer the DEFAULT --
    # see the ceiling block below -- but it remains what a fit gets on the
    # coordinates the ceiling does not reach, which is 266 of 268 axis reads.)
    # Written out independently rather than calling the engine's own helper, so
    # this compares against the formula and not against itself.
    withr::local_options(list(tulpa.recenter.sd_clamp_policy = "clamp"))
    ref <- function(tag, mode_u, sd_u, n_pts = .nl_recenter("n_pts"),
                    span = .nl_recenter("span"),
                    min_sd_u = .nl_recenter("min_sd_u"),
                    max_sd_u = .nl_recenter("max_sd_u")) {
        if (length(tag) != 1L || is.na(tag) ||
            !tag %in% c("log", "logit01", "identity")) return(NULL)
        if (!is.finite(mode_u) || !is.finite(sd_u) || sd_u <= 0) return(NULL)
        sd_u  <- min(max(sd_u, min_sd_u), max_sd_u)
        u_seq <- seq(mode_u - span * sd_u, mode_u + span * sd_u,
                     length.out = as.integer(n_pts))
        nodes <- .joint_pareto_inv(tag, u_seq)$theta
        keep  <- is.finite(nodes)
        if (identical(tag, "logit01")) keep <- keep & nodes > 0 & nodes < 1
        nodes <- sort(unique(nodes[keep]))
        if (length(nodes) < .nl_recenter("min_nodes")) return(NULL)
        nodes
    }
    set.seed(387)
    for (tag in c("log", "logit01", "identity")) {
        for (i in 1:200) {
            m <- stats::rnorm(1, 0, 3)
            s <- stats::rgamma(1, shape = 1, rate = 1) * 4
            expect_identical(.nl_recenter_axis(tag, m, s), ref(tag, m, s))
        }
    }
})

test_that("the ceiling DECLINES by default, and the floor still clamps", {
    # The measured defaults (gcol33/tulpa#387): a clamped ceiling means the
    # stencil could not resolve that direction, and laying an axis from the
    # substituted number never wins a trial and loses none when skipped (7-0
    # paired). A clamped FLOOR widens a too-narrow axis, which is the direction
    # that cannot rail, and declining there loses 22 trials against 9.
    expect_identical(.nl_recenter("sd_clamp_policy"), "decline")
    expect_identical(.nl_recenter("sd_floor_policy"), "clamp")

    rc <- .nl_recenter_axis_full("log", 0, .nl_recenter("max_sd_u") * 10)
    expect_null(rc$nodes)
    expect_identical(rc$reason, "sd_ceiling_unresolved")
    expect_identical(rc$sd_clamp, "ceiling")
    expect_identical(rc$sd_raw, .nl_recenter("max_sd_u") * 10)

    # An SD the stencil DID resolve is untouched by the policy.
    ok <- .nl_recenter_axis_full("log", 0, 0.4)
    expect_null(ok$reason)
    expect_length(ok$nodes, .nl_recenter("n_pts"))

    # The floor carries its own policy, so declining the ceiling does not
    # silently start declining the floor as well.
    fl <- .nl_recenter_axis_full("log", 0, .nl_recenter("min_sd_u") / 10)
    expect_identical(fl$sd_clamp, "floor")
    expect_null(fl$reason)
})

test_that("the relative ceiling caps by the incoming span, or falls back", {
    withr::local_options(list(tulpa.recenter.sd_clamp_policy = "relative"))
    span <- .nl_recenter("span")
    huge <- .nl_recenter("max_sd_u") * 10

    # An incoming axis covering 4 nats caps the re-placed axis at the same 4.
    rel <- .nl_recenter_axis_full("log", 0, huge, ref_span_u = 4)
    expect_identical(rel$sd_clamp, "ceiling")
    expect_equal(diff(range(log(rel$nodes))), 4, tolerance = 1e-12)

    # With no reference span the policy has nothing to be relative TO, and the
    # absolute ceiling applies rather than a guess.
    for (ref in list(NULL, NA_real_, 0, -1)) {
        fb <- .nl_recenter_axis_full("log", 0, huge, ref_span_u = ref)
        expect_equal(diff(range(log(fb$nodes))),
                     2 * span * .nl_recenter("max_sd_u"), tolerance = 1e-12)
    }

    # The floor still wins over a vanishing reference span: the point of
    # re-placing is to bracket the mode with real spread.
    tiny <- .nl_recenter_axis_full("log", 0, huge, ref_span_u = 1e-8)
    expect_equal(diff(range(log(tiny$nodes))),
                 2 * span * .nl_recenter("min_sd_u"), tolerance = 1e-12)
})

test_that("the incoming span is read in each axis's own coordinate", {
    # `.nl_axis_span_u()` is the registry/joint reader (log, logit01, identity);
    # the spatiotemporal driver's `rho` lives on (-1, 1) instead, so it has its
    # own. Both feed the SAME clamp -- the ST path no longer carries a second
    # copy of the bounds.
    expect_equal(.nl_axis_span_u(c(1, exp(2)), "log"), 2, tolerance = 1e-12)
    expect_true(is.na(.nl_axis_span_u(5, "log")))          # one node: no span
    expect_true(is.na(.nl_axis_span_u(NULL, "log")))
    expect_true(is.na(.nl_axis_span_u(c(1, 2), NA_character_)))

    res <- list(theta_grid = cbind(rho = c(-0.5, 0, 0.5)),
                theta_names = "rho")
    expect_equal(.nl_axis_span_u_st(res, "rho"),
                 .st_axis_fwd("rho", 0.5) - .st_axis_fwd("rho", -0.5),
                 tolerance = 1e-12)
})

test_that("a recentred registry fit records what its axis was laid from", {
    # A target that is SHARP in rho and nearly flat in log-sigma: the FD stencil
    # reads a mode SD of 8 nats on sigma, well past the ceiling, so the axis is
    # laid from a substituted spread. That is the #387 case, and the fit has to
    # say so.
    sg <- .nl_grid_axis("field_sd")
    rg <- .nl_grid_axis("bym2_rho")
    gr <- expand.grid(sigma = sg, rho = rg, KEEP.OUT.ATTRS = FALSE)
    theta_grid <- cbind(sigma = gr$sigma, rho = gr$rho)
    synthetic_lm <- function(tm)
        -0.5 * ((log(tm[, 1L]) - log(0.55)) / 8)^2 +
        -0.5 * ((stats::qlogis(tm[, 2L]) - 1.2) / 0.9)^2
    log_marg <- synthetic_lm(theta_grid)
    w <- exp(log_marg - max(log_marg)); w <- w / sum(w)
    res <- .joint_attach_pareto_k_regime(
        list(theta_grid = theta_grid, theta_names = c("sigma", "rho"),
             weights = w, log_marginal = log_marg))
    refit <- function(prior_i) {
        ng <- cbind(sigma = prior_i$sigma_grid, rho = prior_i$rho_grid)
        lm_new <- synthetic_lm(ng)
        w_new  <- exp(lm_new - max(lm_new)); w_new <- w_new / sum(w_new)
        .joint_attach_pareto_k_regime(
            list(theta_grid = ng, theta_names = c("sigma", "rho"),
                 weights = w_new, log_marginal = lm_new))
    }
    run <- function() .nl_registry_grid_rescue(
        res, "bym2", list(type = "bym2"), refit,
        function(prior_i, tm) synthetic_lm(tm), policy = "always")

    # Under `"clamp"` the axis IS laid, from the substituted spread, and the fit
    # says as much.
    out <- withr::with_options(
        list(tulpa.recenter.sd_clamp_policy = "clamp"), run())$res
    expect_identical(out$outer_grid_placement, "auto_recentered")
    cl <- out$outer_grid_recenter_sd_clamp
    expect_identical(unname(cl[["sigma"]]), "ceiling")
    # rho's own curvature is resolved, so it is not clamped -- the record is per
    # axis, not per fit.
    expect_identical(unname(cl[["rho"]]), "none")
    raw  <- out$outer_grid_recenter_sd_raw
    used <- out$outer_grid_recenter_sd_used
    expect_equal(unname(raw[["sigma"]]), 8, tolerance = 0.05)
    expect_identical(unname(used[["sigma"]]), .nl_recenter("max_sd_u"))
    # The unclamped axis reports the same number twice, which is what "the
    # stencil measured this" looks like.
    expect_equal(unname(raw[["rho"]]), unname(used[["rho"]]), tolerance = 1e-12)

    # Under the shipped default the sigma axis is NOT laid -- but rho, whose
    # curvature the stencil did resolve, still is. The decline is per axis.
    dflt <- run()$res
    expect_identical(dflt$outer_grid_placement, "auto_recentered")
    expect_identical(unname(dflt$outer_grid_recenter_sd_clamp[["sigma"]]),
                     "ceiling")
    expect_equal(unname(dflt$outer_grid_recenter_sd_raw[["sigma"]]), 8,
                 tolerance = 0.05)
    expect_false("sigma" %in% names(dflt$outer_grid_recenter_sd_used))
    expect_identical(unname(dflt$outer_grid_recenter_axes), c("sigma", "rho"))

    # And the placement note says the bound was hit, so a reader is not left to
    # infer it from the width.
    note <- .tulpa_grid_placement_note(.tulpa_grid_placement(out))
    expect_match(note, "mode SD hit its bound on .*sigma")
})

test_that("declining on the ceiling is visible as its own reason", {
    sg <- .nl_grid_axis("field_sd")
    theta_grid <- cbind(sigma = sg)
    synthetic_lm <- function(tm) -0.5 * ((log(tm[, 1L]) - log(0.55)) / 8)^2
    log_marg <- synthetic_lm(theta_grid)
    w <- exp(log_marg - max(log_marg)); w <- w / sum(w)
    res <- .joint_attach_pareto_k_regime(
        list(theta_grid = theta_grid, theta_names = "sigma",
             weights = w, log_marginal = log_marg))
    refit <- function(prior_i) {
        ng <- cbind(sigma = prior_i$sigma_grid)
        lm_new <- synthetic_lm(ng)
        w_new  <- exp(lm_new - max(lm_new)); w_new <- w_new / sum(w_new)
        .joint_attach_pareto_k_regime(
            list(theta_grid = ng, theta_names = "sigma",
                 weights = w_new, log_marginal = lm_new))
    }
    rescue <- .nl_registry_grid_rescue(res, "iid", list(type = "iid"), refit,
                                       function(prior_i, tm) synthetic_lm(tm),
                                       policy = "always")
    # The fit says which of the two reasons applied -- not the generic
    # "no_usable_curvature", which would read as "the stencil failed" rather
    # than "the stencil said something the policy refuses to lay an axis from".
    expect_identical(rescue$res$outer_grid_recenter_declined,
                     "sd_ceiling_unresolved")
    expect_false(identical(rescue$res$outer_grid_placement, "auto_recentered"))
    # A declining rescue writes NO axis back onto the prior, so the grid the
    # caller supplied is the grid that gets integrated.
    expect_null(rescue$prior$sigma_grid)
    expect_identical(rescue$res$theta_grid, theta_grid)
})
