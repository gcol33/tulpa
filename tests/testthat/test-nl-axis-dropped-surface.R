# Surfacing the dropped-axis record (gcol33/tulpa#355). gcol33/tulpa#352 writes
# `$axis_fields_dropped` when a supplied axis was an engine default the resolved
# path does not read; this file holds the three documented readers to it, and
# holds the ordinary fit -- the one that used every axis it was given -- silent
# in all three.

.axd_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

.axd_sim <- function(seed = 71L, n_s = 12L, N = 100L) {
    set.seed(seed)
    adj <- .axd_chain_adj(n_s)
    sidx <- sample.int(n_s, N, replace = TRUE)
    phi <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.5))))
    X <- cbind(1, rnorm(N))
    y <- rbinom(N, 1L, plogis(as.numeric(X %*% c(-0.2, 0.5)) + phi[sidx]))
    list(adj = adj, sidx = sidx, X = X, y = y, N = N)
}

.axd_fit <- function(sim, ...) {
    blk <- c(list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors,
                  spatial_idx = sim$sidx), list(...))
    tulpa_nested_laplace(y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
                         prior = blk, family = "binomial",
                         control = list(diagnose_k = FALSE))
}

# --------------------------------------------------------------------------- #
# The formatter                                                                #
# --------------------------------------------------------------------------- #

test_that("the note names the field, the block, the path and the axis used", {
    rec <- data.frame(block = 2L, type = "icar", field = "sigma_grid",
                      path = "registry", integrates = "tau_grid",
                      reason = "default_axis_not_read_by_this_path",
                      stringsAsFactors = FALSE)
    note <- tulpa:::.tulpa_axis_dropped_note(rec)
    expect_length(note, 1L)
    expect_match(note, "`sigma_grid`", fixed = TRUE)
    expect_match(note, "prior block 2", fixed = TRUE)
    expect_match(note, "icar", fixed = TRUE)
    expect_match(note, "`tau_grid`", fixed = TRUE)
    # The path is named with the SAME label the #352 refusal uses.
    expect_match(note, "registry path", fixed = TRUE)

    # A single-block record carries no block number and must not print `NA`.
    rec$block <- NA_integer_
    expect_false(grepl("NA", tulpa:::.tulpa_axis_dropped_note(rec), fixed = TRUE))

    # A path that integrates several axes lists them all.
    rec$type <- "bym2"; rec$integrates <- "sigma_grid, rho_grid"
    note2 <- tulpa:::.tulpa_axis_dropped_note(rec)
    expect_match(note2, "`sigma_grid`, `rho_grid`", fixed = TRUE)

    # Nothing dropped, nothing said.
    expect_null(tulpa:::.tulpa_axis_dropped_note(NULL))
    expect_null(tulpa:::.tulpa_axis_dropped_line(NULL))
})

test_that("the reader unwraps a model package's nested fit", {
    rec <- data.frame(block = NA_integer_, type = "icar", field = "sigma_grid",
                      path = "registry", integrates = "tau_grid",
                      reason = "default_axis_not_read_by_this_path",
                      stringsAsFactors = FALSE)
    inner <- structure(list(axis_fields_dropped = rec), class = "tulpa_fit")
    wrap  <- structure(list(joint_fit = inner), class = "tulpa_fit")
    expect_identical(tulpa:::.tulpa_axis_dropped(wrap), rec)
    # An empty record reads as nothing dropped, not as a zero-row table.
    empty <- structure(list(axis_fields_dropped = rec[0L, ]), class = "tulpa_fit")
    expect_null(tulpa:::.tulpa_axis_dropped(empty))
})

# --------------------------------------------------------------------------- #
# The three readers, on a fit that actually dropped an axis                    #
# --------------------------------------------------------------------------- #

test_that("a dropped axis surfaces through diagnostic_summary, print and summary", {
    skip_on_cran()
    sim <- .axd_sim(seed = 71L)
    fit <- .axd_fit(sim, sigma_grid = tulpa:::.nl_grid_axis("field_sd"))
    rec <- fit$axis_fields_dropped
    expect_s3_class(rec, "data.frame")
    expect_identical(rec$field, "sigma_grid")

    ds <- diagnostic_summary(fit, quiet = TRUE)
    expect_identical(ds$axis_fields_dropped, rec)
    expect_true(any(grepl("`sigma_grid`", ds$recommendations, fixed = TRUE)))
    expect_true(any(grepl("`tau_grid`", ds$recommendations, fixed = TRUE)))
    # And the printed summary carries it, not just the returned list.
    expect_output(print(ds), "sigma_grid")

    out <- paste(utils::capture.output(print(fit)), collapse = "\n")
    expect_match(out, "unused axis fields", fixed = TRUE)
    expect_match(out, "sigma_grid", fixed = TRUE)
    expect_match(out, "diagnostic_summary", fixed = TRUE)

    expect_identical(attr(summary(fit), "axis_fields_dropped"), rec)
})

test_that("an auto_grid()-marked drop surfaces the same way", {
    skip_on_cran()
    sim <- .axd_sim(seed = 72L)
    fit <- .axd_fit(sim, sigma_grid = auto_grid(c(0.4, 0.9)))
    ds <- diagnostic_summary(fit, quiet = TRUE)
    expect_identical(nrow(ds$axis_fields_dropped), 1L)
    expect_true(any(grepl("`sigma_grid`", ds$recommendations, fixed = TRUE)))
})

# --------------------------------------------------------------------------- #
# The ordinary fit is unchanged                                                #
# --------------------------------------------------------------------------- #

test_that("a fit with nothing dropped says nothing anywhere", {
    skip_on_cran()
    sim <- .axd_sim(seed = 73L)
    fit <- .axd_fit(sim, tau_grid = c(1, 4, 9))
    expect_null(fit$axis_fields_dropped)

    ds <- diagnostic_summary(fit, quiet = TRUE)
    expect_null(ds$axis_fields_dropped)
    expect_false(any(grepl("was not used", ds$recommendations, fixed = TRUE)))

    out <- paste(utils::capture.output(print(fit)), collapse = "\n")
    expect_false(grepl("unused axis fields", out, fixed = TRUE))

    expect_null(attr(summary(fit), "axis_fields_dropped"))
})
