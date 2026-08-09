# A reported median / interval is read off a node set, and two node sets that
# both carry a CDF still have different GEOMETRY at the edge (gcol33/tulpa#358).
#
# A tensor grid's values are cell representatives of a partition with known
# spacing: the extreme cell's mass is centred at its coordinate and reaches half
# a spacing past it, so the read runs to the design's own edge
# (`outside = "extend"`, gcol33/tulpa#353). An equal-weight MCMC sample's values
# are order statistics: beyond the largest draw the tail is unknown and half the
# gap between the two extreme draws is not a cell width, so the read clamps.
#
# Both used to be called `"density"`, which made them indistinguishable to a
# caller and gave the sample an outer half-cell it does not have. These tests pin
# the separated taxonomy, the producer that names each kind, and -- the reason
# this is a refactor rather than a fix -- that nothing moves at any probability
# the backends actually report.

# --------------------------------------------------------------- the taxonomy

test_that("every support kind names an outer-edge policy exactly once", {
  expect_identical(.NL_SUPPORT_KINDS, names(.NL_SUPPORT))
  # `match.arg` takes the first entry as the default, so `density` staying first
  # is what keeps every existing caller's unnamed default unchanged.
  expect_identical(.NL_SUPPORT_KINDS[1L], "density")
  expect_setequal(.NL_SUPPORT_KINDS,
                  c("density", "moment_rule", "mixed", "sample"))
  pol <- vapply(.NL_SUPPORT, `[[`, character(1), "outside")
  # The two kinds this issue separates. `mixed` is deliberately not pinned here:
  # its policy is gcol33/tulpa#353's to set, and pinning it in a taxonomy test
  # would put that decision in two files.
  expect_identical(pol[["density"]], "extend")
  expect_identical(pol[["sample"]], "clamp")
  # A moment rule never reaches the quantile read, so it carries no policy.
  expect_true(is.na(pol[["moment_rule"]]))
  # Every policy the table names is one `.nl_wtd_quantile()` accepts.
  expect_true(all(stats::na.omit(pol) %in%
                    eval(formals(.nl_wtd_quantile)$outside)))
  # The second field is the WITHIN-CELL vocabulary (gcol33/tulpa#357), held to
  # `.NL_WITHIN_CELL` the same way. It is a field rather than a fifth kind
  # because `outside` is a fact about the node set and `within` is a caller's
  # choice about how to read it; the setequal pin above is what a fifth kind
  # would have broken.
  wit <- lapply(.NL_SUPPORT, `[[`, "within")
  expect_true(all(vapply(wit, function(x) all(x %in% .NL_WITHIN_CELL),
                         logical(1))))
  # Every kind's default is the shipped read, so no support silently changes
  # construction.
  expect_true(all(vapply(wit, function(x) identical(x[1L], "chord"),
                         logical(1))))
  # Only a cell partition that TILES admits box-uniform.
  expect_true("box_uniform" %in% wit[["density"]])
  expect_false("box_uniform" %in% wit[["mixed"]])
  expect_false("box_uniform" %in% wit[["sample"]])
  expect_false("box_uniform" %in% wit[["moment_rule"]])
})

test_that("a sample support clamps and a density support extends", {
  v <- c(1, 2, 3, 4, 5)
  w <- rep(0.2, 5)
  p <- c(0.005, 0.025, 0.5, 0.975, 0.995)
  expect_identical(.nl_summary_quantile(v, w, p, "positive", "sample"),
                   .nl_wtd_quantile(v, w, p, outside = "clamp"))
  expect_identical(.nl_summary_quantile(v, w, p, "positive", "density"),
                   .nl_wtd_quantile(v, w, p, outside = "extend"))
  # The two differ only outside [w_1/2, 1 - w_n/2] = [0.1, 0.9] here: the sample
  # returns the extreme order statistic, the grid its outer cell's own edge.
  qs <- .nl_summary_quantile(v, w, p, "positive", "sample")
  qd <- .nl_summary_quantile(v, w, p, "positive", "density")
  expect_identical(qs[3L], qd[3L])
  expect_identical(unname(qs[c(1L, 2L)]), c(min(v), min(v)))
  expect_true(all(qd[c(1L, 2L)] < min(v)))
  expect_true(all(qd[c(1L, 2L)] > .nl_cell_edges(v)[1L]))
})

test_that("an unknown support kind is refused rather than translated", {
  v <- c(1, 2, 3); w <- rep(1 / 3, 3)
  expect_error(.nl_summary_quantile(v, w, 0.5, "positive", "draws"))
  expect_error(.nl_summary_quantile(v, w, 0.5, "positive", "mcmc"))
})

test_that(".nl_node_support reads the producer that left the node set", {
  expect_identical(.nl_node_support("grid"), "density")
  expect_identical(.nl_node_support("grid_adaptive"), "density")
  expect_identical(.nl_node_support("ccd"), "moment_rule")
  expect_identical(.nl_node_support("sample"), "sample")
  expect_identical(.nl_node_support(NULL), "density")
  # The per-cell tag still decides ahead of the producer name (gcol33/tulpa#317),
  # and a homogeneous tag still falls through to it.
  expect_identical(.nl_node_support("grid", c("mass", "design", "mass")),
                   "mixed")
  expect_identical(.nl_node_support("grid", rep("mass", 4L)), "density")
  expect_identical(.nl_node_support("ccd", rep("design", 4L)), "moment_rule")
  # A sample carries no quadrature design, so none of its weight is design mass.
  expect_identical(.nl_interval_provenance("sample")$read, "sample")
  expect_identical(.nl_interval_provenance("sample")$design_mass, 0)
})

# ------------------------------------------------- nothing reported moves

test_that("sample and density agree at every probability the backends report", {
  # The policies differ only outside [1 / (2n), 1 - 1 / (2n)]. The reported
  # levels are 0.025 / 0.5 / 0.975, so from n = 21 draws up the two reads are
  # identical -- which is why separating them is a refactor.
  set.seed(4L)
  probs <- c(0.025, 0.5, 0.975)
  for (n in c(21L, 50L, 400L, 2000L)) {
    v <- sort(exp(rnorm(n, -0.3, 0.5)))
    w <- rep(1 / n, n)
    expect_identical(.nl_summary_quantile(v, w, probs, "positive", "sample"),
                     .nl_summary_quantile(v, w, probs, "positive", "density"))
  }
  # Below that they part, and the sample read is the one with a derivation: at
  # n = 400 the grid read fabricates a stub past the largest draw.
  n <- 400L
  v <- sort(exp(rnorm(n, -0.3, 0.5)))
  w <- rep(1 / n, n)
  qs <- .nl_summary_quantile(v, w, 0.999, "positive", "sample")
  qd <- .nl_summary_quantile(v, w, 0.999, "positive", "density")
  expect_identical(unname(qs), max(v))
  expect_gt(qd, max(v))
})

test_that("the derived-summary read is unchanged at the reported probabilities", {
  layout <- list(list(nc = 2L, full = TRUE, k = 3L, label = "g",
                      n_groups = 40L))
  set.seed(9L)
  n <- 500L
  draws <- lapply(seq_len(n), function(i) {
    L <- matrix(c(exp(rnorm(1, -0.2, 0.3)), rnorm(1, 0.2, 0.3),
                  0, exp(rnorm(1, -0.4, 0.3))), 2, 2)
    list(tcrossprod(L))
  })
  w <- rep(1 / n, n)
  smp <- tulpa:::.re_cov_derived_summary(draws, w, layout, support = "sample")
  den <- tulpa:::.re_cov_derived_summary(draws, w, layout, support = "density")
  expect_identical(smp$posterior, den$posterior)
  expect_identical(smp$Sigma_mean, den$Sigma_mean)
  # And the sample read IS the sample quantile, which is what the Gibbs backend
  # has always relied on. `type = 5` is the convention `.nl_wtd_quantile()` uses
  # -- plotting positions `(i - 0.5) / n`, which is what `cumsum(w) - w / 2` is
  # at equal weight. The two conventions agree at the median, which is why
  # test-re-cov-nested.R's median-only pin passes against `type = 7`; they part
  # in the tails, and that difference is the convention, not the support kind.
  s1 <- vapply(draws, function(z) sqrt(z[[1L]][1, 1]), numeric(1))
  row <- smp$posterior[smp$posterior$parameter == "sigma_1", ]
  expect_equal(unname(c(row$ci_lo, row$median, row$ci_hi)),
               unname(stats::quantile(s1, c(0.025, 0.5, 0.975), type = 5)),
               tolerance = 1e-8)
})

# -------------------------------------------------------- on a real fit

test_that("a grid-integrated fit still reads its interval as a density", {
  skip_on_cran()
  skip_if_fast()
  set.seed(31L)
  G <- 40L; npg <- 12L; N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  X <- cbind(1, x); Z <- cbind(1, x)
  u <- matrix(rnorm(2 * G), G, 2) %*% diag(c(0.7, 0.5))
  y <- rbinom(N, 1L, stats::plogis(as.numeric(X %*% c(-0.3, 0.6)) +
                                     rowSums(Z * u[grp, ])))
  rt <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = Z)
  fit <- tulpa_re_cov_nested(y, rep(1L, N), X, rt, family = "binomial",
                             control = list(integration = "grid",
                                            diagnose_k = FALSE))
  expect_identical(.nl_node_support(fit$integration), "density")
  expect_true(all(is.finite(fit$posterior$median)))
  expect_true(all(fit$posterior$ci_lo <= fit$posterior$median + 1e-8))
})

test_that("a Gibbs fit says it produced a sample and reports the same numbers", {
  skip_if_not_slow()
  set.seed(17L)
  G <- 40L; npg <- 12L; N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  X <- cbind(1, x); Z <- cbind(1, x)
  u <- matrix(rnorm(2 * G), G, 2) %*% diag(c(0.7, 0.5))
  y <- rbinom(N, 1L, stats::plogis(as.numeric(X %*% c(-0.3, 0.6)) +
                                     rowSums(Z * u[grp, ])))
  rt <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = Z)
  fit <- tulpa_re_cov_gibbs(y, rep(1L, N), X, rt, family = "binomial",
                            control = list(n_iter = 800L, warmup = 400L,
                                           seed = 5L))
  # The producer names itself, so every downstream reader gets the right kind.
  expect_identical(fit$integration, "sample")
  expect_identical(.nl_node_support(fit$integration), "sample")

  # The reported quantiles are what the pre-#358 `"density"` read produced, on
  # this fit's own draws. This is the no-numerical-change assertion, shown
  # rather than assumed.
  layout <- list(list(nc = 2L, full = TRUE, k = 3L, label = "g",
                      n_groups = G))
  nodes <- lapply(fit$Sigma_draws, function(S) list(S))
  w <- rep(1 / fit$n_kept, fit$n_kept)
  den <- tulpa:::.re_cov_derived_summary(nodes, w, layout, support = "density")
  expect_identical(fit$posterior, den$posterior)

  # And it equals the plain sample quantile of the same derived draws, at the
  # `(i - 0.5) / n` plotting positions `.nl_wtd_quantile()` places.
  s1 <- vapply(fit$Sigma_draws, function(S) sqrt(S[1, 1]), numeric(1))
  row <- fit$posterior[fit$posterior$parameter == "sigma_1", ]
  expect_equal(unname(c(row$ci_lo, row$median, row$ci_hi)),
               unname(stats::quantile(s1, c(0.025, 0.5, 0.975), type = 5)),
               tolerance = 1e-8)
})
