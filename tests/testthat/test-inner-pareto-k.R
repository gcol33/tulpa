# Inner-Laplace importance k-hat (gcol33/tulpa#303): the second score on the
# inner layer, and the one that needs no likelihood derivative -- so it answers
# for a fully coupled model where the cubic term gamma_3 declines permanently.
#
# The inner Gaussian at a fixed theta IS an importance proposal for the exact
# conditional posterior, so PSIS on that ratio scores the inner layer directly.
# It runs on the probed subspace, one dimension per probed index, along the
# same Gaussian-conditional-mean curve gamma_3 expands (src/inner_laplace_is.h).
#
# These are recovery tests, not shape tests. The two inner scores read the same
# misfit off the same curve, so on a fixture where BOTH are computable they must
# MOVE TOGETHER, and where only the k-hat is computable it is held against the
# exact quadrature the coupled fixture provides (helper-coupled-fixture.R).

# --------------------------------------------------------------------------- #
# (1) The shared IS-PSIS core accepts injected draws, exactly and without RNG   #
# --------------------------------------------------------------------------- #

test_that(".nested_is_pareto_k scores injected draws identically and consumes no RNG", {
  # A gaussian target narrower than the proposal: a well-behaved, finite k-hat.
  target <- function(U) -0.65 * rowSums(U^2)
  set.seed(7)
  drawn <- tulpa:::.nested_is_pareto_k(0, matrix(1, 1L, 1L), target,
                                       n_samples = 200L)
  set.seed(7)
  Z <- matrix(stats::rnorm(200L), ncol = 1L)
  before <- .Random.seed
  injected <- tulpa:::.nested_is_pareto_k(0, matrix(1, 1L, 1L), target,
                                          n_samples = 200L, Z = Z)
  # The injected path draws nothing: the fit's own stream is untouched.
  expect_identical(.Random.seed, before)
  expect_identical(injected$pareto_k, drawn$pareto_k)
  expect_identical(injected$is_ess, drawn$is_ess)

  # A mis-sized draw matrix is a bookkeeping fault, never a quiet resample.
  bad <- tulpa:::.nested_is_pareto_k(0, matrix(1, 1L, 1L), target,
                                     n_samples = 200L,
                                     Z = matrix(stats::rnorm(50L), ncol = 1L))
  expect_true(is.na(bad$pareto_k))
  expect_identical(bad$declined, "internal_inconsistency")
})

# --------------------------------------------------------------------------- #
# (2) Structural: summary, materiality gate, decline vocabulary, verdict        #
# --------------------------------------------------------------------------- #

test_that(".tulpa_inner_k_summary bands the worst MATERIAL index, not the worst index", {
  floor_ess <- tulpa:::.nl_diag("inner_k_material_ess")
  # A large shape whose weights are uniform describes no correction: reported,
  # but not banded. This is the guard against flagging a healthy fit.
  s_uniform <- tulpa:::.tulpa_inner_k_summary(c(0.9, 0.2), c(256, 256),
                                              c(1, 1))
  expect_equal(s_uniform$max_pareto_k, 0.9)
  expect_true(s_uniform$weights_uniform)
  expect_equal(s_uniform$n_material, 0L)
  expect_equal(s_uniform$band, "good")

  # The same shape on an index whose weights DO vary bands unreliable.
  s_mat <- tulpa:::.tulpa_inner_k_summary(c(0.9, 0.2), c(160, 256),
                                          c(0.62, 1))
  expect_false(s_mat$weights_uniform)
  expect_equal(s_mat$n_material, 1L)
  expect_equal(s_mat$band, "unreliable")
  expect_equal(s_mat$min_rel_ess, 0.62)

  # A backend reporting no efficiency at all is not silently waved through.
  s_noess <- tulpa:::.tulpa_inner_k_summary(c(0.9), NULL, NULL)
  expect_equal(s_noess$n_material, 1L)
  expect_equal(s_noess$band, "unreliable")

  # Boundary: the floor itself is NOT material (strict `<`).
  s_edge <- tulpa:::.tulpa_inner_k_summary(0.9, 256 * floor_ess, floor_ess)
  expect_true(s_edge$weights_uniform)

  # Nothing scored: NA band, never "good".
  s_none <- tulpa:::.tulpa_inner_k_summary(c(NA_real_, NaN))
  expect_equal(s_none$n_scored, 0L)
  expect_true(is.na(s_none$band))
  expect_null(tulpa:::.tulpa_inner_k_summary(NULL))
})

test_that("an inner-k decline carries a reason from the shared k vocabulary", {
  # `[[`, not `$`: on a declined fit `inner_pareto_k_declined` is the only field
  # carrying the prefix, so `$` would partial-match the k-hat to the reason.
  d <- tulpa:::.inner_k_decline(list(), "not_requested")
  expect_null(d[["inner_pareto_k"]])
  expect_identical(d[["inner_pareto_k_declined"]], "not_requested")
  # The reader must not fall into that trap either.
  expect_true(is.na(tulpa:::.tulpa_inner_k_reliability(d)$max_pareto_k))

  d2 <- tulpa:::.inner_k_decline(list(), "unguessable_axis", "rho_car")
  expect_identical(d2$inner_pareto_k_declined, "unguessable_axis: rho_car")

  # Every reason is one of the outer k-hat's, not a parallel vocabulary.
  for (r in tulpa:::.K_DECLINE_REASONS) {
    lbl <- tulpa:::.inner_k_decline(list(), r)[["inner_pareto_k_declined"]]
    expect_identical(sub(":.*$", "", lbl), r)
  }
  expect_error(tulpa:::.inner_k_decline(list(), "made_up_reason"))

  # A skew decline settles the k-hat too, mapped onto that vocabulary.
  s <- tulpa:::.inner_skew_decline(list(), "coupled_likelihood")
  expect_identical(s[["inner_skew_declined"]], "coupled_likelihood")
  expect_identical(sub(":.*$", "", s[["inner_pareto_k_declined"]]), "not_applicable")
  off <- tulpa:::.inner_skew_decline(list(), "not_requested")
  expect_identical(off[["inner_pareto_k_declined"]], "not_requested")

  # And each reads back as a sentence.
  for (r in c("not_requested", "not_applicable", "draws_too_few",
              "degenerate_proposal", "internal_inconsistency")) {
    expect_true(is.character(tulpa:::.inner_k_decline_note(r)))
  }
  expect_null(tulpa:::.inner_k_decline_note(NA_character_))
})

test_that("the inner k-hat gives a coupled fit an inner verdict instead of 'not assessed'", {
  # The #303 acceptance: gamma_3 declines structurally, the k-hat does not, so
  # the combined verdict names an assessed inner layer.
  before <- tulpa:::.tulpa_combined_reliability(
    "good", NA_character_, inner_declined = "coupled_arm")
  expect_match(before, "not assessed")

  after <- tulpa:::.tulpa_combined_reliability(
    "good", NA_character_, inner_declined = "coupled_arm",
    inner_k_band = "good")
  expect_match(after, "^reliable")
  expect_false(grepl("not assessed", after))

  flagged <- tulpa:::.tulpa_combined_reliability(
    "good", NA_character_, inner_declined = "coupled_arm",
    inner_k_band = "unreliable")
  expect_match(flagged, "^scoped: inner")

  # Where both scores exist the WORSE one governs, in both directions.
  expect_match(tulpa:::.tulpa_combined_reliability("good", "good",
                                                   inner_k_band = "unreliable"),
               "^scoped: inner")
  expect_match(tulpa:::.tulpa_combined_reliability("good", "unreliable",
                                                   inner_k_band = "good"),
               "^scoped: inner")
  expect_match(tulpa:::.tulpa_combined_reliability("good", "good",
                                                   inner_k_band = "good"),
               "^reliable")
  # Neither score: the cubic term's reason is what survives, unchanged.
  expect_match(tulpa:::.tulpa_combined_reliability("good", NA_character_,
                                                   inner_declined = "not_requested"),
               "not assessed")
})

# --------------------------------------------------------------------------- #
# (3) Recovery: quiet where the inner Laplace is analytically EXACT            #
# --------------------------------------------------------------------------- #

test_that("the inner importance weights are uniform for a gaussian fit (the Laplace is exact)", {
  skip_on_cran()
  # An identity-link gaussian log-likelihood is exactly quadratic in eta, so the
  # inner Gaussian IS the conditional posterior and the importance ratio is
  # constant along the probed curve. The reported efficiency must be 1 and the
  # band "good" -- the shape index itself is scale-free and reads a nonzero
  # value on a constant ratio, which is precisely why it is not banded alone.
  set.seed(11)
  n <- 300L
  x <- rnorm(n)
  y <- 1 + 0.5 * x + rnorm(n, 0, 1)
  fit <- tulpa:::cpp_laplace_fit(
    y = as.numeric(y), n = rep(1L, n), X = cbind(1, x),
    re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
    family = "gaussian", compute_skew = TRUE, skew_idx = as.integer(1:2))
  expect_equal(fit$inner_skew, c(0, 0))

  res <- tulpa:::.inner_k_attach(list(), fit)
  expect_true(is.na(res[["inner_pareto_k_declined"]]))
  expect_equal(res[["inner_pareto_k_rel_ess"]], c(1, 1))
  s <- tulpa:::.tulpa_inner_k_summary(res[["inner_pareto_k"]],
                                      res[["inner_pareto_k_is_ess"]],
                                      res[["inner_pareto_k_rel_ess"]])
  expect_true(s$weights_uniform)
  expect_equal(s$band, "good")

  # The log joint along the curve is exactly -z^2/2 + const, so the raw ratio is
  # constant to machine precision. This is the invariant the band gate rests on.
  lr <- fit$inner_is_log_joint[, 1] + 0.5 * fit$inner_is_z^2
  expect_lt(diff(range(lr)), 1e-8 * abs(mean(lr)))
})

# --------------------------------------------------------------------------- #
# (4) Cross-check: gamma_3 and the inner k-hat move together                   #
# --------------------------------------------------------------------------- #

test_that("the inner k-hat tracks gamma_3 across a binomial-intercept skewness ladder", {
  skip_on_cran()
  # Both scores read the same misfit off the same curve, so on a fixture where
  # both compute they must order the same way. The ladder runs from a balanced
  # binomial intercept (the inner Laplace is essentially exact) to a
  # one-success-in-fifteen fit (a textbook Wald-CI failure).
  cases <- list(c(N = 500, S = 230), c(N = 500, S = 60), c(N = 100, S = 3),
                c(N = 20, S = 2), c(N = 15, S = 1))
  got <- lapply(cases, function(cs) {
    N <- cs[["N"]]; S <- cs[["S"]]
    y <- c(rep(1, S), rep(0, N - S))
    fit <- tulpa:::cpp_laplace_fit(
      y = as.numeric(y), n = rep(1L, N), X = matrix(1, N, 1),
      re_idx = numeric(0), n_re_groups = 0L, sigma_re = 1.0,
      family = "binomial", compute_skew = TRUE, skew_idx = 1L)
    res <- tulpa:::.inner_k_attach(list(), fit)
    s <- tulpa:::.tulpa_inner_k_summary(res[["inner_pareto_k"]],
                                        res[["inner_pareto_k_is_ess"]],
                                        res[["inner_pareto_k_rel_ess"]])
    list(gamma3 = abs(fit$inner_skew[1]),
         gamma3_band = tulpa:::.tulpa_gamma3_band(fit$inner_skew[1]),
         k = res$inner_pareto_k[1],
         rel_ess = res$inner_pareto_k_rel_ess[1],
         band = s$band, uniform = s$weights_uniform)
  })
  g3   <- vapply(got, function(g) g$gamma3, numeric(1))
  kh   <- vapply(got, function(g) g$k, numeric(1))
  rel  <- vapply(got, function(g) g$rel_ess, numeric(1))
  expect_true(all(is.finite(g3)), info = "every ladder rung scores gamma_3")
  expect_true(all(is.finite(kh)), info = "every ladder rung scores the k-hat")

  # The ladder IS a skewness ladder: gamma_3 increases monotonically along it,
  # which is what makes an ordering claim about the k-hat meaningful.
  expect_true(all(diff(g3) > 0))

  # The correction the importance sampler has to make tracks the skewness
  # EXACTLY: efficiency falls monotonically as gamma_3 rises.
  expect_true(all(diff(rel) < 0))
  expect_equal(unname(stats::cor(g3, 1 - rel, method = "spearman")), 1)

  # The tail SHAPE tracks it too, up to its own noise on the two rungs where
  # the proposal needs essentially no correction.
  expect_gte(unname(stats::cor(g3, kh, method = "spearman")), 0.8)

  # And the two scores agree on the verdict, rung by rung: every rung gamma_3
  # calls "good" has uniform importance weights, and every rung it calls off
  # "good" has a material correction the k-hat bands off "good" too.
  for (g in got) {
    if (identical(g$gamma3_band, "good")) {
      expect_true(g$uniform)
      expect_equal(g$band, "good")
    } else {
      expect_false(g$uniform)
      expect_false(identical(g$band, "good"))
    }
  }
})

# --------------------------------------------------------------------------- #
# (5) The coupled fixture: the k-hat answers where gamma_3 cannot              #
# --------------------------------------------------------------------------- #

test_that("the inner k-hat scores the coupled fixture gamma_3 declines on", {
  skip_on_cran()
  # The whole point of #303. Every arm of this fit is coupled through a
  # CellCouplingSpec, so gamma_3 has no separable per-observation sum to read
  # and declines permanently -- yet the inner layer still gets a number, and
  # that number is checked against the fixture's exact quadrature.
  coupled_occ_register()
  beta_prec <- 0.25
  d <- coupled_occ_data(seed = 311, n_cells = 100L, n_visits = 4L,
                        b_occ = 0.2, b_det = -0.5)
  fit <- tulpa_nested_laplace_joint(
    responses = coupled_occ_arms(d, beta_prec = beta_prec),
    prior = coupled_occ_flat_prior(d),
    cell_coupling = "test_occupancy_mixture",
    control = list(max_iter = 300L, tol = 1e-12, diagnose_k = FALSE))

  # The cubic term declines, structurally.
  expect_true(all(is.nan(fit$inner_skew)))
  expect_identical(fit$inner_skew_declined, "coupled_arm")

  # The importance k-hat does not.
  expect_length(fit[["inner_pareto_k"]], sum(fit$arm_layout$p))
  expect_true(all(is.finite(fit[["inner_pareto_k"]])))
  expect_true(is.na(fit[["inner_pareto_k_declined"]]))
  rel_all <- fit[["inner_pareto_k_rel_ess"]]
  expect_true(all(rel_all > 0 & rel_all <= 1))

  # Held against ground truth: the fixture's exact marginal skewness comes from
  # a direct two-dimensional quadrature of the same posterior (no Laplace
  # anywhere). The arm whose exact posterior is more skewed is the arm whose
  # inner Gaussian needs the larger importance correction.
  lp <- coupled_occ_log_post(d, beta_prec)
  ctr <- stats::optim(c(0, 0), function(v) -lp(v[1L], v[2L]),
                      method = "BFGS", control = list(reltol = 1e-14))$par
  q <- coupled_occ_quadrature(lp, ctr, half = 16, n_grid = 1601L)
  exact <- c(abs(q$a[["skew"]]), abs(q$b[["skew"]]))
  expect_gt(exact[1], exact[2])                       # occupancy is the skewed arm
  expect_lt(rel_all[which.max(exact)], rel_all[which.min(exact)])

  # The fit's whole-fit verdict now names an assessed inner layer.
  ikk <- tulpa:::.tulpa_inner_k_reliability(fit)
  verdict <- tulpa:::.tulpa_combined_reliability(
    "good", NA_character_, inner_declined = fit$inner_skew_declined,
    inner_k_band = ikk$band, inner_k_declined = ikk$declined)
  expect_false(grepl("not assessed", verdict))
})

# --------------------------------------------------------------------------- #
# (6) Front-door wiring, and the RNG contract                                  #
# --------------------------------------------------------------------------- #

.ikk_chain_prior <- function(n_s, idx) {
  nbr <- lapply(seq_len(n_s), function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(type = "icar", n_spatial_units = n_s, spatial_idx = idx,
       adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn), tau_grid = c(0.5, 1, 2, 4, 8))
}

test_that("tulpa_nested_laplace() reports the inner k-hat and leaves the draws untouched", {
  skip_on_cran()
  set.seed(14)
  n_s <- 12L
  idx <- rep(seq_len(n_s), each = 10L)
  n <- length(idx)
  x <- rnorm(n)
  field <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.4))))
  y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
  prior <- .ikk_chain_prior(n_s, idx)

  set.seed(99)
  on_fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                                 family = "binomial")
  seed_on <- .Random.seed
  set.seed(99)
  off_fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                                  family = "binomial",
                                  control = list(diagnose_skew = FALSE))
  seed_off <- .Random.seed

  # Default probe scope is the fixed-effects coefficients, same as gamma_3.
  expect_length(on_fit$inner_pareto_k, 2L)
  expect_true(all(is.finite(on_fit$inner_pareto_k)))
  expect_true(is.na(on_fit$inner_pareto_k_declined))

  # The diagnostic consumes no randomness: the RNG stream and every non-
  # diagnostic field of the fit are bit-for-bit identical with it on and off.
  expect_identical(seed_on, seed_off)
  # `timing` is wall clock, not model output.
  strip <- function(f) f[!grepl("^inner_|^timing$", names(f))]
  expect_identical(strip(on_fit), strip(off_fit))
  expect_true(length(strip(on_fit)) > 5L)   # the comparison is not vacuous

  # And switching it off is a decline with a reason, never an absent field.
  expect_null(off_fit[["inner_pareto_k"]])
  expect_identical(off_fit[["inner_pareto_k_declined"]], "not_requested")

  # An explicit probe scope narrows both inner scores together.
  one <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                              family = "binomial",
                              control = list(skew_idx = 1L))
  expect_length(one[["inner_pareto_k"]], 1L)
})

test_that("diagnostics() surfaces the inner k-hat when the cubic term declined", {
  skip_on_cran()
  # A minimal i.i.d. shell (mirrors test-inner-skew.R): isolates the reporting
  # layer from the kernel already covered above.
  draws <- matrix(rnorm(400), ncol = 2, dimnames = list(NULL, c("b0", "b1")))
  fit <- structure(list(
    draws = draws, draws_kind = "iid",
    joint_fit = list(weights = rep(1, 5) / 5, pareto_k = 0.3,
                     pareto_k_is_ess = 400, pareto_k_scope = "outer",
                     inner_skew = c(NaN, NaN), inner_skew_idx = c(1L, 2L),
                     inner_skew_dropped = 0L,
                     inner_skew_declined = "coupled_arm",
                     inner_pareto_k = c(0.35, 0.51),
                     inner_pareto_k_is_ess = c(251, 255),
                     inner_pareto_k_rel_ess = c(0.98, 0.996))
  ), class = "tulpa_fit")

  tab <- tulpa:::.tulpa_approx_diag_table(fit)
  expect_equal(attr(tab, "inner_pareto_k"), 0.51)
  expect_equal(attr(tab, "inner_pareto_k_band"), "good")
  expect_true(is.na(attr(tab, "inner_skew_band")))
  # The cubic term declined but the layer IS assessed.
  expect_false(grepl("not assessed", attr(tab, "reliability")))
  expect_output(print(tab), "importance pareto_k")
  expect_equal(attr(tab, "summary")$inner_pareto_k, 0.51)

  # A fit that computed neither inner score still says so.
  fit2 <- fit
  fit2$joint_fit$inner_pareto_k <- NULL
  fit2$joint_fit$inner_pareto_k_is_ess <- NULL
  fit2$joint_fit$inner_pareto_k_rel_ess <- NULL
  fit2$joint_fit$inner_pareto_k_declined <- "not_requested"
  tab2 <- tulpa:::.tulpa_approx_diag_table(fit2)
  expect_match(attr(tab2, "reliability"), "not assessed")
  expect_output(print(tab2), "importance pareto_k = NA")
})
