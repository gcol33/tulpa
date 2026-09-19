# plot_diagnostics() had no draws-provenance gate: plot_rhat() / plot_ess() /
# geweke_test() message and return on a non-chain fit, but plot_diagnostics()
# went on to pick a trace parameter from colnames(fit$draws)[1], which is NULL
# on a draw-less fit, and errored "argument is of length zero" testing NULL
# %in% dimnames(...) (gcol33/tulpa#772). No test file referenced this
# function before.

test_that("plot_diagnostics() gates on a non-chain fit instead of erroring", {
  skip_on_cran()
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("patchwork")
  set.seed(1)
  d <- data.frame(x = rnorm(200)); d$y <- rpois(200, exp(0.3 + 0.4 * d$x))
  f_lap <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  expect_false(tulpa:::.tulpa_is_chain(f_lap))
  expect_message(res <- plot_diagnostics(f_lap), "not an MCMC chain")
  expect_null(res)
})

test_that("plot_diagnostics() still builds a panel for a chain fit", {
  skip_on_cran()
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("patchwork")
  set.seed(1)
  d <- data.frame(x = rnorm(200)); d$y <- rpois(200, exp(0.3 + 0.4 * d$x))
  f_hmc <- tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
                 control = list(n_iter = 200L, warmup = 100L, n_chains = 2L,
                                seed = 1L))
  expect_true(tulpa:::.tulpa_is_chain(f_hmc))
  expect_silent(res <- plot_diagnostics(f_hmc))
  expect_s3_class(res, "patchwork")
})

# plot_pairs() / plot_divergences() read fit$draws directly, gated only on
# is.null(): an agq fit's $draws is a genuine 0 x p matrix (not NULL, "no
# draws" is read through .fit_draws()), so both functions ran on through to
# base graphics with an empty data frame and errored (gcol33/tulpa#782).
test_that("plot_pairs() gates on a fit with a zero-row draws matrix", {
  skip_on_cran()
  set.seed(1); n <- 160; G <- 12
  x <- rnorm(n); g <- factor(sample(seq_len(G), n, replace = TRUE))
  b0 <- rnorm(G, 0, 0.5); b1 <- rnorm(G, 0, 0.3)
  d <- data.frame(x = x, g = g,
                   y = rbinom(n, 1, plogis(-0.3 + x + b0[g] + b1[g] * x)))
  fa <- tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = "agq")

  expect_equal(nrow(fa$draws), 0L)
  expect_null(tulpa:::.fit_draws(fa))
  expect_message(res <- plot_pairs(fa), "carries no posterior draws")
  expect_null(res)
  # agq is not the hmc backend, so plot_divergences() declines on that gate
  # first; its own zero-row-draws gate is exercised directly below.
  expect_message(res2 <- plot_divergences(fa), "only available for HMC")
  expect_null(res2)
})

test_that("plot_divergences() reads draws through .fit_draws() past its hmc gate", {
  # A zero-row $draws that reaches plot_divergences()'s own draws read (past
  # the backend and divergent-transition gates) declines the same way
  # plot_pairs() does, rather than handing an empty frame to base graphics.
  skip_on_cran()
  fake <- structure(
    list(backend = "hmc", diagnostics = list(divergent_idx = 1L),
         draws = matrix(numeric(0), 0, 2, dimnames = list(NULL, c("a", "b")))),
    class = "tulpa_fit"
  )
  expect_equal(tulpa:::n_divergent(fake), 1L)
  expect_null(tulpa:::.fit_draws(fake))
  expect_message(res <- plot_divergences(fake), "carries no posterior draws")
  expect_null(res)
})

test_that("the divergence count and the divergent rows come from one record", {
  # plot_divergences() asks for the count and then for the indices. Reading
  # them off different fields let it believe divergences exist while reporting
  # their indices unavailable (gcol33/tulpa#840), so both go through
  # .tulpa_divergence_record(): on any fit that records divergences at all,
  # n_divergent() equals the length of the rows the plot would mark.
  shapes <- list(
    flat        = list(divergent = c(FALSE, TRUE, FALSE, TRUE)),
    diag_flags  = list(diagnostics = list(divergent = c(TRUE, FALSE, TRUE))),
    diag_idx    = list(diagnostics = list(divergent_idx = c(2L, 5L))),
    diag_count  = list(diagnostics = list(n_divergent = 3L)),
    flat_count  = list(n_divergent = 7L),
    none        = list()
  )
  expected <- c(flat = 2L, diag_flags = 2L, diag_idx = 2L,
                diag_count = 3L, flat_count = 7L, none = 0L)
  has_idx <- c(flat = TRUE, diag_flags = TRUE, diag_idx = TRUE,
               diag_count = FALSE, flat_count = FALSE, none = FALSE)

  for (nm in names(shapes)) {
    fit <- structure(c(list(backend = "hmc"), shapes[[nm]]), class = "tulpa_fit")
    idx <- tulpa:::.tulpa_divergent_idx(fit)
    expect_equal(tulpa:::n_divergent(fit), expected[[nm]], info = nm)
    expect_equal(!is.null(idx), has_idx[[nm]], info = nm)
    # Where indices exist at all, they account for every counted divergence.
    if (!is.null(idx)) expect_length(idx, expected[[nm]])
  }

  # And a fit whose indices are locatable never reports them unavailable.
  fit <- structure(list(backend = "hmc", divergent = c(FALSE, TRUE),
                        draws = matrix(0, 2, 1, dimnames = list(NULL, "a"))),
                   class = "tulpa_fit")
  expect_equal(tulpa:::.tulpa_divergent_idx(fit), 2L)
})
