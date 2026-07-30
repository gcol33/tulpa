# An explicit `mode` that the structure redirects away from (gcol33/tulpa#266).
# The fit does what the model requires, but it says so: a silent downgrade left
# the fit indistinguishable from one that was never asked for a mode, and
# `selection_reason` reported the redirect as though `mode` had been "auto".
#
# The notice is a warning() rather than a message() on purpose -- the reported
# case was a script that promotes warnings, which never saw that its requested
# inference method had been swapped.

make_smooth_re_data <- function(seed = 1L, G = 8L, npg = 12L) {
  set.seed(seed)
  n <- G * npg
  site <- factor(rep(seq_len(G), each = npg))
  x <- runif(n, -3, 3)
  b <- rnorm(G, 0, 0.6)
  data.frame(y = rpois(n, exp(sin(x) + b[as.integer(site)])), x = x, site = site)
}

# Fit, capturing warnings instead of letting testthat's own handlers eat them.
fit_quietly <- function(...) {
  w <- character()
  fit <- withCallingHandlers(
    suppressMessages(tulpa(...)),
    warning = function(cond) {
      w <<- c(w, conditionMessage(cond))
      invokeRestart("muffleWarning")
    })
  list(fit = fit, warnings = w)
}


# --- tier 1: the redirect bookkeeping, no fitting -----------------------------

test_that(".sel_redirect records an override only off an explicit selection", {
  expl <- list(backend = "eb", mode = "structured", requested = "eb",
               explicit = TRUE, reason = "User-specified backend: eb")
  out <- tulpa:::.sel_redirect(expl, "nested_laplace", "smoother present")
  expect_equal(out$backend, "nested_laplace")
  expect_equal(out$overridden$requested, "eb")
  expect_equal(out$overridden$backend, "eb")
  expect_match(out$reason, "^smoother present")
  expect_match(out$reason, "overrides the requested mode = 'eb'")

  auto <- list(backend = "mala", mode = "exact", requested = "auto",
               explicit = FALSE, reason = "default")
  out2 <- tulpa:::.sel_redirect(auto, "nested_laplace", "smoother present")
  expect_null(out2$overridden)
  expect_equal(out2$reason, "smoother present")
})


test_that(".sel_redirect keeps the first override across a redirect chain", {
  sel <- list(backend = "laplace", mode = "structured", requested = "structured",
              explicit = TRUE, reason = "User-specified mode: structured")
  a <- tulpa:::.sel_redirect(sel, "re_cov_nested", "slopes present")
  b <- tulpa:::.sel_redirect(a, "nested_laplace", "smoother present")
  # The request the user actually made, not the intermediate backend a previous
  # redirect chose.
  expect_equal(b$overridden$requested, "structured")
  expect_equal(b$overridden$backend, "laplace")
  # Each redirect REPLACES the reason, so the clause has to be re-appended to the
  # new one -- the second redirect must not leave the fit looking un-overridden.
  expect_match(b$reason, "^smoother present")
  expect_match(b$reason, "overrides the requested mode = 'structured'")
  # Stated once, not once per redirect in the chain.
  expect_equal(length(gregexpr("overrides", b$reason)[[1L]]), 1L)
})


test_that(".sel_redirect names mode and backend separately only when they differ", {
  tier <- tulpa:::.sel_redirect(
    list(backend = "laplace", mode = "structured", requested = "structured",
         explicit = TRUE, reason = "r"), "nested_laplace", "smoother")
  expect_match(tier$reason, "mode = 'structured' \\(backend 'laplace'\\)")

  named <- tulpa:::.sel_redirect(
    list(backend = "eb", mode = "structured", requested = "eb",
         explicit = TRUE, reason = "r"), "nested_laplace", "smoother")
  expect_match(named$reason, "mode = 'eb'")
  expect_false(grepl("backend 'eb'", named$reason))
})


test_that("a documented route records the override without warning about it", {
  # notify = FALSE is for a redirect off a request the structure cannot express
  # at all -- a random slope has no scalar sigma_re for mode = "laplace" to
  # condition on. The fit still records it; it just does not warn on every fit.
  sel <- list(backend = "laplace", mode = "structured", requested = "laplace",
              explicit = TRUE, reason = "r")
  out <- tulpa:::.sel_redirect(sel, "re_cov_nested", "slopes present",
                               notify = FALSE)
  expect_false(out$overridden$notify)
  expect_match(out$reason, "overrides the requested mode = 'laplace'")
})


test_that("a redirect that lands on the same backend is not an override", {
  sel <- list(backend = "nested_laplace", mode = "structured",
              requested = "nested_laplace", explicit = TRUE, reason = "r")
  out <- tulpa:::.sel_redirect(sel, "nested_laplace", "smoother present")
  expect_null(out$overridden)
})


# --- tier 2: the front door ---------------------------------------------------

test_that("an explicit mode overridden by the smoother redirect warns and records it", {
  skip_on_cran()
  d <- make_smooth_re_data()
  # No `sigma_re`: supplying it to mode = "eb" warns on its own (EB estimates the
  # covariance rather than conditioning on it), which would be a second, unrelated
  # warning in this assertion.
  res <- fit_quietly(y ~ s(x, k = 8) + (1 | site), data = d, family = "poisson",
                     mode = "eb")

  expect_equal(res$fit$backend, "nested_laplace")
  expect_length(res$warnings, 1L)
  # The warning names what was asked for, what ran, and why.
  expect_match(res$warnings, "mode = 'eb' was overridden")
  expect_match(res$warnings, "backend 'nested_laplace'")
  expect_match(res$warnings, "smoother")

  # Machine-readable, not only prose.
  expect_equal(res$fit$mode_overridden$requested, "eb")
  expect_equal(res$fit$mode_overridden$backend, "eb")
  # And the fit itself carries the statement for anyone reading it later.
  expect_match(res$fit$selection_reason, "overrides the requested mode = 'eb'")
})


test_that("a tier mode reports both the mode asked for and the backend it resolved to", {
  skip_on_cran()
  d <- make_smooth_re_data()
  res <- fit_quietly(y ~ s(x, k = 8) + (1 | site), data = d, family = "poisson",
                     mode = "structured")
  expect_equal(res$fit$mode_overridden$requested, "structured")
  expect_equal(res$fit$mode_overridden$backend, "laplace")
  expect_match(res$fit$selection_reason,
               "mode = 'structured' \\(backend 'laplace'\\)")
})


test_that("mode = 'auto' is silent: nothing was overridden", {
  skip_on_cran()
  d <- make_smooth_re_data()
  res <- fit_quietly(y ~ s(x, k = 8) + (1 | site), data = d, family = "poisson")
  expect_equal(res$fit$backend, "nested_laplace")
  expect_length(res$warnings, 0L)
  expect_null(res$fit$mode_overridden)
  expect_false(grepl("overrides", res$fit$selection_reason))
})


test_that("naming the backend the structure needs is honored without a warning", {
  skip_on_cran()
  d <- make_smooth_re_data()
  res <- fit_quietly(y ~ s(x, k = 8) + (1 | site), data = d, family = "poisson",
                     mode = "nested_laplace")
  expect_length(res$warnings, 0L)
  expect_null(res$fit$mode_overridden)
})


test_that("a mode the smoother path cannot reach at all still errors, not warns", {
  skip_on_cran()
  d <- make_smooth_re_data()
  # The ModelData samplers do not thread smoother blocks; dropping the terms
  # silently is the one outcome worse than refusing.
  expect_error(
    suppressMessages(tulpa(y ~ s(x, k = 8) + (1 | site), data = d,
                           family = "poisson", mode = "vi")),
    "not threaded through the ModelData samplers")
})
