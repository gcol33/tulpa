# The diagnostics() front door: draws provenance selects which reliability
# question a fit is asked, and the two superseded entry points keep returning
# exactly what they always did.
#
# The routing is the contract worth pinning. A fit whose draws are i.i.d. must
# NOT come back with a chain-mixing verdict: split-Rhat sits at ~1 and ESS ~
# n_draws on i.i.d. draws by construction, so a Rhat table there reads as a
# clean convergence pass while saying nothing about approximation bias.

.diag_chain_fit <- function(n = 400L, p = 2L, seed = 11) {
  set.seed(seed)
  draws <- matrix(rnorm(n * p), n, p,
                  dimnames = list(NULL, letters[seq_len(p)]))
  structure(list(draws = draws, draws_kind = "chain"), class = "tulpa_fit")
}

.diag_iid_fit <- function(n = 800L, p = 3L, seed = 12, k = 0.31) {
  set.seed(seed)
  draws <- matrix(rnorm(n * p), n, p,
                  dimnames = list(NULL, letters[seq_len(p)]))
  structure(list(
    draws = draws, draws_kind = "iid",
    joint_fit = list(weights = c(0.5, 0.3, 0.2), pareto_k = k,
                     pareto_k_is_ess = 700,
                     pareto_k_scope = "outer (hyperparameter) Gaussian proposal")
  ), class = "tulpa_fit")
}

# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #

test_that("a chain fit gets chain diagnostics, an i.i.d. fit gets reliability", {
  ch <- diagnostics(.diag_chain_fit())
  expect_false(inherits(ch, "laplace_diagnostics"))
  expect_setequal(names(ch), c("parameter", "rhat", "ess_bulk", "ess_tail"))

  ap <- diagnostics(.diag_iid_fit())
  expect_s3_class(ap, "laplace_diagnostics")
  expect_equal(attr(ap, "pareto_k"), 0.31)
  # the reliability headline, not a convergence verdict
  expect_true("pareto_k" %in% names(attributes(ap)))
})

test_that("a point fit returns NULL with a message naming the backend", {
  fit <- structure(list(draws_kind = "point", backend = "laplace"),
                   class = "tulpa_fit")
  expect_message(res <- diagnostics(fit), "point summary")
  expect_null(res)
})

test_that("an untagged fit is treated as a chain rather than refused", {
  set.seed(13)
  draws <- matrix(rnorm(400 * 2), 400, 2, dimnames = list(NULL, c("a", "b")))
  fit <- structure(list(draws = draws), class = "tulpa_fit")   # no draws_kind
  res <- diagnostics(fit)
  expect_true("rhat" %in% names(res))
  expect_equal(nrow(res), 2L)
})

test_that("provenance is read from the backend registry when untagged", {
  set.seed(14)
  draws <- matrix(rnorm(600 * 2), 600, 2, dimnames = list(NULL, c("a", "b")))
  # `smc` emits i.i.d. draws per BACKEND_REGISTRY; no explicit draws_kind here.
  fit <- structure(list(draws = draws, backend = "smc"), class = "tulpa_fit")
  expect_s3_class(diagnostics(fit), "laplace_diagnostics")
})

test_that("an unrecognised provenance kind fails loudly", {
  fit <- structure(list(draws = matrix(rnorm(20), 10, 2),
                        draws_kind = "quantum"), class = "tulpa_fit")
  expect_error(diagnostics(fit), "Unknown draws provenance")
})

# --------------------------------------------------------------------------- #
# Argument pass-through                                                        #
# --------------------------------------------------------------------------- #

test_that("measures and probs select columns on the chain route", {
  res <- diagnostics(.diag_chain_fit(),
                     measures = c("rhat_bulk", "mcse_mean", "ess_quantile"),
                     probs = c(0.1, 0.9))
  expect_setequal(names(res),
                  c("parameter", "rhat_bulk", "mcse_mean", "ess_q10", "ess_q90"))
})

test_that("pars restricts both routes", {
  expect_equal(diagnostics(.diag_chain_fit(p = 3L), pars = "b")$parameter, "b")
  expect_equal(diagnostics(.diag_iid_fit(), pars = c("a", "c"))$parameter,
               c("a", "c"))
})

# --------------------------------------------------------------------------- #
# Superseded entry points                                                      #
# --------------------------------------------------------------------------- #

test_that("both deprecated names warn", {
  # "warning" defeats deprecate_warn()'s once-per-session throttle, so the
  # signal is asserted rather than depending on test execution order.
  old <- options(lifecycle_verbosity = "warning")
  on.exit(options(old), add = TRUE)
  expect_warning(mcmc_diagnostics(.diag_chain_fit()), "deprecated")
  expect_warning(laplace_diagnostics(.diag_iid_fit()), "deprecated")
})

test_that("both deprecated names return exactly the diagnostics() value", {
  old <- options(lifecycle_verbosity = "quiet")
  on.exit(options(old), add = TRUE)
  ch <- .diag_chain_fit(); ap <- .diag_iid_fit()
  expect_equal(mcmc_diagnostics(ch), diagnostics(ch))
  expect_equal(laplace_diagnostics(ap), diagnostics(ap))
})

test_that("the deprecated names still route by provenance, not by their name", {
  old <- options(lifecycle_verbosity = "quiet")
  on.exit(options(old), add = TRUE)
  # mcmc_diagnostics() on an i.i.d. fit must still yield the reliability table,
  # which is the behaviour that made the old name wrong in the first place.
  expect_s3_class(mcmc_diagnostics(.diag_iid_fit()), "laplace_diagnostics")
})
