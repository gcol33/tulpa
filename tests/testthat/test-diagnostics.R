# The diagnostics() front door: draws provenance selects which reliability
# question a fit is asked.
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

test_that("a chain-stamped fit with no draws says so instead of returning NULL silently", {
  fit <- structure(list(backend = "gibbs", draws_kind = "chain"),
                   class = "tulpa_fit")
  expect_message(res <- diagnostics(fit), "stamped as an MCMC chain")
  expect_null(res)
})

# --------------------------------------------------------------------------- #
# What the reliability table says it describes                                 #
# --------------------------------------------------------------------------- #

test_that("every backend emitting i.i.d. draws has a reliability scope entry", {
  iid <- names(Filter(function(e) identical(e$emits, "iid"), BACKEND_REGISTRY))
  expect_true(all(iid %in% names(.APPROX_SCOPE)),
              info = paste(setdiff(iid, names(.APPROX_SCOPE)), collapse = ", "))
  for (nm in names(.APPROX_SCOPE)) {
    e <- .APPROX_SCOPE[[nm]]
    expect_true(is.character(e$title) && nzchar(e$title), info = nm)
    expect_true(is.character(e$scope) && nzchar(e$scope), info = nm)
    expect_true(all(e$layers %in% c("outer", "inner")), info = nm)
  }
})

test_that("the header and scope follow the backend, not a nested-Laplace default", {
  set.seed(14)
  draws <- matrix(rnorm(400), ncol = 2, dimnames = list(NULL, c("b0", "b1")))
  shell <- function(backend, ...) {
    structure(list(draws = draws, draws_kind = "iid", backend = backend, ...),
              class = "tulpa_fit")
  }
  for (b in c("smc", "vi", "pathfinder", "ep")) {
    tab <- diagnostics(shell(b))
    out <- paste(capture.output(print(tab)), collapse = "\n")
    expect_match(out, .APPROX_SCOPE[[b]]$title, fixed = TRUE, info = b)
    expect_no_match(out, "Nested-Laplace|OUTER-integration|latent-field|outer PSIS",
                    info = b)
    expect_identical(attr(tab, "scope_backend"), b)
    expect_length(attr(tab, "scope_layers"), 0L)
    expect_true(is.na(attr(tab, "reliability")))
  }

  nested <- diagnostics(shell("re_cov_nested", weights = rep(0.25, 4),
                              pareto_k = 0.3, pareto_k_is_ess = 300,
                              pareto_k_scope = "outer"))
  out <- paste(capture.output(print(nested)), collapse = "\n")
  expect_match(out, "RE-covariance nested-Laplace OUTER-integration", fixed = TRUE)
  expect_match(out, "random-effect covariance Sigma", fixed = TRUE)
  expect_match(attr(nested, "reliability"), "outer integration good")

  # A fit with no registered backend is described by the layers it carries.
  expect_output(print(diagnostics(.diag_iid_fit())), "Nested-Laplace OUTER")
})

test_that("a zero-row draws matrix is read as no draws", {
  fit <- structure(list(draws = matrix(numeric(0), 0L, 2L,
                                       dimnames = list(NULL, c("a", "b"))),
                        draws_kind = "iid", backend = "agq",
                        means = c(a = 0, b = 1)),
                   class = "tulpa_fit")
  expect_message(res <- diagnostics(fit), "carries no posterior draws")
  expect_null(res)
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
