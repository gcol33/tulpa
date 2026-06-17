# Convergence diagnostics: native estimators must reproduce posterior's
# rhat / ess_* / mcse_* (acceptance criterion of gcol33/tulpa#26), and the
# plotting / summary layer must run end-to-end on a multi-chain fit.

# Deterministic AR(1) draws -> [iter, chain, param] array.
make_draws_array <- function(niter = 1000L, nchain = 4L, npar = 3L,
                             phi = 0.6, seed = 20260524L) {
  set.seed(seed)
  pn <- paste0("theta", seq_len(npar))
  arr <- array(0, dim = c(niter, nchain, npar),
               dimnames = list(NULL, NULL, pn))
  for (m in seq_len(nchain)) {
    for (p in seq_len(npar)) {
      e <- rnorm(niter)
      x <- numeric(niter)
      x[1L] <- e[1L]
      for (t in 2:niter) x[t] <- phi * x[t - 1L] + e[t]
      arr[, m, p] <- x + (p - 2) * 0.4 + (m - 1) * 0.03   # small chain offset
    }
  }
  arr
}

# Pool a 3D array into a chain-major [iter*chain x param] matrix.
pool_chain_major <- function(arr) {
  niter <- dim(arr)[1L]; nchain <- dim(arr)[2L]; pn <- dimnames(arr)[[3L]]
  do.call(rbind, lapply(seq_len(nchain), function(m)
    matrix(arr[, m, ], nrow = niter, dimnames = list(NULL, pn))))
}


test_that("mcmc_diagnostics reproduces posterior rhat / ess / mcse", {
  skip_if_not_installed("posterior")
  arr <- make_draws_array()
  fit <- structure(list(draws = arr), class = "tulpa_fit")

  probs <- c(0.05, 0.5, 0.95)
  diag <- mcmc_diagnostics(
    fit,
    measures = c("rhat", "ess_bulk", "ess_tail", "ess_mean", "ess_sd",
                 "mcse_mean", "mcse_sd", "ess_quantile", "mcse_quantile"),
    probs = probs
  )

  expect_equal(nrow(diag), dim(arr)[3L])
  expect_true(all(c("rhat", "ess_bulk", "ess_tail", "ess_mean", "ess_sd",
                    "mcse_mean", "mcse_sd",
                    "ess_q5", "ess_q50", "ess_q95",
                    "mcse_q5", "mcse_q50", "mcse_q95") %in% names(diag)))

  for (p in seq_len(dim(arr)[3L])) {
    x <- arr[, , p]
    nm <- dimnames(arr)[[3L]][p]
    row <- diag[diag$parameter == nm, ]

    expect_equal(row$rhat,      posterior::rhat(x),      tolerance = 1e-4)
    expect_equal(row$ess_bulk,  posterior::ess_bulk(x),  tolerance = 1e-4)
    expect_equal(row$ess_tail,  posterior::ess_tail(x),  tolerance = 1e-4)
    expect_equal(row$ess_mean,  posterior::ess_mean(x),  tolerance = 1e-4)
    expect_equal(row$ess_sd,    posterior::ess_sd(x),    tolerance = 1e-4)
    expect_equal(row$mcse_mean, posterior::mcse_mean(x), tolerance = 1e-4)
    expect_equal(row$mcse_sd,   posterior::mcse_sd(x),   tolerance = 1e-4)
    expect_equal(row$ess_q5,   posterior::ess_quantile(x, 0.05, names = FALSE),  tolerance = 1e-4)
    expect_equal(row$ess_q95,  posterior::ess_quantile(x, 0.95, names = FALSE),  tolerance = 1e-4)
    expect_equal(row$mcse_q5,  posterior::mcse_quantile(x, 0.05, names = FALSE), tolerance = 1e-4)
    expect_equal(row$mcse_q95, posterior::mcse_quantile(x, 0.95, names = FALSE), tolerance = 1e-4)
  }
})


test_that("rhat takes the max of bulk and folded split-Rhat", {
  skip_if_not_installed("posterior")
  arr <- make_draws_array()
  fit <- structure(list(draws = arr), class = "tulpa_fit")
  diag <- mcmc_diagnostics(fit, measures = c("rhat", "rhat_bulk", "rhat_fold"))

  # headline rhat is never below either component, and equals their max
  expect_true(all(diag$rhat >= diag$rhat_bulk - 1e-8))
  expect_true(all(diag$rhat >= diag$rhat_fold - 1e-8))
  expect_equal(diag$rhat, pmax(diag$rhat_bulk, diag$rhat_fold), tolerance = 1e-10)

  for (p in seq_len(dim(arr)[3L])) {
    nm <- dimnames(arr)[[3L]][p]
    expect_equal(diag$rhat[diag$parameter == nm],
                 posterior::rhat(arr[, , p]), tolerance = 1e-4)
  }
})


test_that("default mcmc_diagnostics columns are stable", {
  arr <- make_draws_array(niter = 200L, nchain = 2L, npar = 2L)
  fit <- structure(list(draws = arr), class = "tulpa_fit")
  diag <- mcmc_diagnostics(fit)
  expect_identical(names(diag), c("parameter", "rhat", "ess_bulk", "ess_tail"))
})


test_that("constant parameters yield NA diagnostics, not errors", {
  arr <- make_draws_array(niter = 200L, nchain = 2L, npar = 2L)
  arr[, , 2L] <- 3.5                                   # constant parameter
  fit <- structure(list(draws = arr), class = "tulpa_fit")
  diag <- mcmc_diagnostics(fit)
  const_row <- diag[diag$parameter == "theta2", ]
  expect_true(is.na(const_row$rhat))
  expect_true(is.na(const_row$ess_bulk))
})


test_that("tulpa_draws_array recognises all chain layouts", {
  arr <- make_draws_array(niter = 100L, nchain = 3L, npar = 2L)

  # (a) already a 3D array
  a3 <- tulpa_draws_array(structure(list(draws = arr), class = "tulpa_fit"))
  expect_equal(dim(a3), c(100L, 3L, 2L))
  expect_identical(dimnames(a3)[[3L]], c("theta1", "theta2"))

  # (b) pooled chain-major matrix + n_chains
  pooled <- pool_chain_major(arr)
  a_nc <- tulpa_draws_array(
    structure(list(draws = pooled, n_chains = 3L), class = "tulpa_fit"))
  expect_equal(dim(a_nc), c(100L, 3L, 2L))
  expect_equal(a_nc[, 1L, "theta1"], arr[, 1L, "theta1"], tolerance = 1e-12)

  # (c) pooled matrix + chain_id row map
  cid <- rep(seq_len(3L), each = 100L)
  a_cid <- tulpa_draws_array(
    structure(list(draws = pooled, chain_id = cid), class = "tulpa_fit"))
  expect_equal(dim(a_cid), c(100L, 3L, 2L))

  # (d) single pooled chain
  a1 <- tulpa_draws_array(
    structure(list(draws = arr[, 1L, ]), class = "tulpa_fit"))
  expect_equal(dim(a1), c(100L, 1L, 2L))

  expect_null(tulpa_draws_array(structure(list(), class = "tulpa_fit")))
})


test_that("multi-chain rhat sees between-chain offset that one chain misses", {
  # Build chains with a large between-chain mean shift: pooled multi-chain
  # Rhat must flag it; a single chain (split only) should not.
  arr <- make_draws_array(niter = 500L, nchain = 4L, npar = 1L, phi = 0.3)
  for (m in seq_len(4L)) arr[, m, 1L] <- arr[, m, 1L] + (m - 1) * 2.0
  pooled <- pool_chain_major(arr)

  diag_mc <- mcmc_diagnostics(
    structure(list(draws = pooled, n_chains = 4L), class = "tulpa_fit"))
  diag_1c <- mcmc_diagnostics(
    structure(list(draws = arr[, 1L, , drop = FALSE]), class = "tulpa_fit"))

  expect_gt(diag_mc$rhat, 1.5)            # offset chains: clearly unconverged
  expect_lt(diag_1c$rhat, 1.1)            # one well-mixed chain: fine
})


test_that("select_main_params drops bracketed-index entries", {
  pn <- c("sigma", "phi", "u[10]", "w[3]", "(Intercept)")
  expect_identical(select_main_params(pn), c("sigma", "phi", "(Intercept)"))
  # falls back to the full vector if everything is an indexed entry
  expect_identical(select_main_params(c("u[1]", "u[2]")), c("u[1]", "u[2]"))
})


test_that("plotting and summary layer runs end-to-end on a multi-chain fit", {
  arr <- make_draws_array(niter = 400L, nchain = 4L, npar = 3L)
  pooled <- pool_chain_major(arr)
  n_total <- nrow(pooled)
  fit <- structure(
    list(
      draws = pooled,
      n_chains = 4L,
      backend = "hmc",
      diagnostics = list(
        divergent_idx = c(5L, 120L),
        energy = rnorm(n_total, mean = 100, sd = 5)
      )
    ),
    class = "tulpa_fit"
  )

  # divergence count is read from the populated field
  expect_equal(n_divergent(fit), 2L)

  # parameter resolution helper
  expect_identical(grep_params("theta1", colnames(pooled)), "theta1")
  expect_setequal(grep_params("^theta", colnames(pooled)),
                  c("theta1", "theta2", "theta3"))

  # diagnostic_summary produces a classed result and recommendations
  ds <- diagnostic_summary(fit, quiet = TRUE)
  expect_s3_class(ds, "tulpa_diagnostic_summary")
  expect_equal(ds$backend, "hmc")
  expect_equal(ds$n_divergent, 2L)
  expect_true(is.finite(ds$e_bfmi))

  # quick check returns an invisible logical
  ok <- check_diagnostics(fit, quiet = TRUE)
  expect_type(ok, "logical")
  expect_length(ok, 1L)

  # divergent transitions map onto the (chain, iteration) grid for bayesplot
  npdf <- .tulpa_divergent_np(fit, 400L, 4L)
  expect_equal(sum(npdf$Value), 2L)                    # the two divergent draws
  flagged <- npdf[npdf$Value == 1, ]
  expect_equal(flagged$Chain,     c(1L, 1L))           # idx 5, 120 -> chain 1
  expect_equal(flagged$Iteration, c(5L, 120L))

  # ggplot-dependent layer (Suggests)
  skip_if_not_installed("ggplot2")
  expect_s3_class(plot_rhat(fit), "ggplot")
  expect_s3_class(plot_ess(fit), "ggplot")
  expect_s3_class(plot_ess(fit, type = "tail"), "ggplot")

  # bayesplot-dependent pairs plot must not crash on the divergence path
  skip_if_not_installed("bayesplot")
  expect_no_error(plot_pairs(fit, pars = c("theta1", "theta2")))
})

test_that("every backend declares a draws-provenance (emits) class", {
  emits <- vapply(BACKEND_REGISTRY, function(e) e$emits %||% NA_character_,
                  character(1))
  expect_false(anyNA(emits))
  expect_true(all(emits %in% c("chain", "iid", "point")))
  # emits is orthogonal to tier: an Exact (Tier 1) SMC sampler emits i.i.d.
  # particles, while a Structured (Tier 2) nested-Laplace fit also emits i.i.d.
  expect_equal(BACKEND_REGISTRY$mala$emits, "chain")
  expect_equal(BACKEND_REGISTRY$smc$emits, "iid")
  expect_equal(BACKEND_REGISTRY$nested_laplace$emits, "iid")
})

test_that("draws-kind resolves from tag, registry, or unknown-as-chain", {
  expect_true(.tulpa_is_chain(
    structure(list(backend = "mala"), class = "tulpa_fit")))
  expect_false(.tulpa_is_chain(
    structure(list(backend = "nested_laplace"), class = "tulpa_fit")))
  expect_false(.tulpa_is_chain(
    structure(list(draws_kind = "iid"), class = "tulpa_fit")))   # explicit tag wins
  expect_true(.tulpa_is_chain(
    structure(list(), class = "tulpa_fit")))                     # unknown -> chain
})

test_that("chain diagnostics are withheld on a non-chain (approximation) fit", {
  set.seed(1)
  draws <- matrix(rnorm(2000), 1000, 2, dimnames = list(NULL, c("a", "b")))
  chain <- structure(list(draws = draws, backend = "mala", n_chains = 2L),
                     class = "tulpa_fit")
  iid   <- structure(list(draws = draws, backend = "nested_laplace"),
                     class = "tulpa_fit")

  # chain fit: a real chain Rhat / ESS table. i.i.d. fit: the PSIS reliability
  # table (a `laplace_diagnostics`), NOT a chain table -- so a reader cannot
  # mistake the vacuous Rhat ~ 1 / ESS ~ n of i.i.d. draws for a convergence
  # pass; the headline there is the approximation k-hat, not chain mixing.
  d_chain <- mcmc_diagnostics(chain)
  expect_s3_class(d_chain, "data.frame")
  expect_false(inherits(d_chain, "laplace_diagnostics"))
  expect_true(all(c("rhat", "ess_bulk") %in% names(d_chain)))
  d_iid <- mcmc_diagnostics(iid)
  expect_s3_class(d_iid, "laplace_diagnostics")
  expect_equal(nrow(d_iid), 2L)
  expect_true("pareto_k" %in% names(attributes(d_iid)))

  # check_diagnostics: a real verdict on the chain, NA ("not applicable") on iid.
  expect_false(is.na(check_diagnostics(chain, quiet = TRUE)))
  expect_true(is.na(suppressMessages(check_diagnostics(iid, quiet = TRUE))))

  # typed accessors: summaries see any draws, the chain view is withheld for iid.
  expect_false(is.null(posterior_sample(iid)))
  expect_null(mcmc_draws(iid))
  expect_false(is.null(mcmc_draws(chain)))
})
