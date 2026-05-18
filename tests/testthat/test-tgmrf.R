# ---------------------------------------------------------------------------
# tgmrf() — user-defined GMRF latent block (P1: constructor + formula hook)
#
# Covers:
#   * Constructor validation (Q, prior, init, mu, graph, bounds).
#   * Shape capture (n_latent, theta_dim, theta_names, sparsity pattern).
#   * Periodic AR(1) worked example from generic-todo.md §7.
#   * Formula parser hook: latent(tgmrf(...)) detected, evaluated, stripped.
#
# P2 (Laplace adapter) and beyond are covered in separate test files.
# ---------------------------------------------------------------------------

make_periodic_ar1 <- function(n) {
  # init uses atanh_rho = atanh(0.3) so the off-diagonals are nonzero at init.
  # With rho = 0 the precision collapses to the identity and the captured
  # sparsity pattern would lose the periodic-tridiagonal structure — fine for
  # an inner solve but misleading as a registration-time sanity check.
  tgmrf(
    Q = function(theta) {
      sigma <- exp(theta[1]); rho <- tanh(theta[2])
      d <- rep((1 + rho^2) / sigma^2, n)
      o <- rep(-rho / sigma^2, n)
      M <- Matrix::bandSparse(
        n, k = c(-1L, 0L, 1L), diagonals = list(o, d, o)
      )
      M[1, n] <- -rho / sigma^2
      M[n, 1] <- -rho / sigma^2
      methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    },
    prior = function(theta) {
      stats::dnorm(theta[1], 0, 1, log = TRUE) +
        stats::dnorm(theta[2], 0, 1, log = TRUE)
    },
    init = c(log_sigma = 0, atanh_rho = atanh(0.3)),
    name = "periodic_ar1"
  )
}

test_that("tgmrf() builds the periodic AR1 worked example", {
  blk <- make_periodic_ar1(20)
  expect_s3_class(blk, "tgmrf")
  expect_s3_class(blk, "tulpa_latent_block")
  expect_equal(blk$n_latent, 20L)
  expect_equal(blk$theta_dim, 2L)
  expect_equal(blk$theta_names, c("log_sigma", "atanh_rho"))
  expect_equal(blk$name, "periodic_ar1")
  expect_equal(blk$backend, "r")
})

test_that("tgmrf() captures Q sparsity pattern at init", {
  blk <- make_periodic_ar1(10)
  expect_s4_class(blk$pattern, "dgCMatrix")
  expect_equal(dim(blk$pattern), c(10L, 10L))
  # Periodic tridiagonal: 3 nonzeros per row (self + two neighbours, wrap).
  expect_equal(Matrix::nnzero(blk$pattern), 10L * 3L)
})

test_that("tgmrf() rejects a non-function Q", {
  expect_error(
    tgmrf(Q = NULL,
          prior = function(theta) 0,
          init = c(0)),
    "must be a function"
  )
})

test_that("tgmrf() rejects a non-function prior", {
  expect_error(
    tgmrf(Q = function(theta) Matrix::Diagonal(3),
          prior = "not a function",
          init = c(0)),
    "must be a function"
  )
})

test_that("tgmrf() rejects empty / non-numeric init", {
  Q1 <- function(theta) Matrix::Diagonal(3)
  pr <- function(theta) 0
  expect_error(tgmrf(Q = Q1, prior = pr, init = numeric(0)), "length >= 1")
  expect_error(tgmrf(Q = Q1, prior = pr, init = "abc"),     "numeric")
  expect_error(tgmrf(Q = Q1, prior = pr, init = c(0, NA)),  "finite")
})

test_that("tgmrf() errors when Q(init) is not a sparseMatrix", {
  expect_error(
    tgmrf(Q     = function(theta) diag(3),                 # dense base R
          prior = function(theta) 0,
          init  = c(0)),
    "must return a Matrix::sparseMatrix"
  )
})

test_that("tgmrf() errors on non-square Q", {
  expect_error(
    tgmrf(Q     = function(theta) Matrix::sparseMatrix(i = c(1, 2),
                                                       j = c(1, 2),
                                                       x = c(1, 1),
                                                       dims = c(3, 2)),
          prior = function(theta) 0,
          init  = c(0)),
    "must be square"
  )
})

test_that("tgmrf() errors on asymmetric Q", {
  asym_Q <- function(theta) {
    Matrix::sparseMatrix(i = c(1, 1, 2, 2),
                         j = c(1, 2, 1, 2),
                         x = c(1, 0.3, 0.7, 1),
                         dims = c(2, 2))
  }
  expect_error(
    tgmrf(Q = asym_Q, prior = function(theta) 0, init = c(0)),
    "must be symmetric"
  )
})

test_that("tgmrf() catches a Q() implementation that throws", {
  bad_Q <- function(theta) stop("user bug")
  expect_error(
    tgmrf(Q = bad_Q, prior = function(theta) 0, init = c(0)),
    "raised an error.*user bug"
  )
})

test_that("tgmrf() catches a prior() that throws", {
  expect_error(
    tgmrf(Q     = function(theta) Matrix::Diagonal(2),
          prior = function(theta) stop("prior bug"),
          init  = c(0)),
    "raised an error.*prior bug"
  )
})

test_that("tgmrf() errors when prior(init) is not a finite scalar", {
  Q1 <- function(theta) Matrix::Diagonal(3)
  expect_error(
    tgmrf(Q = Q1, prior = function(theta) c(1, 2), init = c(0)),
    "finite numeric scalar"
  )
  expect_error(
    tgmrf(Q = Q1, prior = function(theta) NA_real_, init = c(0)),
    "finite numeric scalar"
  )
  expect_error(
    tgmrf(Q = Q1, prior = function(theta) Inf, init = c(0)),
    "finite numeric scalar"
  )
})

test_that("tgmrf() accepts a non-zero mu and validates its return shape", {
  Q1 <- function(theta) Matrix::Diagonal(4)
  pr <- function(theta) 0
  good_mu <- function(theta) rep(theta[1], 4)
  blk <- tgmrf(Q = Q1, prior = pr, init = c(0.5), mu = good_mu)
  expect_true(is.function(blk$mu))

  bad_mu <- function(theta) rep(theta[1], 3)         # wrong length
  expect_error(
    tgmrf(Q = Q1, prior = pr, init = c(0.5), mu = bad_mu),
    "length 4"
  )
})

test_that("tgmrf() validates an optional graph pattern", {
  Q1 <- function(theta) {
    Matrix::sparseMatrix(i = c(1, 2, 3, 1, 2),
                         j = c(1, 2, 3, 2, 1),
                         x = c(1, 1, 1, -0.3, -0.3),
                         dims = c(3, 3))
  }
  pr <- function(theta) 0
  # Graph covering the diagonal + (1,2)/(2,1): superset OK.
  g_ok <- Matrix::sparseMatrix(i = c(1, 2, 3, 1, 2),
                               j = c(1, 2, 3, 2, 1),
                               x = c(1, 1, 1, 1, 1),
                               dims = c(3, 3))
  expect_silent(tgmrf(Q = Q1, prior = pr, init = c(0), graph = g_ok))

  # Graph missing the (1,2) edge: subset check must fail.
  g_bad <- Matrix::sparseMatrix(i = c(1, 2, 3),
                                j = c(1, 2, 3),
                                x = c(1, 1, 1),
                                dims = c(3, 3))
  expect_error(
    tgmrf(Q = Q1, prior = pr, init = c(0), graph = g_bad),
    "nonzero entries outside.*graph"
  )

  # Graph with mismatched dims.
  g_bad_dims <- Matrix::Diagonal(4)
  expect_error(
    tgmrf(Q = Q1, prior = pr, init = c(0), graph = g_bad_dims),
    "must be 3 x 3"
  )
})

test_that("tgmrf() validates bounds shape and ordering", {
  Q1 <- function(theta) Matrix::Diagonal(2)
  pr <- function(theta) 0
  expect_silent(
    tgmrf(Q = Q1, prior = pr, init = c(0, 0),
          bounds = list(lower = c(-3, -3), upper = c(3, 3)))
  )
  expect_error(
    tgmrf(Q = Q1, prior = pr, init = c(0, 0),
          bounds = list(lower = c(-3), upper = c(3))),
    "length 2"
  )
  expect_error(
    tgmrf(Q = Q1, prior = pr, init = c(0, 0),
          bounds = list(lower = c(2, 2), upper = c(1, 1))),
    "strictly below"
  )
  expect_error(
    tgmrf(Q = Q1, prior = pr, init = c(0, 0),
          bounds = list(lower = c(-3, -3))),
    "components `lower` and `upper`"
  )
})

test_that("tgmrf() assigns default theta names when init is unnamed", {
  blk <- tgmrf(Q     = function(theta) Matrix::Diagonal(2),
               prior = function(theta) 0,
               init  = c(0.1, 0.2))
  expect_equal(blk$theta_names, c("theta_1", "theta_2"))
  expect_equal(names(blk$init), c("theta_1", "theta_2"))
})

test_that("print.tgmrf() runs without error", {
  blk <- make_periodic_ar1(8)
  expect_output(print(blk), "<tgmrf>")
  expect_output(print(blk), "n_latent.*8")
  expect_output(print(blk), "log_sigma, atanh_rho")
})

test_that("summary.tgmrf() prints prior value and bounds", {
  Q1 <- function(theta) Matrix::Diagonal(3)
  pr <- function(theta) -0.5 * sum(theta^2)
  blk <- tgmrf(Q = Q1, prior = pr, init = c(a = 0.4, b = -0.2),
               bounds = list(lower = c(-3, -3), upper = c(3, 3)))
  expect_output(summary(blk), "log prior at init")
  expect_output(summary(blk), "theta bounds")
})

# ---------------------------------------------------------------------------
# Formula parser hook
# ---------------------------------------------------------------------------

test_that("find_latent_terms() locates latent(...) calls in the AST", {
  blk <- make_periodic_ar1(4)
  f   <- y ~ x + latent(blk)
  matched <- find_latent_terms(f[[3]])
  expect_length(matched, 1L)
  expect_true(is.call(matched[[1]]))
  expect_identical(matched[[1]][[1]], as.name("latent"))
})

test_that("find_latent_terms() returns NULL when no latent term is present", {
  expect_null(find_latent_terms((y ~ x + (1 | g))[[3]]))
  expect_null(find_latent_terms((y ~ x1 + x2)[[3]]))
})

test_that("find_latent_terms() finds multiple latent(...) calls", {
  blk1 <- make_periodic_ar1(4)
  blk2 <- make_periodic_ar1(6)
  f <- y ~ latent(blk1) + latent(blk2)
  expect_length(find_latent_terms(f[[3]]), 2L)
})

test_that("no_latent_terms() strips latent(...) calls", {
  blk <- make_periodic_ar1(4)
  rhs <- (y ~ x + latent(blk))[[3]]
  clean <- no_latent_terms(rhs)
  expect_false(grepl("latent", paste(deparse(clean), collapse = ""), fixed = TRUE))
})

test_that("no_latent_terms() returns NULL when only latent(...) remains", {
  blk <- make_periodic_ar1(4)
  rhs <- (y ~ latent(blk))[[3]]
  expect_null(no_latent_terms(rhs))
})

test_that("tulpa_parse_formula() captures latent blocks and strips them from fixed RHS", {
  blk <- make_periodic_ar1(12)
  pf  <- tulpa_parse_formula(y ~ x1 + x2 + latent(blk))
  expect_s3_class(pf, "tulpa_parsed_formula")
  expect_equal(pf$n_latent_blocks, 1L)
  expect_identical(pf$latent_blocks[[1]], blk)
  expect_false(grepl("latent",
                     paste(deparse(pf$fixed_formula), collapse = ""),
                     fixed = TRUE))
  # Fixed effects formula should still have x1 + x2.
  expect_equal(pf$response, "y")
  expect_true(grepl("x1", paste(deparse(pf$fixed_formula), collapse = "")))
})

test_that("tulpa_parse_formula() handles latent(...) alongside RE bars", {
  blk <- make_periodic_ar1(8)
  pf  <- tulpa_parse_formula(y ~ x + (1 | g) + latent(blk))
  expect_equal(pf$n_re_terms, 1L)
  expect_equal(pf$n_latent_blocks, 1L)
  expect_equal(pf$random_effects[[1]]$group_var, "g")
})

test_that("tulpa_parse_formula() handles a latent-only RHS", {
  blk <- make_periodic_ar1(5)
  pf  <- tulpa_parse_formula(y ~ latent(blk))
  expect_equal(pf$n_latent_blocks, 1L)
  # Fixed formula collapses to `y ~ 1`.
  expect_equal(
    trimws(paste(deparse(pf$fixed_formula), collapse = "")),
    "y ~ 1"
  )
})

test_that("tulpa_parse_formula() supports multiple latent(...) terms", {
  blk1 <- make_periodic_ar1(4)
  blk2 <- make_periodic_ar1(6)
  pf <- tulpa_parse_formula(y ~ latent(blk1) + x + latent(blk2))
  expect_equal(pf$n_latent_blocks, 2L)
  # Preserve order of appearance.
  expect_equal(pf$latent_blocks[[1]]$n_latent, 4L)
  expect_equal(pf$latent_blocks[[2]]$n_latent, 6L)
})

test_that("tulpa_parse_formula() errors when latent() wraps a non-block", {
  not_a_block <- list(foo = 1)
  expect_error(
    tulpa_parse_formula(y ~ latent(not_a_block)),
    "tulpa latent-block object"
  )
})

test_that("tulpa_parse_formula() prints latent-block summary", {
  blk <- make_periodic_ar1(7)
  pf <- tulpa_parse_formula(y ~ x + latent(blk))
  expect_output(print(pf), "Latent blocks: 1")
  expect_output(print(pf), "periodic_ar1")
})
