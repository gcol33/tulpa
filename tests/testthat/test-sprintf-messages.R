# Messages that crashed inside sprintf() before they printed: a format string
# split across two literals passed as separate arguments, or swapped arguments
# (gcol33/tulpa#885).

test_that("the RE-block design message prints (tulpa_eb / re_cov_nested)", {
  set.seed(1)
  J <- 20L
  g <- rep(seq_len(J), each = 6)
  x <- rnorm(length(g))
  y <- rpois(length(g), exp(0.3 + 0.5 * x))
  expect_error(tulpa_eb(y, NULL, cbind(1, x),
                        list(idx = g, n_groups = J, n_coefs = 2L)),
               "RE block 1 \\(n_coefs = 2\\) requires `Z`")
})

test_that("the re_cov_gibbs prior_df messages print", {
  bl_diag <- list(nc = 1L, full = FALSE)
  bl_full <- list(nc = 3L, full = TRUE)
  expect_error(.re_gibbs_block_prior(bl_diag, -1, NULL),
               "`prior_df` = -1 must be > 0 for a proper scalar inverse-gamma")
  expect_error(.re_gibbs_block_prior(bl_full, 1, NULL),
               "`prior_df` = 1 must exceed n_coefs - 1 = 2 for a proper")
})

test_that("print.tulpa_criteria reports high Pareto-k observations", {
  set.seed(1)
  S <- 200L
  n <- 20L
  mu <- matrix(rnorm(S * n), S, n)
  yy <- rnorm(n)
  yy[3] <- 12
  cr <- tulpa_criteria(dnorm(matrix(yy, S, n, byrow = TRUE), mu, 0.3,
                             log = TRUE))
  expect_gt(cr$n_high_k, 0L)
  expect_output(print(cr), sprintf("  %d obs with Pareto k >= ", cr$n_high_k))
})

# The lint the issue asked for, run over the package's own functions: no
# sprintf() / gettextf() call may take a string literal as its second
# positional argument, which is always a split format string.
test_that("no sprintf() call splits its format across two literals", {
  ns <- asNamespace("tulpa")
  bad <- character(0)
  walk <- function(e, where) {
    if (!is.call(e)) return(invisible())
    fn <- e[[1L]]
    if (is.symbol(fn) && as.character(fn) %in% c("sprintf", "gettextf")) {
      args <- as.list(e)[-1L]
      nm <- names(args) %||% rep("", length(args))
      pos <- args[nm == ""]
      if (length(pos) >= 2L && is.character(pos[[2L]])) {
        bad <<- c(bad, where)
      }
    }
    for (a in as.list(e)[-1L]) {
      if (!missing(a)) walk(a, where)
    }
  }
  for (f in ls(ns, all.names = TRUE)) {
    obj <- get(f, envir = ns)
    if (is.function(obj) && !is.primitive(obj)) walk(body(obj), f)
  }
  expect_identical(unique(bad), character(0))
})
