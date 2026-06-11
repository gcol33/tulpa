# Outer/inner thread-budget split for the joint nested-Laplace grid
# (gcol33/tulpa#107). When the outer grid has fewer cells than the outer thread
# pool, the surplus threads are handed to the inner per-observation reduction
# via nested OpenMP, so the long-pole cells are not solved single-threaded while
# freed outer threads idle. The split stays within the outer budget
# (outer_used * inner <= n_threads_outer), so it never oversubscribes, and it is
# a no-op when the grid saturates the pool (n_grid >= n_threads_outer). It must
# leave the fit unchanged up to the floating-point summation order of the inner
# reduction -- the documented threading invariant.

test_that("outer/inner thread split leaves the joint fit unchanged", {
  skip_on_cran()

  chain_adj <- function(S) {
    nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
    nn <- lengths(nb)
    list(adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
         n_neighbors = nn)
  }
  set.seed(7)
  S <- 24L; adj <- chain_adj(S)
  field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
  mk_arm <- function(m, fam) {
    si <- sample(S, m, replace = TRUE); x <- rnorm(m)
    lin <- 0.2 + 0.5 * x + 0.8 * field[si]
    y <- if (fam == "binomial") rbinom(m, 1L, plogis(lin)) else lin + rnorm(m, 0, 0.5)
    list(y = as.numeric(y), n_trials = rep(1L, m), X = cbind(1, x),
         spatial_idx = si, family = fam, phi = if (fam == "gaussian") 0.5 else 1)
  }
  resp <- list(occ = mk_arm(3000L, "binomial"), pos = mk_arm(3000L, "gaussian"))
  prior <- list(type = "icar", n_spatial_units = S,
                adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                n_neighbors = adj$n_neighbors,
                sigma_grid = c(0.3, 0.6, 1.0, 1.6))         # 4 outer cells

  fit_with <- function(outer) suppressMessages(tulpa_nested_laplace_joint(
    responses = resp, prior = prior, control = list(n_threads_outer = outer)))

  serial <- fit_with(1L)                                    # inner = 1
  split  <- fit_with(8L)                                    # 4 cells / 8 -> inner = 2

  expect_equal(split$theta_mean, serial$theta_mean, tolerance = 1e-7)
  expect_equal(sort(split$weights), sort(serial$weights), tolerance = 1e-7)
  expect_equal(split$theta_sd, serial$theta_sd, tolerance = 1e-6)
})
