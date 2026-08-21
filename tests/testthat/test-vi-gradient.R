# Finite-difference checks on the three VI reparameterisation gradients
# (gcol33/tulpa#479).
#
# `cpp_vi_elbo_grad()` evaluates the ELBO and its packed gradient at a supplied
# variational parameter vector, drawing its Monte Carlo sample from a generator
# seeded on entry. The draws do not depend on the variational parameters, so at
# a fixed (seed, mc_samples) the ELBO is a DETERMINISTIC function of the packed
# vector and the reparameterisation gradient is that function's exact
# derivative. A central difference then reproduces it to the difference rule's
# own accuracy -- Monte Carlo noise never enters the comparison, which is what
# lets a real tolerance be asserted instead of a wide band.
#
# Three things the comparison is built around:
#
#   * CENTRAL differences, with the step scaled to each coordinate's own
#     magnitude. One global step is either too coarse for a coordinate near zero
#     or too fine for a large one.
#   * A RELATIVE error read against the gradient's own scale, reported with that
#     scale. An absolute difference means nothing on its own.
#   * The WHOLE packed vector, plus a count of identically-zero analytic
#     entries. A dropped term does not make a number wrong, it makes a block
#     absent, and only a per-block scan sees that.
#
# The entropy is checked a second time on its own, against a DENSE
# log-determinant of the covariance each variant parameterizes. The low-rank
# entropy and its gradient both go through the matrix determinant lemma, so a
# check that reads them against each other would test the lemma against itself.

# --- shared probe -----------------------------------------------------------

.vi_probe_data <- function(seed = 4242L, n = 60L) {
  set.seed(seed)
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  X <- cbind(1, x1, x2)
  eta <- as.numeric(X %*% c(0.3, 0.6, -0.4))
  list(y = as.numeric(rpois(n, exp(eta))),
       n_trials = rep(1L, n),
       X = X)
}

.VI_RANK <- 2L

.vi_call <- function(d, variant, x = NULL, mc_samples = 16L, seed = 7L) {
  cpp_vi_elbo_grad(
    y = d$y, n_trials = d$n_trials, X = d$X, family = "poisson",
    variant = variant, mc_samples = mc_samples, seed = seed,
    rank = .VI_RANK, x = x)
}

# Central difference of `f` at `x`, step scaled per coordinate.
.vi_fd_grad <- function(f, x, h_rel = 1e-5) {
  vapply(seq_along(x), function(k) {
    h <- h_rel * max(1, abs(x[k]))
    xp <- x; xp[k] <- xp[k] + h
    xm <- x; xm[k] <- xm[k] - h
    (f(xp) - f(xm)) / (2 * h)
  }, numeric(1))
}

# Per-entry relative error against the gradient's own scale. An entry more than
# eight orders below that scale is compared against the scale instead of against
# itself: a central difference cannot resolve it either way.
.vi_rel_err <- function(num, ana) {
  scale <- max(abs(ana))
  floor_k <- max(1e-8 * scale, 1e-300)
  abs(num - ana) / pmax(pmax(abs(ana), abs(num)), floor_k)
}

.vi_report <- function(num, ana, what) {
  sprintf(paste0("%s: |grad|_max = %.4g, max abs err = %.3g, ",
                 "max rel err = %.3g, %d of %d analytic entries zero"),
          what, max(abs(ana)), max(abs(num - ana)),
          max(.vi_rel_err(num, ana)), sum(ana == 0), length(ana))
}

# The variational parameter blocks, in the variant's own flatten() layout.
.vi_blocks <- function(variant, D) {
  r <- .VI_RANK
  if (variant == 0L) {
    list(mu = seq_len(D), log_sigma = D + seq_len(D))
  } else if (variant == 1L) {
    list(mu = seq_len(D), L = D + seq_len(D * r),
         log_d = D + D * r + seq_len(D))
  } else {
    list(mu = seq_len(D), L = D + seq_len(D * (D + 1) / 2))
  }
}

# Entropy H[q] written independently of the engine: a DENSE log-determinant of
# the covariance the variant parameterizes.
.vi_entropy_dense <- function(variant, x, D) {
  r <- .VI_RANK
  const <- 0.5 * D * (1 + log(2 * pi))
  if (variant == 0L) {
    Sigma <- diag(exp(2 * x[D + seq_len(D)]), D)
  } else if (variant == 1L) {
    L <- matrix(x[D + seq_len(D * r)], D, r)
    Sigma <- L %*% t(L) + diag(exp(2 * x[D + D * r + seq_len(D)]), D)
  } else {
    L <- matrix(0, D, D)
    idx <- D
    for (i in seq_len(D)) {
      for (j in seq_len(i)) {
        idx <- idx + 1L
        L[i, j] <- x[idx]
      }
    }
    Sigma <- L %*% t(L)
  }
  0.5 * as.numeric(determinant(Sigma, logarithm = TRUE)$modulus) + const
}

# A point away from the default initialisation, so off-diagonal L entries and
# unequal scales are exercised rather than a symmetric special case. The offset
# is bounded by 0.08, which keeps the full-rank diagonal (0.5 at the default)
# clear of the 0.01 floor: at the floor the map from the packed vector to the
# parameters stops being smooth and a finite difference would straddle the kink.
.vi_perturb <- function(x0) {
  x0 + 0.08 * sin(seq_along(x0))
}

# --- 1. the reparameterisation gradients ------------------------------------

for (.spec in list(list(code = 0L, name = "mean-field"),
                   list(code = 1L, name = "low-rank"),
                   list(code = 2L, name = "full-rank"))) {
  local({
    code <- .spec$code
    name <- .spec$name

    test_that(sprintf("the %s VI gradient matches central differences of its own ELBO",
                      name), {
      d <- .vi_probe_data()
      base <- .vi_call(d, code)
      D <- base$D
      x <- .vi_perturb(base$x)
      at <- .vi_call(d, code, x = x)

      # No coordinate was repaired on the way in, so the ELBO is smooth in x.
      expect_equal(at$x, x, tolerance = 1e-12)

      num <- .vi_fd_grad(function(xx) .vi_call(d, code, x = xx)$elbo, x)
      ana <- at$grad

      expect_length(ana, length(x))
      expect_true(all(is.finite(ana)))
      expect_lt(max(.vi_rel_err(num, ana)), 1e-6,
                label = .vi_report(num, ana, paste(name, "ELBO gradient")))

      # No parameter block comes back identically zero: that is how a dropped
      # chain-rule or entropy term hides.
      blocks <- .vi_blocks(code, D)
      for (blk in names(blocks)) {
        expect_gt(max(abs(ana[blocks[[blk]]])), 0,
                  label = sprintf("%s: block '%s' is identically zero",
                                  name, blk))
      }
    })

    test_that(sprintf("the %s VI entropy and its gradient are the same function",
                      name), {
      d <- .vi_probe_data()
      base <- .vi_call(d, code)
      D <- base$D
      x <- .vi_perturb(base$x)
      at <- .vi_call(d, code, x = x)

      # The entropy the engine adds to the ELBO, against a dense
      # log-determinant.
      expect_equal(at$entropy, .vi_entropy_dense(code, x, D), tolerance = 1e-10)

      # Its gradient, against a central difference of that same dense form. No
      # Monte Carlo enters either side.
      num <- .vi_fd_grad(function(xx) .vi_entropy_dense(code, xx, D), x)
      ana <- at$entropy_grad
      expect_lt(max(.vi_rel_err(num, ana)), 1e-6,
                label = .vi_report(num, ana, paste(name, "entropy gradient")))
    })
  })
}

# --- 2. the low-rank start is seed-controlled (gcol33/tulpa#481) -------------

test_that("the low-rank variational start is a function of the seed", {
  d <- .vi_probe_data()
  a1 <- .vi_call(d, 1L, seed = 11L)
  a2 <- .vi_call(d, 1L, seed = 11L)
  b  <- .vi_call(d, 1L, seed = 12L)
  expect_identical(a1$x, a2$x)
  expect_gt(max(abs(a1$x - b$x)), 1e-8)
})

# The other two variants start deterministically, so their start does not move
# with the seed.
test_that("the mean-field and full-rank starts do not move with the seed", {
  d <- .vi_probe_data()
  for (code in c(0L, 2L)) {
    expect_identical(.vi_call(d, code, seed = 11L)$x,
                     .vi_call(d, code, seed = 12L)$x)
  }
})

# --- 3. a loop that cannot run is rejected (gcol33/tulpa#476) ----------------

test_that("VI rejects a max_iter or mc_samples below one", {
  d <- .vi_probe_data(n = 30L)
  for (bad in list(list(vi_max_iter = 0L), list(vi_max_iter = -5L))) {
    expect_error(
      tulpa_sample_glmm(y = d$y, n_trials = d$n_trials, X = d$X,
                        family = "poisson", backend = "vi", control = bad),
      "vi_max_iter")
  }
  expect_error(
    tulpa_sample_glmm(y = d$y, n_trials = d$n_trials, X = d$X,
                      family = "poisson", backend = "vi",
                      control = list(vi_mc_samples = 0L)),
    "vi_mc_samples")
})
