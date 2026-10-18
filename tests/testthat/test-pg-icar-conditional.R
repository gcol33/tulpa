# The ICAR full conditional the Polya-Gamma Gibbs sweep draws from
# (gcol33/tulpa#423).
#
# The single-site sweep for
#   phi_j | rest ~ N( (tau sum_{k~j} phi_k + sum_resid_j) / (tau n_j + sum_omega_j),
#                     1 / (tau n_j + sum_omega_j) )
# is only that conditional if the neighbour sum reads the CURRENT field:
# neighbours the sweep has passed at their new values, the rest at the values
# the sweep started from. update_spatial_icar used to build its output from a
# freshly zeroed vector, so every neighbour with an index above j contributed 0
# and the conditional mean was shrunk toward zero by an amount set by the
# arbitrary labelling of the adjacency.
#
# The arbiter is a replica of the sweep written here in R, drawing from the same
# R stream in the same order, plus the negative control that the zeroing replica
# does NOT reproduce the kernel.

# Connected components by breadth-first scan from unit 1 upward, which is the
# labelling order pg_build_adjacency's traversal produces.
.icar_components <- function(adj_list, J) {
  comp <- rep(NA_integer_, J)
  k <- 0L
  for (s0 in seq_len(J)) {
    if (!is.na(comp[s0])) next
    comp[s0] <- k
    stack <- s0
    while (length(stack)) {
      s <- stack[length(stack)]; stack <- stack[-length(stack)]
      for (t in adj_list[[s]]) {
        if (is.na(comp[t])) { comp[t] <- k; stack <- c(stack, t) }
      }
    }
    k <- k + 1L
  }
  list(component = comp, n_components = k)
}

# The intrinsic prior the Polya-Gamma kernels sample, built densely:
# Qt = diag(w) Q_aug - w w' / W with Q_aug = Q + sum_c 1_c 1_c' / J_c, the
# marginal of the centred field under the engine's diag(w) Q_aug
# (gcol33/tulpa#963). On a connected unweighted graph it is Q.
.icar_Qt <- function(adj_list, J, w = rep(1, J)) {
  Q <- matrix(0, J, J)
  for (j in seq_len(J)) {
    Q[j, j] <- length(adj_list[[j]])
    Q[j, adj_list[[j]]] <- -1
  }
  cc <- .icar_components(adj_list, J)
  if (cc$n_components == 1L) return(w[1] * Q)
  A <- Q
  for (c1 in unique(cc$component)) {
    m <- cc$component == c1
    A[m, m] <- A[m, m] + 1 / sum(m)
  }
  diag(w) %*% A - tcrossprod(w) / sum(w)
}

# `zero_field = TRUE` is the pre-fix behaviour: build the output from zeros and
# read neighbours off it.
# `beta0` / `beta_sd`: the intercept prior every site sees through the field
# mean (gcol33/tulpa#763).
.icar_replica <- function(fx, tau, phi_in, zero_field = FALSE,
                          beta0 = ICAR_BETA0, beta_sd = ICAR_BETA_SD) {
  J <- fx$J
  sum_omega <- numeric(J); sum_resid <- numeric(J)
  for (i in seq_along(fx$kappa)) {
    g <- fx$group[i]
    sum_omega[g] <- sum_omega[g] + fx$omega[i]
    sum_resid[g] <- sum_resid[g] + fx$kappa[i] - fx$omega[i] * fx$offset[i]
  }
  Qt <- .icar_Qt(fx$adj_list, J)

  phi <- if (zero_field) numeric(J) else as.numeric(phi_in)
  for (j in seq_len(J)) {
    prec     <- tau * Qt[j, j] + sum_omega[j]
    mean_num <- -tau * sum(Qt[j, -j] * phi[-j]) + sum_resid[j]
    g <- 1 / J
    base <- beta0 + g * (sum(phi) - phi[j])
    prec <- prec + g^2 / beta_sd^2
    mean_num <- mean_num - g * base / beta_sd^2
    phi[j] <- rnorm(1, mean_num / prec, sqrt(1 / prec))
  }

  cc <- .icar_components(fx$adj_list, J)
  k <- cc$n_components
  if (k > 1L) {
    A <- numeric(k); Bnum <- numeric(k); n_c <- integer(k); S <- numeric(k)
    for (j in seq_len(J)) {
      c1 <- cc$component[j] + 1L
      A[c1]    <- A[c1] + sum_omega[j]
      Bnum[c1] <- Bnum[c1] + sum_resid[j] - sum_omega[j] * phi[j]
      n_c[c1]  <- n_c[c1] + 1L
      S[c1]    <- S[c1] + phi[j]
    }
    A <- A + tau * n_c
    Bnum <- Bnum - tau * S
    d <- vapply(seq_len(k), function(c1) rnorm(1, Bnum[c1] / A[c1],
                                               sqrt(1 / A[c1])), numeric(1))
    aSa <- sum(n_c * n_c / A)
    ad  <- sum(n_c * d)
    d <- d - (n_c / A) * ad / aSa
    phi <- phi + d[cc$component + 1L]
  }

  mean_phi <- mean(phi)
  list(phi = phi - mean_phi, removed_mean = mean_phi)
}

ICAR_BETA0 <- 0.8
ICAR_BETA_SD <- 0.6

.icar_sweep <- function(fx, tau, phi_in, beta0 = ICAR_BETA0,
                        beta_sd = ICAR_BETA_SD) {
  tulpa:::cpp_test_update_spatial_icar(
    kappa = fx$kappa, omega = fx$omega, offset = fx$offset,
    group = fx$group, adj_list = fx$adj_list, n_neighbors = fx$n_neighbors,
    n_units = fx$J, tau = tau, phi = phi_in, beta0 = beta0,
    prior_beta_sd = beta_sd)
}

# A chain 1-2-3-4-5-6, so unit 3 has one lower and one higher neighbour, and
# unit 1 has only a higher one -- the neighbour the zeroing sweep dropped.
.icar_chain <- function(J = 6L, per_unit = 5L, seed = 31L) {
  set.seed(seed)
  adj_list <- lapply(seq_len(J), function(j) {
    as.integer(c(if (j > 1L) j - 1L, if (j < J) j + 1L))
  })
  N <- J * per_unit
  list(J = J,
       adj_list = adj_list,
       n_neighbors = as.integer(vapply(adj_list, length, integer(1))),
       group = as.integer(rep(seq_len(J), each = per_unit)),
       kappa = rnorm(N, 0, 0.5),
       omega = rgamma(N, 2, 4),
       offset = rnorm(N, 0, 0.3))
}

# Two disjoint chains, so the component block update fires as well.
.icar_two_components <- function(seed = 32L) {
  fx <- .icar_chain(J = 6L, per_unit = 5L, seed = seed)
  fx$adj_list <- list(2L, c(1L, 3L), 2L, 5L, c(4L, 6L), 5L)
  fx$adj_list <- lapply(fx$adj_list, as.integer)
  fx$n_neighbors <- as.integer(vapply(fx$adj_list, length, integer(1)))
  fx
}

# A chain of five and an island, unit 6, which carries no observation: its
# value is drawn from the prior alone.
.icar_island <- function(seed = 34L) {
  fx <- .icar_chain(J = 6L, per_unit = 5L, seed = seed)
  fx$adj_list <- lapply(list(2L, c(1L, 3L), c(2L, 4L), c(3L, 5L), 4L,
                             integer(0)), as.integer)
  fx$n_neighbors <- as.integer(vapply(fx$adj_list, length, integer(1)))
  keep <- fx$group != 6L
  fx$group <- fx$group[keep]; fx$kappa <- fx$kappa[keep]
  fx$omega <- fx$omega[keep]; fx$offset <- fx$offset[keep]
  fx
}

test_that("the sweep reproduces the ICAR conditional drawn in R", {
  skip_on_cran()
  for (nm in c("chain", "two_components", "island")) {
    fx <- switch(nm, chain = .icar_chain(),
                 two_components = .icar_two_components(),
                 island = .icar_island())
    phi_in <- seq(-1.5, 1.5, length.out = fx$J)
    for (tau in c(0.5, 4, 40)) {
      set.seed(909)
      got <- .icar_sweep(fx, tau, phi_in)
      set.seed(909)
      want <- .icar_replica(fx, tau, phi_in)
      expect_equal(got$phi, want$phi, tolerance = 1e-12,
                   info = paste(nm, tau))
      expect_equal(got$removed_mean, want$removed_mean, tolerance = 1e-12,
                   info = paste(nm, tau))
    }
  }
})

test_that("the two-component fixture really exercises the block update", {
  fx <- .icar_two_components()
  got <- .icar_sweep(fx, 2, seq(-1.5, 1.5, length.out = fx$J))
  expect_equal(got$n_components, 2L)
  expect_equal(got$component, c(0L, 0L, 0L, 1L, 1L, 1L))
})

test_that("a sweep that zeroes the field does not reproduce the kernel", {
  # The negative control: without it the equivalence above could pass on a
  # replica that shares the defect.
  fx <- .icar_chain()
  phi_in <- seq(-1.5, 1.5, length.out = fx$J)
  set.seed(909)
  got <- .icar_sweep(fx, 40, phi_in)
  set.seed(909)
  bugged <- .icar_replica(fx, 40, phi_in, zero_field = TRUE)
  expect_gt(max(abs(got$phi - bugged$phi)), 0.1)
})

test_that("the kernels' ICAR prior is the marginal of the centred Q_aug field (#963)", {
  skip_on_cran()
  # Integrating the constant direction out of x' diag(w) Q_aug x leaves the
  # Schur complement diag(w) Q_aug - w w' / W, so Qt is checked against that
  # construction, and its quadratic form against the integral it stands for.
  fx_two <- .icar_two_components()
  fx_isl <- .icar_island()
  cases <- list(
    connected = list(fx = .icar_chain(), w = NULL),
    two       = list(fx = fx_two, w = NULL),
    island    = list(fx = fx_isl, w = NULL),
    weighted  = list(fx = fx_isl, w = c(rep(0.7, 5), 2.3))
  )
  set.seed(5)
  for (nm in names(cases)) {
    fx <- cases[[nm]]$fx
    J <- fx$J
    w <- cases[[nm]]$w %||% rep(1, J)
    x <- rnorm(J)
    got <- tulpa:::cpp_test_pg_icar_prior(fx$adj_list, fx$n_neighbors, J, x,
                                          node_prec = cases[[nm]]$w)
    want <- .icar_Qt(fx$adj_list, J, w)
    expect_equal(got$Qt, want, tolerance = 1e-12, info = nm)
    expect_equal(as.numeric(got$Qt %*% rep(1, J)), rep(0, J),
                 tolerance = 1e-12, info = nm)
    expect_equal(got$rank, J - 1L, info = nm)
    expect_equal(qr(got$Qt)$rank, J - 1L, info = nm)
    expect_equal(got$quad, sum(x * (want %*% x)), tolerance = 1e-12,
                 info = nm)
    expect_equal(got$site_prec, diag(want), tolerance = 1e-12, info = nm)
    off <- vapply(seq_len(J), function(j) -sum(want[j, -j] * x[-j]),
                  numeric(1))
    expect_equal(got$site_num, off, tolerance = 1e-12, info = nm)

    # The marginal: min over the level m of (x + m 1)' diag(w) Q_aug (x + m 1)
    # is x' Qt x, since the integral over m of a Gaussian is its minimum up to
    # a constant in x.
    cc <- .icar_components(fx$adj_list, J)
    if (cc$n_components > 1L) {
      Qa <- want + tcrossprod(w) / sum(w)
      m_hat <- -sum(Qa %*% x) / sum(Qa)
      expect_equal(got$quad, sum((x + m_hat) * (Qa %*% (x + m_hat))),
                   tolerance = 1e-10, info = nm)
    }
  }
  # Connected and unweighted, Qt is the plain ICAR precision.
  fx <- .icar_chain()
  got <- tulpa:::cpp_test_pg_icar_prior(fx$adj_list, fx$n_neighbors, fx$J,
                                        rnorm(fx$J))
  Q <- -outer(seq_len(fx$J), seq_len(fx$J),
              function(a, b) as.numeric(abs(a - b) == 1))
  diag(Q) <- fx$n_neighbors
  expect_equal(got$Qt, Q)
})

test_that("an island with no observation is drawn from its N(0, 1 / tau) prior (#963)", {
  # Was pinned to 0 (no data) or N(0, 1000) (with data). Over many sweeps from
  # one incoming field the island's centred value has the conditional variance
  # 1 / (tau Qt_66 + intercept term): order 1 / tau.
  fx <- .icar_island()
  tau <- 4
  set.seed(11)
  isl <- vapply(seq_len(3000L), function(r) {
    s <- .icar_sweep(fx, tau, rep(0, fx$J))
    s$phi[6] + s$removed_mean
  }, numeric(1))
  expect_gt(sd(isl), 0.2)
  expect_lt(sd(isl), 1)
})

test_that("the leading unit's draw matches its closed-form conditional", {
  # Unit 1's only neighbour is unit 2, which the sweep has not reached, so its
  # conditional is a fixed Gaussian in the incoming field -- and it is exactly
  # the term the zeroing sweep dropped. On a connected graph the component
  # block update is skipped, so phi[1] + removed_mean is the raw draw.
  fx <- .icar_chain(seed = 33L)
  phi_in <- c(2.0, -1.25, 0.4, 0.9, -0.6, 1.1)
  tau <- 3.0

  sum_omega <- tapply(fx$omega, fx$group, sum)
  sum_resid <- tapply(fx$kappa - fx$omega * fx$offset, fx$group, sum)
  # The intercept prior the unit sees through the field mean (gcol33/tulpa#763).
  g <- 1 / fx$J
  base <- ICAR_BETA0 + g * sum(phi_in[-1])
  prec <- tau * fx$n_neighbors[1] + sum_omega[[1]] + g^2 / ICAR_BETA_SD^2
  mu   <- (tau * phi_in[2] + sum_resid[[1]] - g * base / ICAR_BETA_SD^2) / prec
  sdev <- sqrt(1 / prec)

  set.seed(77)
  raw <- vapply(seq_len(4000L), function(r) {
    s <- .icar_sweep(fx, tau, phi_in)
    s$phi[1] + s$removed_mean
  }, numeric(1))

  # 4000 draws: the mean's standard error is sdev / 63.
  expect_lt(abs(mean(raw) - mu), 4 * sdev / sqrt(4000))
  expect_lt(abs(sd(raw) / sdev - 1), 0.06)
  # The mean the zeroed field would have produced is well outside that band.
  mu_zeroed <- sum_resid[[1]] / prec
  expect_gt(abs(mu - mu_zeroed), 10 * sdev / sqrt(4000))
})
