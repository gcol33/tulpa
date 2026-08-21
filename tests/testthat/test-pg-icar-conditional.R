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

# `zero_field = TRUE` is the pre-fix behaviour: build the output from zeros and
# read neighbours off it.
.icar_replica <- function(fx, tau, phi_in, isolated_prec, zero_field = FALSE) {
  J <- fx$J
  sum_omega <- numeric(J); sum_resid <- numeric(J)
  for (i in seq_along(fx$kappa)) {
    g <- fx$group[i]
    sum_omega[g] <- sum_omega[g] + fx$omega[i]
    sum_resid[g] <- sum_resid[g] + fx$kappa[i] - fx$omega[i] * fx$offset[i]
  }

  phi <- if (zero_field) numeric(J) else as.numeric(phi_in)
  for (j in seq_len(J)) {
    nb  <- fx$adj_list[[j]]
    n_j <- fx$n_neighbors[j]
    prec     <- tau * n_j + sum_omega[j]
    mean_num <- tau * (if (n_j > 0) sum(phi[nb]) else 0) + sum_resid[j]
    if (n_j == 0) {
      if (sum_omega[j] > 0) {
        prec <- sum_omega[j] + isolated_prec
        mean_num <- sum_resid[j]
      } else {
        phi[j] <- 0
        next
      }
    }
    phi[j] <- rnorm(1, mean_num / prec, sqrt(1 / prec))
  }

  cc <- .icar_components(fx$adj_list, J)
  k <- cc$n_components
  if (k > 1L) {
    A <- numeric(k); Bnum <- numeric(k); n_c <- integer(k)
    for (j in seq_len(J)) {
      c1 <- cc$component[j] + 1L
      A[c1]    <- A[c1] + sum_omega[j]
      Bnum[c1] <- Bnum[c1] + sum_resid[j] - sum_omega[j] * phi[j]
      n_c[c1]  <- n_c[c1] + 1L
    }
    if (all(A > 0)) {
      d <- vapply(seq_len(k), function(c1) rnorm(1, Bnum[c1] / A[c1],
                                                 sqrt(1 / A[c1])), numeric(1))
      aSa <- sum(n_c * n_c / A)
      ad  <- sum(n_c * d)
      if (aSa > 0) d <- d - (n_c / A) * ad / aSa
      phi <- phi + d[cc$component + 1L]
    }
  }

  mean_phi <- mean(phi)
  list(phi = phi - mean_phi, removed_mean = mean_phi)
}

.icar_sweep <- function(fx, tau, phi_in) {
  tulpa:::cpp_test_update_spatial_icar(
    kappa = fx$kappa, omega = fx$omega, offset = fx$offset,
    group = fx$group, adj_list = fx$adj_list, n_neighbors = fx$n_neighbors,
    n_units = fx$J, tau = tau, phi = phi_in)
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

test_that("the sweep reproduces the ICAR conditional drawn in R", {
  for (nm in c("chain", "two_components")) {
    fx <- if (nm == "chain") .icar_chain() else .icar_two_components()
    phi_in <- seq(-1.5, 1.5, length.out = fx$J)
    for (tau in c(0.5, 4, 40)) {
      set.seed(909)
      got <- .icar_sweep(fx, tau, phi_in)
      set.seed(909)
      want <- .icar_replica(fx, tau, phi_in, got$isolated_prec)
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
  bugged <- .icar_replica(fx, 40, phi_in, got$isolated_prec, zero_field = TRUE)
  expect_gt(max(abs(got$phi - bugged$phi)), 0.1)
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
  prec <- tau * fx$n_neighbors[1] + sum_omega[[1]]
  mu   <- (tau * phi_in[2] + sum_resid[[1]]) / prec
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
