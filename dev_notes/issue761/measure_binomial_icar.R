# Does the binomial ICAR Gibbs kernel, which removes the field's mean into the
# intercept every sweep, sample the sum-to-zero model its recorded log_prob
# evaluates (intercept N(0, sd^2), ICAR on a mean-zero field)? A tight intercept
# prior against a data-determined level is where the two can differ.
#
#   Rscript measure_binomial_icar.R <arm> <out.rds>
args <- commandArgs(trailingOnly = TRUE)
arm <- args[1]; out <- args[2]

set.seed(7613)
nr <- 3L; nc <- 3L; J <- nr * nc
W <- matrix(0, J, J)
for (i in seq_len(nr)) for (j in seq_len(nc)) {
  k <- (j - 1L) * nr + i
  if (i < nr) { W[k, k + 1L] <- 1; W[k + 1L, k] <- 1 }
  if (j < nc) { W[k, k + nr] <- 1; W[k + nr, k] <- 1 }
}
unit <- rep(seq_len(J), each = 6L)
N <- length(unit)
X <- matrix(1, N, 1L)
ntr <- rep(10L, N)
phi_true <- as.numeric(scale(rnorm(J, 0, 0.5), scale = FALSE))
y <- as.integer(rbinom(N, ntr, plogis(2.0 + phi_true[unit])))
PRI <- list(beta_sd = 0.7, tau_shape = 2, tau_rate = 1)

if (arm == "gibbs") {
  setwd("C:/GillesC/Documents/dev/tulpa")
  options(pkg.build_extra_flags = FALSE)
  suppressMessages(devtools::load_all(quiet = TRUE))
  al <- adjacency_to_list_tulpa(W)
  res <- lapply(1:4, function(ch) {
    set.seed(300 + ch)
    cpp_pg_binomial_gibbs_spatial(y, ntr, X, rep(1L, N), 0L, unit, J,
      al$adj_list, al$n_neighbors, 60000L, 5000L, 1L,
      prior_beta_sd = PRI$beta_sd, prior_tau_shape = PRI$tau_shape,
      prior_tau_rate = PRI$tau_rate, verbose = FALSE)
  })
  draws <- do.call(rbind, lapply(res, function(r)
    cbind(b0 = r$beta[, 1], tau = r$tau, phi1 = r$spatial[, 1])))
  saveRDS(draws, out)
  quit(save = "no")
}

Q <- diag(rowSums(W)) - W
lpost <- function(th) {
  b <- th[1]; ph <- c(th[2:J], -sum(th[2:J])); tau <- exp(th[J + 1L])
  sum(dbinom(y, ntr, plogis(b + ph[unit]), log = TRUE)) +
    dnorm(b, 0, PRI$beta_sd, log = TRUE) +
    0.5 * (J - 1) * log(tau) - 0.5 * tau * drop(crossprod(ph, Q %*% ph)) +
    dgamma(tau, PRI$tau_shape, PRI$tau_rate, log = TRUE) + th[J + 1L]
}
d <- J + 1L
run_am <- function(seed, n_iter, burn) {
  set.seed(seed)
  th <- c(1, rep(0, J - 1L), 0)
  lp <- lpost(th)
  mu <- th; C <- diag(0.01, d); L <- chol(diag(0.02, d))
  keep <- matrix(NA_real_, n_iter - burn, d)
  for (it in seq_len(n_iter)) {
    prop <- th + drop(crossprod(L, rnorm(d)))
    lpp <- lpost(prop)
    if (is.finite(lpp) && log(runif(1)) < lpp - lp) { th <- prop; lp <- lpp }
    if (it <= burn) {
      w <- 1 / it
      dm <- th - mu; mu <- mu + w * dm
      C <- (1 - w) * C + w * tcrossprod(dm)
      if (it %% 500 == 0 && it > 2000) L <- chol((2.38^2 / d) * C + diag(1e-8, d))
    } else {
      keep[it - burn, ] <- th
    }
  }
  keep
}
ch <- lapply(1:4, function(k) run_am(800 + k, 300000L, 60000L))
th <- do.call(rbind, ch)
saveRDS(cbind(b0 = th[, 1], tau = exp(th[, J + 1L]), phi1 = th[, 2]), out)
