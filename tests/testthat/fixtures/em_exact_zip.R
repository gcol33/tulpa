.em_zip_fixture <- function() {
  y <- c(0, 0, 0, 1, 0, 2)
  group <- rep(1:3, each = 2L)
  stopifnot(length(y) == 6L, length(unique(group)) == 3L,
            all(tabulate(group) == 2L))
  sigma <- 1.25
  prior_sd <- 1.5
  X <- matrix(1, length(y), 1, dimnames = list(NULL, "intercept"))
  weights <- function(fits) {
    if (!length(fits)) return(ifelse(y > 0, 1, .6))
    alpha <- fits$gate$mode[1L]
    rate <- exp(fits$rate$mode[1L] + fits$rate$mode[1L + group])
    log_nonstructural <- plogis(alpha, log.p = TRUE) - rate
    ifelse(y > 0, 1, plogis(log_nonstructural - plogis(-alpha, log.p = TRUE)))
  }
  encode <- function(z, ...) {
    list(gate = list(y = rep(c(1, 0), each = length(y)), X = rbind(X, X),
                     weights = c(z, 1 - z), family = "binomial"),
         rate = list(y = y, X = X, weights = z, family = "poisson",
                     re_list = list(list(idx = group, n_groups = 3L, sigma = sigma))))
  }
  e_step <- function(fits, ...) list(weights = weights(fits))
  e_sample <- function(fits, n_mc, ...) {
    w <- weights(fits)
    lapply(seq_len(n_mc), function(i) rbinom(length(y), 1L, w))
  }
  # The reference marginalizes the binary labels analytically and uses base R
  # densities independently of the engine's encoding and native likelihood.
  log_marginal <- function(theta) {
    alpha <- theta[1L]
    rate <- exp(theta[2L] + theta[2L + group])
    lp1 <- plogis(alpha, log.p = TRUE) + dpois(y, rate, log = TRUE)
    lp0 <- rep(plogis(-alpha, log.p = TRUE), length(y))
    maximum <- pmax(lp0, lp1)
    ll <- lp1
    zero <- which(y == 0)
    ll[zero] <- maximum[zero] +
      log(exp(lp0[zero] - maximum[zero]) + exp(lp1[zero] - maximum[zero]))
    sum(ll) + sum(dnorm(theta[1:2], 0, prior_sd, log = TRUE)) +
      sum(dnorm(theta[3:5], 0, sigma, log = TRUE))
  }
  list(y = y, group = group, sigma = sigma, beta_prior = list(sd = prior_sd),
       e_step = e_step, e_sample = e_sample, encode = encode,
       log_marginal = log_marginal)
}

# Coordinate slice sampling of theta with Z marginalized analytically;
# stepping out and shrinkage leave each scalar conditional invariant.
.em_zip_reference <- function(fixture, n_iter, warmup) {
  theta <- numeric(5L)
  logp <- fixture$log_marginal(theta)
  draws <- matrix(NA_real_, n_iter - warmup, 5L)
  z <- matrix(NA_integer_, n_iter - warmup, length(fixture$y))
  evaluations <- 1L
  evaluate <- function(t) {
    evaluations <<- evaluations + 1L
    fixture$log_marginal(t)
  }
  for (i in seq_len(n_iter)) {
    for (j in seq_along(theta)) {
      level <- logp - rexp(1L)
      left <- theta[j] - runif(1L, 0, 2)
      right <- left + 2
      candidate <- theta
      candidate[j] <- left
      while (evaluate(candidate) > level) {
        left <- left - 2
        candidate[j] <- left
      }
      candidate[j] <- right
      while (evaluate(candidate) > level) {
        right <- right + 2
        candidate[j] <- right
      }
      repeat {
        candidate[j] <- runif(1L, left, right)
        proposed_logp <- evaluate(candidate)
        if (proposed_logp >= level) break
        if (candidate[j] < theta[j]) left <- candidate[j] else right <- candidate[j]
      }
      theta <- candidate
      logp <- proposed_logp
    }
    if (i > warmup) {
      row <- i - warmup
      draws[row, ] <- theta
      rate <- exp(theta[2L] + theta[2L + fixture$group])
      w <- plogis(plogis(theta[1L], log.p = TRUE) - rate -
                    plogis(-theta[1L], log.p = TRUE))
      w[fixture$y > 0] <- 1
      z[row, ] <- rbinom(length(w), 1L, w)
    }
  }
  list(draws = draws, latent = z, density_evaluations = evaluations)
}

.em_summary_function <- function(x) {
  c(mean = mean(x), sd = sd(x), q025 = unname(quantile(x, .025)),
    median = median(x), q975 = unname(quantile(x, .975)))
}

.em_chain_summary <- function(draws, batches = 40L) {
  stopifnot(nrow(draws) >= batches * 50L, !is.null(colnames(draws)))
  batch <- cut(seq_len(nrow(draws)), breaks = batches, labels = FALSE)
  do.call(rbind, lapply(seq_len(ncol(draws)), function(j) {
    estimate <- .em_summary_function(draws[, j])
    by_batch <- vapply(split(draws[, j], batch), .em_summary_function, numeric(5L))
    data.frame(parameter = colnames(draws)[j], statistic = names(estimate),
               estimate = unname(estimate),
               mcse = apply(by_batch, 1L, sd) / sqrt(batches), row.names = NULL)
  }))
}

.em_summary_compare <- function(summary, reference) {
  out <- merge(summary, reference, by = c("parameter", "statistic"),
                suffixes = c("", "_reference"), sort = FALSE)
  stopifnot(nrow(out) == nrow(summary), nrow(out) == nrow(reference))
  out$combined_mcse <- sqrt(out$mcse^2 + out$mcse_reference^2)
  out$z_mcse <- (out$estimate - out$estimate_reference) / out$combined_mcse
  out
}
