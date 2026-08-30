# gcol33/tulpa#631: is the budget-dependent outer k-hat the ESTIMATOR or the
# TARGET?
#
# The observation: holding target, grid and proposal fixed and moving only
# `k_samples`, the reported k-hat climbs monotonically on every non-Gaussian
# target, while a closed-form Pareto control is flat.
#
# What settles it is reading the SAME exceedances three ways:
#
#   k_psis    the shipped Zhang-Stephens GPD fit (`tulpa_psis`)
#   k_hill    the Hill estimator, 1/alpha from the mean log excess -- an
#             independent, assumption-light tail index on the same order
#             statistics
#   k_theory  the closed form for this family of ratios, read at the tail depth
#             the fit actually probes
#
# For a log-ratio that is QUADRATIC in the whitened draw, log w = c z^2 with
# z ~ N(0,1) under the proposal,
#
#     P(w > t) = P(|z| > sqrt(log t / c)) ~ exp(-log t / (2c)) / sqrt(log t / c)
#     alpha_eff(t) = 1/(2c) + 1/(2 log t),   k_eff(t) = 2c / (1 + c / log t).
#
# So k_eff RISES with the tail depth and is BOUNDED ABOVE by 2c. The PSIS tail
# length is min(S/5, 3 sqrt(S)), so the fitted tail FRACTION is 3/sqrt(S) and
# SHRINKS as the budget grows -- the fit probes deeper at a larger budget, and a
# rising k-hat is expected. The bound 2c is what says whether the rise is the
# estimator tracking the true tail or overshooting it.
#
# c is read off the target directly: a regression of lr on z^2 over the upper
# half of the radius range, where the quadratic term dominates the slowly
# varying part. Its slope IS c.

suppressMessages(library(tulpa))
tp <- asNamespace("tulpa")

# Hill estimator on the top `m` order statistics: k = 1 / alpha.
hill_k <- function(w, m) {
    w <- sort(w[is.finite(w) & w > 0], decreasing = TRUE)
    if (length(w) <= m || m < 5L) return(NA_real_)
    mean(log(w[seq_len(m)])) - log(w[m + 1L])
}

read_three <- function(lg, mu, sd, n, seed) {
    set.seed(seed)
    z  <- stats::rnorm(n)
    u  <- mu + sd * z
    lr <- lg(u) + 0.5 * z^2                  # log target - log q, up to a const
    fin <- is.finite(lr)
    lr <- lr[fin]; z2 <- z[fin]^2
    m  <- tp$.psis_tail_len(length(lr))
    hi <- z2 >= stats::quantile(z2, 0.5)
    cc <- unname(stats::coef(stats::lm(lr[hi] ~ z2[hi]))[2L])
    thr  <- unname(stats::quantile(lr, 1 - m / length(lr)))
    logt <- max(lr) - thr                    # tail depth the fit actually probes
    kth  <- if (is.finite(logt) && logt > 0) 2 * cc / (1 + cc / logt) else NA_real_
    c(k_psis = tulpa_psis(lr)$pareto_k,
      k_hill = hill_k(exp(lr - thr), m),
      c_hat = cc, k_bound = 2 * cc, k_theory = kth,
      tail_m = m, tail_frac = m / length(lr), log_depth = logt)
}

s <- 0.4
SPECS <- list(
    list(l = "gaussian",   lg = function(u) stats::dnorm(u / s, log = TRUE)),
    list(l = "skew a=5",   lg = function(u) stats::dnorm(u / s, log = TRUE) +
                                            stats::pnorm(5 * u / s, log.p = TRUE)),
    list(l = "heavy df=8", lg = function(u) stats::dt(u / s, 8, log = TRUE)),
    list(l = "heavy df=3", lg = function(u) stats::dt(u / s, 3, log = TRUE)))

moments_of <- function(lg, lo = -80, hi = 80, n = 400001L) {
    u <- seq(lo * s, hi * s, length.out = n)
    l <- lg(u); l <- l - max(l); w <- exp(l); w <- w / sum(w)
    m <- sum(w * u); list(mu = m, sd = sqrt(sum(w * (u - m)^2)))
}

BUDGETS <- c(500L, 2000L, 10000L, 50000L, 200000L)
rows <- list(); i <- 0L
for (sp in SPECS) {
    mm <- moments_of(sp$lg)
    for (n in BUDGETS) for (sd_i in 1:8) {
        r <- read_three(sp$lg, mm$mu, mm$sd, n, sd_i)
        i <- i + 1L
        rows[[i]] <- data.frame(
            label = sp$l, n_samples = n, seed = sd_i,
            k_psis = unname(r[1]), k_hill = unname(r[2]), c_hat = unname(r[3]),
            k_bound = unname(r[4]), k_theory = unname(r[5]),
            tail_m = unname(r[6]), tail_frac = unname(r[7]),
            log_depth = unname(r[8]), stringsAsFactors = FALSE)
    }
    cat("done", sp$l, "\n")
}
d <- do.call(rbind, rows)
utils::write.csv(d, file.path("dev_notes", "issue631", "tail631.csv"),
                 row.names = FALSE)

cat("\n== the same exceedances, three ways ==\n")
for (lb in unique(d$label)) {
    x <- d[d$label == lb, ]
    cat("\n", lb, "\n", sep = "")
    for (n in BUDGETS) {
        y <- x[x$n_samples == n, ]
        cat(sprintf(
            "  n=%7d m=%5d (%4.1f%%) depth=%5.2f  psis=%8.3f  hill=%7.3f  theory=%6.3f  2c=%6.3f\n",
            n, round(median(y$tail_m)), 100 * median(y$tail_frac),
            median(y$log_depth), median(y$k_psis, na.rm = TRUE),
            median(y$k_hill, na.rm = TRUE), median(y$k_theory, na.rm = TRUE),
            median(y$k_bound, na.rm = TRUE)))
    }
}
