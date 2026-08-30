# Is the low outer k-hat on a SKEWED target a property of the target, or of the
# radius cap?
#
# Ground truth. For a target whose right tail is N(0, s^2) scored against a
# Gaussian proposal of variance v < s^2, the importance weight has a Pareto tail
# of index 1 / (1 - v/s^2), i.e.
#
#     k_true = 1 - v / s^2.
#
# A skew-normal(scale s, shape alpha) has right-tail scale s and variance
# s^2 (1 - 2 delta^2 / pi), delta = alpha / sqrt(1 + alpha^2). So the k-hat a
# moment-matched Gaussian proposal SHOULD read on it is known in closed form and
# RISES with skewness -- 0.509 at alpha = 2, 0.612 at 5, 0.632 at 12.
#
# This script scores the same targets through .nested_is_pareto_k at the exact
# moment-matched proposal, capped (as the engine does) and uncapped, at a large
# draw budget so the k-hat is precise.

suppressMessages(library(tulpa))
tp <- asNamespace("tulpa")

sn_lg <- function(s, alpha) function(u)
    stats::dnorm(u / s, log = TRUE) + stats::pnorm(alpha * u / s, log.p = TRUE)
t_lg  <- function(s, df) function(u) stats::dt(u / s, df, log = TRUE)

moments_of <- function(lg, s, lo = -80, hi = 80, n = 400001L) {
    u <- seq(lo * s, hi * s, length.out = n)
    l <- lg(u); l <- l - max(l); w <- exp(l); w <- w / sum(w)
    m <- sum(w * u); v <- sum(w * (u - m)^2)
    list(mean = m, var = v, skew = sum(w * (u - m)^3) / v^1.5)
}

score <- function(lg, mu, sd, cap, n = 20000L, seed = 1L) {
    set.seed(seed)
    L <- matrix(sd, 1, 1)
    kd <- tp$.nested_is_pareto_k(mu, L,
            function(U) lg(U[, 1]), n_samples = n, radius_cap = cap)
    c(k = kd$pareto_k, ess = kd$is_ess, n_eval = kd$n_eval)
}

s <- 0.4
rows <- list()
for (spec in list(list(kind = "skew",  alpha = 2,  df = Inf),
                  list(kind = "skew",  alpha = 5,  df = Inf),
                  list(kind = "skew",  alpha = 12, df = Inf),
                  list(kind = "heavy", alpha = 0,  df = 8),
                  list(kind = "heavy", alpha = 0,  df = 3),
                  list(kind = "gauss", alpha = 0,  df = Inf))) {
    lg <- switch(spec$kind,
                 skew  = sn_lg(s, spec$alpha),
                 heavy = t_lg(s, spec$df),
                 gauss = function(u) stats::dnorm(u / s, log = TRUE))
    mm <- moments_of(lg, s)
    k_true <- if (spec$kind == "skew") 1 - mm$var / s^2
              else if (spec$kind == "gauss") 0 else NA_real_

    # The engine's own cap on this target: the grid-node envelope. Use the grid
    # the sweep's best-resolved cell uses (41 nodes over +/- 12 s).
    u_grid <- matrix(seq(-12 * s, 12 * s, length.out = 41L), ncol = 1L)
    cap <- tp$.nested_grid_radius_cap(u_grid, mm$mean, matrix(sqrt(mm$var), 1, 1))

    for (sd_i in 1:5) {
        cp <- score(lg, mm$mean, sqrt(mm$var), cap,  seed = sd_i)
        un <- score(lg, mm$mean, sqrt(mm$var), Inf, seed = sd_i)
        rows[[length(rows) + 1L]] <- data.frame(
            kind = spec$kind, alpha = spec$alpha, df = spec$df, seed = sd_i,
            true_skew = mm$skew, k_true = k_true, cap = cap,
            k_capped = cp[["k"]], n_eval_capped = cp[["n_eval"]],
            k_uncapped = un[["k"]], n_eval_uncapped = un[["n_eval"]])
    }
}
d <- do.call(rbind, rows)
a <- aggregate(cbind(true_skew, k_true, cap, k_capped, n_eval_capped,
                     k_uncapped, n_eval_uncapped) ~ kind + alpha + df, d, median)
print(a, digits = 4, row.names = FALSE)
utils::write.csv(d, file.path("dev_notes", "issue629", "cap629.csv"),
                 row.names = FALSE)
