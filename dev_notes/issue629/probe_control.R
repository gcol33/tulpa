# Arbiter for the draw-budget reading: a target/proposal pair whose Pareto k is
# known in CLOSED FORM.
#
# Target N(0, sp^2) against proposal N(0, sq^2) with sq < sp. The importance
# weight is w = exp(a u^2) with a = (1/sq^2 - 1/sp^2)/2 and u ~ N(0, sq^2), so
#     P(w > t) = t^{-1 / (2 a sq^2)},  k = 2 a sq^2 = 1 - sq^2 / sp^2,
# exactly, with no pre-asymptotic polynomial factor. If the engine's PSIS reads
# that number stably across draw budgets, the harness is sound and a k-hat that
# CLIMBS with the budget on another target is a property of that target, not of
# the estimator wiring.

suppressMessages(library(tulpa))
tp <- asNamespace("tulpa")

BUDGETS <- c(200L, 500L, 1000L, 2000L, 5000L, 10000L, 20000L, 50000L)
RATIOS  <- c(0.9, 0.7, 0.5, 0.3)          # sq^2 / sp^2  ->  k_true = 1 - ratio

sp <- 1
rows <- list(); i <- 0L
for (r in RATIOS) {
    sq <- sqrt(r) * sp
    lt <- function(U) stats::dnorm(U[, 1], sd = sp, log = TRUE)
    for (n in BUDGETS) for (sd_i in 1:10) {
        set.seed(sd_i)
        kd <- tp$.nested_is_pareto_k(0, matrix(sq, 1, 1), lt,
                                     n_samples = n, radius_cap = Inf)
        i <- i + 1L
        rows[[i]] <- data.frame(ratio = r, k_true = 1 - r, n_samples = n,
                                seed = sd_i, k_hat = kd$pareto_k)
    }
    cat("done ratio", r, "\n")
}
d <- do.call(rbind, rows)
cat("\n== closed-form control: median k-hat by budget (rows = k_true) ==\n")
tb <- round(tapply(d$k_hat, list(sprintf("k_true=%.1f", d$k_true), d$n_samples),
                   median), 3)
print(tb)
utils::write.csv(d, file.path("dev_notes", "issue629", "control629.csv"),
                 row.names = FALSE)
