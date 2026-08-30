# Does the outer k-hat on a SKEWED (light-tailed) target depend on the draw
# budget it is read at?
#
# The sweep reads every candidate at the shipped k_samples = 500. This script
# holds the target, the grid and the proposal family fixed and moves ONLY
# n_samples, on the best-resolved grid cell of the sweep (41 nodes, +/- 12 s).

suppressMessages(library(tulpa))
tp <- asNamespace("tulpa")
`%||%` <- function(a, b) if (is.null(a)) b else a

s <- 0.4
mk <- function(kind, alpha = 0, df = Inf) switch(kind,
    gaussian = function(u) stats::dnorm(u / s, log = TRUE),
    skew     = function(u) stats::dnorm(u / s, log = TRUE) +
                           stats::pnorm(alpha * u / s, log.p = TRUE),
    heavy    = function(u) stats::dt(u / s, df, log = TRUE))

mk_res <- function(lg, n_nodes, hw) {
    u <- seq(-hw * s, hw * s, length.out = n_nodes)
    l <- lg(u); l <- l - max(l); w <- exp(l); w <- w / sum(w)
    list(theta_grid = matrix(exp(u), ncol = 1L,
                             dimnames = list(NULL, "b1.sigma")),
         weights = w, axis_offsets = c(0L, 1L),
         blocks = list(list(type = "icar")))
}

SPECS <- list(list(l = "gaussian",      kind = "gaussian"),
              list(l = "skew a=2",      kind = "skew", alpha = 2),
              list(l = "skew a=5",      kind = "skew", alpha = 5),
              list(l = "skew a=12",     kind = "skew", alpha = 12),
              list(l = "heavy df=8",    kind = "heavy", df = 8),
              list(l = "heavy df=3",    kind = "heavy", df = 3))
BUDGETS <- c(200L, 500L, 1000L, 2000L, 5000L, 10000L, 20000L)

rows <- list(); i <- 0L
for (sp in SPECS) {
    lg    <- mk(sp$kind, sp$alpha %||% 0, sp$df %||% Inf)
    res   <- mk_res(lg, 41L, 12)
    refit <- function(theta_mat) lg(log(theta_mat[, 1]))
    for (n in BUDGETS) {
        prep <- tp$.joint_pareto_prepare(res, refit, n, NULL)
        vary <- tp$.joint_pareto_vary_axes(prep$Su)
        spec <- tp$.joint_cand_spec(prep, vary, refit)
        for (sd_i in 1:10) {
            set.seed(sd_i)
            g <- tp$.k_score_gaussian(spec, n)
            gm <- g$gm %||% g
            i <- i + 1L
            num <- function(x) if (is.null(x) || !length(x)) NA_real_ else as.numeric(x)[1]
            rows[[i]] <- data.frame(
                label = sp$l, n_samples = n, seed = sd_i,
                k_gm    = if (is.list(gm)) num(gm$pareto_k) else NA_real_,
                k_gauss = if (is.list(g))  num(g$pareto_k)  else NA_real_,
                ess_gm  = if (is.list(gm)) num(gm$is_ess)   else NA_real_)
        }
    }
    cat("done", sp$l, "\n")
}
d <- do.call(rbind, rows)
utils::write.csv(d, file.path("dev_notes", "issue629", "budget629.csv"),
                 row.names = FALSE)
cat("\n== median grid-moment k-hat by draw budget ==\n")
print(round(tapply(d$k_gm, list(d$label, d$n_samples), median), 3))
cat("\n== median moment-matched (best single Gaussian) k-hat ==\n")
print(round(tapply(d$k_gauss, list(d$label, d$n_samples), median), 3))
