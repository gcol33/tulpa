# gcol33/tulpa#630: what the candidate dispatch changes on `tulpa_re_cov_nested`,
# and whether the change is a better proposal or a laundered verdict.
#
# `test-psis.R`'s "outer k-hat orders well-identified below tiny-binary" is the
# arbiter CLAUDE.md's claim rests on -- that a small-group binary RE-covariance
# posterior is genuinely skewed, so its high k-hat is a correct signal rather
# than a defect. This drives its two fixtures through each candidate layer
# separately so the movement is attributable:
#
#   gm      the raw grid-moment / mode-Hessian Gaussian -- what this backend
#           reported before #630
#   gauss   that Gaussian after moment-matching refinement
#   full    the shipped dispatch (adds the skew-normal rescue; no grid here, so
#           the mixture declines)
#
# The skew-normal has Gaussian tails on BOTH sides, so it can absorb asymmetry
# and NOT a heavy tail. Which candidate wins therefore says which the posterior
# had.

suppressMessages(library(tulpa))
tp <- asNamespace("tulpa")
`%||%` <- function(a, b) if (is.null(a)) b else a

# The two fixtures, verbatim from tests/testthat/test-psis.R.
fit_well <- function(seed, k_samples = 150L) {
    set.seed(seed); G <- 30L; per <- 25L; n <- G * per
    grp <- rep(seq_len(G), each = per); x <- rnorm(n)
    eta <- 0.2 + 0.5 * x + rnorm(G, 0, 0.8)[grp] + rnorm(G, 0, 0.5)[grp] * x
    y   <- eta + rnorm(n, 0, 0.5)
    rt  <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
                correlated = TRUE)
    tulpa_re_cov_nested(y, rep(1L, n), cbind(1, x), rt, family = "gaussian",
                        phi = 0.25, hyperprior = "pc_lkj",
                        control = list(diagnose_k = TRUE, k_samples = k_samples))
}
fit_tiny <- function(seed, k_samples = 150L) {
    set.seed(seed); G <- 25L; per <- 3L; n <- G * per
    grp <- rep(seq_len(G), each = per); x <- rnorm(n)
    eta <- -0.2 + 0.4 * x + rnorm(G, 0, 1.0)[grp] + rnorm(G, 0, 0.8)[grp] * x
    y   <- rbinom(n, 1L, plogis(eta))
    rt  <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
                correlated = TRUE)
    tulpa_re_cov_nested(y, rep(1L, n), cbind(1, x), rt, family = "binomial",
                        hyperprior = "pc_lkj",
                        control = list(diagnose_k = TRUE, k_samples = k_samples))
}

# Trace the layers: wrap the scorers so the probe reads the objects the shipped
# run produced, on the shipped run's own RNG stream.
REC  <- new.env(parent = emptyenv())
orig <- list(g = tp$.k_score_gaussian, mix = tp$.k_score_mixture,
             sk = tp$.k_score_skew)
wrap <- function(nm, fn) function(...) {
    out <- fn(...); REC[[nm]] <- out; REC[[paste0(nm, "_called")]] <- TRUE; out
}
utils::assignInNamespace(".k_score_gaussian", wrap("g",   orig$g),   ns = "tulpa")
utils::assignInNamespace(".k_score_mixture",  wrap("mix", orig$mix), ns = "tulpa")
utils::assignInNamespace(".k_score_skew",     wrap("sk",  orig$sk),  ns = "tulpa")

row_of <- function(label, seed, f) {
    rm(list = ls(REC), envir = REC)
    fit <- f(seed)
    g   <- REC$g
    gm  <- g$gm %||% g
    num <- function(x) if (is.null(x) || !length(x)) NA_real_ else as.numeric(x)[1]
    data.frame(
        fixture = label, seed = seed,
        k_gm    = if (is.list(gm)) num(gm$pareto_k) else NA_real_,
        k_gauss = if (is.list(g))  num(g$pareto_k)  else NA_real_,
        k_skew  = if (is.list(REC$sk)) num(REC$sk$pareto_k) else NA_real_,
        k_full  = num(fit$pareto_k),
        src     = fit$pareto_k_proposal_source %||% NA_character_,
        skew_scored = isTRUE(REC$sk_called),
        mix_scored  = isTRUE(REC$mix_called),
        outer_skew  = if (is.null(fit$outer_skew)) NA_real_
                      else max(abs(fit$outer_skew)),
        stringsAsFactors = FALSE)
}

rows <- c(lapply(1:5, function(s) row_of("well_identified", 100L + s, fit_well)),
          lapply(1:5, function(s) row_of("tiny_binary",     200L + s, fit_tiny)))
d <- do.call(rbind, rows)
utils::write.csv(d, file.path("dev_notes", "issue630", "recov630.csv"),
                 row.names = FALSE)
print(d, digits = 4, row.names = FALSE)

cat("\n== medians by fixture ==\n")
print(aggregate(cbind(k_gm, k_gauss, k_skew, k_full) ~ fixture, d,
                function(x) round(median(x, na.rm = TRUE), 3),
                na.action = na.pass))
cat("\nskew candidate scored on:", sum(d$skew_scored), "of", nrow(d), "fits\n")
cat("adopted source:\n"); print(table(d$fixture, d$src))
