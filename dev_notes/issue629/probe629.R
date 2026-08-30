# gcol33/tulpa#629 step (1): does a fit whose reported outer k-hat came from a
# SINGLE-GAUSSIAN candidate, on a hyperparameter posterior a rescue candidate
# fits materially better, actually occur?
#
# The shipped dispatch (.joint_pareto_score_dispatch) already scores three
# proposal families and adopts the best, but each rescue sits behind a gate:
#   * the grid mixture is SCORED only when proposal_source == "grid_moment" AND
#     the grid-moment k is at or above the good band, and ADOPTED only when the
#     hull check says the grid covers the posterior AND it improves on gm;
#   * the skew-normal is SCORED only when the chosen k is above the good band
#     AND the whitened skewness clears .K_DIAG_SKEW_MIN / the z-gate, and
#     ADOPTED only when it strictly improves.
# So the regime #629 posits can only arise through a gate. This probe traces
# what the shipped dispatch actually scored, computes the candidates it skipped
# as counterfactuals, and reports where a skipped-or-rejected candidate is
# materially better than the number the fit reports.
#
# The synthetic target is the device tests/testthat/test-joint-pareto-k-*.R
# already use: a `res` carrying only theta_grid / weights / axis_offsets /
# blocks, plus an analytic `refit_log_marginal`. A `log` axis carries
# log_jac = 0, so the target the PSIS sees is exactly g(u) with u = log(theta):
# the shape is CONTROLLED, not inferred, and the true skewness / excess
# kurtosis are known by quadrature.

suppressMessages(library(tulpa))
tp <- asNamespace("tulpa")
`%||%` <- function(a, b) if (is.null(a)) b else a

# --------------------------------------------------------------------------- #
# targets on the unconstrained axis                                            #
# --------------------------------------------------------------------------- #

mk_target <- function(kind, s = 0.4, alpha = 0, df = Inf) {
    lg <- switch(kind,
        gaussian = function(u) stats::dnorm(u / s, log = TRUE),
        skew     = function(u) stats::dnorm(u / s, log = TRUE) +
                               stats::pnorm(alpha * u / s, log.p = TRUE),
        heavy    = function(u) stats::dt(u / s, df, log = TRUE),
        skewt    = function(u) {
            z <- u / s
            stats::dt(z, df, log = TRUE) +
                stats::pt(alpha * z * sqrt((df + 1) / (z^2 + df)), df + 1,
                          log.p = TRUE)
        },
        stop("unknown target"))
    list(lg = lg, kind = kind, s = s, alpha = alpha, df = df,
         cfg = sprintf("%s(a=%g,df=%g)", kind, alpha, df))
}

true_moments <- function(tg, lo = -80, hi = 80, n = 400001L) {
    u <- seq(lo * tg$s, hi * tg$s, length.out = n)
    l <- tg$lg(u); l <- l - max(l)
    w <- exp(l); w <- w / sum(w)
    m <- sum(w * u); v <- sum(w * (u - m)^2)
    list(sd = sqrt(v), skew = sum(w * (u - m)^3) / v^1.5,
         exkurt = sum(w * (u - m)^4) / v^2 - 3)
}

mk_res <- function(tg, n_nodes, half_width) {
    u <- seq(-half_width * tg$s, half_width * tg$s, length.out = n_nodes)
    l <- tg$lg(u); l <- l - max(l)
    w <- exp(l); w <- w / sum(w)
    list(theta_grid = matrix(exp(u), ncol = 1L,
                             dimnames = list(NULL, "b1.sigma")),
         weights = w, axis_offsets = c(0L, 1L),
         blocks = list(list(type = "icar")))
}

mk_refit <- function(tg) function(theta_mat) tg$lg(log(theta_mat[, 1]))

# --------------------------------------------------------------------------- #
# trace the dispatch: what it scored, and what it skipped                      #
# --------------------------------------------------------------------------- #
# The three candidate scorers are wrapped so the probe reads the EXACT objects
# the shipped run produced (same RNG stream, same draws), rather than
# re-deriving them from a fresh seed and comparing two different realizations.

REC <- new.env(parent = emptyenv())
orig <- list(score   = tp$.joint_pareto_score,
             mixture = tp$.joint_pareto_score_mixture,
             skew    = tp$.joint_pareto_score_skew)

wrap <- function(name, fn) function(...) {
    out <- fn(...)
    REC[[name]] <- out
    REC[[paste0(name, "_called")]] <- TRUE
    out
}
utils::assignInNamespace(".joint_pareto_score",         wrap("g",   orig$score),   ns = "tulpa")
utils::assignInNamespace(".joint_pareto_score_mixture", wrap("mix", orig$mixture), ns = "tulpa")
utils::assignInNamespace(".joint_pareto_score_skew",    wrap("sk",  orig$skew),    ns = "tulpa")

SINGLE_GAUSS <- c("grid_moment", "moment_matched", "mode_hessian")

score_all <- function(tg, n_nodes, half_width, n_samples = 500L, seed = 1L) {
    res   <- mk_res(tg, n_nodes, half_width)
    refit <- mk_refit(tg)
    na_row <- function(reason) data.frame(declined = reason,
                                          stringsAsFactors = FALSE)

    prep <- tp$.joint_pareto_prepare(res, refit, n_samples, NULL)
    if (tp$.k_is_decline(prep)) return(na_row(tp$.k_reason_of(prep)))
    vary <- tp$.joint_pareto_vary_axes(prep$Su)
    if (!length(vary)) return(na_row("no_varying_axis"))

    rm(list = ls(REC), envir = REC)
    set.seed(seed)
    shipped <- tp$.joint_pareto_score_dispatch(prep, vary, refit, n_samples)
    if (tp$.k_is_decline(shipped)) return(na_row(tp$.k_reason_of(shipped)))

    g   <- REC$g
    mix <- REC$mix; mix_scored <- isTRUE(REC$mix_called)
    sk  <- REC$sk;  sk_scored  <- isTRUE(REC$sk_called)

    # Counterfactuals for the candidates the dispatch never scored: continue the
    # SAME stream the shipped run left behind, so nothing is re-seeded.
    mom <- if (!is.null(g)) tp$.joint_pareto_wtd_moments(g$U, g$log_weights,
                                                         g$prop_u, g$prop_L)
           else NULL
    if (!mix_scored) {
        mix <- tryCatch(orig$mixture(prep, vary, refit, n_samples),
                        error = function(e) NULL)
    }
    if (!sk_scored && !is.null(mom)) {
        sk <- tryCatch(orig$skew(prep, vary, refit, n_samples,
                                 g$prop_u, g$prop_L, mom),
                       error = function(e) NULL)
    }

    gm   <- g$gm %||% g
    gm_k <- if (is.list(gm)) gm$pareto_k %||% NA_real_ else NA_real_

    covered <- NA
    if (!is.null(mix) && !is.null(g)) {
        of <- tp$.joint_pareto_hull_weight(
            g$gm_U %||% g$U, g$gm_lw %||% g$log_weights,
            mix$lo - tp$.K_DIAG_HULL_PAD * mix$s,
            mix$hi + tp$.K_DIAG_HULL_PAD * mix$s)
        covered <- is.finite(of) && of <= tp$.K_DIAG_HULL_TOL
    }

    data.frame(
        cfg = tg$cfg, kind = tg$kind, s = tg$s, alpha = tg$alpha, df = tg$df,
        n_nodes = n_nodes, half_width = half_width, seed = seed,
        k_gm      = gm_k,
        k_gauss   = if (is.null(g))   NA_real_ else g$pareto_k,
        k_mix     = if (is.null(mix)) NA_real_ else mix$pareto_k,
        k_skew    = if (is.null(sk))  NA_real_ else sk$pareto_k,
        k_shipped = shipped$best$pareto_k,
        src       = shipped$source,
        mix_scored = mix_scored, sk_scored = sk_scored,
        mix_built  = !is.null(mix), sk_built = !is.null(sk),
        covered = covered,
        skew_est  = if (is.null(mom)) NA_real_ else max(abs(mom$skew)),
        skew_gate = if (is.null(mom)) NA_real_
                    else max(tp$.K_DIAG_SKEW_MIN,
                             tp$.K_DIAG_SKEW_Z * tp$.skew_se(mom$n_eff)),
        declined = NA_character_, stringsAsFactors = FALSE)
}

# --------------------------------------------------------------------------- #
# the sweep                                                                    #
# --------------------------------------------------------------------------- #

TARGETS <- list(
    mk_target("gaussian", s = 0.4),
    mk_target("skew",  s = 0.4, alpha = 2),
    mk_target("skew",  s = 0.4, alpha = 5),
    mk_target("skew",  s = 0.4, alpha = 12),
    mk_target("heavy", s = 0.4, df = 12),
    mk_target("heavy", s = 0.4, df = 8),
    mk_target("heavy", s = 0.4, df = 4),
    mk_target("heavy", s = 0.4, df = 3),
    mk_target("heavy", s = 0.4, df = 2),
    mk_target("skewt", s = 0.4, alpha = 5, df = 4),
    mk_target("skewt", s = 0.4, alpha = 12, df = 3)
)
GRIDS <- expand.grid(n_nodes = c(5L, 9L, 15L, 25L, 41L),
                     half_width = c(3, 6, 12),
                     stringsAsFactors = FALSE)
SEEDS <- 1:20

COLS <- c("cfg","kind","s","alpha","df","n_nodes","half_width","seed",
          "k_gm","k_gauss","k_mix","k_skew","k_shipped","src","mix_scored",
          "sk_scored","mix_built","sk_built","covered","skew_est","skew_gate",
          "declined","true_skew","true_exkurt")

out <- list(); i <- 0L
for (tg in TARGETS) {
    tm <- true_moments(tg)
    for (r in seq_len(nrow(GRIDS))) for (sd in SEEDS) {
        i <- i + 1L
        row <- score_all(tg, GRIDS$n_nodes[r], GRIDS$half_width[r],
                         n_samples = 500L, seed = sd)
        row$cfg <- row$cfg %||% tg$cfg
        row$cfg <- tg$cfg; row$kind <- tg$kind; row$alpha <- tg$alpha
        row$df <- tg$df; row$n_nodes <- GRIDS$n_nodes[r]
        row$half_width <- GRIDS$half_width[r]; row$seed <- sd
        row$true_skew <- tm$skew; row$true_exkurt <- tm$exkurt
        for (nm in COLS) if (is.null(row[[nm]])) row[[nm]] <- NA
        out[[i]] <- row[, COLS]
    }
    cat(sprintf("done %-22s true_skew=%6.3f true_exkurt=%9.3f\n",
                tg$cfg, tm$skew, tm$exkurt))
}
d <- do.call(rbind, out)
utils::write.csv(d, file.path("dev_notes", "issue629", "sweep629.csv"),
                 row.names = FALSE)
cat("rows:", nrow(d), "  declines:", sum(!is.na(d$declined)), "\n")
