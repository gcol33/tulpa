# Regularizing hyperpriors on (sigma, alpha).
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R.

# --- regularizing hyperpriors on (sigma, alpha) ----------------------------
#
# Joint nested-Laplace defaults to a flat hyperprior over the outer grid:
# `log_marginal[i]` is proportional to `p(y | theta_i)` with theta uniform
# over grid cells. The donor amplitude `sigma` is jointly identified by
# every arm; `alpha` is identified only by the copy arm and is right-
# skewed at small copy-arm sample size. A regularizing hyperprior on
# either axis shrinks its upper tail without biasing the modal cell when
# the data identifies it.
#
# Two families, applicable to both sigma and alpha:
#
#  * "pc.prec" (Penalized Complexity, Simpson et al. 2017):
#       pi(theta) = lambda exp(-lambda theta),  lambda = -log(alpha)/U
#    Calibrated by `P(theta > U) = alpha`. Exponential tail decay;
#    concentrates mass near zero only as much as needed. Base model at
#    theta=0 (zero amplitude, no copy). Default-friendly: pick U at the
#    upper end of plausible values and alpha small.
#
#  * "half_normal":
#       pi(theta) = (2/(scale*sqrt(2*pi))) exp(-theta^2/(2*scale^2))
#    Calibrated by `scale`. Sharper decay than PC.
#
# Both densities accept the closed half-line [0, Inf) so the theta=0 grid
# cell carries a finite prior contribution (PC: log(lambda); half-normal:
# log(2/(scale*sqrt(2*pi)))). Both are normalized log-densities -- the
# additive constant is shared across grid cells, doesn't bias moments,
# and is kept for downstream log-evidence consumers.
#
# Returns NULL for spec = NULL (flat prior, no contribution). Returns
# a function(theta) -> log_density (vectorised, theta >= 0) otherwise.
# Validation errors are raised on malformed specs.
.joint_parse_sigma_prior <- function(spec, axis_label) {
    if (is.null(spec)) return(NULL)
    if (!is.list(spec) || length(spec) < 2L) {
        stop(sprintf(
            "`%s` must be a list of the form list(<family>, <params>), e.g. list(\"pc.prec\", c(U = 1.0, alpha = 0.01)).",
            axis_label), call. = FALSE)
    }
    fam <- as.character(spec[[1L]])
    pp  <- as.numeric(spec[[2L]])
    switch(fam,
        "pc.prec" = {
            if (length(pp) != 2L || !all(is.finite(pp)) ||
                pp[1L] <= 0 || pp[2L] <= 0 || pp[2L] >= 1) {
                stop(sprintf(
                    "`%s = list(\"pc.prec\", c(U, alpha))` requires U > 0 and 0 < alpha < 1; got U=%s, alpha=%s.",
                    axis_label, format(pp[1L]), format(pp[2L])),
                    call. = FALSE)
            }
            U <- pp[1L]; a <- pp[2L]
            lambda <- -log(a) / U
            # PC prior density at theta=0 is the finite limit `lambda`,
            # not zero. The strict `s > 0` would zero the boundary grid
            # cell and drop posterior mass on the base-model atom
            # (sigma=0: silent block; alpha=0: no copy). Accept the
            # closed half-line [0, Inf).
            function(s) {
                ok <- is.finite(s) & s >= 0
                out <- rep(-Inf, length(s))
                out[ok] <- log(lambda) - lambda * s[ok]
                out
            }
        },
        "half_normal" = {
            if (length(pp) != 1L || !is.finite(pp) || pp <= 0) {
                stop(sprintf(
                    "`%s = list(\"half_normal\", scale)` requires scale > 0; got scale=%s.",
                    axis_label, format(pp)), call. = FALSE)
            }
            sc <- pp
            const <- log(2) - 0.5 * log(2 * pi) - log(sc)
            # Half-normal density at sigma=0 is the finite peak
            # 2/(scale*sqrt(2*pi)), not zero. Same closed-half-line
            # rationale as the PC branch above.
            function(s) {
                ok <- is.finite(s) & s >= 0
                out <- rep(-Inf, length(s))
                out[ok] <- const - s[ok]^2 / (2 * sc^2)
                out
            }
        },
        stop(sprintf(
            "Unknown hyperprior family for `%s`: '%s'. Supported families: 'pc.prec', 'half_normal'.",
            axis_label, fam), call. = FALSE)
    )
}

# Per-cell log-hyperprior contribution over a joint theta-grid. NULL fns
# contribute zero. Sums independent priors on sigma (donor amplitude)
# and alpha (copy coefficient). Cells missing either column (no-copy
# paths skip the alpha term) get zero contribution from that prior.
.joint_hp_vec_for_grids <- function(theta_grid, fn_sigma, fn_alpha) {
    if (is.null(fn_sigma) && is.null(fn_alpha)) return(NULL)
    if (is.null(theta_grid) || !is.matrix(theta_grid)) return(NULL)
    n <- nrow(theta_grid)
    out <- numeric(n)
    cn <- colnames(theta_grid)
    if (!is.null(fn_sigma) && "sigma" %in% cn) {
        out <- out + fn_sigma(as.numeric(theta_grid[, "sigma"]))
    }
    if (!is.null(fn_alpha) && "alpha" %in% cn) {
        out <- out + fn_alpha(as.numeric(theta_grid[, "alpha"]))
    }
    out
}
