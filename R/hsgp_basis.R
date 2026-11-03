# Basis size and boundary factor of a Hilbert-space GP, from the lengthscales
# the basis has to represent.
#
# Riutort-Mayol et al. (2023, Statistics and Computing 33:17, Sect. 4.3.1) fit
# the relation between the number of basis functions per dimension `m`, the
# boundary factor `c` (the basis boundary is L = c S, S the half-width of the
# input domain) and the lengthscale `ell` at which the approximate covariance
# stays within 1% of the exact one in total variation (their Eq. 18). For the
# squared-exponential kernel, the kernel every tulpa HSGP basis approximates
# (`spectral_density_se`, src/hmc_hsgp_kernels.h):
#
#   m = 1.75 c / (ell / S)                        Eq. (19)
#   c >= 3.2 ell / S   and   c >= 1.2             Eq. (20)
#
# Read the other way, a basis (m, c) is accurate for lengthscales in
#   [1.75 c S / m,  c S / 3.2],
# the smallest lengthscale Eq. (19) allows and the largest Eq. (20) allows. In
# more than one dimension the paper applies the one-dimensional relations to
# each dimension (Sect. 4.3.1); tulpa's basis carries one `m` and one `c` for
# both coordinates of an isotropic lengthscale, so `m` is read off the widest
# dimension and `c` off the narrowest.
.HSGP_RULE <- list(m_coef = 1.75, c_coef = 3.2, c_min = 1.2)

# The basis sizes the HSGP constructors accept. M = m^2 basis functions enter
# every inner solve as dense columns.
.HSGP_M_BOUNDS <- c(3L, 50L)

# Half-widths S of the domain a coordinate matrix spans, per column: the
# bounding box `hsgp_center_L_2d()` (src/hmc_hsgp.h) centres the basis on.
.hsgp_half_widths <- function(coords) {
  rng <- apply(as.matrix(coords), 2L, range)
  (rng[2L, ] - rng[1L, ]) / 2
}

# The lengthscale window a basis (m, c) on half-widths S represents accurately.
.hsgp_lengthscale_window <- function(m, c, S) {
  r <- .HSGP_RULE
  c(lo = r$m_coef * c * max(S) / m, hi = c * min(S) / r$c_coef)
}

# The smallest (m, c) whose window holds [ell_lo, ell_hi]: `c` from Eq. (20) at
# the largest lengthscale, unless it is given, then `m` from Eq. (19) at the
# smallest one with that `c`. `m_req` is Eq. (19) before the constructors'
# bounds; past the upper bound the window does not reach ell_lo.
.hsgp_basis_size <- function(S, ell_lo, ell_hi = ell_lo, c = NULL) {
  r <- .HSGP_RULE
  if (is.null(c)) c <- max(r$c_min, r$c_coef * ell_hi / min(S))
  m_req <- ceiling(r$m_coef * c * max(S) / ell_lo)
  list(m = as.integer(min(max(m_req, .HSGP_M_BOUNDS[1L]), .HSGP_M_BOUNDS[2L])),
       c = c, m_req = m_req)
}

# Quantile of the PC prior on a range in dimension d,
# P(range < U) = alpha (src/pc_prior.h): its CDF is exp(-lambda r^(-d/2)) with
# lambda = -log(alpha) U^(d/2), so the p-quantile is (lambda / -log p)^(2/d).
.pc_range_quantile <- function(p, U, alpha, d = 2) {
  lambda <- -log(alpha) * U^(d / 2)
  (lambda / -log(p))^(2 / d)
}

# The lengthscale prior of an HSGP field over `coords`, as the default
# hyperprior binds it to the nested `lengthscale` axis (`.hp_range_anchor()`
# on the block's coordinate extent), and its median: the initial lengthscale
# guess of the paper's procedure (Sect. 4.5.1, step A1).
.hsgp_prior_lengthscale <- function(coords) {
  ra <- .hp_range_anchor(.hp_coord_fields(coords))
  if (is.null(ra)) return(NA_real_)
  .pc_range_quantile(0.5, ra$anchor[1L], ra$anchor[2L], ra$dim)
}

# The slack of the paper's lengthscale diagnostic (Sect. 4.5.1, steps A5 and
# B6): a basis is accepted when ell_hat + 0.01 >= the smallest lengthscale it
# represents. The paper states it for S = 1, i.e. on the normalized lengthscale
# ell / S of its Fig. 5 and Eqs. (19)-(24), which is where it is applied here.
.HSGP_DIAG_SLACK <- 0.01

# Where the lengthscale estimate `ell` sits against the window of a basis
# (m, c) on half-widths S, each end within the slack on the normalized
# lengthscale: `lower` is the paper's diagnostic (steps A5 / B6), `ell` not
# below the smallest lengthscale Eq. (19) allows; `upper` is `ell` not above
# the largest Eq. (20) allows.
.hsgp_basis_holds <- function(m, c, S, ell) {
  win <- .hsgp_lengthscale_window(m, c, S)
  c(lower = ell + .HSGP_DIAG_SLACK * max(S) >= win[["lo"]],
    upper = ell - .HSGP_DIAG_SLACK * min(S) <= win[["hi"]])
}

# Set the automatic entries of an HSGP spec's basis (`basis_rule$auto`) from a
# lengthscale estimate `ell` (kernel units), the paper's steps A2-A3: `c` from
# Eq. (20) at `ell`, then `m` from Eq. (19) at `ell` and that `c`. `m` never
# shrinks below a size already fitted. Records the step in
# `basis_rule$history`.
.hsgp_resolve_basis <- function(spec, ell, source) {
  br <- spec$basis_rule
  S <- br$half_width
  if (br$auto[["c"]]) spec[["c"]] <- .hsgp_basis_size(S, ell)$c
  sz <- .hsgp_basis_size(S, ell, c = spec[["c"]])
  if (br$auto[["m"]]) spec[["m"]] <- max(sz$m, spec[["m"]] %||% sz$m)
  br$history <- rbind(br$history, data.frame(
    source = source, lengthscale = ell, m = spec[["m"]], c = spec[["c"]],
    stringsAsFactors = FALSE))
  br$capped <- br$auto[["m"]] && sz$m_req > .HSGP_M_BOUNDS[2L]
  spec$basis_rule <- br
  spec
}

# The posterior mean and 95% interval of an HSGP field's lengthscale, in kernel
# units, off whichever posterior the fit carries: the outer grid of a nested
# fit or the `log_lengthscale_hsgp` draws of a sampler fit. NULL when the fit
# carries no posterior over the lengthscale (a fit conditioned on it).
.hsgp_lengthscale_posterior <- function(fit) {
  probs <- c(0.025, 0.975)
  if (!is.null(fit$theta_grid)) {
    s <- tryCatch(.nested_hyper_summary(fit, probs, keep_role = "spatial"),
                  error = function(e) NULL)
    if (is.null(s) || !"lengthscale" %in% rownames(s)) return(NULL)
    v <- as.numeric(s["lengthscale", c("mean", .quantile_colnames(probs))])
  } else {
    dr <- tryCatch(.fit_draws(fit), error = function(e) NULL)
    if (is.null(dr) || !"log_lengthscale_hsgp" %in% colnames(dr)) return(NULL)
    ell <- exp(dr[, "log_lengthscale_hsgp"])
    v <- c(mean(ell), stats::quantile(ell, probs))
  }
  if (!all(is.finite(v)) || any(v <= 0)) return(NULL)
  stats::setNames(v, c("mean", "lo", "hi"))
}

# One iteration of Riutort-Mayol et al. (2023, Sect. 4.5.1, Phase A) on a
# fitted HSGP field whose basis is left to the data: when the posterior mean
# lengthscale fails the paper's diagnostic, the basis it asks for (steps A2-A3,
# with the estimate as the new guess). Returns `list(spec, refit)`: the spec to
# fit next and TRUE, or the fitted spec and FALSE when the diagnostic holds or
# no basis within reach represents a smaller lengthscale.
#
# The refit follows the lower end of the window only, as the paper's diagnostic
# does, and is taken only when it lowers the basis's smallest lengthscale by
# more than the slack; that smallest lengthscale is bounded below (m <= 50,
# c >= 1.2), so the loop stops. With both entries automatic the condition
# always holds on a failing diagnostic: ell_hat + slack lies below the fitted
# basis's smallest lengthscale, and the next basis, sized by Eq. (19) at
# ell_hat, reaches down to ell_hat itself. With `m` supplied, or at its bound,
# it stops the loop once `c` can lower the window no further. The upper end has
# no such ordering: a lengthscale the data barely inform has a posterior mean
# that follows the boundary factor up, so chasing it with `c` need not stop,
# and it is reported by diagnostics() instead.
.hsgp_refit_spec <- function(spec, fit) {
  br <- spec$basis_rule
  keep <- list(spec = spec, refit = FALSE)
  if (is.null(br) || !any(br$auto) || isTRUE(br$capped)) return(keep)
  post <- .hsgp_lengthscale_posterior(fit)
  if (is.null(post)) return(keep)
  S <- br$half_width
  if (.hsgp_basis_holds(spec[["m"]], spec[["c"]], S,
                        post[["mean"]])[["lower"]]) return(keep)
  nxt <- .hsgp_resolve_basis(spec, post[["mean"]], "posterior")
  lo_now <- .hsgp_lengthscale_window(spec[["m"]], spec[["c"]], S)[["lo"]]
  lo_nxt <- .hsgp_lengthscale_window(nxt[["m"]], nxt[["c"]], S)[["lo"]]
  if (lo_nxt >= lo_now - .HSGP_DIAG_SLACK * max(S)) return(keep)
  list(spec = nxt, refit = TRUE)
}

# The basis check of an HSGP fit: the posterior mean lengthscale against both
# ends of the window the fitted basis represents. NULL for a fit without an
# HSGP field or without a lengthscale posterior.
.hsgp_basis_check <- function(fit) {
  sp <- fit$spatial
  if (!identical(tolower(sp$type %||% ""), "hsgp")) return(NULL)
  m <- sp[["m"]]; cc <- sp[["c"]]
  if (is.null(m) || is.null(cc) || is.null(sp$coords_matrix)) return(NULL)
  post <- .hsgp_lengthscale_posterior(fit)
  if (is.null(post)) return(NULL)
  S <- .hsgp_half_widths(sp$coords_matrix)
  k <- .coord_scale(sp)
  list(m = as.integer(m), c = cc,
       window = .hsgp_lengthscale_window(m, cc, S) * k,
       lengthscale = post * k,
       holds = .hsgp_basis_holds(m, cc, S, post[["mean"]]),
       auto = sp$basis_rule$auto %||% c(m = FALSE, c = FALSE),
       capped = isTRUE(sp$basis_rule$capped),
       history = sp$basis_rule$history)
}

.hsgp_basis_note <- function(chk) {
  if (is.null(chk) || all(chk$holds)) return(NULL)
  low <- !chk$holds[["lower"]]
  what <- if (low) {
    sprintf(paste0("represents lengthscales down to %.3g (Riutort-Mayol et ",
                   "al. 2023, Eq. 19), but the posterior mean lengthscale is ",
                   "%.3g"), chk$window[["lo"]], chk$lengthscale[["mean"]])
  } else {
    sprintf(paste0("represents lengthscales up to %.3g (Riutort-Mayol et al. ",
                   "2023, Eq. 20), but the posterior mean lengthscale is %.3g"),
            chk$window[["hi"]], chk$lengthscale[["mean"]])
  }
  fix <- if (low && chk$capped) {
    paste0("; the basis size it needs exceeds m = ", .HSGP_M_BOUNDS[2L])
  } else if (low && !chk$auto[["m"]]) {
    "; refit with a larger `m`, or leave it unset so the fit chooses it"
  } else if (!low) {
    "; a larger `c` (with the `m` Eq. 19 asks for) covers it"
  } else ""
  sprintf("HSGP basis (m = %d per dimension, c = %.2f) %s%s.", chk$m, chk$c,
          what, fix)
}
