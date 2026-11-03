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

# A basis size or boundary factor as an HSGP door accepts it: NULL leaves it to
# the data, anything else is checked and returned in its stored type.
.hsgp_check_m <- function(m, arg = "m") {
  if (is.null(m)) return(NULL)
  if (!is.numeric(m) || length(m) != 1L || is.na(m) ||
      m < .HSGP_M_BOUNDS[1L] || m > .HSGP_M_BOUNDS[2L]) {
    stop(sprintf(paste0("`%s` must be an integer between %d and %d, or NULL ",
                        "to choose it from the data"), arg,
                 .HSGP_M_BOUNDS[1L], .HSGP_M_BOUNDS[2L]), call. = FALSE)
  }
  as.integer(m)
}

.hsgp_check_c <- function(c, arg = "c") {
  if (is.null(c)) return(NULL)
  if (!is.numeric(c) || length(c) != 1L || is.na(c) || c < 1) {
    stop(sprintf(paste0("`%s` (boundary factor) must be >= 1, or NULL to ",
                        "choose it from the data"), arg), call. = FALSE)
  }
  as.numeric(c)
}

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

# The lengthscale prior of an HSGP field over `coords`, read by every route
# that fits one -- the nested blocks (`.hsgp_nl_block()`, the joint door's
# coordinate block) and the samplers (`.hsgp_sampler_spec()`,
# `.svc_sampler_spec()`) -- so a route is a computational choice and never a
# different model. It is the PC prior on a range (Fuglstad et al. 2019),
# P(ell < U) = alpha with U the default fraction of the coordinates' extent,
# in the two dimensions the basis spans. The PC range prior holds for any
# Matern smoothness, so for the squared-exponential limit the basis
# approximates. NULL when the coordinates span nothing.
.HSGP_LENGTHSCALE_PRIOR_DIM <- 2

.hsgp_lengthscale_prior <- function(coords) {
  .hp_range_anchor(.hsgp_coord_fields(coords))$anchor
}

# The coordinate extent an HSGP block carries for its lengthscale prior
# (`.hp_range_anchor()`), in the two dimensions the basis spans.
.hsgp_coord_fields <- function(coords) {
  f <- .hp_coord_fields(coords)
  if (length(f)) f$coord_dim <- .HSGP_LENGTHSCALE_PRIOR_DIM
  f
}

# The median of that prior, or of the PC range prior at the anchor `c(U,
# alpha)`: the initial lengthscale guess of the paper's procedure
# (Sect. 4.5.1, step A1).
.hsgp_prior_lengthscale <- function(coords) {
  .hsgp_prior_median(.hsgp_lengthscale_prior(coords))
}

.hsgp_prior_median <- function(anchor) {
  if (is.null(anchor)) return(NA_real_)
  .pc_range_quantile(0.5, anchor[1L], anchor[2L], .HSGP_LENGTHSCALE_PRIOR_DIM)
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

# The basis rule an HSGP spec carries (`basis_rule`): which of `m` and the
# boundary factor are left to the data (`auto`), the name the spec stores the
# boundary factor under (`c_key`: `c` on spatial_gp() and the joint door's
# coordinate block, `c_boundary` on spatial_svc() and spatial_multiscale()),
# the half-widths S of the coordinates, and the bases fitted so far
# (`history`). Every HSGP door sizes its basis through it, so one rule chooses
# `m` and `c` wherever an HSGP field is fitted.
#
# `.hsgp_basis_init()` attaches the rule and, for an entry left to the data,
# sizes the first basis from the lengthscale prior medians `ell_prior` of the
# fields the basis carries (one per field: two scales of a multi-scale field,
# one per varying coefficient on a shared basis); the paper's step A1. A spec
# whose `m` and boundary factor are both given keeps them.
.hsgp_basis_init <- function(spec, coords, ell_prior, c_key = "c",
                             door = "the HSGP field") {
  br <- spec$basis_rule %||% list()
  br$auto <- br$auto %||%
    c(m = is.null(spec[["m"]]), c = is.null(spec[[c_key]]))
  br$c_key <- c_key
  br$half_width <- .hsgp_half_widths(coords)
  spec$basis_rule <- br
  if (!any(br$auto)) return(spec)
  if (!length(ell_prior) || !all(is.finite(ell_prior)) ||
      any(br$half_width <= 0)) {
    stop(door, ": the coordinates span no area, so the basis size cannot be ",
         "chosen from them; supply `m` and the boundary factor.", call. = FALSE)
  }
  .hsgp_resolve_basis(spec, ell_prior, "prior")
}

# Whether a spec carries a basis left (partly) to the data.
.hsgp_basis_auto <- function(spec) {
  is.list(spec) && any(spec$basis_rule$auto %||% FALSE)
}

# The boundary factor of a spec, under the name its rule stores it.
.hsgp_spec_c <- function(spec) {
  spec[[spec$basis_rule$c_key %||% "c"]]
}

# Set the automatic entries of an HSGP spec's basis from lengthscale estimates
# `ell` (kernel units, one per field on the basis), the paper's steps A2-A3:
# `c` from Eq. (20) at the largest, then `m` from Eq. (19) at the smallest and
# that `c`. `m` never shrinks below a size already fitted. Records the step in
# `basis_rule$history`.
.hsgp_resolve_basis <- function(spec, ell, source) {
  br <- spec$basis_rule
  S <- br$half_width
  ck <- br$c_key %||% "c"
  ell_lo <- min(ell); ell_hi <- max(ell)
  if (br$auto[["c"]]) spec[[ck]] <- .hsgp_basis_size(S, ell_lo, ell_hi)$c
  sz <- .hsgp_basis_size(S, ell_lo, ell_hi, c = spec[[ck]])
  if (br$auto[["m"]]) spec[["m"]] <- max(sz$m, spec[["m"]] %||% sz$m)
  br$history <- rbind(br$history, data.frame(
    source = source, lengthscale = ell_lo, m = spec[["m"]], c = spec[[ck]],
    stringsAsFactors = FALSE))
  br$capped <- br$auto[["m"]] && sz$m_req > .HSGP_M_BOUNDS[2L]
  spec$basis_rule <- br
  spec
}

# The posterior mean and 95% interval of every HSGP lengthscale a fit carries,
# in kernel units, one row per field: off the outer grid of a nested fit (each
# `lengthscale` axis, or the columns named in `axes`), or off the draws of a
# sampler fit (`log_lengthscale_hsgp`, or one `log_phi_svc[j]` per term of an
# HSGP varying-coefficient field). NULL when the fit carries no posterior over
# a lengthscale (a fit conditioned on it).
.hsgp_lengthscale_posterior <- function(fit, axes = NULL) {
  probs <- c(0.025, 0.975)
  if (!is.null(fit$theta_grid)) {
    tg <- fit$theta_grid
    cn <- colnames(tg) %||% fit$theta_names
    j <- if (is.null(axes)) which(sub("^b[0-9]+\\.", "", cn) == "lengthscale")
         else match(axes, cn)
    if (!length(j) || anyNA(j)) return(NULL)
    s <- tryCatch(.nested_hyper_summary(fit, probs), error = function(e) NULL)
    if (is.null(s) || nrow(s) != length(cn)) return(NULL)
    v <- as.matrix(s[j, c("mean", .quantile_colnames(probs)), drop = FALSE])
    rownames(v) <- cn[j]
  } else {
    dr <- tryCatch(.fit_draws(fit), error = function(e) NULL)
    if (is.null(dr)) return(NULL)
    cols <- grep("^log_lengthscale_hsgp$|^log_phi_svc\\[[0-9]+\\]$",
                 colnames(dr), value = TRUE)
    if (!length(cols)) return(NULL)
    v <- t(vapply(cols, function(k) {
      ell <- exp(dr[, k])
      c(mean(ell), stats::quantile(ell, probs, names = FALSE))
    }, numeric(3)))
  }
  if (!all(is.finite(v)) || any(v <= 0)) return(NULL)
  colnames(v) <- c("mean", "lo", "hi")
  v
}

# One iteration of Riutort-Mayol et al. (2023, Sect. 4.5.1, Phase A) on a
# fitted HSGP basis left to the data: when the smallest posterior mean
# lengthscale on the basis fails the paper's diagnostic, the basis it asks for
# (steps A2-A3, with the estimates as the new guess). Returns
# `list(spec, refit)`: the spec to fit next and TRUE, or the fitted spec and
# FALSE when the diagnostic holds or no basis within reach represents a
# smaller lengthscale. The basis rule's `axes` names the fit's lengthscale
# columns that sit on this basis (all of them when NULL).
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
.hsgp_refit_spec <- function(spec, fit, axes = spec$basis_rule$axes) {
  br <- spec$basis_rule
  keep <- list(spec = spec, refit = FALSE)
  if (!.hsgp_basis_auto(spec) || isTRUE(br$capped)) return(keep)
  post <- .hsgp_lengthscale_posterior(fit, axes)
  if (is.null(post)) return(keep)
  S <- br$half_width
  cc <- .hsgp_spec_c(spec)
  if (.hsgp_basis_holds(spec[["m"]], cc, S,
                        min(post[, "mean"]))[["lower"]]) return(keep)
  nxt <- .hsgp_resolve_basis(spec, post[, "mean"], "posterior")
  lo_now <- .hsgp_lengthscale_window(spec[["m"]], cc, S)[["lo"]]
  lo_nxt <- .hsgp_lengthscale_window(nxt[["m"]], .hsgp_spec_c(nxt), S)[["lo"]]
  if (lo_nxt >= lo_now - .HSGP_DIAG_SLACK * max(S)) return(keep)
  list(spec = nxt, refit = TRUE)
}

# The bases a spec carries, as a list of basis specs: one per scale of a
# multi-scale field (`scale_basis`, each scale on its own basis), the spec
# itself otherwise. `.hsgp_basis_set_parts()` writes them back.
.hsgp_basis_parts <- function(spec) {
  spec$scale_basis %||% list(spec)
}

.hsgp_basis_set_parts <- function(spec, parts) {
  if (!is.null(spec$scale_basis)) {
    spec$scale_basis[] <- parts
    return(spec)
  }
  parts[[1L]]
}

# The basis (m, c) an HSGP block is built on: its scale's own basis on a
# multi-scale field (`block_name` the scale), the spec's otherwise.
.hsgp_block_basis <- function(spec, block_name = NULL) {
  b <- if (!is.null(spec$scale_basis)) spec$scale_basis[[block_name]] else spec
  list(m = as.integer(b[["m"]]), c = as.numeric(.hsgp_spec_c(b)))
}

# Fit an HSGP model through the paper's Phase A: `fit_once(spec)` fits at the
# spec's bases, and every basis left to the data is refitted on the basis its
# lengthscale posterior asks for (`.hsgp_refit_spec()`) until each holds its
# own. A spec whose bases are given is fitted once. `parts` / `set_parts` read
# and write the bases a spec carries (`.hsgp_basis_parts()`); the joint door,
# whose spec is the list of its coordinate blocks' bases, passes identities.
# The warnings of a fit that is replaced describe a fit the caller does not
# receive, so only the returned fit's are raised. Returns `list(fit, spec)`,
# the spec the fit was made at.
.hsgp_fit_basis_loop <- function(spec, fit_once,
                                 parts = .hsgp_basis_parts,
                                 set_parts = .hsgp_basis_set_parts) {
  auto <- is.list(spec) &&
    any(vapply(parts(spec), .hsgp_basis_auto, logical(1)))
  repeat {
    fit_warnings <- list()
    fit <- withCallingHandlers(
      fit_once(spec),
      warning = function(w) {
        if (auto) {
          fit_warnings[[length(fit_warnings) + 1L]] <<- w
          invokeRestart("muffleWarning")
        }
      })
    if (!auto || !is.list(fit)) break
    steps <- lapply(parts(spec), .hsgp_refit_spec, fit = fit)
    spec <- set_parts(spec, lapply(steps, `[[`, "spec"))
    if (!any(vapply(steps, `[[`, logical(1), "refit"))) break
  }
  for (w in fit_warnings) warning(w)
  list(fit = fit, spec = spec)
}

# The basis check of an HSGP fit: the smallest and the largest posterior mean
# lengthscale on a basis against the two ends of the window it represents, in
# the units of the coordinates as supplied. `spec` carries the bases: the one a
# door that fits outside the formula front door records as `hsgp_basis`, else
# the fit's spatial spec. Of several bases (a multi-scale field's) the first
# whose window misses its lengthscales is reported, else the first. NULL for a
# fit without an HSGP basis rule or without a lengthscale posterior.
.hsgp_basis_check <- function(fit, spec = fit$hsgp_basis %||% fit$spatial) {
  if (!is.list(spec)) return(NULL)
  k <- .coord_scale(spec)
  chks <- Filter(Negate(is.null), lapply(.hsgp_basis_parts(spec), function(b) {
    br <- b$basis_rule
    if (is.null(br) || is.null(br$half_width)) return(NULL)
    m <- b[["m"]]; cc <- .hsgp_spec_c(b)
    if (is.null(m) || is.null(cc)) return(NULL)
    post <- .hsgp_lengthscale_posterior(fit, br$axes)
    if (is.null(post)) return(NULL)
    S <- br$half_width
    lo <- post[which.min(post[, "mean"]), ]
    hi <- post[which.max(post[, "mean"]), ]
    list(m = as.integer(m), c = cc,
         window = .hsgp_lengthscale_window(m, cc, S) * k,
         lengthscale = lo * k, lengthscale_hi = hi * k,
         holds = c(lower = .hsgp_basis_holds(m, cc, S, lo[["mean"]])[["lower"]],
                   upper = .hsgp_basis_holds(m, cc, S, hi[["mean"]])[["upper"]]),
         auto = br$auto, capped = isTRUE(br$capped), history = br$history)
  }))
  if (!length(chks)) return(NULL)
  bad <- Filter(function(ch) !all(ch$holds), chks)
  if (length(bad)) bad[[1L]] else chks[[1L]]
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
            chk$window[["hi"]], chk$lengthscale_hi[["mean"]])
  }
  fix <- if (low && chk$capped) {
    paste0("; the basis size it needs exceeds m = ", .HSGP_M_BOUNDS[2L])
  } else if (low && !chk$auto[["m"]]) {
    "; refit with a larger `m`, or leave it unset so the fit chooses it"
  } else if (!low) {
    "; a larger boundary factor (with the `m` Eq. 19 asks for) covers it"
  } else ""
  sprintf("HSGP basis (m = %d per dimension, c = %.2f) %s%s.", chk$m, chk$c,
          what, fix)
}
