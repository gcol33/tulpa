# The kernels an HSGP basis approximates, and the basis size and boundary
# factor each needs for the lengthscales it has to represent.
#
# The spectral density of each kernel is evaluated in one place,
# `hsgp_spectral_density_2d()` (src/hsgp_spectral.h), which reads the kernel by
# its `code`. `corr` is the kernel's correlation at distance r = x * ell.
#
# Riutort-Mayol et al. (2023, Statistics and Computing 33:17, Sect. 4.3.1) fit
# the relation between the number of basis functions per dimension `m`, the
# boundary factor `c` (the basis boundary is L = c S, S the half-width of the
# input domain) and the lengthscale `ell` at which the approximate covariance
# stays within 1% of the exact one in total variation (their Eq. 18):
#
#   m = m_coef c / (ell / S),   c >= c_coef ell / S,   c >= 1.2
#
# with (m_coef, c_coef) = (1.75, 3.2) for the squared-exponential kernel
# (Eqs. 19-20), (2.65, 4.1) for Matern 5/2 (Eqs. 21-22) and (3.42, 4.5) for
# Matern 3/2 (Eqs. 23-24). Read the other way, a basis (m, c) is accurate for
# lengthscales in [m_coef c S / m, c S / c_coef], the smallest lengthscale the
# first relation allows and the largest the second allows. In two dimensions,
# the tensor expansion also retains 99% of the joint spectral variance;
# `.hsgp_m_coefficient()` integrates the spectrum over its frequency rectangle.
# One m and c serve both coordinates of an isotropic lengthscale, so m uses
# the widest dimension and c the narrowest.
.HSGP_KERNELS <- list(
  squared_exponential = list(
    code = 0L, nu = Inf, m_coef = 1.75, c_coef = 3.2,
    label = "squared exponential",
    corr = function(x) exp(-x^2 / 2)),
  matern32 = list(
    code = 1L, nu = 1.5, m_coef = 3.42, c_coef = 4.5,
    label = "Matern 3/2",
    corr = function(x) (1 + sqrt(3) * x) * exp(-sqrt(3) * x)),
  matern52 = list(
    code = 2L, nu = 2.5, m_coef = 2.65, c_coef = 4.1,
    label = "Matern 5/2",
    corr = function(x) {
      a <- sqrt(5) * x
      (1 + a + a^2 / 3) * exp(-a)
    })
)
.HSGP_C_MIN <- 1.2

# The kernel an HSGP field approximates when its door is given none.
.HSGP_DEFAULT_KERNEL <- "matern32"

# The kernel an HSGP door's `cov` / `nu` name: `"squared_exponential"`, or
# `"matern"` with `nu` 1.5 or 2.5. NULL `cov` is the default kernel. The
# exponential kernel (Matern 1/2) is refused: the paper gives no basis rule for
# it, and its spectral density decays too slowly for a finite basis.
.hsgp_kernel <- function(cov = NULL, nu = 1.5, door = "the HSGP field") {
  if (is.null(cov)) return(.HSGP_DEFAULT_KERNEL)
  if (!is.character(cov) || length(cov) != 1L || is.na(cov)) {
    stop(door, ": `cov` must be one string.", call. = FALSE)
  }
  if (identical(cov, "squared_exponential")) return("squared_exponential")
  if (identical(cov, "matern")) {
    if (!is.numeric(nu) || length(nu) != 1L || is.na(nu)) {
      stop(door, ": `nu` must be 1.5 or 2.5.", call. = FALSE)
    }
    if (isTRUE(all.equal(nu, 1.5))) return("matern32")
    if (isTRUE(all.equal(nu, 2.5))) return("matern52")
    stop(door, ": an HSGP basis approximates a Matern kernel with nu = 1.5 ",
         "or 2.5; got nu = ", format(nu), ".", call. = FALSE)
  }
  stop(door, ": an HSGP basis approximates cov = \"matern\" (nu = 1.5 or ",
       "2.5) or cov = \"squared_exponential\"; got cov = \"", cov, "\". ",
       "The exponential kernel has no HSGP basis rule (Riutort-Mayol et al. ",
       "2023); use approx = \"nngp\" for it.", call. = FALSE)
}

# The table row of a kernel name.
.hsgp_kernel_info <- function(kernel = .HSGP_DEFAULT_KERNEL) {
  k <- .HSGP_KERNELS[[kernel %||% .HSGP_DEFAULT_KERNEL]]
  if (is.null(k)) {
    stop("Internal: unknown HSGP kernel '", kernel, "'.", call. = FALSE)
  }
  k
}

# The covariance a spatial spec states, for printing: the kernel an HSGP basis
# approximates, else the NNGP `cov` with its smoothness.
.spatial_cov_label <- function(x) {
  if (!is.null(x$kernel)) return(.hsgp_kernel_info(x$kernel)$label)
  if (identical(x$cov, "matern") && !is.null(x$nu)) {
    return(sprintf("matern (nu = %.1f)", x$nu))
  }
  x$cov %||% "exponential"
}

# The kernel of a nested-Laplace HSGP block, given by its basis or by its
# coordinates: the `kernel` a door already resolved, else the block's `cov` /
# `nu` read as spatial_gp() reads them.
.hsgp_block_kernel <- function(p) {
  p$kernel %||% .hsgp_kernel(p$cov, p$nu %||% 1.5,
                             paste0("an HSGP block (type '", p$type %||% "hsgp",
                                    "')"))
}

# The integer code the C++ side reads (`HSGPData::kernel`, a block spec's
# `kernel` element).
.hsgp_kernel_code <- function(kernel = .HSGP_DEFAULT_KERNEL) {
  .hsgp_kernel_info(kernel)$code
}

# The distance at which a kernel's correlation falls to `rho`, per unit
# lengthscale.
.hsgp_kernel_range_factor <- function(kernel = .HSGP_DEFAULT_KERNEL,
                                      rho = 0.05) {
  corr <- .hsgp_kernel_info(kernel)$corr
  stats::uniroot(function(x) corr(x) - rho, c(1e-6, 50),
                 tol = 1e-12)$root
}

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
  xy <- as.matrix(coords)
  domain <- cpp_hsgp_domain_2d(xy, 1, expanded = FALSE)
  stats::setNames(as.numeric(domain[3:4]), colnames(xy))
}

# The lengthscale window a basis (m, c) on half-widths S represents accurately
# for `kernel`.
.hsgp_lengthscale_window <- function(m, c, S, kernel = .HSGP_DEFAULT_KERNEL) {
  if (all(S == 0)) return(c(lo = 0, hi = 0))
  r <- .hsgp_kernel_info(kernel)
  c(lo = .hsgp_m_coefficient(S, kernel) * c * max(S) / m,
    hi = .hsgp_boundary_lengthscale(c, S, c * min(S) / r$c_coef, kernel))
}

# The smallest (m, c) whose window holds [ell_lo, ell_hi] for `kernel`: `c`
# from the boundary relation at the largest lengthscale, unless it is given,
# then `m` from the basis-size relation at the smallest one with that `c`.
# `m_req` is that relation before the constructors' bounds; past the upper
# bound the window does not reach ell_lo.
.hsgp_basis_size <- function(S, ell_lo, ell_hi = ell_lo, c = NULL,
                             kernel = .HSGP_DEFAULT_KERNEL) {
  r <- .hsgp_kernel_info(kernel)
  if (is.null(c)) {
    c <- .hsgp_boundary_factor(max(.HSGP_C_MIN, r$c_coef * ell_hi / min(S)),
                               S, ell_hi, kernel)
  }
  m_req <- ceiling(.hsgp_m_coefficient(S, kernel) * c * max(S) / ell_lo)
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
# P(2 ell < U) = alpha with U the default fraction of the coordinates' extent,
# in the two dimensions the basis spans. The PC range prior holds for any
# Matern smoothness, so for each kernel an HSGP basis approximates (Matern 3/2
# and 5/2, and the squared-exponential limit). NULL when the coordinates span
# nothing.
.HSGP_LENGTHSCALE_PRIOR_DIM <- 2


.hsgp_range_prior <- function(coords) {
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
  .hsgp_prior_median(.hsgp_range_prior(coords))
}

.hsgp_prior_median <- function(anchor) {
  if (is.null(anchor)) return(NA_real_)
  .pc_range_quantile(0.5, anchor[1L], anchor[2L], .HSGP_LENGTHSCALE_PRIOR_DIM) /
    .SPATIAL_RANGE_FACTOR
}

# The slack of the paper's lengthscale diagnostic (Sect. 4.5.1, steps A5 and
# B6): a basis is accepted when ell_hat + 0.01 >= the smallest lengthscale it
# represents. The paper states it for S = 1, i.e. on the normalized lengthscale
# ell / S of its Fig. 5 and Eqs. (19)-(24), which is where it is applied here.
.HSGP_DIAG_SLACK <- 0.01

# Where the lengthscale estimate `ell` sits against the window of a basis
# (m, c) on half-widths S, each end within the slack on the normalized
# lengthscale: `lower` is the paper's diagnostic (steps A5 / B6), `ell` not
# below the smallest lengthscale the kernel's basis-size relation allows;
# `upper` is `ell` not above the largest its boundary relation allows.
.hsgp_basis_holds <- function(m, c, S, ell, kernel = .HSGP_DEFAULT_KERNEL) {
  win <- .hsgp_lengthscale_window(m, c, S, kernel)
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
  br$kernel <- spec$kernel %||% br$kernel %||% .HSGP_DEFAULT_KERNEL
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

# The kernel a basis spec approximates.
.hsgp_spec_kernel <- function(spec) {
  spec$kernel %||% spec$basis_rule$kernel %||% .HSGP_DEFAULT_KERNEL
}

# The boundary factor of a spec, under the name its rule stores it.
.hsgp_spec_c <- function(spec) {
  spec[[spec$basis_rule$c_key %||% "c"]]
}

# Set the automatic entries of an HSGP spec's basis from lengthscale estimates
# `ell` (kernel units, one per field on the basis), the paper's steps A2-A3:
# `c` from the kernel's boundary relation at the largest, then `m` from its
# basis-size relation at the smallest and that `c`. `m` never shrinks below a
# size already fitted, and is at least `m_min` (Phase B, step B2). Records the
# step in `basis_rule$history`.
.hsgp_resolve_basis <- function(spec, ell, source, m_min = NULL) {
  br <- spec$basis_rule
  S <- br$half_width
  ck <- br$c_key %||% "c"
  kern <- .hsgp_spec_kernel(spec)
  ell_lo <- min(ell); ell_hi <- max(ell)
  if (br$auto[["c"]]) {
    spec[[ck]] <- .hsgp_basis_size(S, ell_lo, ell_hi, kernel = kern)$c
  }
  sz <- .hsgp_basis_size(S, ell_lo, ell_hi, c = spec[[ck]], kernel = kern)
  if (br$auto[["m"]]) {
    m_new <- max(sz$m, spec[["m"]] %||% sz$m, m_min %||% 0L)
    spec[["m"]] <- as.integer(min(m_new, .HSGP_M_BOUNDS[2L]))
  }
  br$history <- rbind(br$history, data.frame(
    source = source, lengthscale = ell_lo, m = spec[["m"]], c = spec[[ck]],
    stringsAsFactors = FALSE))
  br$capped <- br$auto[["m"]] &&
    max(sz$m_req, m_min %||% 0L) > .HSGP_M_BOUNDS[2L]
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
# basis's smallest lengthscale, and the next basis, sized by the basis-size
# relation at ell_hat, reaches down to ell_hat itself. With `m` supplied, or at its bound,
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
  kern <- .hsgp_spec_kernel(spec)
  if (.hsgp_basis_holds(spec[["m"]], cc, S, min(post[, "mean"]),
                        kern)[["lower"]]) return(keep)
  nxt <- .hsgp_resolve_basis(spec, post[, "mean"], "posterior")
  lo_now <- .hsgp_lengthscale_window(spec[["m"]], cc, S, kern)[["lo"]]
  lo_nxt <- .hsgp_lengthscale_window(nxt[["m"]], .hsgp_spec_c(nxt), S,
                                     kern)[["lo"]]
  if (lo_nxt >= lo_now - .HSGP_DIAG_SLACK * max(S)) return(keep)
  list(spec = nxt, refit = TRUE)
}

# Riutort-Mayol et al. (2023, Sect. 4.5.1, Phase B), entered once every basis
# left to the data holds its lengthscale (Phase A). Each step adds five basis
# functions per dimension to every basis whose `m` is left to the data (step
# B2), never fewer than the basis-size relation asks for at the latest
# estimate, and fits again (B5). The boundary factor stays where Phase A left
# it: step B3 would move it with the estimate, and an estimate the data barely
# inform has a posterior mean that follows the boundary factor up (see
# `.hsgp_refit_spec()`), so the steps would chase it; Phase A has already set
# the factor its diagnostic needs. The procedure ends when the refitted model passes the lengthscale
# diagnostic and is stable against the previous fit (B6); otherwise it adds
# five more. `m` is bounded (`.HSGP_M_BOUNDS`), so the steps stop.
#
# The paper names what has to be stable, the lengthscale estimate and the
# predictive accuracy, and gives no threshold for it. The one used here takes
# its scale from the posterior: a quantity is stable when the refitted
# posterior mean lies inside the previous fit's 95% posterior interval for it.
# That is read for every lengthscale on every basis
# (`.hsgp_lengthscale_posterior()`) and for the linear predictor at every
# observation (`posterior_linpred(summary = TRUE)`), the in-sample predictive
# distribution, on whichever of the two a fit carries.

# The next Phase B basis of one spec: `list(spec, refit)`, `refit` FALSE when
# its `m` is supplied, already at its bound, or the fit carries no lengthscale
# posterior.
.hsgp_phase_b_spec <- function(spec, fit, axes = spec$basis_rule$axes) {
  br <- spec$basis_rule
  keep <- list(spec = spec, refit = FALSE)
  if (!.hsgp_basis_auto(spec) || !isTRUE(br$auto[["m"]])) return(keep)
  m_now <- as.integer(spec[["m"]])
  if (m_now >= .HSGP_M_BOUNDS[2L]) return(keep)
  post <- .hsgp_lengthscale_posterior(fit, axes)
  if (is.null(post)) return(keep)
  held <- spec
  held$basis_rule$auto[["c"]] <- FALSE
  nxt <- .hsgp_resolve_basis(held, post[, "mean"], "phase B",
                             m_min = m_now + 5L)
  nxt$basis_rule$auto <- br$auto
  list(spec = nxt, refit = nxt[["m"]] > m_now)
}

# What Phase B compares between two fits: the lengthscale posterior of each
# basis (one matrix per part, NULL where the fit carries none) and the summary
# of the linear predictor's posterior at the training rows (NULL when the fit
# reports none).
.hsgp_fit_state <- function(fit, parts) {
  lp <- tryCatch(posterior_linpred(fit, summary = TRUE),
                 error = function(e) NULL)
  if (!is.data.frame(lp) || !all(is.finite(c(lp$conf.low, lp$conf.high)))) {
    lp <- NULL
  }
  list(ell = lapply(parts, function(b)
         .hsgp_lengthscale_posterior(fit, b$basis_rule$axes)),
       linpred = lp)
}

# Whether `cur` is stable against `prev` (step B6): every posterior mean of
# `cur` inside the matching 95% interval of `prev`.
.hsgp_fit_stable <- function(prev, cur) {
  inside <- function(x, lo, hi) {
    length(x) == length(lo) && all(x >= lo & x <= hi)
  }
  ell_ok <- all(vapply(seq_along(cur$ell), function(b) {
    a <- prev$ell[[b]]
    z <- cur$ell[[b]]
    if (is.null(a) || is.null(z)) return(TRUE)
    inside(z[, "mean"], a[, "lo"], a[, "hi"])
  }, logical(1)))
  lp_ok <- is.null(prev$linpred) || is.null(cur$linpred) ||
    inside(cur$linpred$estimate, prev$linpred$conf.low,
           prev$linpred$conf.high)
  ell_ok && lp_ok
}

# Whether every basis of a fit passes the lengthscale diagnostic (steps A5 /
# B6).
.hsgp_parts_hold <- function(parts, fit) {
  all(vapply(parts, function(b) {
    if (!.hsgp_basis_auto(b)) return(TRUE)
    post <- .hsgp_lengthscale_posterior(fit, b$basis_rule$axes)
    if (is.null(post)) return(TRUE)
    .hsgp_basis_holds(b[["m"]], .hsgp_spec_c(b), b$basis_rule$half_width,
                      min(post[, "mean"]), .hsgp_spec_kernel(b))[["lower"]]
  }, logical(1)))
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

# The basis (m, c) an HSGP block is built on, and the kernel it approximates: its scale's own basis on a
# multi-scale field (`block_name` the scale), the spec's otherwise.
.hsgp_block_basis <- function(spec, block_name = NULL) {
  b <- if (!is.null(spec$scale_basis)) spec$scale_basis[[block_name]] else spec
  list(m = as.integer(b[["m"]]), c = as.numeric(.hsgp_spec_c(b)),
       kernel = .hsgp_spec_kernel(b))
}

# Fit an HSGP model through the paper's procedure (Sect. 4.5.1):
# `fit_once(spec)` fits at the spec's bases. In Phase A every basis left to the
# data is refitted on the basis its lengthscale posterior asks for
# (`.hsgp_refit_spec()`) until each holds its own; Phase B then adds basis
# functions until the fit is stable (`.hsgp_phase_b_spec()`,
# `.hsgp_fit_stable()`), and the last fit is returned. A spec whose bases are
# given is fitted once. `parts` / `set_parts` read
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
  phase_b <- FALSE
  prev <- NULL
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
    if (!phase_b) {
      steps <- lapply(parts(spec), .hsgp_refit_spec, fit = fit)
      if (any(vapply(steps, `[[`, logical(1), "refit"))) {
        spec <- set_parts(spec, lapply(steps, `[[`, "spec"))
        next
      }
      phase_b <- TRUE
    } else if (.hsgp_parts_hold(parts(spec), fit) &&
               .hsgp_fit_stable(prev, .hsgp_fit_state(fit, parts(spec)))) {
      break
    }
    steps <- lapply(parts(spec), .hsgp_phase_b_spec, fit = fit)
    if (!any(vapply(steps, `[[`, logical(1), "refit"))) break
    prev <- .hsgp_fit_state(fit, parts(spec))
    spec <- set_parts(spec, lapply(steps, `[[`, "spec"))
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
    kern <- .hsgp_spec_kernel(b)
    lo <- post[which.min(post[, "mean"]), ]
    hi <- post[which.max(post[, "mean"]), ]
    list(m = as.integer(m), c = cc, kernel = kern,
         window = .hsgp_lengthscale_window(m, cc, S, kern) * k,
         lengthscale = lo * k, lengthscale_hi = hi * k,
         holds = c(lower = .hsgp_basis_holds(m, cc, S, lo[["mean"]],
                                             kern)[["lower"]],
                   upper = .hsgp_basis_holds(m, cc, S, hi[["mean"]],
                                             kern)[["upper"]]),
         auto = br$auto, capped = isTRUE(br$capped), history = br$history)
  }))
  if (!length(chks)) return(NULL)
  bad <- Filter(function(ch) !all(ch$holds), chks)
  if (length(bad)) bad[[1L]] else chks[[1L]]
}

.hsgp_basis_note <- function(chk) {
  if (is.null(chk) || all(chk$holds)) return(NULL)
  low <- !chk$holds[["lower"]]
  kinfo <- .hsgp_kernel_info(chk$kernel)
  what <- if (low) {
    sprintf(paste0("represents lengthscales down to %.3g (2-D spectral ",
                   "variance criterion), posterior mean lengthscale ",
                   "%.3g"), chk$window[["lo"]],
            chk$lengthscale[["mean"]])
  } else {
    sprintf(paste0("represents lengthscales up to %.3g (2-D boundary ",
                   "variance criterion), posterior mean lengthscale %.3g"),
            chk$window[["hi"]], chk$lengthscale_hi[["mean"]])
  }
  fix <- if (low && chk$capped) {
    paste0("; the basis size it needs exceeds m = ", .HSGP_M_BOUNDS[2L])
  } else if (low && !chk$auto[["m"]]) {
    "; refit with a larger `m`, or leave it unset so the fit chooses it"
  } else if (!low) {
    sprintf("; a larger boundary factor (with the `m` Eq. %d asks for) covers it",
            eq[["m"]])
  } else ""
  sprintf("HSGP basis (%s kernel, m = %d per dimension, c = %.2f) %s%s.",
          kinfo$label, chk$m, chk$c, what, fix)
}
