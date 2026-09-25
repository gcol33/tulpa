# Default hyperpriors on the outer hyperparameter axes (gcol33/tulpa#730).
#
# Every outer axis carries a proper, normalised prior density when the caller
# states none: a PC prior on a scale (standard deviation, variance, precision), a
# PC prior on a Matern range anchored at the coordinates' extent, a uniform on a
# bounded axis's domain, and the caller's own density wherever a block or a front
# door declares one. The constants are `.NL_HYPERPRIOR` (`R/settings.R`); the
# densities are pc_prior.h's (`cpp_hyperprior_log_density()`).
#
# THE CONVENTION. A density is taken to the axis's INTEGRATION coordinate (log
# for a log-scale axis, the natural value otherwise), change-of-variables term
# included, and folded into `log_marginal`. `log_hyperprior` records the fold,
# `log_hyperprior_axes` names the axes it covers, and
# `log_hyperprior_declined` names every integrated axis carrying no normalised
# density, with the reason. A density the kernel already added to its own
# marginal (a tgmrf block's `prior(theta)`) is recorded in `log_hyperprior` and
# again in `log_hyperprior_in_kernel`, so `log_hyperprior -
# log_hyperprior_in_kernel` is what R added. That is the convention the AR1 rho, SPDE and joint
# priors already followed, so everything reading `log_marginal` as the outer
# posterior on the grid coordinate -- the weights, the Pareto-k target, the
# placement stencils, the axis quantiles -- reads the posterior without knowing
# the prior exists. A cell's measure (`log_quad`) stays its width on that
# coordinate, so the evidence is `lse(log_marginal + absolute log cell width)`
# and needs no renormalisation over the grid.

# Why an integrated axis carries no normalised density.
.HP_DECLINE_REASONS <- c(
  # A range axis on a block that carries neither a user range anchor nor the
  # coordinates the default anchor is read from.
  "range_extent_unknown",
  # A per-arm dispersion whose family has no prior sourced yet.
  "dispersion_prior_unsourced",
  # A free-covariance block's log-Cholesky columns form a tensor neither in
  # those coordinates nor, for two fields, in (log sigma_1, log sigma_2, rho),
  # so its cells have no measure the grid itself defines.
  "logchol_design_measure",
  # A free-covariance block with some of its log-Cholesky columns held fixed:
  # the joint density on Sigma is not a density on the axes that remain.
  "logchol_partial_block",
  # An axis no table binds a density to.
  "unclassified_axis",
  # The caller chose `hyperprior = "flat"` and stated no density on the axis.
  "flat_hyperprior"
)

.hp_decline <- function(reason) {
  stopifnot(reason %in% .HP_DECLINE_REASONS)
  reason
}

# Families whose dispersion `phi` is a residual VARIANCE, the one dispersion the
# SD PC prior describes by change of variables.
.HP_VARIANCE_DISPERSION <- c("gaussian", "lognormal")

# Families whose dispersion `phi` is the negative-binomial SIZE, which carries
# the PC prior on its overdispersion 1 / size (`.NL_HYPERPRIOR$nb_size_lambda`).
.HP_NB_SIZE_DISPERSION <- c("neg_binomial_2", "truncated_neg_binomial_2")

# Families whose dispersion carries an exponential prior on `phi` itself, by
# base family, with the `.NL_HYPERPRIOR` rate it reads: R-INLA's defaults for a
# gamma shape and a beta precision (gcol33/tulpa#736).
.HP_EXP_DISPERSION <- c(gamma = "gamma_shape_rate", beta = "beta_precision_rate")

# The columns of `tg` that are integrated: a column holding one value is a
# fixed setting, part of the model rather than an axis of the grid.
.hp_integrated_axes <- function(tg) {
  if (is.null(tg) || !ncol(tg)) return(character(0))
  keep <- vapply(seq_len(ncol(tg)), function(j) {
    v <- tg[, j]
    length(unique(v[is.finite(v)])) > 1L
  }, logical(1))
  colnames(tg)[keep]
}

# PC density of a positive scale-type axis on its log coordinate.
.hp_log_scale_density <- function(x, kind, anchor = .nl_scale_anchor(), d = 2) {
  suppressWarnings(
    cpp_hyperprior_log_density(log(as.numeric(x)), kind, "log",
                               anchor[1L], anchor[2L], d))
}

# The coordinate extent and dimension of a spatial block, or NULL. A block
# declares `coord_extent` / `coord_dim` itself where the coordinates do not
# travel with it (an HSGP basis, an SPDE mesh), otherwise they are read off
# `coords`.
.hp_block_extent <- function(p) {
  if (!is.null(p$coord_extent)) {
    ext <- as.numeric(p$coord_extent)[1L]
    dim <- as.numeric(p$coord_dim %||% 2)[1L]
  } else if (is.matrix(p$coords) && nrow(p$coords) >= 2L) {
    rng <- apply(p$coords, 2L, range)
    ext <- sqrt(sum((rng[2L, ] - rng[1L, ])^2))
    # The dimension the points span, not the column count: a coordinate column
    # that is constant, or a column a linear combination of the others, adds no
    # direction the field varies over.
    dim <- qr(sweep(p$coords, 2L, colMeans(p$coords)))$rank
  } else {
    return(NULL)
  }
  if (!is.finite(ext) || ext <= 0 || !is.finite(dim) || dim < 1) return(NULL)
  list(extent = ext, dim = dim)
}

# The coordinate extent and dimension of a coordinate matrix, as the two fields
# a block carries them in.
.hp_coord_fields <- function(coords) {
  ex <- .hp_block_extent(list(coords = as.matrix(coords)))
  if (is.null(ex)) return(list())
  list(coord_extent = ex$extent, coord_dim = ex$dim)
}

# The default range PC anchor `c(U, alpha)` for a field over `coords`:
# P(range < U) = alpha with U the `.NL_HYPERPRIOR` fraction of the coordinates'
# extent. NULL when the coordinates span nothing.
.nl_default_range_prior <- function(coords) {
  ex <- .hp_block_extent(list(coords = as.matrix(coords)))
  if (is.null(ex)) return(NULL)
  c(.nl_hyperprior("range_extent_fraction") * ex$extent,
    .nl_hyperprior("range_alpha"))
}

# The SPDE field's hyperprior on (range, sigma) at natural values, as a density
# on (log range, log sigma): the coordinates every SPDE outer integrator works
# in, grid, CCD and k-hat alike (gcol33/tulpa#731).
#' SPDE field hyperprior log density on (range, sigma)
#'
#' The SPDE field's hyperprior on `(range, sigma)`, evaluated as a density on
#' `(log range, log sigma)` -- the coordinates every SPDE outer integrator
#' works in (grid, CCD and k-hat alike, gcol33/tulpa#731). Intended for a
#' caller running its own outer-grid loop over `(range, sigma)` around a
#' coupled model tulpa's own SPDE front doors do not fit directly.
#'
#' @param range,sigma Numeric vectors of range / marginal-SD nodes (natural
#'   scale, recycled against each other).
#' @param sp A `spatial_spde()` / `spatial_spde_custom()` spec, carrying the
#'   declared or defaulted `prior_range` / `prior_sigma` anchors.
#' @param hyperprior `"proper"` (default) or `"flat"`; see `?spatial_spde`.
#' @return Numeric vector of log-density values, one per `(range, sigma)`
#'   pair.
#' @seealso [tulpa_spde_precision_Q()], [spatial_spde()]
#' @examples
#' n <- 6L
#' G <- Matrix::bandSparse(n, k = c(-1, 0, 1),
#'                         diagonals = list(rep(-1, n - 1), c(1, rep(2, n - 2), 1),
#'                                          rep(-1, n - 1)))
#' sp <- spatial_spde_custom(Matrix::Diagonal(n), G, Matrix::Diagonal(n),
#'                           prior_range = c(1, 0.5), prior_sigma = c(1, 0.01))
#' tulpa_spde_log_hyperprior(range = c(0.5, 1, 2), sigma = 1, sp = sp)
#' @export
tulpa_spde_log_hyperprior <- function(range, sigma, sp, hyperprior = "proper") {
  .spde_log_hyperprior(range, sigma, sp, hyperprior)
}

.spde_log_hyperprior <- function(range, sigma, sp, hyperprior = "proper") {
  .spde_hyperprior_record(range, sigma, sp, hyperprior)$lp
}

# The same density as a `.hp_collect()` record over (range, sigma). The spec
# carries an anchor for both axes whether the caller stated it or the
# constructor defaulted it (the grid is placed on it either way), so
# `sp$prior_stated` is what says which were stated: under `"flat"` only those
# keep their density.
.spde_hyperprior_record <- function(range, sigma, sp, hyperprior = "proper") {
  flat <- identical(.hp_choice(hyperprior), "flat")
  stated <- sp$prior_stated %||% c(range = TRUE, sigma = TRUE)
  block <- list(type = "spde",
                prior_range = if (!flat || stated[["range"]]) sp$prior_range,
                prior_sigma = if (!flat || stated[["sigma"]]) sp$prior_sigma)
  tg <- cbind(range = as.numeric(range), sigma = as.numeric(sigma))
  .hp_collect(tg, function(a) .hp_axis_prior(a, block, hyperprior = hyperprior),
              axes = colnames(tg))
}

# The range PC anchor `c(U, alpha)` and dimension for a block: the caller's
# `prior_range` where the block carries one (a 2-D SPDE convention), the
# extent-scaled default otherwise, NULL when neither is available.
.hp_range_anchor <- function(p) {
  if (!is.null(p$prior_range)) {
    return(list(anchor = as.numeric(p$prior_range)[1:2], dim = 2))
  }
  ex <- .hp_block_extent(p)
  if (is.null(ex)) return(NULL)
  list(anchor = c(.nl_hyperprior("range_extent_fraction") * ex$extent,
                  .nl_hyperprior("range_alpha")),
       dim = ex$dim)
}

# The default density of ONE axis. `bare` is the axis name without a block
# prefix, `block` the prior block that owns it (its `type` and the fields a
# density reads), `family` the arm family of a `phi_<arm>` dispersion axis.
#
# Returns one of
#   list(fn = function(x))   the log density on the integration coordinate at
#                            natural-scale values x, to fold into log_marginal;
#   list(spec = TRUE)        the axis spec already declares a normalised density
#                            (the copy scale's slab), nothing to fold;
#   list(reason = "<why>")   no normalised density.
.hp_axis_default <- function(bare, block = list(), family = NULL) {
  type <- tolower(block$type %||% "")
  scale <- function(kind, anchor = .nl_scale_anchor()) {
    force(kind); force(anchor)
    list(fn = function(x) .hp_log_scale_density(x, kind, anchor))
  }
  uniform <- function(bounds) {
    force(bounds)
    list(fn = function(x) rep(-log(diff(as.numeric(bounds))), length(x)))
  }
  range_pc <- function() {
    ra <- .hp_range_anchor(block)
    if (is.null(ra)) return(list(reason = .hp_decline("range_extent_unknown")))
    list(fn = function(x) .hp_log_scale_density(x, "range", ra$anchor, ra$dim))
  }

  if (type %in% c("mcar", "miid") && .hp_is_logchol_col(bare)) {
    return(list(group = "logchol"))
  }
  switch(paste(type, bare, sep = ":"),
    "spde:sigma" = return(scale(
      "sd", as.numeric(block$prior_sigma %||% .nl_scale_anchor())[1:2])),
    "bym2:rho"       = return(uniform(c(0, 1))),
    "car_proper:rho" = return(uniform(block$rho_bounds %||%
                                        .nl_grid_par("car_rho", "bounds"))),
    "hsgp_mo:rho"    = return(uniform(c(-1, 1))),
    "ar1:rho" = {
      # Beta(a, b) on u = (rho + 1) / 2, normalised and carried to rho.
      ab <- .ar1_rho_beta_ab(block$rho_prior)
      return(list(fn = function(x) {
        stats::dbeta(0.5 * (as.numeric(x) + 1), ab[1L], ab[2L], log = TRUE) +
          log(0.5)
      }))
    },
    NULL
  )
  if (bare %in% c("range", "phi_gp", "lengthscale", "ell")) return(range_pc())
  if (identical(bare, "alpha")) return(list(spec = TRUE))
  if (identical(bare, "rho_car")) {
    return(uniform(.nl_grid_par("car_rho", "bounds")))
  }
  if (startsWith(bare, "phi_")) {
    fam <- tolower(family %||% "")
    if (fam %in% .HP_VARIANCE_DISPERSION) return(scale("variance"))
    base <- if (nzchar(fam)) .family_base(fam) else ""
    if (base %in% names(.HP_EXP_DISPERSION)) {
      rate <- .nl_hyperprior(.HP_EXP_DISPERSION[[base]])
      return(list(fn = function(x) .hyper_prior_carry(
        x, function(v) stats::dexp(v, rate, log = TRUE),
        log_scale = TRUE, coord = "natural")))
    }
    if (fam %in% .HP_NB_SIZE_DISPERSION) {
      return(list(fn = function(x) suppressWarnings(cpp_hyperprior_log_density(
        log(as.numeric(x)), "nb_size", "log", 1, 0.5,
        lambda = .nl_hyperprior("nb_size_lambda")))))
    }
    return(list(reason = .hp_decline("dispersion_prior_unsourced")))
  }
  if (startsWith(bare, "tau")) return(scale("precision"))
  if (identical(bare, "sigma2")) return(scale("variance"))
  if (startsWith(bare, "sigma")) return(scale("sd"))
  list(reason = .hp_decline("unclassified_axis"))
}

# The prior a front door gives every outer axis its caller states no density
# on. `"proper"` is `.hp_axis_default()`'s per-axis set; `"flat"` adds no density
# of the engine's own, so such an axis is integrated under its cell measure
# alone and the evidence declines. A density the caller states is applied under
# either: a `prior_sigma` / `prior_alpha` / `prior_phi` argument, a block's
# `rho_prior`, `prior_range` or `prior_sigma`, the copy scale's declared slab,
# and a tgmrf block's own prior.
.HP_CHOICES <- c("proper", "flat")

.hp_choice <- function(hyperprior) {
  if (!is.character(hyperprior) || length(hyperprior) != 1L ||
      !hyperprior %in% .HP_CHOICES) {
    stop(sprintf("`hyperprior` must be one of %s.",
                 paste0('"', .HP_CHOICES, '"', collapse = ", ")), call. = FALSE)
  }
  hyperprior
}

# The density of ONE axis under a hyperprior choice, in `.hp_axis_default()`'s
# return shape. Under `"flat"` an axis keeps only a density its block states.
.hp_axis_prior <- function(bare, block = list(), family = NULL,
                           hyperprior = "proper") {
  if (identical(.hp_choice(hyperprior), "proper")) {
    return(.hp_axis_default(bare, block, family))
  }
  type <- tolower(block$type %||% "")
  stated <- identical(bare, "alpha") ||
    (identical(type, "ar1") && identical(bare, "rho") && !is.null(block$rho_prior)) ||
    (identical(type, "spde") && identical(bare, "sigma") && !is.null(block$prior_sigma)) ||
    (bare %in% c("range", "phi_gp", "lengthscale", "ell") && !is.null(block$prior_range))
  if (stated) return(.hp_axis_default(bare, block, family))
  list(reason = .hp_decline("flat_hyperprior"))
}

# Fold densities over the integrated columns of `tg` into a record: the summed
# per-cell density, the axes it covers, the declined axes with their reasons.
# `resolve(axis)` returns the `.hp_axis_default()` shape for a column; `name`
# maps a column to the name it is reported under; `axes` are the columns read.
# The caller states `axes` from the grid it declared, never from `tg`: a batch
# of cells (a placement stencil, a single probe row, a refinement slice) holds
# columns constant that the fit integrates (gcol33/tulpa#760). `logchol` is the
# same declaration for the free-covariance blocks among `axes`, each block's
# design keyed by its column prefix (`.hp_logchol_designs()`): a batch of a
# block's rows is a tensor in no coordinates at all.
.hp_collect <- function(tg, resolve, name = identity, axes, logchol = list()) {
  n <- nrow(tg)
  out <- list(lp = numeric(n), lp_in_kernel = numeric(n),
              axes = character(0), declined = character(0))
  groups <- list()
  for (a in axes) {
    r <- resolve(a)
    if (identical(r$group, "logchol")) {
      key <- .hp_logchol_key(.hp_col_prefix(a))
      groups[[key]] <- c(groups[[key]], a)
    } else if (!is.null(r$fn)) {
      v <- as.numeric(r$fn(as.numeric(tg[, a])))
      v[is.na(v)] <- -Inf
      out$lp   <- out$lp + v
      out$axes <- c(out$axes, name(a))
    } else if (!is.null(r$reason)) {
      out$declined[[name(a)]] <- r$reason
    }
  }
  # A free-covariance block's density is one density over all of its columns,
  # evaluated at each row in the coordinates its declared design names.
  for (key in names(groups)) {
    pre <- .hp_logchol_key_pre(key)
    design <- logchol[[key]]
    if (is.null(design)) {
      stop(sprintf(paste0("The free-covariance block '%s' has no declared ",
                          "design; resolve it off the declared grid with ",
                          "`.hp_declare()`."), pre), call. = FALSE)
    }
    cols <- .hp_logchol_block_cols(colnames(tg), pre)
    d <- if (is.null(cols)) list(reason = .hp_decline("logchol_partial_block"))
         else .hp_logchol_log_density(tg[, cols, drop = FALSE], design)
    if (!is.null(d$lp)) {
      v <- d$lp
      v[is.na(v)] <- -Inf
      out$lp   <- out$lp + v
      out$axes <- c(out$axes, vapply(groups[[key]], name, character(1)))
    } else {
      for (a in groups[[key]]) out$declined[[name(a)]] <- d$reason
    }
  }
  out
}

# Free-covariance blocks (`mcar`, `miid`) lay their outer axes as the
# p(p + 1) / 2 log-Cholesky coordinates of Sigma = L L', one column per entry
# of L in column-major lower-triangle order and named `L<i><j>`: `log L_ii` on
# the diagonal, the raw `L_ij` below it (`.re_logchol_to_L()`). Their prior and
# their cell measure are properties of the block, not of any one column.
.hp_is_logchol_col <- function(bare) grepl("^L[0-9][0-9]$", bare)

# The block prefix of a column (`b2.` of `b2.L21`, empty when unprefixed).
.hp_col_prefix <- function(col) {
  m <- regmatches(col, regexpr("^b[0-9]+[.]", col))
  if (length(m)) m else ""
}

# A block's log-Cholesky columns among `cols`, in the column-major order
# `.re_logchol_to_L()` reads, or NULL when the set is not a complete p x p
# triangle.
.hp_logchol_block_cols <- function(cols, prefix) {
  bare <- sub("^b[0-9]+[.]", "", cols)
  mine <- cols[.hp_col_prefix_vec(cols) == prefix & .hp_is_logchol_col(bare)]
  if (!length(mine)) return(NULL)
  ij <- do.call(rbind, lapply(sub("^b[0-9]+[.]L", "", sub("^L", "", mine)),
                              function(s) as.integer(strsplit(s, "")[[1L]])))
  p <- max(ij)
  want <- character(0)
  for (j in seq_len(p)) for (i in j:p) want <- c(want, sprintf("%sL%d%d", prefix, i, j))
  if (!setequal(want, mine)) return(NULL)
  want
}
.hp_col_prefix_vec <- function(cols) vapply(cols, .hp_col_prefix, character(1),
                                            USE.NAMES = FALSE)

# Is a node matrix a full tensor over its columns' distinct values? Values are
# compared at 12 significant digits, the resolution a coordinate recovered
# through `exp` / `sqrt` keeps.
.hp_is_tensor <- function(M) {
  M <- signif(as.matrix(M), 12L)
  n_lev <- vapply(seq_len(ncol(M)), function(j) length(unique(M[, j])),
                  integer(1))
  nrow(unique(M)) == nrow(M) && nrow(M) == prod(n_lev)
}

# Lists keyed by a free-covariance block's column prefix. An unprefixed
# single-block grid has the empty prefix, which is not a usable list name.
.hp_logchol_key <- function(pre) paste0("<", pre, ">")
.hp_logchol_key_pre <- function(key) sub("^<(.*)>$", "\\1", key)

# The coordinates a free-covariance block's grid is a tensor in, which are the
# coordinates it integrates on. `M` holds the block's columns in
# `.hp_logchol_block_cols()` order. The design is a property of the block's
# grid, so it is read off the block's DISTINCT rows: a joint grid repeats them
# once per row of every other block, and that is still the block's own tensor.
# Returns `list(design, p)` with `design` one of
#   "logchol"  a tensor in the log-Cholesky columns themselves;
#   "sd_rho"   two fields laid as a tensor in (log sigma_1, log sigma_2, rho),
#              the default grid (`.mcar_default_logchol_grid()`);
# or `list(reason)`. The two cannot both hold on a grid whose three columns are
# all integrated: rho levels fix the (L21, L22) pairs along s2, and s2 levels fix
# them along rho.
.hp_logchol_design <- function(M) {
  M <- as.matrix(M)
  m <- ncol(M)
  p <- (sqrt(8 * m + 1) - 1) / 2
  if (!m || p != round(p)) return(list(reason = .hp_decline("logchol_design_measure")))
  M <- M[!duplicated(signif(M, 12L)), , drop = FALSE]
  varies <- vapply(seq_len(m), function(j) length(unique(M[, j])) > 1L,
                   logical(1))
  if (!all(varies)) return(list(reason = .hp_decline("logchol_partial_block")))
  if (.hp_is_tensor(M)) return(list(design = "logchol", p = as.integer(p)))
  if (p == 2) {
    sd_rho <- list(design = "sd_rho", p = 2L)
    if (.hp_is_tensor(.hp_logchol_coords(M, sd_rho))) return(sd_rho)
  }
  list(reason = .hp_decline("logchol_design_measure"))
}

# A block's rows in the coordinates of `design`: the log-Cholesky columns
# themselves, or (log sigma_1, log sigma_2, rho) for "sd_rho". Both are
# pointwise maps, so they carry any rows, not only the design's own nodes.
.hp_logchol_coords <- function(M, design) {
  M <- as.matrix(M)
  if (identical(design$design, "logchol")) return(M)
  s1 <- exp(M[, 1L])
  s2 <- sqrt(M[, 2L]^2 + exp(2 * M[, 3L]))
  cbind(log(s1), log(s2), M[, 2L] / s2)
}

# Each free-covariance block among `axes` with the design its grid declares,
# keyed by `.hp_logchol_key()`. `grid` is the grid the fit declared -- a block's
# own grid or a joint tensor over it -- never a batch evaluated from it.
.hp_logchol_designs <- function(grid, axes = colnames(grid)) {
  out <- list()
  lc <- axes[.hp_is_logchol_col(sub("^b[0-9]+[.]", "", axes))]
  for (pre in unique(.hp_col_prefix_vec(lc))) {
    cols <- .hp_logchol_block_cols(colnames(grid), pre)
    out[[.hp_logchol_key(pre)]] <-
      if (is.null(cols)) list(reason = .hp_decline("logchol_partial_block"))
      else .hp_logchol_design(grid[, cols, drop = FALSE])
  }
  out
}

# What a declared grid fixes for every batch evaluated from it: the columns it
# integrates and each free-covariance block's design.
.hp_declare <- function(grid, axes = .hp_integrated_axes(grid)) {
  list(axes = axes, logchol = .hp_logchol_designs(grid, axes))
}

# The declaration for points laid in the grid's COLUMN coordinates -- a CCD
# design and its mode-find, importance draws on the identity transform a
# free-covariance axis takes: every block with a design takes the log-Cholesky
# one. On a two-field block that is the "sd_rho" density carried by the map's
# Jacobian, so the density over those points is a density in the coordinates
# they were laid in.
.hp_declare_on_columns <- function(declared) {
  declared$logchol <- lapply(declared$logchol, function(d) {
    if (!is.null(d$design)) d$design <- "logchol"
    d
  })
  declared
}

# Per-cell log prior density of a free-covariance block at the rows of `M`, in
# the coordinates of `design` (a `.hp_logchol_design()` record): independent PC
# priors on the marginal standard deviations and an LKJ prior on the
# correlation, the `re_cov_pc_lkj_prior()` default at the engine's anchor. On a
# log-Cholesky design that is the PC + LKJ density pushed to the log-Cholesky
# coordinates; on the two-field default it is the same prior on
# (log sigma_1, log sigma_2, rho) directly, where the LKJ(eta) density at d = 2
# is (1 - rho^2)^(eta - 1) / c_2(eta).
.hp_logchol_log_density <- function(M, design) {
  if (is.null(design$design)) return(list(reason = design$reason))
  anchor <- .nl_scale_anchor()
  eta <- .nl_hyperprior("lkj_eta")
  if (identical(design$design, "sd_rho")) {
    C <- .hp_logchol_coords(M, design)
    rho <- C[, 3L]
    lp <- .hp_log_scale_density(exp(C[, 1L]), "sd", anchor) +
      .hp_log_scale_density(exp(C[, 2L]), "sd", anchor) +
      (eta - 1) * log1p(-rho^2) - .lkj_log_normaliser(2L, eta)
  } else {
    f <- .re_cov_block_logprior(design$p, TRUE, anchor, eta)
    lp <- apply(as.matrix(M), 1L, f)
  }
  list(lp = as.numeric(lp))
}

# Per-cell default log hyperprior of one registry block over its own axis
# columns (bare names, natural values). A tgmrf block's own `prior(theta)` is
# folded inside the kernel (`log_prior_theta_per_grid`), so it is recorded here
# and not added again. `declared` is the `.hp_declare()` record of the block's
# declared grid.
.nl_block_log_hyperprior <- function(p, tg, hyperprior = "proper", declared) {
  axes <- intersect(declared$axes, colnames(tg))
  if (identical(tolower(p$type %||% ""), "tgmrf")) {
    n <- nrow(tg)
    lpk <- p$log_prior_theta_per_grid
    return(list(lp = numeric(n),
                lp_in_kernel = if (length(lpk) == n) as.numeric(lpk) else numeric(n),
                axes = axes, declined = character(0)))
  }
  .hp_collect(tg, function(a) .hp_axis_prior(a, p, hyperprior = hyperprior),
              axes = axes, logchol = declared$logchol)
}

# The default-or-user hyperprior over a joint grid. `blocks` is the block list
# the grid's `b<k>.` prefixes index (a single-block joint grid is unprefixed and
# passes its one block), `families` names each arm's family by the `phi_<arm>`
# suffix, and `user` holds the caller's natural-scale densities by role
# (`sigma`, `alpha`, `phi`), each a function or a per-block list the front door
# parsed. A user density replaces the default on every axis of its role, and
# `hyperprior` (`.HP_CHOICES`) sets what every other axis carries.
#
# `declared` holds the columns the fit integrates and each free-covariance
# block's design, read once off the grid the caller declared (`.hp_declare()` of
# the whole grid, or `.joint_multi_declared_axes()`), and never off `theta_grid`
# itself. The
# drivers evaluate the record over batches -- refinement slices, CCD and
# mode-find points, importance draws -- whose cells share coordinates on axes
# the fit integrates, and a column constant in a batch is not a fixed setting of
# the model (gcol33/tulpa#760).
.joint_hyperprior <- function(theta_grid, blocks, families = NULL,
                              user = list(), copy_atom_mass = .TULPA_COPY_ATOM_MASS,
                              hyperprior = "proper", declared) {
  tg <- as.matrix(theta_grid)
  n  <- nrow(tg)
  if (is.null(colnames(tg)) || !n) {
    return(list(lp = numeric(n), lp_in_kernel = numeric(n),
                axes = character(0), declined = character(0)))
  }
  block_of <- function(col) {
    k <- suppressWarnings(as.integer(sub("^b([0-9]+)[.].*$", "\\1", col)))
    if (grepl("^b[0-9]+[.]", col) && !is.na(k)) {
      return(list(block = if (k <= length(blocks)) blocks[[k]] else list(),
                  index = k))
    }
    list(block = if (length(blocks) == 1L) blocks[[1L]] else list(),
         index = if (length(blocks) == 1L) 1L else NA_integer_)
  }
  user_fn <- function(role, index) {
    fn <- user[[role]]
    if (is.null(fn) || is.function(fn)) return(fn)
    .joint_hp_fn_for_block(fn, index)
  }
  .hp_collect(tg, function(col) {
    bare <- sub("^b[0-9]+[.]", "", col)
    bo   <- block_of(col)
    role <- if (startsWith(bare, "phi_")) "phi"
            else if (identical(bare, "alpha")) "alpha"
            else if (identical(bare, "sigma")) "sigma"
            else NA_character_
    fn <- if (is.na(role)) NULL else user_fn(role, bo$index)
    if (!is.null(fn)) {
      log_scale <- isTRUE(.hyper_axis_scale(bare))
      is_alpha  <- identical(role, "alpha")
      return(list(fn = function(x) {
        lp <- .hyper_prior_carry(x, fn, log_scale, "natural")
        # The copy scale's zero is its declared point mass, which the density
        # does not describe (gcol33/tulpa#624, gcol33/tulpa#626).
        if (is_alpha) {
          lp[.hyper_is_atom_level(x, list(atom_mass = copy_atom_mass,
                                          log_scale = log_scale))] <- 0
        }
        lp
      }))
    }
    fam <- if (identical(role, "phi")) families[[sub("^phi_", "", bare)]] else NULL
    .hp_axis_prior(bare, bo$block, fam, hyperprior)
  }, axes = intersect(declared$axes, colnames(tg)), logchol = declared$logchol)
}

# Fold one or more hyperprior records into a kernel result. Each record's axis
# names are the ones the result's `theta_grid` carries.
.nl_fold_hyperprior <- function(res, parts) {
  for (hp in parts) {
    res$log_marginal <- res$log_marginal + hp$lp
    res$log_hyperprior <- (res$log_hyperprior %||% 0) + hp$lp + hp$lp_in_kernel
    if (any(hp$lp_in_kernel != 0)) {
      res$log_hyperprior_in_kernel <- (res$log_hyperprior_in_kernel %||% 0) +
        hp$lp_in_kernel
    }
    res$log_hyperprior_axes <- union(res$log_hyperprior_axes %||% character(0),
                                     hp$axes)
    res$log_hyperprior_declined <- c(res$log_hyperprior_declined, hp$declined)
  }
  res
}

# The per-cell hyperprior R folded into `log_marginal`, over `n` cells: zero when
# nothing was folded, NULL when the record does not align with the grid.
.nl_log_hyperprior_folded <- function(res, n) {
  lh <- as.numeric(res$log_hyperprior %||% 0) -
    as.numeric(res$log_hyperprior_in_kernel %||% 0)
  if (length(lh) == 1L) return(rep(lh, n))
  if (length(lh) != n) return(NULL)
  lh
}

# Prefix every axis name a block record carries with the block tag the
# multi-block grid uses (`b2.tau`).
.hp_prefix <- function(hp, prefix) {
  if (length(hp$axes)) hp$axes <- paste0(prefix, hp$axes)
  if (length(hp$declined)) names(hp$declined) <- paste0(prefix, names(hp$declined))
  hp
}

# The integrated axes a grid's evidence cannot be read over: every axis carrying
# neither a folded density nor a normalised one declared on its spec, with the
# reason. `specs` are the axis specs the cell measure was built from.
.nl_evidence_uncovered <- function(tg, specs, res) {
  integrated <- .hp_integrated_axes(tg)
  if (!length(integrated)) return(character(0))
  covered <- res$log_hyperprior_axes %||% character(0)
  declared <- character(0)
  for (s in specs %||% list()) {
    if (!is.null(s$slab_log_density)) declared <- c(declared, s$name)
  }
  out <- res$log_hyperprior_declined %||% character(0)
  for (a in setdiff(integrated, c(covered, declared, names(out)))) {
    out[[a]] <- .hp_decline("unclassified_axis")
  }
  out[names(out) %in% integrated]
}
