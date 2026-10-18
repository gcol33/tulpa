# field_effects.R
# ------------------------------------------------------------------------------
# Per-unit posterior of a fit's spatial field. The field lives in a different
# place on every backend -- the conditional-Laplace mode, a sampler's draws, a
# Gibbs chain's field columns, a nested-Laplace grid's per-cell modes, an SPDE
# fit's mesh-node effects -- and each reader below locates it through the same
# helper the rest of the package reads that backend's field with
# (`R/posterior_predict.R`, `R/temporal_rtr_posteriors.R`,
# `R/nested_laplace.R`), so the reported field is the one eta was assembled
# from.
# ------------------------------------------------------------------------------

#' Spatial field effects per unit
#'
#' @description
#' The posterior of a fit's spatial field, one row per spatial unit: an areal
#' region (a row of the adjacency matrix) or an SPDE mesh node. The value
#' reported is the field's contribution to the linear predictor at that unit,
#' the quantity a covariate is read against.
#'
#' @details
#' Where the posterior comes from follows what the fit computes:
#' \itemize{
#'   \item A ModelData sampler fit (`mode = "hmc"` and its siblings) with an
#'     ICAR, BYM2 or proper-CAR field: every draw's field contribution is
#'     evaluated through the engine's own linear-predictor assembly, so an
#'     intrinsic field is centred as its density centres it and a BYM2 field is
#'     the combined `sigma * (sqrt(rho) * s * phi + sqrt(1 - rho) * theta)` at
#'     that draw's `sigma` and `rho` (`s` the scaling factor). A unit that no
#'     observation falls in contributes nothing to the linear predictor and is
#'     reported as `NA`.
#'   \item A Polya-Gamma Gibbs fit (`mode = "gibbs"`) stores the field it
#'     samples (`phi_spatial[k]`); for BYM2 each draw's two unit-scale
#'     components are combined into the same
#'     `sigma * (sqrt(rho) * s * phi + sqrt(1 - rho) * theta)` the sampler
#'     path reports. The rows summarize those draws.
#'   \item A nested-Laplace fit with an ICAR or proper-CAR field: each outer
#'     cell contributes its conditional mode, and its marginal variance where
#'     the fit retained one, weighted by the cell's posterior weight. With the
#'     variances the summary is the exact mean, SD and quantiles of that
#'     Gaussian mixture (`source = "mixture"`); without them the weighted mean
#'     of the cell modes, with no spread (`source = "mode"`). A BYM2 field is
#'     reported where the fit stores its grid-weighted combined field
#'     (`fit_st_nested()`); a `tulpa()` nested BYM2 fit carries its two
#'     components with cell-dependent mixing and errors.
#'   \item A conditional-Laplace areal fit (`mode = "laplace"` / `"eb"`)
#'     reports the field at its mode, with no spread. For BYM2 this is the
#'     combined field at the `sigma = 1`, `rho = 0.5` the conditional kernel
#'     fixes.
#'   \item An SPDE fit ([fit_spde()]) reports the mesh-node field: its draws on
#'     the sampler path, its mode (the posterior-mean hyperparameters' fit on
#'     the nested path) otherwise.
#' }
#' A fit whose field cannot be located this way (a continuous GP / NNGP /
#' HSGP field, a restricted field, an inline `spatial()` varying-coefficient
#' field) errors naming the field and the backend, as does a fit with no
#' spatial field. A temporal field is read by [temporal()].
#'
#' @param object A `tulpa_fit` carrying a spatial field.
#' @return A data frame with one row per spatial unit: `unit` (the node
#'   index), `label` (the adjacency row name where it has one, else the index),
#'   `estimate`, `sd`, the 2.5% / 97.5% bounds `conf.low` / `conf.high`, and
#'   `source`: `"sampled"` for a posterior draw summary, `"mixture"` for a
#'   nested-Laplace Gaussian mixture, `"mode"` for a point at the mode (`sd`
#'   and the bounds `NA`). The field type is attached as attribute `"field"`.
#' @seealso [smooth_effects()] for an `s(x)` term, [ranef()] for random
#'   effects, [temporal()] for a temporal field.
#' @examples
#' \donttest{
#' set.seed(1)
#' K <- 12
#' W <- matrix(0, K, K)
#' for (i in 1:K) { j <- if (i < K) i + 1 else 1; W[i, j] <- W[j, i] <- 1 }
#' region <- sample(1:K, 200, replace = TRUE)
#' d <- data.frame(region = region, x = rnorm(200))
#' d$y <- rpois(200, exp(0.3 + 0.5 * d$x + sin(2 * pi * region / K)))
#' fit <- tulpa(y ~ x + spatial(region), data = d, family = "poisson",
#'              spatial = list(type = "icar", adjacency = W), mode = "laplace")
#' fe <- field_effects(fit)
#' plot(fe$unit, fe$estimate, type = "b")
#' }
#' @export
field_effects <- function(object) {
  read <- .field_effects_read(object)
  n <- nrow(read$stats)
  out <- data.frame(
    unit   = seq_len(n),
    label  = .field_unit_labels(object, n),
    read$stats,
    source = read$source,
    row.names = NULL, stringsAsFactors = FALSE
  )
  attr(out, "field") <- read$field
  out
}

.FIELD_EFFECTS_PROBS <- c(0.025, 0.975)

.FIELD_RESTRICTED_WHY <- paste0(
  "a restricted field enters the linear predictor through its projection, ",
  "so its coordinates are not the per-unit field.")

# Draw columns of an areal field on a ModelData sampler fit, as
# `sampler_param_names()` (`src/sampler_model_data.h`) names them: the field's
# own coordinates and the hyperparameters its contribution to eta reads.
.SAMPLER_AREAL_FIELD_COLS <- paste0(
  "^(phi_spatial|theta_spatial)\\[[0-9]+\\]$|",
  "^(log_tau_spatial|log_sigma_spatial|logit_rho_bym2|logit_rho_car)$")

# One reader's result: the per-unit estimate / sd / bounds table, the
# construction it came from, and the field type.
.field_read <- function(stats, source, field) {
  list(stats = stats, source = source, field = field)
}

.field_point_stats <- function(v) {
  data.frame(estimate = as.numeric(v), sd = NA_real_, conf.low = NA_real_,
             conf.high = NA_real_)
}

# Per-unit summary of an S x n_units draw matrix through `.ranef_empirical()`;
# a column that is not finite on every draw (a unit the reader could not
# reach) is a row of NA.
.field_draws_stats <- function(U) {
  n <- ncol(U)
  out <- data.frame(estimate = rep(NA_real_, n), sd = NA_real_,
                    conf.low = NA_real_, conf.high = NA_real_)
  ok <- apply(is.finite(U), 2L, all)
  if (any(ok)) out[ok, ] <- .ranef_empirical(U[, ok, drop = FALSE])
  out
}

.field_unit_labels <- function(object, n) {
  adj <- object$spatial$adjacency
  rn <- if (!is.null(adj)) rownames(adj)
  if (length(rn) == n) as.character(rn) else as.character(seq_len(n))
}

.field_type <- function(object) {
  tolower(object$spatial$type %||% object$spatial_type %||% "")
}

.field_effects_unavailable <- function(object, field, why) {
  backend <- object$backend %||% class(object)[1L]
  stop(sprintf("field_effects() cannot read the '%s' spatial field of a '%s' fit: %s",
               field, backend, why), call. = FALSE)
}

.field_effects_read <- function(object) {
  if (!inherits(object, "tulpa_fit")) {
    stop("field_effects() needs a `tulpa_fit` object.", call. = FALSE)
  }
  type <- .field_type(object)
  nl <- .nl_spatial_field_layout(object)
  if (!nzchar(type) && !is.null(nl)) type <- nl$type
  if (!nzchar(type)) {
    hint <- if (!is.null(object$temporal)) {
      " Its temporal field is read by temporal()."
    } else ""
    stop("field_effects(): this fit carries no spatial field.", hint,
         call. = FALSE)
  }

  if (type == "svc") {
    stop("field_effects(): this fit carries spatially-varying coefficients, ",
         "not a spatial field. Use svc() to read them.", call. = FALSE)
  }
  if (isTRUE(object$spatial$rsr)) {
    .field_effects_unavailable(object, type, .FIELD_RESTRICTED_WHY)
  }
  if (type == "spde") return(.field_effects_spde(object))
  if (identical(.tulpa_linpred_source(object), "sampler_model")) {
    return(.field_effects_sampler(object, type))
  }
  if (is.matrix(object$draws) && nrow(object$draws) > 0L) {
    U <- .gibbs_field_unit_draws(object, seq_len(nrow(object$draws)))
    if (!is.null(U)) {
      return(.field_read(.field_draws_stats(U), "sampled", type))
    }
  }
  if (is.matrix(object$modes) && !is.null(object$weights)) {
    return(.field_effects_nested(object, nl, type))
  }
  n_obs <- if (is.matrix(object$model_matrix)) nrow(object$model_matrix)
           else length(object$spatial$spatial_idx)
  lf <- .laplace_areal_field(object, n_obs)
  if (!is.null(lf)) {
    v <- as.numeric(lf$design$unit_map %*% lf$u)
    return(.field_read(.field_point_stats(v), "mode", type))
  }
  .field_effects_unavailable(object, type, paste0(
    "no reader locates this field on this backend (the areal fields are read ",
    "on the sampler, Gibbs, nested-Laplace and conditional-Laplace paths, an ",
    "SPDE field on fit_spde() fits)."))
}

# SPDE mesh-node field: the sampler's field draws where it exposes them
# (`field_draws`, `R/fit_spde_nuts.R`), else the mode the Laplace and nested
# paths store as `spatial_effects`.
.field_effects_spde <- function(object) {
  if (is.matrix(object$field_draws) && nrow(object$field_draws) > 0L) {
    return(.field_read(.field_draws_stats(object$field_draws), "sampled",
                       "spde"))
  }
  if (is.numeric(object$spatial_effects) && length(object$spatial_effects)) {
    return(.field_read(.field_point_stats(object$spatial_effects), "mode",
                       "spde"))
  }
  .field_effects_unavailable(object, "spde",
    "it stores neither mesh-node field draws nor mesh-node effects.")
}

# Areal field on a ModelData sampler fit. Every draw is evaluated through the
# engine's own eta assembly (`.sampler_engine_eta()`) with every coordinate
# outside the field and its hyperparameters set to zero and no offset: the
# linear predictor is linear in each block's latent coordinates and zero at
# zero, so what is left at observation i is the field's contribution there,
# centred and scaled exactly as the density put it into eta. Each unit's value
# is read at its first observation.
.field_effects_sampler <- function(object, type) {
  ss <- object$model_inputs$spatial_spec
  stype <- tolower(ss$type %||% "")
  D <- object$draws
  dn <- colnames(D)
  if (!stype %in% c("icar", "bym2", "car_proper") || is.null(dn) ||
      !any(grepl("^phi_spatial\\[", dn))) {
    .field_effects_unavailable(object, type,
      "the sampler path reads an ICAR, BYM2 or proper-CAR field only.")
  }
  D[, !grepl(.SAMPLER_AREAL_FIELD_COLS, dn)] <- 0
  eta <- .sampler_engine_eta(object, D, offset = NULL)
  n_units <- as.integer(ss$n_spatial_units)
  first <- match(seq_len(n_units), as.integer(ss$spatial_idx))
  U <- matrix(NA_real_, nrow(D), n_units)
  hit <- !is.na(first)
  U[, hit] <- eta[, first[hit], drop = FALSE]
  .field_read(.field_draws_stats(U), "sampled", stype)
}

# Where a nested-Laplace fit's spatial field sits in its latent vector: the
# layout `fit_st_nested()` records (`field_cols`, `spatial_type`), else the
# spatial-role block of the fit's prior located through `.nl_role_cols()`.
# A single-block solve lays out [fixed, RE, field], so a lone ICAR /
# proper-CAR block, one coordinate per unit, is the tail of that width. NULL
# when the fit carries no spatial block.
.nl_spatial_field_layout <- function(object) {
  M <- object$modes
  if (!is.matrix(M)) return(NULL)
  n_x <- ncol(M)
  sp_cols <- object$field_cols$spatial
  if (!is.null(sp_cols) && is.character(object$spatial_type)) {
    type <- tolower(object$spatial_type)
    n_cols <- length(sp_cols)
    return(list(type = type, cols = as.integer(sp_cols),
                n_units = if (type == "bym2") n_cols %/% 2L else n_cols,
                projected = FALSE))
  }
  prior <- object$prior
  roles <- .nl_block_roles(prior)
  k <- which(roles == "spatial")[1L]
  if (is.na(k)) return(NULL)
  single <- !is.null(prior$type)
  blk <- if (single) prior else prior[[k]]
  type <- tolower(blk$type %||% "")
  n_units <- as.integer(blk$n_spatial_units %||% NA_integer_)
  cols <- .nl_role_cols(prior, object$block_latent_offsets, n_x, "spatial")
  if (is.null(cols) && single && type %in% c("icar", "car_proper") &&
      !is.na(n_units) && n_units >= 1L && n_units <= n_x) {
    cols <- seq.int(n_x - n_units + 1L, n_x)
  }
  list(type = type, cols = cols, n_units = n_units,
       projected = !is.null(blk$projector))
}

# A nested-Laplace areal field. An ICAR / proper-CAR block's coordinates ARE
# the per-unit field (loading 1), so the grid posterior is the per-cell
# Gaussian mixture `.nl_field_mixture()` assembles, summarized exactly by
# `.nl_gauss_mixture_summary()` -- the construction ranef() and temporal()
# report a nested fit's latent coordinates with.
.field_effects_nested <- function(object, nl, type) {
  if (is.null(nl)) {
    .field_effects_unavailable(object, type,
      "the fit records no spatial block in its latent layout.")
  }
  if (isTRUE(nl$projected)) {
    .field_effects_unavailable(object, nl$type, .FIELD_RESTRICTED_WHY)
  }
  if (nl$type %in% c("icar", "car", "car_proper")) {
    if (is.null(nl$cols) || length(nl$cols) != nl$n_units) {
      .field_effects_unavailable(object, nl$type,
        "its block's columns cannot be located from the recorded block sizes.")
    }
    mix <- .nl_field_mixture(object, nl$cols)
    mx <- if (!is.null(mix)) {
      .nl_gauss_mixture_summary(mix$mu, mix$var, mix$w,
                                probs = .FIELD_EFFECTS_PROBS)
    }
    if (is.null(mx)) {
      .field_effects_unavailable(object, nl$type,
        "no outer-grid cell carries a finite mode and a positive weight.")
    }
    stats <- data.frame(estimate = mx$mean, sd = mx$sd,
                        conf.low = mx$quantiles[, 1L],
                        conf.high = mx$quantiles[, 2L])
    return(.field_read(stats, if (is.null(mix$var)) "mode" else "mixture",
                       nl$type))
  }
  if (nl$type == "bym2") {
    se <- object$spatial_effects
    if (is.numeric(se) && length(se) == nl$n_units) {
      return(.field_read(.field_point_stats(se), "mode", "bym2"))
    }
    .field_effects_unavailable(object, "bym2", paste0(
      "its latent carries the structured and unstructured components ",
      "separately, mixed by the cell's (sigma, rho), and the fit stores no ",
      "combined per-unit field."))
  }
  .field_effects_unavailable(object, nl$type,
    "the nested-Laplace path reads an ICAR, proper-CAR or BYM2 field only.")
}
