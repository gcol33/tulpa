# fit_st_nested.R
# ------------------------------------------------------------------------------
# Additive spatiotemporal GLM from a response vector, a design matrix and
# integer indices: a spatial field (areal icar / bym2 / car_proper, continuous
# hsgp / nngp) plus a temporal field (rw1 / rw2 / ar1), each a latent block of
# one multi-block nested-Laplace fit. The blocks carry the `role` tags tulpa()
# gives its `spatial =` / `temporal =` fields, so the fit is the one-arm joint
# fit `tulpa_nested_laplace()` makes of a multi-block prior: placed, refined and
# scored by the joint door, and read back by the generic accessors through the
# block roles.

# The spatial block of a fit_st_nested() fit, in the form a `spatial =` field
# reaches the nested path (`.spatial_spec_to_nl_prior()` for the areal fields,
# the shared HSGP and NNGP block builders for the continuous ones).
.st_spatial_block <- function(spatial_type, spatial_idx, adjacency, coords, nn,
                              cov_type, hsgp_m, hsgp_c, kernel) {
  blk <- switch(spatial_type,
    hsgp = .hsgp_nl_block(coords, hsgp_m, hsgp_c, kernel),
    nngp = {
      n_s <- nrow(coords)
      nn  <- min(as.integer(nn), n_s - 1L)
      b <- .nngp_nl_block(list(unique_coords = coords, n_spatial = n_s,
                               obs_to_loc = spatial_idx),
                          compute_nngp_neighbors(coords, nn), nn)
      b$cov_type <- as.integer(cov_type)
      b
    },
    {
      spec <- list(type = spatial_type, adjacency = adjacency,
                   spatial_idx = spatial_idx)
      if (spatial_type == "bym2") {
        sc <- .bym2_component_scaling(adjacency)
        spec$scale_factor <- sc$scale_factor
        spec$node_prec <- sc$node_prec
      }
      if (spatial_type == "car_proper") {
        spec$rho_bounds <- compute_car_rho_bounds(adjacency)
      }
      .spatial_spec_to_nl_prior(spec)
    })
  blk$role <- "spatial"
  blk
}

# Grid-weighted posterior means of the two fields. An areal field is read
# through its per-unit loading (`.nl_spatial_field_load()`: weight 1 on an
# ICAR / proper-CAR coordinate, BYM2's cell-dependent mix of its two
# components); an NNGP block's coordinates are the field at its locations; an
# HSGP field is its basis coordinates projected through each cell's spectral
# loading at the observations.
.st_field_means <- function(fit, spatial_type, sp_block, N) {
  M <- fit$modes
  n_x <- ncol(M)
  w <- as.numeric(fit$weights)
  tg <- fit$theta_grid
  offs <- fit$block_latent_offsets
  sp_cols <- .nl_role_cols(fit$prior, offs, n_x, "spatial")
  te_cols <- .nl_role_cols(fit$prior, offs, n_x, "temporal")
  live <- which(is.finite(w) & w > 0)
  spatial <- if (spatial_type == "hsgp") {
    phi <- as.matrix(sp_block$phi_basis)
    code <- .hsgp_kernel_code(sp_block$kernel)
    acc <- numeric(N)
    for (k in live) {
      spectral <- cpp_hsgp_sqrt_spectral(sp_block$lambda_eig,
        tg[k, "b1.sigma2"], tg[k, "b1.lengthscale"], code)
      loading <- if (ncol(phi) > N)
        cpp_hsgp_observation_projection(phi, phi, spectral)$basis
        else sweep(phi, 2L, spectral, "*")
      acc <- acc + w[k] * as.numeric(loading %*% M[k, sp_cols])
    }
    acc
  } else if (spatial_type == "nngp") {
    as.numeric(crossprod(w[live], M[live, sp_cols, drop = FALSE]))
  } else {
    ld <- .nl_spatial_field_load(fit$prior, offs, n_x, tg)
    acc <- numeric(ld$n_units)
    for (k in live) {
      wk <- ld$weights(k)
      acc <- acc + w[k] * Reduce(`+`, lapply(seq_along(ld$cols), function(c)
        wk[[c]] * M[k, ld$cols[[c]]]))
    }
    acc
  }
  list(spatial = spatial,
       temporal = as.numeric(crossprod(w[live], M[live, te_cols, drop = FALSE])))
}


#' Fit an additive spatiotemporal GLM by nested Laplace
#'
#' @description
#' Fits `y ~ X beta + u_spatial[s] + v_temporal[t]` with a spatial field --
#' areal (`icar` / `bym2` / `car_proper`) or continuous (`hsgp` / `nngp`) --
#' and a temporal field (`rw1` / `rw2` / `ar1`). The two fields are latent
#' blocks of one nested-Laplace fit ([tulpa_nested_laplace()] with a two-block
#' `prior`), which integrates the spatial hyperparameter(s), the temporal
#' precision and (for `ar1`) the temporal autocorrelation on its outer grid.
#' The fixed-effect posterior is the grid-marginalised mixture; the spatial and
#' temporal field posterior means are the grid-weighted latent modes.
#'
#' @param y Response vector.
#' @param X Fixed-effects design matrix (`nrow(X) == length(y)`).
#' @param spatial_idx Integer per-observation spatial-unit index (1-based).
#'   For `spatial_type = "icar"/"bym2"/"car_proper"`, an areal unit in
#'   `[1, nrow(adjacency)]`; for `"nngp"`, a location in `[1, nrow(coords)]`.
#'   Ignored for `"hsgp"` (the field is evaluated directly at each
#'   observation's own coordinates); pass any placeholder (e.g. `seq_len(N)`).
#' @param adjacency Spatial adjacency (a symmetric 0/1 matrix or `sparseMatrix`).
#'   Required for `spatial_type = "icar"/"bym2"/"car_proper"`; ignored (pass
#'   `NULL`) for `"hsgp"`/`"nngp"`, which take `coords` instead.
#' @param temporal_idx Integer per-observation time index (1-based).
#' @param n_times Number of distinct time points.
#' @param spatial_type `"icar"` (default), `"bym2"`, `"car_proper"`, `"hsgp"`
#'   (Hilbert-space GP basis), or `"nngp"` (nearest-neighbour GP).
#' @param temporal_type `"ar1"` (default), `"rw1"`, or `"rw2"`.
#' @param family Response family (see [family_names()]).
#' @param n_trials Binomial denominators, or `NULL` (= 1).
#' @template phi
#' @param cyclic Logical; wrap the temporal field (seasonal). Default `FALSE`.
#' @param re_idx,n_re_groups,sigma_re Optional single iid random-intercept term
#'   alongside the fields (conditioned on `sigma_re`); `n_re_groups = 0` (default)
#'   is no RE term.
#' @template hyperprior
#' @param coords Coordinate matrix for a continuous spatial field, required
#'   when `spatial_type` is `"hsgp"` or `"nngp"` (ignored otherwise). For
#'   `"hsgp"`, an `N x 2` matrix (one row per observation, matching
#'   `spatial_gp(approx = "hsgp")`'s basis convention -- the basis is built by
#'   `cpp_hsgp_basis_2d()`, 2D only). For `"nngp"`, an `n_spatial x d` matrix
#'   of unique locations that `spatial_idx` indexes into (any `d`, matching
#'   [spatial_gp()]'s NNGP convention).
#' @param nn Number of nearest neighbours per location, `spatial_type = "nngp"`
#'   only. Default 10 (clamped to `nrow(coords) - 1`).
#' @param cov_type Integer NNGP covariance code (`spatial_type = "nngp"` only):
#'   0 = exponential, 1 = Matern 3/2, 2 = Matern 5/2 (default), 3 = Gaussian.
#' @param hsgp_m,hsgp_c Hilbert-space GP basis size per dimension (3 to 50)
#'   and boundary factor (`>= 1`), `spatial_type = "hsgp"` only -- the
#'   parameterisation of `spatial_gp(approx = "hsgp")`. A value given is held
#'   as given; `NULL` (default) leaves it to the data by the rule and the
#'   refit loop [spatial_gp()] describes, and the fit records the bases it
#'   went through as `hsgp_basis`.
#' @param hsgp_cov,hsgp_nu The kernel the HSGP basis approximates,
#'   `spatial_type = "hsgp"` only, read as [spatial_gp()] reads `cov` / `nu`:
#'   `"matern"` (`nu` 1.5 or 2.5) or `"squared_exponential"`; `NULL` (default)
#'   is Matern 3/2.
#' @param control The numerical knobs of [tulpa_nested_laplace()] (`max_iter`,
#'   `tol`, `n_threads`, `auto_recenter`, `axis_refine`, `within_cell`, ...),
#'   passed to it unchanged. The outer grid starts from each block family's
#'   default axes and is placed and refined there.
#'
#' @return A `tulpa_fit` (subclass `tulpa_st_nested` of
#'   `tulpa_nested_laplace`): the multi-block fit [tulpa_nested_laplace()]
#'   returns for the two blocks, block 1 the spatial field and block 2 the
#'   temporal one, so its outer axes are named `b1.<axis>` / `b2.<axis>`
#'   (`b1.tau` for an ICAR or proper-CAR field, `b1.sigma` / `b1.rho` for
#'   BYM2, `b1.sigma2` with `b1.lengthscale` / `b1.phi_gp` for HSGP / NNGP;
#'   `b2.tau`, and `b2.rho` under `ar1`). It also carries `spatial_effects`
#'   and `temporal_effects` (grid-weighted field posterior means: per areal
#'   unit or NNGP location, per observation for HSGP, per time point),
#'   `temporal` (the time layout [temporal()] reads), `spatial_type` and
#'   `temporal_type`.
#'
#' @seealso [tulpa()] (front door), [tulpa_nested_laplace()].
#' @examples
#' \donttest{
#' set.seed(1)
#' n_s <- 16L; n_t <- 8L; N <- 400L
#' adj <- matrix(0, n_s, n_s)
#' for (i in 1:(n_s - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
#' s <- sample(n_s, N, TRUE); tt <- sample(n_t, N, TRUE)
#' us <- as.numeric(scale(cumsum(rnorm(n_s)))); vt <- as.numeric(scale(cumsum(rnorm(n_t))))
#' x <- rnorm(N)
#' y <- rbinom(N, 1, plogis(0.2 + 0.5 * x + 0.7 * us[s] + 0.6 * vt[tt]))
#' fit <- fit_st_nested(y, cbind(1, x), s, adj, tt, n_t, family = "binomial")
#' }
#' @export
fit_st_nested <- function(y, X, spatial_idx, adjacency, temporal_idx, n_times,
                          spatial_type = c("icar", "bym2", "car_proper",
                                          "hsgp", "nngp"),
                          temporal_type = c("ar1", "rw1", "rw2"),
                          family = "binomial", n_trials = NULL, phi = 1.0,
                          cyclic = FALSE,
                          re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                          hyperprior = c("proper", "flat"),
                          control = list(),
                          coords = NULL, nn = 10L, cov_type = 2L,
                          hsgp_m = NULL, hsgp_c = NULL,
                          hsgp_cov = NULL, hsgp_nu = 1.5) {
  tulpa_check_control(control, .CONTROL_KEYS$nested_laplace, "fit_st_nested")
  spatial_type  <- match.arg(spatial_type)
  door <- "fit_st_nested(spatial_type = 'hsgp')"
  if (spatial_type == "hsgp" && is.null(coords)) {
    stop("`coords` is required for spatial_type = 'hsgp' (an N x 2 matrix, ",
         "one row per observation).", call. = FALSE)
  }
  if (spatial_type == "hsgp" && (is.null(hsgp_m) || is.null(hsgp_c))) {
    # A basis left to the data: the fit below runs at a given basis, through
    # the HSGP basis loop that sizes it.
    cm <- .coords_2col(as.matrix(coords), door)
    spec <- .hsgp_basis_init(list(m = .hsgp_check_m(hsgp_m, "hsgp_m"),
                                  c = .hsgp_check_c(hsgp_c, "hsgp_c"),
                                  kernel = .hsgp_kernel(hsgp_cov, hsgp_nu, door)),
                             cm, .hsgp_prior_lengthscale(cm), "c", door)
    cl <- match.call()
    env <- parent.frame()
    run <- .hsgp_fit_basis_loop(spec, function(s) {
      cl$hsgp_m <- s[["m"]]
      cl$hsgp_c <- s[["c"]]
      eval(cl, env)
    })
    run$fit$hsgp_basis <- run$spec
    return(run$fit)
  }
  temporal_type <- match.arg(temporal_type)
  hyperprior    <- match.arg(hyperprior)
  family <- .canonical_family(family)
  .family_or_stop(family)
  X <- as.matrix(X)
  N <- .validate_glm_design(y, X, n_trials, "fit_st_nested",
                            missing_response = TRUE)$N

  if (length(temporal_idx) != N) {
    stop("`temporal_idx` must have length length(y).", call. = FALSE)
  }
  if (max(temporal_idx) > n_times || min(temporal_idx) < 1L) {
    stop("`temporal_idx` must be 1-based indices in [1, n_times].", call. = FALSE)
  }
  if (spatial_type == "hsgp") {
    coords <- .coords_2col(as.matrix(coords), door)
    if (nrow(coords) != N) {
      stop("`coords` must have one row per observation (", N, ") for ",
           "spatial_type = 'hsgp'; got ", nrow(coords), " row(s).",
           call. = FALSE)
    }
  } else {
    if (spatial_type == "nngp") {
      if (is.null(coords)) {
        stop("`coords` is required for spatial_type = 'nngp' (an n_spatial x d ",
             "matrix of unique locations `spatial_idx` indexes into).",
             call. = FALSE)
      }
      coords <- as.matrix(coords)
      storage.mode(coords) <- "double"
      n_s <- nrow(coords)
      what <- "nrow(coords)"
    } else {
      n_s <- nrow(as.matrix(adjacency))
      what <- "nrow(adjacency)"
    }
    if (length(spatial_idx) != N) {
      stop("`spatial_idx` must have length length(y).", call. = FALSE)
    }
    if (max(spatial_idx) > n_s || min(spatial_idx) < 1L) {
      stop("`spatial_idx` must be 1-based indices in [1, ", what, "].",
           call. = FALSE)
    }
  }

  sp <- .st_spatial_block(spatial_type, as.integer(spatial_idx), adjacency,
                          coords, nn, cov_type, hsgp_m, hsgp_c,
                          .hsgp_kernel(hsgp_cov, hsgp_nu, door))
  tm <- list(type = temporal_type, temporal_idx = as.integer(temporal_idx),
             n_times = as.integer(n_times), role = "temporal")
  if (temporal_type %in% c("rw1", "rw2")) tm$cyclic <- isTRUE(cyclic)

  fit <- tulpa_nested_laplace(y, n_trials, X, prior = list(sp, tm),
                              re_idx = re_idx, n_re_groups = n_re_groups,
                              sigma_re = sigma_re, family = family, phi = phi,
                              hyperprior = hyperprior, control = control)

  fx <- .st_field_means(fit, spatial_type, sp, N)
  fit$spatial_effects  <- fx$spatial
  fit$temporal_effects <- fx$temporal
  fit$temporal <- structure(
    list(type = temporal_type, n_times = as.integer(n_times), n_groups = 1L,
         time_levels = seq_len(n_times), cyclic = isTRUE(cyclic)),
    class = c("tulpa_temporal", "list"))
  fit$spatial_type  <- spatial_type
  fit$temporal_type <- temporal_type
  class(fit) <- c("tulpa_st_nested", class(fit))
  fit
}
