#' Inference Mode System
#'
#' @description
#' tulpa uses an explicit tier system for inference that encodes
#' **epistemic guarantees**, not just runtime characteristics.
#'
#' This design makes the difference between inference methods **first-class
#' and unavoidable**, rather than hiding them as implementation details.
#'
#' @section The Three Tiers:
#'
#' **Tier 1 - Exact:**
#' Asymptotically correct posterior inference (up to Monte Carlo error).
#' Credible intervals are interpretable as posterior uncertainty.
#' This is the reference standard.
#'
#' **Tier 2 - Structured:**
#' Accurate inference *conditional on explicit structural assumptions*.
#' Typically requires latent Gaussian structure, conditional independence,
#' smooth posteriors. Very fast for the right model class, but can be wrong
#' outside that class. Failure modes are predictable and explainable.
#'
#' **Tier 3 - Optimized:**
#' No general correctness guarantee beyond empirical usefulness.
#' Point estimates usually good, but uncertainty often underestimated.
#' Tails and correlations unreliable. Failure is usually silent.
#' This is optimization, not sampling.
#'
#' @section Auto Mode:
#'
#' `mode = "auto"` chooses between Tier 1 and Tier 2 only.
#' It will **never** silently choose Tier 3 (Optimized).
#'
#' The contract: "Use the most reliable method that is expected to finish
#' for this model."
#'
#' Auto decisions are deterministic, explainable, and overrideable.
#'
#' @section Implementation Rules:
#'
#' 1. Modes change semantics, not just runtime.
#'    Intervals from Optimized do not mean the same as Exact.
#'
#' 2. The mode must always be visible in output.
#'
#' 3. No silent upgrading or downgrading.
#'    If Exact fails, we error - we do not switch to Structured.
#'
#' 4. Backends slot into tiers.
#'    Adding a backend never introduces a new epistemic promise.
#'
#' @name inference_modes
#' @keywords internal
NULL


# ==============================================================================
# Tier metadata + backend registry (single source of truth)
# ==============================================================================

#' Tier metadata: the epistemic guarantee attached to each tier.
#'
#' Tier *membership* (which backends live in a tier) is not stored here -- it
#' is derived from `BACKEND_REGISTRY`. This table holds only the per-tier
#' description/guarantee so the two never drift.
#'
#' @keywords internal
TIER_META <- list(
  exact = list(
    tier = 1L,
    name = "Exact",
    description = "Asymptotically correct posterior inference",
    guarantee = "Credible intervals interpretable as posterior uncertainty",
    note = "Reference standard"
  ),
  structured = list(
    tier = 2L,
    name = "Structured",
    description = "Accurate inference conditional on structural assumptions",
    guarantee = "Correct if model meets structural assumptions",
    note = "Controlled approximation, not heuristics"
  ),
  optimized = list(
    tier = 3L,
    name = "Optimized",
    description = "Optimization-based approximate inference",
    guarantee = "No general correctness guarantee",
    note = "Uncertainty often underestimated; requires explicit opt-in"
  )
)


#' Backend registry -- single source of truth for the inference backends.
#'
#' @description
#' One entry per backend. Adding a backend is a single entry here; tier
#' membership, family support, the input contract, and R-level reachability
#' all derive from this list. Replaces the previously scattered per-tier
#' `backends` vectors and the separate `BACKEND_FAMILY_SUPPORT` table.
#'
#' Fields:
#' * `tier`     -- tier key (`"exact"`, `"structured"`, `"optimized"`).
#' * `input`    -- the input contract the backend's fitter consumes:
#'     - `"design"`    : design-matrix bundle (`y`, `n_trials`, `X`, RE structure).
#'     - `"logpost"`   : a `log_posterior(theta)` closure plus dimension/init.
#'     - `"modeldata"` : a tulpa `ModelData` / `LikelihoodSpec` (C-ABI NUTS path).
#'     - `"nested"`    : a design-matrix bundle plus one or more latent prior
#'                       blocks (`latent(tgmrf(...))`), integrated over the block
#'                       hyperparameters by the nested-Laplace driver.
#' * `fitter`   -- name of the R function implementing the backend, or `NULL`
#'     when only a C++ kernel exists with no R entry point yet. Stored as a
#'     *string* (not the function object) so the registry is independent of
#'     source-load order; resolved lazily via [resolve_backend_fitter()].
#'     `NULL` => not selectable from R; dispatch fails loudly.
#' * `families` -- character vector of supported family identifiers, or `NULL`
#'     for an unrestricted backend.
#' * `cabi`     -- the registered C-ABI callable backing the backend (the symbol
#'     a model package reaches via `LinkingTo: tulpa`, and the one an R wrapper
#'     would call), or `NULL`.
#' * `note`     -- optional human-readable note.
#'
#' Family identity (for `families`) is checked against `family$name`,
#' `family$distribution`, and `family$numerator$distribution` (numdenom-style
#' ratio families nest a per-process distribution).
#'
#' @keywords internal
BACKEND_REGISTRY <- list(
  # ---- Tier 1: Exact ----
  hmc = list(
    tier = "exact", input = "modeldata", fitter = NULL, families = NULL,
    cabi = "tulpa_run_nuts_generic",
    note = "Model packages drive NUTS through the C ABI; no generic R fitter yet"
  ),
  ess = list(
    tier = "exact", input = "logpost", fitter = NULL, families = NULL,
    cabi = "tulpa_run_ess_sampler"
  ),
  pg = list(
    tier = "exact", input = "design", fitter = NULL,
    families = c("binomial", "beta_binomial", "beta_binomial_fixed",
                 "negbin_negbin", "neg_binomial_2", "negative_binomial"),
    cabi = "tulpa_pg_binomial_gibbs"
  ),
  gibbs = list(
    tier = "exact", input = "design", fitter = "tulpa_gibbs",
    families = c("poisson_gamma", "negbin_negbin", "binomial",
                 "negbin_gamma", "gamma_gamma", "lognormal",
                 "beta_binomial", "lognormal_lognormal",
                 "beta_binomial_fixed",
                 "poisson", "neg_binomial_2", "negative_binomial"),
    cabi = "tulpa_cpp_gibbs_spatial"
  ),
  re_cov_gibbs = list(
    tier = "exact", input = "design", fitter = "tulpa_re_cov_gibbs",
    families = NULL, cabi = NULL,
    note = paste("Correlated random-slope term (1 + x | g): exact",
                 "Metropolis-within-Gibbs debias of the RE covariance Sigma.",
                 "Auto-selected from the Laplace path with control$re_cov = 'gibbs'")
  ),
  sghmc = list(
    tier = "exact", input = "logpost", fitter = NULL, families = NULL,
    cabi = "tulpa_sghmc_fit"
  ),
  sgld = list(
    tier = "exact", input = "logpost", fitter = NULL, families = NULL,
    cabi = "tulpa_sgld_fit"
  ),
  mclmc = list(
    tier = "exact", input = "logpost", fitter = NULL, families = NULL,
    cabi = "tulpa_mclmc_fit",
    note = "Microcanonical Langevin Monte Carlo; C-ABI kernel only"
  ),
  smc = list(
    tier = "exact", input = "logpost", fitter = NULL, families = NULL,
    cabi = "tulpa_smc_fit",
    note = "Sequential Monte Carlo; C-ABI kernel only"
  ),
  imh_laplace = list(
    tier = "exact", input = "logpost", fitter = "imh_laplace",
    families = NULL, cabi = NULL
  ),
  mala = list(
    tier = "exact", input = "logpost", fitter = "mala",
    families = NULL, cabi = NULL
  ),
  # ---- Tier 2: Structured ----
  laplace = list(
    tier = "structured", input = "design", fitter = "tulpa_laplace",
    families = NULL, cabi = "tulpa_laplace_mode_dense_multi_re"
  ),
  re_cov_nested = list(
    tier = "structured", input = "design", fitter = "tulpa_re_cov_nested",
    families = NULL, cabi = NULL,
    note = paste("Correlated random-slope term (1 + x | g): nested-Laplace",
                 "integration over the RE covariance Sigma (CCD design + PC/LKJ",
                 "prior). Auto-selected from the Laplace path for a single",
                 "correlated term")
  ),
  pathfinder = list(
    tier = "structured", input = "logpost", fitter = "pathfinder",
    families = NULL, cabi = NULL
  ),
  agq = list(
    tier = "structured", input = "design", fitter = "agq_fit",
    families = NULL, cabi = NULL
  ),
  nested_laplace = list(
    tier = "structured", input = "nested", fitter = "tulpa_nested_laplace",
    families = NULL, cabi = "cpp_nested_laplace_multi",
    note = "Single-arm nested Laplace; integrates latent-block hyperparameters"
  ),
  nested_laplace_joint = list(
    tier = "structured", input = "nested", fitter = "tulpa_nested_laplace_joint",
    families = NULL, cabi = "cpp_nested_laplace_joint_multi",
    note = paste("Joint multi-arm nested Laplace; driven by model packages, not the",
                 "single-response tulpa() formula (cannot express multiple arms)")
  ),
  # ---- Tier 3: Optimized ----
  vi = list(
    tier = "optimized", input = "logpost", fitter = NULL, families = NULL,
    cabi = "tulpa_fit_vi",
    note = "Generic VI via C ABI; tgmrf VI exposed as tulpa_tgmrf_vi"
  )
)


#' Backend names belonging to a tier (derived from the registry).
#' @keywords internal
.tier_backends <- function(tier_key) {
  names(Filter(function(e) e$tier == tier_key, BACKEND_REGISTRY))
}


#' Inference tiers with their properties (derived from `TIER_META` +
#' `BACKEND_REGISTRY`). Preserves the `INFERENCE_TIERS$<tier>$backends`
#' interface relied on by downstream code and tests.
#' @keywords internal
INFERENCE_TIERS <- local({
  out <- vector("list", length(TIER_META))
  names(out) <- names(TIER_META)
  for (tk in names(TIER_META)) {
    out[[tk]] <- c(TIER_META[[tk]], list(backends = .tier_backends(tk)))
  }
  out
})


#' Backend -> supported families, derived from the registry.
#' @keywords internal
BACKEND_FAMILY_SUPPORT <- local({
  out <- list()
  for (b in names(BACKEND_REGISTRY)) {
    fam <- BACKEND_REGISTRY[[b]]$families
    if (!is.null(fam)) out[[b]] <- fam
  }
  out
})


#' All registered backends across every tier (derived from the registry).
#' @keywords internal
ALL_BACKENDS <- names(BACKEND_REGISTRY)


#' Test whether a backend supports the given family.
#' @keywords internal
backend_supports_family <- function(backend, family) {
  supported <- BACKEND_REGISTRY[[backend]]$families
  if (is.null(supported)) return(TRUE)  # unrestricted backend

  fam_name <- family$name %||% ""
  fam_dist <- family$distribution %||% (family$numerator$distribution %||% "")
  fam_name %in% supported || fam_dist %in% supported || fam_dist == "binomial"
}


#' Get tier for a backend
#' @param backend Character string naming the backend
#' @return List with tier information
#' @keywords internal
get_backend_tier <- function(backend) {
  entry <- BACKEND_REGISTRY[[backend]]
  if (is.null(entry)) {
    stop(sprintf("Unknown backend: '%s'", backend), call. = FALSE)
  }
  meta <- TIER_META[[entry$tier]]
  list(
    tier = meta$tier,
    name = meta$name,
    mode = entry$tier,
    description = meta$description,
    guarantee = meta$guarantee,
    note = meta$note
  )
}


# ==============================================================================
# Reachability + dispatch spine
# ==============================================================================

#' Is a backend reachable from R (does it have an R-level fitter)?
#' @keywords internal
backend_is_reachable <- function(backend) {
  entry <- BACKEND_REGISTRY[[backend]]
  !is.null(entry) && !is.null(entry$fitter)
}


#' Error if a selected backend has no R-level fitter.
#'
#' @description
#' Enforces the registry honesty contract: a backend may ship a C++ kernel
#' reachable from model packages (via `LinkingTo: tulpa`) yet have no R entry
#' point. Such a backend must never be *silently* selectable from R --
#' selecting it errors with a precise message naming the C-ABI symbol, rather
#' than pretending to dispatch.
#'
#' @keywords internal
assert_backend_reachable <- function(backend) {
  entry <- BACKEND_REGISTRY[[backend]]
  if (is.null(entry)) {
    stop(sprintf("Unknown backend: '%s'", backend), call. = FALSE)
  }
  if (is.null(entry$fitter)) {
    meta <- TIER_META[[entry$tier]]
    cabi <- entry$cabi %||% "(none)"
    stop(sprintf(paste0(
      "Backend '%s' is registered (Tier %d, %s) but has no R-level fitter.\n",
      "Its C++ kernel is reachable from model packages through the C ABI\n",
      "(%s), but it cannot be driven directly from R yet. Choose a reachable\n",
      "backend, or wire an R wrapper over the kernel."),
      backend, meta$tier, meta$name, cabi),
      call. = FALSE)
  }
  invisible(TRUE)
}


#' Resolve the R fitter function for a backend (errors if unreachable).
#'
#' Looks up the fitter *name* string in the registry and resolves it lazily,
#' so the registry stays independent of source-file load order.
#' @keywords internal
resolve_backend_fitter <- function(backend) {
  assert_backend_reachable(backend)
  match.fun(BACKEND_REGISTRY[[backend]]$fitter)
}


#' Dispatch a fit to the backend chosen by the mode/tier system.
#'
#' @description
#' The routing spine. Resolves `mode` to a concrete backend via
#' [select_inference_mode()], asserts the backend is R-reachable (fails loudly
#' otherwise), then calls its fitter with `fitter_args`. The caller supplies
#' `fitter_args` matching the backend's input contract
#' (`BACKEND_REGISTRY$<backend>$input`).
#'
#' The selected mode/tier/backend are stamped onto the returned fit (without
#' overwriting any the fitter already set), so the inference contract is always
#' visible in the output.
#'
#' @param mode User-specified mode (`"auto"`, a tier, or a backend name).
#' @param fitter_args Named list of arguments forwarded to the backend fitter.
#' @param family,n_obs,has_spatial,has_temporal,has_latent,spatial_type,temporal
#'   Model characteristics forwarded to [select_inference_mode()].
#' @return The fitter's result, with `inference_mode`, `inference_tier`,
#'   `backend`, and `selection_reason` ensured.
#' @keywords internal
tulpa_dispatch <- function(mode,
                           fitter_args = list(),
                           family = NULL,
                           n_obs = NULL,
                           has_spatial = FALSE,
                           has_temporal = FALSE,
                           has_latent = FALSE,
                           spatial_type = NULL,
                           temporal = NULL) {
  sel <- select_inference_mode(
    mode,
    family = family, n_obs = n_obs,
    has_spatial = has_spatial, has_temporal = has_temporal,
    has_latent = has_latent, spatial_type = spatial_type, temporal = temporal
  )

  fitter <- resolve_backend_fitter(sel$backend)
  fit <- do.call(fitter, fitter_args)

  if (is.list(fit)) {
    fit$inference_mode <- fit$inference_mode %||% sel$mode
    fit$inference_tier <- fit$inference_tier %||% sel$tier
    fit$backend <- fit$backend %||% sel$backend
    fit$selection_reason <- fit$selection_reason %||% sel$reason
    # Guarantee a consistent return type: lower-level fitters (e.g.
    # tulpa_laplace) return a bare result list for model packages to wrap;
    # through dispatch the inference contract -- including the class -- must be
    # visible, so tag any unclassed result as a tulpa_fit.
    if (!inherits(fit, "tulpa_fit")) {
      class(fit) <- c(oldClass(fit), "tulpa_fit")
    }
  }
  fit
}


#' Map mode to valid backends
#' @param mode Character: "auto", "exact", "structured", or "optimized"
#' @return Character vector of valid backends for this mode
#' @keywords internal
get_mode_backends <- function(mode) {
  mode <- tolower(mode)

  if (mode == "auto") {
    # Auto can choose Tier 1 or Tier 2, never Tier 3
    return(c(INFERENCE_TIERS$exact$backends, INFERENCE_TIERS$structured$backends))
  }

  if (mode %in% names(INFERENCE_TIERS)) {
    return(INFERENCE_TIERS[[mode]]$backends)
  }

  stop(sprintf(
    "Unknown mode: '%s'. Use one of: auto, exact, structured, optimized",
    mode
  ), call. = FALSE)
}


# ==============================================================================
# Mode Selection Logic
# ==============================================================================

#' Select inference mode and backend
#'
#' @description
#' Implements the mode selection logic for tulpa. Accepts either tier names
#' (auto, exact, structured, optimized) or backend names (hmc, ess, pg, laplace, vi).
#'
#' When mode is "auto", selects between Tier 1 (Exact) and Tier 2 (Structured)
#' based on model characteristics. Never selects Tier 3 (Optimized) automatically.
#'
#' @param mode User-specified mode or backend name
#' @param family Model family object
#' @param n_obs Number of observations
#' @param has_spatial Whether model has spatial effects
#' @param has_temporal Whether model has temporal effects
#' @param has_latent Whether model has latent factors
#'
#' @return List with:
#'   - mode: The selected mode name
#'   - backend: The selected backend
#'   - tier: The tier number (1, 2, or 3)
#'   - tier_name: The tier name
#'   - reason: Explanation for the selection
#'
#' @keywords internal
select_inference_mode <- function(mode,
                                  family,
                                  n_obs,
                                  has_spatial = FALSE,
                                  has_temporal = FALSE,
                                  has_latent = FALSE,
                                  spatial_type = NULL,
                                  temporal = NULL) {

  mode <- tolower(mode)

  # Check if mode is actually a backend name (derived from INFERENCE_TIERS)
  if (mode %in% ALL_BACKENDS) {
    # User specified a backend directly
    backend <- mode
    tier_info <- get_backend_tier(backend)

    return(list(
      mode = tier_info$mode,
      backend = backend,
      tier = tier_info$tier,
      tier_name = tier_info$name,
      reason = sprintf("User-specified backend: %s", backend)
    ))
  }

  # Validate tier name
  valid_modes <- c("auto", names(INFERENCE_TIERS))
  if (!mode %in% valid_modes) {
    stop(sprintf(
      "Invalid mode: '%s'. Use one of: %s, or a backend: %s",
      mode,
      paste(valid_modes, collapse = ", "),
      paste(ALL_BACKENDS, collapse = ", ")
    ), call. = FALSE)
  }

  # Auto-selection logic
  if (mode == "auto") {
    return(auto_select_mode(family, n_obs, has_spatial, has_temporal, has_latent, temporal,
                             spatial_type))
  }

  # Explicit tier mode: select best backend within that tier
  backend <- select_backend_for_mode(mode, family, n_obs, has_spatial, has_temporal,
                                     has_latent)
  tier_info <- get_backend_tier(backend)

  return(list(
    mode = mode,
    backend = backend,
    tier = tier_info$tier,
    tier_name = tier_info$name,
    reason = sprintf("User-specified mode: %s", mode)
  ))
}


#' Auto-select mode (Tier 1 or Tier 2 only)
#'
#' @description
#' Implements the "auto" mode selection. Chooses the most reliable method
#' that is expected to finish for the given model.
#'
#' **Critical rule**: Auto never selects Tier 3 (Optimized).
#'
#' @keywords internal
auto_select_mode <- function(family, n_obs, has_spatial, has_temporal, has_latent, temporal = NULL,
                             spatial_type = NULL) {

  # Latent prior blocks (`latent(tgmrf(...))`) integrate their hyperparameters
  # via nested Laplace -- the designed Tier 2 hot path for latent Gaussian
  # structure. This takes precedence over the dataset-size heuristics below:
  # a latent-block model goes nested regardless of n.
  if (has_latent) {
    return(list(
      mode = "structured",
      backend = "nested_laplace",
      tier = 2L,
      tier_name = "Structured",
      reason = "latent prior block(s); nested-Laplace hyperparameter integration"
    ))
  }

  # Thresholds
  VERY_LARGE <- 50000
  LARGE <- 10000

  # Decision logic
  reason_parts <- character(0)

  # Check if Laplace (Tier 2) is appropriate
  use_structured <- FALSE

  if (n_obs > VERY_LARGE) {
    use_structured <- TRUE
    reason_parts <- c(reason_parts, sprintf("large dataset (n=%s)", format(n_obs, big.mark = ",")))
  }

  # Latent Gaussian models are ideal for Laplace
  if (has_spatial && !has_latent && n_obs > LARGE) {
    use_structured <- TRUE
    reason_parts <- c(reason_parts, "spatial model with large n")
  }

  if (use_structured) {
    return(list(
      mode = "structured",
      backend = "laplace",
      tier = 2L,
      tier_name = "Structured",
      reason = paste(reason_parts, collapse = "; ")
    ))
  }

  # Gibbs for ICAR spatial-only models (no temporal, no latent)
  # Gibbs is ~18x faster than HMC for ICAR because it updates phi
  # component-wise, avoiding HMC's curse of dimensionality
  is_gibbs_spatial <- has_spatial && !is.null(spatial_type) &&
                      spatial_type %in% c("icar", "car", "bym2", "hsgp")
  # Gibbs supports: no temporal, TVC, or RW1/AR1 GMRF temporal
  has_tvc_only <- !is.null(temporal) && inherits(temporal, "tulpa_tvc")
  has_gmrf_temporal <- !is.null(temporal) && inherits(temporal, "tulpa_temporal") &&
                       !is.null(temporal$type) && temporal$type %in% c("rw1", "ar1")
  gibbs_temporal_ok <- !has_temporal || has_tvc_only || has_gmrf_temporal
  if (is_gibbs_spatial && gibbs_temporal_ok && !has_latent) {
    if (backend_supports_family("gibbs", family)) {
      return(list(
        mode = "exact",
        backend = "gibbs",
        tier = 1L,
        tier_name = "Exact",
        reason = sprintf("ICAR/BYM2 spatial model (Gibbs %s)", spatial_type)
      ))
    }
  }

  # Default: Tier 1 (Exact) with HMC
  # HMC is the most general and robust
  backend <- "hmc"
  reason <- "default (full MCMC)"

  return(list(
    mode = "exact",
    backend = backend,
    tier = 1L,
    tier_name = "Exact",
    reason = reason
  ))
}


#' Select best backend within a mode
#' @keywords internal
select_backend_for_mode <- function(mode, family, n_obs, has_spatial, has_temporal,
                                    has_latent = FALSE) {

  if (mode == "exact") {
    # HMC is the default for Exact - most general
    return("hmc")
  }

  if (mode == "structured") {
    # Latent prior blocks need the nested-Laplace path; plain (conditional)
    # Laplace does not consume them. Otherwise Laplace is the Tier 2 default.
    if (has_latent) return("nested_laplace")
    return("laplace")
  }

  if (mode == "optimized") {
    # VI is currently the only Tier 3 backend
    return("vi")
  }

  # Fallback
  return("hmc")
}


# ==============================================================================
# Mode Validation
# ==============================================================================

#' Validate that a fit used the expected mode
#'
#' @description
#' Ensures the fit object was created with the expected inference mode.
#' Useful for enforcing mode requirements in downstream analysis.
#'
#' @param fit A tulpa_fit object
#' @param expected_mode Mode that should have been used
#' @param error If TRUE (default), error on mismatch. If FALSE, return logical.
#'
#' @return If error = FALSE, returns TRUE/FALSE. Otherwise errors on mismatch.
#'
#' @export
validate_mode <- function(fit, expected_mode, error = TRUE) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  actual_mode <- fit$inference_mode %||% "unknown"
  actual_tier <- fit$inference_tier %||% NA

  expected_mode <- tolower(expected_mode)

  # Check if modes match
  match <- (actual_mode == expected_mode)

  # Also accept tier-based matching
 if (!match && expected_mode %in% names(INFERENCE_TIERS)) {
    expected_tier <- INFERENCE_TIERS[[expected_mode]]$tier
    match <- !is.na(actual_tier) && (actual_tier == expected_tier)
  }

  if (!match && error) {
    stop(sprintf(
      "Mode mismatch: fit used '%s' (Tier %s) but expected '%s'.\n%s",
      actual_mode,
      ifelse(is.na(actual_tier), "?", actual_tier),
      expected_mode,
      "Refit the model with the correct mode, or adjust your expectations."
    ), call. = FALSE)
  }

  return(match)
}


# ==============================================================================
# User-Facing Information
# ==============================================================================
#' Print inference mode information
#'
#' @description
#' Displays information about the inference modes available in tulpa,
#' including their tiers, guarantees, and appropriate use cases.
#'
#' @export
inference_mode_info <- function() {
  cat("tulpa Inference Modes\n")
  cat("========================\n\n")

  cat("Modes encode EPISTEMIC GUARANTEES, not just runtime characteristics.\n\n")

  for (tier_name in c("exact", "structured", "optimized")) {
    tier <- INFERENCE_TIERS[[tier_name]]

    cat(sprintf("TIER %d: %s\n", tier$tier, toupper(tier$name)))
    cat(sprintf("  %s\n", tier$description))
    cat(sprintf("  Guarantee: %s\n", tier$guarantee))
    cat(sprintf("  Backends: %s\n", paste(tier$backends, collapse = ", ")))
    cat(sprintf("  Note: %s\n", tier$note))
    cat("\n")
  }

  cat("AUTO MODE\n")
  cat("  Selects between Tier 1 and Tier 2 only.\n")
  cat("  NEVER silently chooses Tier 3 (Optimized).\n")
  cat("  Contract: 'Use the most reliable method expected to finish.'\n\n")

  cat("Usage (by tier):\n")
  cat("  tulpa(..., mode = 'auto')       # Tier 1 or 2 (default)\n")
  cat("  tulpa(..., mode = 'exact')      # Tier 1\n")
  cat("  tulpa(..., mode = 'structured') # Tier 2\n")
  cat("  tulpa(..., mode = 'optimized')  # Tier 3 (explicit opt-in)\n\n")

  cat("Backends ([R] = callable from R, [C-ABI] = model-package kernel only):\n")
  for (tk in c("exact", "structured", "optimized")) {
    for (b in .tier_backends(tk)) {
      entry <- BACKEND_REGISTRY[[b]]
      tag <- if (is.null(entry$fitter)) "[C-ABI]" else "[R]    "
      cat(sprintf("  %s mode = '%-11s # Tier %d (%s)\n",
                  tag, paste0(b, "'"), TIER_META[[tk]]$tier, TIER_META[[tk]]$name))
    }
  }

  invisible(NULL)
}


#' Format tier information for display
#'
#' @param tier_info List from get_backend_tier()
#' @param verbose Include full description
#' @return Character string
#' @keywords internal
format_tier_info <- function(tier_info, verbose = FALSE) {
  if (verbose) {
    sprintf(
      "Tier %d (%s): %s",
      tier_info$tier,
      tier_info$name,
      tier_info$description
    )
  } else {
    sprintf("Tier %d (%s)", tier_info$tier, tier_info$name)
  }
}


#' Format mode selection message
#'
#' @param selection List from select_inference_mode()
#' @return Character string for display
#' @keywords internal
format_mode_selection <- function(selection) {
  lines <- c(
    sprintf("Mode: %s", selection$tier_name),
    sprintf("Backend: %s", selection$backend),
    sprintf("Reason: %s", selection$reason)
  )
  paste(lines, collapse = "\n")
}
