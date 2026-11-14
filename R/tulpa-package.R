#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib tulpa, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom Matrix sparseMatrix
#' @importFrom grDevices adjustcolor colorRampPalette rgb
#' @importFrom graphics abline hist lines pairs par polygon
#' @importFrom methods as
#' @importFrom stats approx ar as.formula ave binom.test coef cor cov density dist dnorm fitted ks.test logLik lowess median model.frame model.matrix model.response na.pass nobs pnorm predict qnorm reshape residuals runif sd setNames simulate var vcov
#' @importFrom utils flush.console head
## usethis namespace: end
NULL

# `.data` is the rlang/ggplot2 pronoun used inside conditional ggplot2 plotting
# helpers; `ratio` is the downstream model-package (tulpaRatio) accessor invoked by
# prepare_map_data_from_fit when mapping a ratio fit; `.phl_w` names the weights
# column post_hoc_lm() attaches to the model frame so lm() resolves it. None is a
# tulpa symbol; register them as globals for the non-standard-evaluation sites above.
utils::globalVariables(c("cpp_gradient_mode_names", ".data", "ratio", ".phl_w", "cpp_spde_orphan_mass",
                         "cpp_device_info", "cpp_device_validate", "cpp_device_configure",
                         "cpp_device_begin", "cpp_device_end",
                         "cpp_device_snapshot", "cpp_device_probe", "cpp_device_test",
                         "cpp_device_axpy", "cpp_device_cholesky", "cpp_device_nngp",
                         "cpp_device_operations", "cpp_device_sparse", "cpp_device_sparse_state",
                         "cpp_device_family", "cpp_device_model_gradient",
                         "cpp_device_likelihood_resources",
                         "cpp_device_dense", "cpp_device_temporal_dense", "cpp_device_hsgp_dense",
                         "cpp_device_dense_adjoint", "cpp_device_projected_dense",
                         "cpp_device_hsgp_sampler",
                         "cpp_device_hsgp_cross",
                         "cpp_device_basis_moments", "cpp_device_dense_threads",
                         "cpp_family_loglik_batch",
                         "cpp_hsgp_sum_projection", "cpp_outer_tensor_interpolate",
                         "cpp_outer_read_design", "cpp_outer_tensor_partition",
                         "cpp_nl_group_logsumexp", "cpp_nl_axis_row_id",
                         "cpp_precision_density", "cpp_precision_math", "cpp_beta_precision_score",
                         "cpp_beta_density_terms", "cpp_spde_density_terms",
                         "cpp_precision_density_paths", "cpp_precision_ad_math",
                         "cpp_beta_grouped_terms", "cpp_test_field_coordinates",
                         "cpp_test_sampler_controls"))

# The package version, read from the one place that defines it. A literal here
# (or in src/) is a second definition that nothing forces anyone to update at
# release, so it can only ever agree with DESCRIPTION by coincidence.
# TULPA_ABI_VERSION is deliberately NOT this: it is a compiled constant because
# it must describe the DLL a model package linked against, not the metadata
# sitting beside it.
#' @keywords internal
tulpa_version <- function() {
  as.character(utils::packageVersion("tulpa"))
}

# Scoped set.seed: seeds the RNG and restores the caller's RNG state when the
# calling function exits, so a user-supplied `seed` does not clobber the
# session RNG stream. No-op when `seed` is NULL. The on.exit restore is
# registered in `envir` (the caller's frame), the base-R equivalent of
# withr::defer().
#' @keywords internal
# Snapshot the global RNG stream (`.Random.seed`), returning NULL when unset.
# The single source the three seed helpers below share.
.snapshot_seed <- function() {
  if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv)
  } else NULL
}

# A closure that restores the global RNG stream to `old_seed`, removing
# `.Random.seed` when it was unset at snapshot time. `old_seed` is forced here so
# the snapshot is captured now, not lazily inside the closure after the caller's
# randomized step has already advanced the stream.
.make_seed_restore <- function(old_seed) {
  force(old_seed)
  function() {
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }
}

.seed_scoped <- function(seed, envir = parent.frame()) {
  if (is.null(seed)) return(invisible(FALSE))
  restore <- .make_seed_restore(.snapshot_seed())
  do.call(on.exit, list(as.call(list(restore)), add = TRUE), envir = envir)
  set.seed(as.integer(seed))
  invisible(TRUE)
}

# Snapshot the session RNG stream and restore it when `envir` (the caller's
# frame by default) exits, so an internal randomized step (a Pareto-k
# importance batch, a probe vector) leaves `.Random.seed` exactly as the
# caller had it. The complement of `.seed_scoped`: that one also seeds.
#' @keywords internal
.preserve_seed_in_frame <- function(envir = parent.frame()) {
  restore <- .make_seed_restore(.snapshot_seed())
  do.call(on.exit, list(as.call(list(restore)), add = TRUE), envir = envir)
  invisible(NULL)
}

# Evaluate `expr` and restore the caller's RNG stream afterwards, so an
# internal randomized step (diagnostic importance draws, a probe matrix)
# leaves the session RNG untouched. Complements .seed_scoped(): that one
# seeds for the rest of the calling function, this one is RNG-neutral
# around a single expression.
#' @keywords internal
.with_preserved_seed <- function(expr) {
  restore <- .make_seed_restore(.snapshot_seed())
  on.exit(restore(), add = TRUE)
  expr
}
