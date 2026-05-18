#' User-defined GMRF latent block, compiled C++ backend
#'
#' @description
#' `tgmrf_cpp()` is the compiled-C++ analogue of [tgmrf()]. Where `tgmrf()`
#' takes R closures `Q(theta)` / `prior(theta)`, `tgmrf_cpp()` takes a
#' user-written `.cpp` file that defines the same kernels as templated C++
#' (so the same source compiles against every AD type tulpa uses). The
#' returned object has the same S3 class as `tgmrf()` -- `c("tgmrf",
#' "tulpa_latent_block")` -- so every downstream consumer (formula parser,
#' inference layers, S3 methods) treats the two paths identically. Only the
#' dispatch on `Q(theta)` differs: registry-stored function pointer
#' (`tgmrf_cpp()`) vs Rcpp::Function callback (`tgmrf()`).
#'
#' The user `.cpp` file must include `<tulpa/tgmrf.h>` and call the
#' `TULPA_REGISTER_TGMRF(id, Q_fn, mu_fn, log_prior_fn)` macro once. See
#' `inst/examples/tgmrf_periodic_ar1.cpp` for a worked example.
#'
#' @param cpp_file Absolute path to the user's `.cpp` file. The file is
#'   compiled via [Rcpp::sourceCpp()] with caching keyed on
#'   `digest::sha256(file_contents) + TULPA_ABI_VERSION` so repeated calls
#'   with an unchanged source skip the rebuild.
#' @param id Character: the stable id used as the registry key. Must match
#'   the first argument of `TULPA_REGISTER_TGMRF` in the user's `.cpp`.
#' @param init,mu,graph,bounds,obs_idx,name Same arguments as [tgmrf()].
#'   `init` is required and supplies the canonical theta names. `mu` is
#'   currently ignored on the C++ path -- the mu kernel registered by
#'   `TULPA_REGISTER_TGMRF` is the source of truth; pass the argument here
#'   only to keep call sites symmetric with `tgmrf()`.
#' @param cache_dir Directory for Rcpp's `sourceCpp` cache. Defaults to a
#'   per-user location under `tools::R_user_dir("tulpa", "cache")`.
#' @param rebuild Force a recompile even if the cached DLL is up to date.
#'   Default `FALSE`.
#'
#' @return An object of class `c("tgmrf", "tulpa_latent_block")` with the
#'   same fields as [tgmrf()] plus `backend = "cpp"` and `cpp_id = id`. The
#'   `Q` / `prior` fields are R wrapper closures around the registered C++
#'   kernels, so callers (e.g. NUTS-over-theta) that hold the object can
#'   still call `block$Q(theta)` directly.
#'
#' @seealso [tgmrf()] for the R-closure path; [latent()] for the formula
#'   slot that consumes a `tgmrf` block.
#' @export
tgmrf_cpp <- function(cpp_file, id, init,
                      mu        = NULL,
                      graph     = NULL,
                      bounds    = NULL,
                      obs_idx   = NULL,
                      name      = NULL,
                      cache_dir = tulpa_cache_dir(),
                      rebuild   = FALSE) {

  if (!is.character(cpp_file) || length(cpp_file) != 1L || is.na(cpp_file)) {
    stop("`cpp_file` must be a single non-NA character path.", call. = FALSE)
  }
  if (!file.exists(cpp_file)) {
    stop("`cpp_file` does not exist: ", cpp_file, call. = FALSE)
  }
  if (!is.character(id) || length(id) != 1L || is.na(id) || !nzchar(id)) {
    stop("`id` must be a single non-empty character string ",
         "matching the first argument of TULPA_REGISTER_TGMRF.", call. = FALSE)
  }
  if (!is.numeric(init) || length(init) < 1L || anyNA(init)) {
    stop("`init` must be a finite numeric vector of length >= 1.", call. = FALSE)
  }
  theta_dim   <- length(init)
  theta_names <- if (!is.null(names(init))) names(init) else paste0("theta_", seq_len(theta_dim))
  names(init) <- theta_names

  # ---- 1. Compile (cached) ---------------------------------------------------
  # Cache is keyed on (sha256 of file contents, ABI version). Two sources
  # with identical bytes but different ABI versions of tulpa rebuild; an
  # unchanged source against an unchanged ABI hits the Rcpp cache.
  if (!dir.exists(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  }
  if (isTRUE(rebuild)) {
    # Force a rebuild by clearing Rcpp's cache for this exact file. Cheapest
    # robust trigger: remove any stale .o / .so / .dll matching the file's
    # basename under the cache directory.
    base <- tools::file_path_sans_ext(basename(cpp_file))
    stale <- list.files(cache_dir, pattern = paste0("^", base),
                        full.names = TRUE)
    if (length(stale)) unlink(stale, recursive = TRUE, force = TRUE)
  }

  # Make sure the user's `#include <tulpa/tgmrf.h>` resolves. Under an
  # installed package, Rcpp::depends(tulpa) picks up
  # <install>/include/tulpa/ automatically. Under devtools::load_all() the
  # headers live at <source>/inst/include/tulpa/ but Rcpp's LinkingTo
  # resolver looks at <source>/include/ (which does not exist). Prepend
  # the right path via PKG_CPPFLAGS so both paths work.
  inst_inc <- system.file("include", package = "tulpa")
  if (!nzchar(inst_inc) || !dir.exists(inst_inc)) {
    stop("Cannot locate tulpa's inst/include directory.", call. = FALSE)
  }
  prev_flags <- Sys.getenv("PKG_CPPFLAGS", unset = NA)
  add_flag <- paste0('-I"', inst_inc, '"')
  new_flags <- if (is.na(prev_flags)) add_flag else paste(add_flag, prev_flags)
  Sys.setenv(PKG_CPPFLAGS = new_flags)
  on.exit({
    if (is.na(prev_flags)) Sys.unsetenv("PKG_CPPFLAGS")
    else Sys.setenv(PKG_CPPFLAGS = prev_flags)
  }, add = TRUE)

  # sourceCpp loads the user DLL; its static initializer fires
  # `tulpa_register_tgmrf(id, &spec)` via R_GetCCallable. After this point
  # the registry holds the spec under `id`. Rcpp's dependency-resolver
  # tries a few paths relative to the .cpp file's directory before falling
  # back to system.file("include", package = "tulpa") -- the first probes
  # fail with a normalizePath warning when the .cpp lives outside a source
  # tree (e.g. under inst/examples of the installed package). Suppress
  # that warning so the user sees a clean compile log.
  withCallingHandlers(
    Rcpp::sourceCpp(file = cpp_file, cacheDir = cache_dir,
                    rebuild = isTRUE(rebuild)),
    warning = function(w) {
      if (grepl("inst.?/include", conditionMessage(w))) {
        invokeRestart("muffleWarning")
      }
    }
  )

  # ---- 2. Verify registration ------------------------------------------------
  if (!.tulpa_tgmrf_registry_has(id)) {
    stop("The compiled DLL did not register a tgmrf spec under id = '", id,
         "'. Check that `cpp_file` contains exactly one ",
         "`TULPA_REGISTER_TGMRF(\"", id, "\", ...)` call.", call. = FALSE)
  }

  # ---- 3. One-shot Q(init) evaluation to capture n_latent + sparsity --------
  eval0 <- tryCatch(
    .tulpa_tgmrf_eval(id, as.numeric(init)),
    error = function(e) {
      stop("Registered Q(init) raised an error: ", conditionMessage(e),
           call. = FALSE)
    }
  )
  n_latent <- as.integer(eval0$n)
  if (n_latent < 1L) {
    stop("Registered Q(init) must have at least one row/column.", call. = FALSE)
  }
  # Reconstruct as dgCMatrix to capture the sparsity pattern in the same
  # form as tgmrf() does for the R path.
  Qc <- Matrix::sparseMatrix(
    i = eval0$i, p = eval0$p, x = eval0$x,
    dims = c(n_latent, n_latent), index1 = FALSE
  )
  Qc <- methods::as(methods::as(Qc, "generalMatrix"), "CsparseMatrix")
  max_abs <- if (length(Qc@x) > 0L) max(abs(Qc@x)) else 0
  sym_err <- max(abs(Qc - Matrix::t(Qc)))
  if (is.finite(sym_err) && sym_err > 1e-8 * max(1, max_abs)) {
    stop("Registered Q(init) must be symmetric; max |Q - t(Q)| = ",
         format(sym_err), ".", call. = FALSE)
  }
  pattern <- Qc
  pattern@x <- rep(1, length(pattern@x))

  # Validate graph hint against captured pattern (same logic as tgmrf()).
  if (!is.null(graph)) {
    if (!methods::is(graph, "sparseMatrix")) {
      stop("`graph` must be NULL or a Matrix::sparseMatrix.", call. = FALSE)
    }
    if (nrow(graph) != n_latent || ncol(graph) != n_latent) {
      stop("`graph` must be ", n_latent, " x ", n_latent, "; got ",
           nrow(graph), " x ", ncol(graph), ".", call. = FALSE)
    }
    gC <- methods::as(graph, "CsparseMatrix")
    extra <- Matrix::which((pattern != 0) & (gC == 0), arr.ind = TRUE)
    if (nrow(extra) > 0) {
      stop("Registered Q(init) has ", nrow(extra),
           " nonzero entries outside the supplied `graph` pattern.",
           call. = FALSE)
    }
  }

  # log p(init) sanity check.
  if (!is.numeric(eval0$log_prior) || !is.finite(eval0$log_prior)) {
    stop("Registered log_prior(init) must return a finite numeric scalar; got ",
         deparse(eval0$log_prior), ".", call. = FALSE)
  }

  # Validate obs_idx, if supplied.
  if (!is.null(obs_idx)) {
    if (!is.numeric(obs_idx) || anyNA(obs_idx)) {
      stop("`obs_idx` must be a finite integer vector.", call. = FALSE)
    }
    obs_idx <- as.integer(obs_idx)
    if (any(obs_idx < 1L) || any(obs_idx > n_latent)) {
      stop("`obs_idx` entries must be in [1, n_latent = ", n_latent, "].",
           call. = FALSE)
    }
  }

  # Validate bounds, if supplied (same shape contract as tgmrf()).
  if (!is.null(bounds)) {
    if (!is.list(bounds) || !all(c("lower", "upper") %in% names(bounds))) {
      stop("`bounds` must be a list with components `lower` and `upper`.",
           call. = FALSE)
    }
    if (length(bounds$lower) != theta_dim || length(bounds$upper) != theta_dim) {
      stop("`bounds$lower` and `bounds$upper` must each have length ",
           theta_dim, ".", call. = FALSE)
    }
    if (any(bounds$lower >= bounds$upper)) {
      stop("`bounds$lower` must be strictly below `bounds$upper` for every theta.",
           call. = FALSE)
    }
  }

  # ---- 4. Build R-callable wrappers around the registered kernels -----------
  # These are not used during the inner Newton loop (the multi-block driver
  # talks to the C++ kernel directly via the registry), but they let
  # NUTS-over-theta / IMH / VI compute Q(theta') at arbitrary proposals
  # without round-tripping through R closures.
  Q_wrapper <- function(theta) {
    th <- as.numeric(theta)
    res <- .tulpa_tgmrf_eval(id, th)
    Matrix::sparseMatrix(
      i = res$i, p = res$p, x = res$x,
      dims = c(n_latent, n_latent), index1 = FALSE
    )
  }
  prior_wrapper <- function(theta) {
    res <- .tulpa_tgmrf_eval(id, as.numeric(theta))
    as.numeric(res$log_prior)
  }
  mu_wrapper <- function(theta) {
    .tulpa_tgmrf_eval_mu(id, as.numeric(theta))
  }
  # Only attach mu_wrapper if the spec actually carries a mu kernel.
  has_mu <- !is.null(mu_wrapper(init))
  mu_field <- if (has_mu) mu_wrapper else NULL

  structure(
    list(
      type        = "tgmrf",
      Q           = Q_wrapper,
      prior       = prior_wrapper,
      mu          = mu_field,
      init        = init,
      theta_names = theta_names,
      theta_dim   = as.integer(theta_dim),
      n_latent    = as.integer(n_latent),
      pattern     = pattern,
      graph       = graph,
      bounds      = bounds,
      obs_idx     = obs_idx,
      name        = name,
      backend     = "cpp",
      cpp_id      = id,
      cpp_file    = normalizePath(cpp_file, mustWork = FALSE)
    ),
    class = c("tgmrf", "tulpa_latent_block")
  )
}

#' Default cache directory for `tgmrf_cpp()`-compiled DLLs
#'
#' @description
#' Returns a per-user cache directory under
#' `tools::R_user_dir("tulpa", "cache")` and creates it if missing.
#'
#' @return Absolute path (character) to the cache directory.
#' @export
tulpa_cache_dir <- function() {
  d <- tools::R_user_dir("tulpa", "cache")
  if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

# -- Internal accessors over the Rcpp-exported registry helpers ----------------
# Wrapped so .R files outside this one do not depend on the bare cpp_*
# symbol names (which would otherwise need to live in utils::globalVariables
# or be exported via @importFrom).
.tulpa_tgmrf_registry_has <- function(id) {
  cpp_tgmrf_registry_has(id)
}

.tulpa_tgmrf_eval <- function(id, theta) {
  cpp_tgmrf_eval(id, as.numeric(theta))
}

.tulpa_tgmrf_eval_mu <- function(id, theta) {
  cpp_tgmrf_eval_mu(id, as.numeric(theta))
}
