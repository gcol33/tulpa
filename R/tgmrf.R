#' User-defined GMRF latent block
#'
#' @description
#' `tgmrf()` lets a user define a Gaussian Markov random field as a latent
#' block in a tulpa model by supplying two R closures: a precision-matrix
#' factory `Q(theta)` and a log-prior on theta `prior(theta)`. The resulting
#' object plugs into a formula via `latent(tgmrf(...))` and is consumed by
#' every tulpa inference tier (Laplace, EM+Laplace, VI, nested_laplace+CCD,
#' NUTS, IMH-Laplace).
#'
#' This is the latent-side dual of `LikelihoodSpec`. `LikelihoodSpec` lets a
#' model package own the observation model; `tgmrf()` lets a script own a
#' single latent block.
#'
#' @details
#' For a Gaussian latent block `z ~ N(mu(theta), Q(theta)^{-1})`:
#'
#' \deqn{\log p(z\mid\theta) = \tfrac{1}{2}\log\det Q(\theta) - \tfrac{1}{2}(z - \mu(\theta))^\top Q(\theta)(z - \mu(\theta)) + \mathrm{const}}
#'
#' the gradient `-Q(z - mu)` and Hessian `-Q` wrt `z` are closed form, so the
#' user never writes gradient code. The Laplace inner step needs only
#' `Q(theta)` and `mu(theta)` at numeric `theta`. NUTS and VI additionally
#' use a forward finite-difference gradient on theta, costing `dim(theta)`
#' extra `Q` calls per outer step — cheap for the typical `dim(theta) <= 5`.
#'
#' See `dev_notes/plans/generic-todo.md` for the full design rationale and the rollout phases.
#'
#' @param Q A function `function(theta)` returning a sparse precision matrix
#'   of class `Matrix::sparseMatrix` (typically `dgCMatrix`). The matrix must
#'   be square and symmetric. Called once at registration with `theta = init`
#'   to infer `n_latent` and capture the sparsity pattern.
#' @param prior A function `function(theta)` returning a finite numeric
#'   scalar — the log-prior density at `theta`.
#' @param init Numeric vector of starting values for `theta`. Names, if
#'   present, become the canonical theta names.
#' @param mu Optional function `function(theta)` returning a numeric vector
#'   of length `n_latent`. Default `NULL` is equivalent to a zero mean.
#' @param graph Optional sparse matrix specifying the upper bound on `Q`'s
#'   sparsity pattern. If supplied, the registration check verifies that
#'   the nonzero pattern of `Q(init)` is a subset.
#' @param bounds Optional list with components `lower` and `upper`, each a
#'   numeric vector of length `length(init)`. Used by `nested_laplace + CCD`
#'   to build the outer grid; unused by NUTS / VI.
#' @param obs_idx Optional integer vector mapping each observation to a
#'   latent slot in `[1, n_latent]`. If `NULL` (default), the fit-time
#'   driver assumes `N == n_latent` and uses `seq_len(N)` — i.e. one
#'   observation per latent slot, in row order.
#' @param name Optional character; cosmetic label used by `print()` /
#'   `summary()`.
#'
#' @return An object of class `c("tgmrf", "tulpa_latent_block")` with
#'   components:
#'   * `Q`, `prior`, `mu` — the user closures (or `NULL` for `mu`).
#'   * `init`, `theta_names`, `theta_dim` — hyperparameter metadata.
#'   * `n_latent` — inferred from `Q(init)`.
#'   * `pattern` — captured sparsity pattern as a `dgCMatrix` of 1s.
#'   * `graph` — user-supplied pattern, validated against `pattern` if given.
#'   * `bounds`, `name` — passed through.
#'
#' @examples
#' # Periodic AR(1) block, wrap-around tridiagonal precision.
#' periodic_ar1 <- function(n) {
#'   tgmrf(
#'     Q = function(theta) {
#'       sigma <- exp(theta[1]); rho <- tanh(theta[2])
#'       d <- rep((1 + rho^2) / sigma^2, n)
#'       o <- rep(-rho / sigma^2, n)
#'       M <- Matrix::bandSparse(
#'         n, k = c(-1, 0, 1), diagonals = list(o, d, o)
#'       )
#'       M[1, n] <- M[n, 1] <- -rho / sigma^2
#'       methods::as(M, "CsparseMatrix")
#'     },
#'     prior = function(theta) {
#'       stats::dnorm(theta[1], 0, 1, log = TRUE) +
#'         stats::dnorm(theta[2], 0, 1, log = TRUE)
#'     },
#'     init = c(log_sigma = 0, atanh_rho = 0),
#'     name = "periodic_ar1"
#'   )
#' }
#'
#' blk <- periodic_ar1(20)
#' print(blk)
#'
#' @seealso [latent()] for the formula slot that consumes a `tgmrf` block.
#' @export
tgmrf <- function(Q, prior, init,
                  mu      = NULL,
                  graph   = NULL,
                  bounds  = NULL,
                  obs_idx = NULL,
                  name    = NULL) {

  if (!is.function(Q))     stop("`Q` must be a function of theta.",     call. = FALSE)
  if (!is.function(prior)) stop("`prior` must be a function of theta.", call. = FALSE)
  if (!is.null(mu) && !is.function(mu)) {
    stop("`mu` must be NULL or a function of theta.", call. = FALSE)
  }

  if (!is.numeric(init) || length(init) < 1L || anyNA(init)) {
    stop("`init` must be a finite numeric vector of length >= 1.", call. = FALSE)
  }
  theta_dim   <- length(init)
  theta_names <- if (!is.null(names(init))) names(init) else paste0("theta_", seq_len(theta_dim))
  names(init) <- theta_names

  # Evaluate Q(init) once to (1) catch user errors at registration, (2) infer
  # n_latent, (3) capture the sparsity pattern.
  Q0 <- tryCatch(
    Q(init),
    error = function(e) {
      stop("`Q(init)` raised an error: ", conditionMessage(e), call. = FALSE)
    }
  )
  if (!methods::is(Q0, "sparseMatrix")) {
    stop("`Q(init)` must return a Matrix::sparseMatrix (typically dgCMatrix); ",
         "got class ", paste(class(Q0), collapse = "/"), ".", call. = FALSE)
  }
  if (nrow(Q0) != ncol(Q0)) {
    stop("`Q(init)` must be square; got ", nrow(Q0), " x ", ncol(Q0), ".",
         call. = FALSE)
  }
  n_latent <- nrow(Q0)
  if (n_latent < 1L) {
    stop("`Q(init)` must have at least one row/column.", call. = FALSE)
  }

  # Symmetry check on numeric values (Q is the precision matrix). Coerce
  # through generalMatrix first so unit-diagonal / unit-triangular shortcuts
  # (e.g. ddiMatrix(diag = "U"), dtCMatrix) materialise their structural
  # entries into the dgCMatrix slot. Tolerance loose enough to allow
  # user-supplied dsCMatrix or floating-point round-trips through
  # bandSparse / sparseMatrix.
  Qc  <- methods::as(methods::as(Q0, "generalMatrix"), "CsparseMatrix")
  max_abs <- if (length(Qc@x) > 0L) max(abs(Qc@x)) else 0
  sym_err <- max(abs(Qc - Matrix::t(Qc)))
  if (is.finite(sym_err) && sym_err > 1e-8 * max(1, max_abs)) {
    stop("`Q(init)` must be symmetric; max |Q - t(Q)| = ", format(sym_err),
         ".", call. = FALSE)
  }

  # Sparsity pattern (1 where Qc has a stored entry; ignore exact-zero stored
  # entries to avoid penalising users who keep structural zeros for sparsity
  # reuse).
  pattern <- Qc
  pattern@x <- rep(1, length(pattern@x))

  # Validate graph hint against captured pattern.
  if (!is.null(graph)) {
    if (!methods::is(graph, "sparseMatrix")) {
      stop("`graph` must be NULL or a Matrix::sparseMatrix.", call. = FALSE)
    }
    if (nrow(graph) != n_latent || ncol(graph) != n_latent) {
      stop("`graph` must be ", n_latent, " x ", n_latent, "; got ",
           nrow(graph), " x ", ncol(graph), ".", call. = FALSE)
    }
    gC <- methods::as(graph, "CsparseMatrix")
    # Subset check: every (i, j) with pattern[i, j] != 0 must have gC[i, j] != 0.
    extra <- Matrix::which((pattern != 0) & (gC == 0), arr.ind = TRUE)
    if (nrow(extra) > 0) {
      stop("`Q(init)` has ", nrow(extra),
           " nonzero entries outside the supplied `graph` pattern.",
           call. = FALSE)
    }
  }

  # Evaluate the user prior once at init.
  lp0 <- tryCatch(
    prior(init),
    error = function(e) {
      stop("`prior(init)` raised an error: ", conditionMessage(e), call. = FALSE)
    }
  )
  if (!is.numeric(lp0) || length(lp0) != 1L || !is.finite(lp0)) {
    stop("`prior(init)` must return a finite numeric scalar; got ",
         deparse(lp0), ".", call. = FALSE)
  }

  # Evaluate mu(init) once, if supplied.
  if (!is.null(mu)) {
    mu0 <- tryCatch(
      mu(init),
      error = function(e) {
        stop("`mu(init)` raised an error: ", conditionMessage(e), call. = FALSE)
      }
    )
    if (!is.numeric(mu0) || length(mu0) != n_latent || anyNA(mu0)) {
      stop("`mu(init)` must return a finite numeric vector of length ",
           n_latent, ".", call. = FALSE)
    }
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

  # Validate bounds, if supplied.
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

  structure(
    list(
      type        = "tgmrf",          # registry key for .NL_REGISTRY dispatch
      Q           = Q,
      prior       = prior,
      mu          = mu,
      init        = init,
      theta_names = theta_names,
      theta_dim   = as.integer(theta_dim),
      n_latent    = as.integer(n_latent),
      pattern     = pattern,
      graph       = graph,
      bounds      = bounds,
      obs_idx     = obs_idx,
      name        = name,
      backend     = "r"
    ),
    class = c("tgmrf", "tulpa_latent_block")
  )
}


#' Mark an expression as a latent block in a tulpa formula
#'
#' @description
#' Wraps a user-defined latent block (currently a [tgmrf()] object) so the
#' formula parser can recognise and route it to the inference layer. The
#' call is structural — it is never executed at fit time. The parser
#' evaluates the inner expression in the formula's environment, removes
#' the `latent(...)` term from the fixed-effects formula, and attaches the
#' resulting object to `parsed$latent_blocks`.
#'
#' @param block A `tgmrf` (or, in the future, `tgeneric`) object.
#' @return Returns `block` invisibly. Outside a formula context the call is
#'   a no-op pass-through.
#' @export
latent <- function(block) {
  invisible(block)
}


#' @export
print.tgmrf <- function(x, ...) {
  label <- x$name %||% "tgmrf"
  cat("<tgmrf>", if (!is.null(x$name)) paste0(" \"", x$name, "\"") else "", "\n", sep = "")
  cat("  n_latent : ", x$n_latent, "\n", sep = "")
  cat("  theta    : ", x$theta_dim, " dim (",
      paste(x$theta_names, collapse = ", "), ")\n", sep = "")
  init_str <- paste(format(x$init, digits = 4), collapse = ", ")
  cat("  init     : ", init_str, "\n", sep = "")
  nnz <- length(x$pattern@x)
  cat("  Q nnz    : ", nnz, " (", format(100 * nnz / x$n_latent^2, digits = 3),
      "% of dense)\n", sep = "")
  if (!is.null(x$bounds)) {
    cat("  bounds   : supplied\n")
  }
  if (!is.null(x$graph)) {
    cat("  graph    : supplied\n")
  }
  if (!is.null(x$mu)) {
    cat("  mu()     : supplied (non-zero mean)\n")
  }
  cat("  backend  : ", x$backend, "\n", sep = "")
  invisible(x)
}


#' @export
summary.tgmrf <- function(object, ...) {
  print(object, ...)
  cat("\nlog prior at init: ", format(object$prior(object$init), digits = 6),
      "\n", sep = "")
  if (!is.null(object$bounds)) {
    cat("theta bounds:\n")
    tab <- rbind(lower = object$bounds$lower, upper = object$bounds$upper)
    colnames(tab) <- object$theta_names
    print(tab)
  }
  invisible(object)
}
