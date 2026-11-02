# Outer hyperparameter mode-find.
#
# `fit_spde_nested_ccd()` and `fit_st_nested()`'s auto-grid both locate an outer
# mode the same way: box-constrained L-BFGS-B on a marginal that arrives from a
# compiled kernel with no analytic gradient, so both depend on `optim()`'s
# central-difference step and both treat a failed call as "no usable mode".
# The invocation lives here once; the tuning each one uses is in
# `.NL_MODE_FIND` (R/settings.R).

# Resolve a consumer's mode-find tuning, applying a caller's `control$mode_find`
# overrides on top of the settings defaults.
#
# `control` is the already-validated front-door control list, or NULL. Names
# inside `mode_find` are checked against the consumer's own key set, so a
# misspelled sub-knob errors rather than silently fitting the default -- the
# same contract `tulpa_check_control()` gives the outer list.
#' @keywords internal
.nl_mode_find_tuning <- function(consumer, control = NULL) {
  keys <- names(.NL_MODE_FIND[[consumer]])
  if (is.null(keys)) {
    stop("Unknown mode-find consumer '", consumer, "'.", call. = FALSE)
  }
  tune <- stats::setNames(
    lapply(keys, function(k) .nl_mode_find(consumer, k)), keys)

  ov <- control[["mode_find"]]
  if (is.null(ov)) return(tune)
  if (!is.list(ov)) {
    stop("`control$mode_find` must be a list.", call. = FALSE)
  }
  if (length(ov) == 0L) return(tune)

  nm <- names(ov)
  if (is.null(nm) || any(!nzchar(nm))) {
    stop("every `control$mode_find` entry must be named.", call. = FALSE)
  }
  unknown <- setdiff(nm, keys)
  if (length(unknown)) {
    stop(sprintf(
      "Unknown control$mode_find knob(s): %s.\nAllowed: %s.",
      paste(sQuote(unknown, q = FALSE), collapse = ", "),
      paste(sort(keys), collapse = ", ")), call. = FALSE)
  }
  for (k in nm) {
    v <- ov[[k]]
    if (!is.numeric(v) || length(v) != 1L || !is.finite(v) || v <= 0) {
      stop(sprintf("`control$mode_find$%s` must be one positive finite number.",
                   k), call. = FALSE)
    }
    tune[[k]] <- v
  }
  tune
}

# Box-constrained L-BFGS-B, with a numerical gradient by default. `gr`, when
# supplied, is an analytic gradient of `fn` (see fit_spde.R's
# spde_log_marginal_grad for the one caller that has one); `optim()`'s own
# central-difference step (tuned by `ndeps`) is what runs otherwise, exactly as
# before. Returns the `optim()` result, or NULL if the call errored -- callers
# decide what an absent mode means for them.
#' @keywords internal
.nl_lbfgsb_mode_find <- function(par, fn, lower, upper, tuning,
                                 hessian = FALSE, gr = NULL) {
  tryCatch(
    stats::optim(par = par, fn = fn, gr = gr,
                 method = "L-BFGS-B", lower = lower, upper = upper,
                 hessian = hessian,
                 control = list(factr = tuning$factr,
                                maxit = tuning$maxit,
                                ndeps = rep(tuning$ndeps, length(par)))),
    error = function(e) NULL
  )
}

# Unconstrained minimisation of `fn` run to a stationary point. BFGS moves
# from `par` to the basin; Newton steps on the curvature at the point it
# stopped then continue until the Newton decrement g' H^-1 g / 2 -- the
# quadratic model's estimate of how far `fn` still sits above its minimum, in
# the objective's own units -- is at most `tol`. BFGS's relative-change test
# reads the objective's scale instead: on a log-likelihood of a few hundred it
# stops wherever successive values agree to the tolerance times that size,
# which can leave a scale parameter short of the maximum by a margin the
# likelihood still resolves. `gr` is an analytic gradient of `fn`, or NULL for
# central differences. Returns `optim()`'s fields (`par`, `value`, `counts`,
# `hessian` at the returned point, `convergence` 0 once the decrement is
# within `tol`) plus `newton_decrement`.
#' @keywords internal
.optim_to_stationary <- function(par, fn, gr = NULL, tol = 1e-8,
                                 maxit = 200L, newton_max = 25L) {
  opt <- stats::optim(par = par, fn = fn, gr = gr, method = "BFGS",
                      control = list(maxit = maxit, reltol = 1e-10))
  grad <- if (is.null(gr)) {
    function(x) {
      h <- 1e-5 * pmax(abs(x), 1)
      vapply(seq_along(x), function(j) {
        xp <- x; xp[j] <- xp[j] + h[j]
        xm <- x; xm[j] <- xm[j] - h[j]
        (fn(xp) - fn(xm)) / (2 * h[j])
      }, numeric(1))
    }
  } else gr
  x <- opt$par
  f <- opt$value
  dec <- Inf
  n_newton <- 0L
  repeat {
    H <- stats::optimHess(x, fn, gr)
    g <- grad(x)
    step <- tryCatch(solve((H + t(H)) / 2, g), error = function(e) NULL)
    if (is.null(step) || any(!is.finite(step))) break
    dec <- 0.5 * sum(g * step)
    if (!is.finite(dec) || dec < 0) break
    if (dec <= tol || n_newton >= newton_max) break
    # Damped step: halve until the objective does not increase.
    t_step <- 1
    accepted <- FALSE
    for (i in seq_len(30L)) {
      xn <- x - t_step * step
      fn_x <- fn(xn)
      if (is.finite(fn_x) && fn_x <= f) { accepted <- TRUE; break }
      t_step <- t_step / 2
    }
    if (!accepted) break
    x <- xn; f <- fn_x
    n_newton <- n_newton + 1L
  }
  opt$par <- x
  opt$value <- f
  # Every exit from the loop leaves H evaluated at the returned point.
  opt$hessian <- H
  opt$newton_decrement <- dec
  opt$counts <- c(opt$counts, newton = n_newton)
  opt$convergence <- if (is.finite(dec) && dec >= 0 && dec <= tol) 0L else 1L
  opt
}
