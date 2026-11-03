# Censored and truncated Gaussian responses: `y | cens(lower, upper)` and
# `y | trunc(lb, ub)` on the left of a tulpa() formula.
#
# Both are the gaussian family read through a per-observation bound. The engine
# owns the two kernels (`interval_gaussian`, `truncated_gaussian`,
# src/laplace_family_link.h); this file is the front door's half: reading the
# response term off the formula's parse tree, evaluating the bounds against the
# data, checking them, and the R-side density / draw the accessors read a
# bounded fit through.

# The response forms. `args` is the signature the call is matched against, so
# positional and named arguments resolve the way R resolves them; `open` is the
# value an argument left out takes (none for `cens`, whose two bounds are the
# observation itself).
.RESPONSE_BOUND_FORMS <- list(
  cens = list(
    family = "interval_gaussian",
    args   = function(lower, upper) NULL,
    open   = NULL
  ),
  trunc = list(
    family = "truncated_gaussian",
    args   = function(lb = -Inf, ub = Inf) NULL,
    open   = list(lb = -Inf, ub = Inf)
  )
)

# The bounded-response families, by the kernel family name a fit carries.
.BOUND_FAMILIES <- vapply(.RESPONSE_BOUND_FORMS, `[[`, character(1), "family")

#' @keywords internal
.family_reads_bounds <- function(family) {
  is.character(family) && length(family) == 1L && family %in% .BOUND_FAMILIES
}

# Read a `y | cens(lower, upper)` / `y | trunc(lb, ub)` response off the left
# side of a formula. `response_expr` is the parse tree of that side; anything
# not a call to `|` is an ordinary response and returns NULL. Returns
# list(kind, family, response_expr, lower, upper): the response's own
# expression and the two bound expressions, unevaluated.
#' @keywords internal
.parse_response_bounds <- function(response_expr) {
  if (!is.call(response_expr) || !identical(response_expr[[1L]], as.name("|"))) {
    return(NULL)
  }
  if (length(response_expr) != 3L) {
    stop("A `|` on the left of `~` takes a response and one bound term, ",
         "e.g. `y | cens(lower, upper)` or `y | trunc(lb, ub)`.", call. = FALSE)
  }
  term <- response_expr[[3L]]
  head <- if (is.call(term) && is.name(term[[1L]])) as.character(term[[1L]])
  if (is.null(head) || !head %in% names(.RESPONSE_BOUND_FORMS)) {
    stop("The term after `|` on the left of `~` must be cens(lower, upper) or ",
         "trunc(lb, ub); got `",
         paste(deparse(term, width.cutoff = 500L), collapse = ""), "`.",
         call. = FALSE)
  }
  form <- .RESPONSE_BOUND_FORMS[[head]]
  matched <- tryCatch(match.call(form$args, term), error = function(e) {
    stop(sprintf("`%s` in the response: %s", head, conditionMessage(e)),
         call. = FALSE)
  })
  given <- as.list(matched)[-1L]
  formals_nm <- names(formals(form$args))
  vals <- lapply(formals_nm, function(nm) {
    if (!is.null(given[[nm]])) return(given[[nm]])
    if (is.null(form$open)) {
      stop(sprintf("`%s()` in the response needs both of %s.", head,
                   paste0("`", formals_nm, "`", collapse = " and ")),
           call. = FALSE)
    }
    form$open[[nm]]
  })
  list(kind = head, family = form$family,
       response_expr = response_expr[[2L]],
       lower = vals[[1L]], upper = vals[[2L]])
}

# Check a pair of bound vectors and return them as doubles. `kind` is "cens"
# (a class interval: lower <= upper, a finite lower == upper an exactly
# observed value) or "trunc" (truncation bounds: lower < upper, and `y`, where
# present, inside them). NA is refused here: the formula door has already read
# a missing bound as an open side, and the joint door takes -Inf / Inf for it.
# `prefix` and `nm` name the arguments in the message.
#' @keywords internal
.check_response_bounds <- function(lower, upper, N, kind, prefix = "",
                                   nm = c("lower", "upper"), y = NULL) {
  lower <- as.numeric(lower)
  upper <- as.numeric(upper)
  if (length(lower) != N || length(upper) != N) {
    stop(sprintf("%slength(%s)/length(%s) must equal length(y) (%d).",
                 prefix, nm[1L], nm[2L], N), call. = FALSE)
  }
  if (anyNA(lower) || anyNA(upper)) {
    stop(sprintf("%s`%s` / `%s` must be non-NA; -Inf / Inf leave a side open.",
                 prefix, nm[1L], nm[2L]), call. = FALSE)
  }
  bad <- if (identical(kind, "cens")) {
    lower > upper | (lower == upper & !is.finite(lower))
  } else {
    lower >= upper
  }
  if (any(bad)) {
    i <- which(bad)[1L]
    stop(sprintf(paste0(
      "%s%s at row %d: `%s` = %s, `%s` = %s. %s"), prefix,
      if (identical(kind, "cens")) "An empty interval" else "An empty truncation range",
      i, nm[1L], format(lower[i]), nm[2L], format(upper[i]),
      if (identical(kind, "cens"))
        "Each lower lies at or below its upper; equal finite bounds record an exact value."
      else "Each lower truncation point lies strictly below its upper."),
      call. = FALSE)
  }
  if (!is.null(y) && identical(kind, "trunc")) {
    out <- !is.na(y) & (y < lower | y > upper)
    if (any(out)) {
      i <- which(out)[1L]
      stop(sprintf(paste0(
        "%sthe response at row %d (%s) lies outside its truncation range ",
        "[%s, %s]; a truncated response is observed inside it."), prefix, i,
        format(y[i]), format(lower[i]), format(upper[i])), call. = FALSE)
    }
  }
  list(lower = lower, upper = upper)
}

# Evaluate the bounds of a parsed bounded response against `data`. A bound may
# be a column or any expression of the data, a scalar recycled over the rows;
# NA reads as the open side (-Inf for a lower bound, Inf for an upper one), the
# convention survival::Surv(type = "interval2") uses.
#
# For `cens` the response `y` is not what the likelihood reads -- the interval
# is -- so a row whose value is unknown may carry NA there. `y`, where present,
# lies in its interval. A row with both bounds open carries no information and
# is a missing response (`y_kernel` NA), read for its linear predictor alone as
# every door masks an NA response; every other row hands the kernel a finite
# placeholder in `y_kernel`, since an NA there would mask it.
#' @keywords internal
.eval_response_bounds <- function(rb, data, env, N, y) {
  ev <- function(expr, what, open) {
    v <- tryCatch(eval(expr, envir = data, enclos = env), error = function(e) {
      stop(sprintf("The %s bound of `%s()` in the response could not be read: %s",
                   what, rb$kind, conditionMessage(e)), call. = FALSE)
    })
    if (!is.numeric(v) && !is.logical(v)) {
      stop(sprintf("The %s bound of `%s()` in the response must be numeric.",
                   what, rb$kind), call. = FALSE)
    }
    v <- as.numeric(v)
    if (length(v) == 1L) v <- rep(v, N)
    if (length(v) != N) {
      stop(sprintf(paste0("The %s bound of `%s()` in the response has length ",
                          "%d; it must be a scalar or one value per row (%d)."),
                   what, rb$kind, length(v), N), call. = FALSE)
    }
    v[is.na(v)] <- open
    v
  }
  lower <- ev(rb$lower, "lower", -Inf)
  upper <- ev(rb$upper, "upper", Inf)
  nm <- if (identical(rb$kind, "cens")) c("lower", "upper") else c("lb", "ub")
  b <- .check_response_bounds(lower, upper, N, rb$kind,
                              sprintf("`%s()` in the response: ", rb$kind), nm,
                              y = y)
  y <- as.numeric(y)
  y_kernel <- y
  if (identical(rb$kind, "cens")) {
    out <- !is.na(y) & (y < b$lower | y > b$upper)
    if (any(out)) {
      i <- which(out)[1L]
      stop(sprintf(paste0(
        "`cens()` in the response: the response at row %d (%s) lies outside ",
        "its interval [%s, %s]."), i, format(y[i]), format(b$lower[i]),
        format(b$upper[i])), call. = FALSE)
    }
    informative <- is.finite(b$lower) | is.finite(b$upper)
    fill <- ifelse(is.finite(b$lower), b$lower, b$upper)
    y_kernel <- ifelse(informative, ifelse(is.na(y), fill, y), NA_real_)
  }
  list(kind = rb$kind, family = rb$family, lower = b$lower, upper = b$upper,
       y_kernel = y_kernel)
}

# The bounded family a `cens()` / `trunc()` response turns `family` into. The
# kernels are the Gaussian identity-link density read through a bound, so the
# family the caller names has to be that one.
#' @keywords internal
.bounded_family <- function(family, rb) {
  if (!identical(family, "gaussian")) {
    stop(sprintf(paste0(
      "`%s()` in the response is the censored / truncated Gaussian; it takes ",
      "family = 'gaussian' (identity link), not '%s'."), rb$kind, family),
      call. = FALSE)
  }
  rb$family
}

# The joint-arm fields carrying a bounded response's bounds.
#' @keywords internal
.response_bound_arm_fields <- function(rb) {
  if (is.null(rb)) return(list())
  if (identical(rb$kind, "cens")) {
    list(lower = rb$lower, upper = rb$upper)
  } else {
    list(trunc_lower = rb$lower, trunc_upper = rb$upper)
  }
}

# Per-observation log-likelihood of a bounded response, from the compiled
# kernels the fit solved under. `phi` is the residual VARIANCE (the R-level
# convention, `.phi_to_kernel()`), a scalar or one value per element.
#' @keywords internal
.bound_loglik <- function(eta, y, family, bounds, phi) {
  # A [draws x obs] matrix holds one observation per column, so each bound is
  # repeated down its column.
  n <- length(eta)
  per <- if (is.matrix(eta)) nrow(eta) else 1L
  lo <- rep_len(rep(as.numeric(bounds$lower), each = per), n)
  hi <- rep_len(rep(as.numeric(bounds$upper), each = per), n)
  yy <- rep_len(as.numeric(y), n)
  yy[is.na(yy)] <- 0
  sg <- .phi_to_kernel(family, as.numeric(phi))
  cpp_bound_gaussian_loglik(yy, lo, hi, as.numeric(eta), sg, family)
}

# One response draw per element of `eta`. A censored response is the latent
# Gaussian value the interval was recorded from, which is what an exact row
# observes and what a censored row hides; a truncated response is drawn from
# the truncated law itself, by inverting its CDF on the retained mass.
#' @keywords internal
.bound_sample <- function(eta, family, bounds, phi) {
  n <- length(eta)
  sd <- rep_len(sqrt(as.numeric(phi)), n)
  if (identical(family, "interval_gaussian")) {
    return(stats::rnorm(n, eta, sd))
  }
  lo <- rep_len(as.numeric(bounds$lower), n)
  hi <- rep_len(as.numeric(bounds$upper), n)
  pa <- stats::pnorm(lo, eta, sd)
  pb <- stats::pnorm(hi, eta, sd)
  u <- stats::runif(n)
  # Read in the upper tail where the retained mass sits there, so a range far
  # above eta keeps its resolution.
  upper_tail <- pa > 0.5
  out <- numeric(n)
  lt <- !upper_tail
  out[lt] <- stats::qnorm(pa[lt] + u[lt] * (pb[lt] - pa[lt]), eta[lt], sd[lt])
  if (any(upper_tail)) {
    qa <- stats::pnorm(lo, eta, sd, lower.tail = FALSE)[upper_tail]
    qb <- stats::pnorm(hi, eta, sd, lower.tail = FALSE)[upper_tail]
    out[upper_tail] <- stats::qnorm(qa - u[upper_tail] * (qa - qb),
                                    eta[upper_tail], sd[upper_tail],
                                    lower.tail = FALSE)
  }
  pmin(pmax(out, lo), hi)
}

# The bounds a joint arm of a bounded family carries, in the shape the
# accessors read (`lower`, `upper`), or NULL.
#' @keywords internal
.joint_arm_bounds <- function(arm) {
  N <- length(arm$y)
  if (identical(arm$family, "interval_gaussian") && !is.null(arm$lower)) {
    return(list(kind = "cens", lower = arm$lower, upper = arm$upper))
  }
  if (identical(arm$family, "truncated_gaussian") &&
      (!is.null(arm$trunc_lower) || !is.null(arm$trunc_upper))) {
    return(list(kind = "trunc", lower = arm$trunc_lower %||% rep(-Inf, N),
                upper = arm$trunc_upper %||% rep(Inf, N)))
  }
  NULL
}

# A bounded family's density and draws are functions of the bounds; reading
# one without them would be a different model.
#' @keywords internal
.bounds_or_stop <- function(family, bounds) {
  if (is.null(bounds)) {
    stop(sprintf(paste0(
      "family '%s' reads per-observation bounds, and none reached this ",
      "accessor. At new data, supply the bound columns of the response term."),
      family), call. = FALSE)
  }
  bounds
}

# The bounds of a bounded fit's response term evaluated at `newdata`: the bound
# expressions of the formula read against the new rows. NULL when the fit has
# no bounded response, or when `newdata` does not carry what the bounds read
# (a censored response is then still drawn, a truncated one refused by
# `.bounds_or_stop()`).
#' @keywords internal
.response_bounds_at <- function(object, newdata) {
  if (!.family_reads_bounds(object$family) || is.null(object$formula)) {
    return(NULL)
  }
  rb <- tulpa_parse_formula(object$formula)$response_bounds
  if (is.null(rb)) return(NULL)
  n <- nrow(newdata)
  tryCatch(
    .eval_response_bounds(rb, newdata, environment(object$formula), n,
                          rep(NA_real_, n)),
    error = function(e) NULL)
}
