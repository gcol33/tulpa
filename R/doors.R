# doors.R
# ------------------------------------------------------------------------------
# Named front doors: tglmm(), tgam(). Each is a contract-narrowing VIEW on
# tulpa() -- the same parser, the same tier/mode selection, the same backends.
# A door adds nothing to the engine; it removes reach. Two things follow from
# that, and both live in the registry below rather than in the door bodies:
#
#   * a signature that omits the arguments the model class cannot use, so the
#     door cannot express a model outside its class, and
#   * a structural contract on the formula (what must be present, what must not),
#     checked before dispatch so the failure names the model class rather than
#     surfacing as a backend error further in.
#
# Adding a door is one registry entry plus a stub whose body is the shared
# .door_fit() call; the contract logic, the withheld-argument redirect, and the
# display hook are written once here.
#
# The doors deliberately carry an explicit signature rather than deriving one
# from formals(tulpa) at build time: it is what a user reads and tab-completes.
# test-doors.R asserts each door's signature is exactly tulpa()'s minus its
# withheld arguments, so a new statistical argument on tulpa() surfaces as a
# failing test rather than a silently stale door.
# ------------------------------------------------------------------------------


# Structural features a formula can carry, as read off the parse. The doors
# state their contract in these names, so a new latent structure is added to
# the vocabulary once and every door can then require or forbid it.
#' @keywords internal
.DOOR_FEATURES <- list(
  re       = function(p) (p$n_re_terms %||% 0L) > 0L,
  smooth   = function(p) (p$n_smooth_terms %||% 0L) > 0L,
  latent   = function(p) (p$n_latent_blocks %||% 0L) > 0L,
  spatial  = function(p) (p$n_spatial_field_blocks %||% 0L) > 0L ||
                          !is.null(p$spatial_var),
  temporal = function(p) (p$n_temporal_field_blocks %||% 0L) > 0L ||
                          !is.null(p$temporal_var)
)

# How each feature reads in a formula, for the contract error messages. Written
# as a mid-sentence noun phrase carrying its own article, so the templates below
# interpolate it without agreement fixes.
#' @keywords internal
.DOOR_FEATURE_SYNTAX <- c(
  re       = "a random-effect term such as (1 | g)",
  smooth   = "an s(...) smoother term such as s(x)",
  latent   = "a latent(...) block",
  spatial  = "a spatial field",
  temporal = "a temporal field"
)

#' @keywords internal
.TULPA_DOORS <- list(
  tglmm = list(
    subclass = "tulpa_glmm",
    label    = "GLMM",
    long     = "generalized linear mixed model",
    # What the model degenerates to when the required structure is absent, for
    # the contract error: a GLMM without a random effect is a GLM.
    degenerate = "GLM",
    requires = "re",
    forbids  = c("smooth", "latent", "spatial", "temporal"),
    withheld = c("spatial", "temporal")
  ),
  tgam = list(
    subclass = "tulpa_gam",
    label    = "GAM",
    long     = "generalized additive model",
    degenerate = "GLM",
    # Random effects are allowed alongside the smooths (the GAMM case); only
    # the structures with their own integrator are out of scope.
    requires = "smooth",
    forbids  = c("latent", "spatial", "temporal"),
    withheld = c("spatial", "temporal")
  )
)


# The door label for a fit, from its class. NULL for a fit that came through
# tulpa() itself, which is what print.tulpa_fit falls back on.
#' @keywords internal
.door_label_for <- function(x) {
  cls <- oldClass(x)
  for (d in names(.TULPA_DOORS)) {
    if (.TULPA_DOORS[[d]]$subclass %in% cls) return(.TULPA_DOORS[[d]]$label)
  }
  NULL
}


# Arguments the caller actually supplied, evaluated. Read from the door's own
# match.call() (so partial matching is already resolved) rather than from
# missing(), and left absent when unsupplied so tulpa() applies its own
# defaults -- the door never restates a default and so cannot drift from one.
#' @keywords internal
.door_args <- function(cl, env, fn) {
  nms      <- setdiff(names(formals(fn)), "...")
  supplied <- setdiff(names(as.list(cl))[-1L], "")
  mget(intersect(supplied, nms), envir = env)
}


# An argument a door withholds is not a typo -- it is a model the door cannot
# express -- so it gets a redirect to tulpa() rather than the generic unknown-
# argument error. Partial matches are caught too: `spat = ` would otherwise
# partial-match tulpa()'s `spatial` on the way through and fit the very model
# the door exists to exclude.
#' @keywords internal
.door_reject_withheld <- function(door, spec, dots) {
  if (!length(dots) || !length(spec$withheld)) return(invisible(NULL))
  nm  <- names(dots) %||% rep("", length(dots))
  hit <- spec$withheld[stats::na.omit(pmatch(nm, spec$withheld))]
  if (length(hit)) {
    stop(sprintf(paste0(
      "%s() is the %s door and does not take `%s`; a model carrying %s is\n",
      "outside the class it fits. Use tulpa(), which carries every latent\n",
      "structure the engine has."),
      door, spec$long, paste(hit, collapse = "`, `"),
      paste(.DOOR_FEATURE_SYNTAX[hit], collapse = " or ")), call. = FALSE)
  }
  invisible(NULL)
}


# The structural contract, checked against the parsed formula before dispatch.
#' @keywords internal
.door_check_contract <- function(door, spec, parsed) {
  present <- vapply(.DOOR_FEATURES, function(f) isTRUE(f(parsed)), logical(1))

  absent <- setdiff(spec$requires, names(present)[present])
  if (length(absent)) {
    stop(sprintf(paste0(
      "%s() fits a %s, so the formula needs %s.\n",
      "Without one this is a plain %s, which tulpa() fits."),
      door, spec$long,
      paste(.DOOR_FEATURE_SYNTAX[absent], collapse = " and "),
      spec$degenerate), call. = FALSE)
  }

  extra <- intersect(spec$forbids, names(present)[present])
  if (length(extra)) {
    stop(sprintf(paste0(
      "%s() fits a %s; the formula also carries %s. Fit it with tulpa(),\n",
      "which carries every latent structure the engine has."),
      door, spec$long,
      paste(.DOOR_FEATURE_SYNTAX[extra], collapse = " and ")), call. = FALSE)
  }
  invisible(NULL)
}


# The shared door body: check the contract, dispatch through tulpa(), stamp the
# subclass and the door's own call. Everything a door does that tulpa() does not
# is in these four lines.
#' @keywords internal
.door_fit <- function(door, args, dots, cl) {
  spec <- .TULPA_DOORS[[door]]
  .door_reject_withheld(door, spec, dots)
  # No formula to check the contract against: let the dispatch below raise the
  # standard missing-argument error rather than parsing NULL into a worse one.
  if (!is.null(args$formula)) {
    .door_check_contract(door, spec, tulpa_parse_formula(args$formula))
  }
  fit <- do.call(tulpa, c(args, dots))
  if (is.list(fit)) {
    fit$call <- cl
    fit <- .finalize_fit(fit, extra_class = spec$subclass)
  }
  fit
}


#' Fit a generalized linear mixed model
#'
#' @description
#' The GLMM door onto [tulpa()]: fixed effects plus random effects, and nothing
#' else. Same families, same `mode` selection, same backends -- the formula is
#' required to carry at least one `(1 | g)` / `(1 + x | g)` term, and the
#' smoother, `latent(...)`, spatial, and temporal structures are refused with a
#' pointer to [tulpa()] rather than fit.
#'
#' `spatial` and `temporal` are absent from the signature for the same reason.
#' A model that needs one is not a GLMM, and [tulpa()] fits it.
#'
#' @inheritParams tulpa
#' @return A `tulpa_fit` with `"tulpa_glmm"` prepended to its class, so
#'   [print()] reports the random-effect covariance alongside the fixed effects.
#'   Every `tulpa_fit` accessor ([coef()], [summary()], [ranef()], [VarCorr()],
#'   [mcmc_diagnostics()]) applies unchanged.
#' @seealso [tulpa()] for the general engine door, [tgam()] for smoothers,
#'   [VarCorr()] for the covariance summary, [tulpa_eb()] to estimate the
#'   random-effect covariance rather than condition on it.
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 300L
#' g <- rep(seq_len(20), length.out = n)
#' b <- rnorm(20, 0, 0.7)
#' d <- data.frame(x = rnorm(n), g = factor(g))
#' d$y <- rpois(n, exp(0.4 + 0.5 * d$x + b[g]))
#' fit <- tglmm(y ~ x + (1 | g), data = d, family = "poisson", mode = "eb")
#' fit
#' VarCorr(fit)
#' }
#' @export
tglmm <- function(formula, data,
                  family = "gaussian",
                  mode = "auto",
                  sigma_re = NULL,
                  n_trials = NULL,
                  weights = NULL,
                  phi = 1.0,
                  estimate_phi = FALSE,
                  phi2 = NULL,
                  beta_prior = NULL,
                  re_prior = NULL,
                  ziformula = NULL,
                  zi_prior = NULL,
                  warm_start = NULL,
                  control = list(),
                  ...) {
  .door_fit("tglmm", .door_args(match.call(), environment(), sys.function()),
            list(...), match.call())
}


#' Fit a generalized additive model
#'
#' @description
#' The GAM door onto [tulpa()]: fixed effects plus `s(x)` covariate smoothers,
#' optionally with random effects (the GAMM case). Each smoother is an RW2
#' (default) or RW1 GMRF over the binned covariate, and its smoothness
#' hyperparameter is integrated by nested Laplace rather than selected -- so
#' the fitted smooth carries the uncertainty of not knowing how smooth it is.
#'
#' The formula must carry at least one `s(...)` term. `latent(...)` blocks and
#' spatial / temporal fields are refused with a pointer to [tulpa()], which
#' fits a smoother alongside them.
#'
#' @inheritParams tulpa
#' @return A `tulpa_fit` with `"tulpa_gam"` prepended to its class, so [print()]
#'   lists the smoother terms and [plot()] draws the fitted smooths. Every
#'   `tulpa_fit` accessor applies unchanged; [smooth_effects()] returns a
#'   smooth's node-by-node posterior mean.
#' @seealso [tulpa()] for the general engine door, [smooth_effects()] for the
#'   fitted values, [tglmm()] for the mixed-model door.
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(x = runif(300, -2, 2))
#' d$y <- rpois(300, exp(0.3 + sin(2 * d$x)))
#' fit <- tgam(y ~ s(x), data = d, family = "poisson")
#' fit
#' head(smooth_effects(fit))
#' }
#' @export
tgam <- function(formula, data,
                 family = "gaussian",
                 mode = "auto",
                 sigma_re = NULL,
                 n_trials = NULL,
                 weights = NULL,
                 phi = 1.0,
                 estimate_phi = FALSE,
                 phi2 = NULL,
                 beta_prior = NULL,
                 re_prior = NULL,
                 ziformula = NULL,
                 zi_prior = NULL,
                 warm_start = NULL,
                 control = list(),
                 ...) {
  .door_fit("tgam", .door_args(match.call(), environment(), sys.function()),
            list(...), match.call())
}


# A door's display is the generic fixed-effect report, plus whatever the backend
# that ran has to add, plus the section specific to the model class. The middle
# piece is what a door would otherwise lose: dispatching to the door subclass
# skips the backend's own print method, and a fit whose hyperparameters were
# integrated should still report its grid and outer Pareto-k.
#' @keywords internal
.print_door_base <- function(x, ...) {
  print.tulpa_fit(x, ...)
  if (inherits(x, "tulpa_nested_laplace")) .print_nested_laplace_body(x)
  invisible(x)
}


# A GLMM's second half is the covariance its random effects are drawn from,
# with VarCorr's own estimated / sampled / conditioned label carried through --
# a conditioned sigma_re is an input echoed back, not a result, and the door
# must not present it as one.
#' @export
print.tulpa_glmm <- function(x, ...) {
  .print_door_base(x, ...)
  vc <- tryCatch(VarCorr(x), error = function(e) NULL)
  if (!is.null(vc) && nrow(vc)) {
    cat("\nRandom effects:\n")
    print(data.frame(term = vc$term, coef = vc$coef, sd = round(vc$sd, 4),
                     source = vc$source, row.names = NULL))
  }
  invisible(x)
}


#' @export
print.tulpa_gam <- function(x, ...) {
  .print_door_base(x, ...)
  sm <- x$smooth_terms
  if (length(sm)) {
    cat("\nSmooth terms:\n")
    print(data.frame(
      term  = vapply(sm, function(s) paste0("s(", s$var, ")"), character(1)),
      basis = vapply(sm, function(s) toupper(s$type), character(1)),
      nodes = vapply(sm, function(s) as.integer(s$k), integer(1)),
      row.names = NULL))
    cat("\nFitted values: smooth_effects(fit), plot(fit)\n")
  }
  invisible(x)
}


#' Plot a GAM fit
#'
#' @param x A fit from [tgam()].
#' @param type `"smooth"` (default) draws the fitted smooth of each `s(...)`
#'   term; the other values are the [plot.tulpa_fit()] posterior views.
#' @param term Which smoother to draw: index or covariate name. `NULL`
#'   (default) draws every one.
#' @param ... Passed to [plot()] for the smooth panels, or to
#'   [plot.tulpa_fit()] for the other types.
#' @return `x`, invisibly.
#' @export
plot.tulpa_gam <- function(x, type = c("smooth", "density", "trace", "pairs"),
                           term = NULL, ...) {
  type <- match.arg(type)
  if (type != "smooth") return(plot.tulpa_fit(x, type = type, ...))

  sm <- x$smooth_terms
  if (!length(sm)) {
    stop("This fit carries no s(...) smoother terms to plot.", call. = FALSE)
  }
  which_terms <- if (is.null(term)) seq_along(sm) else term
  if (length(which_terms) > 1L) {
    op <- graphics::par(mfrow = grDevices::n2mfrow(length(which_terms)))
    on.exit(graphics::par(op), add = TRUE)
  }
  for (j in which_terms) {
    e <- smooth_effects(x, j)
    graphics::plot(e$x, e$estimate, type = "l",
                   xlab = attr(e, "var"),
                   ylab = paste0("s(", attr(e, "var"), ")"), ...)
  }
  invisible(x)
}
