#' Temporal structure specifications for tulpa
#'
#' @description
#' Functions to specify temporal random effects for tulpa models.
#' Temporal effects are shared between processes by default,
#' which helps prevent bias from temporally-structured
#' unmeasured confounders.
#'
#' @return The temporal constructors documented in this family
#'   ([temporal_rw1()], [temporal_rw2()], [temporal_ar1()], and the others) each
#'   return a `tulpa_temporal` specification object to pass to the `temporal`
#'   argument of [tulpa()].
#'
#' @name tulpa_temporal
NULL
