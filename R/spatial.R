#' Spatial structure specifications for tulpa
#'
#' @description
#' Functions to specify spatial random effects for tulpa models.
#' Spatial effects are shared between processes by default,
#' which helps prevent bias from spatially-structured
#' unmeasured confounders.
#'
#' @return The spatial constructors documented in this family
#'   ([spatial_car()], [spatial_bym2()], [spatial_gp()], and the others) each
#'   return a `tulpa_spatial` (or related) specification object to pass to the
#'   `spatial` argument of [tulpa()].
#'
#' @name tulpa_spatial
NULL
