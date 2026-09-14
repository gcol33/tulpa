#' @param hyperprior The prior an outer hyperparameter axis carries when the
#'   call states no density for it, `"proper"` (default) or `"flat"`.
#'   `"proper"` folds a normalised density on the axis's integration coordinate
#'   into `log_marginal`: the PC prior `P(sigma > 3) = 0.01` on a standard
#'   deviation, variance or precision axis, the PC range prior at 0.2 times the
#'   coordinates' bounding-box diagonal on a range or lengthscale axis, a uniform
#'   on a bounded axis, and the PC + LKJ prior over a free-covariance block. An
#'   axis with no sourced density is named in `log_hyperprior_declined`.
#'   `"flat"` folds no density of the engine's own, so such an axis is
#'   integrated under its cell measure alone, is named in
#'   `log_hyperprior_declined` as `"flat_hyperprior"`, and the fit's
#'   `log_evidence` declines with `"improper_hyperprior"`. A density the call
#'   states -- a `prior_*` argument, a block's `rho_prior`, `prior_range` or
#'   `prior_sigma`, the copy coefficient's slab, a [tgmrf()] block's own prior --
#'   applies under either choice. The same two choices, under the same names,
#'   are offered by [tulpa()], [tulpa_eb()], [tulpa_re_cov_nested()] and
#'   [fit_spde()].
