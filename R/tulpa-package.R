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
# tulpa symbol, so register them to keep R CMD check's global-variable analysis quiet.
utils::globalVariables(c(".data", "ratio", ".phl_w"))

# Scoped set.seed: seeds the RNG and restores the caller's RNG state when the
# calling function exits, so a user-supplied `seed` does not clobber the
# session RNG stream. No-op when `seed` is NULL. The on.exit restore is
# registered in `envir` (the caller's frame), the base-R equivalent of
# withr::defer().
#' @keywords internal
.seed_scoped <- function(seed, envir = parent.frame()) {
  if (is.null(seed)) return(invisible(FALSE))
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) {
    get(".Random.seed", envir = .GlobalEnv)
  } else NULL
  restore <- function() {
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }
  do.call(on.exit, list(as.call(list(restore)), add = TRUE), envir = envir)
  set.seed(as.integer(seed))
  invisible(TRUE)
}
