#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib tulpa, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom Matrix sparseMatrix
#' @importFrom grDevices adjustcolor colorRampPalette rgb
#' @importFrom graphics abline hist lines pairs par polygon
#' @importFrom methods as
#' @importFrom stats approx ar as.formula ave binom.test coef cor cov density
#'   dist dnorm fitted ks.test logLik lowess median model.frame model.matrix
#'   model.response na.pass pnorm predict qnorm reshape residuals runif sd
#'   setNames simulate var
#' @importFrom utils head
## usethis namespace: end
NULL

# `.data` is the rlang/ggplot2 pronoun used inside conditional ggplot2 plotting
# helpers; `ratio` is the downstream model-package (numdenom) accessor invoked by
# prepare_map_data_from_fit when mapping a ratio fit. Neither is a tulpa symbol,
# so register them to keep R CMD check's global-variable analysis quiet.
utils::globalVariables(c(".data", "ratio"))
