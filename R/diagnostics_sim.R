#' Simulation-Based Diagnostics for tulpa Models
#'
#' @description
#' Model-checking tools based on posterior predictive simulation.
#' These are native R implementations equivalent to DHARMa's test suite,
#' with no external dependencies. They work with any model that provides
#' `simulate()`, `fitted()`, and `residuals()` methods.
#'
#' @return The diagnostic functions documented in this family return their
#'   individual results (a test-statistic object, a data frame of residuals, or
#'   a `check_model` summary); see each function's own help page.
#'
#' @name tulpa_diagnostics
NULL


# ==============================================================================
# PIT Residuals
# ==============================================================================

#' PIT (Probability Integral Transform) residuals
#'
#' For each observation, computes the quantile of the observed value within
#' the posterior predictive distribution. If the model is correct, PIT
#' residuals follow Uniform(0, 1).
#'
#' For integer-valued responses, a randomisation step avoids discrete
#' artefacts: the residual is drawn uniformly between P(sim < obs) and
#' P(sim <= obs).
#'
#' @param object A fitted model with a `simulate()` method, or a matrix
#'   of simulated values (n_obs x nsim)
#' @param observed Observed response vector (required if `object` is a matrix)
#' @param nsim Number of simulations (default 250)
#' @param seed Random seed (default 123)
#'
#' @return Numeric vector of length n_obs with values in `[0, 1]`
#'
#' @export
pit_residuals <- function(object, observed = NULL, nsim = 250L, seed = 123L) {

  if (is.matrix(object)) {
    sims <- object
    if (is.null(observed)) stop("observed required when object is a matrix", call. = FALSE)
    obs <- observed
  } else {
    sims <- as.matrix(simulate(object, nsim = nsim, seed = seed))
    obs <- observed %||% object$y %||% object$.internal$fit_args$y
    if (is.null(obs)) {
      stop("Cannot extract observed data. Provide `observed` argument.", call. = FALSE)
    }
  }

  n <- length(obs)

  # P(sim < obs) and P(sim <= obs) per observation
  lower <- rowMeans(sims < obs)
  upper <- rowMeans(sims <= obs)

  # Randomise between lower and upper for integer data
  .seed_scoped(seed + 1L)
  lower + runif(n) * (upper - lower)
}


# ==============================================================================
# Test Uniformity (KS test on PIT residuals)
# ==============================================================================

#' Test uniformity of PIT residuals
#'
#' If the model is correct, PIT residuals should follow Uniform(0, 1).
#' Applies a Kolmogorov-Smirnov test against that null.
#' Equivalent to `DHARMa::testUniformity()`.
#'
#' @param object A fitted model, a numeric vector of PIT residuals, or a
#'   matrix of simulations
#' @param observed Observed data (if object is simulations matrix)
#' @param nsim Number of simulations (default 250)
#' @param seed Random seed (default 123)
#' @param plot If TRUE, draws a QQ plot
#'
#' @return An `htest` object (KS test result)
#'
#' @export
test_uniformity <- function(object, observed = NULL, nsim = 250L, seed = 123L,
                            plot = FALSE) {

  if (is.numeric(object) && is.null(dim(object))) {
    r <- object
  } else {
    r <- pit_residuals(object, observed = observed, nsim = nsim, seed = seed)
  }

  result <- suppressWarnings(ks.test(r, "punif"))

  if (plot) {
    n <- length(r)
    expected <- (seq_len(n) - 0.5) / n
    plot(sort(r), expected,
         xlab = "PIT residuals", ylab = "Expected Uniform",
         main = sprintf("QQ Uniform (KS p = %.3f)", result$p.value),
         pch = 19, cex = 0.6)
    abline(0, 1, col = "red", lty = 2)
  }

  result
}


# ==============================================================================
# Test Dispersion
# ==============================================================================

#' Test for over- or underdispersion
#'
#' Compares the variance of observed data to the variance expected under
#' the fitted model (via simulation). Ratio > 1 = overdispersion;
#' < 1 = underdispersion. Equivalent to `DHARMa::testDispersion()`.
#'
#' @param object A fitted model with `simulate()` method
#' @param observed Observed response vector (optional -- extracted from fit)
#' @param nsim Number of simulations (default 250)
#' @param seed Random seed (default 123)
#' @param alternative `"two.sided"`, `"greater"`, or `"less"`
#'
#' @return An `htest` object with dispersion ratio and p-value
#'
#' @export
test_dispersion <- function(object, observed = NULL, nsim = 250L, seed = 123L,
                            alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)

  sims <- as.matrix(simulate(object, nsim = nsim, seed = seed))
  obs <- observed %||% object$y %||% object$.internal$fit_args$y

  var_obs <- var(obs)
  var_sim <- apply(sims, 2, var)
  ratio <- var_obs / median(var_sim)

  p_greater <- mean(var_sim >= var_obs)
  p_less <- mean(var_sim <= var_obs)
  p_val <- switch(alternative,
    two.sided = min(2 * min(p_greater, p_less), 1),
    greater = p_greater,
    less = p_less
  )

  structure(
    list(
      statistic = c("dispersion ratio" = ratio),
      p.value = p_val,
      alternative = alternative,
      method = "Simulation-based dispersion test",
      data.name = deparse(substitute(object))
    ),
    class = "htest"
  )
}


# ==============================================================================
# Test Outliers
# ==============================================================================

#' Test for outliers (simulation envelope)
#'
#' Counts how many observations fall outside the min-to-max range of
#' all simulations. Under a correct model, the expected number is
#' approximately `2 * N / (nsim + 1)`. Equivalent to `DHARMa::testOutliers()`.
#'
#' @param object A fitted model with `simulate()` method
#' @param observed Observed response vector (optional)
#' @param nsim Number of simulations (default 250)
#' @param seed Random seed (default 123)
#'
#' @return An `htest` object (binomial test)
#'
#' @export
test_outliers <- function(object, observed = NULL, nsim = 250L, seed = 123L) {

  sims <- as.matrix(simulate(object, nsim = nsim, seed = seed))
  obs <- observed %||% object$y %||% object$.internal$fit_args$y
  N <- length(obs)

  sim_min <- apply(sims, 1, min)
  sim_max <- apply(sims, 1, max)
  n_outliers <- sum(obs < sim_min | obs > sim_max)

  p_expected <- 2 / (nsim + 1)
  result <- binom.test(n_outliers, N, p = p_expected, alternative = "greater")
  result$method <- sprintf("Simulation outlier test (%d/%d outside envelope)",
                           n_outliers, N)
  result
}


# ==============================================================================
# Test Zero Inflation
# ==============================================================================

#' Test for zero inflation
#'
#' Compares the number of zeros in observed data to the distribution
#' expected under the fitted model. Equivalent to `DHARMa::testZeroInflation()`.
#'
#' @param object A fitted model with `simulate()` method
#' @param observed Observed response vector (optional)
#' @param nsim Number of simulations (default 250)
#' @param seed Random seed (default 123)
#'
#' @return An `htest` object with zero-inflation ratio and p-value
#'
#' @export
test_zero_inflation <- function(object, observed = NULL, nsim = 250L, seed = 123L) {

  sims <- as.matrix(simulate(object, nsim = nsim, seed = seed))
  obs <- observed %||% object$y %||% object$.internal$fit_args$y

  n_zero_obs <- sum(obs == 0)
  n_zero_sim <- colSums(sims == 0)
  ratio <- n_zero_obs / max(median(n_zero_sim), 1)

  p_greater <- mean(n_zero_sim >= n_zero_obs)
  p_less <- mean(n_zero_sim <= n_zero_obs)
  p_val <- min(2 * min(p_greater, p_less), 1)

  structure(
    list(
      statistic = c("zero-inflation ratio" = ratio),
      parameter = c("observed zeros" = n_zero_obs,
                     "expected zeros" = median(n_zero_sim)),
      p.value = p_val,
      alternative = "two.sided",
      method = "Simulation-based zero-inflation test",
      data.name = deparse(substitute(object))
    ),
    class = "htest"
  )
}


# ==============================================================================
# Moran's I
# ==============================================================================

#' Moran's I test for spatial autocorrelation in residuals
#'
#' Tests whether residuals exhibit spatial structure after model fitting.
#' Supports inverse-distance and k-nearest-neighbour weight matrices.
#' No external dependencies (uses normal approximation for inference).
#'
#' @param object A fitted model, or a numeric vector of residuals
#' @param coords N x 2 coordinate matrix (required)
#' @param weights Weight scheme: `"inverse"` or `"knn"`
#' @param k Number of neighbours for knn (default 10)
#' @param resid_type Residual type if extracting from model (default `"pearson"`)
#' @param alternative `"two.sided"`, `"greater"`, or `"less"`
#'
#' @return An `htest` object with Moran's I, expected I, and p-value
#'
#' @examples
#' set.seed(1)
#' coords <- cbind(runif(50), runif(50))
#' resid  <- rnorm(50)
#' moran_i(resid, coords)
#'
#' @export
moran_i <- function(object, coords,
                    weights = c("inverse", "knn"), k = 10L,
                    resid_type = "pearson",
                    alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  weights <- match.arg(weights)

  if (is.numeric(object) && is.null(dim(object))) {
    x <- object
  } else {
    x <- residuals(object, type = resid_type)
  }

  coords <- as.matrix(coords)
  N <- length(x)
  if (nrow(coords) != N) {
    stop(sprintf("coords has %d rows but residuals has length %d",
                 nrow(coords), N), call. = FALSE)
  }

  D <- as.matrix(dist(coords))
  if (weights == "inverse") {
    W <- 1 / D
    diag(W) <- 0
    W[!is.finite(W)] <- 0
    method_str <- "Moran's I (inverse-distance weights)"
  } else {
    k <- min(k, N - 1L)
    W <- matrix(0, N, N)
    for (i in seq_len(N)) {
      # Drop self explicitly rather than assuming the zero self-distance sorts
      # first: a coincident coordinate (another point at distance 0) can outrank
      # it, which would make a point its own neighbour and drop a true one.
      nn <- setdiff(order(D[i, ]), i)[seq_len(k)]
      W[i, nn] <- 1
    }
    method_str <- sprintf("Moran's I (k=%d nearest neighbours)", k)
  }

  xbar <- mean(x)
  dx <- x - xbar
  ss <- sum(dx^2)
  S0 <- sum(W)
  I <- (N / S0) * (as.numeric(dx %*% W %*% dx) / ss)

  # Expected value and variance under randomisation
  EI <- -1 / (N - 1)
  S1 <- 0.5 * sum((W + t(W))^2)
  S2 <- sum((rowSums(W) + colSums(W))^2)
  n2 <- N^2
  k2 <- (sum(dx^4) / N) / (ss / N)^2
  VI <- (N * ((n2 - 3 * N + 3) * S1 - N * S2 + 3 * S0^2) -
         k2 * (N * (N - 1) * S1 - 2 * N * S2 + 6 * S0^2)) /
        ((N - 1) * (N - 2) * (N - 3) * S0^2) - EI^2

  Z <- (I - EI) / sqrt(max(VI, 1e-10))
  p_val <- switch(alternative,
    two.sided = 2 * pnorm(abs(Z), lower.tail = FALSE),
    greater = pnorm(Z, lower.tail = FALSE),
    less = pnorm(Z, lower.tail = TRUE)
  )

  structure(
    list(
      statistic = c("Moran's I" = I),
      parameter = c("Expected I" = EI),
      p.value = p_val,
      alternative = alternative,
      method = method_str,
      data.name = deparse(substitute(object))
    ),
    class = "htest"
  )
}


# ==============================================================================
# Durbin-Watson
# ==============================================================================

#' Durbin-Watson test for temporal autocorrelation
#'
#' Tests first-order autocorrelation in temporally-ordered residuals.
#'
#' @param object A numeric vector of temporally-ordered residuals
#' @param alternative `"two.sided"`, `"greater"` (positive autocorr), or `"less"`
#'
#' @return An `htest` object with DW statistic, lag-1 r, and p-value
#'
#' @export
durbin_watson <- function(object, alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  x <- as.numeric(object)
  n <- length(x)
  if (n < 3L) stop("need at least 3 observations for Durbin-Watson test", call. = FALSE)

  e <- x - mean(x)
  dw <- sum(diff(e)^2) / sum(e^2)
  r1 <- 1 - dw / 2

  Z <- (dw - 2) / sqrt(4 / n)
  p_val <- switch(alternative,
    two.sided = 2 * pnorm(abs(Z), lower.tail = FALSE),
    greater = pnorm(Z, lower.tail = TRUE),
    less = pnorm(Z, lower.tail = FALSE)
  )

  structure(
    list(
      statistic = c("DW" = dw),
      parameter = c("lag-1 r" = r1),
      p.value = p_val,
      alternative = alternative,
      method = "Durbin-Watson test for temporal autocorrelation",
      data.name = deparse(substitute(object))
    ),
    class = "htest"
  )
}


# ==============================================================================
# Variogram
# ==============================================================================

#' Empirical semivariogram of residuals
#'
#' Computes the empirical semivariogram in distance bins for visual
#' assessment of remaining spatial structure in residuals.
#'
#' @param object A fitted model, or a numeric vector of residuals
#' @param coords N x 2 coordinate matrix (required)
#' @param n_bins Number of distance bins (default 15)
#' @param max_dist Maximum distance (default: half the maximum pairwise distance)
#' @param resid_type Residual type if extracting from model (default `"pearson"`)
#'
#' @return A `tulpa_variogram` data.frame with columns `dist`, `gamma`, `n_pairs`
#'
#' @export
tulpa_variogram <- function(object, coords, n_bins = 15L, max_dist = NULL,
                            resid_type = "pearson") {

  if (is.numeric(object) && is.null(dim(object))) {
    x <- object
  } else {
    x <- residuals(object, type = resid_type)
  }

  coords <- as.matrix(coords)
  N <- length(x)

  D <- dist(coords)
  d_vec <- as.numeric(D)
  if (is.null(max_dist)) max_dist <- max(d_vec) / 2

  idx <- which(lower.tri(matrix(0, N, N)), arr.ind = TRUE)
  sq_diff <- (x[idx[, 1]] - x[idx[, 2]])^2

  breaks <- seq(0, max_dist, length.out = n_bins + 1L)
  mids <- (breaks[-1] + breaks[-(n_bins + 1L)]) / 2

  gamma <- numeric(n_bins)
  n_pairs <- integer(n_bins)
  for (b in seq_len(n_bins)) {
    in_bin <- d_vec >= breaks[b] & d_vec < breaks[b + 1L]
    n_pairs[b] <- sum(in_bin)
    if (n_pairs[b] > 0) gamma[b] <- mean(sq_diff[in_bin]) / 2
  }

  out <- data.frame(dist = mids, gamma = gamma, n_pairs = n_pairs)
  out <- out[out$n_pairs > 0, ]
  class(out) <- c("tulpa_variogram", "data.frame")
  out
}


#' @export
plot.tulpa_variogram <- function(x, ...) {
  plot(x$dist, x$gamma,
       xlab = "Distance", ylab = "Semivariance",
       main = "Empirical Semivariogram",
       pch = 19, cex = sqrt(x$n_pairs / max(x$n_pairs)) * 2, ...)
  lines(x$dist, x$gamma, lty = 2, col = "grey50")
  invisible(x)
}


# ==============================================================================
# checkModel -- diagnostic panel plot
# ==============================================================================

#' Diagnostic panel plot
#'
#' Produces a 2x2 or 1x3 panel of diagnostic plots:
#' 1. QQ plot of PIT residuals vs Uniform (with KS p-value)
#' 2. Residuals vs fitted (with lowess smoother)
#' 3. Dispersion histogram (observed variance vs simulated)
#' 4. Spatial correlogram via Moran's I (if coords provided)
#'
#' @param object A fitted model with `simulate()`, `fitted()`, `residuals()`
#' @param coords Optional N x 2 coordinate matrix for spatial panel
#' @param nsim Number of simulations (default 250)
#' @param seed Random seed (default 123)
#'
#' @return Invisible list with `ks_p`, `disp_ratio`, `moran` (if spatial)
#'
#' @export
check_model <- function(object, coords = NULL, nsim = 250L, seed = 123L) {

  sims <- as.matrix(simulate(object, nsim = nsim, seed = seed))
  obs <- object$y %||% object$.internal$fit_args$y
  pit <- pit_residuals(sims, observed = obs, nsim = nsim, seed = seed)

  has_coords <- !is.null(coords)
  old_par <- par(mfrow = if (has_coords) c(2, 2) else c(1, 3),
                 mar = c(4, 4, 2.5, 1))
  on.exit(par(old_par))

  # Panel 1: QQ Uniform
  n <- length(pit)
  expected <- (seq_len(n) - 0.5) / n
  ks_p <- suppressWarnings(ks.test(pit, "punif"))$p.value
  plot(sort(pit), expected,
       xlab = "PIT residuals", ylab = "Expected Uniform",
       main = sprintf("QQ Uniform (KS p = %.3f)", ks_p),
       pch = 19, cex = 0.5, col = adjustcolor("black", 0.6))
  abline(0, 1, col = "red", lty = 2, lwd = 1.5)

  # Panel 2: Residuals vs fitted
  fv <- fitted(object)
  r <- residuals(object, type = "pearson")
  plot(fv, r,
       xlab = "Fitted", ylab = "Pearson residuals",
       main = "Residuals vs Fitted",
       pch = 19, cex = 0.5, col = adjustcolor("black", 0.6))
  abline(h = 0, col = "red", lty = 2, lwd = 1.5)
  lo <- tryCatch(lowess(fv, r), error = function(e) NULL)
  if (!is.null(lo)) lines(lo, col = "blue", lwd = 1.5)

  # Panel 3: Dispersion
  var_sim <- apply(sims, 2, var)
  var_obs <- var(obs)
  ratio <- var_obs / median(var_sim)
  hist(var_sim, breaks = 20,
       main = sprintf("Dispersion (ratio = %.2f)", ratio),
       xlab = "Simulated variance", col = "grey80", border = "grey50")
  abline(v = var_obs, col = "red", lwd = 2)

  # Panel 4: Spatial correlogram
  moran_result <- NULL
  if (has_coords) {
    coords <- as.matrix(coords)
    ks_vals <- c(3, 5, 8, 12, 20)
    ks_vals <- ks_vals[ks_vals < nrow(coords)]
    I_vals <- numeric(length(ks_vals))
    p_vals <- numeric(length(ks_vals))
    for (j in seq_along(ks_vals)) {
      mi <- moran_i(r, coords = coords, weights = "knn", k = ks_vals[j])
      I_vals[j] <- mi$statistic
      p_vals[j] <- mi$p.value
    }
    sig <- p_vals < 0.05
    plot(ks_vals, I_vals, type = "b",
         xlab = "k neighbours", ylab = "Moran's I",
         main = "Spatial Correlogram",
         pch = ifelse(sig, 19, 1),
         col = ifelse(sig, "red", "black"))
    abline(h = -1 / (nrow(coords) - 1), col = "grey50", lty = 2)
    moran_result <- data.frame(k = ks_vals, I = I_vals, p = p_vals)
  }

  invisible(list(ks_p = ks_p, disp_ratio = ratio, moran = moran_result))
}
