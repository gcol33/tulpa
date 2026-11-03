#' Central composite design (CCD) grid for nested-Laplace integration
#'
#' @description
#' Produces a structured set of standardised hyperparameter points
#' \eqn{z \in \mathbb{R}^k} for use as integration nodes in a nested
#' Laplace approximation when \eqn{k \ge 3}. CCD scales much better than
#' the full tensor `expand.grid()` used by the 1D and 2D backends:
#' a CCD has \eqn{1 + 2k + 2^{k - q}} points (1 centre, 2k axial,
#' \eqn{2^{k - q}} factorial), versus \eqn{m^k} for an `m`-per-axis
#' tensor product.
#'
#' Point layout (with centre+axial+factorial scaling \eqn{f_0}):
#'   * 1 centre point at the origin;
#'   * 2k axial points at \eqn{\pm f_0} along each coordinate axis;
#'   * \eqn{2^{k - q}} factorial points at corners of the hypercube,
#'     scaled to lie on a sphere of radius \eqn{f_0}.
#'
#' For \eqn{k \le 6} the factorial portion is the full \eqn{2^k}
#' design. For \eqn{k \ge 7} a half-fraction (\eqn{q = 1}) using the
#' defining word \eqn{x_1 \cdots x_k} keeps the point count reasonable
#' while preserving Resolution V.
#'
#' Used by [tulpa_nested_laplace()] for higher-dimensional hyperparameter
#' blocks. The standardised z-coordinates are mapped to physical
#' hyperparameters \eqn{\theta} via `ccd_to_theta()`.
#'
#' `ccd_to_theta()` converts standardised CCD z-coordinates (in
#' \eqn{\mathbb{R}^k}) to physical hyperparameters \eqn{\theta} via the affine
#' map \eqn{\theta = \hat\theta + L \cdot z} with optional log-scale
#' transform per component. `L` is typically a Cholesky factor of the
#' negative Hessian inverse evaluated at the (working) mode \eqn{\hat\theta}
#' -- i.e. it scales `z` to one posterior standard deviation per axis.
#'
#' `ccd_weights()` gives the per-point design weights for a `ccd_grid()` used
#' as nested-Laplace integration nodes. It implements the corrected R-INLA
#' central-composite-design weights: the formula in Rue, Martino and Chopin
#' (2009) has a typo, corrected by the R-INLA team to give a positive central
#' weight. With `m` hyperparameters, `np` design points and standardized
#' scaling `f0`, every non-centre point gets
#' \deqn{w = 1 / ((np - 1)(1 + e^{-m f0^2 / 2}(f0^2 - 1)))}
#' and the centre point gets \eqn{w_0 = 1 - (np - 1) w}. Used as
#' \eqn{\Delta_k}: the integration weight of node \eqn{k} is
#' \eqn{\Delta_k \exp(\log\,\mathrm{marginal}_k)}, renormalized. On a
#' standardized (whitened) hyperparameter posterior these reproduce the
#' Gaussian moments exactly.
#'
#' @param k Number of hyperparameters (length of theta). Must be >= 1.
#' @param f_0 Radius of the design sphere (default \eqn{\sqrt{k}}). Larger
#'   `f_0` spreads points further out; smaller concentrates near the
#'   centre. INLA's default scales like \eqn{\sqrt{k}}.
#'
#' @return `ccd_grid()`: a list with components:
#'   * `z`: numeric matrix `[n_points x k]` of standardised hyperparameter
#'     coordinates.
#'   * `n_points`: integer; total grid size.
#'   * `kind`: character vector labelling each point as
#'     `"center"`, `"axial"`, or `"factorial"`.
#'   * `f_0`: the sphere radius used.
#'
#'   `ccd_to_theta()`: a numeric matrix `[n_points x k]` of physical
#'   theta-values.
#'
#'   `ccd_weights()`: a numeric vector of length `ccd$n_points`, the design
#'   weight for each point (the centre weight `w0` for the central point, `w`
#'   otherwise).
#'
#' @keywords internal
#' @examples
#' g <- ccd_grid(3)
#' g$n_points
#' table(g$kind)
#' ccd_weights(g)
#'
#' g2 <- ccd_grid(2)
#' ccd_to_theta(g2$z, theta_hat = c(0, 1), L = diag(c(0.5, 0.2)))
#' @export
ccd_grid <- function(k, f_0 = sqrt(k)) {
  # A fractional `k` is refused, not truncated: ccd_grid(2.7) was a 2-D design.
  k <- .check_count(k, "k", min = 1L)
  .check_scalar(f_0, "f_0", positive = TRUE)

  # Centre point.
  z_center <- matrix(0, nrow = 1L, ncol = k)
  kind <- "center"

  # Axial points: +-f_0 along each coordinate axis.
  if (k >= 1L) {
    z_ax <- matrix(0, nrow = 2L * k, ncol = k)
    for (j in seq_len(k)) {
      z_ax[2L * j - 1L, j] <- f_0
      z_ax[2L * j, j]     <- -f_0
    }
    kind <- c(kind, rep("axial", 2L * k))
  } else {
    z_ax <- matrix(0, nrow = 0L, ncol = k)
  }

  # Factorial points: corners of (+-1)^k scaled so each corner has norm f_0.
  # For k <= 6 use the full 2^k design; for k >= 7 use a Resolution-V
  # half-fraction (defining word: product of all x_j == +1) so the
  # number of factorial points stays at 2^(k-1).
  # Skip for k == 1 since the factorial corners coincide with the axials.
  if (k <= 1L) {
    z_fac <- matrix(0, nrow = 0L, ncol = k)
  } else if (k <= 6L) {
    signs <- as.matrix(do.call(expand.grid, rep(list(c(-1, 1)), k)))
    dimnames(signs) <- NULL
    s <- f_0 / sqrt(k)
    z_fac <- s * signs
    kind <- c(kind, rep("factorial", nrow(z_fac)))
  } else {
    # Half-fraction via defining word x_1 * ... * x_k = +1.
    signs <- as.matrix(do.call(expand.grid, rep(list(c(-1, 1)), k - 1L)))
    dimnames(signs) <- NULL
    last <- apply(signs, 1L, prod)         # +1 fraction
    signs <- cbind(signs, last)
    s <- f_0 / sqrt(k)
    z_fac <- s * signs
    kind <- c(kind, rep("factorial", nrow(z_fac)))
  }

  z <- rbind(z_center, z_ax, z_fac)
  list(
    z = unname(z),
    n_points = nrow(z),
    kind = kind,
    f_0 = f_0
  )
}

#' @rdname ccd_grid
#' @param z Matrix `[n_points x k]` of standardised coordinates from
#'   `ccd_grid()`.
#' @param theta_hat Numeric vector of length `k`: centre of the design,
#'   in either physical or log-space (per `log_scale`).
#' @param L Numeric `[k x k]` matrix: scale/rotation applied to z. Pass
#'   `diag(sd)` for a diagonal axis-aligned grid where `sd` is the
#'   per-axis posterior SD; pass a Cholesky factor to capture
#'   correlations between hyperparameters.
#' @param log_scale Logical (or logical vector of length k). If `TRUE`
#'   for component j, `theta_hat[j]` and column j of the affine
#'   transform live on log-scale and are exponentiated afterward
#'   (useful for positive parameters like tau). Default `FALSE` everywhere.
#' @export
ccd_to_theta <- function(z, theta_hat, L, log_scale = FALSE) {
  if (!is.matrix(z)) stop("`z` must be a matrix.", call. = FALSE)
  k <- ncol(z)
  if (length(theta_hat) != k)
    stop("`theta_hat` must have length k = ncol(z).", call. = FALSE)
  if (!identical(dim(L), c(k, k)))
    stop("`L` must be k x k where k = ncol(z).", call. = FALSE)
  if (length(log_scale) == 1L) log_scale <- rep(log_scale, k)
  if (length(log_scale) != k)
    stop("`log_scale` must be length 1 or k.", call. = FALSE)

  theta <- sweep(z %*% t(L), 2L, theta_hat, FUN = "+")
  if (any(log_scale)) {
    cols <- which(log_scale)
    theta[, cols] <- exp(theta[, cols])
  }
  unname(theta)
}

#' @rdname ccd_grid
#' @param ccd A `ccd_grid()` result. The standardized scaling is recovered as
#'   `f0 = ccd$f_0 / sqrt(m)`: a `ccd_grid(m, f_0 = sqrt(m) * f0)` design places
#'   the factorial corners at `+/- f0`, matching the INLA convention (`f0 = 1.1`
#'   is the INLA default).
#' @export
ccd_weights <- function(ccd) {
  m  <- ncol(ccd$z)
  np <- ccd$n_points
  if (np <= 1L) return(rep(1, max(np, 1L)))
  f0 <- ccd$f_0 / sqrt(m)
  w  <- 1 / ((np - 1) * (1 + exp(-m * f0^2 / 2) * (f0^2 - 1)))
  w0 <- 1 - (np - 1) * w
  ifelse(ccd$kind == "center", w0, w)
}

# Absolute log measure of each CCD node on the coordinate theta = theta_hat +
# L z the design was mapped to, so that sum_k exp(g(theta_k) + measure_k)
# approximates the integral of exp(g).
#
# With phi_u(z) = exp(-|z|^2 / 2), the weights v_k = Delta_k phi_u(z_k) /
# sum_j Delta_j phi_u(z_j) are a moment rule for N(0, I): they sum to one,
# reproduce the unit second moments (the condition `ccd_weights()` is solved
# from), and every odd and mixed moment vanishes by the design's symmetry, so the
# rule is exact for polynomials of degree 3. Writing exp(g(z)) = r(z) phi_u(z),
# the integral is (2 pi)^(k/2) |det L| E_phi[r], and the rule reads
#   E_phi[r] ~ sum_k v_k r(z_k) = sum_k Delta_k exp(g_k) / sum_j Delta_j phi_u(z_j),
# exact whenever the ratio of the target to the design's Gaussian is a cubic in
# z, in particular when exp(g) is itself Gaussian at that centre and scale.
#' @keywords internal
.ccd_log_measure <- function(ccd, L) {
  k <- ncol(ccd$z)
  ld <- log(ccd_weights(ccd))
  log_norm <- .tulpa_logsumexp(ld - 0.5 * rowSums(ccd$z^2))
  ld - log_norm + 0.5 * k * log(2 * pi) +
    as.numeric(determinant(as.matrix(L), logarithm = TRUE)$modulus)
}

# The ALIGNED TENSOR: `n_per_axis` nodes per axis over +/- `span` on the
# whitened coordinates z, mapped to the outer coordinates as
# `theta = theta_hat + L z` with `L` the Cholesky factor of the inverse mode
# Hessian -- the orientation the CCD is laid in, so the nodes follow the outer
# mode's correlation, but a tensor whose uniform cells discretize the density
# (`.nl_node_support("grid")`) rather than a moment rule. The default design for
# up to two outer coordinates (`.nl_outer_design()`).
.nl_aligned_tensor <- function(k, n_per_axis, span) {
  n_per_axis <- as.integer(n_per_axis)
  ax <- seq(-span, span, length.out = n_per_axis)
  z <- as.matrix(expand.grid(rep(list(ax), k)))
  dimnames(z) <- NULL
  list(z = z, k = as.integer(k), h = 2 * span / (n_per_axis - 1L))
}

# The cells an aligned tensor `z` (spacing `h`) adds past its ends. A tensor
# laid at +/- `span` SDs of the mode's Gaussian cuts a marginal whose tail is
# heavier than that Gaussian's: the left tail of a log-scale posterior reaches
# several SDs past it. Per whitened axis and side, while the outermost level
# carries more than `2 * tail_mass` of the axis's marginal -- the box read
# leaves half of it, the half-cell past the node, to a region no node measures
# -- the axis gains one level `h` beyond, laid across every level of the other
# axes so the tensor stays a tensor of uniform cells. `lw` is each cell's log
# mass.
#
# Only the new cells beyond the rows of the edge level that hold all but
# `row_tail` of its mass are solved (`solve`); the rest carry no mass, as the
# rows they continue carry none to speak of, and so does a corner cell past two
# axes grown in the same round. Returns `list(z, solve)` for the new rows of
# `z`, NULL when no end qualifies.
.nl_tensor_edge_cells <- function(z, lw, h,
                                  tail_mass = .nl_diag("edge_tail_mass"),
                                  row_tail = .nl_diag("level_row_tail")) {
  ok <- is.finite(lw)
  if (!any(ok)) return(NULL)
  w <- numeric(length(lw))
  w[ok] <- exp(lw[ok] - max(lw[ok]))
  w <- w / sum(w)
  key <- round(2 * z / h)
  key_id <- function(K) do.call(paste, as.data.frame(K))
  lev <- lapply(seq_len(ncol(z)), function(j) sort(unique(key[, j])))
  grow <- list()
  for (j in seq_len(ncol(z))) {
    l <- lev[[j]]
    p <- vapply(l, function(v) sum(w[key[, j] == v]), numeric(1))
    if (p[1L] / 2 > tail_mass) {
      grow[[length(grow) + 1L]] <- list(j = j, edge = l[1L], step = -2)
      lev[[j]] <- c(l[1L] - 2, lev[[j]])
    }
    if (p[length(p)] / 2 > tail_mass) {
      grow[[length(grow) + 1L]] <- list(j = j, edge = l[length(l)], step = 2)
      lev[[j]] <- c(lev[[j]], l[length(l)] + 2)
    }
  }
  if (!length(grow)) return(NULL)
  full <- as.matrix(expand.grid(lev, KEEP.OUT.ATTRS = FALSE))
  dimnames(full) <- NULL
  full <- full[!(key_id(full) %in% key_id(key)), , drop = FALSE]
  held <- character(0)
  for (g in grow) {
    idx <- which(key[, g$j] == g$edge & w > 0)
    o <- idx[order(w[idx], decreasing = TRUE)]
    share <- cumsum(w[o]) / sum(w[o])
    keep <- o[c(TRUE, share[-length(share)] < 1 - row_tail)]
    nxt <- key[keep, , drop = FALSE]
    nxt[, g$j] <- nxt[, g$j] + g$step
    held <- c(held, key_id(nxt))
  }
  list(z = full * h / 2, solve = key_id(full) %in% held)
}

# Absolute log measure of each aligned-tensor node on the coordinate
# `theta = theta_hat + L z`: the whitened cell volume `h^k` times `|det L|`.
.nl_aligned_tensor_log_measure <- function(tens, L) {
  rep(tens$k * log(tens$h) + sum(log(abs(diag(as.matrix(L))))), nrow(tens$z))
}
