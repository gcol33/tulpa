# BRASIL: Best Rational Approximation by Successive Interval Length adjustment
# (Hofreither 2021, doi:10.1007/s11075-020-01042-0). Computes the degree-(m, m)
# best uniform (minimax) rational approximation of a scalar function on an
# interval, returned in barycentric form with its zeros and poles extracted.
#
# Faithful port of the reference implementation (c-f-h/baryrat): Chebyshev-node
# initialization, Knockaert (2008) Loewner-matrix barycentric interpolation, the
# two-phase node-relocation / interval-length-adjustment iteration, and
# golden-section local-error maxima. Pole / zero extraction uses base-R
# polyroot on the cleared barycentric denominator / numerator (the secular
# polynomial), so no generalized-eigensolver dependency is needed.
#
# This is the coefficient engine for the fractional-SPDE rational approximation
#: rational_spde_coefficients calls it for fractional nu, and
# fit_spde()'s fractional path (.spde_nested_logmarginal_at) consumes those
# coefficients to fit and integrate a fractional Matern field.

# Chebyshev nodes of the first kind, n points in [a, b].
.cheb_nodes <- function(n, a, b) {
  t <- (1 - cos((2 * seq_len(n) - 1) / (2 * n) * pi))   # in [0, 2]
  t * ((b - a) / 2) + a
}

# Coefficients (ascending powers) of prod(x - roots).
.poly_from_roots <- function(roots) {
  coef <- 1
  for (r in roots) coef <- c(0, coef) - c(r * coef, 0)   # multiply by (x - r)
  coef
}

# Barycentric rational interpolant of (nodes, values) with odd length 2n+1;
# numerator and denominator both have degree <= n. Knockaert (2008) Loewner
# construction: the weights are a nullspace vector of the Loewner matrix.
.bary_interp_rat <- function(nodes, values) {
  N <- length(values)
  if (N %% 2L != 1L) stop("number of interpolation nodes must be odd")
  ia <- seq(1L, N, by = 2L)   # 1st, 3rd, ... (the "a" set)
  ib <- seq(2L, N, by = 2L)   # 2nd, 4th, ... (the "b" set)
  xa <- nodes[ia]; xb <- nodes[ib]
  va <- values[ia]; vb <- values[ib]
  # Loewner matrix B[i, j] = (vb[i] - va[j]) / (xb[i] - xa[j]).
  B <- outer(seq_along(xb), seq_along(xa),
             function(i, j) (vb[i] - va[j]) / (xb[i] - xa[j]))
  # Nullspace vector = right singular vector of the smallest singular value.
  sv <- svd(B, nu = 0, nv = ncol(B))
  w <- sv$v[, ncol(sv$v)]
  list(nodes = xa, values = va, weights = w)
}

# Evaluate a barycentric rational at points x (vectorized), exact at nodes.
.bary_eval <- function(br, x) {
  z <- br$nodes; f <- br$values; w <- br$weights
  vapply(x, function(xi) {
    d <- xi - z
    hit <- which(d == 0)
    if (length(hit)) return(f[hit[1]])
    c <- w / d
    sum(c * f) / sum(c)
  }, numeric(1))
}

# Poles of the barycentric rational: roots of the cleared denominator
# q(x) = sum_j w_j prod_{k != j} (x - z_k). Zeros: same with w_j * f_j.
.bary_secular_roots <- function(z, w) {
  N <- length(z)
  acc <- numeric(N)                       # degree N-1 polynomial, ascending
  for (j in seq_len(N)) {
    pj <- .poly_from_roots(z[-j])         # prod_{k != j} (x - z_k)
    acc <- acc + w[j] * pj
  }
  # Drop leading (highest-power) zeros so polyroot sees the true degree.
  while (length(acc) > 1L && acc[length(acc)] == 0) acc <- acc[-length(acc)]
  if (length(acc) <= 1L) return(numeric(0))
  r <- polyroot(acc)
  Re(r[abs(Im(r)) < 1e-6 * (1 + abs(Re(r)))])
}
.bary_poles <- function(br) .bary_secular_roots(br$nodes, br$weights)
.bary_zeros <- function(br) .bary_secular_roots(br$nodes, br$weights * br$values)

# Per-subinterval local maxima of g via vectorized golden-section search,
# plus boundary searches on the two end subintervals.
.local_maxima_golden <- function(g, nodes, num_iter = 30L) {
  gm <- (3 - sqrt(5)) / 2
  bnd <- function(lo, hi) {
    z <- c(lo, lo + (hi - lo) * gm, hi)
    gb <- g(z[2])
    for (k in seq_len(num_iter)) {
      mid <- (z[1] + z[3]) / 2
      far <- if (z[2] <= mid) z[3] else z[1]
      xx <- z[2] + gm * (far - z[2]); gx <- g(xx)
      if (gx > gb) {
        if (xx > z[2]) z[1] <- z[2] else z[3] <- z[2]
        z[2] <- xx; gb <- gx
      } else {
        if (xx < z[2]) z[1] <- xx else z[3] <- xx
      }
    }
    c(z[2], gb)
  }
  out_x <- numeric(length(nodes) - 1)
  out_v <- numeric(length(nodes) - 1)
  for (i in seq_len(length(nodes) - 1)) {
    bv <- bnd(nodes[i], nodes[i + 1])
    out_x[i] <- bv[1]; out_v[i] <- bv[2]
  }
  list(x = out_x, v = out_v)
}

# Best degree-(m, m) rational approximation of f on [a, b] via BRASIL. Returns
# the barycentric interpolant plus convergence info; .bary_zeros / .bary_poles
# read off the numerator / denominator roots.
.brasil <- function(f, interval, m, tol = 1e-4, maxiter = 1000L,
                    max_step_size = 0.1, step_factor = 0.1, num_iter = 30L,
                    init_steps = 100L) {
  a <- interval[1]; b <- interval[2]
  if (!(a < b)) stop("invalid interval")
  nn <- 2L * m + 1L
  nodes <- .cheb_nodes(nn, a, b)
  stepsize <- NA_real_
  converged <- FALSE; max_err <- NA_real_; deviation <- NA_real_
  br <- NULL
  for (k in seq_len(init_steps + maxiter)) {
    br <- .bary_interp_rat(nodes, f(nodes))
    errfun <- function(x) abs(f(x) - .bary_eval(br, x))
    lm <- .local_maxima_golden(errfun, c(a, nodes, b), num_iter)
    max_err <- max(lm$v); min_err <- min(lm$v)
    deviation <- max_err / min_err - 1
    converged <- deviation <= tol
    if (converged || k == init_steps + maxiter) break

    if (k <= init_steps) {
      # Phase 1: move the node nearest the lowest-error interval onto the
      # highest-error point.
      max_i <- which.max(lm$v); max_x <- lm$x[max_i]
      if (max_x == a) max_x <- (3 * a + nodes[1]) / 4
      else if (max_x == b) max_x <- (nodes[nn] + 3 * b) / 4
      min_k <- which.min(lm$v)
      if (min_k == 1L) {
        min_j <- 1L
      } else if (min_k == length(lm$v)) {
        min_j <- nn
      } else {
        min_j <- if (abs(max_x - nodes[min_k - 1L]) < abs(max_x - nodes[min_k]))
          min_k else min_k - 1L
      }
      nodes[min_j] <- max_x
      nodes <- sort(nodes)
    } else {
      # Phase 2: scale subinterval lengths by normalized local-error deviation.
      lens <- diff(c(a, nodes, b))
      mean_err <- mean(lm$v)
      max_dev <- max(abs(lm$v - mean_err))
      ndev <- (lm$v - mean_err) / max_dev
      stepsize <- min(max_step_size, step_factor * max_dev / mean_err)
      lens <- lens * (1 - stepsize)^ndev
      lens <- lens * (b - a) / sum(lens)
      nodes <- (a + cumsum(lens))[-length(lens)]
    }
  }
  list(br = br, converged = converged, error = max_err, deviation = deviation)
}


# BRASIL roots for the rational SPDE. `.spde_rational_roots()` returns the
# degree-(m, m) best uniform rational approximation of x^{-beta_rem} on
# [spectrum_ratio, 1], as its numerator factors `rc` (= 1 / zero), denominator
# factors `rb` (= 1 / pole) and the scale constant, with
#   R(x) = scale * prod(x - 1 / rc) / prod(x - 1 / rb).
# `spectrum_ratio` = l_min / l_max of the operator whose fractional power is
# approximated; the approximation interval is [spectrum_ratio, 1] on the
# l / l_max-normalized axis. `.spde_rational_terms()` turns these roots into the
# partial-fraction covariance terms the field is assembled from.
#
# The fitted path is the compiled search (`src/brasil.h`): the outer integrator
# visits one spectrum interval per (range, sigma) cell -- 1102 of them for a
# single n = 40 fit, essentially all distinct -- so this runs once per cell and
# at R speed costs ~0.55 s apiece, the whole run time of a fractional fit
# (gcol33/tulpa#818). `.spde_rational_roots_r()` below is the reference oracle
# the port reproduces; test-brasil-cpp-oracle.R pins them together.
.spde_rational_roots <- function(order, beta, spectrum_ratio, tol = 1e-7) {
  cpp_spde_rational_roots(as.integer(order), as.numeric(beta),
                          as.numeric(spectrum_ratio), as.numeric(tol))
}

.spde_rational_roots_r <- function(order, beta, spectrum_ratio, tol = 1e-7) {
  if (order < 1L) stop("rational order must be >= 1.", call. = FALSE)
  if (!(spectrum_ratio > 0 && spectrum_ratio < 1)) {
    stop("spectrum_ratio must be in (0, 1).", call. = FALSE)
  }
  m_beta <- max(1L, as.integer(floor(beta)))
  beta_rem <- beta - (m_beta - 1)
  res <- .brasil(function(x) x^(-beta_rem), c(spectrum_ratio, 1), order, tol = tol)
  zr <- .bary_zeros(res$br)
  pr <- .bary_poles(res$br)
  xm <- (spectrum_ratio + 1) / 2
  scale <- .bary_eval(res$br, xm) * prod(xm - pr) / prod(xm - zr)
  list(
    rb       = 1 / pr,            # denominator factors -> Pl
    rc       = 1 / zr,            # numerator factors   -> Pr
    scale    = scale,
    m_beta   = m_beta,
    beta_rem = beta_rem,
    error    = res$error          # rational approximation error of x^{-beta_rem}
  )
}


# Partial-fraction terms of the covariance-based rational SPDE (Bolin, Simas &
# Xiong 2024). The Matern field of operator order alpha = nu + d/2 has the
# covariance operator tau^-2 L^-alpha. With m = floor(alpha) and b = alpha - m in
# (0, 1), the fractional factor l^-b is a Stieltjes function, so its best
# rational approximation on the spectrum [kappa^2, l_max] has negative real poles
# and positive residues:
#   l^-b ~= c0 + sum_j B_j / (l + e_j),         c0 > 0, B_j > 0, e_j > 0,
# and therefore
#   l^-alpha ~= c0 l^-m + sum_j B_j l^-m / (l + e_j),
# a sum of m_rat + 1 covariance operators each of which is the covariance of a
# proper GMRF. The roots come from `.spde_rational_roots()` on the
# l / l_max-normalized axis: with zeros z = 1 / rc and poles p = 1 / rb of
#   R(x) = scale prod(x - z) / prod(x - p) ~= x^-b,  x = l / l_max,
# the partial fractions are R(x) = scale + sum_j rho_j / (x - p_j), and in l
#   c0 = scale l_max^-b,  B_j = rho_j l_max^(1 - b),  e_j = -l_max p_j.
# A degenerate approximation (non-finite or non-positive term) cannot be a
# covariance and stops the assembly.
.spde_rational_terms <- function(nu, order, kappa, l_max, d = 2) {
  alpha <- nu + d / 2
  m <- as.integer(floor(alpha))
  b <- alpha - m
  spectrum_ratio <- min(kappa^2 / l_max, 0.5)
  rr <- .spde_rational_roots(as.integer(order), b, spectrum_ratio)
  z <- 1 / rr$rc
  p <- 1 / rr$rb
  rho <- vapply(seq_along(p), function(j) {
    rr$scale * prod(p[j] - z) / prod(p[j] - p[-j])
  }, numeric(1))
  terms <- list(m = m, b = b, c0 = rr$scale * l_max^(-b),
                B = rho * l_max^(1 - b), e = -l_max * p,
                spectrum_ratio = spectrum_ratio, error = rr$error)
  ok <- is.finite(terms$c0) && terms$c0 > 0 &&
    all(is.finite(terms$B)) && all(terms$B > 0) &&
    all(is.finite(terms$e)) && all(terms$e > 0)
  if (!ok) {
    stop(sprintf(paste0(
      "The order-%d rational approximation of the fractional SPDE operator ",
      "is degenerate on the spectrum ratio %.3g (a pole or residue is not ",
      "positive), so it does not define a covariance; use a lower ",
      "`rational_order`."), as.integer(order), spectrum_ratio), call. = FALSE)
  }
  terms
}

# Assemble the fractional SPDE field from FEM matrices as a sum of independent
# GMRF components, one per term of `.spde_rational_terms()`. With the lumped
# mass C (diagonal `C0`), K = kappa^2 C + G and the discrete operator
# L = C^-1 K, the field covariance tau^-2 L^-alpha C^-1 is approximated by
#   u = u_0 + sum_j u_j,
#   Q_0 = (tau^2 / c0)  K (C^-1 K)^(m - 1),
#   Q_j = (tau^2 / B_j) (K + e_j C) C^-1 K (C^-1 K)^(m - 1),
# each u_k independent with precision Q_k. The latent is the stack
# x = (u_0, u_1, ...), its precision the block diagonal of the Q_k, and the
# field u = Pr x with Pr = [I, I, ...]. Every block is a polynomial of degree at
# most m + 1 in L with positive coefficients, so each is as well conditioned as
# the integer-order precision of that degree, and the Laplace marginal is formed
# in precision space like the integer path's. `l_max` bounds the spectrum of L
# from above (`.spde_operator_lambda_max()` plus kappa^2).
.spde_rational_assemble <- function(C0, G, kappa, tau, nu, order,
                                    d = 2, l_max) {
  n  <- length(C0)
  Cm <- Matrix::Diagonal(n, C0)
  Ci <- Matrix::Diagonal(n, 1 / C0)
  K  <- kappa^2 * Cm + G
  terms <- .spde_rational_terms(nu, order, kappa, l_max, d = d)

  base <- K
  if (terms$m > 1L) for (i in seq_len(terms$m - 1L)) base <- base %*% Ci %*% K
  sym <- function(M) as(Matrix::forceSymmetric(M), "CsparseMatrix")
  blocks <- c(
    list(sym((tau^2 / terms$c0) * base)),
    lapply(seq_along(terms$B), function(j) {
      sym((tau^2 / terms$B[j]) * (K + terms$e[j] * Cm) %*% Ci %*% base)
    }))
  n_comp <- length(blocks)
  Q  <- Matrix::bdiag(blocks)
  Pr <- do.call(cbind, rep(list(Matrix::Diagonal(n)), n_comp))
  list(Q = as(Matrix::forceSymmetric(Q), "CsparseMatrix"),
       Pr = as(Pr, "CsparseMatrix"), terms = terms, n_comp = n_comp,
       l_max = l_max, spectrum_ratio = terms$spectrum_ratio)
}

# Upper bound on the spectrum of C^-1 G over the mesh nodes that carry mass,
# by Gershgorin's theorem: every eigenvalue of C^-1 G lies below
# max_i sum_j |G_ij| / C_i. It is a property of the mesh, independent of
# (range, sigma); the operator L = kappa^2 I + C^-1 G then has
# l_max = kappa^2 + this bound, which the rational approximation interval
# covers.
.spde_operator_lambda_max <- function(C0, G) {
  keep <- which(C0 > 1e-12)
  Gk <- abs(as(G[keep, keep, drop = FALSE], "CsparseMatrix"))
  max(Matrix::rowSums(Gk) / C0[keep])
}
