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
# (gcol33/tulpa#71); it is self-contained and not yet wired into any fitter.

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


# Rational-SPDE coefficients (gcol33/tulpa#71, stage 2; gated -- not yet wired
# into a fitter). For a Matern field of operator order alpha = nu + d/2 with
# beta = alpha / 2, the rSPDE operator-based construction
# (Bolin & Kirchner 2020) assembles
#   Pl = C (CiL)^{m_beta-1} prod_i (I - CiL * rb_i),   m_beta = max(1, floor(beta))
#   Pr =                    prod_i (I - CiL * rc_i)
#   Q  = Pl' Ci Pl,   x ~ N(0, Q^{-1}),   u = Pr x
# whose field-mode variance (in the C-inner-product eigenbasis, CiL phi = l phi)
# is [prod(1 - l rc)]^2 / [l^{2(m_beta-1)} prod(1 - l rb)^2]. Matching the Matern
# spectral density l^{-2 beta} requires
#   prod(1 - l rc) / [l^{m_beta-1} prod(1 - l rb)] ~ l^{-beta},
# i.e. a rational approximation of l^{-beta_rem} (beta_rem = beta - (m_beta-1)) on
# the scaled spectrum. BRASIL gives the best such (m, m) rational; its zeros map
# to rc (= 1/zero) and poles to rb (= 1/pole), with a scale constant. Validated
# against the Matern spectral density in test-spde-rational.R.
#
# `spectrum_ratio` = l_min / l_max of CiL (the generalized eigenvalues); the
# approximation interval is [spectrum_ratio, 1] on the L / l_max-normalized axis.
.spde_rational_roots <- function(order, beta, spectrum_ratio, tol = 1e-7) {
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
