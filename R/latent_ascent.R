# Block-coordinate ascent for a grouped (community) model carrying a shared
# Gaussian Markov random field and / or latent factors on one arm.
#
# The linear predictor of cell (site i, group s) is
#
#   eta_{s,i} = X_i . (mu + b_s) + sum_k W[i,k] F[u(i),k]
#                                + sum_q lambda_{s,q} zeta_{q,i}
#
# with b_s ~ N(0, Sigma) the per-group coefficient deviations, F the field(s)
# (an all-ones weight column for an intercept field, covariate values for a
# varying-coefficient field) and (zeta, lambda) the per-site latent factors and
# their per-group loadings.
#
# The latent structure enters each group's log-likelihood as a fixed per-cell
# offset, so a coefficient update is an ordinary grouped Laplace-EM supplied by
# the caller (`em_fit`); the field and factor updates given the coefficients are
# self-contained Newton solves. Alternating the blocks converges to the joint
# mode without a finite-difference Hessian over an O(n_sites) field.
#
# The family enters through ONE callback, the oracle `make_oracle(em)` returns:
#
#   working(eta)  (required) list(score, curv), each [n_sites x n_groups]: the
#                 score and curvature of the family's marginal log-likelihood
#                 with respect to an additive offset on the arm's predictor.
#                 Poisson gives (y - mu, mu), a Bernoulli (y - p, p (1 - p)).
#   data_ll(eta)  (required) the summed data log-likelihood.
#   ll_cell(eta, idx = NULL)
#                 (required with factors) the [n_sites x n_groups] matrix of
#                 per-cell marginal log-likelihoods; `idx` (a subset of sites)
#                 may be honoured to evaluate only those rows, and must then
#                 return just those rows.
#   cov_curv(eta) (optional) the curvature used for the field's Laplace
#                 covariance in the precision M-step, when a family stabilises
#                 it away from the Newton curvature; defaults to working()$curv.
#
# The field geometry enters only through a precision builder and a few index
# objects (see `tulpa_latent_ascent()`), so ICAR, proper CAR, SPDE and BYM2
# fields share the same solver.


# ---------------------------------------------------------------------------
# Factor loadings by marginal likelihood
# ---------------------------------------------------------------------------

# The factors are a latent, zeta_i ~ N(0, I_Q). Holding them at their joint
# mode makes (zeta, lambda) a joint-likelihood estimate with n_sites * Q
# incidental parameters (the Neyman-Scott situation, in which the joint mode is
# inconsistent): the site factors' estimation error lands in the fitted
# co-occurrence and the loadings absorb it. The joint-mode objective is also
# unbounded in the magnitude direction (zeta -> 0, lambda -> Inf at fixed
# product drives the N(0, I) prior cost to zero). The loadings are therefore
# estimated from the joint site marginal with the factors integrated out; the
# joint-mode update and the 1-D magnitude search below only initialise that.

# Adaptive Gauss-Hermite grid for the JOINT site marginal, integrating zeta_i
# over all Q dimensions with the groups at a site kept together:
#
#   L_i = integral prod_s f(y_is | eta_is + lambda_s' z) N(z; 0, I_Q) dz
#
# on a tensor Gauss-Hermite rule. Returns the per-site node positions and the
# unnormalised per-node log terms: their log-sum-exp is the marginal, and
# normalised over the nodes they are the posterior p(z_i | y_i) the loading
# M-step averages over.
#
# The nodes are placed at the mode and curvature of each site's own integrand.
# Prior-scale nodes are accurate only when the per-site likelihood is flat
# against the N(0, I) prior (a Bernoulli site); a Poisson site with counts of a
# few carries a sharply peaked, shifted integrand that a handful of prior-scale
# nodes does not resolve, and that quadrature error moves the magnitude argmax.
# When the mode-find cannot proceed (a non-finite score far from the mode, which
# the count marginals can return) the rule falls back to the prior-scale one,
# the zhat = 0, A = I special case of the same construction.
.latent_joint_grid <- function(oracle, eta_base, lambda, gh, adapt = TRUE) {
  Ns <- nrow(eta_base); Qk <- ncol(lambda)
  nd <- as.matrix(expand.grid(rep(list(gh$nodes), Qk)))
  lw <- rowSums(as.matrix(expand.grid(rep(list(log(gh$weights)), Qk))))

  zhat <- matrix(0, Ns, Qk)
  A <- array(0, dim = c(Ns, Qk, Qk))
  for (a in seq_len(Qk)) A[, a, a] <- 1
  logdetA <- numeric(Ns)

  if (isTRUE(adapt)) {
    # h_i(z) = sum_s ll_cell(eta_is + lambda_s' z) - z'z / 2, the objective
    # whose mode and curvature the nodes adapt to. The Newton Hessian entry
    # (a, b) is sum_s curv_is lambda_sa lambda_sb + delta_ab.
    # `idx` lets an idx-aware oracle skip the sites this call does not need; an
    # oracle that ignores it evaluates every site and the slice below still
    # returns the right length(idx) rows.
    hval <- function(z, idx = NULL) {
      cells <- oracle$ll_cell(eta_base + tcrossprod(z, lambda), idx = idx)
      zz <- if (is.null(idx)) z else z[idx, , drop = FALSE]
      v <- rowSums(cells) - 0.5 * rowSums(zz^2)
      ifelse(is.finite(v), v, -Inf)
    }
    # A Newton step is not guaranteed to ascend, and an exp-linked marginal
    # overflows on an undamped overshoot, which would place every node at an
    # arbitrary point. Backtrack per site: halve only the sites whose step did
    # not improve, re-evaluating just the still-pending sites each round.
    ok <- TRUE
    h0 <- hval(zhat)
    if (any(!is.finite(h0))) ok <- FALSE
    for (it in seq_len(12L)) {
      if (!ok) break
      wk <- oracle$working(eta_base + tcrossprod(zhat, lambda))
      if (any(!is.finite(wk$score)) || any(!is.finite(wk$curv))) { ok <- FALSE; break }
      G <- wk$score %*% lambda - zhat
      step <- matrix(0, Ns, Qk)
      for (i in seq_len(Ns)) {
        Hi <- matrix(0, Qk, Qk)
        for (a in seq_len(Qk)) for (b in seq_len(Qk)) {
          Hi[a, b] <- sum(wk$curv[i, ] * lambda[, a] * lambda[, b]) +
            (if (a == b) 1 else 0)
        }
        step[i, ] <- tryCatch(solve(Hi, G[i, ]),
                              error = function(e) rep(NA_real_, Qk))
      }
      step[!is.finite(step)] <- 0
      tt <- rep(1, Ns)
      moved <- rep(FALSE, Ns)
      for (bt in seq_len(25L)) {
        pend <- !moved & (rowSums(abs(step)) > 0)
        if (!any(pend)) break
        idx <- which(pend)
        cand <- zhat + step * tt
        hc <- hval(cand, idx = idx)
        take <- is.finite(hc) & (hc > h0[idx])
        if (any(take)) {
          ti <- idx[take]
          zhat[ti, ] <- cand[ti, , drop = FALSE]
          h0[ti] <- hc[take]
          moved[ti] <- TRUE
        }
        tt[idx[!take]] <- tt[idx[!take]] / 2
      }
      if (max(abs(step * tt)) < 1e-8) break
    }
    if (ok) {
      wk <- oracle$working(eta_base + tcrossprod(zhat, lambda))
      ok <- all(is.finite(wk$score)) && all(is.finite(wk$curv))
    }
    if (ok) {
      # A_i = H_i^{-1/2} via the Cholesky H = L L', A = L^{-T}: then A A' =
      # H^{-1} and log|det A| = -sum(log(diag(L))).
      for (i in seq_len(Ns)) {
        Hi <- matrix(0, Qk, Qk)
        for (a in seq_len(Qk)) for (b in seq_len(Qk)) {
          Hi[a, b] <- sum(wk$curv[i, ] * lambda[, a] * lambda[, b]) +
            (if (a == b) 1 else 0)
        }
        Li <- tryCatch(chol(Hi), error = function(e) NULL)
        if (is.null(Li)) { ok <- FALSE; break }
        A[i, , ] <- backsolve(Li, diag(Qk))       # L^{-T}, since chol() is upper
        logdetA[i] <- -sum(log(diag(Li)))
      }
    }
    if (!ok) {
      zhat[] <- 0
      A[] <- 0
      for (a in seq_len(Qk)) A[, a, a] <- 1
      logdetA[] <- 0
    }
  }

  cphi <- -0.5 * Qk * log(2 * pi)
  Z <- vector("list", nrow(nd))
  L <- vector("list", nrow(nd))
  # The per-node data log-likelihood is kept: the loading M-step's starting
  # objective is this same quantity at the same nodes, so returning it saves
  # re-running the oracle over every node.
  D <- vector("list", nrow(nd))
  for (k in seq_len(nrow(nd))) {
    Zk <- zhat
    for (b in seq_len(Qk)) Zk <- Zk + A[, , b] * nd[k, b]
    Z[[k]] <- Zk
    D[[k]] <- rowSums(oracle$ll_cell(eta_base + tcrossprod(Zk, lambda)))
    L[[k]] <- lw[k] + D[[k]] +
      (cphi - 0.5 * rowSums(Zk^2)) - (cphi - 0.5 * sum(nd[k, ]^2))
  }
  list(Z = Z, logterm = L, logdetA = logdetA, ll = D)
}

# The joint site marginal: log-sum-exp of the grid's per-node log terms.
.latent_joint_marginal <- function(oracle, eta_base, lambda, gh, adapt = TRUE) {
  g <- .latent_joint_grid(oracle, eta_base, lambda, gh, adapt = adapt)
  m <- Reduce(pmax, g$logterm)
  sum(g$logdetA + m +
        log(Reduce(`+`, lapply(g$logterm, function(z) exp(z - m)))))
}

# The factor scale: the scalar c maximising the joint marginal at loadings
# c * lambda. It initialises the loading EM on the first outer pass, where the
# joint-mode direction it is handed can be an order of magnitude out, by a
# GLOBAL search over an expanding bracket that a local ascent cannot perform.
#
# It has to be the JOINT marginal. One group's own marginal integrates zeta to
# one dimension and depends on lambda only through ||lambda_s||, but a single
# Bernoulli per site carries no replication to see overdispersion: a
# normal-mixed logit is nearly a rescaled logit, leaving the magnitude and the
# coefficient scale confounded along a ridge. The magnitude is identified by
# zeta_i being SHARED across groups at a site, i.e. by the cross-group
# co-occurrence only the joint integral sees. One shared scalar, because a
# per-group magnitude is weakly identified at this stage; the loading EM then
# refines every loading.
#
# The bracket expands rather than clamping, so the result is an argmax and not
# a window edge; the "saturated" attribute reports when even the widened
# bracket did not close.
.latent_factor_scale <- function(oracle, eta_base, lambda, gh) {
  if (!is.function(oracle$ll_cell)) {
    stop("the latent factor magnitude needs the oracle's `ll_cell(eta)` ",
         "returning an [n_sites x n_species] matrix of per-cell marginal ",
         "log-likelihoods; this oracle supplies none.", call. = FALSE)
  }
  if (all(sqrt(rowSums(lambda^2)) < 1e-8)) return(1)

  prof_at <- function(g)
    .latent_joint_marginal(oracle, eta_base, g * lambda, gh)

  lo <- 0.2; hi <- 1.5
  grid <- exp(seq(log(lo), log(hi), length.out = 14L))
  prof <- vapply(grid, prof_at, numeric(1))
  saturated <- FALSE
  for (expand in seq_len(6L)) {
    j <- which.max(prof)
    if (j != 1L && j != length(grid)) break
    if (j == 1L) {
      lo <- lo / 4
      add <- exp(seq(log(lo), log(grid[1L]), length.out = 5L))[-5L]
      grid <- c(add, grid); prof <- c(vapply(add, prof_at, numeric(1)), prof)
    } else {
      hi <- hi * 4
      add <- exp(seq(log(grid[length(grid)]), log(hi), length.out = 5L))[-1L]
      grid <- c(grid, add); prof <- c(prof, vapply(add, prof_at, numeric(1)))
    }
    if (expand == 6L) saturated <- TRUE
  }
  j <- which.max(prof)
  if (j == 1L || j == length(grid)) saturated <- TRUE
  out <- if (j == 1L || j == length(grid)) {
    grid[j]
  } else {
    x <- grid[(j - 1L):(j + 1L)]; y <- prof[(j - 1L):(j + 1L)]
    d <- (y[1] - y[2]) * (x[2] - x[3]) - (y[2] - y[3]) * (x[1] - x[2])
    if (!is.finite(d) || abs(d) < 1e-12) {
      grid[j]
    } else {
      v <- x[2] - 0.5 * ((x[2] - x[1])^2 * (y[2] - y[3]) -
                         (x[2] - x[3])^2 * (y[2] - y[1])) / d
      if (is.finite(v) && v >= x[1] && v <= x[3]) v else grid[j]
    }
  }
  attr(out, "saturated") <- saturated
  out
}

# The loadings by MARGINAL maximum likelihood: an EM on the joint site
# marginal over all S * Q loadings (Dempster, Laird & Rubin 1977). The E-step is
# the posterior p(z_i | y_i) on the adaptive grid; the M-step maximises the
# expected complete-data log-likelihood, which separates across groups into a
# Q-dimensional weighted Newton per group with the nodes as design rows.
#
# Warm-started at the scale step's answer: the marginal is not concave in
# lambda, and the 1-D bracket is a cheap global pass over the magnitude, the
# direction a joint-mode iterate is grossly wrong in. This EM then refines
# magnitude and direction locally.
#
# `center` centres the loadings across groups after each M-step (used when a
# shared field is also present, so the field owns the shared spatial mean and
# the factors the between-group differences); the E-step re-adapts to the
# constrained loadings, so the sequence stays a projected ascent.
.latent_factor_mmle <- function(oracle, eta_base, lambda, gh,
                                em.iter = 10L, newton = 2L, tol = 1e-5,
                                center = FALSE) {
  Ns <- nrow(eta_base); S <- ncol(eta_base); Qk <- ncol(lambda)
  zeta0 <- matrix(0, Ns, Qk)
  # Collapsed loadings carry no latent structure, so the offset is zero rather
  # than absent (the driver differences it against the previous pass).
  if (all(sqrt(rowSums(lambda^2)) < 1e-8)) {
    return(list(lambda = lambda, zeta = zeta0, converged = TRUE,
                offset = matrix(0, Ns, S), loglik = NA_real_))
  }

  # The expected complete-data log-likelihood at `lam` over the E-step's fixed
  # nodes and weights: the quantity the M-step Newton step is a step on.
  qobj <- function(lam, Z, W) {
    v <- 0
    for (k in seq_along(Z)) {
      v <- v + sum(W[[k]] *
                     rowSums(oracle$ll_cell(eta_base + tcrossprod(Z[[k]], lam))))
    }
    if (is.finite(v)) v else -Inf
  }

  # The grid is re-adapted every E-step, so successive marginals are not on
  # identical quadrature and the sequence is only near-monotone: carry the best.
  best <- list(ll = -Inf, lambda = lambda, zeta = zeta0)
  ll_prev <- -Inf
  converged <- FALSE

  for (it in seq_len(em.iter)) {
    # ---- E-step: posterior weights over the nodes, and E[z_i | y_i] ----
    g   <- .latent_joint_grid(oracle, eta_base, lambda, gh)
    m   <- Reduce(pmax, g$logterm)
    tot <- Reduce(`+`, lapply(g$logterm, function(z) exp(z - m)))
    ll  <- sum(g$logdetA + m + log(tot))
    if (!is.finite(ll)) break
    W <- lapply(g$logterm, function(z) exp(z - m) / tot)
    if (!all(vapply(W, function(w) all(is.finite(w)), logical(1)))) break

    zeta <- matrix(0, Ns, Qk)
    for (k in seq_along(W)) zeta <- zeta + W[[k]] * g$Z[[k]]
    if (ll > best$ll) best <- list(ll = ll, lambda = lambda, zeta = zeta)
    if (is.finite(ll_prev) && abs(ll - ll_prev) < tol * (abs(ll) + 1)) {
      converged <- TRUE
      break
    }
    ll_prev <- ll

    # ---- M-step: weighted Newton on lambda_s, group by group ----
    # The grid's per-node data log-likelihood IS this objective at the current
    # lambda, accumulated in the order qobj() uses, so the backtracking
    # comparison starts from the identical value.
    q0 <- 0
    for (k in seq_along(W)) q0 <- q0 + sum(W[[k]] * g$ll[[k]])
    if (!is.finite(q0)) q0 <- -Inf
    for (nw in seq_len(newton)) {
      if (!is.finite(q0)) break
      grad <- matrix(0, S, Qk)
      H    <- array(0, dim = c(Qk, Qk, S))
      ok   <- TRUE
      for (k in seq_along(W)) {
        Zk <- g$Z[[k]]; wk <- W[[k]]
        wr <- oracle$working(eta_base + tcrossprod(Zk, lambda))
        if (any(!is.finite(wr$score)) || any(!is.finite(wr$curv))) {
          ok <- FALSE
          break
        }
        # grad[s, a] = sum_ik w_ik score_isk Z_ika
        # H[a, b, s] = sum_ik w_ik curv_isk Z_ika Z_ikb
        grad <- grad + crossprod(wr$score, Zk * wk)
        for (a in seq_len(Qk)) for (b in a:Qk) {
          h <- as.numeric(crossprod(wr$curv, wk * Zk[, a] * Zk[, b]))
          H[a, b, ] <- H[a, b, ] + h
          if (b != a) H[b, a, ] <- H[b, a, ] + h
        }
      }
      if (!ok) break

      D <- matrix(0, S, Qk)
      for (s in seq_len(S)) {
        Hs <- matrix(H[, , s], Qk, Qk)
        D[s, ] <- tryCatch(as.numeric(solve(Hs, grad[s, ])), error = function(e)
          tryCatch(as.numeric(solve(Hs + diag(1e-6, Qk), grad[s, ])),
                   error = function(e2) rep(NA_real_, Qk)))
      }
      if (!all(is.finite(D))) break

      # Exp-linked families overshoot, so backtrack on the M-step objective.
      t <- 1; taken <- FALSE
      while (t > 1e-4) {
        cand <- lambda + t * D
        q <- qobj(cand, g$Z, W)
        if (q > q0) { lambda <- cand; q0 <- q; taken <- TRUE; break }
        t <- t / 2
      }
      if (!taken) break
    }
    if (isTRUE(center)) lambda <- sweep(lambda, 2L, colMeans(lambda), "-")
  }

  # `loglik` is the achieved joint marginal at the returned loadings, so a
  # caller can compare two fits on the objective the estimator ascends.
  list(lambda = best$lambda, zeta = best$zeta, converged = converged,
       offset = .latent_factor_offset(oracle, eta_base, best$lambda, gh),
       loglik = best$ll)
}

# The per-cell offset the coefficient EM conditions on.
#
# The coefficient update maximises sum_is ll_cell(eta_is(beta) + off_is); the
# target is the integrated sum_is E_z[ll_cell(eta_is(beta) + lambda_s' z_i)],
# and no plug-in point offset reproduces it through a nonlinear link (the
# posterior-mean plug-in carries too little latent variance, a prior-spread one
# too much). The SCORE is matched instead: the two objectives have stationary
# conditions
#
#   sum_is  score(eta_is + off_is)        d eta_is / d beta = 0     (plug-in)
#   sum_is  E_z[score(eta_is + lam_s'z)]  d eta_is / d beta = 0     (integrated)
#
# so off_is solving score(eta_is + off_is) = E_z[score(...)] makes them the
# same condition for any family, with no knowledge of the link. For a Poisson
# log link it reduces to off = lambda' zhat + v / 2, the Jensen correction.
.latent_factor_offset <- function(oracle, eta_base, lambda, gh) {
  g <- .latent_joint_grid(oracle, eta_base, lambda, gh)
  m   <- Reduce(pmax, g$logterm)
  tot <- Reduce(`+`, lapply(g$logterm, function(z) exp(z - m)))
  W   <- lapply(g$logterm, function(z) exp(z - m) / tot)

  # The posterior-mean plug-in offset E[lambda_s' z_i | y_i]: needs only the
  # grid, so it is the fallback where working() cannot be evaluated.
  off <- matrix(0, nrow(eta_base), ncol(eta_base))
  for (k in seq_along(W)) off <- off + W[[k]] * tcrossprod(g$Z[[k]], lambda)

  sbar <- matrix(0, nrow(eta_base), ncol(eta_base))
  for (k in seq_along(W)) {
    wr <- oracle$working(eta_base + tcrossprod(g$Z[[k]], lambda))
    if (any(!is.finite(wr$score))) return(off)
    sbar <- sbar + W[[k]] * wr$score
  }

  # Newton on f(off) = score(eta_base + off) - sbar from the plug-in offset,
  # steps capped so a near-flat curvature cannot throw a cell into the region
  # where an exp link overflows.
  for (it in seq_len(30L)) {
    wr <- oracle$working(eta_base + off)
    if (any(!is.finite(wr$score)) || any(!is.finite(wr$curv))) break
    step <- (wr$score - sbar) / pmax(wr$curv, 1e-8)
    step[!is.finite(step)] <- 0
    step <- pmax(pmin(step, 1), -1)
    off <- off + step
    if (max(abs(step)) < 1e-8) break
  }
  off
}


# ---------------------------------------------------------------------------
# Field solve
# ---------------------------------------------------------------------------

# One ridge ladder for every singular-Hessian retry: the field Newton, the field
# covariance and the factor Newton. Tiers are none, relative and then absolute
# against the mean diagonal, which is guarded so a non-finite diagonal cannot
# give a NaN ridge. `g` absent solves for the inverse. A solve failing at every
# tier returns NA (NULL for the inverse) rather than throwing.
.latent_ridge_solve <- function(H, g = NULL) {
  n  <- nrow(H)
  d  <- mean(Matrix::diag(H))
  dd <- if (is.finite(d) && d > 0) d else 1
  out <- function(r) {
    Hr <- if (is.null(r)) H else H + Matrix::Diagonal(n, r)
    if (is.null(g)) Matrix::solve(Hr)
    else if (is.matrix(g)) as.matrix(Matrix::solve(Hr, g))
    else as.numeric(Matrix::solve(Hr, g))
  }
  for (r in list(NULL, 1e-8 * dd, max(1e-4 * dd, 1e-6))) {
    v <- tryCatch(out(r), error = function(e) NULL)
    if (!is.null(v)) return(v)
  }
  if (is.null(g)) NULL
  else if (is.matrix(g)) matrix(NA_real_, nrow(g), ncol(g))
  else rep(NA_real_, length(g))
}

# Sum-to-zero constraint matrix over the first `n` coordinates of a vector of
# length `n_total`: one row per node set in `components`, ones on the set's
# coordinates (shifted by `offset`). NULL when there is no constraint.
.latent_constraint_matrix <- function(components, n_total, offset = 0L) {
  if (!length(components)) return(NULL)
  Cm <- matrix(0, length(components), n_total)
  for (k in seq_along(components)) Cm[k, offset + components[[k]]] <- 1
  Cm
}

# Newton step for the ascent direction H^-1 g restricted to {s : Cm s = 0},
# the exact step on the constrained subspace (conditioning by kriging, Rue and
# Held 2005, sec. 2.3.3): s - H^-1 Cm' (Cm H^-1 Cm')^-1 Cm s, from one solve
# against [g, Cm']. Projecting an unconstrained step onto the subspace instead
# is not a Newton step there, and with several constrained components it can
# cycle and diverge.
.latent_constrained_step <- function(H, g, Cm = NULL) {
  if (is.null(Cm)) return(.latent_ridge_solve(H, g))
  X <- .latent_ridge_solve(H, cbind(g, t(Cm)))
  if (!all(is.finite(X))) return(rep(NA_real_, length(g)))
  s <- X[, 1L]; Z <- X[, -1L, drop = FALSE]
  as.numeric(s - Z %*% solve(Cm %*% Z, Cm %*% s))
}

# Backtracking along a Newton step: the largest t in 1, 1/2, ... (down to
# 2^-19) with obj(t) > f0. 0 when no such t exists (the iterate is held).
.latent_backtrack <- function(obj, f0) {
  t <- 1
  while (t > 1e-6) {
    f <- obj(t)
    if (is.finite(f) && f > f0) return(t)
    t <- t / 2
  }
  0
}

# Centre each column of `Fm` over every node set in `components` (NULL: none).
.latent_center_components <- function(Fm, components) {
  for (cc in components) {
    Fm[cc, ] <- sweep(Fm[cc, , drop = FALSE], 2L,
                      colMeans(Fm[cc, , drop = FALSE]), "-")
  }
  Fm
}

# Newton update for K covariate-weighted fields with prior tau_k Q. `eta_base`
# [n_sites x n_groups] is the arm's predictor from everything except these
# fields. The per-site field contribution is sum_k W[i,k] (M' F)[i,k]; joint
# Newton over the K * n_nodes field vector (block (k,l) is
# M diag(W[,k] W[,l] curv) M' + (k==l) tau_k Q), then a closed-form tau M-step.
# Each field is centred over every node set in `components` (one sum-to-zero
# per set; NULL leaves a full-rank field unconstrained), and `rankdef` (the
# number of constraints) is the rank the M-step discounts.
.latent_field_solve <- function(oracle, eta_base, Q, Fm, tau, W, M = NULL,
                                max_iter = 50L, tol = 1e-8,
                                components = list(seq_len(nrow(Fm))),
                                rankdef = length(components)) {
  Ns <- nrow(eta_base); K <- ncol(W); Nn <- nrow(Fm)
  if (is.null(M)) M <- Matrix::Diagonal(Ns)
  Mt <- Matrix::t(M)
  build_H <- function(cw) {
    blocks <- vector("list", K * K)
    for (k in seq_len(K)) for (l in seq_len(K)) {
      Dw <- M %*% Matrix::Diagonal(x = W[, k] * W[, l] * cw) %*% Mt
      blocks[[(k - 1L) * K + l]] <- if (k == l) Dw + tau[k] * Q else Dw
    }
    do.call(rbind, lapply(seq_len(K), function(k)
      do.call(cbind, blocks[((k - 1L) * K + 1L):(k * K)])))
  }
  field_off_of <- function(Ff) rowSums(W * as.matrix(Mt %*% Ff))
  obj_of <- function(Ff) {
    v <- oracle$data_ll(eta_base + field_off_of(Ff)) -
      0.5 * sum(vapply(seq_len(K), function(k)
        tau[k] * as.numeric(crossprod(Ff[, k], Q %*% Ff[, k])), 0))
    if (is.finite(v)) v else -Inf
  }
  Cm <- if (length(components))
    do.call(rbind, lapply(seq_len(K), function(k)
      .latent_constraint_matrix(components, K * Nn, (k - 1L) * Nn))) else NULL

  # The data term M diag(w) M' has rank at most n_sites; a mesh with more nodes
  # than sites relies on tau Q for full rank, so a small tau can leave H
  # near-singular and the solve goes through the ridge ladder.
  for (it in seq_len(max_iter)) {
    wk <- oracle$working(eta_base + field_off_of(Fm))
    sc <- rowSums(wk$score); cw <- rowSums(wk$curv)
    g  <- unlist(lapply(seq_len(K), function(k)
      as.numeric(M %*% (W[, k] * sc)) - tau[k] * as.numeric(Q %*% Fm[, k])))
    step <- .latent_constrained_step(build_H(cw), g, Cm)
    # A non-finite step is held rather than applied: it would poison F and make
    # the convergence test NA.
    if (!all(is.finite(step))) break
    S  <- matrix(step, Nn, K)
    f0 <- obj_of(Fm)
    t  <- .latent_backtrack(function(t) obj_of(Fm + t * S), f0)
    if (t == 0) break
    Fm <- .latent_center_components(Fm + t * S, components)
    if (t * max(abs(step)) < tol) break
  }

  # tau M-step. The field's Laplace covariance uses cov_curv when the family
  # supplies it (an occupancy marginal can drive the Newton curvature to ~0 at
  # undetected sites, which would blow the covariance up).
  eta_full <- eta_base + field_off_of(Fm)
  cw  <- rowSums(if (is.null(oracle$cov_curv)) oracle$working(eta_full)$curv
                 else oracle$cov_curv(eta_full))
  Hc  <- build_H(cw)
  Cov <- .latent_ridge_solve(Hc)
  tau_new <- numeric(K)
  df <- Nn - as.integer(rankdef)
  for (k in seq_len(K)) {
    idx  <- (k - 1L) * Nn + seq_len(Nn)
    quad <- as.numeric(t(Fm[, k]) %*% (Q %*% Fm[, k])) +
            sum(Matrix::diag(Q %*% Cov[idx, idx, drop = FALSE]))
    # Floored away from zero: with more field nodes than sites the data term is
    # rank-deficient, and a collapsing tau would leave the next H singular.
    tau_new[k] <- max(df / max(quad, 1e-8), 1e-6)
  }
  list(Fm = Fm, tau = tau_new)
}

# Laplace marginal of the field solve at its mode, the objective the field's
# precision hyperparameters are profiled on: the family data log-likelihood plus
# the GMRF prior (0.5 (df log tau + log|Q|) - 0.5 tau f'Q f) minus the Laplace
# normaliser 0.5 log|H|. `logdetQ` is log|Q| per unit tau on the constrained
# subspace (`.latent_logdet()`).
.latent_field_marginal <- function(oracle, eta_base, Q, Fm, tau, W, logdetQ,
                                   rankdef = 0L, M = NULL) {
  Ns <- nrow(eta_base); K <- ncol(W); Nn <- nrow(Fm)
  if (is.null(M)) M <- Matrix::Diagonal(Ns)
  Mt <- Matrix::t(M)
  field_off <- rowSums(W * as.matrix(Mt %*% Fm))
  eta_full  <- eta_base + field_off
  ll  <- oracle$data_ll(eta_full)
  cw  <- rowSums(oracle$working(eta_full)$curv)
  blocks <- vector("list", K * K)
  for (k in seq_len(K)) for (l in seq_len(K)) {
    Dw <- M %*% Matrix::Diagonal(x = W[, k] * W[, l] * cw) %*% Mt
    blocks[[(k - 1L) * K + l]] <- if (k == l) Dw + tau[k] * Q else Dw
  }
  H <- do.call(rbind, lapply(seq_len(K), function(k)
    do.call(cbind, blocks[((k - 1L) * K + 1L):(k * K)])))
  df <- Nn - as.integer(rankdef)
  prior <- 0
  for (k in seq_len(K)) {
    prior <- prior + 0.5 * (df * log(tau[k]) + logdetQ) -
             0.5 * tau[k] * as.numeric(t(Fm[, k]) %*% (Q %*% Fm[, k]))
  }
  ll + prior - 0.5 * as.numeric(Matrix::determinant(H, logarithm = TRUE)$modulus)
}

# log-determinant of a unit-scale precision on the subspace its `n_constr`
# sum-to-zero constraints leave: the full sparse log|Q| when unconstrained, else
# the sum of log of all but the n_constr smallest eigenvalues (the directions
# the constraints remove), each floored at 1e-10.
.latent_logdet <- function(Q, n_constr = 0L) {
  if (n_constr == 0L) {
    return(as.numeric(Matrix::determinant(Q, logarithm = TRUE)$modulus))
  }
  ev <- sort(eigen(as.matrix(Q), symmetric = TRUE, only.values = TRUE)$values,
             decreasing = TRUE)
  sum(log(pmax(ev[seq_len(length(ev) - n_constr)], 1e-10)))
}

# BYM2 field update (one shared field). The combined field phi = a v + b u
# enters eta, with v the structured part (precision `Qic`, sum-to-zero per
# component) and u ~ N(0, I) the unstructured part; a = sigma sqrt(rho / scale),
# b = sigma sqrt(1 - rho). Joint Newton over (v, u) at fixed (sigma, rho); `m`
# is the Laplace field marginal the hyperparameters are profiled on. u is held
# at overall sum zero: its mean enters eta exactly as the intercept does, and
# the intercept carries a flat prior, so the mean is redundant. Left free it is
# a ridge the block-coordinate ascent creeps along (intercept in the coefficient
# EM, field level here), which does not converge in the outer passes it is given.
.latent_bym2_solve <- function(oracle, eta_base, Qic, v, u, a, b,
                               max_iter = 50L, tol = 1e-8,
                               components = list(seq_along(v))) {
  Ns <- nrow(eta_base); Imat <- Matrix::Diagonal(Ns)
  # A tiny ridge regularises the structured null directions (one per connected
  # component, singular when a is small); v is centred per component each step,
  # so the ridged constant directions are removed anyway.
  ridge <- Matrix::Diagonal(2L * Ns, 1e-6)
  build_H <- function(cw, a, b) {
    Dw <- Matrix::Diagonal(x = cw)
    rbind(cbind(a * a * Dw + Qic, a * b * Dw),
          cbind(a * b * Dw,       b * b * Dw + Imat)) + ridge
  }
  obj_of <- function(v, u) {
    val <- oracle$data_ll(eta_base + a * v + b * u) -
      0.5 * as.numeric(crossprod(v, Qic %*% v)) - 0.5 * sum(u * u)
    if (is.finite(val)) val else -Inf
  }
  Cm <- rbind(.latent_constraint_matrix(components, 2L * Ns),
              .latent_constraint_matrix(list(seq_len(Ns)), 2L * Ns, Ns))
  u <- u - mean(u)
  for (it in seq_len(max_iter)) {
    phi <- a * v + b * u
    wk  <- oracle$working(eta_base + phi)
    sc  <- rowSums(wk$score); cw <- rowSums(wk$curv)
    gv  <- a * sc - as.numeric(Qic %*% v)
    gu  <- b * sc - u
    H   <- build_H(cw, a, b)
    step <- .latent_constrained_step(H, c(gv, gu), Cm)
    if (!all(is.finite(step))) break
    sv <- step[seq_len(Ns)]; su <- step[Ns + seq_len(Ns)]
    t  <- .latent_backtrack(function(t) obj_of(v + t * sv, u + t * su),
                            obj_of(v, u))
    if (t == 0) break
    v <- v + t * sv; u <- u + t * su
    v <- as.numeric(.latent_center_components(matrix(v), components))
    u <- u - mean(u)
    if (t * max(abs(step)) < tol) break
  }
  # Laplace marginal at the mode (constant normalisers dropped).
  phi <- a * v + b * u
  eta_full <- eta_base + phi
  cw  <- rowSums(oracle$working(eta_full)$curv)
  H   <- build_H(cw, a, b)
  m   <- oracle$data_ll(eta_full) -
         0.5 * as.numeric(t(v) %*% (Qic %*% v)) - 0.5 * sum(u * u) -
         0.5 * as.numeric(Matrix::determinant(H, logarithm = TRUE)$modulus)
  list(v = v, u = u, m = m)
}

# Profile a field's hyperparameters over the Cartesian grid `axes` (a named
# list, the first axis varying fastest) by the field marginal `cell_fit(hyper)`
# returns as `m`. Each cell is an independent solve from the same start, so the
# grid runs through `tulpa_hyper_grid()`; the cell states are kept in call
# order and the cell with the largest finite marginal is returned (the first
# such cell on a tie). A cell whose solve errors carries no marginal. NULL when
# no cell is finite.
.latent_profile_hyper <- function(axes, cell_fit) {
  cells <- list()
  inner <- function(hyper) {
    r <- tryCatch(cell_fit(hyper), error = function(e) NULL)
    cells[[length(cells) + 1L]] <<- list(hyper = hyper, fit = r)
    list(log_marginal = if (is.null(r)) -Inf else r$m)
  }
  specs <- lapply(names(axes), function(nm) hyper_axis_spec(nm, grid = axes[[nm]]))
  hg <- tulpa_hyper_grid(specs, inner, combine = "none")
  lm <- hg$log_marginal
  if (!any(is.finite(lm))) return(NULL)
  j <- which.max(lm)
  if (length(cells) != nrow(hg$theta_grid) ||
      !isTRUE(all(cells[[j]]$hyper == hg$theta_grid[j, ]))) {
    stop("hyperparameter profile: cell states out of step with the grid.",
         call. = FALSE)
  }
  c(cells[[j]]$fit, list(hyper = cells[[j]]$hyper))
}


# ---------------------------------------------------------------------------
# Latent factor initialisation
# ---------------------------------------------------------------------------

# Joint-mode factor update: refine the per-site factors zeta [n_sites x Q] and
# per-group loadings lambda [n_groups x Q] against `eta_base`. Alternating Newton
# (zeta_i | lambda; lambda_s | zeta) with an N(0, I) factor prior and a weak
# loading ridge; the factor columns are centred and scaled to unit variance each
# pass (the scale folds into lambda). Site i's working weights depend on
# zeta[i, ] alone and group s's on lambda[s, ] alone, so one working() call per
# half-pass is exact. Used only to supply a starting direction.
.latent_factor_update <- function(oracle, eta_base, zeta, lambda,
                                  inner = 15L, center = FALSE) {
  Ns <- nrow(eta_base); S <- ncol(eta_base); Qk <- ncol(zeta)
  ridge <- 1e-3

  # The penalised objective both half-passes ascend; an eta far enough out to
  # overflow an exp link reads as -Inf, so the line search rejects it.
  obj <- function(z, l) {
    v <- oracle$data_ll(eta_base + tcrossprod(z, l)) -
      0.5 * sum(z * z) - ridge / 2 * sum(l * l)
    if (is.finite(v)) v else -Inf
  }

  nstep <- function(H, g) .latent_ridge_solve(H, g)

  # Halve a step until it ascends; hold the previous iterate if it never does.
  # Within a half-pass the updates are mutually independent, so one shared
  # scale is exact for whichever step it accepts.
  ascend <- function(f0, build) {
    t <- 1
    while (t > 1e-4) {
      cand <- build(t)
      f <- obj(cand$zeta, cand$lambda)
      if (f > f0) return(c(cand, list(f = f)))
      t <- t / 2
    }
    NULL
  }

  for (it in seq_len(inner)) {
    f0 <- obj(zeta, lambda)
    if (!is.finite(f0)) break

    wk <- oracle$working(eta_base + tcrossprod(zeta, lambda))
    if (any(!is.finite(wk$score)) || any(!is.finite(wk$curv))) break
    Dz <- matrix(0, Ns, Qk)
    for (i in seq_len(Ns)) {
      g <- as.numeric(crossprod(lambda, wk$score[i, ])) - zeta[i, ]
      H <- crossprod(lambda * sqrt(wk$curv[i, ])) + diag(Qk)
      Dz[i, ] <- nstep(H, g)
    }
    if (all(is.finite(Dz))) {
      cand <- ascend(f0, function(t) list(zeta = zeta + t * Dz, lambda = lambda))
      if (!is.null(cand)) { zeta <- cand$zeta; f0 <- cand$f }
    }

    wk <- oracle$working(eta_base + tcrossprod(zeta, lambda))
    if (any(!is.finite(wk$score)) || any(!is.finite(wk$curv))) break
    Dl <- matrix(0, S, Qk)
    for (s in seq_len(S)) {
      g <- as.numeric(crossprod(zeta, wk$score[, s])) - ridge * lambda[s, ]
      H <- crossprod(zeta * sqrt(wk$curv[, s])) + ridge * diag(Qk)
      Dl[s, ] <- nstep(H, g)
    }
    if (all(is.finite(Dl))) {
      cand <- ascend(f0, function(t) list(zeta = zeta, lambda = lambda + t * Dl))
      if (!is.null(cand)) { lambda <- cand$lambda; f0 <- cand$f }
    }

    for (q in seq_len(Qk)) {
      zeta[, q] <- zeta[, q] - mean(zeta[, q])
      sdq <- stats::sd(zeta[, q])
      if (is.finite(sdq) && sdq > 1e-6) {
        zeta[, q] <- zeta[, q] / sdq; lambda[, q] <- lambda[, q] * sdq
      }
    }
    # With a shared field also present, centre the loadings across groups so the
    # field owns the shared spatial mean and the factors the between-group
    # differences.
    if (isTRUE(center)) lambda <- sweep(lambda, 2L, colMeans(lambda), "-")
  }
  list(zeta = zeta, lambda = lambda)
}

# A principal-factor starting DIRECTION (Lawley & Maxwell 1971): the top-Q
# eigenvectors of the cross-group covariance of the Pearson working residuals
# score / sqrt(curv) at zero factor offset. Pearson rather than Newton residuals
# (score / curv), whose denominator is the curvature itself, so a near-zero
# fitted mean would let a few low-count cells dominate the covariance. Returns
# NULL when the decomposition is not well posed.
.latent_factor_eigen_init <- function(oracle, eta_base, Qk) {
  wk <- oracle$working(eta_base)
  if (any(!is.finite(wk$score)) || any(!is.finite(wk$curv))) return(NULL)
  resid <- wk$score / sqrt(pmax(wk$curv, 1e-8))
  if (!all(is.finite(resid)) || nrow(resid) < 2L) return(NULL)
  resid <- scale(resid, center = TRUE, scale = FALSE)
  Sigma_res <- crossprod(resid) / (nrow(resid) - 1)
  ee <- tryCatch(eigen(Sigma_res, symmetric = TRUE), error = function(e) NULL)
  if (is.null(ee) || length(ee$values) < Qk) return(NULL)
  ev <- ee$values[seq_len(Qk)]
  if (any(!is.finite(ev)) || any(ev <= 1e-8)) return(NULL)
  V <- ee$vectors[, seq_len(Qk), drop = FALSE]
  # A direction only: the magnitude search sets the scale on the marginal, so
  # the pair goes to it directly rather than through the joint-mode ascent.
  lambda <- V
  zeta <- resid %*% V
  for (q in seq_len(Qk)) {
    zeta[, q] <- zeta[, q] - mean(zeta[, q])
    sdq <- stats::sd(zeta[, q])
    if (is.finite(sdq) && sdq > 1e-6) {
      zeta[, q] <- zeta[, q] / sdq
      lambda[, q] <- lambda[, q] * sdq
    }
  }
  if (!all(is.finite(zeta)) || !all(is.finite(lambda))) return(NULL)
  list(zeta = zeta, lambda = lambda)
}

# `k` pseudo-random starting DIRECTIONS for the first-pass candidate search.
# The seed is fixed, so the candidate set (and the fit) is reproducible, and
# the caller's RNG stream is restored afterwards.
.latent_factor_random_starts <- function(Ns, S, Qk, k = 6L, seed = 20250157L) {
  .seed_scoped(seed)
  lapply(seq_len(k), function(i) {
    list(zeta = matrix(stats::rnorm(Ns * Qk), Ns, Qk),
         lambda = matrix(stats::rnorm(S * Qk, 0, 0.3), S, Qk),
         ascend = TRUE)
  })
}


# ---------------------------------------------------------------------------
# Block-coordinate driver
# ---------------------------------------------------------------------------

#' Block-coordinate ascent for a grouped model with a shared field and latent factors
#'
#' Fits a grouped (community) model whose linear predictor on one arm carries a
#' shared Gaussian Markov random field, latent factors, or both:
#' \deqn{\eta_{s,i} = X_i (\mu + b_s) + \sum_k W_{ik} (M^\top F)_{ik} +
#'   \lambda_s^\top \zeta_i,}
#' with \eqn{b_s \sim N(0, \Sigma)} per-group coefficient deviations, \eqn{F}
#' the field(s) on the geometry's nodes and \eqn{\zeta_i \sim N(0, I_Q)}
#' per-site factors with per-group loadings \eqn{\lambda_s}. Each outer pass
#' runs (a) the caller's coefficient EM with the combined latent structure as a
#' per-cell offset, (b) a Newton field solve with a closed-form precision
#' M-step, and (c) the loading update. The loadings are estimated by marginal
#' maximum likelihood: an EM on the joint site marginal with the factors
#' integrated out on an adaptive Gauss-Hermite grid, initialised on the first
#' pass from several candidate directions (a joint-mode ascent from the
#' supplied start, a principal-factor direction, and fixed-seed random
#' restarts), each scaled by a bracketed 1-D magnitude search and selected on
#' the converged marginal. The offset the coefficient EM conditions on matches
#' the integrated score per cell. After the loop the field's precision
#' hyperparameters are profiled over `geometry$hyper_grid` by the Laplace field
#' marginal, through [tulpa_hyper_grid()].
#'
#' @param make_oracle `function(em)` returning the family oracle at the current
#'   coefficients: a list with `working(eta)` (a list of `score` and `curv`,
#'   each `[n_sites x n_species]`: the score and curvature of the marginal
#'   log-likelihood with respect to an additive offset on the arm's predictor),
#'   `data_ll(eta)` (the summed data log-likelihood), `ll_cell(eta, idx = NULL)`
#'   (per-cell marginal log-likelihoods; required when `factors` is given; may
#'   honour `idx` by returning only those site rows) and optionally
#'   `cov_curv(eta)` (the curvature used for the field's Laplace covariance).
#' @param em_fit `function(site_off, fac_off, em_prev)` running the caller's
#'   coefficient EM with the field offset `site_off` (length `n_sites`) and the
#'   factor offset `fac_off` (`[n_sites x n_species]`), warm-started from
#'   `em_prev` (`NULL` on the first pass); returns the EM state.
#' @param offset_of `function(em)` returning the arm's `[n_sites x n_species]`
#'   coefficient predictor at that EM state.
#' @param n_sites,n_species Dimensions of the cell grid.
#' @param geometry `NULL` (no field) or a list describing the field:
#'   * `type` -- a label returned with the result; `"bym2"` selects the BYM2
#'     field \eqn{\phi = a v + b u} (structured \eqn{v}, unstructured
#'     \eqn{u \sim N(0, I)}, \eqn{a = \sigma\sqrt{\rho / scale}},
#'     \eqn{b = \sigma\sqrt{1 - \rho}}), any other value a field with prior
#'     \eqn{\tau_k Q}.
#'   * `Q` -- `function(hyper)` returning the unit-scale sparse precision at the
#'     named numeric hyperparameters `hyper` (the structured precision for
#'     BYM2).
#'   * `hyper` -- named numeric hyperparameters used during the ascent (BYM2:
#'     `sigma`, `rho`); `numeric(0)` when `Q` has none.
#'   * `hyper_grid` -- named list of axis grids profiled after the ascent, the
#'     first axis varying fastest (`NULL` or empty: none).
#'   * `components` -- list of node index sets, each held at sum zero (one per
#'     null direction of an intrinsic precision); `NULL` leaves the field
#'     unconstrained. Its length is the rank deficiency the precision M-step and
#'     the profile log-determinant discount.
#'   * `M` -- the `[n_nodes x n_sites]` node-from-site map (a group incidence or
#'     a projector transpose), or `NULL` for one node per site.
#'   * `W` -- `[n_sites x K]` per-field weights (ones for an intercept field).
#'   * `n_nodes` -- the number of field nodes.
#'   * `scale` -- BYM2 only: the generalised variance the structured part is
#'     divided by.
#' @param factors `NULL` (no factors) or `list(zeta, lambda)`: starting values,
#'   `[n_sites x Q]` and `[n_species x Q]`.
#' @param control List of
#'   * `tol` (`1e-4`) -- convergence tolerance on the largest change of the
#'     field or factor offset between passes.
#'   * `max_outer` (`NULL`) -- outer-pass budget; `NULL` takes `factor_outer`
#'     when factors are present, else 400 (a field run stops at `tol`, so this
#'     is a cap; how many passes it needs depends on how strongly the field
#'     overlaps the coefficient block).
#'   * `factor_outer` (`25`) -- the budget a factor model takes by default (the
#'     factor block alternates with the coefficient block along a slow mode and
#'     rarely meets `tol`).
#'   * `factor_starts` (`8`) -- candidate starting directions on the first pass.
#'   * `n_quad` (`5`) -- Gauss-Hermite nodes per factor dimension.
#'   * `verbose` (`FALSE`), `label` (`"latent"`) -- per-pass progress message
#'     and its prefix.
#'
#' @return A list with
#'   * `em` -- the last coefficient EM state.
#'   * `geometry` -- the geometry as supplied.
#'   * `settings` -- the resolved `max_outer`, `tol`, `factor_outer`,
#'     `factor_starts`, `n_quad` (the factor knobs `NA` without factors).
#'   * `field` (`NULL` without a field) -- `Fm` (`[n_nodes x K]`), `tau`, `type`,
#'     `hyper` (the selected hyperparameters), `site_off` (the per-site field
#'     contribution) and, for BYM2, `bym2 = list(sigma, rho, scale, phi)`.
#'   * `factor` (`NULL` without factors) -- `n_factors`, `zeta`, `lambda`,
#'     `offset` (the score-matched per-cell offset the coefficients were fitted
#'     against), `magnitude_saturated`, `loading_em_converged` and
#'     `marginal_loglik` (the joint marginal at the returned loadings).
#'
#' @seealso [tulpa_re_em()] for a coefficient EM `em_fit` can wrap,
#'   [tulpa_hyper_grid()], [tulpa_gauss_hermite()], [tulpa_car_precision_Q()],
#'   [tulpa_spde_precision_Q()].
#' @examples
#' # Poisson counts with a shared latent factor and one free intercept per
#' # species, the intercepts updated by a closed-form Newton step.
#' set.seed(1)
#' Ns <- 40L; S <- 5L
#' lam <- matrix(rnorm(S, 0, 0.7), S, 1L)
#' y <- matrix(rpois(Ns * S, exp(0.5 + tcrossprod(rnorm(Ns), lam))), Ns, S)
#' oracle <- list(
#'   working = function(eta) list(score = y - exp(eta), curv = exp(eta)),
#'   ll_cell = function(eta, idx = NULL) {
#'     if (!is.null(idx)) { eta <- eta[idx, , drop = FALSE]; yy <- y[idx, , drop = FALSE] }
#'     else yy <- y
#'     yy * eta - exp(eta)
#'   },
#'   data_ll = function(eta) sum(y * eta - exp(eta)))
#' em_fit <- function(site_off, fac_off, em_prev) {
#'   b <- if (is.null(em_prev)) rep(0, S) else em_prev
#'   for (it in 1:20) b <- b + colSums(y - exp(outer(rep(1, Ns), b) + fac_off)) /
#'     colSums(exp(outer(rep(1, Ns), b) + fac_off))
#'   b
#' }
#' res <- tulpa_latent_ascent(
#'   make_oracle = function(em) oracle, em_fit = em_fit,
#'   offset_of = function(b) outer(rep(1, Ns), b), n_sites = Ns, n_species = S,
#'   factors = list(zeta = matrix(scale(cos(seq_len(Ns)))[, 1], Ns, 1L),
#'                  lambda = matrix(0.1, S, 1L)),
#'   control = list(max_outer = 3L, factor_starts = 2L))
#' cor(res$factor$lambda, lam)
#' @export
tulpa_latent_ascent <- function(make_oracle, em_fit, offset_of,
                                n_sites, n_species,
                                geometry = NULL, factors = NULL,
                                control = list()) {
  known <- c("tol", "max_outer", "factor_outer", "factor_starts", "n_quad",
             "verbose", "label")
  bad <- setdiff(names(control), known)
  if (length(bad)) {
    stop("tulpa_latent_ascent(): unknown control key(s): ",
         paste(bad, collapse = ", "), call. = FALSE)
  }
  if (is.null(geometry) && is.null(factors)) {
    stop("tulpa_latent_ascent(): give a field `geometry`, `factors`, or both.",
         call. = FALSE)
  }
  S  <- as.integer(n_species)
  Ns <- as.integer(n_sites)
  has_field  <- !is.null(geometry)
  has_factor <- !is.null(factors)
  tol          <- control[["tol"]] %||% 1e-4
  factor_outer <- control[["factor_outer"]] %||% 25L
  verbose      <- isTRUE(control[["verbose"]])
  label        <- control[["label"]] %||% "latent"

  # A field run stops at `tol`, so its budget is a cap set well above what it
  # takes: how many passes it needs grows with how much the field overlaps the
  # coefficient block (a BYM2 field's iid part overlaps a site covariate's
  # effect, and takes tens to a few hundred). The factor block alternates with
  # the coefficient block along a slow mode and takes `factor_outer` passes
  # unless the caller sets `max_outer`.
  max_outer <- control[["max_outer"]]
  if (is.null(max_outer)) max_outer <- if (has_factor) factor_outer else 400L
  max_outer <- as.integer(max_outer)

  # Each candidate direction costs a full loading-EM run on the first pass.
  factor_starts <- control[["factor_starts"]]
  if (is.null(factor_starts)) factor_starts <- 8L
  factor_starts <- max(1L, as.integer(factor_starts))

  n_quad <- control[["n_quad"]]
  if (is.null(n_quad)) n_quad <- 5L
  n_quad <- max(1L, as.integer(n_quad))

  # ---- field setup ----
  Ffield <- NULL; tau <- NULL; Q <- NULL
  is_bym2 <- FALSE
  hyper <- numeric(0)
  components <- NULL
  if (has_field) {
    is_bym2 <- identical(geometry$type, "bym2")
    hyper <- geometry$hyper %||% numeric(0)
    components <- geometry$components
    Q <- geometry$Q(hyper)
    if (is_bym2) {
      Qic <- Q
      bym_scale <- geometry$scale
      bym_v <- numeric(geometry$n_nodes); bym_u <- numeric(geometry$n_nodes)
      bym_sigma <- hyper[["sigma"]]; bym_rho <- hyper[["rho"]]
      bym_phi <- numeric(geometry$n_nodes)
    } else {
      Ffield <- matrix(0, geometry$n_nodes, ncol(geometry$W))
      tau <- rep(1, ncol(geometry$W))
    }
  }
  Mt_field <- if (has_field && !is.null(geometry$M)) Matrix::t(geometry$M) else NULL
  site_field_off <- function(Ff) {
    Fs <- if (is.null(Mt_field)) Ff else as.matrix(Mt_field %*% Ff)
    rowSums(geometry$W * Fs)
  }
  cur_site_off <- function() {
    if (!has_field) numeric(Ns) else if (is_bym2) bym_phi else site_field_off(Ffield)
  }

  # ---- factor setup ----
  zeta <- factors$zeta; lambda <- factors$lambda
  # (zeta, lambda) seeds the first pass only; (zeta_hat, lambda_hat), the
  # marginal-likelihood pair, is the loop's only carried factor state. Running
  # the joint-mode update alongside it every pass does not converge: the
  # coefficient EM conditions on E[z | y], whose spread is shrunk, the update
  # grows lambda to explain the missing variance, and the pair ratchets.
  lambda_hat <- lambda
  zeta_hat   <- zeta
  fac_offset <- matrix(0, Ns, S)
  gh_joint <- if (has_factor) gauss_hermite_prob(n_quad) else NULL
  fac_saturated <- FALSE
  fac_em_converged <- FALSE
  fac_loglik <- NA_real_

  em <- NULL
  for (outer in seq_len(max_outer)) {
    site_off <- cur_site_off()
    fac_off  <- fac_offset
    # (a) coefficient EM given the combined latent offset.
    em <- em_fit(site_off, fac_off, em)
    eta_coef <- offset_of(em)
    oracle   <- make_oracle(em)
    delta <- 0
    # (b) field update (its base holds the coefficients + the factor part).
    if (has_field && is_bym2) {
      a  <- bym_sigma * sqrt(bym_rho / bym_scale); b <- bym_sigma * sqrt(1 - bym_rho)
      br <- .latent_bym2_solve(oracle, eta_coef + fac_off, Qic,
                               bym_v, bym_u, a, b, components = components)
      phi_new <- a * br$v + b * br$u
      delta <- max(delta, max(abs(phi_new - bym_phi)))
      bym_v <- br$v; bym_u <- br$u; bym_phi <- phi_new
    } else if (has_field) {
      fu <- .latent_field_solve(oracle, eta_coef + fac_off, Q, Ffield, tau,
                                geometry$W, M = geometry$M,
                                components = components)
      delta <- max(delta, max(abs(fu$Fm - Ffield))); Ffield <- fu$Fm; tau <- fu$tau
    }
    # (c) factor update (its base holds the coefficients + the field part).
    if (has_factor) {
      base_fac <- eta_coef + matrix(cur_site_off(), Ns, S)
      if (outer == 1L) {
        # The marginal's gradient in lambda vanishes at lambda = 0, so the
        # loading EM needs a non-degenerate start. A local ascent inherits the
        # basin of its starting DIRECTION, and rescaling a bad direction does
        # not escape it, so several directions are tried (the supplied start
        # through the joint-mode ascent, a principal-factor direction, fixed-seed
        # random restarts), each run through the magnitude search and the
        # loading EM to convergence, and the one with the highest converged
        # marginal is kept. The comparison is on the converged marginal because
        # the candidates' unconverged values do not rank the same way.
        starts <- list(list(zeta = zeta, lambda = lambda, ascend = TRUE))
        if (factor_starts >= 2L) {
          eig <- .latent_factor_eigen_init(oracle, base_fac, ncol(lambda))
          if (!is.null(eig)) starts <- c(starts, list(c(eig, list(ascend = FALSE))))
        }
        n_rand <- max(0L, factor_starts - 2L)
        if (n_rand > 0L) {
          starts <- c(starts, .latent_factor_random_starts(
            Ns, S, ncol(lambda), k = n_rand))
        }

        cand <- NULL
        for (st in starts) {
          gu  <- if (isTRUE(st$ascend)) {
            .latent_factor_update(oracle, base_fac, st$zeta, st$lambda,
                                  center = has_field)
          } else {
            st
          }
          sc   <- .latent_factor_scale(oracle, base_fac, gu$lambda, gh_joint)
          lam0 <- gu$lambda * as.numeric(sc)
          sat  <- isTRUE(attr(sc, "saturated"))
          mm_st <- .latent_factor_mmle(oracle, base_fac, lam0, gh_joint,
                                       center = has_field)
          m <- mm_st$loglik
          if (is.null(cand) ||
              (is.finite(m) && (!is.finite(cand$m) || m > cand$m))) {
            cand <- list(m = m, mm = mm_st, saturated = sat)
          }
        }
        if (isTRUE(cand$saturated)) fac_saturated <- TRUE
        mm <- cand$mm
      } else {
        mm <- .latent_factor_mmle(oracle, base_fac, lambda_hat, gh_joint,
                                  center = has_field)
      }
      lambda_hat <- mm$lambda
      zeta_hat   <- mm$zeta
      fac_offset <- mm$offset
      fac_em_converged <- isTRUE(mm$converged)
      fac_loglik <- mm$loglik
      delta <- max(delta, max(abs(fac_offset - fac_off)))
    }
    if (verbose) {
      message(sprintf("[%s latent %d] delta=%.2e%s", label, outer, delta,
                      if (has_factor)
                        sprintf("  |lambda_hat|=%.3f  sd(zeta_hat)=%.3f",
                                sqrt(sum(lambda_hat^2)),
                                stats::sd(as.numeric(zeta_hat)))
                      else ""))
    }
    if (outer > 2L && delta < tol) break
  }

  # ---- field hyperparameter profile at the converged coefficients ----
  axes <- if (has_field) geometry$hyper_grid else NULL
  if (has_field && length(axes)) {
    eta_coef <- offset_of(em)
    oracle   <- make_oracle(em)
    base_f   <- eta_coef + (if (has_factor) fac_offset else 0)
    n_constr <- length(components)
    if (is_bym2) {
      best <- .latent_profile_hyper(axes, function(h) {
        a <- h[["sigma"]] * sqrt(h[["rho"]] / bym_scale)
        b <- h[["sigma"]] * sqrt(1 - h[["rho"]])
        br <- .latent_bym2_solve(oracle, base_f, Qic, bym_v, bym_u,
                                 a, b, components = components)
        c(br, list(a = a, b = b))
      })
      if (!is.null(best)) {
        bym_v <- best$v; bym_u <- best$u
        bym_sigma <- best$hyper[["sigma"]]; bym_rho <- best$hyper[["rho"]]
        bym_phi <- best$a * best$v + best$b * best$u
        hyper[names(best$hyper)] <- best$hyper
      }
    } else {
      best <- .latent_profile_hyper(axes, function(h) {
        Qh <- geometry$Q(h)
        ld <- .latent_logdet(Qh, n_constr)
        fr <- .latent_field_solve(oracle, base_f, Qh, Ffield, tau, geometry$W,
                                  M = geometry$M, components = components)
        mm <- .latent_field_marginal(oracle, base_f, Qh, fr$Fm, fr$tau,
                                     geometry$W, ld, rankdef = n_constr,
                                     M = geometry$M)
        list(m = mm, Fm = fr$Fm, tau = fr$tau)
      })
      if (!is.null(best)) {
        Ffield <- best$Fm; tau <- best$tau
        hyper[names(best$hyper)] <- best$hyper
      }
    }
  }

  list(em = em, geometry = geometry,
       # The settings the fit ran under; the factor knobs read NA without a
       # factor block, which they do not act on.
       settings = list(
         max_outer = max_outer, tol = tol,
         factor_outer  = if (has_factor) as.integer(factor_outer) else NA_integer_,
         factor_starts = if (has_factor) factor_starts else NA_integer_,
         n_quad        = if (has_factor) n_quad else NA_integer_),
       field = if (!has_field) NULL else list(
         Fm = Ffield, tau = tau, type = geometry$type, hyper = hyper,
         site_off = cur_site_off(),
         bym2 = if (is_bym2) list(sigma = bym_sigma, rho = bym_rho,
                                  scale = bym_scale, phi = bym_phi) else NULL),
       factor = if (!has_factor) NULL else list(
         n_factors = ncol(lambda), zeta = zeta_hat, lambda = lambda_hat,
         offset = fac_offset,
         # TRUE when the initialising magnitude search hit the end of even its
         # widened bracket: the loadings, and so the residual covariance, are
         # then suspect in scale (the row-normalised correlation survives).
         magnitude_saturated = fac_saturated,
         # TRUE when the last loading EM met its tolerance.
         loading_em_converged = fac_em_converged,
         marginal_loglik = fac_loglik))
}
