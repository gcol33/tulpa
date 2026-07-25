# Standalone check of the exact Laplace HESSIAN derived in
# dev_notes/laplace_exact_hessian.md. Everything is built from scratch in base R
# -- no tulpa code -- so agreement with a finite difference of the analytic
# gradient (itself checked in proto_exact_gradient.R) is evidence about the MATH.
#
# Model: y ~ family(eta), eta = X beta + sum_m Z_m b_m, b_{m,g} ~ N(0, Sigma_m).
# Latent x = (beta, b_1, b_2, ...), each b_m group-major within its block.
# Multiple RE blocks are supported so the cross-block Hessian coupling -- which
# the gradient has no analogue for -- is exercised, not just the diagonal.

set.seed(11)

# ---- families: loglik and its first FOUR eta-derivatives --------------------
fam_poisson <- list(
  ll = function(y, eta) sum(y * eta - exp(eta) - lgamma(y + 1)),
  d1 = function(y, eta) y - exp(eta),
  d2 = function(y, eta) -exp(eta),
  d3 = function(y, eta) -exp(eta),
  d4 = function(y, eta) -exp(eta),
  kind = "poisson"
)
fam_binom <- list(
  ll = function(y, eta) sum(y * eta - log1p(exp(eta))),
  d1 = function(y, eta) y - plogis(eta),
  d2 = function(y, eta) { p <- plogis(eta); -p * (1 - p) },
  d3 = function(y, eta) { p <- plogis(eta); -p * (1 - p) * (1 - 2 * p) },
  d4 = function(y, eta) { p <- plogis(eta); -(1 - 6 * p + 6 * p^2) * p * (1 - p) },
  kind = "binom"
)
fam_gauss <- function(sd) list(
  ll = function(y, eta) sum(dnorm(y, eta, sd, log = TRUE)),
  d1 = function(y, eta) (y - eta) / sd^2,
  d2 = function(y, eta) rep(-1 / sd^2, length(eta)),
  d3 = function(y, eta) rep(0, length(eta)),
  d4 = function(y, eta) rep(0, length(eta)),
  kind = "gauss"
)

# ---- theta (log-Cholesky) -> L, dSigma/dtheta, d2Sigma/dtheta2 --------------
n_coord <- function(nc, full) as.integer(if (!full) nc else nc * (nc + 1) / 2)

theta_to_L <- function(theta, nc, full) {
  L <- matrix(0, nc, nc)
  if (!full) { diag(L) <- exp(theta); return(L) }
  idx <- 0
  for (j in seq_len(nc)) for (i in j:nc) {
    idx <- idx + 1
    L[i, j] <- if (i == j) exp(theta[idx]) else theta[idx]
  }
  L
}

# First derivatives dL/dtheta_p, with a flag for the diagonal (log) coordinates.
L_first_derivs <- function(L, nc, full) {
  Lp <- list(); is_diag <- logical(0); di <- integer(0)
  if (!full) {
    for (i in seq_len(nc)) {
      dL <- matrix(0, nc, nc); dL[i, i] <- L[i, i]
      Lp[[length(Lp) + 1L]] <- dL; is_diag <- c(is_diag, TRUE); di <- c(di, i)
    }
  } else {
    for (j in seq_len(nc)) for (i in j:nc) {
      dL <- matrix(0, nc, nc)
      if (i == j) { dL[i, i] <- L[i, i]; is_diag <- c(is_diag, TRUE);  di <- c(di, i) }
      else        { dL[i, j] <- 1;       is_diag <- c(is_diag, FALSE); di <- c(di, NA) }
      Lp[[length(Lp) + 1L]] <- dL
    }
  }
  list(Lp = Lp, is_diag = is_diag, di = di)
}

dSigma_dtheta <- function(theta, nc, full) {
  L  <- theta_to_L(theta, nc, full)
  fd <- L_first_derivs(L, nc, full)
  lapply(fd$Lp, function(dL) dL %*% t(L) + L %*% t(dL))
}

# d2Sigma/dtheta_p dtheta_q = L_pq L' + L_p L_q' + L_q L_p' + L L_pq',
# with L_pq nonzero only when p == q is a diagonal (log) coordinate.
d2Sigma_dtheta <- function(theta, nc, full) {
  L  <- theta_to_L(theta, nc, full)
  fd <- L_first_derivs(L, nc, full)
  Lp <- fd$Lp; k <- length(Lp)
  out <- lapply(seq_len(k), function(.) vector("list", k))
  for (p in seq_len(k)) for (q in seq_len(k)) {
    d2 <- Lp[[p]] %*% t(Lp[[q]]) + Lp[[q]] %*% t(Lp[[p]])
    if (p == q && fd$is_diag[p]) {
      i <- fd$di[p]
      Lpp <- matrix(0, nc, nc); Lpp[i, i] <- L[i, i]     # d2 L / dtheta^2 = E_ii L_ii
      d2 <- d2 + Lpp %*% t(L) + L %*% t(Lpp)
    }
    out[[p]][[q]] <- d2
  }
  out
}

# ---- inner Newton solve for the joint mode ---------------------------------
inner_solve <- function(y, A, P, fam, x0 = NULL) {
  n_x <- ncol(A)
  x <- if (is.null(x0)) rep(0, n_x) else x0
  for (it in 1:300) {
    eta <- as.numeric(A %*% x)
    g   <- as.numeric(t(A) %*% fam$d1(y, eta)) - as.numeric(P %*% x)
    w   <- -fam$d2(y, eta)
    H   <- t(A) %*% (w * A) + P
    step <- solve(H, g)
    x <- x + step
    if (max(abs(step)) < 1e-13) break
  }
  eta <- as.numeric(A %*% x)
  w   <- -fam$d2(y, eta)
  list(x = x, eta = eta, H = t(A) %*% (w * A) + P, w = w)
}

# ---- stacked theta <-> per-block ; P assembly ------------------------------
unpack_theta <- function(theta, blocks) {
  out <- list(); pos <- 0L
  for (m in seq_along(blocks)) {
    k <- n_coord(blocks[[m]]$nc, blocks[[m]]$full)
    out[[m]] <- theta[pos + seq_len(k)]; pos <- pos + k
  }
  out
}

build_P <- function(theta, data) {
  th <- unpack_theta(theta, data$blocks)
  L_list  <- lapply(seq_along(data$blocks),
                    function(m) theta_to_L(th[[m]], data$blocks[[m]]$nc, data$blocks[[m]]$full))
  Om_list <- lapply(L_list, function(L) solve(L %*% t(L)))
  Pb <- c(list(data$P_beta),
          lapply(seq_along(data$blocks),
                 function(m) kronecker(diag(data$blocks[[m]]$G), Om_list[[m]])))
  list(P = as.matrix(Matrix::bdiag(Pb)), L_list = L_list, Om_list = Om_list, th = th)
}

# group-coordinate slice within x for block m, group g (1-based)
grp_slice <- function(data, m, g) {
  bl <- data$blocks[[m]]
  data$p_fix + bl$off + (g - 1L) * bl$nc + seq_len(bl$nc)
}
blk_slice <- function(data, m) {
  bl <- data$blocks[[m]]
  data$p_fix + bl$off + seq_len(bl$G * bl$nc)
}

# ---- log Laplace marginal (theta-free constants dropped) --------------------
log_marginal <- function(theta, data) {
  mp  <- build_P(theta, data)
  fit <- inner_solve(data$y, data$A, mp$P, data$fam)
  ld_H <- as.numeric(determinant(fit$H, logarithm = TRUE)$modulus)
  ld_P <- 0
  for (m in seq_along(data$blocks))
    ld_P <- ld_P + data$blocks[[m]]$G *
      as.numeric(determinant(mp$Om_list[[m]], logarithm = TRUE)$modulus)
  data$fam$ll(data$y, fit$eta) - 0.5 * sum(fit$x * (mp$P %*% fit$x)) +
    0.5 * ld_P - 0.5 * ld_H
}

# ---- base quantities shared by gradient and Hessian ------------------------
base_quantities <- function(theta, data) {
  mp  <- build_P(theta, data)
  fit <- inner_solve(data$y, data$A, mp$P, data$fam)
  x <- fit$x; H <- fit$H; Hinv <- solve(H)
  eta <- fit$eta
  A <- data$A

  s      <- rowSums((A %*% Hinv) * A)              # (A Hinv A')_ii
  dwdeta <- -data$fam$d3(data$y, eta)              # dw/deta = -l'''
  r      <- dwdeta * s
  u      <- as.numeric(Hinv %*% (t(A) %*% r))

  R_list <- V_list <- C_list <- M0_list <- S_list <- list()
  for (m in seq_along(data$blocks)) {
    bl <- data$blocks[[m]]; nc <- bl$nc; G <- bl$G
    idx <- blk_slice(data, m)
    b_m <- matrix(x[idx], nrow = nc); u_m <- matrix(u[idx], nrow = nc)
    R_m <- b_m %*% t(b_m)
    V_m <- matrix(0, nc, nc)
    for (g in seq_len(G)) { sl <- grp_slice(data, m, g); V_m <- V_m + Hinv[sl, sl, drop = FALSE] }
    Cp <- b_m %*% t(u_m); C_m <- 0.5 * (Cp + t(Cp))
    Om <- mp$Om_list[[m]]; M0 <- R_m + V_m - C_m
    R_list[[m]] <- R_m; V_list[[m]] <- V_m; C_list[[m]] <- C_m
    M0_list[[m]] <- M0; S_list[[m]] <- 0.5 * (Om %*% M0 %*% Om - G * Om)
  }
  list(mp = mp, fit = fit, x = x, H = H, Hinv = Hinv, eta = eta,
       s = s, dwdeta = dwdeta, r = r, u = u,
       M0 = M0_list, S = S_list, th = mp$th)
}

# ---- analytic gradient (M0 form) -------------------------------------------
analytic_gradient <- function(theta, data) {
  bq <- base_quantities(theta, data)
  g <- numeric(0)
  for (m in seq_along(data$blocks)) {
    bl <- data$blocks[[m]]
    dS <- dSigma_dtheta(bq$th[[m]], bl$nc, bl$full)
    g  <- c(g, vapply(dS, function(dSig) sum(dSig * bq$S[[m]]), numeric(1)))
  }
  g
}

# ---- analytic Hessian ------------------------------------------------------
analytic_hessian <- function(theta, data) {
  bq <- base_quantities(theta, data)
  A <- data$A; x <- bq$x; Hinv <- bq$Hinv
  s <- bq$s; dwdeta <- bq$dwdeta; u <- bq$u
  d2wdeta2 <- -data$fam$d4(data$y, bq$eta)
  v_r <- as.numeric(t(A) %*% bq$r)

  # coordinate bookkeeping: stacked index -> (block, local)
  ktot <- length(theta); blk_of <- integer(ktot); loc_of <- integer(ktot)
  pos <- 0L
  for (m in seq_along(data$blocks)) {
    k <- n_coord(data$blocks[[m]]$nc, data$blocks[[m]]$full)
    blk_of[pos + seq_len(k)] <- m; loc_of[pos + seq_len(k)] <- seq_len(k); pos <- pos + k
  }
  gstart <- c(0, cumsum(vapply(seq_along(data$blocks),
              function(m) n_coord(data$blocks[[m]]$nc, data$blocks[[m]]$full), integer(1))))

  dS_by_block <- lapply(seq_along(data$blocks), function(m)
    dSigma_dtheta(bq$th[[m]], data$blocks[[m]]$nc, data$blocks[[m]]$full))
  d2S_by_block <- lapply(seq_along(data$blocks), function(m)
    d2Sigma_dtheta(bq$th[[m]], data$blocks[[m]]$nc, data$blocks[[m]]$full))

  Hm <- matrix(0, ktot, ktot)
  for (k in seq_len(ktot)) {
    mk <- blk_of[k]; lk <- loc_of[k]
    blk <- data$blocks[[mk]]; nck <- blk$nc; Gk <- blk$G
    Om_k <- bq$mp$Om_list[[mk]]
    dSig_k <- dS_by_block[[mk]][[lk]]
    dOm_k  <- -Om_k %*% dSig_k %*% Om_k

    # dP_k: dOm_k on each group of block mk
    n_x <- length(x); dP_k <- matrix(0, n_x, n_x)
    for (g in seq_len(Gk)) { sl <- grp_slice(data, mk, g); dP_k[sl, sl] <- dOm_k }

    J_k     <- -as.numeric(Hinv %*% (dP_k %*% x))       # mode Jacobian column
    eta_dot <- as.numeric(A %*% J_k)
    dW_k    <- dwdeta * eta_dot
    dH_k    <- t(A) %*% (dW_k * A) + dP_k
    dHinv_k <- -Hinv %*% dH_k %*% Hinv
    ds_k    <- rowSums((A %*% dHinv_k) * A)
    dr_k    <- (d2wdeta2 * eta_dot) * s + dwdeta * ds_k
    du_k    <- as.numeric(dHinv_k %*% v_r) + as.numeric(Hinv %*% (t(A) %*% dr_k))

    for (m in seq_along(data$blocks)) {
      bl <- data$blocks[[m]]; nc <- bl$nc; G <- bl$G
      idx <- blk_slice(data, m)
      b_m  <- matrix(x[idx],    nrow = nc)
      Jb_m <- matrix(J_k[idx],  nrow = nc)
      u_m  <- matrix(u[idx],    nrow = nc)
      du_m <- matrix(du_k[idx], nrow = nc)

      dR_m <- Jb_m %*% t(b_m) + b_m %*% t(Jb_m)
      dV_m <- matrix(0, nc, nc)
      for (g in seq_len(G)) { sl <- grp_slice(data, m, g); dV_m <- dV_m + dHinv_k[sl, sl, drop = FALSE] }
      Cp   <- Jb_m %*% t(u_m) + b_m %*% t(du_m); dC_m <- 0.5 * (Cp + t(Cp))
      dM0_m <- dR_m + dV_m - dC_m

      Om_m <- bq$mp$Om_list[[m]]; M0_m <- bq$M0[[m]]
      dS_m <- 0.5 * (Om_m %*% dM0_m %*% Om_m)
      if (m == mk)
        dS_m <- dS_m + 0.5 * (dOm_k %*% M0_m %*% Om_m + Om_m %*% M0_m %*% dOm_k - G * dOm_k)

      # term B: <dSigma_j, dS_m> for every j in block m
      dS_list <- dS_by_block[[m]]
      for (lj in seq_along(dS_list)) {
        gj <- gstart[m] + lj
        Hm[gj, k] <- Hm[gj, k] + sum(dS_list[[lj]] * dS_m)
      }
    }

    # term A: <d2Sigma_{j,k}, S_mk> for j in block mk
    S_mk <- bq$S[[mk]]
    for (lj in seq_len(n_coord(nck, blk$full))) {
      gj <- gstart[mk] + lj
      Hm[gj, k] <- Hm[gj, k] + sum(d2S_by_block[[mk]][[lj]][[lk]] * S_mk)
    }
  }
  Hm                                  # raw (unsymmetrized): its natural symmetry is a check
}

# ---- finite differences -----------------------------------------------------
fd_gradient <- function(theta, data, h = 1e-5) {
  vapply(seq_along(theta), function(j) {
    tp <- theta; tp[j] <- tp[j] + h
    tm <- theta; tm[j] <- tm[j] - h
    (log_marginal(tp, data) - log_marginal(tm, data)) / (2 * h)
  }, numeric(1))
}
fd_hessian_from_grad <- function(theta, data, h = 1e-5) {
  k <- length(theta); H <- matrix(0, k, k)
  for (j in seq_len(k)) {
    tp <- theta; tp[j] <- tp[j] + h
    tm <- theta; tm[j] <- tm[j] - h
    H[, j] <- (analytic_gradient(tp, data) - analytic_gradient(tm, data)) / (2 * h)
  }
  (H + t(H)) / 2
}

# ---- model builder ----------------------------------------------------------
# blockspec: list(nc, full, G, per). per = obs per group used to size n; each
# block re-samples its own grouping over the shared n, so blocks are crossed.
build_data <- function(fam, blockspecs, seed = 11) {
  set.seed(seed)
  n <- as.integer(max(vapply(blockspecs, function(b) b$G * b$per, numeric(1))))
  X <- cbind(1, rnorm(n)); p_fix <- ncol(X)
  beta_true <- c(0.3, 0.5)
  eta <- as.numeric(X %*% beta_true)

  blocks <- list(); Zs <- list(); off <- 0L
  for (m in seq_along(blockspecs)) {
    sp <- blockspecs[[m]]; nc <- sp$nc; full <- sp$full; G <- sp$G
    grp  <- sample(rep(seq_len(G), length.out = n))
    Zcol <- if (nc == 1) matrix(1, n, 1) else cbind(1, rnorm(n))
    Z <- matrix(0, n, G * nc)
    for (i in seq_len(n)) { sl <- (grp[i] - 1L) * nc + seq_len(nc); Z[i, sl] <- Zcol[i, ] }
    # a valid probe theta for this block
    if (!full) theta_m <- rep(log(0.6), nc) else {
      theta_m <- numeric(n_coord(nc, full)); idx <- 0
      for (j in seq_len(nc)) for (i in j:nc) { idx <- idx + 1; theta_m[idx] <- if (i == j) log(0.6) else 0.2 }
    }
    Sig <- { L <- theta_to_L(theta_m, nc, full); L %*% t(L) }
    btrue <- t(chol(Sig)) %*% matrix(rnorm(nc * G), nc, G)
    eta <- eta + rowSums(Zcol * t(btrue)[grp, , drop = FALSE])
    blocks[[m]] <- list(nc = nc, full = full, G = G, off = off, theta = theta_m)
    Zs[[m]] <- Z; off <- off + G * nc
  }
  A <- do.call(cbind, c(list(X), Zs))
  y <- switch(fam$kind,
              poisson = rpois(n, exp(eta)),
              binom   = rbinom(n, 1, plogis(eta)),
              gauss   = rnorm(n, eta, 0.7))
  theta <- unlist(lapply(blocks, function(b) b$theta))
  list(y = y, A = A, fam = fam, p_fix = p_fix, P_beta = diag(1e-4, p_fix),
       blocks = blocks, theta = theta)
}

# ---- runner -----------------------------------------------------------------
run_case <- function(label, fam, blockspecs, seed = 11) {
  data  <- build_data(fam, blockspecs, seed)
  theta <- data$theta

  # sub-check: d2Sigma vs FD of dSigma, per block
  d2err <- 0
  for (m in seq_along(data$blocks)) {
    bl <- data$blocks[[m]]; th <- bl$theta; kk <- n_coord(bl$nc, bl$full)
    d2 <- d2Sigma_dtheta(th, bl$nc, bl$full); hh <- 1e-6
    for (p in seq_len(kk)) {
      tp <- th; tp[p] <- tp[p] + hh; tm <- th; tm[p] <- tp[p] - 2 * hh
      fdp <- (dSigma_dtheta(tp, bl$nc, bl$full)[[1]] -
              dSigma_dtheta(tm, bl$nc, bl$full)[[1]]) / (2 * hh)  # only col 1, cheap
      d2err <- max(d2err, max(abs(d2[[1]][[p]] - fdp)))
    }
  }

  ga <- analytic_gradient(theta, data)
  gf <- fd_gradient(theta, data)
  Ha_raw <- analytic_hessian(theta, data)          # raw, unsymmetrized
  Ha <- (Ha_raw + t(Ha_raw)) / 2
  Hf <- fd_hessian_from_grad(theta, data)

  rel_g <- max(abs(ga - gf) / pmax(abs(gf), 1e-8))
  rel_H <- max(abs(Ha - Hf) / pmax(abs(Hf), 1e-6))
  abs_H <- max(abs(Ha - Hf))
  asym  <- max(abs(Ha_raw - t(Ha_raw)))            # natural asymmetry of the analytic form

  cat(sprintf("\n== %s ==\n", label))
  cat(sprintf("  k = %d,  gradient max|rel| = %.2e\n", length(theta), rel_g))
  cat(sprintf("  Hessian  max|abs| vs FD  = %.2e\n", abs_H))
  cat(sprintf("  Hessian  max|rel| vs FD  = %.2e\n", rel_H))
  cat(sprintf("  analytic asymmetry       = %.2e\n", asym))
  cat(sprintf("  d2Sigma vs FD (sub-check)= %.2e\n", d2err))
  invisible(list(Ha = Ha, Hf = Hf, ga = ga, gf = gf))
}

fg <- fam_gauss(0.7)

run_case("poisson  scalar RE",            fam_poisson, list(list(nc = 1, full = FALSE, G = 10, per = 6)))
run_case("poisson  correlated RE (nc=2)", fam_poisson, list(list(nc = 2, full = TRUE,  G = 10, per = 6)))
run_case("binomial scalar RE",            fam_binom,   list(list(nc = 1, full = FALSE, G = 10, per = 6)))
run_case("binomial correlated RE (nc=2)", fam_binom,   list(list(nc = 2, full = TRUE,  G = 10, per = 6)))
run_case("gaussian scalar RE (control)",  fg,          list(list(nc = 1, full = FALSE, G = 10, per = 6)))
run_case("poisson  two crossed blocks",   fam_poisson, list(list(nc = 1, full = FALSE, G = 8, per = 6),
                                                            list(nc = 1, full = FALSE, G = 6, per = 8)))
run_case("binomial block + slope block",  fam_binom,   list(list(nc = 1, full = FALSE, G = 8, per = 7),
                                                            list(nc = 2, full = TRUE,  G = 6, per = 7)))
