# The SMC evidence adds the normalizing constant each latent block's prior
# leaves out (compute_log_prior_normalizer). Two arbiters, neither of them the
# engine's density code:
#   * pointwise: the engine's prior plus its normalizer, at random parameter
#     vectors, equals the prior density written out here from its definition
#     (dense matrices, base R densities);
#   * end to end: on a gaussian response with a Gaussian latent field the
#     evidence is a Gaussian marginal, integrated over the field's
#     hyperparameter by quadrature, and the SMC estimate sits within its
#     Monte Carlo error of it.

lmvn_prec <- function(x, P) {
  R <- chol(P)
  sum(log(diag(R))) - 0.5 * length(x) * log(2 * pi) - 0.5 * sum((R %*% x)^2)
}

lmvn_cov <- function(x, S) {
  R <- chol(S)
  z <- backsolve(R, x, transpose = TRUE)
  -sum(log(diag(R))) - 0.5 * length(x) * log(2 * pi) - 0.5 * sum(z^2)
}

# CSR adjacency (0-based column indices) from a symmetric 0/1 matrix.
adj_csr <- function(W) {
  J <- nrow(W)
  cols <- lapply(seq_len(J), function(i) which(W[i, ] != 0) - 1L)
  list(adj_row_ptr = as.integer(c(0L, cumsum(lengths(cols)))),
       adj_col_idx = as.integer(unlist(cols)),
       n_neighbors = as.integer(lengths(cols)))
}

# The prior on the fixed effects, in every model below.
lp_beta <- function(b, sdb) sum(dnorm(b, 0, sdb, log = TRUE))

prior_check_fixture <- function(J = 8L, n = 40L, seed = 1L) {
  set.seed(seed)
  x <- rnorm(n)
  list(X = cbind(1, x), y = rnorm(n), n = n)
}

# Rows of plausible parameter values, named by the engine's own layout.
theta_rows <- function(names, n_rows = 4L, seed = 2L) {
  set.seed(seed)
  m <- matrix(rnorm(n_rows * length(names), 0, 0.6), n_rows,
              dimnames = list(NULL, names))
  m
}

# Two components and an island: {1..4} a path with a chord, {5..7} a
# triangle, {8} alone.
areal_W <- function() {
  W <- matrix(0, 8, 8)
  e <- rbind(c(1, 2), c(2, 3), c(3, 4), c(1, 3), c(5, 6), c(6, 7), c(5, 7))
  W[e] <- 1; W[e[, 2:1]] <- 1
  W
}

test_that("the ICAR / BYM2 / proper-CAR normalizers make the prior a density", {
  load_cpp_fixture("smc_normalizer")
  f <- prior_check_fixture()
  sdb <- 2; s_hc <- 2.5
  W <- areal_W(); J <- nrow(W)
  csr <- adj_csr(W)
  Q <- diag(rowSums(W)) - W
  comps <- list(1:4, 5:7, 8)
  Qaug <- Q
  for (cc in comps) Qaug[cc, cc] <- Qaug[cc, cc] + 1 / length(cc)
  sidx <- as.integer(rep_len(seq_len(J), f$n))

  # ICAR
  sp <- c(list(type = "icar", spatial_idx = sidx, n_spatial_units = J), csr)
  probe <- cpp_test_log_prior_normalized(
    matrix(0, 0, 0), f$y, integer(0), f$X, "gaussian", 1, sdb, s_hc,
    spatial_spec = sp)
  th <- theta_rows(probe$names)
  got <- cpp_test_log_prior_normalized(th, f$y, integer(0), f$X, "gaussian",
                                       1, sdb, s_hc, spatial_spec = sp)
  expect_length(got$declined, 0L)
  want <- apply(th, 1, function(r) {
    tau <- exp(r[["log_tau_spatial"]])
    phi <- r[grep("^phi_spatial", names(r))]
    lp_beta(r[1:2], sdb) +
      dgamma(tau, 1, 0.01, log = TRUE) + log(tau) +
      lmvn_prec(phi, tau * Qaug)
  })
  expect_equal(got$log_prior, unname(want), tolerance = 1e-10)

  # BYM2, with a per-component precision multiplier on phi
  w <- c(rep(1.3, 4), rep(0.7, 3), 2)
  sp_b <- modifyList(sp, list(type = "bym2", scale_factor = 1, node_prec = w))
  probe <- cpp_test_log_prior_normalized(
    matrix(0, 0, 0), f$y, integer(0), f$X, "gaussian", 1, sdb,
    s_hc, spatial_spec = sp_b)
  th <- theta_rows(probe$names)
  got <- cpp_test_log_prior_normalized(th, f$y, integer(0), f$X, "gaussian",
                                       1, sdb, s_hc, spatial_spec = sp_b)
  expect_length(got$declined, 0L)
  want <- apply(th, 1, function(r) {
    ls <- r[["log_sigma_spatial"]]; s <- exp(ls)
    rho <- plogis(r[["logit_rho_bym2"]])
    phi <- r[grep("^phi_spatial", names(r))]
    tht <- r[grep("^theta_spatial", names(r))]
    lp_beta(r[1:2], sdb) +
      log(2 / (pi * s_hc)) - log1p((s / s_hc)^2) + ls +
      log(rho) + log1p(-rho) +
      lmvn_prec(phi, diag(w) %*% Qaug) + sum(dnorm(tht, log = TRUE))
  })
  expect_equal(got$log_prior, unname(want), tolerance = 1e-10)

  # Proper CAR on a connected graph (a ring of six with one chord)
  Wc <- matrix(0, 6, 6)
  e <- rbind(cbind(1:6, c(2:6, 1)), c(1, 4))
  Wc[e] <- 1; Wc[e[, 2:1]] <- 1
  d <- rowSums(Wc)
  Wn <- Wc / sqrt(outer(d, d))
  bounds <- c(-0.6, 0.95)
  sp_c <- c(list(type = "car_proper", spatial_idx = as.integer(rep_len(1:6, f$n)),
                 n_spatial_units = 6L,
                 adj_eigenvalues = eigen(Wn, symmetric = TRUE)$values,
                 rho_lower = bounds[1], rho_upper = bounds[2]), adj_csr(Wc))
  probe <- cpp_test_log_prior_normalized(
    matrix(0, 0, 0), f$y, integer(0), f$X, "gaussian", 1, sdb, s_hc,
    spatial_spec = sp_c)
  th <- theta_rows(probe$names)
  got <- cpp_test_log_prior_normalized(th, f$y, integer(0), f$X, "gaussian",
                                       1, sdb, s_hc, spatial_spec = sp_c)
  expect_length(got$declined, 0L)
  want <- apply(th, 1, function(r) {
    tau <- exp(r[["log_tau_spatial"]])
    u <- plogis(r[["logit_rho_car"]])
    rho <- bounds[1] + diff(bounds) * u
    phi <- r[grep("^phi_spatial", names(r))]
    lp_beta(r[1:2], sdb) +
      dgamma(tau, 1, 0.01, log = TRUE) + log(tau) + log(u) + log1p(-u) +
      lmvn_prec(phi, tau * (diag(d) - rho * Wc))
  })
  expect_equal(got$log_prior, unname(want), tolerance = 1e-10)
})

# The structure matrix of a random walk of order k over Tn nodes.
rw_structure <- function(Tn, order, cyclic) {
  D <- diag(Tn)
  for (k in seq_len(order)) {
    D <- if (cyclic) (D - D[c(Tn, seq_len(Tn - 1)), ]) else diff(D)
  }
  crossprod(D)
}

test_that("the temporal normalizers make the prior a density", {
  load_cpp_fixture("smc_normalizer")
  f <- prior_check_fixture()
  sdb <- 2; Tn <- 10L
  tidx <- as.integer(rep_len(seq_len(Tn), f$n))
  run <- function(tp, lp_field, tol = 1e-10) {
    probe <- cpp_test_log_prior_normalized(
      matrix(0, 0, 0), f$y, integer(0), f$X, "gaussian", 1, sdb, 2.5,
      temporal_spec = tp)
    th <- theta_rows(probe$names)
    got <- cpp_test_log_prior_normalized(th, f$y, integer(0), f$X,
                                         "gaussian", 1, sdb, 2.5,
                                         temporal_spec = tp)
    expect_length(got$declined, 0L)
    want <- apply(th, 1, function(r) {
      lp_beta(r[1:2], sdb) + lp_field(r, r[grep("^phi_temporal", names(r))])
    })
    expect_equal(got$log_prior, unname(want), tolerance = tol,
                 info = paste(tp$type, isTRUE(tp$cyclic)))
  }
  base <- list(time_idx = tidx, n_times = Tn, n_groups = 1L, group_idx = NULL)
  for (cfg in list(list("rw1", 1L, FALSE), list("rw1", 1L, TRUE),
                   list("rw2", 2L, TRUE))) {
    R <- rw_structure(Tn, cfg[[2]], cfg[[3]])
    Raug <- R + 1 / Tn
    run(c(base, list(type = cfg[[1]], cyclic = cfg[[3]])), function(r, x) {
      tau <- exp(r[["log_tau_temporal"]])
      dgamma(tau, 1, 0.01, log = TRUE) + log(tau) + lmvn_prec(x, tau * Raug)
    })
  }
  # AR1 under a Beta(2, 3) on (rho + 1) / 2
  run(c(base, list(type = "ar1", cyclic = FALSE, rho_prior_a = 2,
                   rho_prior_b = 3)), function(r, x) {
    tau <- exp(r[["log_tau_temporal"]])
    u <- plogis(r[["logit_rho_ar1"]]); rho <- 2 * u - 1
    P <- diag(c(1, rep(1 + rho^2, Tn - 2), 1))
    P[cbind(1:(Tn - 1), 2:Tn)] <- P[cbind(2:Tn, 1:(Tn - 1))] <- -rho
    dgamma(tau, 1, 0.01, log = TRUE) + log(tau) +
      dbeta(u, 2, 3, log = TRUE) + log(u) + log1p(-u) +
      lmvn_prec(x, tau * P)
  })
  # Temporal GP, both parameterizations, on irregular instants. The engine
  # factors a dense covariance with a 1e-8 nugget, carried here too, and the OU
  # chain of the exponential kernel floors 1 - rho^2, which moves its density
  # in the eighth significant digit.
  tv <- cumsum(c(0, rexp(Tn - 1, 2)))
  lo <- 0.05; hi <- 4
  gp_hyper <- function(r) {
    ls2 <- r[["log_sigma2_temporal_gp"]]
    ell <- lo + (hi - lo) * plogis(r[["logit_phi_temporal_gp"]])
    lam <- -log(0.01) / 1
    list(s2 = exp(ls2), ell = ell,
         lp = log(lam) - lam * exp(ls2 / 2) - log(2) + ls2 / 2 +
              log((ell - lo) * (hi - ell) / (hi - lo)) - log(hi - lo))
  }
  for (cov in c("exponential", "matern")) {
    for (par in c("centered", "noncentered")) {
      run(c(base, list(type = "gp", time_values = tv, cov = cov, nu = 1.5,
                       parameterization = par, phi_prior_lower = lo,
                       phi_prior_upper = hi)), function(r, x) {
        h <- gp_hyper(r)
        u <- abs(outer(tv, tv, "-")) / h$ell
        K <- h$s2 * if (cov == "exponential") exp(-u) else
          (1 + sqrt(3) * u) * exp(-sqrt(3) * u)
        h$lp + if (par == "centered")
          lmvn_cov(x, K + diag(1e-8, Tn)) else sum(dnorm(x, log = TRUE))
      }, tol = 1e-6)
    }
  }
})

# PC prior on a standard deviation, carried to log(sigma2) and to log(tau).
lp_pc_log_sigma2 <- function(ls2, U, a) {
  lam <- -log(a) / U
  log(lam) - lam * exp(ls2 / 2) - log(2) + ls2 / 2
}
lp_pc_log_tau <- function(lt, U, a) {
  lam <- -log(a) / U
  log(lam) - lam * exp(-lt / 2) - log(2) - lt / 2
}
# PC prior on a d = 2 range, carried to log(range).
lp_pc_log_range <- function(lr, U, a) {
  lam <- -log(a) * U
  log(lam) - 2 * lr - lam * exp(-lr) + lr
}

test_that("the GP / HSGP / SVC / TVC normalizers make the prior a density", {
  load_cpp_fixture("smc_normalizer")
  sdb <- 2
  set.seed(3)
  n <- 9L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n), t = rep(1:3, 3))
  d$y <- rnorm(n)
  X <- model.matrix(~ x, d)
  check <- function(spec_args, lp_rest, tol = 1e-10, info = "") {
    call_fx <- function(th) do.call(cpp_test_log_prior_normalized,
      c(list(th, d$y, integer(0), X, "gaussian", 1, sdb, 2.5), spec_args))
    nm <- call_fx(matrix(0, 0, 0))$names
    th <- theta_rows(nm)
    got <- call_fx(th)
    expect_length(got$declined, 0L)
    want <- apply(th, 1, function(r) lp_beta(r[1:2], sdb) + lp_rest(r))
    expect_equal(got$log_prior, unname(want), tolerance = tol, info = info)
  }

  # NNGP with every earlier location a neighbour: the Vecchia density is then
  # the full GP density, centered; and the non-centered z.
  for (par in c("centered", "noncentered")) {
    v <- validate_gp(spatial_gp(~ lon + lat, nn = n - 1L,
                                parameterization = par), d)
    sp <- tulpa:::.gp_sampler_spec(v)
    check(list(spatial_spec = sp), function(r) {
      ls2 <- r[["log_sigma2_gp"]]; lph <- r[["log_phi_gp"]]
      w <- r[grep("^gp_w", names(r))]
      lp_pc_log_sigma2(ls2, sp$sigma2_prior_U, sp$sigma2_prior_alpha) +
        lp_pc_log_range(lph, sp$phi_prior_U, sp$phi_prior_alpha) +
        if (par == "centered") {
          K <- exp(ls2) * exp(-as.matrix(dist(sp$coords)) / exp(lph))
          lmvn_cov(w, K)
        } else sum(dnorm(w, log = TRUE))
    }, tol = 1e-6, info = paste("nngp", par))
  }

  # Multi-scale NNGP: two independent fields, each with every earlier location
  # a neighbour.
  for (smp in c("centered", "noncentered")) {
    v <- validate_gp(spatial_multiscale(~ lon + lat, nn_local = n - 1L,
                                        range_local = c(0.05, 1),
                                        range_regional = c(1, 10),
                                        nn_regional = n - 1L, sampler = smp), d)
    sp <- tulpa:::.msgp_sampler_spec(v)
    check(list(spatial_spec = sp), function(r) {
      lp <- 0
      for (s in c("local", "regional")) {
        ls2 <- r[[paste0("log_sigma2_gp_", s)]]
        lph <- r[[paste0("log_phi_gp_", s)]]
        w <- r[grep(paste0("^gp_", s, "\\["), names(r))]
        lp <- lp +
          lp_pc_log_sigma2(ls2, sp[[paste0("sigma2_", s, "_prior_U")]],
                           sp[[paste0("sigma2_", s, "_prior_alpha")]]) +
          lp_pc_log_range(lph, sp[[paste0("range_", s, "_lower")]],
                          sp[[paste0("range_", s, "_prior_alpha")]]) +
          if (smp == "centered") {
            lmvn_cov(w, exp(ls2) * exp(-as.matrix(dist(sp$coords)) / exp(lph)))
          } else sum(dnorm(w, log = TRUE))
      }
      lp
    }, tol = 1e-6, info = paste("multiscale", smp))
  }

  # HSGP
  v <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp", m = 3), d)
  sp <- tulpa:::.hsgp_sampler_spec(v)
  check(list(spatial_spec = sp), function(r) {
    lp_pc_log_sigma2(r[["log_sigma2_hsgp"]], 1, 0.01) +
      dnorm(r[["log_lengthscale_hsgp"]], log = TRUE) +
      sum(dnorm(r[grep("^hsgp_beta", names(r))], log = TRUE))
  }, info = "hsgp")

  # SVC on either basis (non-centered NNGP z, HSGP coefficients)
  for (approx in c("nngp", "hsgp")) {
    v <- validate_svc(spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 4L,
                                  approx = approx, m = 3), d, X)
    sp <- tulpa:::.svc_sampler_spec(v, X)
    check(list(svc_spec = sp), function(r) {
      ls2 <- r[["log_sigma2_svc[1]"]]
      lp <- log(2 / pi) - log1p(exp(ls2)) - log(2) + ls2 / 2 +
        sum(dnorm(r[grep("^svc_w", names(r))], log = TRUE))
      lp + if (approx == "nngp")
        lp_pc_log_range(r[["log_phi_svc[1]"]], sp$phi_prior_U, sp$phi_prior_alpha)
      else dnorm(r[["log_phi_svc[1]"]], log = TRUE)
    }, info = paste("svc", approx))
  }

  # TVC: RW1, AR1 and GP coefficients over three instants
  Tn <- 3L
  for (st in c("rw1", "ar1", "gp")) {
    v <- tulpa:::validate_tvc(temporal_tvc("t", terms = ~ x - 1, structure = st,
                                           cov = "exponential"), d, X)
    sp <- tulpa:::.tvc_sampler_spec(v, X)
    check(list(tvc_spec = sp), function(r) {
      w <- r[grep("^tvc_w", names(r))]
      if (st == "gp") {
        tv <- sp$time_values
        lo <- sp$phi_prior_lower %||% 0.01; hi <- sp$phi_prior_upper %||% 10
        ell <- lo + (hi - lo) * plogis(r[["logit_phi_tvc_gp[1]"]])
        ls2 <- r[["log_sigma2_tvc_gp[1]"]]
        return(lp_pc_log_sigma2(ls2, 1, 0.01) +
                 log((ell - lo) * (hi - ell) / (hi - lo)) - log(hi - lo) +
                 lmvn_cov(w, exp(ls2) * exp(-abs(outer(tv, tv, "-")) / ell)))
      }
      lt <- r[["log_tau_tvc[1]"]]; tau <- exp(lt)
      lp <- lp_pc_log_tau(lt, 1, 0.01)
      if (st == "rw1") return(lp + lmvn_prec(w, tau * (rw_structure(Tn, 1, FALSE) + 1 / Tn)))
      u <- plogis(r[["logit_rho_tvc[1]"]]); rho <- 2 * u - 1
      P <- diag(c(1, rep(1 + rho^2, Tn - 2), 1))
      P[cbind(1:(Tn - 1), 2:Tn)] <- P[cbind(2:Tn, 1:(Tn - 1))] <- -rho
      lp + log(u) + log1p(-u) + lmvn_prec(w, tau * P)
    }, tol = if (st == "gp") 1e-6 else 1e-10, info = paste("tvc", st))
  }
})

test_that("an improper latent prior leaves the evidence declined, by name", {
  load_cpp_fixture("smc_normalizer")
  f <- prior_check_fixture()
  Tn <- 10L
  base <- list(time_idx = as.integer(rep_len(seq_len(Tn), f$n)), n_times = Tn,
               n_groups = 1L, group_idx = NULL, cyclic = FALSE)
  # An acyclic RW2 keeps its linear null direction.
  got <- cpp_test_log_prior_normalized(
    matrix(0, 0, 0), f$y, integer(0), f$X, "gaussian", 1, 2, 2.5,
    temporal_spec = modifyList(base, list(type = "rw2")))
  expect_identical(got$declined, "temporal:rw2_improper")
  expect_true(is.na(got$normalizer))
  # RW1 over two groups leaves the group contrast free.
  two <- modifyList(base, list(type = "rw1", n_groups = 2L,
                               group_idx = as.integer(rep_len(1:2, f$n))))
  got <- cpp_test_log_prior_normalized(
    matrix(0, 0, 0), f$y, integer(0), f$X, "gaussian", 1, 2, 2.5,
    temporal_spec = two)
  expect_identical(got$declined, "temporal:group_contrasts_improper")
})

test_that("the multi-scale temporal normalizer makes the prior a density", {
  load_cpp_fixture("smc_normalizer")
  f <- prior_check_fixture()
  sdb <- 2; Tn <- 8L; per <- 4L
  tp <- list(type = "multiscale",
             time_index = as.integer(rep_len(seq_len(Tn), f$n)),
             n_times = Tn, n_groups = 1L, group_index = NULL, trend = "rw1",
             seasonal = per, short_term = "ar1", shared = TRUE,
             trend_parameterization = "centered")
  call_fx <- function(th) cpp_test_log_prior_normalized(
    th, f$y, integer(0), f$X, "gaussian", 1, sdb, 2.5, temporal_spec = tp)
  th <- theta_rows(call_fx(matrix(0, 0, 0))$names)
  # The trend is read on its values (the parameterization every backend but
  # the exact sampler takes), whose density the reference below writes out.
  # The short-term correlation's Beta(a, b) on (rho + 1) / 2 is the front
  # door's Beta(2, 2) and a skewed pair: the normalizer follows the pair the
  # density reads.
  for (ab in list(c(2, 2), c(3, 1.5))) {
    tp$rho_prior_a <- ab[1L]; tp$rho_prior_b <- ab[2L]
    got <- call_fx(th)
    expect_length(got$declined, 0L)
    want <- apply(th, 1, function(r) {
      tr <- r[grep("^trend\\[", names(r))]
      se <- r[grep("^seasonal\\[", names(r))]
      sh <- r[grep("^short_term\\[", names(r))]
      s2t <- exp(r[["log_sigma2_trend"]]); s2s <- exp(r[["log_sigma2_seasonal"]])
      s2h <- exp(r[["log_sigma2_short"]])
      u <- plogis(r[["logit_rho_short"]]); rho <- 2 * u - 1
      P <- diag(c(1, rep(1 + rho^2, Tn - 2), 1))
      P[cbind(1:(Tn - 1), 2:Tn)] <- P[cbind(2:Tn, 1:(Tn - 1))] <- -rho
      lp_beta(r[1:2], sdb) +
        lp_pc_log_sigma2(r[["log_sigma2_trend"]], 1, 0.01) +
        lmvn_prec(tr, (rw_structure(Tn, 1, FALSE) + 1 / Tn) / s2t) +
        lp_pc_log_sigma2(r[["log_sigma2_seasonal"]], 1, 0.01) +
        lmvn_prec(se, (rw_structure(per, 1, TRUE) + 1 / per) / s2s) +
        lp_pc_log_sigma2(r[["log_sigma2_short"]], 1, 0.01) +
        dbeta(u, ab[1L], ab[2L], log = TRUE) + log(u) + log1p(-u) +
        lmvn_prec(sh, P / s2h)
    })
    expect_equal(got$log_prior, unname(want), tolerance = 1e-10)
  }
  # An RW2 trend keeps its linear null direction.
  tp$trend <- "rw2"
  expect_identical(call_fx(matrix(0, 0, 0))$declined,
                   "multiscale_temporal:trend_improper")
})

# End to end: an ICAR field on a gaussian response, its precision integrated by
# quadrature over log(tau) of the Gaussian marginal y ~ N(0, phi I + sdb^2 XX'
# + A C (tau Q_aug)^-1 C A'), C the centring the field takes into eta.
test_that("the SMC evidence of an ICAR model matches quadrature", {
  skip_if_not_slow()
  set.seed(20)
  J <- 20; W <- matrix(0, J, J)
  for (i in 1:(J - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
  for (i in 1:(J - 3)) if (i %% 3 == 0) W[i, i + 3] <- W[i + 3, i] <- 1
  n <- 100; x <- rnorm(n); X <- cbind(1, x); s <- rep_len(1:J, n)
  u <- cumsum(rnorm(J, 0, 0.4)); u <- u - mean(u)
  phi <- 0.25; sdb <- 2
  y <- 1 + 0.5 * x + u[s] + rnorm(n, 0, 0.5)
  Qaug <- diag(rowSums(W)) - W + 1 / J
  C <- diag(J) - 1 / J; A <- outer(s, 1:J, "==") * 1
  Sf <- A %*% C %*% solve(Qaug) %*% C %*% t(A)
  lmarg <- function(S) {
    R <- chol(S); z <- backsolve(R, y, transpose = TRUE)
    -0.5 * n * log(2 * pi) - sum(log(diag(R))) - 0.5 * sum(z^2)
  }
  lg <- seq(-6, 12, length.out = 1501)
  lv <- vapply(lg, function(lt) lmarg(phi * diag(n) + sdb^2 * tcrossprod(X) +
    Sf / exp(lt)), 0) + dgamma(exp(lg), 1, 0.01, log = TRUE) + lg
  m <- max(lv); exact <- m + log(sum(exp(lv - m)) * diff(lg)[1])
  sp <- c(list(type = "icar", spatial_idx = as.integer(s), n_spatial_units = J),
          adj_csr(W))
  fit <- tulpa_sample_glmm(y, NULL, X, "gaussian", "smc", phi = phi,
                           sigma_beta = sdb, spatial_spec = sp,
                           control = list(n_particles = 1000L, seed = 1L))
  expect_null(fit$log_evidence_declined)
  expect_lt(abs(fit$log_evidence - exact), 4 * fit$log_evidence_se)
  expect_true(length(fit$n_moves_reference) >= 1L)
})

# At the default settings, across seeds: the mean error of the log evidence
# against quadrature sits within its Monte Carlo error, and the spread of the
# errors matches the standard error the fit reports. An AR1 field on 15 times,
# its precision and correlation integrated on a grid over (log tau, logit u),
# rho = 2u - 1, Gamma(1, 0.01) on tau and a uniform u.
test_that("the SMC evidence of an AR1 model is calibrated across seeds", {
  skip_if_not_slow()
  set.seed(20)
  Tn <- 15; n <- 90; x <- rnorm(n); X <- cbind(1, x); tt <- rep_len(1:Tn, n)
  f <- as.numeric(arima.sim(list(ar = 0.6), Tn, sd = 0.4))
  phi <- 0.25; sdb <- 2
  y <- 1 + 0.5 * x + f[tt] + rnorm(n, 0, 0.5)
  A <- outer(tt, 1:Tn, "==") * 1
  base <- phi * diag(n) + sdb^2 * tcrossprod(X)
  lg <- seq(-4, 10, length.out = 141); lr <- seq(-8, 8, length.out = 161)
  lv <- outer(lg, lr, Vectorize(function(lt, l) {
    u <- plogis(l); rho <- 2 * u - 1
    P <- diag(c(1, rep(1 + rho^2, Tn - 2), 1))
    P[cbind(1:(Tn - 1), 2:Tn)] <- P[cbind(2:Tn, 1:(Tn - 1))] <- -rho
    lmvn_cov(y, base + A %*% solve(exp(lt) * P) %*% t(A)) +
      dgamma(exp(lt), 1, 0.01, log = TRUE) + lt + log(u) + log1p(-u)
  }))
  m <- max(lv)
  exact <- m + log(sum(exp(lv - m)) * diff(lg)[1] * diff(lr)[1])
  tp <- list(type = "ar1", time_idx = as.integer(tt), n_times = Tn,
             n_groups = 1L, group_idx = NULL, cyclic = FALSE)
  res <- t(vapply(1:12, function(s) {
    fit <- tulpa_sample_glmm(y, NULL, X, "gaussian", "smc", phi = phi,
                             sigma_beta = sdb, temporal_spec = tp,
                             control = list(n_particles = 1000L, seed = s))
    c(fit$log_evidence - exact, fit$log_evidence_se)
  }, numeric(2)))
  err <- res[, 1]
  expect_lt(abs(mean(err)), 3 * sd(err) / sqrt(length(err)))
  ratio <- sd(err) / mean(res[, 2])
  expect_gt(ratio, 0.5)
  expect_lt(ratio, 1.6)
})
