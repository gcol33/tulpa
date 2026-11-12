full_st_neighbors <- function(coords, time, k) {
    n <- nrow(coords)
    stopifnot(k == n - 1L)
    index <- matrix(0L, n, k)
    space <- temporal <- matrix(0, n, k)
    for (i in seq_len(n)[-1L]) {
        j <- seq_len(i - 1L)
        index[i, j] <- j
        delta <- sweep(coords[j, , drop = FALSE], 2L, coords[i, ], "-")
        space[i, j] <- sqrt(rowSums(delta^2))
        temporal[i, j] <- abs(time[j] - time[i])
    }
    list(nn_idx = index, nn_dist_space = space, nn_dist_time = temporal, nn_order = seq_len(n), nn_order_inv = seq_len(n), 
        k = k)
}

test_that("cpp_test_st_gp_log_lik is the covariance it claims", {
    unit_kernel <- function(d, phi, cov) {
        switch(cov, exponential = exp(-d/phi), matern = (1 + sqrt(3) * d/phi) * exp(-sqrt(3) * d/phi), 
            gaussian = exp(-0.5 * (d/phi)^2), spherical = ifelse(d >= phi, 0, 1 - 1.5 * (d/phi) + 0.5 * 
                (d/phi)^3), stop("unknown kernel"))
    }
    st_corr_ref <- function(h, u, phi_s, phi_t, nonsep, cov_s, cov_t) {
        if (nonsep == "gneiting") {
            base <- (u/phi_t)^2 + 1
            return(exp(-(h/phi_s)/sqrt(base))/base)
        }
        unit_kernel(h, phi_s, cov_s) * unit_kernel(u, phi_t, cov_t)
    }
    dense_logpdf <- function(w, Sigma) {
        L <- chol(Sigma)
        z <- backsolve(L, w, transpose = TRUE)
        -0.5 * length(w) * log(2 * pi) - sum(log(diag(L))) - 0.5 * sum(z^2)
    }
    set.seed(11)
    S <- 5L
    T <- 4L
    n <- S * T
    xy <- cbind(runif(S), runif(S))
    coords <- xy[rep(seq_len(S), each = T), , drop = FALSE]
    times <- rep(seq_len(T)/T, times = S)
    H <- as.matrix(dist(coords))
    U <- abs(outer(times, times, "-"))
    nb <- full_st_neighbors(coords, times, k = n - 1L)
    w <- rnorm(n)
    nngp <- function(sigma2, phi_s, phi_t, nonsep, cov) {
        tulpa:::cpp_test_st_gp_log_lik(w = w, sigma2 = sigma2, phi_space = phi_s, phi_time = phi_t, coords = coords, 
            time_values = times, nn = n - 1L, nn_idx = as.integer(t(nb$nn_idx)), nn_dist_space = as.numeric(t(nb$nn_dist_space)), 
            nn_dist_time = as.numeric(t(nb$nn_dist_time)), nn_order = as.integer(nb$nn_order), nonsep_type = nonsep, 
            cov_space = cov, cov_time = cov)
    }
    grid <- expand.grid(nonsep = c("product", "gneiting"), cov = c("exponential", "matern"), sigma2 = c(0.4, 
        2.5), stringsAsFactors = FALSE)
    for (i in seq_len(nrow(grid))) {
        nonsep <- grid$nonsep[i]
        cov <- grid$cov[i]
        sigma2 <- grid$sigma2[i]
        phi_s <- 0.35
        phi_t <- 0.6
        R <- matrix(st_corr_ref(as.vector(H), as.vector(U), phi_s, phi_t, nonsep, cov, cov), n, n)
        expect_equal(nngp(sigma2, phi_s, phi_t, nonsep, cov), dense_logpdf(w, sigma2 * R), tolerance = 1e-05, 
            info = paste(nonsep, cov, "sigma2 =", sigma2))
    }
    dev_at <- function(phi_s) {
        R <- matrix(st_corr_ref(as.vector(H), as.vector(U), phi_s, 0.6, "product", "gaussian", "gaussian"), 
            n, n)
        e <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
        c(cond = max(e)/min(e), dev = abs(nngp(0.4, phi_s, 0.6, "product", "gaussian") - dense_logpdf(w, 
            0.4 * R)))
    }
    coarse <- dev_at(0.35)
    fine <- dev_at(0.15)
    expect_gt(coarse[["cond"]]/fine[["cond"]], 5)
    expect_equal(coarse[["dev"]]/fine[["dev"]], coarse[["cond"]]/fine[["cond"]], tolerance = 0.25)
})

test_that("the density falls to -Inf on a structure that is not filled", {
    set.seed(3)
    n <- 6L
    coords <- cbind(runif(n), runif(n))
    times <- as.numeric(seq_len(n))
    w <- rnorm(n)
    got <- tulpa:::cpp_test_st_gp_log_lik(w = w, sigma2 = 1, phi_space = 0.5, phi_time = 0.5, coords = coords, 
        time_values = times, nn = 2L, nn_idx = integer(0), nn_dist_space = numeric(0), nn_dist_time = numeric(0), 
        nn_order = seq_len(n))
    expect_true(is.infinite(got) && got < 0)
})

test_that("an additive space-time kernel cannot carry an interaction", {
    set.seed(5)
    S <- 6L
    T <- 5L
    xy <- cbind(runif(S), runif(S))
    coords <- xy[rep(seq_len(S), each = T), , drop = FALSE]
    times <- rep(seq_len(T), times = S)
    H <- as.matrix(dist(coords))
    U <- abs(outer(times, times, "-"))
    ks <- exp(-H/0.3)
    kt <- exp(-U/0.4)
    rank_of <- function(M) {
        e <- eigen(M, symmetric = TRUE, only.values = TRUE)$values
        sum(e > 1e-08 * max(e))
    }
    expect_equal(rank_of(0.5 * (ks + kt)), S + T - 1L)
    expect_equal(rank_of(ks * kt), S * T)
    v <- numeric(S * T)
    v[1] <- 1
    v[2] <- -1
    v[T + 1L] <- -1
    v[T + 2L] <- 1
    expect_lt(abs(as.numeric(t(v) %*% (0.5 * (ks + kt)) %*% v)), 1e-10)
    expect_gt(as.numeric(t(v) %*% (ks * kt) %*% v), 0.001)
})

test_that("the margin penalty's normalizer is 0.5 log|I + B' Sigma B| with its derivatives (#151)", {
    set.seed(12)
    S <- 5L
    T <- 4L
    n <- S * T
    xy <- cbind(runif(S), runif(S))
    coords <- xy[rep(seq_len(S), each = T), , drop = FALSE]
    times <- rep(seq_len(T)/T, times = S)
    H <- as.matrix(dist(coords))
    U <- abs(outer(times, times, "-"))
    nb <- full_st_neighbors(coords, times, k = n - 1L)
    slice <- rep(seq_len(T), times = S)
    region <- rep(seq_len(S), each = T)
    B <- cbind(sapply(seq_len(T), function(t) (slice == t)/(0.001 * S)), sapply(seq_len(S), function(s) (region == 
        s)/(0.001 * T)))
    dense <- function(tau, phi_s, phi_t) {
        base <- (U/phi_t)^2 + 1
        Sigma <- exp(-(H/phi_s)/sqrt(base))/base/tau
        0.5 * as.numeric(determinant(diag(ncol(B)) + t(B) %*% Sigma %*% B)$modulus)
    }
    engine <- function(tau, phi_s, phi_t) {
        tulpa:::cpp_test_st_gp_margin_normalizer(tau = tau, phi_space = phi_s, phi_time = phi_t, n_spatial = S, 
            n_times = T, coords = coords, time_values = times, nn = n - 1L, nn_idx = as.integer(t(nb$nn_idx)), 
            nn_dist_space = as.numeric(t(nb$nn_dist_space)), nn_dist_time = as.numeric(t(nb$nn_dist_time)), 
            nn_order = as.integer(nb$nn_order))
    }
    for (p in list(c(0.5, 0.2, 0.3), c(3, 2, 0.1), c(0.25, 8, 1))) {
        r <- engine(p[1], p[2], p[3])
        expect_equal(r$value, dense(p[1], p[2], p[3]), tolerance = 1e-08)
        h <- 1e-05
        fd <- sapply(1:3, function(k) {
            up <- dn <- log(p)
            up[k] <- up[k] + h
            dn[k] <- dn[k] - h
            (dense(exp(up[1]), exp(up[2]), exp(up[3])) - dense(exp(dn[1]), exp(dn[2]), exp(dn[3])))/(2 * 
                h)
        })
        expect_equal(r$grad, fd, tolerance = 1e-05)
    }
    expect_gt(abs(engine(1, 8, 0.3)$value - engine(1, 0.2, 0.3)$value), 1)
})

test_that("the constrained AR1 interaction has a dense Gaussian normalizer", {
    S <- 5L
    T <- 4L
    tau <- 2
    ar1_R <- function(rho) {
        R <- diag(c(1, rep(1 + rho^2, T - 2), 1))
        for (i in 1:(T - 1)) R[i, i + 1] <- R[i + 1, i] <- -rho
        R
    }
    Qicar <- matrix(0, S, S)
    for (i in 1:(S - 1)) Qicar[i, i + 1] <- Qicar[i + 1, i] <- -1
    diag(Qicar) <- -rowSums(Qicar)
    projection <- kronecker(diag(S) - matrix(1/S, S, S), diag(T) - matrix(1/T, T, T))
    for (type in c("II", "IV")) {
        log_mass <- function(rho) {
            spatial <- if (type == "II") 
                diag(S)
            else Qicar
            Q <- tau * (projection %*% kronecker(spatial, ar1_R(rho)) %*% projection + diag(S * T) - 
                projection)
            tulpa:::cpp_test_st_ar1_constrained_prior(rho, S, T, type)[1] - as.numeric(determinant(Q)$modulus)/2
        }
        expect_lt(diff(range(vapply(c(-0.6, 0, 0.5, 0.9), log_mass, 0))), 0.001)
        for (rho in c(-0.6, 0.3, 0.8)) {
            h <- 1e-06
            fd <- (tulpa:::cpp_test_st_ar1_constrained_prior(rho + h, S, T, type)[1] - tulpa:::cpp_test_st_ar1_constrained_prior(rho - 
                h, S, T, type)[1])/(2 * h)
            expect_equal(tulpa:::cpp_test_st_ar1_constrained_prior(rho, S, T, type)[2], fd, tolerance = 1e-06)
        }
    }
})
