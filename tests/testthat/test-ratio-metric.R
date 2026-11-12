metric_reference_epsilon <- function(q, momenta, C) {
    epsilon <- 1
    trial <- function(p) {
        H0 <- sum(q * q)/2 + drop(crossprod(p, C %*% p))/2
        half <- p - epsilon * q/2
        qt <- q + epsilon * drop(C %*% half)
        pt <- half - epsilon * qt/2
        H0 - sum(qt * qt)/2 - drop(crossprod(pt, C %*% pt))/2
    }
    threshold <- log(0.8)
    direction <- if (trial(momenta[1, ]) > threshold) 
        1
    else -1
    used <- 1L
    for (i in seq.int(2L, nrow(momenta))) {
        used <- i
        delta <- trial(momenta[i, ])
        if ((direction == 1 && delta <= threshold) || (direction == -1 && delta >= threshold)) 
            break
        epsilon <- epsilon * 2^direction
    }
    stopifnot(used == nrow(momenta))
    list(epsilon = epsilon, calls = 3L * used)
}

test_that("metric products, energies and momenta match an explicit matrix", {
    for (n in c(5L, 17L)) for (type in c("identity", "diag", "dense", "block")) {
        for (block in c("none", "precision", "sparse", "kronecker")) {
            z <- tulpa:::cpp_test_metric_geometry(type, block, n)
            C <- z$inverse_mass
            p <- z$momentum
            expect_equal(drop(z$velocity), drop(C %*% p), tolerance = 1e-12)
            expect_equal(z$kinetic, drop(crossprod(p, C %*% p))/2, tolerance = 1e-12)
            velocity <- drop(C %*% p)
            expect_identical(z$uturn, velocity[1] - velocity[2] < 0)
            M <- solve(C)
            se <- sqrt((outer(diag(M), diag(M)) + M^2)/20000)
            expect_lt(max(abs(z$covariance - M)/se), 5)
            expect_lt(max(abs(z$mean)/sqrt(diag(M)/20000)), 5)
        }
    }
})

test_that("the production leapfrog follows the metric Hamiltonian", {
    for (n in c(5L, 17L)) for (type in c("identity", "diag", "dense", "block")) {
        for (block in c("none", "precision", "sparse", "kronecker")) {
            epsilon <- 0.15
            z <- tulpa:::cpp_test_metric_step(type, block, n, epsilon)
            half <- z$p0 - epsilon * z$q0/2
            q <- z$q0 + epsilon * drop(z$inverse_mass %*% half)
            p <- half - epsilon * q/2
            expect_equal(z$q, q, tolerance = 1e-12)
            expect_equal(z$p, p, tolerance = 1e-12)
            expect_equal(z$log_prob, -sum(q * q)/2, tolerance = 1e-12)
            expect_false(z$divergent)
        }
    }
    expect_true(tulpa:::cpp_test_metric_step("dense", "none", 5L, 0.15, TRUE)$divergent)
})

test_that("epsilon search uses one new score per trial with every metric", {
    for (type in c("identity", "diag", "dense", "block")) {
        z <- tulpa:::cpp_test_metric_epsilon(type)
        ref <- metric_reference_epsilon(z$q, z$momenta, z$inverse_mass)
        expect_identical(z$epsilon, ref$epsilon)
        expect_identical(z$calls, ref$calls)
    }
})

test_that("all gradient modes retain reversible metric trajectories after an arena gradient", {
    skip_on_cran()
    invisible(tulpa:::cpp_test_gradient_trajectory("identity", "none", "A_r", "copy"))
    for (n in c(5L, 17L)) for (mode in c("N", "A", "A_r", "H", "auto", "reallocate")) {
        for (type in c("identity", "diag", "dense", "block")) {
            for (block in c("none", "precision", "sparse", "kronecker")) {
                paths <- c("copy", "inplace", "tree", "walnuts")
                for (path in paths) {
                  epsilon <- 0.1
                  steps <- 7L
                  z <- tulpa:::cpp_test_gradient_trajectory(type, block, mode, path, n, steps, epsilon)
                  q <- z$q0
                  p <- z$p0
                  for (j in seq_len(steps)) {
                    p <- p - epsilon * q/2
                    q <- q + epsilon * drop(z$inverse_mass %*% p)
                    p <- p - epsilon * q/2
                  }
                  info <- paste(n, mode, type, block, path, sep = "/")
                  score_error <- if (mode == "N") 
                    64 * .Machine$double.eps * sum(q * q)/1e-05
                  else 0
                  expect_lt(max(abs(z$q - q)), 1e-12 + steps * epsilon * score_error, label = info)
                  expect_lt(max(abs(z$p - p)), 1e-12 + steps * epsilon * score_error, label = info)
                  expect_lt(max(abs(z$grad + z$q)), 1e-12 + score_error, label = info)
                  roundoff <- 64 * .Machine$double.eps * max(1, abs(z$q0), abs(z$p0))
                  expect_lt(max(abs(z$reverse_q - z$q0)), roundoff, label = info)
                  expect_lt(max(abs(z$reverse_p + z$p0)), roundoff, label = info)
                  ref <- metric_reference_epsilon(z$q0, z$momenta, z$inverse_mass)
                  expect_identical(z$epsilon, ref$epsilon, info = info)
                }
            }
        }
    }
})
