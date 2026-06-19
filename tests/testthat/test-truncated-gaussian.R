# Finite-difference + reduction check of the upper-truncated Gaussian kernel
# (gcol33/tulpa#122). The latent response is Normal(eta, sigma^2) conditioned on
# y <= u; the log-density is
#   ll = -0.5 log(2 pi sigma^2) - 0.5 z^2 - log Phi((u - eta)/sigma),
# z = (y - eta)/sigma. The family operates on whatever response it is given (a
# consumer gets a truncated-lognormal by feeding log(cover), u = log(ceiling); the
# response Jacobian is the consumer's). cpp_truncated_gaussian_terms() returns the
# analytic (ll, d ll/d eta, -d2 ll/d eta2), checked against central differences and
# against the plain gaussian arm in the u -> +Inf limit.

# Reference log-density from R (truncated Gaussian, no response Jacobian).
.tg_ll_R <- function(y, u, eta, sigma) {
    z <- (y - eta) / sigma
    logPhi <- if (is.finite(u)) stats::pnorm((u - eta) / sigma, log.p = TRUE) else 0
    -0.5 * log(2 * pi * sigma^2) - 0.5 * z^2 - logPhi
}

test_that("analytic grad / neg_hess match central differences", {
    h <- 1e-5
    grid <- expand.grid(
        y     = c(-3.0, -1.2, -0.2, -0.02),  # response = log-cover scale, y <= u
        eta   = c(-3.0, -1.0, -0.2, 0.1),
        sigma = c(0.4, 0.7, 1.2),
        u     = c(0, log(2))
    )
    for (r in seq_len(nrow(grid))) {
        y <- grid$y[r]; eta <- grid$eta[r]; sg <- grid$sigma[r]; u <- grid$u[r]
        out <- tulpa:::cpp_truncated_gaussian_terms(y, u, eta, sg)

        expect_equal(out[["ll"]], .tg_ll_R(y, u, eta, sg), tolerance = 1e-10)

        g_fd <- (.tg_ll_R(y, u, eta + h, sg) - .tg_ll_R(y, u, eta - h, sg)) / (2 * h)
        expect_equal(out[["grad"]], g_fd, tolerance = 1e-5)

        nh_fd <- -(.tg_ll_R(y, u, eta + h, sg) - 2 * .tg_ll_R(y, u, eta, sg) +
                   .tg_ll_R(y, u, eta - h, sg)) / h^2
        expect_equal(out[["neg_hess"]], nh_fd, tolerance = 1e-3)
        expect_gte(out[["neg_hess"]], 0)   # log-concave in eta
    }
})

test_that("u = +Inf reduces exactly to the gaussian arm", {
    for (y in c(-3.0, -1.2, -0.2)) for (eta in c(-1.0, -0.2)) for (sg in c(0.4, 0.7)) {
        out <- tulpa:::cpp_truncated_gaussian_terms(y, Inf, eta, sg)
        expect_equal(out[["ll"]],  stats::dnorm(y, eta, sg, log = TRUE), tolerance = 1e-12)
        expect_equal(out[["grad"]], (y - eta) / sg^2, tolerance = 1e-12)
        expect_equal(out[["neg_hess"]], 1 / sg^2, tolerance = 1e-12)
    }
})

test_that("deep truncation (predicted mean far above the bound) stays finite", {
    # eta well above u: nearly all mass is truncated; lambda ~ -a is large but the
    # log-space inverse-Mills keeps ll / grad / neg_hess finite, neg_hess floored.
    out <- tulpa:::cpp_truncated_gaussian_terms(-4.6, 0.0, 5.0, 0.3)
    expect_true(is.finite(out[["ll"]]))
    expect_true(is.finite(out[["grad"]]))
    expect_true(is.finite(out[["neg_hess"]]) && out[["neg_hess"]] >= 0)
})

test_that("truncated-normal moment identities hold (MLE target)", {
    eta <- -0.5; sg <- 0.8; u <- 0
    a <- (u - eta) / sg
    lambda <- stats::dnorm(a) / stats::pnorm(a)
    set.seed(1)
    draws <- rnorm(2e6, eta, sg); draws <- draws[draws <= u]
    expect_equal(eta - sg * lambda, mean(draws), tolerance = 5e-3)              # E[t | t <= u]
    expect_equal(sg^2 * (1 - lambda * (a + lambda)), var(draws), tolerance = 5e-3)  # Var[t | t <= u]
})
