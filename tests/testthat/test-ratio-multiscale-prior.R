test_that("multiscale prior scores differentiate their dense densities at small variances", {
  n <- 8L
  rho <- 0.3
  logit_rho <- qlogis((rho + 1) / 2)
  lambda <- -log(0.01)
  for (structure in c("rw1", "cyclic", "rw2", "ar1", "iid")) {
    D <- switch(structure,
      rw1 = diff(diag(n)),
      cyclic = rbind(diff(diag(n)), diag(n)[1, ] - diag(n)[n, ]),
      rw2 = diff(diag(n), differences = 2),
      ar1 = {
        d <- diag(n)
        d[1, 1] <- sqrt(1 - rho^2)
        d[cbind(2:n, 1:(n - 1))] <- -rho
        d
      },
      iid = diag(n))
    Q <- crossprod(D)
    rank <- if (structure == "rw2") n - 1L else n
    if (structure %in% c("rw1", "cyclic", "rw2")) Q <- Q + 1 / n
    for (x in c(-12, -20, -25, -30)) {
      sigma2 <- exp(x)
      phi <- sqrt(sigma2) * sin(seq_len(n))
      result <- tulpa:::cpp_test_multiscale_prior(structure, phi, x, logit_rho)
      quad <- drop(crossprod(phi, Q %*% phi))
      expected_value <- -quad / (2 * sigma2) - rank / 2 * log(2 * pi * sigma2) +
        log(lambda / 2) + x / 2 - lambda * sqrt(sigma2)
      if (structure == "ar1") expected_value <- expected_value + log(1 - rho^2) / 2
      expected_score <- c(-drop(Q %*% phi) / sigma2,
        -rank / 2 + quad / (2 * sigma2) + (1 - lambda * sqrt(sigma2)) / 2)
      expect_equal(unname(result$value), rep(expected_value, 3), tolerance = 1e-13)
      expect_equal(unname(result$gradient[seq_len(n + 1L), ]),
        matrix(expected_score, n + 1L, 3), tolerance = 1e-12)

      parameters <- c(phi, x, logit_rho)
      for (j in seq_along(parameters)) {
        h <- if (j <= n) sqrt(sigma2) * 1e-4 else 1e-4
        plus <- minus <- parameters
        plus[j] <- plus[j] + h
        minus[j] <- minus[j] - h
        value <- function(p) tulpa:::cpp_test_multiscale_prior(
          structure, p[seq_len(n)], p[n + 1L], p[n + 2L])$value
        finite_difference <- (value(plus) - value(minus)) / (2 * h)
        expect_equal(unname(result$gradient[j, ]), unname(finite_difference),
                     tolerance = 1e-7, info = paste(structure, x, j))
      }
    }
  }
})
