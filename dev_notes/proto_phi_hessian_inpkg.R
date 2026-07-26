# In-package check of the wired phi Hessian: the closed (k+1)x(k+1) outer
# Hessian from .laplace_exact_re_grad(want_hessian, want_phi) against a central
# difference of the analytic outer gradient (itself the object the optimizer
# drove). This is the proto's math check, now run through the actual assembly:
# the family derivatives, the phi column/diagonal, and the log-phi J column.
REPO <- "C:/Users/Gilles Colling/Documents/dev/tulpa"
suppressMessages(devtools::load_all(REPO, quiet = TRUE))
options(digits = 10)
gfn <- function(nm) getFromNamespace(nm, "tulpa")
block_layout <- gfn(".re_cov_block_layout")
theta_to_L   <- gfn(".re_cov_theta_to_L_list")
L_to_theta   <- gfn(".re_cov_L_list_to_theta")
build_re     <- gfn(".re_cov_build_re_list")
exact_grad   <- gfn(".laplace_exact_re_grad")

sim <- function(fam, seed = 11, G = 8L, per = 8L) {
  set.seed(seed)
  n <- G * per; grp <- rep(seq_len(G), each = per)
  X <- cbind(1, rnorm(n))
  b <- rnorm(G, 0, 0.7)
  eta <- as.numeric(X %*% c(0.3, 0.5)) + b[grp]
  phi0 <- switch(fam, gaussian = 0.55, neg_binomial_2 = 2.5)
  y <- switch(fam,
    gaussian       = rnorm(n, eta, sqrt(phi0)),
    neg_binomial_2 = rnbinom(n, size = phi0, mu = exp(eta)))
  list(y = y, X = X, grp = grp, G = G, fam = fam, phi0 = phi0,
       re_terms = list(list(idx = grp, n_groups = G, n_coefs = 1L)))
}

check <- function(fam) {
  d <- sim(fam)
  layout <- block_layout(d$re_terms, length(d$y))
  k <- sum(vapply(layout, `[[`, integer(1), "k"))

  # g(chi) and H(chi): fit at (theta_re, phi), read the analytic grad + Hessian.
  gH <- function(chi, want_h = FALSE) {
    theta_re <- chi[seq_len(k)]; phi <- exp(chi[[k + 1L]])
    L_list <- theta_to_L(theta_re, layout)
    fit <- tulpa_laplace(y = d$y, n_trials = rep(1, length(d$y)), X = d$X,
      re_list = build_re(L_list, layout), family = fam, phi = phi,
      return_hessian = TRUE, return_joint_hessian = TRUE,
      max_iter = 200L, tol = 1e-12)
    r <- exact_grad(fit = fit, y = d$y, X = d$X, n_trials = rep(1, length(d$y)),
      offset = NULL, weights = NULL, re_list = build_re(L_list, layout),
      layout = layout, L_list = L_list, family = fam, phi = phi,
      want_hessian = want_h, want_phi = TRUE)
    if (want_h) list(g = r$grad, H = r$H, J = r$J) else r
  }

  chi0 <- c(log(0.7), log(d$phi0))
  out <- gH(chi0, want_h = TRUE)
  h <- 1e-5
  Hfd <- matrix(0, k + 1L, k + 1L)
  for (j in seq_len(k + 1L)) {
    cp <- chi0; cp[j] <- cp[j] + h
    cm <- chi0; cm[j] <- cm[j] - h
    Hfd[, j] <- (gH(cp) - gH(cm)) / (2 * h)
  }
  Hfd <- (Hfd + t(Hfd)) / 2
  rel <- max(abs(out$H - Hfd) / pmax(abs(Hfd), 1e-6))
  cat(sprintf("%-15s  H (closed vs grad-FD) rel = %.2e   H_tp = %+.4f   dim(J)=%dx%d\n",
              fam, rel, out$H[1, k + 1L], nrow(out$J), ncol(out$J)))
  invisible(rel)
}

cat("in-package phi Hessian (chi = [theta..., log phi])\n")
cat("==================================================\n")
r1 <- check("gaussian")
r2 <- check("neg_binomial_2")
stopifnot(r1 < 1e-4, r2 < 1e-4)
cat("OK\n")
