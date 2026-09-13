# Is logLik() on a nested-Laplace fit the model evidence?
#
# logLik.tulpa_fit() returns logsumexp(log_marginal) over the outer grid
# (R/methods_generic.R). The engine builds the same grid's posterior weights from
# log_marginal + log_quad, log_quad being each cell's prior mass
# (.nl_normalise_weights_safe(..., log_quad =), .nl_grid_log_quad()). If
# log_marginal is the integrand p(y | theta_k), the evidence is
# logsumexp(log_marginal + log_quad), and the unweighted sum depends on the grid.
#
# check_fit() takes any fit (or joint_fit) carrying log_marginal, log_quad and
# weights, and reports:
#   - which formula reproduces the stored weights (settles what log_marginal is)
#   - the two log-sums and their difference
# Run it on two fits of the same data at different node counts to see which of
# the two quantities moves with the grid.

lse <- function(v) { v <- v[is.finite(v)]; m <- max(v); m + log(sum(exp(v - m))) }

check_fit <- function(jf, label = "") {
  lm <- jf$log_marginal; lq <- jf$log_quad; w <- jf$weights / sum(jf$weights)
  stopifnot(!is.null(lm), !is.null(lq), !is.null(w), length(lq) == length(lm))
  ok <- is.finite(lm) & is.finite(lq)
  A <- lse(lm); B <- lse(lm[ok] + lq[ok])
  wq <- numeric(length(lm)); wq[ok] <- exp(lm[ok] + lq[ok] - B)
  wf <- numeric(length(lm)); wf[is.finite(lm)] <- exp(lm[is.finite(lm)] - A)
  cat(sprintf(paste0("%s K=%d | stored weights reproduced by: +log_quad %.1e, flat %.1e",
                     " | sum(exp(log_quad)) %.4f, sd %.3f\n",
                     "    logLik formula lse(lm) = %.3f | quadrature lse(lm + lq) = %.3f | diff %.3f\n"),
              label, sum(ok), max(abs(wq - w)), max(abs(wf - w)),
              sum(exp(lq[is.finite(lq)])), sd(lq[is.finite(lq)]), A, B, A - B))
  invisible(c(A = A, B = B))
}

# Recorded on base fits stored inside tulpaObs SBC arms (tulpa 0.3.0,
# tulpaObs 0.2.0, beta occu_cover, same simulated data within each pair,
# only the dispersion node count differing):
#
#   J=10 phi 21 nodes: K=2653  weights +log_quad 6.5e-16, flat 3.1e-02
#                      lse(lm) -155.714  lse(lm+lq) -164.805  diff 9.091
#   J=10 phi 55 nodes: K=6938  weights +log_quad 8.3e-17, flat 1.6e-02
#                      lse(lm) -154.836  lse(lm+lq) -164.553  diff 9.717
#   J=3  phi 11 nodes: K= 864  weights +log_quad 5.3e-16, flat 5.8e-02
#                      lse(lm)  -94.912  lse(lm+lq) -102.581  diff 7.669
#   J=3  phi 55 nodes: K=4296  weights +log_quad 9.2e-16, flat 2.0e-02
#                      lse(lm)  -93.325  lse(lm+lq) -102.578  diff 9.253
#
#   J=10, 21 -> 55 nodes: logLik formula +0.878, quadrature evidence +0.252
#   J=3,  11 -> 55 nodes: logLik formula +1.587, quadrature evidence +0.003
#
# log_quad sd was 0.85 to 0.93 on these tensor grids, and sum(exp(log_quad))
# 0.76 to 0.82 rather than 1.
