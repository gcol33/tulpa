// re_cov_grad_export.cpp
// R-facing surface for recov_block_grad (src/re_cov_chol.h:115), the chain rule
// from a Sigma-space gradient onto one block's log-Cholesky coordinates.
//
// The AGHQ path reaches that routine through cpp_aghq_objective_grad, which is
// tied to the AGHQ oracle and to the Fisher-identity second moment. The exact
// Laplace gradient needs the SAME chain rule against a different second moment
// (R + V + C, see dev_notes/laplace_exact_gradient.md), so it needs the chain
// rule on its own. Re-deriving it in R would be a second copy of the log L_ii /
// log sigma_i / LKJ algebra, which is exactly the drift this export avoids.

#include <Rcpp.h>
#include <Eigen/Dense>
#include <vector>

#include "re_cov_chol.h"

// Sigma-space gradient -> log-Cholesky coordinates for one RE block.
//
// `Smat` is the symmetric nc x nc matrix the caller has already assembled in
// the convention recov_block_grad expects, namely
//     Smat = 0.5 * Omega * (M - G * Sigma) * Omega
// for a second moment M. `L` is the block's Cholesky factor with Sigma = L L'.
// Returns the k = nc(nc+1)/2 (correlated) or nc (diagonal) coordinates in
// column-major lower-triangular order, matching .re_cov_theta_to_L_list().
//
// [[Rcpp::export]]
Rcpp::NumericVector cpp_recov_block_grad(Rcpp::NumericMatrix Smat,
                                         Rcpp::NumericMatrix L,
                                         bool full,
                                         double lkj_eta = 1.0) {
  const int nc = Smat.nrow();
  if (Smat.ncol() != nc) {
    Rcpp::stop("cpp_recov_block_grad: Smat must be square (got %d x %d).",
               nc, (int)Smat.ncol());
  }
  if (L.nrow() != nc || L.ncol() != nc) {
    Rcpp::stop("cpp_recov_block_grad: L must be %d x %d (got %d x %d).",
               nc, nc, (int)L.nrow(), (int)L.ncol());
  }

  Eigen::MatrixXd S(nc, nc), Lm(nc, nc);
  for (int j = 0; j < nc; j++) {
    for (int i = 0; i < nc; i++) {
      S(i, j) = Smat(i, j);
      Lm(i, j) = L(i, j);
    }
  }

  // The two-argument constructor owns the `full && nc > 1` guard and the k
  // formula, so a 1 x 1 block declared correlated collapses the same way here
  // as everywhere else.
  tulpa::ReCovBlock b(nc, full);

  std::vector<double> out(b.k, 0.0);
  tulpa::recov_block_grad(S, Lm, b, lkj_eta, out.data());
  return Rcpp::wrap(out);
}
