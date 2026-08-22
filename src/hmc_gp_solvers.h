inline GPSolver parse_gp_solver(const std::string& s) {
  static const tulpa::EnumEntry<GPSolver> table[] = {
      {"auto", GPSolver::AUTO},
      {"cholesky", GPSolver::CHOLESKY},
      {"cg", GPSolver::CG},
      {"pcg", GPSolver::PCG},
      {"gpu", GPSolver::GPU}
  };
  return tulpa::parse_enum(s, table, GPSolver::AUTO);
}

// The iterative branch of both entries below: copy the leading n_nb x n_nb
// block of C and the right-hand side into row-major scratch, solve, and write
// the answer back. Function-local buffers: the system is k x k with k <= nn
// (typically 10-20), so the allocation is negligible next to the solve, and
// lazily-initialized thread_local vectors inside an OpenMP region corrupt the
// heap under the mingw toolchain.
inline bool cg_neighbor_solve(
    const Eigen::MatrixXd& C_eigen, int n_nb,
    const Eigen::VectorXd& rhs,
    Eigen::VectorXd& out,
    const GPSolverConfig& cfg,
    bool preconditioned
) {
  std::vector<double> C_buf((size_t)n_nb * n_nb);
  std::vector<double> b_buf(n_nb);
  std::vector<double> x_buf(n_nb);
  for (int j1 = 0; j1 < n_nb; j1++) {
    for (int j2 = 0; j2 < n_nb; j2++) {
      C_buf[j1 * n_nb + j2] = C_eigen(j1, j2);
    }
    b_buf[j1] = rhs(j1);
  }
  int it = preconditioned
    ? dense_pcg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                      cfg.cg_tol, cfg.cg_maxiter)
    : dense_cg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                     cfg.cg_tol, cfg.cg_maxiter);
  if (it < 0) return false;
  if (out.size() < n_nb) out.resize(n_nb);
  for (int j = 0; j < n_nb; j++) out(j) = x_buf[j];
  return true;
}

// -----------------------------------------------------------------------------
// Neighbor-system solver dispatch
// -----------------------------------------------------------------------------
//
// Single source of truth for "solve C * alpha = c" inside the per-observation
// NNGP loop. Branches on `cfg.effective_solver()` and either
//   (a) does an Eigen LLT factorization in-place (Cholesky path), or
//   (b) calls dense_cg_solve / dense_pcg_solve from hmc_gp_cg.h.
//
// `llt` is reused as workspace by the Cholesky branch and ignored by CG.
// Returns true on success, false on failure (non-PSD or CG non-convergence).
//
// CG is an explicit choice on GPSolverConfig; we do NOT
// silently fall back to Cholesky on CG failure — the caller treats failure
// the same way as a Cholesky non-PSD failure (typically: -INFINITY for
// log-lik, or zero contribution for gradients), so HMC will reject the step.
inline bool solve_neighbor_system(
    Eigen::MatrixXd& C_eigen, int n_nb,
    const Eigen::VectorXd& c_eigen,
    Eigen::VectorXd& alpha_out,
    Eigen::LLT<Eigen::MatrixXd>& llt,
    const GPSolverConfig& cfg
) {
  GPSolver effective = cfg.effective_solver();

  if (effective == GPSolver::CG || effective == GPSolver::PCG) {
    return cg_neighbor_solve(C_eigen, n_nb, c_eigen, alpha_out, cfg,
                             effective == GPSolver::PCG);
  }

  // Default / Cholesky path
  llt.compute(C_eigen.topLeftCorner(n_nb, n_nb));
  if (llt.info() != Eigen::Success) return false;
  if (alpha_out.size() < n_nb) alpha_out.resize(n_nb);
  alpha_out.head(n_nb) = llt.solve(c_eigen.head(n_nb));
  return true;
}

// Same as above but solves a SECOND system reusing the already-factored
// matrix when possible. For Cholesky, that's `llt.solve(rhs)`. For CG, we
// just call the iterative solver again — there is no factor to reuse.
inline bool solve_neighbor_system_second(
    const Eigen::MatrixXd& C_eigen, int n_nb,
    const Eigen::VectorXd& rhs,
    Eigen::VectorXd& out,
    const Eigen::LLT<Eigen::MatrixXd>& llt,
    const GPSolverConfig& cfg
) {
  GPSolver effective = cfg.effective_solver();

  if (effective == GPSolver::CG || effective == GPSolver::PCG) {
    return cg_neighbor_solve(C_eigen, n_nb, rhs, out, cfg,
                             effective == GPSolver::PCG);
  }

  if (out.size() < n_nb) out.resize(n_nb);
  out.head(n_nb) = llt.solve(rhs.head(n_nb));
  return true;
}

