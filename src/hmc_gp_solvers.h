enum class GPSolver { AUTO, CHOLESKY, CG, PCG };
struct GPSolverConfig {
    GPSolver solver = GPSolver::CHOLESKY;
    double cg_tol = 1e-6;
    int cg_maxiter = 200;
    GPSolver effective_solver() const { return solver; }
};

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
// Single source of truth for "solve C * x = rhs" inside the per-observation
// NNGP loop, against a factor of C that is already in `llt`. Branches on
// `cfg.effective_solver()` and either
//   (a) solves through the precomputed Eigen LLT factor (Cholesky path), or
//   (b) calls dense_cg_solve / dense_pcg_solve from hmc_gp_cg.h, which has no
//       factor to reuse and ignores `llt`.
// Returns true on success, false on CG non-convergence.
//
// CG is an explicit choice on GPSolverConfig; we do NOT
// silently fall back to Cholesky on CG failure — the caller treats failure
// the same way as a Cholesky non-PSD failure (typically: -INFINITY for
// log-lik, or zero contribution for gradients), so HMC will reject the step.
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

// The first solve of a neighbour system: on the Cholesky path factor the
// leading n_nb x n_nb block of C into `llt` (reused as workspace, and left
// holding the factor for later solve_neighbor_system_second calls), then
// solve. Returns false on a non-PSD factorization or CG non-convergence.
inline bool solve_neighbor_system(
    Eigen::MatrixXd& C_eigen, int n_nb,
    const Eigen::VectorXd& c_eigen,
    Eigen::VectorXd& alpha_out,
    Eigen::LLT<Eigen::MatrixXd>& llt,
    const GPSolverConfig& cfg
) {
  GPSolver effective = cfg.effective_solver();
  if (effective != GPSolver::CG && effective != GPSolver::PCG) {
    llt.compute(C_eigen.topLeftCorner(n_nb, n_nb));
    if (llt.info() != Eigen::Success) return false;
  }
  return solve_neighbor_system_second(C_eigen, n_nb, c_eigen, alpha_out, llt,
                                      cfg);
}

