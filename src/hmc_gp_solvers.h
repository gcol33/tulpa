inline GPSolver parse_gp_solver(const std::string& s) {
  if (s == "auto") return GPSolver::AUTO;
  if (s == "cholesky") return GPSolver::CHOLESKY;
  if (s == "cg") return GPSolver::CG;
  if (s == "pcg") return GPSolver::PCG;
  if (s == "gpu") return GPSolver::GPU;
  return GPSolver::AUTO;
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
// CG is an explicit user choice (`spatial_gp(solver = "cg")`); we do NOT
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
    // CG path: copy the (top-left n_nb x n_nb) block of C into a row-major
    // scratch buffer for the solver.
    static thread_local std::vector<double> C_buf;
    static thread_local std::vector<double> b_buf;
    static thread_local std::vector<double> x_buf;
    C_buf.resize(n_nb * n_nb);
    b_buf.resize(n_nb);
    x_buf.resize(n_nb);
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = 0; j2 < n_nb; j2++) {
        C_buf[j1 * n_nb + j2] = C_eigen(j1, j2);
      }
      b_buf[j1] = c_eigen(j1);
    }
    int it = (effective == GPSolver::PCG)
      ? dense_pcg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                        cfg.cg_tol, cfg.cg_maxiter)
      : dense_cg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                       cfg.cg_tol, cfg.cg_maxiter);
    if (it < 0) return false;
    if (alpha_out.size() < n_nb) alpha_out.resize(n_nb);
    for (int j = 0; j < n_nb; j++) alpha_out(j) = x_buf[j];
    return true;
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
    static thread_local std::vector<double> C_buf;
    static thread_local std::vector<double> b_buf;
    static thread_local std::vector<double> x_buf;
    C_buf.resize(n_nb * n_nb);
    b_buf.resize(n_nb);
    x_buf.resize(n_nb);
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = 0; j2 < n_nb; j2++) {
        C_buf[j1 * n_nb + j2] = C_eigen(j1, j2);
      }
      b_buf[j1] = rhs(j1);
    }
    int it = (effective == GPSolver::PCG)
      ? dense_pcg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                        cfg.cg_tol, cfg.cg_maxiter)
      : dense_cg_solve(C_buf.data(), n_nb, b_buf.data(), x_buf.data(),
                       cfg.cg_tol, cfg.cg_maxiter);
    if (it < 0) return false;
    if (out.size() < n_nb) out.resize(n_nb);
    for (int j = 0; j < n_nb; j++) out(j) = x_buf[j];
    return true;
  }

  if (out.size() < n_nb) out.resize(n_nb);
  out.head(n_nb) = llt.solve(rhs.head(n_nb));
  return true;
}

