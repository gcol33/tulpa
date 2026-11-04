#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]
Rcpp::List exact_ms_grid(Rcpp::List axes, Eigen::MatrixXd D,
                        Eigen::VectorXd y, double phi) {
    if (axes.size() != 4 || D.rows() != y.size() || D.cols() != y.size() ||
        !std::isfinite(phi) || phi <= 0)
        Rcpp::stop("Four axes, a square distance matrix and positive noise variance are required.");
    std::vector<Rcpp::NumericVector> ax;
    for (int j = 0; j < 4; ++j) {
        ax.push_back(Rcpp::as<Rcpp::NumericVector>(axes[j]));
        if (ax.back().size() < 2) Rcpp::stop("Each integration axis needs at least two levels.");
    }
    int n = y.size(), nr = ax[3].size();
    std::vector<Eigen::MatrixXd> El, Er;
    for (double x : ax[1]) El.push_back((-D.array() / std::exp(x)).exp());
    for (double x : ax[3]) Er.push_back((-D.array() / std::exp(x)).exp());
    Eigen::MatrixXd base = Eigen::MatrixXd::Constant(n, n, 6.25);
    base.diagonal().array() += phi;
    std::vector<std::vector<double>> part[4];
    for (int j = 0; j < 4; ++j) part[j].resize(nr, std::vector<double>(ax[j].size(), 0));
    std::vector<double> totals(nr, 0);
    const double ls = -std::log(.05) / 2;
    const double shift = -.5 * n * std::log(2 * M_PI * phi);
    #pragma omp parallel for num_threads(2) schedule(dynamic)
    for (int ir = 0; ir < nr; ++ir) {
        for (int il = 0; il < ax[1].size(); ++il) {
            double lp_r = 0;
            for (int b = 0; b < 2; ++b) {
                double x = b ? ax[3][ir] : ax[1][il];
                double lr = -std::log(.05) * (b ? 1.5 : .2);
                lp_r += std::log(lr) - x - lr * std::exp(-x);
            }
            for (int sl = 0; sl < ax[0].size(); ++sl) {
                Eigen::MatrixXd V0 = base + std::exp(ax[0][sl]) * El[il];
                double sigl = std::exp(.5 * ax[0][sl]);
                for (int sr = 0; sr < ax[2].size(); ++sr) {
                    Eigen::MatrixXd V = V0 + std::exp(ax[2][sr]) * Er[ir];
                    Eigen::LLT<Eigen::MatrixXd> chol(V);
                    Eigen::MatrixXd L = chol.matrixL();
                    Eigen::VectorXd z = chol.matrixL().solve(y);
                    double sigr = std::exp(.5 * ax[2][sr]);
                    double lp = lp_r + 2 * std::log(ls / 2) - ls * (sigl + sigr) +
                        .5 * (ax[0][sl] + ax[2][sr]) -
                        L.diagonal().array().log().sum() - .5 * z.squaredNorm() -
                        .5 * n * std::log(2 * M_PI);
                    double w = std::exp(lp - shift);
                    part[0][ir][sl] += w; part[1][ir][il] += w;
                    part[2][ir][sr] += w; part[3][ir][ir] += w;
                    totals[ir] += w;
                }
            }
        }
    }
    double total = std::accumulate(totals.begin(), totals.end(), 0.0);
    Rcpp::List marg(4);
    double logh = 0;
    for (int j = 0; j < 4; ++j) {
        Rcpp::NumericVector m(ax[j].size());
        for (int ir = 0; ir < nr; ++ir) for (int k = 0; k < m.size(); ++k) m[k] += part[j][ir][k];
        marg[j] = m / total;
        logh += std::log(ax[j][1] - ax[j][0]);
    }
    return Rcpp::List::create(Rcpp::Named("marg") = marg,
        Rcpp::Named("log_evidence") = std::log(total) + shift + logh);
}
