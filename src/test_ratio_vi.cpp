#include <RcppEigen.h>
#include "test_prior_model.h"
#include "vi_meanfield.h"
#include "vi_fullrank.h"
#include "vi_lowrank.h"

namespace {
struct GaussianGradient {
    Eigen::VectorXd mean;
    Eigen::MatrixXd loading;
};
GaussianGradient gaussian_reference(const Eigen::MatrixXd& loading,
                                    std::mt19937& rng) {
    const int draws=10;
    Eigen::MatrixXd noise(loading.cols(),draws);
    std::normal_distribution<double> normal(0,1);
    for(int j=0;j<draws;++j) for(int i=0;i<noise.rows();++i)
        noise(i,j)=normal(rng);
    const Eigen::MatrixXd theta=loading*noise;
    return {-theta.rowwise().mean(),loading-theta*noise.transpose()/draws};
}
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_vi_math() {
    using namespace tulpa::vi;
    tulpa_hmc::set_gradient_mode(tulpa_hmc::GradientMode::HANDCODED);
    struct ResetMode {
        ~ResetMode() { tulpa_hmc::set_gradient_mode(tulpa_hmc::GradientMode::AUTO); }
    } reset;
    tulpa_hmc::ModelData data;
    tulpa::LikelihoodSpec spec;
    tulpa_probe::gaussian_prior(data,spec,2);
    const auto layout=tulpa_hmc::compute_param_layout(data);
    MeanFieldParams mf(2);mf.log_sigma.setZero();
    FullRankParams fr(2);fr.L.setIdentity();
    LowRankParams lr(2,1,1701);lr.L.setZero();lr.L(0,0)=0.6;
    lr.log_d << std::log(0.8),0.0;
    std::mt19937 rng(1701);
    const auto a=compute_meanfield_elbo_grad(mf,data,layout,10,rng);
    const auto b=compute_fullrank_elbo_grad(fr,data,layout,10,rng);
    const auto c=compute_lowrank_elbo_grad(lr,data,layout,10,rng);
    std::mt19937 reference_rng(1701);
    const auto ar=gaussian_reference(Eigen::MatrixXd::Identity(2,2),reference_rng);
    const auto br=gaussian_reference(Eigen::MatrixXd::Identity(2,2),reference_rng);
    Eigen::MatrixXd loading(2,3);
    loading.leftCols(1)=lr.L;
    loading.rightCols(2)=lr.log_d.array().exp().matrix().asDiagonal();
    const auto cr=gaussian_reference(loading,reference_rng);
    const Eigen::MatrixXd lower=br.loading.triangularView<Eigen::Lower>();
    const Eigen::VectorXd scale_grad=cr.loading.rightCols(2).diagonal().array()*lr.log_d.array().exp();
    Eigen::VectorXd diagonal(3);diagonal << 0.5,0.8,1.1;
    Eigen::MatrixXd L(3,2);L << 0.2,0.1,-0.1,0.3,0.4,-0.2;
    Eigen::VectorXd delta(3);delta << 0.4,-1.2,0.7;
    const tulpa::vi::LowRankPrecision precision(diagonal,L);
    Eigen::MatrixXd covariance=L*L.transpose();
    covariance.diagonal().array()+=diagonal.array().square();
    const double inverse_error=(precision.apply(delta)-covariance.llt().solve(delta)).norm();
    Eigen::MatrixXd rotation(2,2);
    rotation << std::cos(0.7),-std::sin(0.7),std::sin(0.7),std::cos(0.7);
    const double alignment_error=(tulpa::vi::align_factor_gradient(L*rotation,2*L*rotation,L)-2*L).norm();
    tulpa::vi::ConvergenceChecker ordinary(1e-4,0.01,50),shifted(1e-4,0.01,50),constant(1e-4,0.01,50);
    bool same=true,rejected=true;
    for(int i=0;i<200;++i) {
      Eigen::VectorXd g(2);g << (i%2 ? 1.0:-1.0),0.0;
      same &= ordinary.check(-1000+i*0.5,g)==shifted.check(i*0.5,g);
      rejected &= constant.check(-1000,Eigen::VectorXd::Constant(2,10)).empty();
    }
    return Rcpp::NumericVector::create(
      Rcpp::Named("meanfield")=(a.grad_mu-ar.mean).norm()+(a.grad_log_sigma-ar.loading.diagonal()).norm(),
      Rcpp::Named("fullrank")=(b.grad_mu-br.mean).norm()+(b.grad_L-lower).norm(),
      Rcpp::Named("lowrank")=(c.grad_mu-cr.mean).norm()+(c.grad_L-cr.loading.leftCols(1)).norm()+(c.grad_log_d-scale_grad).norm(),
      Rcpp::Named("inverse_error")=inverse_error,
      Rcpp::Named("alignment_error")=alignment_error,
      Rcpp::Named("offset_invariant")=same,Rcpp::Named("constant_rejected")=rejected);
}
