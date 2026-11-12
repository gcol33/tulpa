// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "test_prior_model.h"

#include <tulpa/walnuts.h>
#include "hmc_mass_drift.h"

template<class Run>
auto with_test_metric(std::string type, std::string block, int n, Run run) {
    using namespace tulpa_hmc;
    DenseMassMatrix mass;
    const auto kind = type == "dense" ? MassMatrixType::DENSE
                    : type == "block" ? MassMatrixType::BLOCK_DIAG : MassMatrixType::DIAG;
    mass.init(n,kind); mass.adapted = true;
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n,n);
    for (int i = 0; i < n; ++i) {
        C(i,i) = type == "identity" ? 1.0 : 1.2+0.1*i;
        mass.inv_mass_diag[i] = C(i,i);
        mass.sqrt_mass_diag[i] = 1/std::sqrt(C(i,i));
    }
    if (type == "dense") {
        C.array() += 0.08;
        Eigen::MatrixXd L = C.llt().matrixL();
        mass.inv_mass_dense.assign(C.data(),C.data()+n*n);
        mass.L_inv_mass.assign(L.data(),L.data()+n*n);
    } else if (type == "block") {
        mass.in_block.assign(n,false);
        for (int start : {0,2}) {
            MassBlock b; b.init(start,2); b.adapted = true;
            C(start,start+1) = C(start+1,start) = 0.12;
            Eigen::Matrix2d B = C.block<2,2>(start,start);
            Eigen::Matrix2d L = B.llt().matrixL();
            for(int j=0;j<2;++j) for(int i=0;i<2;++i) {
                b.inv_mass[i*4+j] = B(i,j); b.L_inv_mass[i*4+j] = L(i,j);
            }
            mass.blocks.push_back(b);
            mass.in_block[start] = mass.in_block[start+1] = true;
        }
    }
    if (block != "none") {
        Eigen::Matrix2d Q; Q << 3.0,0.4,0.4,2.0;
        Eigen::Matrix2d inverse = Q.inverse();
        if(block=="kronecker") {
            Eigen::Matrix2d Qt;Qt << 2.0,0.2,0.2,1.5;
            const Eigen::Matrix2d Ti=Qt.inverse();
            for(int i=1;i<5;++i) { C.row(i).setZero();C.col(i).setZero(); }
            for(int t=0;t<2;++t) for(int u=0;u<2;++u)
                for(int a=0;a<2;++a) for(int b=0;b<2;++b)
                    C(1+a+2*t,1+b+2*u)=inverse(a,b)*Ti(t,u);
        } else {
            C.row(1).setZero();C.row(2).setZero();
            C.col(1).setZero();C.col(2).setZero();
            C.block<2,2>(1,1)=inverse;
        }
        mass.type=MassMatrixType::DENSE;
        const Eigen::MatrixXd L=C.llt().matrixL();
        mass.inv_mass_dense.assign(C.data(),C.data()+n*n);
        mass.L_inv_mass.assign(L.data(),L.data()+n*n);
    }
    return run(mass,C);
}

void metric_gaussian_gradient(const std::vector<double>& q,
    const tulpa_hmc::ModelData&, const tulpa_hmc::ParamLayout&,
    std::vector<double>& gradient, double* lp) {
    gradient.resize(q.size());
    double sum = 0.0;
    for (size_t i=0;i<q.size();++i) { gradient[i] = -q[i]; sum += q[i]*q[i]; }
    if(lp) *lp = -0.5*sum;
}

void reallocating_gaussian_gradient(const std::vector<double>& q,
    const tulpa_hmc::ModelData& data, const tulpa_hmc::ParamLayout& layout,
    std::vector<double>& gradient, double* lp) {
    std::vector<double> replacement;
    metric_gaussian_gradient(q,data,layout,replacement,lp);
    gradient.swap(replacement);
}

template<class Metric>
struct TestWalnutsModel {
    const Metric& metric;
    const tulpa_hmc::ModelData& data;
    const tulpa_hmc::ParamLayout& layout;
    tulpa_hmc::GradientFn fn;
    void gradient(const std::vector<double>& q, std::vector<double>& g, double* lp) {
        fn(q,data,layout,g,lp);
    }
    void drift(double step, double* q, const double* p, double* scratch) const {
        tulpa_hmc::apply_drift(step,q,p,metric,scratch,metric.n);
    }
    double kinetic_energy(const double* p) const { return metric.kinetic_energy(p); }
};

// [[Rcpp::export]]
Rcpp::List cpp_test_gradient_trajectory(std::string type, std::string block,
    std::string mode, std::string path, int n = 5, int steps = 7,
    double epsilon = 0.1) {
  using namespace tulpa_hmc;
  const auto selected = mode == "reallocate" ? GradientMode::AUTO : parse_gradient_mode(mode);
  set_gradient_mode(selected);
  struct ModeScope { ~ModeScope() { set_gradient_mode(GradientMode::AUTO); } } scope;
  ModelData data;
  tulpa::LikelihoodSpec spec;
  tulpa_probe::gaussian_prior(data,spec,n);
  if(mode=="reallocate") spec.gradient_fn=reallocating_gaussian_gradient;
  const auto layout = compute_param_layout(data);
  const auto fn = mode == "reallocate" ? reallocating_gaussian_gradient
                                       : resolve_gradient_fn(selected,data,layout);
  return with_test_metric(type,block,n,[&](const auto& mass,const auto& C) {
   const auto run = [&](const auto& metric) {
    std::vector<double> q0(n), p0(n), q(n), p(n), g(n), scratch(n), displacement(n);
    const double scale=mode=="N" ? 1e-3 : 1.0;
    for (int i=0;i<n;++i) {
      q0[i]=scale*(0.1+0.15*i);
      p0[i]=scale*(0.3+0.2*i);
    }
    q=q0; p=p0;
    double lp;
    fn(q,data,layout,g,&lp);
    NUTSWorkspace ws; ws.init(n,2); ws.gradient_fn=fn;
    tulpa::WalnutsWorkspace wws; wws.init(n,2);
    TestWalnutsModel<std::decay_t<decltype(metric)>> model{metric,data,layout,fn};
    using Model = std::decay_t<decltype(model)>;
    struct Context {
      Model& model;
      tulpa::WalnutsWorkspace& ws;
    } context{model,wws};
    const auto step = [&]() {
      if (path == "copy" || path == "cached") {
        auto out=leapfrog_step(q,p,epsilon,data,layout,nullptr,&mass);
        q=std::move(out.q); p=std::move(out.p); lp=out.log_prob;
        fn(q,data,layout,g,nullptr);
      } else if (path == "inplace" || path == "tree") {
        std::copy(q.begin(),q.end(),ws.q_at(0));
        std::copy(p.begin(),p.end(),ws.p_at(0));
        std::copy(g.begin(),g.end(),ws.grad_at(0));
        ws.logp_at(0)=lp;
        if (path == "inplace") leapfrog_step_inplace(ws,0,epsilon,mass,data,layout);
        else {
          std::mt19937 rng(229);
          build_tree_fast(ws,0,1,0,epsilon,mass,-lp+mass.kinetic_energy(p.data()),
                          1000.0,data,layout,rng);
        }
        q.assign(ws.q_at(0),ws.q_at(0)+n);
        p.assign(ws.p_at(0),ws.p_at(0)+n);
        g.assign(ws.grad_at(0),ws.grad_at(0)+n); lp=ws.logp_at(0);
      } else if (path == "walnuts") {
        auto& state=wws.step_state;
        state.theta=q; state.rho=p; state.grad=g;
        tulpa::walnuts_detail::leapfrog(context,state,epsilon,1);
        q=state.theta; p=state.rho; g=state.grad; lp=state.logp_pos;
      } else {
        Rcpp::stop("Unknown trajectory path");
      }
    };
    for (int j=0;j<steps;++j) step();
    const auto forward_q=q, forward_p=p, forward_g=g;
    for (double& x:p) x=-x;
    for (int j=0;j<steps;++j) step();
    std::mt19937 rng(170183), clone=rng;
    std::vector<double> search_p(n);
    metric.sample_momentum(search_p.data(),clone);
    const double search_epsilon=find_reasonable_epsilon(q0,data,layout,rng,mass);
    std::vector<std::vector<double>> momentum_draws;
    clone=std::mt19937(170183);
    while(clone!=rng && momentum_draws.size()<201) {
      std::vector<double> draw(n);mass.sample_momentum(draw.data(),clone);
      momentum_draws.push_back(std::move(draw));
    }
    Rcpp::NumericMatrix momenta(momentum_draws.size(),n);
    for(size_t i=0;i<momentum_draws.size();++i) for(int j=0;j<n;++j) momenta(i,j)=momentum_draws[i][j];
    return Rcpp::List::create(Rcpp::Named("q0")=q0,Rcpp::Named("p0")=p0,
      Rcpp::Named("q")=forward_q,Rcpp::Named("p")=forward_p,
      Rcpp::Named("grad")=forward_g,Rcpp::Named("reverse_q")=q,
      Rcpp::Named("reverse_p")=p,Rcpp::Named("inverse_mass")=C,
      Rcpp::Named("epsilon")=search_epsilon,Rcpp::Named("momenta")=momenta);
   };
   return run(mass);
  });
}

// [[Rcpp::export]]
Rcpp::List cpp_test_metric_step(std::string type, std::string block, int n = 5,
                                double epsilon = 0.15, bool extreme = false) {
  using namespace tulpa_hmc;
  return with_test_metric(type,block,n,[&](const auto& mass,const auto& C) {
    NUTSWorkspace ws; ws.init(n,1); ws.gradient_fn = metric_gaussian_gradient;
    Rcpp::NumericVector q0(n), p0(n);
    for(int i=0;i<n;++i) {
      q0[i] = extreme ? std::numeric_limits<double>::infinity() : 0.1+0.15*i;
      p0[i] = 0.3+0.2*i;
      ws.q_at(0)[i] = q0[i]; ws.p_at(0)[i] = p0[i]; ws.grad_at(0)[i] = -q0[i];
    }
    ModelData data; ParamLayout layout;
    const auto step = leapfrog_step_inplace(ws,0,epsilon,mass,data,layout);
    Rcpp::NumericVector q(ws.q_at(0),ws.q_at(0)+n), p(ws.p_at(0),ws.p_at(0)+n);
    return Rcpp::List::create(Rcpp::Named("q0")=q0,Rcpp::Named("p0")=p0,
        Rcpp::Named("q")=q,Rcpp::Named("p")=p,Rcpp::Named("inverse_mass")=C,
        Rcpp::Named("log_prob")=step.log_prob,Rcpp::Named("divergent")=step.divergent);
  });
}

namespace {
thread_local int metric_gradient_calls=0;
void counted_metric_gradient(const std::vector<double>& q,const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,std::vector<double>& g,double* lp) {
  ++metric_gradient_calls;
  metric_gaussian_gradient(q,data,layout,g,lp);
}
}

// [[Rcpp::export]]
Rcpp::List cpp_test_metric_epsilon(std::string type,int n=5) {
  using namespace tulpa_hmc;
  return with_test_metric(type,"none",n,[&](const auto& mass,const auto& C) {
    ModelData data; tulpa::LikelihoodSpec spec;
    tulpa_probe::gaussian_prior(data,spec,n);
    spec.gradient_fn=counted_metric_gradient;
    const auto layout=compute_param_layout(data);
    std::vector<double> q(n,0.3),p(n);
    std::mt19937 rng(170183),clone=rng;
    metric_gradient_calls=0;
    const double epsilon=find_reasonable_epsilon(q,data,layout,rng,mass);
    const int trials=metric_gradient_calls/3;
    Rcpp::NumericMatrix momenta(trials,n);
    for(int i=0;i<trials;++i) {
      mass.sample_momentum(p.data(),clone);
      for(int j=0;j<n;++j) momenta(i,j)=p[j];
    }
    return Rcpp::List::create(Rcpp::Named("epsilon")=epsilon,
      Rcpp::Named("calls")=metric_gradient_calls,Rcpp::Named("q")=q,
      Rcpp::Named("momenta")=momenta,Rcpp::Named("inverse_mass")=C);
  });
}

// [[Rcpp::export]]
Rcpp::List cpp_test_metric_geometry(std::string type, std::string block,
                                    int n = 5, int n_draws = 20000) {
  return with_test_metric(type,block,n,[&](const auto& mass,const auto& C) {
    Eigen::VectorXd p = Eigen::VectorXd::LinSpaced(n,0.3,1.2), velocity(n);
    mass.inv_mass_times_p(p.data(),velocity.data());
    Eigen::VectorXd qminus = Eigen::VectorXd::Zero(n), qplus = qminus, scratch(n);
    qplus[0] = 1.0; qplus[1] = -1.0;
    const bool uturn = !tulpa_hmc::compute_criterion(velocity.data(),velocity.data(),qplus.data(),n);
    Eigen::MatrixXd draws(n,n_draws);
    std::mt19937 rng(170183);
    for (int j=0;j<n_draws;++j) mass.sample_momentum(draws.col(j).data(),rng);
    Eigen::VectorXd mean = draws.rowwise().mean();
    Eigen::MatrixXd centered = draws.colwise()-mean;
    Eigen::MatrixXd covariance = centered*centered.transpose()/(n_draws-1.0);
    return Rcpp::List::create(Rcpp::Named("inverse_mass")=C,
        Rcpp::Named("momentum")=p,Rcpp::Named("velocity")=velocity,
        Rcpp::Named("kinetic")=mass.kinetic_energy(p.data()),
        Rcpp::Named("covariance")=covariance,Rcpp::Named("mean")=mean,
        Rcpp::Named("uturn")=uturn);
  });
}
