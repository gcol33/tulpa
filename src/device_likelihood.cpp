#include "device_likelihood.h"
#include "laplace_builtin_family_spec.h"
#include "tulpa/likelihood_math.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tulpa { namespace device {

LikelihoodWorkspace::LikelihoodWorkspace(ObservationPlan p, DeviceLikelihood k) :
    plan_(std::move(p)), kernel_(std::move(k)) {
    const auto& plan = plan_;
    if (plan.n<1 || plan.np<1 || plan.np>8 || plan.width<1 ||
        plan.rowptr.size()!=static_cast<std::size_t>(plan.n)*plan.np+1 ||
        plan.beta_start.size()!=static_cast<std::size_t>(plan.np) ||
        plan.beta_count.size()!=static_cast<std::size_t>(plan.np) ||
        plan.offset.size()!=static_cast<std::size_t>(plan.n)*plan.np ||
        plan.column.size()!=plan.weight.size() ||
        plan.rowptr.back()!=static_cast<int>(plan.column.size()))
        throw std::invalid_argument("invalid likelihood observation plan");
    for (int col : plan.column) if (col<0 || col>=plan.width)
        throw std::invalid_argument("likelihood projection column outside coefficient vector");
    const int cols = std::accumulate(plan.beta_count.begin(),plan.beta_count.end(),0);
    if (plan.x.size()!=static_cast<std::size_t>(plan.n)*cols)
        throw std::invalid_argument("likelihood dense design size mismatch");
    for (int k=0; k<plan.np; ++k) if (plan.beta_count[k]<0 || plan.beta_start[k]<0 ||
        plan.beta_start[k]+plan.beta_count[k]>plan.width)
        throw std::invalid_argument("likelihood beta span outside coefficient vector");
    if (kernel_.source.empty()) throw std::invalid_argument("empty likelihood device source");
}
LikelihoodWorkspace::~LikelihoodWorkspace() = default;

Status LikelihoodWorkspace::refresh_projection(ObservationPlan plan) {
    if (plan.n!=plan_.n || plan.np!=plan_.np || plan.width!=plan_.width ||
        plan.beta_start!=plan_.beta_start || plan.beta_count!=plan_.beta_count)
        return Status::read("likelihood",-1,"refresh_projection","observation layout changed within a fit");
    Status s;
    if (route_) s = route_->refresh_projection(plan);
    plan_=std::move(plan);
    return s;
}

Status LikelihoodWorkspace::evaluate(const std::vector<double>& parameters, int chains,
    bool batched, bool weights, LikelihoodResult& result) {
    if (chains<1 || chains>65535 || parameters.size()!=static_cast<std::size_t>(chains)*plan_.width)
        return Status::read("likelihood",-1,"evaluate","coefficient dimensions mismatch");
    if (!route_) {
        Backend* b = backend();
        if (!b) return backend_status();
        route_ = b->likelihood(plan_);
        if (!route_) return Status::read("likelihood",-1,"evaluate","the backend returned no route");
    }
    return route_->evaluate(kernel_, parameters, chains, batched, weights, result);
}

bool builtin_device_likelihood(const ModelData& data, const ParamLayout&,
    const void* response, bool initialize, DeviceLikelihood& kernel) {
    const auto& r = *static_cast<const BuiltinFamilyResponse*>(response);
    using math::FusedFamily;
    static const std::vector<std::pair<std::string,FusedFamily>> families = {
        {"gaussian",FusedFamily::Gaussian},{"binomial",FusedFamily::Binomial},
        {"poisson",FusedFamily::Poisson},{"neg_binomial_2",FusedFamily::Negbin2},
        {"gamma",FusedFamily::Gamma},{"beta",FusedFamily::Beta},
        {"beta_binomial",FusedFamily::BetaBinomial},
        {"truncated_poisson",FusedFamily::TruncatedPoisson},
        {"truncated_neg_binomial_2",FusedFamily::TruncatedNegbin2},
        {"neg_binomial_1",FusedFamily::Negbin1}};
    const auto found=std::find_if(families.begin(),families.end(),[&](const auto& x){return x.first==r.family;});
    if (found==families.end() || !r.beta_y.empty() || !r.y || !r.prepared ||
        r.ll_const.size()!=static_cast<std::size_t>(data.N)) return false;
    const bool zi = data.n_processes==2 || data.zi_type!=ZIType::NONE;
    const int family=static_cast<int>(found->second);
    kernel.variant=family*2+(zi ? 1 : 0);
    kernel.constants={r.phi};
    if (initialize) {
        kernel.source="#include \"tulpa/likelihood_math.h\"\n__device__ void tulpa_likelihood_row(const double* r,int n,int i,const double* e,const double* c,double& v,double* g,double* h) { tulpa::math::fused_likelihood_row<"+
            std::to_string(family)+","+(zi ? "true" : "false")+">(r,n,i,e,c,v,g,h); }\n";
        kernel.response.resize(static_cast<std::size_t>(data.N)*4);
        for (int i=0; i<data.N; ++i) {
            const int trials=r.n_trials ? r.n_trials[i] : 1;
            kernel.response[i]=r.y[i]; kernel.response[data.N+i]=trials;
            kernel.response[2*data.N+i]=r.weights ? r.weights[i] : 1;
            kernel.response[3*data.N+i]=r.ll_const[i];
        }
    }
    return true;
}

bool select_likelihood(const ModelData& data, const ParamLayout& layout,
    DeviceLikelihood& kernel, Policy& policy, Status& status, bool weights) {
    const auto* spec=static_cast<const LikelihoodSpec*>(data.likelihood_spec);
    if (!spec || !spec->device_fn || !data.N || policy.choice()==Choice::Cpu) return false;
    if (!spec->device_fn(data,layout,data.model_response_data,false,kernel)) return false;
    return policy.select(weights ? "likelihood_weights" : "likelihood",data.N,kernel.variant,status);
}

} }
