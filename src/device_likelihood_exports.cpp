#include <RcppEigen.h>
#include "device_likelihood.h"
#include "sampler_model_data.h"
#include "log_post_impl.h"
#include "sysmem.h"
#include <chrono>

namespace {
using FusedClock=std::chrono::steady_clock;
double fused_seconds(FusedClock::time_point begin) {
    return std::chrono::duration<double>(FusedClock::now()-begin).count();
}
bool fused_bad_device(const tulpa::ModelData& data, const tulpa::ParamLayout& layout,
    const void* response, bool initialize, tulpa::DeviceLikelihood& kernel) {
    if (!tulpa::device::builtin_device_likelihood(data,layout,response,initialize,kernel)) return false;
    if (initialize) kernel.source.insert(kernel.source.rfind('}')," g[0] = -g[0]; ");
    return true;
}
template<class T> T fused_extra_ll(int i, const T* eta, const T& zi, const T& oi,
    const std::vector<T>& params, const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout, const void* response) {
    const T combined=eta[0]+params[layout.total_params-1];
    const auto& r=*static_cast<const tulpa::BuiltinFamilyResponse*>(response);
    return tulpa::math::log_lik_poisson<T>(static_cast<int>(r.y[i]),tulpa::math::safe_exp(combined));
}
bool fused_extra_device(const tulpa::ModelData& data, const tulpa::ParamLayout& layout,
    const void* response, bool initialize, tulpa::DeviceLikelihood& kernel) {
    if (!tulpa::device::builtin_device_likelihood(data,layout,response,initialize,kernel)) return false;
    kernel.variant=-1183; kernel.parameter_indices={layout.total_params-1};
    if (initialize) kernel.source=
        "#include \"tulpa/likelihood_math.h\"\n__device__ void tulpa_likelihood_row(const double* r,int n,int i,const double* e,const double* c,double& v,double* g,double* h) { double eta[1]={e[0]+e[1]}, score[1], curvature[1]; tulpa::math::fused_likelihood_row<2,false>(r,n,i,eta,c,v,score,curvature); g[0]=g[1]=score[0]; for(int k=0;k<4;++k) h[k]=curvature[0]; }\n";
    return true;
}
void fused_cpu_evaluate(const tulpa::device::ObservationPlan& plan,
    const tulpa::LikelihoodSpec& spec, tulpa::ModelData& data,
    const tulpa::ParamLayout& layout, const tulpa::BuiltinFamilyResponse& response,
    const std::vector<double>& coefficients, int chains, bool arena, bool weights,
    std::size_t& arena_nodes, std::size_t& arena_capacity,
    tulpa::device::LikelihoodResult& result) {
    const int n=plan.n, np=plan.np, width=plan.width;
    result.value.assign(chains,0); result.gradient.assign(coefficients.size(),0);
    result.eta.resize(weights ? static_cast<std::size_t>(chains)*n*np : 0);
    result.score.resize(result.eta.size()); result.curvature.resize(result.eta.size()*np);
    for (int chain=0; chain<chains; ++chain) {
        const auto* par=coefficients.data()+chain*width;
        std::vector<double> parameters(par,par+width);
        tulpa::arena::ArenaScope scope;
        auto active=arena ? tulpa::arena::make_vars(scope.arena(),parameters) : std::vector<tulpa::arena::Var>{};
        tulpa::arena::Var lp(0);
        for (int i=0; i<n; ++i) {
            double eta[8], score[8], curvature[64];
            tulpa::arena::Var ad_eta[8];
            std::size_t xoff=0;
            for (int k=0; k<np; ++k) {
                const int row=k*n+i;
                eta[k]=plan.offset[row];
                if (arena) ad_eta[k]=tulpa::arena::Var(eta[k]);
                for (int j=0; j<plan.beta_count[k]; ++j) {
                    const double x=plan.x[xoff+static_cast<std::size_t>(j)*n+i];
                    eta[k]+=x*par[plan.beta_start[k]+j];
                    if (arena) ad_eta[k]=ad_eta[k]+tulpa::arena::Var(x)*active[plan.beta_start[k]+j];
                }
                xoff+=static_cast<std::size_t>(n)*plan.beta_count[k];
                for (int j=plan.rowptr[row]; j<plan.rowptr[row+1]; ++j) {
                    eta[k]+=plan.weight[j]*par[plan.column[j]];
                    if (arena) ad_eta[k]=ad_eta[k]+tulpa::arena::Var(plan.weight[j])*active[plan.column[j]];
                }
                if (weights) result.eta[(static_cast<std::size_t>(chain)*n+i)*np+k]=eta[k];
            }
            if (arena) {
                const auto z=np==2 ? ad_eta[1] : tulpa::arena::Var(0);
                lp=lp+tulpa::builtin_family_ll_ad(i,ad_eta,z,tulpa::arena::Var(0),active,data,layout,&response);
                if (!weights) continue;
            }
            result.value[chain]+=spec.ll_eta_weights_fn(i,eta,0,0,parameters,data,layout,&response,score,curvature);
            xoff=0;
            for (int k=0; k<np; ++k) {
                if (weights) result.score[(static_cast<std::size_t>(chain)*np+k)*n+i]=score[k];
                for (int j=0; j<plan.beta_count[k]; ++j)
                    result.gradient[chain*width+plan.beta_start[k]+j]+=plan.x[xoff+static_cast<std::size_t>(j)*n+i]*score[k];
                xoff+=static_cast<std::size_t>(n)*plan.beta_count[k];
                const int row=k*n+i;
                for (int j=plan.rowptr[row]; j<plan.rowptr[row+1]; ++j)
                    result.gradient[chain*width+plan.column[j]]+=plan.weight[j]*score[k];
            }
            if (weights) std::copy(curvature,curvature+np*np,result.curvature.begin()+(static_cast<std::size_t>(chain)*n+i)*np*np);
        }
        if (arena) {
            arena_nodes=std::max(arena_nodes,static_cast<std::size_t>(scope.arena()->size()));
            arena_capacity=std::max(arena_capacity,static_cast<std::size_t>(scope.arena()->capacity()));
            lp.backward(); result.value[chain]=lp.val();
            const auto gradient=tulpa::arena::get_adjoints(active);
            std::copy(gradient.begin(),gradient.end(),result.gradient.begin()+chain*width);
        }
    }
}
}

// [[Rcpp::export]]
Rcpp::List cpp_device_likelihood_resources() {
    std::size_t free=0,total=0;
    auto& context=tulpa::device::Context::instance();
    auto status=context.initialize();
    if (status) {
        tulpa::device::Lease lease(context,status);
        if (status) status=context.memory_available(free,total);
    }
    return Rcpp::List::create(Rcpp::Named("host_available")=static_cast<double>(tulpa::available_ram_bytes()),
        Rcpp::Named("device_available")=static_cast<double>(free),
        Rcpp::Named("device_total")=static_cast<double>(total),Rcpp::Named("device_reason")=status.message);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_family(std::string family, Rcpp::NumericVector y,
    Rcpp::IntegerVector trials, Rcpp::NumericMatrix X, Rcpp::NumericMatrix coefficients,
    double phi, bool zi, Rcpp::NumericVector offset, Rcpp::NumericVector weights,
    Rcpp::IntegerVector groups, std::string choice, int repetitions=1, bool batched=true,
    bool return_weights=true) {
    const int n=y.size(), p=X.ncol(), np=zi ? 2 : 1, chains=coefficients.ncol();
    if (n<1 || X.nrow()!=n || trials.size()!=n || offset.size()!=n || weights.size()!=n || groups.size()!=n || repetitions<1)
        Rcpp::stop("invalid fused likelihood fixture dimensions");
    int ng=0;
    for (int g : groups) { if (g<0 || g==NA_INTEGER) Rcpp::stop("invalid field group"); ng=std::max(ng,g); }
    const int width=p*np+ng;
    if (coefficients.nrow()!=width || chains<1) Rcpp::stop("invalid fused coefficient dimensions");
    tulpa::ModelData data; data.N=n; data.n_processes=np;
    data.zi_type=zi ? tulpa::ZIType::ZI : tulpa::ZIType::NONE;
    auto spec=tulpa::builtin_family_spec(family,zi);
    data.likelihood_spec=&spec;
    tulpa::BuiltinFamilyResponse response;
    response.y=y.begin(); response.n_trials=trials.begin(); response.weights=weights.begin();
    response.N=n; response.family=family; response.phi=phi; response.prepare();
    data.model_response_data=&response;
    tulpa::ParamLayout layout;
    tulpa::device::ObservationPlan plan; plan.n=n; plan.np=np; plan.width=width;
    for (int k=0; k<np; ++k) {
        plan.add_design(X.begin(),p,k*p,true);
    }
    plan.rowptr.push_back(0);
    for (int k=0; k<np; ++k) for (int i=0; i<n; ++i) {
        plan.offset.push_back(k==0 ? offset[i] : 0);
        if (k==0 && groups[i]>0) { plan.column.push_back(p*np+groups[i]-1); plan.weight.push_back(1); }
        plan.rowptr.push_back(static_cast<int>(plan.column.size()));
    }
    std::vector<double> parameters(coefficients.begin(),coefficients.end());
    tulpa::device::LikelihoodResult result;
    const auto setup=FusedClock::now();
    std::unique_ptr<tulpa::device::LikelihoodWorkspace> workspace;
    bool cuda=false;
    if (choice!="arena") {
        tulpa::device::Policy policy(tulpa::device::parse_choice(choice),tulpa::device::thresholds());
        tulpa::DeviceLikelihood kernel; tulpa::device::Status status;
        cuda=tulpa::device::select_likelihood(data,layout,kernel,policy,status,return_weights);
        if (!status) Rcpp::stop(status.message);
        if (cuda) {
            spec.device_fn(data,layout,&response,true,kernel);
            workspace=std::make_unique<tulpa::device::LikelihoodWorkspace>(std::move(plan),std::move(kernel));
            status=workspace->evaluate(parameters,chains,batched,return_weights,result);
            if (!status) Rcpp::stop(status.message);
        }
    }
    const double setup_seconds=fused_seconds(setup);
    std::size_t arena_nodes=0, arena_capacity=0;
    const auto begin=FusedClock::now();
    for (int rep=0; rep<repetitions; ++rep) {
        if (cuda) {
            const auto status=workspace->evaluate(parameters,chains,batched,return_weights,result);
            if (!status) Rcpp::stop(status.message);
        } else fused_cpu_evaluate(plan,spec,data,layout,response,parameters,chains,choice=="arena",return_weights,
                                  arena_nodes,arena_capacity,result);
    }
    return Rcpp::List::create(Rcpp::Named("value")=result.value,Rcpp::Named("gradient")=result.gradient,
        Rcpp::Named("eta")=result.eta,Rcpp::Named("score")=result.score,Rcpp::Named("curvature")=result.curvature,
        Rcpp::Named("used_cuda")=cuda,Rcpp::Named("seconds")=fused_seconds(begin)/repetitions,
        Rcpp::Named("arena_nodes")=static_cast<double>(arena_nodes),
        Rcpp::Named("arena_capacity")=static_cast<double>(arena_capacity),
        Rcpp::Named("host_available_bytes")=static_cast<double>(tulpa::available_ram_bytes()),
        Rcpp::Named("setup_seconds")=setup_seconds);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_model_gradient(Rcpp::NumericVector y, Rcpp::IntegerVector trials,
    Rcpp::NumericMatrix X, std::string family, double phi, Rcpp::NumericVector params,
    std::string choice, Rcpp::Nullable<Rcpp::List> re=R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial=R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal=R_NilValue,
    Rcpp::Nullable<Rcpp::List> zi=R_NilValue, bool verify=true, int repetitions=1,
    bool device_slot=true, bool corrupt=false, bool consumer_extra=false) {
    if (repetitions<1) Rcpp::stop("repetitions must be positive");
    tulpa::SamplerModelInputs in;
    tulpa::build_sampler_model_inputs(in,y,trials,X,family,phi,NA_REAL,2.5,{},1,re,spatial,temporal,R_NilValue,R_NilValue,zi);
    if (!device_slot) in.spec.device_fn=nullptr;
    if (corrupt) in.spec.device_fn=&fused_bad_device;
    if (consumer_extra) {
        ++in.layout.total_params;
        in.spec.ll_double=&fused_extra_ll<double>;
        in.spec.ll_arena=&fused_extra_ll<tulpa::arena::Var>;
        in.spec.device_fn=&fused_extra_device;
    }
    auto policy=std::make_shared<tulpa::device::Policy>(tulpa::device::parse_choice(choice),tulpa::device::thresholds());
    auto previous=tulpa::device::exchange_policy(policy);
    struct Restore { std::shared_ptr<tulpa::device::Policy> previous;
        ~Restore() { tulpa::device::exchange_policy(previous); } } restore{previous};
    tulpa_hmc::GradientModeFitScope gradient_scope;
    std::vector<double> q(params.begin(),params.end());
    if (q.empty()) q.assign(in.layout.total_params,.1);
    if (q.size()!=static_cast<std::size_t>(in.layout.total_params)) Rcpp::stop("parameter count mismatch");
    if (verify) tulpa_hmc::ensure_gradient_verified(q,in.data,in.layout);
    auto fn=tulpa_hmc::resolve_gradient_fn(tulpa_hmc::get_gradient_mode(),in.data,in.layout);
    std::vector<double> gradient(q.size()); double lp=0;
    fn(q,in.data,in.layout,gradient,&lp);
    const auto begin=FusedClock::now();
    for (int i=0; i<repetitions; ++i) fn(q,in.data,in.layout,gradient,&lp);
    return Rcpp::List::create(Rcpp::Named("value")=lp,Rcpp::Named("gradient")=gradient,
        Rcpp::Named("params")=q,Rcpp::Named("seconds")=fused_seconds(begin)/repetitions,
        Rcpp::Named("verified")=tulpa_hmc::g_gradient_verified,
        Rcpp::Named("used_cuda")=fn==&tulpa_hmc::compute_gradient_generic_device);
}
