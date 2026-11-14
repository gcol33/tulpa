#include "device_likelihood.h"
#include "laplace_builtin_family_spec.h"
#include "tulpa/likelihood_math.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tulpa { namespace device {

struct LikelihoodWorkspace::Impl {
    ObservationPlan plan;
    DeviceLikelihood kernel;
    Context& context = Context::instance();
    Status status;
    enum Slot { X, Offset, Rowptr, Column, Weight, Colptr, Row, Transpose,
        Response, Constants, BetaStart, BetaCount, Parameters, Scores, Curvature,
        Partial, Gradient, Eta, Count };
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::vector<void*> streams;
    int capacity = 0, blocks = 0;
    bool have_weights = false;

    Impl(ObservationPlan p, DeviceLikelihood k) : plan(std::move(p)), kernel(std::move(k)), buffers(Count) {
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
        if (kernel.source.empty()) throw std::invalid_argument("empty likelihood device source");
    }
    ~Impl() {
        if (buffers[X]) {
            Status cleanup;
            Lease lease(context,cleanup);
            cleanup.absorb(context.synchronize());
            for (void* stream : streams) cleanup.absorb(context.destroy_stream(stream));
            buffers.clear();
        }
    }
    unsigned long long address(Slot slot) const { return buffers[slot] ? buffers[slot]->address() : 0; }
    void allocate(Slot slot, std::size_t bytes) {
        buffers[slot].reset();
        buffers[slot] = std::make_unique<Buffer>(context,bytes,status);
    }
    template<class T> void upload(Slot slot, const std::vector<T>& data) {
        allocate(slot,data.size()*sizeof(T));
        if (status) status.absorb(buffers[slot]->upload(data.data(),data.size()*sizeof(T)));
    }
    void upload_projection() {
        upload(Rowptr,plan.rowptr); upload(Column,plan.column); upload(Weight,plan.weight);
        std::vector<int> colptr(plan.width+1,0), rows(plan.column.size());
        std::vector<double> transpose(plan.weight.size());
        for (int col : plan.column) ++colptr[col+1];
        std::partial_sum(colptr.begin(),colptr.end(),colptr.begin());
        auto next=colptr;
        for (int row=0; row<plan.n*plan.np; ++row)
            for (int j=plan.rowptr[row]; j<plan.rowptr[row+1]; ++j) {
                const int dest=next[plan.column[j]]++;
                rows[dest]=row; transpose[dest]=plan.weight[j];
            }
        upload(Colptr,colptr); upload(Row,rows); upload(Transpose,transpose);
    }
    void initialize(int chains, bool weights) {
        if (!buffers[X]) {
            upload(X,plan.x); upload(Offset,plan.offset); upload(Response,kernel.response);
            upload(BetaStart,plan.beta_start); upload(BetaCount,plan.beta_count);
            upload_projection();
            blocks = std::min(4096,(plan.n+255)/256);
        }
        if (chains>capacity || (weights && !have_weights)) {
            capacity = std::max(chains,capacity); have_weights = weights || have_weights;
            allocate(Parameters,static_cast<std::size_t>(capacity)*plan.width*sizeof(double));
            allocate(Scores,static_cast<std::size_t>(capacity)*plan.n*plan.np*sizeof(double));
            allocate(Partial,static_cast<std::size_t>(capacity)*blocks*sizeof(double));
            allocate(Gradient,static_cast<std::size_t>(capacity)*plan.width*sizeof(double));
            if (have_weights) {
                allocate(Curvature,static_cast<std::size_t>(capacity)*plan.n*plan.np*plan.np*sizeof(double));
                allocate(Eta,static_cast<std::size_t>(capacity)*plan.n*plan.np*sizeof(double));
            }
        }
        if (!buffers[Constants] || buffers[Constants]->size()!=kernel.constants.size()*sizeof(double))
            allocate(Constants,kernel.constants.size()*sizeof(double));
        while (status && static_cast<int>(streams.size())<chains) {
            void* stream = nullptr;
            status.absorb(context.create_stream(stream));
            if (stream) streams.push_back(stream);
        }
    }
};

LikelihoodWorkspace::LikelihoodWorkspace(ObservationPlan plan, DeviceLikelihood kernel) :
    impl_(new Impl(std::move(plan),std::move(kernel))) {}
LikelihoodWorkspace::~LikelihoodWorkspace() = default;
const ObservationPlan& LikelihoodWorkspace::plan() const { return impl_->plan; }
DeviceLikelihood& LikelihoodWorkspace::kernel() { return impl_->kernel; }
Status LikelihoodWorkspace::refresh_projection(ObservationPlan plan) {
    auto& w=*impl_;
    if (plan.n!=w.plan.n || plan.np!=w.plan.np || plan.width!=w.plan.width ||
        plan.beta_start!=w.plan.beta_start || plan.beta_count!=w.plan.beta_count)
        return Status::read("likelihood",-1,"refresh_projection","observation layout changed within a fit");
    Status s;
    if (w.buffers[Impl::X]) {
        Lease lease(w.context,s);
        w.status={};
        if (plan.x!=w.plan.x) w.upload(Impl::X,plan.x);
        if (plan.offset!=w.plan.offset) w.upload(Impl::Offset,plan.offset);
        const bool changed=plan.rowptr!=w.plan.rowptr || plan.column!=w.plan.column || plan.weight!=w.plan.weight;
        w.plan=std::move(plan);
        if (changed) w.upload_projection();
        s.absorb(w.status);
    } else w.plan=std::move(plan);
    return s;
}

Status LikelihoodWorkspace::evaluate(const std::vector<double>& parameters, int chains,
    bool batched, bool weights, LikelihoodResult& result) {
    auto& w = *impl_;
    if (chains<1 || chains>65535 || parameters.size()!=static_cast<std::size_t>(chains)*w.plan.width)
        return Status::read("likelihood",-1,"evaluate","coefficient dimensions mismatch");
    Status s = w.context.supports_operation("likelihood");
    if (!s) return s;
    {
        Lease lease(w.context,s);
        w.status = {};
        w.initialize(chains,weights); s.absorb(w.status);
        if (s) s = w.buffers[Impl::Parameters]->upload(parameters.data(),parameters.size()*sizeof(double));
        if (s) s = w.buffers[Impl::Constants]->upload(w.kernel.constants.data(),w.kernel.constants.size()*sizeof(double));
        auto launch = [&](int first, int count, void* stream) {
            if (!s) return;
            int n=w.plan.n, np=w.plan.np, width=w.plan.width;
            auto x=w.address(Impl::X), offset=w.address(Impl::Offset), rowptr=w.address(Impl::Rowptr);
            auto column=w.address(Impl::Column), projection=w.address(Impl::Weight), response=w.address(Impl::Response);
            auto constants=w.address(Impl::Constants), start=w.address(Impl::BetaStart), columns=w.address(Impl::BetaCount);
            auto par=w.address(Impl::Parameters)+static_cast<std::size_t>(first)*width*sizeof(double);
            auto score=w.address(Impl::Scores)+static_cast<std::size_t>(first)*n*np*sizeof(double);
            auto curvature=weights ? w.address(Impl::Curvature)+static_cast<std::size_t>(first)*n*np*np*sizeof(double) : 0;
            auto partial=w.address(Impl::Partial)+static_cast<std::size_t>(first)*w.blocks*sizeof(double);
            auto eta=weights ? w.address(Impl::Eta)+static_cast<std::size_t>(first)*n*np*sizeof(double) : 0;
            void* args[]={&n,&np,&width,&start,&columns,&x,&offset,&rowptr,&column,&projection,&response,&constants,&par,&score,&curvature,&partial,&eta};
            Launch shape; shape.grid_x=w.blocks; shape.grid_y=count; shape.stream=stream;
            s = w.context.launch(w.kernel.source+kernel_source("likelihood"),"tulpa_likelihood",shape,args);
            auto cp=w.address(Impl::Colptr), row=w.address(Impl::Row), wt=w.address(Impl::Transpose);
            auto gradient=w.address(Impl::Gradient)+static_cast<std::size_t>(first)*width*sizeof(double);
            void* scatter[]={&n,&np,&width,&cp,&row,&wt,&score,&gradient};
            shape.grid_x=std::min(width,4096);
            if (s) s = w.context.launch(w.kernel.source+kernel_source("likelihood"),"tulpa_likelihood_scatter",shape,scatter);
            for (int chain=first; s && chain<first+count; ++chain) {
                std::size_t xoff=0;
                for (int k=0; s && k<np; ++k) {
                    const int p=w.plan.beta_count[k];
                    if (p) s = w.context.gemv(w.address(Impl::X)+xoff*sizeof(double),
                        w.address(Impl::Scores)+(static_cast<std::size_t>(chain)*np+k)*n*sizeof(double),
                        w.address(Impl::Gradient)+(static_cast<std::size_t>(chain)*width+w.plan.beta_start[k])*sizeof(double),
                        n,p,true,1,stream);
                    xoff += static_cast<std::size_t>(n)*p;
                }
            }
        };
        if (batched) launch(0,chains,w.streams[0]);
        else for (int chain=0; s && chain<chains; ++chain) launch(chain,1,w.streams[chain]);
        for (int chain=0; chain<(batched ? 1 : chains) && chain<static_cast<int>(w.streams.size()); ++chain)
            s.absorb(w.context.synchronize_stream(w.streams[chain]));
        std::vector<double> partial(static_cast<std::size_t>(chains)*w.blocks);
        result.gradient.resize(parameters.size()); result.value.assign(chains,0);
        if (s) s = w.buffers[Impl::Partial]->download(partial.data(),partial.size()*sizeof(double));
        if (s) s = w.buffers[Impl::Gradient]->download(result.gradient.data(),result.gradient.size()*sizeof(double));
        if (s) for (int chain=0; chain<chains; ++chain)
            result.value[chain]=std::accumulate(partial.begin()+chain*w.blocks,partial.begin()+(chain+1)*w.blocks,0.0);
        result.score.clear(); result.curvature.clear(); result.eta.clear();
        if (s && weights) {
            result.score.resize(static_cast<std::size_t>(chains)*w.plan.n*w.plan.np);
            result.curvature.resize(result.score.size()*w.plan.np);
            result.eta.resize(result.score.size());
            s = w.buffers[Impl::Scores]->download(result.score.data(),result.score.size()*sizeof(double));
            if (s) s = w.buffers[Impl::Curvature]->download(result.curvature.data(),result.curvature.size()*sizeof(double));
            if (s) s = w.buffers[Impl::Eta]->download(result.eta.data(),result.eta.size()*sizeof(double));
        }
    }
    s.absorb(w.status);
    return s;
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
