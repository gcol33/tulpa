#include "device_sparse.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tulpa { namespace device {
namespace sparse_detail {
using Handle = void*;
using Address = unsigned long long;
// cuDSS 1.x C ABI, from NVIDIA's cudss.h and cudss_data_types.h.
constexpr int real64 = 1, int32 = 10, spd = 3, upper = 2;
constexpr int analysis = 3, factorization = 4, solve_phase = 1008;
constexpr int data_info = 0, data_diag = 8, data_memory = 13;
constexpr int matching = 3, pivot_epsilon = 9;

struct Frame {
    Context& context;
    Status status;
    bool pushed;
    explicit Frame(Context& c) : context(c), status(c.push()), pushed(bool(status)) {}
    ~Frame() { if (pushed) status.absorb(context.pop()); }
};

struct Api {
    int (*create)(Handle*) = nullptr;
    int (*destroy)(Handle) = nullptr;
    int (*config_create)(Handle*) = nullptr;
    int (*config_destroy)(Handle) = nullptr;
    int (*config_set)(Handle, int, const void*, std::size_t) = nullptr;
    int (*data_create)(Handle, Handle*) = nullptr;
    int (*data_destroy)(Handle, Handle) = nullptr;
    int (*get)(Handle, Handle, int, void*, std::size_t, std::size_t*) = nullptr;
    int (*set)(Handle, Handle, int, const void*, std::size_t) = nullptr;
    int (*execute)(Handle, int, Handle, Handle, Handle, Handle, Handle) = nullptr;
    int (*set_stream)(Handle, Handle) = nullptr;
    int (*csr)(Handle*, std::int64_t, std::int64_t, std::int64_t,
               const void*, const void*, const void*, const void*, int, int, int,
               int, int, int) = nullptr;
    int (*dense)(Handle*, std::int64_t, std::int64_t, std::int64_t, const void*, int, int) = nullptr;
    int (*matrix_destroy)(Handle) = nullptr;
    Status load(Context& c) {
        Status s;
#define SPARSE_SYMBOL(member, name) member = reinterpret_cast<decltype(member)>(c.symbol(Capability::Sparse, name, s))
        SPARSE_SYMBOL(create, "cudssCreate");
        SPARSE_SYMBOL(destroy, "cudssDestroy");
        SPARSE_SYMBOL(config_create, "cudssConfigCreate");
        SPARSE_SYMBOL(config_destroy, "cudssConfigDestroy");
        SPARSE_SYMBOL(config_set, "cudssConfigSet");
        SPARSE_SYMBOL(data_create, "cudssDataCreate");
        SPARSE_SYMBOL(data_destroy, "cudssDataDestroy");
        SPARSE_SYMBOL(get, "cudssDataGet");
        SPARSE_SYMBOL(set, "cudssDataSet");
        SPARSE_SYMBOL(execute, "cudssExecute");
        SPARSE_SYMBOL(set_stream, "cudssSetStream");
        SPARSE_SYMBOL(csr, "cudssMatrixCreateCsr");
        SPARSE_SYMBOL(dense, "cudssMatrixCreateDn");
        SPARSE_SYMBOL(matrix_destroy, "cudssMatrixDestroy");
#undef SPARSE_SYMBOL
        return s;
    }
};
}

struct SparseFactor::Impl {
    Context& context = Context::instance();
    std::shared_ptr<Policy> policy = current_policy();
    sparse_detail::Api api;
    Status status;
    void *handle = nullptr, *config = nullptr, *data = nullptr, *matrix = nullptr;
    void *rhs = nullptr, *solution = nullptr, *stream = nullptr;
    bool own_stream = false, factored = false;
    int n = 0, nnz = 0, width = 0;
    std::size_t count = 0, bytes = 0;
    double logdet = std::numeric_limits<double>::quiet_NaN();
    std::unique_ptr<Buffer> p, i, values, b, x;
    std::vector<int> permutation;

    void read(int code, const char* operation) { status.absorb(Status::read("cuDSS", code, operation)); }
    void* pointer(const std::unique_ptr<Buffer>& v) { return reinterpret_cast<void*>(v->address()); }
    void destroy_matrix(void*& object) {
        if (object) { read(api.matrix_destroy(object), "cudssMatrixDestroy"); object = nullptr; }
    }
    void reserve(int columns) {
        if (!status || columns == width) return;
        status.absorb(context.synchronize_stream(stream));
        destroy_matrix(rhs); destroy_matrix(solution);
        b.reset(); x.reset(); width = 0;
        const std::size_t size = static_cast<std::size_t>(n) * columns * sizeof(double);
        b.reset(new Buffer(context, size, status));
        x.reset(new Buffer(context, size, status));
        if (status) read(api.dense(&rhs, n, columns, n, pointer(b), sparse_detail::real64, 0), "cudssMatrixCreateDn(rhs)");
        if (status) read(api.dense(&solution, n, columns, n, pointer(x), sparse_detail::real64, 0), "cudssMatrixCreateDn(solution)");
        if (status) width = columns;
    }
    void execute(int phase) {
        if (status) read(api.execute(handle, phase, config, data, matrix, solution, rhs), "cudssExecute");
        if (status) status.absorb(context.synchronize_stream(stream));
        if (status && phase == sparse_detail::solve_phase) {
            int info = 0; std::size_t written = 0;
            read(api.get(handle, data, sparse_detail::data_info, &info, sizeof(info), &written), "cudssDataGet(solve info)");
            status.absorb(Status::read("cuDSS", info, "solve device info"));
        }
    }
    void launch(const char* entry, const Launch& shape, void** args) {
        if (!status) return;
        const auto source = kernel_source("sparse_inverse");
        if (source.empty()) { status = Status::read("CUDA", -1, entry, "sparse_inverse kernel source is missing"); return; }
        Lease lease(context, status);
        status.absorb(context.launch(source, entry, shape, args));
    }
    template<class Gather>
    void inverse_blocks(int count, Buffer& order, Gather gather) {
        auto oa = order.address();
        for (int first = 0; status && first < count; first += 64) {
            int columns = std::min(64, count - first);
            reserve(columns);
            auto ba = b->address(), xa = x->address();
            Launch shape; shape.stream = stream;
            shape.grid_x = static_cast<unsigned>((static_cast<std::size_t>(n)*columns + 255)/256);
            void* args[] = {&ba, &n, &columns, &first, &oa};
            launch("sparse_identity", shape, args);
            execute(sparse_detail::solve_phase);
            gather(xa, first, columns, shape);
        }
        if (status) status.absorb(context.synchronize_stream(stream));
    }
    ~Impl() {
        if (!handle && !own_stream) return;
        sparse_detail::Frame frame(context);
        if (!frame.status) return;
        status.absorb(context.synchronize_stream(stream));
        destroy_matrix(rhs); destroy_matrix(solution); destroy_matrix(matrix);
        if (data) read(api.data_destroy(handle, data), "cudssDataDestroy");
        if (config) read(api.config_destroy(config), "cudssConfigDestroy");
        if (handle) read(api.destroy(handle), "cudssDestroy");
        p.reset(); i.reset(); values.reset(); b.reset(); x.reset();
        if (own_stream) status.absorb(context.destroy_stream(stream));
        if (!status) policy->fail(status);
    }
};

SparseFactor::SparseFactor() : impl_(new Impl) {}
SparseFactor::~SparseFactor() = default;
Status SparseFactor::analyze(int n, const int* p, const int* i, const double* x, void* stream) {
    if (n <= 0 || !p || !i || !x) return Status::read("cuDSS", -1, "analyze", "invalid sparse dimensions or null arrays");
    impl_.reset(new Impl);
    auto& s = *impl_;
    s.status = s.api.load(s.context);
    if (!s.status) return s.status;
    sparse_detail::Frame frame(s.context);
    s.status.absorb(frame.status);
    s.n = n; s.nnz = p[n]; s.stream = stream;
    if (s.status) s.read(s.api.create(&s.handle), "cudssCreate");
    if (s.status && !stream) { s.status.absorb(s.context.create_stream(s.stream)); s.own_stream = s.stream != nullptr; }
    if (s.status) s.read(s.api.set_stream(s.handle, s.stream), "cudssSetStream");
    if (s.status) s.read(s.api.config_create(&s.config), "cudssConfigCreate");
    if (s.status) s.read(s.api.data_create(s.handle, &s.data), "cudssDataCreate");
    int no_matching = 0;
    double no_perturbation = 0;
    if (s.status) s.read(s.api.config_set(s.config, sparse_detail::matching, &no_matching, sizeof(int)), "cudssConfigSet(matching)");
    if (s.status) s.read(s.api.config_set(s.config, sparse_detail::pivot_epsilon, &no_perturbation, sizeof(double)), "cudssConfigSet(pivot epsilon)");
    s.p.reset(new Buffer(s.context, (n + 1ULL) * sizeof(int), s.status));
    s.i.reset(new Buffer(s.context, s.nnz * sizeof(int), s.status));
    s.values.reset(new Buffer(s.context, s.nnz * sizeof(double), s.status));
    if (s.status) s.status.absorb(s.p->upload(p, s.p->size()));
    if (s.status) s.status.absorb(s.i->upload(i, s.i->size()));
    if (s.status) s.status.absorb(s.values->upload(x, s.values->size()));
    // Symmetric lower CSC is upper CSR, including the same numeric ordering.
    if (s.status) s.read(s.api.csr(&s.matrix, n, n, s.nnz, s.pointer(s.p), nullptr,
        s.pointer(s.i), s.pointer(s.values), sparse_detail::int32, sparse_detail::int32,
        sparse_detail::real64, sparse_detail::spd, sparse_detail::upper, 0), "cudssMatrixCreateCsr");
    s.reserve(1);
    s.execute(sparse_detail::analysis);
    if (s.status) {
        std::size_t written = 0;
        s.read(s.api.get(s.handle, s.data, sparse_detail::data_memory, nullptr, 0, &written), "cudssDataGet(memory size)");
        std::vector<std::int64_t> memory((written + sizeof(std::int64_t) - 1)/sizeof(std::int64_t));
        if (s.status) s.read(s.api.get(s.handle, s.data, sparse_detail::data_memory, memory.data(), written, &written), "cudssDataGet(memory)");
        if (s.status && memory.size() >= 4) s.bytes = static_cast<std::size_t>(memory[0] + memory[2]);
        s.permutation.resize(n);
        if (s.status) s.read(s.api.get(s.handle, s.data, 4, s.permutation.data(), n * sizeof(int), &written), "cudssDataGet(permutation)");
        ++s.count;
    }
    return s.status;
}
Status SparseFactor::refactor(const double* x, bool& positive_definite) {
    auto& s = *impl_;
    positive_definite = false; s.factored = false;
    sparse_detail::Frame frame(s.context);
    s.status.absorb(frame.status);
    const int clear_info = 0;
    if (s.status) s.read(s.api.set(s.handle, s.data, sparse_detail::data_info, &clear_info, sizeof(int)), "cudssDataSet(clear info)");
    if (s.status) s.status.absorb(s.values->upload(x, s.values->size()));
    s.execute(sparse_detail::factorization);
    int info = 0; std::size_t written = 0;
    if (s.status) s.read(s.api.get(s.handle, s.data, sparse_detail::data_info, &info, sizeof(info), &written), "cudssDataGet(info)");
    if (!s.status || info) return s.status;
    std::vector<double> diag(s.n);
    s.read(s.api.get(s.handle, s.data, sparse_detail::data_diag, diag.data(), diag.size() * sizeof(double), &written), "cudssDataGet(diagonal)");
    if (s.status) s.status.absorb(s.context.synchronize_stream(s.stream));
    s.logdet = 0;
    for (double v : diag) {
        if (!(v > 0) || !std::isfinite(v)) return s.status;
        s.logdet += 2 * std::log(v);
    }
    positive_definite = s.factored = bool(s.status);
    return s.status;
}
Status SparseFactor::solve(const double* b, double* x, int columns) {
    auto& s = *impl_;
    if (!s.factored || columns <= 0) return Status::read("cuDSS", -1, "solve", "no usable factor or invalid RHS count");
    sparse_detail::Frame frame(s.context);
    s.status.absorb(frame.status);
    s.reserve(columns);
    if (s.status) s.status.absorb(s.b->upload(b, s.b->size()));
    s.execute(sparse_detail::solve_phase);
    if (s.status) s.status.absorb(s.x->download(x, s.x->size()));
    return s.status;
}

Status SparseFactor::inverse_entries(const std::vector<int>& p, const std::vector<int>& i,
                                     const std::vector<int>& perm, std::vector<double>& z) {
    auto& s = *impl_;
    if (!s.factored) return Status::read("cuDSS", -1, "selected inverse", "no usable factor");
    if (p.size() != static_cast<std::size_t>(s.n + 1) || perm.size() != static_cast<std::size_t>(s.n) ||
        p[0] != 0 || p.back() < 0 || static_cast<std::size_t>(p.back()) != i.size())
        return Status::read("cuDSS", -1, "selected inverse", "invalid selected pattern dimensions");
    std::vector<bool> seen(s.n, false);
    for (int j = 0; j < s.n; ++j) {
        if (perm[j] < 0 || perm[j] >= s.n || seen[perm[j]] || p[j] < 0 ||
            p[j + 1] < p[j] || p[j + 1] > p.back())
            return Status::read("cuDSS", -1, "selected inverse", "invalid permutation or selected pattern");
        seen[perm[j]] = true;
        for (int k = p[j]; k < p[j + 1]; ++k)
            if (i[k] < j || i[k] >= s.n || (k > p[j] && i[k] <= i[k - 1]))
                return Status::read("cuDSS", -1, "selected inverse", "invalid selected row indices");
    }
    sparse_detail::Frame frame(s.context);
    s.status.absorb(frame.status);
    Buffer dp(s.context, p.size()*sizeof(int), s.status), di(s.context, i.size()*sizeof(int), s.status);
    Buffer order(s.context, perm.size()*sizeof(int), s.status), out(s.context, i.size()*sizeof(double), s.status);
    if (s.status) s.status.absorb(dp.upload(p.data(), dp.size()));
    if (s.status) s.status.absorb(di.upload(i.data(), di.size()));
    if (s.status) s.status.absorb(order.upload(perm.data(), order.size()));
    auto pa = dp.address(), ia = di.address(), oa = order.address(), za = out.address();
    s.inverse_blocks(s.n, order, [&](sparse_detail::Address xa, int first, int width, Launch launch) {
        launch.grid_x = static_cast<unsigned>(width);
        void* gather[] = {&xa, &s.n, &first, &width, &pa, &ia, &oa, &za};
        s.launch("sparse_gather", launch, gather);
    });
    z.resize(i.size());
    if (s.status) s.status.absorb(out.download(z.data(), out.size()));
    return s.status;
}
Status SparseFactor::inverse_diagonal(const std::vector<int>& indices, std::vector<double>& z) {
    auto& s = *impl_;
    if (!s.factored) return Status::read("cuDSS", -1, "inverse diagonal", "no usable factor");
    for (int index : indices) if (index < 0 || index >= s.n) return Status::read("cuDSS", -1, "inverse diagonal", "index out of range");
    sparse_detail::Frame frame(s.context);
    s.status.absorb(frame.status);
    Buffer order(s.context, indices.size()*sizeof(int), s.status), out(s.context, indices.size()*sizeof(double), s.status);
    if (s.status) s.status.absorb(order.upload(indices.data(), order.size()));
    auto oa = order.address(), za = out.address();
    s.inverse_blocks(static_cast<int>(indices.size()), order, [&](sparse_detail::Address xa, int first, int width, Launch launch) {
        launch.grid_x = (width + 255)/256;
        void* gather[] = {&xa, &s.n, &first, &width, &oa, &za};
        s.launch("sparse_gather_diagonal", launch, gather);
    });
    z.resize(indices.size());
    if (s.status) s.status.absorb(out.download(z.data(), out.size()));
    return s.status;
}
double SparseFactor::log_determinant() const { return impl_->logdet; }
std::size_t SparseFactor::factor_bytes() const { return impl_->bytes; }
std::size_t SparseFactor::analyses() const { return impl_->count; }
const std::vector<int>& SparseFactor::permutation() const { return impl_->permutation; }
void* SparseFactor::stream() const { return impl_->stream; }
} }
