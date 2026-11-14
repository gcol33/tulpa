#include "device.h"
#include "linalg_fast.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#define TULPA_DEVICE_CALL __stdcall
#else
#include <dlfcn.h>
#define TULPA_DEVICE_CALL
#endif

namespace tulpa { namespace device {

Status Status::read(const char* library, int code, const char* operation,
                    const std::string& detail) {
    if (!code) return {};
    return {code, std::string(library) + "::" + operation + " failed (status " +
        std::to_string(code) + ")" + (detail.empty() ? "" : ": " + detail)};
}

namespace detail {
std::mutex settings_mutex;
std::vector<Threshold> settings;
std::map<std::string, std::string> kernel_sources;
std::size_t source_generation=0;
std::shared_ptr<Policy> active_policy;

class Library {
    void* handle_ = nullptr;
public:
    std::string name, reason;
    ~Library() {
        if (!handle_) return;
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
    }
    bool open(const std::vector<std::string>& names) {
#ifndef TULPA_DISABLE_CUDA
        for (const auto& candidate : names) {
#ifdef _WIN32
            handle_ = LoadLibraryExA(candidate.c_str(), nullptr,
                candidate.find('/') == std::string::npos ? 0 : LOAD_WITH_ALTERED_SEARCH_PATH);
#else
            handle_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            if (handle_) { name = candidate; return true; }
        }
#endif
        reason = "cannot load " + names.front();
        return false;
    }
    void* find(const char* symbol, Status& status) {
        if (!status) return nullptr;
#ifdef _WIN32
        void* result = handle_ ? reinterpret_cast<void*>(
            GetProcAddress(static_cast<HMODULE>(handle_), symbol)) : nullptr;
#else
        void* result = handle_ ? dlsym(handle_, symbol) : nullptr;
#endif
        if (!result) status = Status::read("loader", -1, symbol,
            handle_ ? "entry point absent from " + name : reason);
        return result;
    }
    bool loaded() const { return handle_ != nullptr; }
};

std::vector<std::string> library_names(const char* windows, const char* posix) {
#ifdef _WIN32
    std::vector<std::string> result{windows};
    for (const char* variable : {"CUDA_PATH", "CUDSS_DIR"}) {
        const char* root = std::getenv(variable);
        if (!root || !*root) continue;
        for (const char* suffix : {"/bin/x64/", "/bin/"})
            result.push_back(std::string(root) + suffix + windows);
    }
    return result;
#else
    return {posix};
#endif
}

using Address = unsigned long long;
using Handle = void*;
#define DEVICE_FN(name, args) using name##_type = int (TULPA_DEVICE_CALL*) args; name##_type name = nullptr
}
using namespace detail;

struct Context::Impl {
    std::once_flag once;
    std::mutex execution;
    Status status;
    Status capability[5];
    Library libraries[5];
    Handle context = nullptr, blas = nullptr, solver = nullptr;
    int device = 0;
    struct CompiledKernel { Handle module = nullptr, function = nullptr; };
    std::map<std::tuple<std::size_t,std::string,std::string>, CompiledKernel> kernels;
    Info info;
    DEVICE_FN(init, (unsigned));
    DEVICE_FN(driver_version, (int*));
    DEVICE_FN(device_count, (int*));
    DEVICE_FN(device_get, (int*, int));
    DEVICE_FN(device_name, (char*, int, int));
    DEVICE_FN(device_memory, (std::size_t*, int));
    DEVICE_FN(device_attribute, (int*, int, int));
    DEVICE_FN(retain, (Handle*, int));
    DEVICE_FN(release, (int));
    DEVICE_FN(push, (Handle));
    DEVICE_FN(pop, (Handle*));
    DEVICE_FN(allocate, (Address*, std::size_t));
    DEVICE_FN(free, (Address));
    DEVICE_FN(upload, (Address, const void*, std::size_t));
    DEVICE_FN(download, (void*, Address, std::size_t));
    DEVICE_FN(sync, ());
    DEVICE_FN(memory_info, (std::size_t*, std::size_t*));
    DEVICE_FN(stream_create, (Handle*, unsigned));
    DEVICE_FN(stream_destroy, (Handle));
    DEVICE_FN(stream_sync, (Handle));
    DEVICE_FN(module_load, (Handle*, const void*));
    DEVICE_FN(module_unload, (Handle));
    DEVICE_FN(function, (Handle*, Handle, const char*));
    DEVICE_FN(launch, (Handle, unsigned, unsigned, unsigned, unsigned, unsigned,
                       unsigned, unsigned, Handle, void**, void**));
    DEVICE_FN(blas_create, (Handle*));
    DEVICE_FN(blas_destroy, (Handle));
    DEVICE_FN(blas_stream, (Handle, Handle));
    DEVICE_FN(gemv, (Handle, int, int, int, const double*, const double*, int,
                     const double*, int, const double*, double*, int));
    DEVICE_FN(solver_create, (Handle*));
    DEVICE_FN(solver_destroy, (Handle));
    DEVICE_FN(potrf, (Handle, int, int, double**, int, int*, int));
    DEVICE_FN(program_create, (Handle*, const char*, const char*, int,
                               const char* const*, const char* const*));
    DEVICE_FN(program_compile, (Handle, int, const char* const*));
    DEVICE_FN(program_destroy, (Handle*));
    DEVICE_FN(ptx_size, (Handle, std::size_t*));
    DEVICE_FN(ptx, (Handle, char*));
    DEVICE_FN(log_size, (Handle, std::size_t*));
    DEVICE_FN(log, (Handle, char*));
    DEVICE_FN(sparse_property, (int, int*));

    template<class T> void resolve(T& target, int lib, const char* symbol, Status& s) {
        target = reinterpret_cast<T>(libraries[lib].find(symbol, s));
    }
    void initialize() {
        ++info.init_count;
#ifdef TULPA_DISABLE_CUDA
        info.compiled = false;
        status = Status::read("CUDA", -1, "initialize", "TULPA_DISABLE_CUDA is defined");
#else
        libraries[0].open(library_names("nvcuda.dll", "libcuda.so.1"));
#define RESOLVE(member, symbol) resolve(member, 0, symbol, status)
        RESOLVE(init, "cuInit");
        RESOLVE(driver_version, "cuDriverGetVersion");
        RESOLVE(device_count, "cuDeviceGetCount");
        RESOLVE(device_get, "cuDeviceGet");
        RESOLVE(device_name, "cuDeviceGetName");
        RESOLVE(device_memory, "cuDeviceTotalMem_v2");
        RESOLVE(device_attribute, "cuDeviceGetAttribute");
        RESOLVE(retain, "cuDevicePrimaryCtxRetain");
        RESOLVE(release, "cuDevicePrimaryCtxRelease_v2");
        RESOLVE(push, "cuCtxPushCurrent_v2");
        RESOLVE(pop, "cuCtxPopCurrent_v2");
        RESOLVE(allocate, "cuMemAlloc_v2");
        RESOLVE(free, "cuMemFree_v2");
        RESOLVE(upload, "cuMemcpyHtoD_v2");
        RESOLVE(download, "cuMemcpyDtoH_v2");
        RESOLVE(sync, "cuCtxSynchronize");
        RESOLVE(memory_info, "cuMemGetInfo_v2");
        RESOLVE(stream_create, "cuStreamCreate");
        RESOLVE(stream_destroy, "cuStreamDestroy_v2");
        RESOLVE(stream_sync, "cuStreamSynchronize");
        RESOLVE(module_load, "cuModuleLoadData");
        RESOLVE(module_unload, "cuModuleUnload");
        RESOLVE(function, "cuModuleGetFunction");
        RESOLVE(launch, "cuLaunchKernel");
#undef RESOLVE
        if (!status) return;
#define DRIVER(call) do { status = Status::read("CUDA", call, #call); if (!status) return; } while (false)
        DRIVER(init(0));
        DRIVER(driver_version(&info.driver_version));
        DRIVER(device_count(&info.count));
        if (!info.count) { status = Status::read("CUDA", -1, "initialize", "no CUDA devices"); return; }
        int major = 0, minor = 0;
        char name[256] = {};
        std::size_t memory = 0;
        DRIVER(device_get(&device, 0));
        DRIVER(device_name(name, sizeof(name), device));
        DRIVER(device_memory(&memory, device));
        DRIVER(device_attribute(&major, 75, device));
        DRIVER(device_attribute(&minor, 76, device));
        DRIVER(retain(&context, device));
        info.name = name;
        info.memory_mb = memory / (1024.0 * 1024.0);
        info.compute_capability = std::to_string(major) + "." + std::to_string(minor);
        DRIVER(push(context));
#undef DRIVER
        libraries[1].open(library_names("cusolver64_12.dll", "libcusolver.so.12"));
        libraries[2].open(library_names("cublas64_13.dll", "libcublas.so.13"));
        libraries[3].open(library_names("nvrtc64_130_0.dll", "libnvrtc.so.13"));
        auto sparse_names = library_names("cudss64_1.dll", "libcudss.so.1");
        auto sparse_legacy_names = library_names("cudss64_0.dll", "libcudss.so.0");
        sparse_names.insert(sparse_names.end(), sparse_legacy_names.begin(), sparse_legacy_names.end());
        libraries[4].open(sparse_names);
        resolve(solver_create, 1, "cusolverDnCreate", capability[1]);
        resolve(solver_destroy, 1, "cusolverDnDestroy", capability[1]);
        resolve(potrf, 1, "cusolverDnDpotrfBatched", capability[1]);
        if (capability[1]) capability[1] = Status::read("cuSOLVER", solver_create(&solver), "cusolverDnCreate");
        resolve(blas_create, 2, "cublasCreate_v2", capability[2]);
        resolve(blas_destroy, 2, "cublasDestroy_v2", capability[2]);
        resolve(blas_stream, 2, "cublasSetStream_v2", capability[2]);
        resolve(gemv, 2, "cublasDgemv_v2", capability[2]);
        if (capability[2]) capability[2] = Status::read("cuBLAS", blas_create(&blas), "cublasCreate_v2");
        resolve(program_create, 3, "nvrtcCreateProgram", capability[3]);
        resolve(program_compile, 3, "nvrtcCompileProgram", capability[3]);
        resolve(program_destroy, 3, "nvrtcDestroyProgram", capability[3]);
        resolve(ptx_size, 3, "nvrtcGetPTXSize", capability[3]);
        resolve(ptx, 3, "nvrtcGetPTX", capability[3]);
        resolve(log_size, 3, "nvrtcGetProgramLogSize", capability[3]);
        resolve(log, 3, "nvrtcGetProgramLog", capability[3]);
        resolve(sparse_property, 4, "cudssGetProperty", capability[4]);
        int sparse_major = 0;
        if (capability[4]) capability[4] = Status::read("cuDSS", sparse_property(0, &sparse_major), "cudssGetProperty");
        if (capability[4] && sparse_major != 1)
            capability[4] = Status::read("cuDSS", -1, "ABI", "cuDSS 1.x is required");
        Handle previous = nullptr;
        status = Status::read("CUDA", pop(&previous), "cuCtxPopCurrent_v2");
#endif
    }
    ~Impl() {
        if (!context) return;
        Status cleanup = Status::read("CUDA", push(context), "cuCtxPushCurrent_v2");
        if (cleanup) {
            for (const auto& item : kernels)
                cleanup.absorb(Status::read("CUDA", module_unload(item.second.module), "cuModuleUnload"));
            if (solver) cleanup.absorb(Status::read("cuSOLVER", solver_destroy(solver), "cusolverDnDestroy"));
            if (blas) cleanup.absorb(Status::read("cuBLAS", blas_destroy(blas), "cublasDestroy_v2"));
            Handle previous = nullptr;
            cleanup.absorb(Status::read("CUDA", pop(&previous), "cuCtxPopCurrent_v2"));
        }
        cleanup.absorb(Status::read("CUDA", release(device), "cuDevicePrimaryCtxRelease_v2"));
    }
};
#undef DEVICE_FN

Context::Context() : impl_(new Impl) {}
Context::~Context() = default;
Context& Context::instance() { static Context context; return context; }
Status Context::initialize() {
    std::call_once(impl_->once, [this] { impl_->initialize(); });
    return impl_->status;
}
Status Context::supports(Capability capability) {
    Status s = initialize();
    if (!s) return s;
    return impl_->capability[static_cast<int>(capability)];
}
Info Context::info() {
    Status s = initialize();
    Info result = impl_->info;
    result.available = static_cast<bool>(s);
    result.reason = s.message;
    if (!s) result.count = 0;
    for (int i = 0; i < 5; ++i) {
        result.libraries.push_back(impl_->libraries[i].name);
        result.library_reasons.push_back(s ? impl_->capability[i].message : s.message);
    }
    return result;
}
std::mutex& Context::execution_mutex() { return impl_->execution; }
void* Context::symbol(Capability capability, const char* name, Status& status) {
    status.absorb(supports(capability));
    return status ? impl_->libraries[static_cast<int>(capability)].find(name, status) : nullptr;
}
Status Context::push() { Status s = initialize(); return s ? Status::read("CUDA", impl_->push(impl_->context), "cuCtxPushCurrent_v2") : s; }
Status Context::pop() { Handle previous = nullptr; return Status::read("CUDA", impl_->pop(&previous), "cuCtxPopCurrent_v2"); }
Status Context::allocate(Address& pointer, std::size_t bytes) { return Status::read("CUDA", impl_->allocate(&pointer, bytes), "cuMemAlloc_v2"); }
Status Context::release(Address pointer) { return Status::read("CUDA", impl_->free(pointer), "cuMemFree_v2"); }
Status Context::upload(Address pointer, const void* data, std::size_t bytes) {
    Status s = Status::read("CUDA", impl_->upload(pointer, data, bytes), "cuMemcpyHtoD_v2");
    // Pageable host copies may return after staging. Complete the default
    // stream transfer before a nonblocking solver stream reads the allocation.
    if (s) s.absorb(synchronize_stream(nullptr));
    return s;
}
Status Context::download(void* data, Address pointer, std::size_t bytes) { return Status::read("CUDA", impl_->download(data, pointer, bytes), "cuMemcpyDtoH_v2"); }
Status Context::synchronize() { return Status::read("CUDA", impl_->sync(), "cuCtxSynchronize"); }
Status Context::create_stream(void*& stream) { return Status::read("CUDA", impl_->stream_create(&stream, 1), "cuStreamCreate"); }
Status Context::destroy_stream(void* stream) { return Status::read("CUDA", impl_->stream_destroy(stream), "cuStreamDestroy_v2"); }
Status Context::synchronize_stream(void* stream) { return Status::read("CUDA", impl_->stream_sync(stream), "cuStreamSynchronize"); }
Status Context::memory_available(std::size_t& free, std::size_t& total) {
    return Status::read("CUDA",impl_->memory_info(&free,&total),"cuMemGetInfo_v2");
}
Status Context::gemv(Address matrix, Address x, Address y, int rows, int cols,
                     bool transpose, double beta, Handle stream) {
    Status s = supports(Capability::Blas);
    if (s) s = Status::read("cuBLAS", impl_->blas_stream(impl_->blas, stream), "cublasSetStream_v2");
    const double alpha = 1;
    if (s) s = Status::read("cuBLAS", impl_->gemv(impl_->blas, transpose ? 1 : 0,
        rows, cols, &alpha, reinterpret_cast<const double*>(matrix), rows,
        reinterpret_cast<const double*>(x), 1, &beta, reinterpret_cast<double*>(y), 1),
        "cublasDgemv_v2");
    if (impl_->blas_stream && impl_->blas)
        s.absorb(Status::read("cuBLAS", impl_->blas_stream(impl_->blas, nullptr), "cublasSetStream_v2"));
    return s;
}
Status Context::factor(Address pointers, int n, int batch, Address info) {
    return Status::read("cuSOLVER", impl_->potrf(impl_->solver, 0, n,
        reinterpret_cast<double**>(pointers), n, reinterpret_cast<int*>(info), batch), "cusolverDnDpotrfBatched");
}

Lease::Lease(Context& context, Status& status) : context_(context), status_(status),
    lock_(context.execution_mutex()) {
    if (status_) { status_.absorb(context_.push()); pushed_ = static_cast<bool>(status_); }
}
Lease::~Lease() { if (pushed_) status_.absorb(context_.pop()); }
Buffer::Buffer(Context& context, std::size_t bytes, Status& status) :
    context_(context), status_(status), bytes_(bytes) {
    if (bytes && status_) status_.absorb(context_.allocate(pointer_, bytes));
}
Buffer::~Buffer() { if (pointer_) status_.absorb(context_.release(pointer_)); }
Status Buffer::upload(const void* data, std::size_t bytes) {
    if (bytes > bytes_) return Status::read("buffer", -1, "upload", "copy exceeds allocation");
    return bytes ? context_.upload(pointer_, data, bytes) : Status{};
}
Status Buffer::download(void* data, std::size_t bytes) {
    if (bytes > bytes_) return Status::read("buffer", -1, "download", "copy exceeds allocation");
    return bytes ? context_.download(data, pointer_, bytes) : Status{};
}

Status Context::compile(const std::string& source, const std::string& name,
                         Handle& module, Handle& kernel) {
    Status s = supports(Capability::Compiler);
    if (!s) return s;
    Handle program = nullptr;
    std::vector<std::string> header_names, header_text;
    {
        std::lock_guard<std::mutex> lock(settings_mutex);
        for (const auto& entry : kernel_sources) if (entry.first.find(".h") != std::string::npos) {
            header_names.push_back(entry.first); header_text.push_back(entry.second);
            if (entry.first.rfind("tulpa/",0)==0) {
                header_names.push_back(entry.first.substr(6)); header_text.push_back(entry.second);
            }
        }
    }
    std::vector<const char*> names, headers;
    for (std::size_t i=0; i<header_names.size(); ++i) {
        names.push_back(header_names[i].c_str()); headers.push_back(header_text[i].c_str());
    }
    s = Status::read("NVRTC", impl_->program_create(&program, source.c_str(), "tulpa.cu",
        static_cast<int>(names.size()), headers.data(), names.data()), "nvrtcCreateProgram");
    if (!s) return s;
    const auto destroy = [this, &s](void* value) {
        s.absorb(Status::read("NVRTC", impl_->program_destroy(&value), "nvrtcDestroyProgram"));
    };
    std::unique_ptr<void, decltype(destroy)> owned_program(program, destroy);
    // The driver's device capability determines PTX's target architecture.
    const std::string target = "--gpu-architecture=compute_" +
        impl_->info.compute_capability.substr(0, impl_->info.compute_capability.find('.')) +
        impl_->info.compute_capability.substr(impl_->info.compute_capability.find('.') + 1);
    const char* options[] = {"--std=c++17", "--fmad=false", target.c_str()};
    int code = impl_->program_compile(program, 3, options);
    if (code) {
        std::size_t length = 0;
        Status log_status = Status::read("NVRTC", impl_->log_size(program, &length), "nvrtcGetProgramLogSize");
        std::vector<char> log(length ? length : 1, 0);
        if (log_status) log_status = Status::read("NVRTC", impl_->log(program, log.data()), "nvrtcGetProgramLog");
        s = Status::read("NVRTC", code, "nvrtcCompileProgram", log_status ? log.data() : log_status.message);
    }
    std::size_t length = 0;
    if (s) s = Status::read("NVRTC", impl_->ptx_size(program, &length), "nvrtcGetPTXSize");
    std::vector<char> ptx(length);
    if (s) s = Status::read("NVRTC", impl_->ptx(program, ptx.data()), "nvrtcGetPTX");
    owned_program.reset();
    if (s) s = Status::read("CUDA", impl_->module_load(&module, ptx.data()), "cuModuleLoadData");
    if (s) s = Status::read("CUDA", impl_->function(&kernel, module, name.c_str()), "cuModuleGetFunction");
    if (!s && module) { s.absorb(unload(module)); module = nullptr; kernel = nullptr; }
    return s;
}
Status Context::unload(Handle module) { return Status::read("CUDA", impl_->module_unload(module), "cuModuleUnload"); }
Status Context::launch(const std::string& source, const std::string& name,
                        const Launch& shape, void** arguments) {
    Status s;
    std::size_t generation;
    { std::lock_guard<std::mutex> lock(settings_mutex); generation=source_generation; }
    const auto key = std::make_tuple(generation,source,name);
    auto found = impl_->kernels.find(key);
    if (found == impl_->kernels.end()) {
        Impl::CompiledKernel entry;
        s = compile(source, name, entry.module, entry.function);
        if (!s) return s;
        const auto destroy = [this, &s](void* value) { s.absorb(unload(value)); };
        std::unique_ptr<void, decltype(destroy)> owned_module(entry.module, destroy);
        found = impl_->kernels.emplace(key, entry).first;
        owned_module.release();
    }
    return Status::read("CUDA", impl_->launch(found->second.function,
        shape.grid_x, shape.grid_y, shape.grid_z,
        shape.block_x, shape.block_y, shape.block_z,
        shape.shared_bytes, shape.stream, arguments, nullptr), "cuLaunchKernel");
}

const std::vector<Operation>& operations() {
    static const std::vector<Operation> table = {
        {"nngp_cholesky", {Capability::Solver}}, {"axpy", {Capability::Compiler}},
        {"sparse_cholesky", {Capability::Sparse}},
        {"likelihood", {Capability::Compiler, Capability::Blas}},
        {"likelihood_weights", {Capability::Compiler, Capability::Blas}}
    };
    return table;
}
static const Operation* registered_operation(const std::string& name) {
    const auto& table=operations();
    auto found=std::find_if(table.begin(),table.end(),[&](const Operation& op) { return op.name==name; });
    return found==table.end() ? nullptr : &*found;
}
Status Context::supports_operation(const std::string& name) {
    const auto* operation=registered_operation(name);
    if (!operation) return Status::read("policy",-1,"select","unregistered operation "+name);
    Status status;
    for (auto capability : operation->capabilities) {
        status=supports(capability);
        if (!status) break;
    }
    return status;
}
Choice parse_choice(const std::string& value) {
    if (value == "auto") return Choice::Auto;
    if (value == "cpu") return Choice::Cpu;
    if (value == "cuda") return Choice::Cuda;
    throw std::invalid_argument("control$device must be one of 'auto', 'cpu', 'cuda'");
}
void configure(const std::vector<Threshold>& value,
                const std::map<std::string, std::string>& sources) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    settings = value; kernel_sources = sources; ++source_generation;
}
std::vector<Threshold> thresholds() { std::lock_guard<std::mutex> lock(settings_mutex); return settings; }
std::string kernel_source(const std::string& name) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    auto found = kernel_sources.find(name);
    return found == kernel_sources.end() ? "" : found->second;
}
std::shared_ptr<Policy> current_policy() {
    auto result = std::atomic_load(&active_policy);
    return result ? result : std::make_shared<Policy>(Choice::Auto, thresholds());
}
std::shared_ptr<Policy> exchange_policy(std::shared_ptr<Policy> policy) {
    return std::atomic_exchange(&active_policy, std::move(policy));
}
Policy::Policy(Choice choice, const std::vector<Threshold>& thresholds) : choice_(choice), thresholds_(thresholds) {}
bool Policy::select(const std::string& name, std::size_t size, int variant, Status& status) const {
    if (!registered_operation(name)) { status = Status::read("policy", -1, "select", "unregistered operation " + name); return false; }
    if (choice_ == Choice::Cpu || !size) return false;
    if (choice_ == Choice::Auto) {
        auto threshold = std::find_if(thresholds_.begin(), thresholds_.end(), [&](const Threshold& x) {
            return x.operation == name && x.variant == variant;
        });
        if (threshold == thresholds_.end() || size < threshold->size || !std::isfinite(threshold->size)) return false;
    }
    Status available = Context::instance().supports_operation(name);
    if (!available) {
        if (choice_ == Choice::Cuda) status = available;
        return false;
    }
    return true;
}
void Policy::record(const std::string& name, bool cuda) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto stage = std::find_if(stages_.begin(), stages_.end(), [&](const Stage& x) { return x.operation == name; });
    if (stage == stages_.end()) { stages_.push_back({name, 0, 0}); stage = stages_.end() - 1; }
    if (cuda) ++stage->cuda; else ++stage->cpu;
}
void Policy::fail(const Status& status) { std::lock_guard<std::mutex> lock(mutex_); failure_.absorb(status); }
Status Policy::failure() const { std::lock_guard<std::mutex> lock(mutex_); return failure_; }
std::vector<Stage> Policy::stages() const { std::lock_guard<std::mutex> lock(mutex_); return stages_; }
std::shared_ptr<void> Policy::workspace(const void* owner, const void* context,
    const std::string& kind, const std::function<std::shared_ptr<void>()>& create) {
    const auto key = std::make_tuple(std::this_thread::get_id(),owner,context,kind);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = workspaces_.find(key);
        if (found != workspaces_.end()) return found->second;
    }
    auto value = create();
    std::lock_guard<std::mutex> lock(mutex_);
    return workspaces_.emplace(key,std::move(value)).first->second;
}

Status cholesky_batch(std::vector<std::vector<double>>& matrices, int n,
                       Policy& policy, bool& used_cuda,
                       const std::vector<int>* widths) {
    used_cuda = false;
    Status s;
    if (n <= 0 || matrices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return Status::read("device", -1, "cholesky_batch", "invalid batch dimensions");
    const std::size_t stride = static_cast<std::size_t>(n) * n;
    for (const auto& matrix : matrices) if (matrix.size() != stride)
        return Status::read("device", -1, "cholesky_batch", "matrix size differs from n*n");
    if (widths) {
        if (widths->size() != matrices.size())
            return Status::read("device", -1, "cholesky_batch", "batch widths differ from batch size");
        for (int width : *widths) if (width <= 0 || width > n)
            return Status::read("device", -1, "cholesky_batch", "invalid effective matrix width");
    }
    if (matrices.empty()) return s;
    const bool cuda = policy.select("nngp_cholesky", matrices.size(), n, s);
    if (!s) return s;
    if (cuda) {
        std::vector<double> packed(matrices.size() * stride);
        for (std::size_t b = 0; b < matrices.size(); ++b)
            std::copy(matrices[b].begin(), matrices[b].end(), packed.begin() + b * stride);
        auto& context = Context::instance();
        {
            Lease lease(context, s);
            Buffer data(context, packed.size() * sizeof(double), s);
            Buffer pointers(context, matrices.size() * sizeof(Address), s);
            Buffer codes(context, matrices.size() * sizeof(int), s);
            std::vector<Address> addresses(matrices.size());
            for (std::size_t b = 0; b < matrices.size(); ++b)
                addresses[b] = data.address() + b * stride * sizeof(double);
            if (s) s = data.upload(packed.data(), data.size());
            if (s) s = pointers.upload(addresses.data(), pointers.size());
            if (s) s = context.factor(pointers.address(), n, static_cast<int>(matrices.size()), codes.address());
            if (s) s = context.synchronize();
            std::vector<int> info(matrices.size());
            if (s) s = codes.download(info.data(), codes.size());
            if (s) for (std::size_t b = 0; b < info.size(); ++b) {
                if (info[b]) { s = Status::read("cuSOLVER", info[b], "potrf matrix", "batch index " + std::to_string(b)); break; }
            }
            if (s) s = data.download(packed.data(), data.size());
        }
        if (!s) return s;
        // Positive pivots and the diagonal of L L' check every matrix. One
        // complete CPU factor also verifies the column-major handoff.
        for (std::size_t b = 0; b < matrices.size(); ++b) {
            const double* L = packed.data() + b * stride;
            for (int r = 0; r < n; ++r) {
                double diagonal = 0;
                for (int c = 0; c <= r; ++c) diagonal += L[c * n + r] * L[c * n + r];
                if (!(L[r * n + r] > 0) || !std::isfinite(diagonal) ||
                    std::abs(diagonal - matrices[b][r * n + r]) >
                        1e-8 * std::max(std::abs(matrices[b][r * n + r]), 1e-12))
                    return Status::read("cuSOLVER", -1, "verify factor", "batch index " + std::to_string(b));
            }
        }
        auto reference = matrices[matrices.size() / 2];
        if (!tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(reference.data(), reference.data(), n, n, 0.0))
            return Status::read("CPU", -1, "verify factor", "reference is not positive definite");
        const double* probe = packed.data() + (matrices.size() / 2) * stride;
        double scale = 0, difference = 0;
        for (int r = 0; r < n; ++r) for (int c = 0; c <= r; ++c) {
            scale = std::max(scale, std::abs(reference[r * n + c]));
            difference = std::max(difference, std::abs(reference[r * n + c] - probe[c * n + r]));
        }
        if (!(difference <= 1e-8 * std::max(scale, 1e-12)))
            return Status::read("cuSOLVER", -1, "verify factor", "CPU factor differs");
        for (std::size_t b = 0; b < matrices.size(); ++b) {
            auto& L = matrices[b];
            const double* column_major = packed.data() + b * stride;
            for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c)
                L[r * n + c] = c <= r ? column_major[c * n + r] : 0.0;
        }
        used_cuda = true;
    } else {
        for (std::size_t b = 0; b < matrices.size(); ++b) {
            auto& L = matrices[b];
            const int width = widths ? (*widths)[b] : n;
            if (!tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(L.data(), L.data(), width, n, 0.0))
                return Status::read("CPU", 1, "cholesky_batch", "matrix is not positive definite");
            for (int r = 0; r < n; ++r) for (int c = r + 1; c < n; ++c) L[r * n + c] = 0.0;
        }
    }
    policy.record("nngp_cholesky", used_cuda);
    return s;
}

Status axpy(const std::vector<double>& x, std::vector<double>& y, double a,
             Policy& policy, bool& used_cuda) {
    Status s;
    used_cuda = false;
    if (x.size() != y.size() || x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return Status::read("device", -1, "axpy", "incompatible vector sizes");
    const bool cuda = policy.select("axpy", x.size(), 0, s);
    if (!s) return s;
    if (cuda) {
        auto& context = Context::instance();
        {
            Lease lease(context, s);
            Buffer dx(context, x.size() * sizeof(double), s), dy(context, y.size() * sizeof(double), s);
            if (s) s = dx.upload(x.data(), dx.size());
            if (s) s = dy.upload(y.data(), dy.size());
            Address px = dx.address(), py = dy.address();
            int n = static_cast<int>(x.size());
            void* arguments[] = {&px, &py, &a, &n};
            Launch shape;
            shape.grid_x = (static_cast<unsigned>(n) + shape.block_x - 1) / shape.block_x;
            if (s) s = context.launch(kernel_source("axpy"), "tulpa_axpy", shape, arguments);
            if (s) s = context.synchronize();
            if (s) s = dy.download(y.data(), dy.size());
        }
        if (!s) return s;
        used_cuda = true;
    } else {
        for (std::size_t i = 0; i < x.size(); ++i) y[i] += a * x[i];
    }
    policy.record("axpy", used_cuda);
    return s;
}

} }
