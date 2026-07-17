// checkpoint_io.h
// Generic content-addressed append-log checkpoint.
//
// The outer loop of a long fit -- a hyperparameter grid (nested Laplace), a
// covariance-CCD node grid, or a set of independent MCMC chains -- is a list of
// independent units, each producing a self-contained result. This file owns the
// shared on-disk format and the load/append/torn-tail logic ONCE, as a template
// over the per-unit payload type. A consumer specializes it by providing two
// free functions for its payload, found by ADL:
//
//     std::string ckpt_serialize(const Payload&);
//     bool        ckpt_deserialize(CkptReader&, Payload&);
//
// and a `using MyCheckpoint = CheckpointLog<Payload>;`. GridCheckpoint
// (LaplaceResult) and ChainCheckpoint (HMCResultCpp) are the two instantiations.
//
// File format (little-endian, host-native; not portable across endianness,
// acceptable for a resume-on-the-same-machine workflow):
//
//   Header (once):
//     char[8]  magic = "TLPACKP1"
//     uint64   fingerprint
//   Record (appended per completed unit):
//     uint32   key_len
//     bytes    key            (raw bytes of the unit coordinate)
//     uint32   payload_len
//     bytes    payload        (ckpt_serialize output)
//     uint64   checksum       (FNV-1a over key ++ payload)
//
// A torn final record (process killed mid-write) is detected on load by a short
// read or a checksum mismatch and discarded; that unit is simply re-run. A
// header fingerprint mismatch errors rather than resuming onto a stale result.

#ifndef TULPA_CHECKPOINT_IO_H
#define TULPA_CHECKPOINT_IO_H

#include <Rcpp.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tulpa {

// FNV-1a 64-bit. Used for both the header fingerprint and per-record checksums.
inline std::uint64_t fnv1a64(const void* data, std::size_t n,
                             std::uint64_t h = 1469598103934665603ULL) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Append a POD value's raw bytes to a byte buffer.
template <typename T>
inline void ckpt_put(std::string& buf, const T& v) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "ckpt_put requires a trivially-copyable type");
    buf.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

// Append a contiguous span of POD values (length-prefixed by the caller).
template <typename T>
inline void ckpt_put_span(std::string& buf, const std::vector<T>& v) {
    std::uint64_t n = v.size();
    ckpt_put(buf, n);
    if (n) buf.append(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
}

// Running FNV-1a fingerprint of everything that changes a unit's result given
// its coordinate (data, designs, solver settings, layout). Each fitter folds
// its own fields; the value guards the checkpoint header. fold_vec writes only
// the data bytes (no length prefix) -- callers fold the dimensions that
// disambiguate lengths separately.
struct Fingerprint {
    std::uint64_t h = 1469598103934665603ULL;

    void fold(const void* d, std::size_t n) { h = fnv1a64(d, n, h); }

    template <typename T>
    void fold_pod(const T& v) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "fold_pod requires a trivially-copyable type");
        fold(&v, sizeof(T));
    }

    void fold_str(const std::string& s) { fold(s.data(), s.size()); }

    template <typename T>
    void fold_vec(const std::vector<T>& v) {
        if (!v.empty()) fold(v.data(), v.size() * sizeof(T));
    }

    std::uint64_t value() const { return h; }
};

// Recursively fold an arbitrary R object's contents into a fingerprint, so a
// heterogeneous spec (adjacency, coords, type strings, nested lists) invalidates
// a stale resume without enumerating its fields by hand.
inline void fold_sexp(Fingerprint& fp, SEXP x) {
    int type = TYPEOF(x);
    fp.fold_pod(type);
    switch (type) {
        case REALSXP: {
            R_xlen_t n = Rf_xlength(x);
            if (n) fp.fold(REAL(x), (std::size_t)n * sizeof(double));
            break;
        }
        case INTSXP:
        case LGLSXP: {
            R_xlen_t n = Rf_xlength(x);
            if (n) fp.fold(INTEGER(x), (std::size_t)n * sizeof(int));
            break;
        }
        case STRSXP: {
            R_xlen_t n = Rf_xlength(x);
            for (R_xlen_t i = 0; i < n; i++) {
                const char* c = CHAR(STRING_ELT(x, i));
                fp.fold(c, std::strlen(c));
            }
            break;
        }
        case VECSXP: {
            R_xlen_t n = Rf_xlength(x);
            for (R_xlen_t i = 0; i < n; i++) fold_sexp(fp, VECTOR_ELT(x, i));
            break;
        }
        default:
            break;
    }
}

// Builds the per-cell keys for a rectilinear (or paired) outer grid. Each axis
// is a length-n_grid column of per-cell coordinate values; key[k] is the
// concatenated raw double bytes of every axis at k.
class CellKeyBuilder {
public:
    explicit CellKeyBuilder(int n_grid) : keys_(n_grid) {}

    void add_axis(const double* col, std::size_t stride = 1) {
        for (std::size_t k = 0; k < keys_.size(); k++) {
            double v = col[k * stride];
            keys_[k].append(reinterpret_cast<const char*>(&v), sizeof(double));
        }
    }

    void add_axis(const std::vector<double>& col) {
        // A degenerate (constant) axis carries no per-cell values -- e.g. rw1 /
        // rw2 pass an empty rho_grid. It contributes nothing to the keys;
        // reading n_grid doubles from an empty column would run off the buffer.
        if (col.empty()) return;
        add_axis(col.data(), 1);
    }

    std::vector<std::string> take() { return std::move(keys_); }

private:
    std::vector<std::string> keys_;
};

// Cursor-based reader over a byte buffer. Every read is bounds-checked; a read
// past the end sets `ok = false` so a truncated tail is detected, not faulted.
struct CkptReader {
    const char* p;
    const char* end;
    bool        ok = true;

    CkptReader(const char* data, std::size_t n) : p(data), end(data + n) {}

    template <typename T>
    T get() {
        T v{};
        if (!ok || p + sizeof(T) > end) { ok = false; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }

    template <typename T>
    std::vector<T> get_span() {
        std::vector<T> v;
        std::uint64_t n = get<std::uint64_t>();
        if (!ok) return v;
        // Cap n against the bytes actually remaining BEFORE computing
        // n * sizeof(T) (which could overflow a uint64 and wrap past the check)
        // or resizing (a torn/corrupt length field would otherwise allocate
        // gigabytes). A length that cannot fit is a truncated tail, not a value.
        std::size_t remaining = static_cast<std::size_t>(end - p);
        if (n > remaining / sizeof(T)) { ok = false; return v; }
        std::size_t nn = static_cast<std::size_t>(n);
        v.resize(nn);
        if (nn) {
            std::memcpy(v.data(), p, nn * sizeof(T));
            p += nn * sizeof(T);
        }
        return v;
    }
};

// Customization points: each payload header provides a non-template overload
// (`ckpt_serialize(const LaplaceResult&)`, etc.) found by overload resolution at
// instantiation. These primary templates make the names visible at the template
// definition below; they are never instantiated (a non-template overload always
// wins) and so are intentionally left undefined.
template <typename T> std::string ckpt_serialize(const T&);
template <typename T> bool        ckpt_deserialize(CkptReader&, T&);

// Generic per-fit checkpoint handle over a payload type. Constructed once per
// run where the unit coordinates (keys) are known; passed by pointer into the
// outer loop. `ckpt_serialize` / `ckpt_deserialize` for Payload must be visible
// (overload resolution) at instantiation.
template <typename Payload>
class CheckpointLog {
public:
    static constexpr char MAGIC[8] = {'T','L','P','A','C','K','P','1'};

    CheckpointLog(const std::string& path,
                  std::uint64_t fingerprint,
                  std::vector<std::string> keys)
        : path_(path), keys_(std::move(keys)) {
        std::ifstream in(path_, std::ios::binary);
        if (in.good()) {
            std::uintmax_t good_bytes = load_existing(in, fingerprint);
            in.close();
            // Drop any torn tail (a partial record from a killed write) by
            // truncating to the last fully-valid record before appending; else
            // new records would be written AFTER the orphaned bytes and a later
            // load would stop at the torn boundary and never see them.
            std::error_code ec;
            std::filesystem::resize_file(path_, good_bytes, ec);
            out_.open(path_, std::ios::binary | std::ios::app);
        } else {
            out_.open(path_, std::ios::binary | std::ios::trunc);
            if (out_) {
                out_.write(MAGIC, sizeof(MAGIC));
                out_.write(reinterpret_cast<const char*>(&fingerprint),
                           sizeof(fingerprint));
                out_.flush();
            }
        }
        if (!out_) {
            Rcpp::stop("checkpoint: cannot open '%s' for writing.", path_.c_str());
        }
    }

    bool has(int k) const {
        if (k < 0 || k >= static_cast<int>(keys_.size())) return false;
        return loaded_.find(keys_[k]) != loaded_.end();
    }

    const Payload& get(int k) const { return loaded_.at(keys_[k]); }

    // Append unit k's completed result. Thread-safe: the outer loop may call
    // this from inside an OpenMP parallel region.
    void save(int k, const Payload& res) {
        if (k < 0 || k >= static_cast<int>(keys_.size())) return;
        const std::string& key = keys_[k];
        std::string payload = ckpt_serialize(res);
        std::uint64_t chk = fnv1a64(payload.data(), payload.size(),
                                    fnv1a64(key.data(), key.size()));
        std::lock_guard<std::mutex> lock(mu_);
        if (loaded_.find(key) != loaded_.end()) return;  // idempotent
        std::uint32_t key_len = static_cast<std::uint32_t>(key.size());
        std::uint32_t pay_len = static_cast<std::uint32_t>(payload.size());
        out_.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        out_.write(key.data(), key.size());
        out_.write(reinterpret_cast<const char*>(&pay_len), sizeof(pay_len));
        out_.write(payload.data(), payload.size());
        out_.write(reinterpret_cast<const char*>(&chk), sizeof(chk));
        out_.flush();
        loaded_.emplace(key, res);
    }

    int n_loaded() const {
        int n = 0;
        for (const auto& k : keys_) if (loaded_.find(k) != loaded_.end()) n++;
        return n;
    }

private:
    // Reads the header (validating magic + fingerprint) and every intact
    // record, stopping at EOF or a torn tail. Returns the byte offset just past
    // the last fully-valid record (the truncation point), so the caller can drop
    // a partial trailing record before appending.
    std::uintmax_t load_existing(std::ifstream& in, std::uint64_t fingerprint) {
        char magic[8];
        in.read(magic, sizeof(magic));
        if (!in || std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
            Rcpp::stop("checkpoint: '%s' is not a tulpa checkpoint file "
                       "(bad magic). Point `checkpoint$path` at a fresh "
                       "location or remove the file.", path_.c_str());
        }
        std::uint64_t stored_fp = 0;
        in.read(reinterpret_cast<char*>(&stored_fp), sizeof(stored_fp));
        if (!in) Rcpp::stop("checkpoint: '%s' header is truncated.", path_.c_str());
        if (stored_fp != fingerprint) {
            Rcpp::stop("checkpoint: '%s' was written for different data or "
                       "solver settings (fingerprint mismatch). Resume needs "
                       "the same inputs + grid + control; use a fresh path "
                       "or set checkpoint$resume = FALSE to start over.",
                       path_.c_str());
        }
        // File size, so a torn/garbage record length cannot drive a multi-GiB
        // string allocation before the short read is detected as a torn tail.
        std::streampos after_hdr = in.tellg();
        in.seekg(0, std::ios::end);
        std::uintmax_t file_size = static_cast<std::uintmax_t>(in.tellg());
        in.seekg(after_hdr);
        std::uintmax_t good_bytes = sizeof(MAGIC) + sizeof(stored_fp);
        while (true) {
            std::uint32_t key_len = 0, pay_len = 0;
            in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
            if (!in) break;
            if (key_len > file_size - static_cast<std::uintmax_t>(in.tellg())) break;
            std::string key(key_len, '\0');
            in.read(&key[0], key_len);
            if (!in) break;
            in.read(reinterpret_cast<char*>(&pay_len), sizeof(pay_len));
            if (!in) break;
            if (pay_len > file_size - static_cast<std::uintmax_t>(in.tellg())) break;
            std::string payload(pay_len, '\0');
            in.read(&payload[0], pay_len);
            if (!in) break;
            std::uint64_t chk = 0;
            in.read(reinterpret_cast<char*>(&chk), sizeof(chk));
            if (!in) break;
            std::uint64_t want = fnv1a64(payload.data(), payload.size(),
                                         fnv1a64(key.data(), key.size()));
            if (chk != want) break;  // torn / corrupt tail
            CkptReader rd(payload.data(), payload.size());
            Payload r;
            if (!ckpt_deserialize(rd, r)) break;
            loaded_[key] = std::move(r);
            good_bytes = static_cast<std::uintmax_t>(in.tellg());
        }
        return good_bytes;
    }

    std::string                              path_;
    std::vector<std::string>                 keys_;
    std::unordered_map<std::string, Payload> loaded_;
    std::ofstream                            out_;
    std::mutex                               mu_;
};

template <typename Payload>
constexpr char CheckpointLog<Payload>::MAGIC[8];

} // namespace tulpa

#endif // TULPA_CHECKPOINT_IO_H
