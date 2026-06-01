// nested_laplace_checkpoint.h
// Grid-cell checkpoint/resume for the nested-Laplace outer grid (gcol33/tulpa#50).
//
// The outer grid is the natural checkpoint boundary: each cell over the
// hyperparameter coordinate is independent and produces a self-contained
// LaplaceResult. A killed or interrupted fit can resume from the last completed
// cell instead of restarting, which matters for EVA-scale joint fits that run
// for hours.
//
// Design
// ------
// The file is a content-addressed append log keyed by the cell's hyperparameter
// coordinate (the theta-grid row plus any per-arm phi-grid value). Keying by the
// coordinate -- not the integer grid index -- means an adaptive-grid refinement
// pass that appends newly spawned cells stores them under their own keys, and a
// resume that re-runs the whole driver (initial grid then refinement) hits every
// previously completed cell regardless of the order cells are visited. Two runs
// with the same data + settings but a changed grid simply miss (every cell
// re-solved), never return a wrong cached result.
//
// A header fingerprint guards against using a checkpoint written for different
// data or solver settings: the constructor errors if a present file's
// fingerprint disagrees, rather than silently resuming onto a stale result.
// "Fresh vs resume" is decided by the R front door (it removes the file before
// the first kernel call when resume = FALSE); at this layer a present, matching
// file is always loaded, so the several kernel calls within one fit share it.
//
// File format (little-endian, as written by the host; checkpoints are not
// portable across architectures with different endianness, which is acceptable
// for a resume-on-the-same-machine workflow):
//
//   Header (once):
//     char[8]  magic = "TLPACKP1"
//     uint64   fingerprint
//   Record (appended per completed cell):
//     uint32   key_len
//     bytes    key            (raw double bytes of the cell coordinate)
//     uint32   payload_len
//     bytes    payload        (serialized LaplaceResult, see (de)serialize)
//     uint64   checksum       (FNV-1a over key ++ payload)
//
// A torn final record (process killed mid-write) is detected on load by a short
// read or a checksum mismatch and discarded; that cell is simply re-solved.

#ifndef TULPA_NESTED_LAPLACE_CHECKPOINT_H
#define TULPA_NESTED_LAPLACE_CHECKPOINT_H

#include "laplace_core.h"
#include <Rcpp.h>
#include <cstdint>
#include <cstring>
#include <fstream>
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
        if (p + n * sizeof(T) > end) { ok = false; return v; }
        v.resize(n);
        if (n) {
            std::memcpy(v.data(), p, n * sizeof(T));
            p += n * sizeof(T);
        }
        return v;
    }
};

// Serialize a LaplaceResult to a byte buffer. Every field that a resumed cell
// must reproduce bit-for-bit is written, so a loaded cell_results[k] is
// indistinguishable from a freshly solved one.
inline std::string serialize_laplace_result(const LaplaceResult& r) {
    std::string buf;
    ckpt_put(buf, r.log_marginal);
    ckpt_put(buf, r.log_det_Q);
    ckpt_put<std::int32_t>(buf, r.n_iter);
    ckpt_put<std::uint8_t>(buf, r.converged ? 1u : 0u);
    ckpt_put_span(buf, r.mode);
    ckpt_put<std::int32_t>(buf, r.Q_csc_n);
    ckpt_put_span(buf, r.Q_csc_p);
    ckpt_put_span(buf, r.Q_csc_i);
    ckpt_put_span(buf, r.Q_csc_x);
    ckpt_put_span(buf, r.re_cov_flat);
    ckpt_put_span(buf, r.re_cov_block_sizes);
    return buf;
}

inline bool deserialize_laplace_result(CkptReader& rd, LaplaceResult& r) {
    r.log_marginal       = rd.get<double>();
    r.log_det_Q          = rd.get<double>();
    r.n_iter             = rd.get<std::int32_t>();
    r.converged          = (rd.get<std::uint8_t>() != 0);
    r.mode               = rd.get_span<double>();
    r.Q_csc_n            = rd.get<std::int32_t>();
    r.Q_csc_p            = rd.get_span<int>();
    r.Q_csc_i            = rd.get_span<int>();
    r.Q_csc_x            = rd.get_span<double>();
    r.re_cov_flat        = rd.get_span<double>();
    r.re_cov_block_sizes = rd.get_span<int>();
    return rd.ok;
}

// Per-fit checkpoint handle. Constructed once per kernel call where the cell
// coordinates are known; passed by pointer into run_nested_laplace_grid.
class GridCheckpoint {
public:
    static constexpr char    MAGIC[8] = {'T','L','P','A','C','K','P','1'};

    // `cell_keys[k]` is the raw-byte key of cell k's hyperparameter coordinate.
    GridCheckpoint(const std::string& path,
                   std::uint64_t fingerprint,
                   std::vector<std::string> cell_keys)
        : path_(path), keys_(std::move(cell_keys)) {
        std::ifstream in(path_, std::ios::binary);
        bool exists = in.good();
        if (exists) {
            load_existing(in, fingerprint);
            in.close();
            // Reopen for append; the header is already present.
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
            Rcpp::stop("checkpoint: cannot open '%s' for writing.",
                       path_.c_str());
        }
    }

    // True when cell k's coordinate was completed in a prior pass / run.
    bool has(int k) const {
        if (k < 0 || k >= static_cast<int>(keys_.size())) return false;
        return loaded_.find(keys_[k]) != loaded_.end();
    }

    // The stored result for cell k (caller guards with has()).
    const LaplaceResult& get(int k) const {
        return loaded_.at(keys_[k]);
    }

    // Append cell k's completed result. Thread-safe: the outer grid may call
    // this from inside an OpenMP parallel region.
    void save(int k, const LaplaceResult& res) {
        if (k < 0 || k >= static_cast<int>(keys_.size())) return;
        const std::string& key = keys_[k];
        std::string payload = serialize_laplace_result(res);
        std::uint64_t chk = fnv1a64(payload.data(), payload.size(),
                                    fnv1a64(key.data(), key.size()));
        std::lock_guard<std::mutex> lock(mu_);
        // Skip a re-save of an already-present coordinate (idempotent).
        if (loaded_.find(key) != loaded_.end()) return;
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

    // Count of cells in THIS grid that were loaded (for progress accounting).
    int n_loaded() const {
        int n = 0;
        for (const auto& k : keys_) {
            if (loaded_.find(k) != loaded_.end()) n++;
        }
        return n;
    }

private:
    void load_existing(std::ifstream& in, std::uint64_t fingerprint) {
        char magic[8];
        in.read(magic, sizeof(magic));
        if (!in || std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
            Rcpp::stop("checkpoint: '%s' is not a tulpa checkpoint file "
                       "(bad magic). Point `checkpoint$path` at a fresh "
                       "location or remove the file.", path_.c_str());
        }
        std::uint64_t stored_fp = 0;
        in.read(reinterpret_cast<char*>(&stored_fp), sizeof(stored_fp));
        if (!in) {
            Rcpp::stop("checkpoint: '%s' header is truncated.", path_.c_str());
        }
        if (stored_fp != fingerprint) {
            Rcpp::stop("checkpoint: '%s' was written for different data or "
                       "solver settings (fingerprint mismatch). Resume needs "
                       "the same responses + grid + control; use a fresh path "
                       "or set checkpoint$resume = FALSE to start over.",
                       path_.c_str());
        }
        // Records: read until a short read or checksum mismatch (torn tail).
        while (true) {
            std::uint32_t key_len = 0, pay_len = 0;
            in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
            if (!in) break;
            std::string key(key_len, '\0');
            in.read(&key[0], key_len);
            if (!in) break;
            in.read(reinterpret_cast<char*>(&pay_len), sizeof(pay_len));
            if (!in) break;
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
            LaplaceResult r;
            if (!deserialize_laplace_result(rd, r)) break;
            loaded_[key] = std::move(r);
        }
    }

    std::string                                   path_;
    std::vector<std::string>                      keys_;
    std::unordered_map<std::string, LaplaceResult> loaded_;
    std::ofstream                                 out_;
    std::mutex                                    mu_;
};

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_CHECKPOINT_H
