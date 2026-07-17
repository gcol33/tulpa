// hmc_chain_checkpoint.h
// Per-chain checkpoint/resume for multi-chain NUTS/HMC.
//
// The natural checkpoint boundary for the sampler is a whole chain: each chain
// is run independently with its own (seed, chain_id), so its result is a
// deterministic function of the data, sampler settings and that chain index.
// Checkpointing at the chain boundary is therefore correct by construction --
// a resumed chain c is bit-for-bit identical to the uninterrupted chain c, with
// no mid-trajectory RNG / dual-averaging / metric state to restore -- and lets
// a crashed M-chain run keep its completed chains and re-run only the missing
// ones. The on-disk format and load/append/torn-tail logic are the generic
// CheckpointLog in checkpoint_io.h; this file is the HMCResultCpp specialization
// (ChainCheckpoint), keyed by the chain index.

#ifndef TULPA_HMC_CHAIN_CHECKPOINT_H
#define TULPA_HMC_CHAIN_CHECKPOINT_H

#include "checkpoint_io.h"
#include "hmc_sampler_chain_state.h"   // HMCResultCpp
#include <string>
#include <vector>

// HMCResultCpp payload (de)serialization -- the customization points the generic
// CheckpointLog finds by ADL. They live in tulpa_hmc (the namespace of
// HMCResultCpp) so argument-dependent lookup from CheckpointLog::save/load
// reaches them. Every field consumed downstream (cpp_to_r_result, the warm-start
// outputs, the collapsed-mode draws) is written so a loaded chain is
// indistinguishable from a freshly sampled one.
namespace tulpa_hmc {

using tulpa::ckpt_put;
using tulpa::ckpt_put_span;
using tulpa::CkptReader;

inline std::string ckpt_serialize(const HMCResultCpp& r) {
    std::string buf;
    ckpt_put_span(buf, r.samples_flat);
    ckpt_put<std::int32_t>(buf, r.n_params_stored);
    ckpt_put_span(buf, r.log_prob);
    ckpt_put_span(buf, r.accept_prob);
    ckpt_put_span(buf, r.n_leapfrog);
    ckpt_put_span(buf, r.divergent);
    ckpt_put_span(buf, r.treedepth);
    ckpt_put(buf, r.epsilon);
    ckpt_put<std::int32_t>(buf, r.n_warmup);
    ckpt_put<std::int32_t>(buf, r.n_sample);
    ckpt_put<std::int32_t>(buf, r.chain_id);
    ckpt_put<std::int32_t>(buf, r.n_max_treedepth);
    std::uint64_t slen = r.sampler.size();
    ckpt_put(buf, slen);
    if (slen) buf.append(r.sampler.data(), r.sampler.size());
    ckpt_put_span(buf, r.inv_metric_diag);
    ckpt_put_span(buf, r.final_position);
    ckpt_put<std::int32_t>(buf, r.n_gp_collapsed);
    ckpt_put<std::int32_t>(buf, r.n_icar_collapsed);
    ckpt_put_span(buf, r.gp_w_star_flat);
    ckpt_put_span(buf, r.icar_phi_star_flat);
    ckpt_put_span(buf, r.bym2_theta_star_flat);
    return buf;
}

inline bool ckpt_deserialize(CkptReader& rd, HMCResultCpp& r) {
    r.samples_flat    = rd.get_span<double>();
    r.n_params_stored = rd.get<std::int32_t>();
    r.log_prob        = rd.get_span<double>();
    r.accept_prob     = rd.get_span<double>();
    r.n_leapfrog      = rd.get_span<int>();
    r.divergent       = rd.get_span<int>();
    r.treedepth       = rd.get_span<int>();
    r.epsilon         = rd.get<double>();
    r.n_warmup        = rd.get<std::int32_t>();
    r.n_sample        = rd.get<std::int32_t>();
    r.chain_id        = rd.get<std::int32_t>();
    r.n_max_treedepth = rd.get<std::int32_t>();
    std::uint64_t slen = rd.get<std::uint64_t>();
    if (!rd.ok) return false;
    if (slen) {
        if (rd.p + slen > rd.end) { rd.ok = false; return false; }
        r.sampler.assign(rd.p, rd.p + slen);
        rd.p += slen;
    } else {
        r.sampler.clear();
    }
    r.inv_metric_diag     = rd.get_span<double>();
    r.final_position      = rd.get_span<double>();
    r.n_gp_collapsed      = rd.get<std::int32_t>();
    r.n_icar_collapsed    = rd.get<std::int32_t>();
    r.gp_w_star_flat      = rd.get_span<double>();
    r.icar_phi_star_flat  = rd.get_span<double>();
    r.bym2_theta_star_flat = rd.get_span<double>();
    return rd.ok;
}

} // namespace tulpa_hmc

namespace tulpa {

using ChainCheckpoint = CheckpointLog<tulpa_hmc::HMCResultCpp>;

// Keys are the chain index encoded as raw bytes (a chain is a "cell"); the
// fingerprint must already cover the data + sampler settings + per-chain seed so
// a resume onto a file from a different run errors. n_chains keys.
inline std::vector<std::string> chain_checkpoint_keys(int n_chains) {
    std::vector<std::string> keys(n_chains);
    for (int c = 0; c < n_chains; c++) {
        keys[c].append(reinterpret_cast<const char*>(&c), sizeof(int));
    }
    return keys;
}

} // namespace tulpa

#endif // TULPA_HMC_CHAIN_CHECKPOINT_H
