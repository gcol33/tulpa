// hmc_progress.h
// Progress reporting for long-running HMC/NUTS sampling
// Provides real-time feedback without excessive overhead

#ifndef TULPA_HMC_PROGRESS_H
#define TULPA_HMC_PROGRESS_H

#include <Rcpp.h>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

namespace tulpa_progress {

// Check if R is running interactively
inline bool is_interactive() {
  Rcpp::Environment base("package:base");
  Rcpp::Function interactive = base["interactive"];
  return Rcpp::as<bool>(interactive());
}

// Progress reporter for HMC sampling
// Thread-safe when used from a single chain
class ProgressReporter {
public:
  int total_iter;
  int warmup_iter;
  int chain_id;
  int n_chains;
  int update_interval;
  bool verbose;
  bool show_progress_bar;
  bool is_interactive_session;

  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point last_update;
  int last_iter;
  int n_divergent;
  double mean_accept_prob;

  ProgressReporter(int total, int warmup, int chain, int n_ch,
                   bool verbose_mode = true, bool progress_bar = true)
    : total_iter(total), warmup_iter(warmup), chain_id(chain), n_chains(n_ch),
      verbose(verbose_mode), show_progress_bar(progress_bar),
      last_iter(0), n_divergent(0), mean_accept_prob(0.0) {

    // Check if in interactive session (only show \r updates in interactive mode)
    is_interactive_session = is_interactive();

    // Update every 5% or at least every 10 iterations
    update_interval = std::max(10, total_iter / 20);
    start_time = std::chrono::steady_clock::now();
    last_update = start_time;
  }

  // Format duration as human-readable string
  static std::string format_duration(double seconds) {
    if (seconds < 60) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(1) << seconds << "s";
      return oss.str();
    } else if (seconds < 3600) {
      int mins = static_cast<int>(seconds) / 60;
      int secs = static_cast<int>(seconds) % 60;
      std::ostringstream oss;
      oss << mins << "m " << secs << "s";
      return oss.str();
    } else {
      int hours = static_cast<int>(seconds) / 3600;
      int mins = (static_cast<int>(seconds) % 3600) / 60;
      std::ostringstream oss;
      oss << hours << "h " << mins << "m";
      return oss.str();
    }
  }

  // Create progress bar string
  static std::string progress_bar(double pct, int width = 20) {
    int filled = static_cast<int>(pct * width);
    std::string bar = "[";
    for (int i = 0; i < width; i++) {
      if (i < filled) bar += "=";
      else if (i == filled) bar += ">";
      else bar += " ";
    }
    bar += "]";
    return bar;
  }

  // Update progress (call from sampling loop)
  void update(int iter, double accept_prob, bool divergent) {
    if (!verbose) return;

    // Track divergent transitions
    if (divergent) n_divergent++;

    // Update running mean accept probability
    mean_accept_prob = (mean_accept_prob * iter + accept_prob) / (iter + 1);

    // Only print at intervals
    if (iter - last_iter < update_interval && iter < total_iter - 1) {
      return;
    }
    last_iter = iter;

    // Skip intermediate progress updates in non-interactive mode
    // to avoid flooding output with \r-prefixed lines
    if (!is_interactive_session && iter < total_iter - 1) {
      return;
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();

    // Estimate remaining time
    double pct = static_cast<double>(iter + 1) / total_iter;
    double eta = (pct > 0.01) ? elapsed * (1.0 - pct) / pct : 0.0;

    // Determine phase
    std::string phase = (iter < warmup_iter) ? "Warmup" : "Sample";

    // Build progress message
    std::ostringstream msg;

    if (n_chains > 1) {
      msg << "Chain " << (chain_id + 1) << "/" << n_chains << ": ";
    }

    if (show_progress_bar && is_interactive_session) {
      msg << progress_bar(pct) << " ";
    }

    msg << std::fixed << std::setprecision(0) << (pct * 100) << "% ";
    msg << "(" << (iter + 1) << "/" << total_iter << ") ";
    msg << phase;

    if (eta > 1.0 && is_interactive_session) {
      msg << " | ETA: " << format_duration(eta);
    }

    // Print (carriage return for in-place update only in interactive mode)
    if (is_interactive_session) {
      Rcpp::Rcout << "\r" << msg.str() << std::flush;
    }

    // Final newline at completion
    if (iter >= total_iter - 1) {
      Rcpp::Rcout << std::endl;
    }

    last_update = now;
  }

  // Print summary after chain completion
  void summary() {
    if (!verbose) return;

    auto end_time = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    double iter_per_sec = total_iter / total_seconds;
    int post_warmup = total_iter - warmup_iter;

    Rcpp::Rcout << "Chain " << (chain_id + 1);
    if (n_chains > 1) {
      Rcpp::Rcout << "/" << n_chains;
    }
    Rcpp::Rcout << " finished in " << format_duration(total_seconds);
    Rcpp::Rcout << " (" << std::fixed << std::setprecision(1) << iter_per_sec << " iter/s)";

    if (n_divergent > 0) {
      double div_pct = 100.0 * n_divergent / post_warmup;
      Rcpp::Rcout << " | " << n_divergent << " divergent";
      if (div_pct > 1.0) {
        Rcpp::Rcout << " (" << std::setprecision(1) << div_pct << "%)";
      }
    }

    Rcpp::Rcout << std::endl;
  }
};

// Aggregate progress for multiple chains
class MultiChainProgress {
public:
  int n_chains;
  int total_iter;
  int warmup_iter;
  std::vector<int> chain_progress;
  std::vector<int> chain_divergent;
  std::chrono::steady_clock::time_point start_time;
  bool verbose;

  MultiChainProgress(int chains, int total, int warmup, bool verbose_mode = true)
    : n_chains(chains), total_iter(total), warmup_iter(warmup),
      chain_progress(chains, 0), chain_divergent(chains, 0),
      verbose(verbose_mode) {
    start_time = std::chrono::steady_clock::now();
  }

  void update_chain(int chain_id, int iter, bool divergent) {
    chain_progress[chain_id] = iter;
    if (divergent) chain_divergent[chain_id]++;
  }

  void print_summary() {
    if (!verbose) return;

    auto end_time = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    int total_divergent = 0;
    for (int d : chain_divergent) total_divergent += d;

    int post_warmup_total = n_chains * (total_iter - warmup_iter);

    Rcpp::Rcout << "\n=== Sampling complete ===" << std::endl;
    Rcpp::Rcout << "Total time: " << ProgressReporter::format_duration(total_seconds) << std::endl;
    Rcpp::Rcout << "Chains: " << n_chains << " x " << (total_iter - warmup_iter)
                << " post-warmup samples" << std::endl;

    if (total_divergent > 0) {
      double div_pct = 100.0 * total_divergent / post_warmup_total;
      Rcpp::Rcout << "Divergent transitions: " << total_divergent;
      if (div_pct > 0.1) {
        Rcpp::Rcout << " (" << std::fixed << std::setprecision(1) << div_pct << "%)";
      }
      Rcpp::Rcout << std::endl;

      if (div_pct > 10.0) {
        Rcpp::Rcout << "WARNING: High divergence rate. Consider reparameterization "
                    << "or increasing adapt_delta." << std::endl;
      }
    }
  }
};

}  // namespace tulpa_progress

#endif  // QUOTR_HMC_PROGRESS_H
