// gpu_backend.cpp
// R interface for GPU support functions

#include <Rcpp.h>
#include <string>
#include <vector>
#include <algorithm>
#include "gpu_backend.h"

using namespace Rcpp;

// [[Rcpp::export]]
bool cpp_gpu_available() {
  return tulpa_gpu::gpu_available();
}

// Which batched-CUDA implementation this build linked: "cuda" (the real
// dynamic-loading one, the default) or "stub" (TULPA_DISABLE_CUDA).
//
// Distinct from cpp_gpu_available(), which asks whether a usable GPU is present
// at RUN time. This asks what was COMPILED, and it is exported because the two
// used to be indistinguishable: gpu_backend.h and gpu_nngp_laplace.h defined
// the same `inline` functions differently, so which one every caller got was a
// link-order accident that nothing could report (gcol33/tulpa#396).
//
// This translation unit reaches it through gpu_backend.h and the NNGP kernels
// reach it through gpu_cuda.h, so a test asserting they AGREE is what pins the
// single definition.
// [[Rcpp::export]]
std::string cpp_gpu_backend_kind() {
  return std::string(tulpa_gpu::cuda_backend_kind());
}

// [[Rcpp::export]]
List cpp_gpu_info() {
  tulpa_gpu::GPUInfo info = tulpa_gpu::get_gpu_info();

  List devices;
  for (const auto& dev : info.devices) {
    devices.push_back(List::create(
      Named("name") = dev.name,
      Named("memory_mb") = dev.memory_mb,
      Named("compute_capability") = dev.compute_capability,
      Named("device_id") = dev.device_id
    ));
  }

  List result = List::create(
    Named("available") = info.available,
    Named("backend") = info.backend,
    Named("device_count") = info.device_count,
    Named("devices") = devices
  );

  result.attr("class") = "tulpa_gpu_info";
  return result;
}

// Drive gpu_batched_cholesky_solve() on one batch and report both the result
// and whether the GPU produced it.
//
// The composition chains a batched Cholesky into two triangular solves, and the
// factor travels between the three steps in a buffer whose layout no other
// entry point observes: cuSOLVER writes it column-major, the CPU consumers read
// it row-major, and cuBLAS reads it column-major again. Getting that wrong
// returns true and produces finite, plausible numbers, so the only thing that
// sees it is scoring alpha against an independent solve of the same system --
// which is what this exists for.
//
// C_flat: batch_size x (k*k) SPD matrices, one per row, flattened.
// c_rhs:  batch_size x k right-hand sides.
// Returns used_gpu (false when no device ran it, alpha then all NA) and the
// batch_size x k solution alpha = C^-1 c.
// [[Rcpp::export]]
List cpp_gpu_batched_cholesky_solve(NumericMatrix C_flat, NumericMatrix c_rhs,
                                    int k) {
  const int batch_size = C_flat.nrow();
  if (k <= 0) stop("k must be positive");
  if (C_flat.ncol() != k * k) stop("C_flat must have k * k columns");
  if (c_rhs.nrow() != batch_size || c_rhs.ncol() != k) {
    stop("c_rhs must be batch_size x k");
  }

  std::vector<std::vector<double>> C_batch(batch_size,
                                           std::vector<double>(k * k));
  std::vector<std::vector<double>> c_batch(batch_size, std::vector<double>(k));
  for (int b = 0; b < batch_size; b++) {
    for (int j = 0; j < k * k; j++) C_batch[b][j] = C_flat(b, j);
    for (int j = 0; j < k; j++) c_batch[b][j] = c_rhs(b, j);
  }

  std::vector<std::vector<double>> alpha_batch;
  const bool used_gpu = tulpa_gpu::gpu_batched_cholesky_solve(
      C_batch, c_batch, alpha_batch, k);

  NumericMatrix alpha(batch_size, k);
  if (used_gpu) {
    for (int b = 0; b < batch_size; b++) {
      for (int j = 0; j < k; j++) alpha(b, j) = alpha_batch[b][j];
    }
  } else {
    std::fill(alpha.begin(), alpha.end(), NA_REAL);
  }

  return List::create(Named("used_gpu") = used_gpu,
                      Named("alpha") = alpha);
}
