// gpu_backend.cpp
// R interface for GPU support functions

#include <Rcpp.h>
#include <string>
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
