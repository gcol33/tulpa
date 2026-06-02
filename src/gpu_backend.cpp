// gpu_backend.cpp
// R interface for GPU support functions

#include <Rcpp.h>
#include "gpu_backend.h"

using namespace Rcpp;

// [[Rcpp::export]]
bool cpp_gpu_available() {
  return tulpa_gpu::gpu_available();
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
