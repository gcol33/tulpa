#ifndef TULPA_ENGINE_DEVICE_LIKELIHOOD_H
#define TULPA_ENGINE_DEVICE_LIKELIHOOD_H

#include "device.h"
#include "tulpa/likelihood.h"
#include <utility>
#include <stdexcept>

namespace tulpa { namespace device {

// The observation plan and kernel of one fit's device likelihood. The plan is
// validated here; the backend route holds the device copies.
class LikelihoodWorkspace {
    ObservationPlan plan_;
    DeviceLikelihood kernel_;
    std::unique_ptr<LikelihoodRoute> route_;
public:
    LikelihoodWorkspace(ObservationPlan plan, DeviceLikelihood kernel);
    ~LikelihoodWorkspace();
    const ObservationPlan& plan() const { return plan_; }
    DeviceLikelihood& kernel() { return kernel_; }
    Status refresh_projection(ObservationPlan plan);
    Status evaluate(const std::vector<double>& parameters, int chains,
        bool batched, bool weights, LikelihoodResult& result);
};

bool builtin_device_likelihood(const ModelData& data, const ParamLayout& layout,
    const void* response, bool initialize, DeviceLikelihood& kernel);

bool select_likelihood(const ModelData& data, const ParamLayout& layout,
    DeviceLikelihood& kernel, Policy& policy, Status& status, bool weights=false);

} }
#endif
