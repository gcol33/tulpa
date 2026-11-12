#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <tulpa/model_point_api.h>
#include "hmc_sampler.h"
#include "hmc_field_coordinates.h"
#include "sampler_field_store.h"
#include "shim_guard.h"

namespace {

void model_point(const tulpa::ModelData* data, const tulpa::ParamLayout* layout,
                 const tulpa::ModelPointRequest* request) {
    TULPA_SHIM_GUARD_BEGIN
    if (!data || !layout || !request)
        Rcpp::stop("Model evaluation requires data, a layout and a request");
    if (request->n_params != layout->total_params)
        Rcpp::stop("Model position length must equal the parameter layout");
    const int n = request->n_params;
    if (n && !request->position) Rcpp::stop("Model position is null");
    std::vector<double> q(n);
    if (n) std::copy_n(request->position, n, q.begin());
    tulpa_hmc::FieldCoordinates coordinates;
    if (request->coordinate_anchor) {
        std::vector<double> anchor(request->coordinate_anchor, request->coordinate_anchor + n);
        coordinates = tulpa_hmc::make_field_coordinates(anchor, *data, *layout);
        if (request->sampling_position) {
            coordinates.inverse(anchor);
            std::copy(anchor.begin(), anchor.end(), request->sampling_position);
        }
    } else if (request->sampling_position && n) {
        std::copy(q.begin(), q.end(), request->sampling_position);
    }
    tulpa_hmc::FieldCoordinateScope coordinate_scope(request->coordinate_anchor ? &coordinates : nullptr);
    if (request->gradient) {
        std::vector<double> gradient;
        tulpa_hmc::compute_gradient(q, *data, *layout, gradient, request->value);
        std::copy(gradient.begin(), gradient.end(), request->gradient);
    } else if (request->value) {
        *request->value = tulpa_hmc::compute_log_post(q, *data, *layout);
    }
    if (request->prior || request->stored) {
        std::vector<double> native = q;
        const double jacobian = request->coordinate_anchor ? coordinates.forward(q, native) : 0.0;
        tulpa_hmc::FieldCoordinateScope raw(nullptr);
        if (request->prior)
            *request->prior = tulpa_hmc::compute_log_prior(native, *data, *layout) + jacobian;
        if (request->stored) {
            std::copy(native.begin(), native.end(), request->stored);
            tulpa::store_field_draw(request->stored, native, *data, *layout);
        }
    }
    TULPA_SHIM_GUARD_END("tulpa_model_point")
}

}

void tulpa_register_model_point(DllInfo*) {
    R_RegisterCCallable("tulpa", "tulpa_model_point", reinterpret_cast<DL_FUNC>(&model_point));
}
