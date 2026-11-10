#ifndef TULPA_FIELD_SETUP_API_H
#define TULPA_FIELD_SETUP_API_H

#include <tulpa/nuts_api.h>

namespace tulpa {

enum class FieldSetup { MULTISCALE_TEMPORAL, SVC_HSGP, ST_HSGP, TEMPORAL_RESTRICTION };
using FieldSetupFn = void (*)(ModelData*, FieldSetup);

inline void prepare_field(ModelData& data, FieldSetup setup) {
    check_abi_version();
    const auto fn = reinterpret_cast<FieldSetupFn>(
        R_GetCCallable("tulpa", "tulpa_prepare_field"));
    fn(&data, setup);
}

}
#endif
