#ifndef TULPA_TEST_PRIOR_MODEL_H
#define TULPA_TEST_PRIOR_MODEL_H
#include "hmc_sampler.h"
#include <tulpa/likelihood.h>
namespace tulpa_probe {
template<class T>
T zero_observation(int,const T*,const T&,const T&,const std::vector<T>&,
    const tulpa::ModelData&,const tulpa::ParamLayout&,const void*) { return T(0); }
inline void gaussian_prior(tulpa::ModelData& data,tulpa::LikelihoodSpec& spec,int n) {
    spec.n_processes=1;spec.ll_double=zero_observation<double>;
    spec.ll_arena=zero_observation<tulpa::arena::Var>;
    data.n_processes=1;data.processes.resize(1);data.processes[0].p=n;
    data.N=0;data.sigma_beta=1;data.likelihood_spec=&spec;data.sharing.init(1);
}
}
#endif
