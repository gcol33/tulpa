#include <tulpa/precision_math.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

volatile double precision_sink=0;
template<class F> double median_ns(F fn, int iterations) {
    std::vector<double> times;
    for(int rep=0;rep<9;++rep) {
        auto start=std::chrono::steady_clock::now();
        fn(iterations);
        times.push_back(std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/iterations);
    }
    std::sort(times.begin(),times.end());
    return times[times.size()/2];
}
int main() {
    std::cout<<std::setprecision(10);
    std::cout<<"long_double_digits "<<std::numeric_limits<long double>::digits<<"\n";
    for(double t : {.0001,.01,.015625,.03,.0625,.1,.3,.5,1.,2.,4.,8.,9.999,10.,
                    12.,14.,16.,18.,25.,41.,94.,100.,401.,8608.,1e8,1e9}) {
        auto original_start=std::chrono::steady_clock::now();
        for(int i=0;i<2000;++i) precision_sink+=tulpa::math::stirling_remainder(t+i*1e-8);
        std::cout<<"previous_ns_per_call "<<t<<" "<<std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-original_start).count()/2000<<"\n";
        double ns=median_ns([&](int n) {
            for(int i=0;i<n;++i) precision_sink+=tulpa::math::stirling_remainder(t+(i%2000)*1e-8);
        },200000);
        std::cout<<"ns_per_call "<<t<<" "<<ns<<"\n";
    }
    for(double t : {-0.7,-.1,1e-8,.1,1.,8.}) {
        double ns=median_ns([&](int n) {
            for(int i=0;i<n;++i) precision_sink+=tulpa::math::log1pmx(t+(i%2000)*1e-12);
        },200000);
        std::cout<<"log1pmx_ns_per_call "<<t<<" "<<ns<<"\n";
    }
    constexpr int nobs=100000;
    std::vector<double> mu(nobs),y(nobs),ly(nobs),lq(nobs);
    for(int i=0;i<nobs;++i) {
        mu[i]=.15+.7*(i%997)/996.;
        y[i]=mu[i]*(1+.09*std::sin(i*.01));
        ly[i]=std::log(y[i]); lq[i]=std::log1p(-y[i]);
    }
    if(mu.size()!=nobs || y.size()!=nobs) std::abort();
    std::cout<<"nobs "<<nobs<<"\n";
    for(double phi : {.3,8.,100.,1e8}) {
        for(bool beta : {false,true}) {
            double ns=median_ns([&](int n) {
                for(int rep=0;rep<n;++rep) {
                    tulpa::math::GammaPrecision cache(phi);
                    double sum=0;
                    for(int i=0;i<nobs;++i)
                        sum+=beta ? tulpa::math::beta_log_density(y[i],mu[i],1-mu[i],phi,ly[i],lq[i])
                                  : cache.log_density(y[i],mu[i],ly[i]);
                    precision_sink=sum;
                }
            },5);
            std::cout<<"likelihood_ms "<<(beta ? "beta" : "gamma")<<" "<<phi<<" "<<ns/1e6<<" "<<precision_sink<<"\n";
        }
    }
}
