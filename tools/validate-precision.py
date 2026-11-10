import csv
import ctypes
import json
import math
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import mpmath as mp

parser = argparse.ArgumentParser(description='Compare precision primitives with mpmath at binary64 inputs.')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--baseline', default='4e2b1d86')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
WORK = args.output.resolve()
WORK.mkdir(parents=True, exist_ok=True)
compiler = shutil.which(os.environ.get('CXX', 'g++')) or shutil.which('clang++')
if compiler is None:
    parser.error('A C++17 compiler is required (CXX, g++, or clang++).')
mp.mp.dps = 50
original = subprocess.check_output(['git', 'show', args.baseline+':inst/include/tulpa/precision_math.h'], cwd=ROOT, text=True)
current = (ROOT/'inst/include/tulpa/precision_math.h').read_text()
coefficients = (ROOT/'inst/include/tulpa/precision_coefficients.h').read_text()
current = current.replace('#include "precision_coefficients.h"', coefficients)
prefix = '''#include <cmath>
#include <limits>
static int perturb = 0, calls = 0;
inline long double test_lgamma(long double x) {
    ++calls;
    const long double exact = std::lgamma(x);
    if (perturb == 0) return exact;
    double y = static_cast<double>(exact);
    for (int i=0; i<std::abs(perturb); ++i)
        y = std::nextafter(y, perturb > 0 ? INFINITY : -INFINITY);
    return y;
}
'''
suffix = '''
extern "C" EXPORT double eval(double x, int k, int mode) {
    perturb = mode;
    return k < 0 ? tulpa::math::log1pmx(x) : tulpa::math::stirling_remainder_derivative(x,k);
}
extern "C" EXPORT int call_count() { return calls; }
'''
prefix = ('#define EXPORT __declspec(dllexport)\n' if os.name == 'nt' else '#define EXPORT\n')+prefix
libs = {}
for name, header in [('before', original), ('after', current), ('before_double', original), ('after_double', current)]:
    if name.endswith('_double'):
        header = header.replace('long double', 'double')
        import re
        header = re.sub(r'(?<=[0-9.])[lL]\b', '', header)
    header = header.replace('std::lgamma(', 'test_lgamma(')
    source = WORK/(name+'.cpp')
    source.write_text(prefix+header+suffix)
    extension = '.dll' if os.name == 'nt' else '.so'
    flags = ['-shared', '-static-libgcc', '-static-libstdc++'] if os.name == 'nt' else ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    binary = WORK/(name+extension)
    subprocess.run([compiler, '-std=c++17', '-O2', '-ffp-contract=off', *flags,
                    str(source), '-o', str(binary)], check=True)
    lib = ctypes.CDLL(str(binary))
    lib.eval.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_int]
    lib.eval.restype = ctypes.c_double
    libs[name] = lib

fixture = list(csv.DictReader((ROOT/'tests/testthat/fixtures/precision-primitives.csv').open()))
xs = set(10.**(-8+16*i/10000) for i in range(10001))
xs.update(i/2 for i in range(1,201))
xs.update(float(r['x']) for r in fixture if r['kind']=='s')
for t in [2.**e for e in range(-6,4)]+[.25, (math.sqrt(8)-1)/2, 4.5, 10.,
          138619474.,8608.,401.,94.,41.,25.,18.,14.,12.,
          240095971.,12872.,554.,124.,53.,31.,21.,17.,
          339546979.,16941.,698.,151.,63.,36.,19.,16.,
          438353265.,20937.,838.,177.,73.,28.,15.,13.,11.,
          536870912.,24899.,977.,203.,82.,46.,23.]:
    xs.update([math.nextafter(t, -math.inf),t,math.nextafter(t,math.inf)])
mxs = set(10.**(-8+16*i/10000) for i in range(10001))
mxs.update(-1+10.**(-16+16*i/5000) for i in range(5001))
mxs.update(-10.**(-8+8*i/5000) for i in range(5001))
mxs.update(float(r['x']) for r in fixture if r['kind']=='m')
for t in [-.75, -.125, .125, 2., 2/9, -2/11,
          2*math.sqrt(.125)/(1-math.sqrt(.125)),
          -2*math.sqrt(.125)/(1+math.sqrt(.125))]:
    mxs.update([math.nextafter(t,-math.inf),t,math.nextafter(t,math.inf)])
for y in [.00003,.005,.0625,.125,.25]:
    r=math.sqrt(y)
    for t in [2*r/(1-r),-2*r/(1+r)]:
        mxs.update([math.nextafter(t,-math.inf),t,math.nextafter(t,math.inf)])

def reference(x, k):
    z = mp.mpf(float(x))
    if k < 0:
        return z-mp.log1p(z)
    if x >= 1e10:
        return mp.fsum(mp.bernoulli(2*j)/(2*j*(2*j-1))*(-1)**k*mp.rf(2*j-1,k)*z**(-(2*j-1+k)) for j in range(1,15))
    if k == 0:
        return mp.loggamma(z)-((z-mp.mpf('.5'))*mp.log(z)-z+mp.log(2*mp.pi)/2)
    if k == 1:
        return mp.digamma(z)-mp.log(z)+1/(2*z)
    sign = (-1)**k
    main = sign*mp.factorial(k-2)*(z**(1-k)+mp.mpf('.5')*(k-1)*z**(-k))
    return mp.polygamma(k-1,z)-main

summary = {}
perturbation = {n: 0 for n in libs}
fixture_errors = []
started = time.time()
with (WORK/'grid.csv').open('w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['kind','order','x','reference','before_ulp','after_ulp','before_double_ulp','after_double_ulp'])
    for k, points in [(k,sorted(xs)) for k in range(5)]+[(-1,sorted(mxs))]:
        for x in points:
            ref = reference(x,k)
            rounded = float(ref)
            branch = ('s_small' if x < .015625 else 's_polynomial' if x < 10 else 's_asymptotic') if k>=0 else ('m_series' if -.75<=x<=2 else 'm_log1p')
            key = f'{branch}:order{k}'
            errs = []
            for name, lib in libs.items():
                actual = lib.eval(x,k,0)
                if math.isnan(actual):
                    raise AssertionError(f'{name}: NaN at {x=}, {k=}')
                err = float(abs(mp.mpf(actual)-ref)/math.ulp(rounded)) if math.isfinite(rounded) else 0 if actual==rounded else math.inf
                errs.append(err)
                record = summary.setdefault(name,{}).setdefault(key,{'max_ulp':-1, 'x':None, 'count':0})
                record['count'] += 1
                if err > record['max_ulp']:
                    record.update(max_ulp=err,x=x)
                for mode in [-4,4]:
                    shifted = lib.eval(x,k,mode)
                    delta = abs(shifted-actual)/math.ulp(actual) if math.isfinite(actual) else 0 if shifted==actual else math.inf
                    perturbation[name] = max(perturbation[name],delta)
            writer.writerow(['s' if k>=0 else 'm',k,repr(x),mp.nstr(ref,25),*errs])
        print('finished',k,'seconds',time.time()-started,flush=True)
for r in fixture:
    x,k = float(r['x']),int(r['order']) if r['kind']=='s' else -1
    ref = reference(x,k)
    expected = float(r['expected'])
    if float(ref)!=expected:
        fixture_errors.append(dict(r,correct=repr(float(ref))))
result = dict(summary=summary,perturbation_max_ulp=perturbation,lgamma_calls={n:l.call_count() for n,l in libs.items()},fixture_rounding_differences=fixture_errors,seconds=time.time()-started)
(WORK/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
for name in ['after', 'after_double']:
    assert all(v['max_ulp'] <= 2 for v in summary[name].values()), name
    assert perturbation[name] == 0 and libs[name].call_count() == 0, name
assert not fixture_errors, fixture_errors
