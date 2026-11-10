"""Generate binary64 Chebyshev interpolants and Stirling series coefficients."""
import argparse
from pathlib import Path
import mpmath as mp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] /
                    'inst/include/tulpa/precision_coefficients.h')
args = parser.parse_args()
mp.mp.dps = 80

def remainder(x, order):
    if order == 0:
        return mp.loggamma(x) - (x-mp.mpf('.5'))*mp.log(x) + x - mp.log(2*mp.pi)/2
    if order == 1:
        return mp.digamma(x) - mp.log(x) + 1/(2*x)
    return mp.polygamma(order-1, x) - (-1)**order*mp.factorial(order-2) * (
        x**(1-order) + mp.mpf('.5')*(order-1)*x**(-order))

def row(values, indent='    '):
    values = [float(x).hex() for x in values]
    return '{\n' + ',\n'.join(indent + ', '.join(values[i:i+4])
                              for i in range(0, len(values), 4)) + '\n}'

fits = []
for order in range(5):
    branches = []
    for exponent in range(-6, 4):
        a = mp.mpf(2)**exponent
        coefficients, error = mp.chebyfit(lambda t: remainder(a*(t+mp.mpf('1.5')), order) *
            (a*(t+mp.mpf('1.5')))**order, [-mp.mpf('.5'), mp.mpf('.5')], 22, error=True)
        branches.append(coefficients)
        print('order', order, 'interval', float(a), float(2*a),
              'estimated_absolute_error', mp.nstr(error, 8), flush=True)
    fits.append(branches)

small = [mp.chebyfit(lambda x: (-1)**k*(mp.factorial(k-1)/2-mp.factorial(k-2)*x) +
         x**k*mp.polygamma(k-1, 1+x), [0, mp.mpf(1)/16], 16) for k in range(2, 5)]
elementary = [mp.chebyfit(lambda x: mp.loggamma(1+x)/x, [0, mp.mpf(1)/16], 12),
              mp.chebyfit(lambda x: mp.digamma(1+x), [0, mp.mpf(1)/16], 12)]
for coefficients, radius in ([(c, mp.mpf('.5')) for branches in fits for c in branches] +
                             [(c, mp.mpf(1)/16) for c in small+elementary]):
    assert mp.fsum(abs(c)*radius**power for power, c in
                   enumerate(reversed(coefficients[:-1]), start=1)) < abs(coefficients[-1])
series = [[mp.bernoulli(2*j)/(2*j*(2*j-1)) * (-1)**k * mp.rf(2*j-1, k)
           for j in range(1, 15)] for k in range(5)]
cutoffs = [[mp.ceil((abs(series[k][n])/abs(series[k][0])*2**59)**(mp.mpf(1)/(2*n)))
            for n in range(1, 14)] for k in range(5)]
tail = lambda y: (mp.atanh(mp.sqrt(y))-mp.sqrt(y))/(y*mp.sqrt(y))
log_tails = [mp.chebyfit(lambda t: tail(t+center), [lo-center, hi-center], 16)
             for lo, hi, center in [(mp.mpf(0), mp.mpf(1)/8, mp.mpf(1)/16),
                                    (mp.mpf(1)/8, mp.mpf(9)/25, mp.mpf(1)/4)]]

text = '''#ifndef TULPA_PRECISION_COEFFICIENTS_H
#define TULPA_PRECISION_COEFFICIENTS_H

namespace tulpa { namespace math { namespace detail {

// Degree-21 Chebyshev interpolants of x^k s^(k)(x), in t=x/2^e-3/2,
// -1/2 <= t <= 1/2, e=-6,...,3. Coefficients are in descending powers.
static constexpr double stirling_coefficients[5][10][22] = {
'''
text += ',\n'.join('{\n' + ',\n'.join(row(c) for c in branches) + '\n}' for branches in fits) + '\n};\n\n'
for name, values, shape, comment in [
    ('stirling_small_coefficients', small, '[3][16]',
     '// x^k s^(k)(x), k=2,3,4, after removal of the pole at zero.\n'),
    ('stirling_elementary_coefficients', elementary, '[2][12]',
     '// log Gamma(1+x)/x and psi(1+x) on [0,1/16].\n'),
    ('stirling_series_coefficients', series, '[5][14]',
     '// (-1)^k (2n-1)_k B_(2n)/(2n(2n-1)), n=1,...,14.\n'),
    ('stirling_series_cutoffs', cutoffs, '[5][13]',
     '// First omitted term / leading term <= 2^-59 above each cutoff.\n'),
    ('log1pmx_tail_coefficients', log_tails, '[2][16]',
     '// S(y) on [0,1/8] and [1/8,9/25], centered at 1/16 and 1/4.\n')]:
    text += comment + f'static constexpr double {name}{shape} = {{\n'
    text += ',\n'.join(row(c) for c in values) + '\n};\n\n'
for name, values, shape in [
    ('stirling_constant_low', [[c[-1]-mp.mpf(float(c[-1])) for c in branches] for branches in fits], '[5][10]'),
    ('stirling_small_constant_low', [[c[-1]-mp.mpf(float(c[-1])) for c in small]], '[1][3]'),
    ('stirling_elementary_constant_low', [[c[-1]-mp.mpf(float(c[-1])) for c in elementary]], '[1][2]')]:
    text += f'static constexpr double {name}{shape} = {{\n'
    text += ',\n'.join(row(c) for c in values) + '\n};\n\n'
text += '// Coefficients of sum_(j>=0) y^j/(2j+3), 0 <= y <= 1/200.\n'
text += 'static constexpr double log1pmx_coefficients[8] = ' + row([mp.mpf(1)/(2*j+3) for j in range(8)]) + ';\n\n'
text += '} } }\n#endif\n'
args.output.write_text(text, encoding='utf-8')
