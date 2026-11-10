"""Compile and time the shipped precision header against a Git baseline."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--baseline', default='4e2b1d86')
parser.add_argument('--mac', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
work = args.output.resolve()
work.mkdir(parents=True, exist_ok=True)
main = (root/'tools/benchmark-precision.cpp').read_text()
compiler = shutil.which(os.environ.get('CXX', 'g++')) or shutil.which('clang++')
for name in ['before', 'after']:
    header = subprocess.check_output(['git', 'show', args.baseline+':inst/include/tulpa/precision_math.h'],
             cwd=root, text=True) if name == 'before' else (root/'inst/include/tulpa/precision_math.h').read_text()
    if name == 'after':
        header = header.replace('#include "precision_coefficients.h"',
                                (root/'inst/include/tulpa/precision_coefficients.h').read_text())
    source = work/f'{name}.cpp'
    source.write_text(main.replace('#include <tulpa/precision_math.h>', header))
    binary = work/(name+('.exe' if os.name == 'nt' else ''))
    flags = ['-static-libgcc', '-static-libstdc++'] if os.name == 'nt' else []
    subprocess.run([compiler, '-std=c++17', '-O2', '-ffp-contract=off', *flags,
                    str(source), '-o', str(binary)], check=True)
    with (work/f'windows-{name}.log').open('w') as output:
        subprocess.run([str(binary)], stdout=output, check=True)
    print('finished native', name, flush=True)

if args.mac:
    remote = '/tmp/tulpa-r19b-precision-20261110'
    subprocess.run(['ssh', '-o', 'BatchMode=yes', 'mac', f'mkdir -p {remote}'], check=True)
    subprocess.run(['scp', str(work/'before.cpp'), str(work/'after.cpp'), f'mac:{remote}/'], check=True)
    for name in ['before', 'after']:
        for contraction in ['off', 'on']:
            flag = '-ffp-contract=off' if contraction == 'off' else ''
            binary = f'bench-{name}-{contraction}'
            command = (f'cd {remote} && clang++ -std=c++17 -O2 {flag} {name}.cpp -o {binary}'
                       f' && ./{binary} > {binary}.log')
            subprocess.run(['ssh', 'mac', command], check=True)
            subprocess.run(['scp', f'mac:{remote}/{binary}.log', str(work/f'mac-{name}-{contraction}.log')], check=True)
            print('finished Mac', name, contraction, flush=True)
