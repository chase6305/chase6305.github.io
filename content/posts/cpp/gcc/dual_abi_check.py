"""Reproduce libstdc++ dual-ABI matching without changing system libraries.

Python standard library, Linux g++ and GNU nm. Builds temporary object files;
two deliberately mismatched links must fail. This is not a GLIBCXX version test.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++', help='Compiler executable, without flags')
    parser.add_argument('--output', type=Path, default=Path('dual-abi-results.json'))
    args = parser.parse_args()
    compiler = shutil.which(args.cxx)
    if compiler is None or shutil.which('nm') is None:
        parser.error('g++ and GNU nm must be available')
    environment = dict(os.environ, LC_ALL='C')

    def run(command, cwd, check=True):
        result = subprocess.run(command, cwd=cwd, env=environment,
                                text=True, capture_output=True, timeout=30)
        if check and result.returncode:
            raise RuntimeError(f'{command!r}\n{result.stdout}\n{result.stderr}')
        return result

    with tempfile.TemporaryDirectory(prefix='libstdcxx-dual-abi-') as directory:
        root = Path(directory)
        (root/'label.hpp').write_text('#pragma once\n#include <string>\nstd::string label();\n')
        (root/'library.cpp').write_text('#include "label.hpp"\nstd::string label() { return "robot"; }\n')
        (root/'main.cpp').write_text('#include "label.hpp"\n#include <iostream>\n'
                                   'int main() { auto value = label(); std::cout << value << "\\n"; '
                                   'return value == "robot" ? 0 : 1; }\n')
        symbols = {}
        for abi in (0, 1):
            run([compiler, '-std=c++11', '-Wall', '-Wextra',
                 f'-D_GLIBCXX_USE_CXX11_ABI={abi}', '-c', 'library.cpp',
                 '-o', f'library-{abi}.o'], root)
            output = run(['nm', '-C', '--defined-only', f'library-{abi}.o'], root).stdout
            names = [line.split(' T ', 1)[1] for line in output.splitlines() if ' T label' in line]
            assert names == (['label()'] if abi == 0 else ['label[abi:cxx11]()']), names
            symbols[str(abi)] = names[0]
        rows = []
        for library_abi, caller_abi in ((0, 0), (1, 1), (0, 1), (1, 0)):
            command = [compiler, '-std=c++17', '-Wall', '-Wextra',
                       f'-D_GLIBCXX_USE_CXX11_ABI={caller_abi}', 'main.cpp',
                       f'library-{library_abi}.o', '-o', 'app']
            linked = run(command, root, check=False)
            expected_success = library_abi == caller_abi
            assert (linked.returncode == 0) == expected_success, linked.stderr
            row = {'library_abi': library_abi, 'caller_abi': caller_abi,
                   'library_standard': 'c++11', 'caller_standard': 'c++17',
                   'link_success': linked.returncode == 0}
            if expected_success:
                output = run([str(root/'app')], root).stdout.strip()
                assert output == 'robot', output
                row['program_output'] = output
            else:
                errors = [line for line in linked.stderr.splitlines()
                          if 'undefined reference' in line and 'label' in line]
                assert errors, linked.stderr
                # Omit compiler-generated temporary object names from the report.
                row['missing_symbol'] = 'label[abi:cxx11]()' if caller_abi else 'label()'
            rows.append(row)
        version = run([compiler, '--version'], root).stdout.splitlines()[0]
    report = {'compiler': version, 'library_defined_symbols': symbols, 'cases': rows,
              'scope': 'One std::string-returning function across two translation units. '
                       'No shared-library replacement, GLIBCXX version failure, '
                       'cross-compiler compatibility claim or application ABI audit.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
