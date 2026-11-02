"""Check that the conditional-moment definitions share a native PDF page with the formula."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files = sorted(args.directory.glob('pion-*.pdf'))
    if len(files) != 5:
        raise SystemExit('Expected five native PDF exports')
    rows = []
    for file in files:
        text = subprocess.check_output(['pdftotext', '-layout', str(file), '-'],
                                       text=True, encoding='utf-8', timeout=30)
        pages = [''.join(page.split()) for page in text.split('\f')]
        definition = [i for i, page in enumerate(pages, 1) if '记保留组内' in page]
        formula = [i for i, page in enumerate(pages, 1) if 'Var(d)=' in page]
        rows.append({'pdf': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                     'definition_pages': definition, 'formula_pages': formula,
                     'passed': len(definition) == len(formula) == 1 and definition == formula})
    result = {'pdfs': len(files), 'cases': len(rows),
              'failures': sum(not row['passed'] for row in rows), 'rows': rows,
              'scope': 'Actual PDF text page positions; no inference from CSS rules.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('pdfs', 'cases', 'failures')}))
    if args.check and result['failures']:
        raise SystemExit('Conditional-moment definition is separated from its formula')


if __name__ == '__main__':
    main()
