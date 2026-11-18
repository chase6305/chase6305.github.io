#!/usr/bin/env python3
"""Check source rejection and exclusive output of the optional runtime probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
PROBE=ROOT/'content/posts/ai/pyrua-lean/probe_runtime.py'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    def call(*extra):
        return subprocess.run([sys.executable,'-B',str(PROBE),*map(str,extra)],
                              text=True,capture_output=True,timeout=30)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='pyrua-probe-contracts-') as temporary:
        work=Path(temporary)
        result=work/'result.json'
        valid=call('--source',source,'--output',result)
        assert valid.returncode==0,valid.stderr
        data=json.loads(result.read_text())
        assert data['show_then_move_attaches_end_of_cell_frame']
        checks.append('verified source, no third-party imports, recording behavior')
        digest=hashlib.sha256(result.read_bytes()).hexdigest()
        existing=call('--source',source,'--output',result)
        assert existing.returncode==2 and 'already exists' in existing.stderr
        assert hashlib.sha256(result.read_bytes()).hexdigest()==digest
        checks.append('existing output rejected and preserved byte for byte')
        mismatch=call('--source',ROOT,'--output',work/'wrong-commit.json')
        assert mismatch.returncode==2 and 'Expected official source commit' in mismatch.stderr
        assert not (work/'wrong-commit.json').exists()
        checks.append('wrong commit rejected before source execution or output creation')
        fixture=work/'modified-source'
        package=fixture/'src/pyrualean'
        package.mkdir(parents=True)
        gitdir=subprocess.check_output(['git','-C',str(source),'rev-parse','--absolute-git-dir'],text=True).strip()
        (fixture/'.git').write_text('gitdir: '+gitdir+'\n')
        for name in data['verified_modules']:
            shutil.copyfile(source/'src/pyrualean'/(name+'.py'),package/(name+'.py'))
        target=package/'arms.py'
        target.write_text(target.read_text()+'\nraise RuntimeError("tampered source executed")\n')
        modified=call('--source',fixture,'--output',work/'modified.json')
        assert modified.returncode==2 and 'Source SHA-256 mismatch: arms.py' in modified.stderr
        assert 'tampered source executed' not in modified.stderr
        assert not (work/'modified.json').exists()
        checks.append('modified source rejected before import or output creation')
        assert not list(source.rglob('*.pyc')), 'Probe added source bytecode'
        checks.append('official source kept free of Python bytecode')
    report={'checks':checks,'passed':len(checks),'source':str(source)}
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
