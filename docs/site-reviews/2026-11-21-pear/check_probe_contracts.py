#!/usr/bin/env python3
"""Exercise PEAR probe source rejection and exclusive-output preservation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
PROBE=ROOT/'content/posts/ai/pear/probe_pear_contracts.py'
REVISION='2a81b2eec95e6e6f19c3fc0a80a38aebb40a30e2'


def run(command, expected=0):
    result=subprocess.run(command,text=True,capture_output=True,timeout=45)
    assert result.returncode==expected, (command,result.returncode,result.stdout,result.stderr)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    checks=[]
    with tempfile.TemporaryDirectory(prefix='pear-source-contracts-') as temporary:
        tmp=Path(temporary)
        valid=tmp/'valid.json'
        command=[sys.executable,'-B',str(PROBE),'--source',str(source),'--output',str(valid)]
        result=run(command)
        report=json.loads(result.stdout)
        assert report['source_commit']==REVISION and report['gripper_sign_not_magnitude']
        assert len(report['verified_files'])==9
        assert report['data_manifests']['stage3_nominal_row_weights']=={'libero':2000,'robodojo':1968}
        assert report['data_manifests']['official_smoke']['rows']==52
        checks.append('valid pinned source and bounded recorders')
        original=valid.read_bytes()
        result=run(command,2)
        assert 'Output already exists' in result.stderr and valid.read_bytes()==original
        checks.append('existing output preserved')
        worktree=tmp/'worktree'
        run(['git','-C',str(source),'worktree','add','--detach',str(worktree),REVISION])
        try:
            protocol=worktree/'pear/agent_loop/protocol.py'
            original_source=protocol.read_bytes()
            protocol.write_bytes(original_source+b'\nraise RuntimeError("MUST_NOT_EXECUTE")\n')
            output=tmp/'tampered.json'
            result=run([sys.executable,'-B',str(PROBE),'--source',str(worktree),'--output',str(output)],2)
            assert 'Source hash mismatch' in result.stderr and not output.exists()
            assert 'Traceback' not in result.stderr
            checks.append('tampered source rejected before execution/output')
            protocol.write_bytes(original_source)
            manifest=worktree/'pear/envs/robodojo/configs/official_smoke.json'
            original_manifest=manifest.read_bytes()
            manifest.write_bytes(original_manifest+b'\n')
            output=tmp/'tampered-manifest.json'
            result=run([sys.executable,'-B',str(PROBE),'--source',str(worktree),'--output',str(output)],2)
            assert 'Source hash mismatch: pear/envs/robodojo/configs/official_smoke.json' in result.stderr
            assert 'Traceback' not in result.stderr and not output.exists()
            checks.append('tampered manifest rejected before execution/output')
            manifest.write_bytes(original_manifest)
            run(['git','-C',str(worktree),'-c','user.name=Probe Fixture','-c','user.email=probe@example.invalid',
                 'commit','--allow-empty','-m','probe wrong revision fixture'])
            output=tmp/'wrong-revision.json'
            result=run([sys.executable,'-B',str(PROBE),'--source',str(worktree),'--output',str(output)],2)
            assert 'Source commit mismatch' in result.stderr and not output.exists()
            checks.append('wrong revision rejected before source/output')
        finally:
            run(['git','-C',str(source),'worktree','remove','--force',str(worktree)])
        output=tmp/'missing.json'
        run([sys.executable,'-B',str(PROBE),'--source',str(tmp/'absent'),'--output',str(output)],2)
        assert not output.exists()
        checks.append('missing checkout leaves no output')
    assert not list((source/'pear/agent_loop').glob('__pycache__'))
    assert not list((ROOT/'content/posts/ai/pear').glob('__pycache__'))
    checks.append('no source or article bytecode emitted')
    result={'checks':checks,'passed':len(checks)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
