"""Build/check the deterministic downloadable Pion arithmetic example."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'content/posts/ai/pion-muon-vla-rlvr'
ARCHIVE=BUNDLE/'pion-spectral-lab.zip'
NAMES=('README.txt','pion_spectral.py','pion_matrix.py','make_figures.py',
       'probe_official_pion.py','pion_alignment.py','pion_step.py','probe_official_steps.py',
       'probe_official_vla_buffers.py','assets/vla-buffer-reference.json',
       'probe_official_vla_padding.py','assets/vla-padding-reference.json',
       'flat_spectrum.py','make_rank_figures.py','pion_snr.py','assets/snr-reference.json',
       'snr_correlated_groups.py','assets/snr-correlated-groups-reference.json',
       'grpo_group_signal.py','assets/grpo-group-signal-reference.json',
       'vla_gradient_rank.py','assets/vla-gradient-rank-reference.json',
       'assets/flat-spectrum-reference.json','assets/flat-spectrum-bf16-reference.json',
       'assets/step-reference.json','assets/official-steps.json','assets/spectral-reference.json',
       'assets/alignment-reference.json','assets/trace-reference.json',
       'assets/per-head-reference.json','assets/official-precision.json','check_examples.py')


def source(name):
    return ROOT/'scripts/test_pion_examples.py' if name=='check_examples.py' else BUNDLE/name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--numpy',action='store_true')
    args=parser.parse_args()
    if args.check:
        with zipfile.ZipFile(ARCHIVE) as archive:
            if archive.namelist()!=list(NAMES):raise SystemExit('Pion ZIP member list is stale')
            for name in NAMES:
                if archive.read(name)!=source(name).read_bytes():raise SystemExit('Stale Pion ZIP member: '+name)
    else:
        with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name in NAMES:
                info=zipfile.ZipInfo(name,date_time=(2026,10,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                info.create_system=3;info.external_attr=0o100644<<16;archive.writestr(info,source(name).read_bytes())
    if args.test:
        with tempfile.TemporaryDirectory(prefix='pion-lab-') as parent:
            folder=Path(parent,'download')
            folder.mkdir()
            # Mimic unpacking inside a checkout. The tests must select their
            # downloaded siblings rather than silently importing repo sources.
            decoy=Path(parent,'content/posts/ai/pion-muon-vla-rlvr/pion_spectral.py')
            decoy.parent.mkdir(parents=True)
            decoy.write_text('raise RuntimeError("Downloaded checks selected the parent repository")\n')
            with zipfile.ZipFile(ARCHIVE) as archive:archive.extractall(folder)
            subprocess.run([sys.executable,'-B','check_examples.py']+(['--numpy'] if args.numpy else []),cwd=folder,check=True,timeout=30)
            actual=json.loads(subprocess.check_output([sys.executable,'-B','pion_spectral.py'],cwd=folder,timeout=30))
            expected=json.loads(Path(folder,'assets/spectral-reference.json').read_text())
            if actual!=expected:raise SystemExit('Saved Pion scalar reference changed')
            actual=json.loads(subprocess.check_output([sys.executable,'-B','pion_alignment.py'],cwd=folder,timeout=30))
            expected=json.loads(Path(folder,'assets/alignment-reference.json').read_text())
            if actual!=expected:raise SystemExit('Saved Pion signal-alignment reference changed')
            actual=json.loads(subprocess.check_output([sys.executable,'-B','flat_spectrum.py'],cwd=folder,timeout=30))
            expected=json.loads(Path(folder,'assets/flat-spectrum-reference.json').read_text())
            if actual!=expected:raise SystemExit('Saved Pion flat-spectrum reference changed')
            actual=json.loads(subprocess.check_output([sys.executable,'-B','pion_snr.py'],cwd=folder,timeout=30))
            expected=json.loads(Path(folder,'assets/snr-reference.json').read_text())
            if actual!=expected:raise SystemExit('Saved Pion SNR reference changed')
            actual=json.loads(subprocess.check_output([sys.executable,'-B','snr_correlated_groups.py'],cwd=folder,timeout=30))
            expected=json.loads(Path(folder,'assets/snr-correlated-groups-reference.json').read_text())
            if actual!=expected:raise SystemExit('Saved Pion correlated-group reference changed')
            actual=json.loads(subprocess.check_output([sys.executable,'-B','grpo_group_signal.py'],cwd=folder,timeout=30))
            expected=json.loads(Path(folder,'assets/grpo-group-signal-reference.json').read_text())
            if actual!=expected:raise SystemExit('Saved GRPO group-signal reference changed')
            actual=json.loads(subprocess.check_output([sys.executable,'-B','vla_gradient_rank.py'],cwd=folder,timeout=30))
            expected=json.loads(Path(folder,'assets/vla-gradient-rank-reference.json').read_text())
            if actual!=expected:raise SystemExit('Saved VLA gradient-rank reference changed')
            expected=json.loads(Path(folder,'assets/trace-reference.json').read_text())
            for kp in (1,2):
                actual=json.loads(subprocess.check_output([sys.executable,'-B','pion_spectral.py','--sigma','.35','--kp',str(kp),'--trace'],cwd=folder,timeout=30))
                if actual!=expected[str(kp)]:raise SystemExit('Saved Pion iteration trace changed')
            if args.numpy:
                expected=json.loads(Path(folder,'assets/step-reference.json').read_text())
                for policy in ['skip','decay']:
                    actual=json.loads(subprocess.check_output([sys.executable,'-B','pion_step.py','--missing-gradient',policy],cwd=folder,timeout=30))
                    if actual.keys()!=expected[policy].keys() or actual['settings']!=expected[policy]['settings']:
                        raise SystemExit('Saved Pion step settings changed: '+policy)
                    # BLAS order can change insignificant Float64 tails.
                    def compare(value,reference):
                        if isinstance(reference,dict):
                            return value.keys()==reference.keys() and all(compare(value[k],v)for k,v in reference.items())
                        if isinstance(reference,list):
                            return len(value)==len(reference) and all(compare(a,b)for a,b in zip(value,reference))
                        if isinstance(reference,float):
                            return abs(value-reference)<=max(5e-324,abs(reference)*1e-10)
                        return value==reference
                    if not compare(actual,expected[policy]):raise SystemExit('Saved Pion step reference changed: '+policy)
                actual=json.loads(subprocess.check_output([sys.executable,'-B','pion_matrix.py','--head-report'],cwd=folder,timeout=30))
                expected=json.loads(Path(folder,'assets/per-head-reference.json').read_text())
                for key in ['input_diagonal','whole_diagonal','per_head_diagonal','input_block_norms','whole_block_norms','per_head_block_norms']:
                    if len(actual[key])!=len(expected[key]):raise SystemExit('Saved Pion head reference has a different shape: '+key)
                    for value,reference in zip(actual[key],expected[key]):
                        if abs(value-reference)>max(1e-300,abs(reference)*1e-10):raise SystemExit('Saved Pion head reference changed: '+key)
            # An incomplete extraction must also fail locally, even if a parent
            # checkout offers a module with the same name.
            for name in ('pion_spectral.py','pion_alignment.py','pion_matrix.py','pion_step.py','flat_spectrum.py','pion_snr.py','snr_correlated_groups.py','grpo_group_signal.py','vla_gradient_rank.py'):
                local_source=Path(folder,name)
                saved_source=local_source.read_bytes()
                # Even a hostile PYTHONPATH must not satisfy a missing sibling.
                fallback=Path(parent,'fallback')
                fallback.mkdir(exist_ok=True)
                (fallback/name).write_text('raise RuntimeError("Downloaded checks imported a PYTHONPATH fallback")\n')
                try:
                    local_source.unlink()
                    missing=subprocess.run([sys.executable,'-B','check_examples.py'],cwd=folder,
                        env={**os.environ,'PYTHONPATH':str(fallback)},capture_output=True,text=True,timeout=30)
                    if missing.returncode==0 or 'Missing local Pion example source' not in missing.stderr:
                        raise SystemExit('Incomplete download did not reject its missing local source: '+name)
                    if 'Downloaded checks imported' in missing.stderr or 'Downloaded checks selected' in missing.stderr:
                        raise SystemExit('Incomplete download imported a fallback: '+name)
                finally:
                    local_source.write_bytes(saved_source)
    print('Pion archive matches source'+('; extracted checks and reference passed' if args.test else ''))


if __name__=='__main__':main()
