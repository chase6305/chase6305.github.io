#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,json,time,hashlib
parser=argparse.ArgumentParser(description='Serial browser/PDF checks for the three paper articles.')
parser.add_argument('--base',required=True)
parser.add_argument('--port',default='9337')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--axe',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[3]
review=root/'docs/site-reviews/2026-11-05-three-paper-threehour'
output=args.output;output.mkdir(parents=True,exist_ok=True)
base=args.base;port=args.port;axe=args.axe
items=[
 ('reading',root/'docs/site-reviews/2026-11-05-last-omnicalib/check_articles.cjs',[axe]),
 ('whitening',review/'check_whitening.cjs',[axe]),
 ('prior',review/'check_prior.cjs',[axe]),
 ('touch',review/'check_touch.cjs',[]),
 ('chapters',review/'check_chapters.cjs',['--all-posts']),
 ('overflow',review/'check_overflow.cjs',[]),
 ('math',root/'scripts/check_blog_math.cjs',[]),
 ('print-large',root/'docs/site-reviews/2026-11-05-last-omnicalib/check_print_geometry.cjs',[]),
 ('print-large-forced',root/'docs/site-reviews/2026-11-05-last-omnicalib/check_print_geometry.cjs',['--forced-colors']),
 ('native',review/'check_native_print.cjs',[]),
 ('lab-print-states',review/'check_lab_print_states.cjs',[]),
 ('print-canvas',review/'check_print_canvas.cjs',[]),
 ('print-position',review/'check_print_position.cjs',[]),
]
source_files=[]
for name in ['content/posts/ai/last-action-tokenizer','content/posts/calibration/omnicalib','content/posts/ai/machine-learning-for-robotics']:
 source_files.extend(p for p in (root/name).rglob('*') if p.is_file())
source_files.extend(root/name for name in ['assets/css/custom.css','assets/css/last-whitening.css','assets/js/last-whitening.js','assets/css/omnicalib-prior.css','assets/js/omnicalib-prior.js','layouts/_shortcodes/last-whitening.html','layouts/_shortcodes/omnicalib-prior.html','docs/blog-editorial-review.json','data/blog_topics.json'])
freeze={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)}
(output/'source-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
report={'base':base,'source_hashes':freeze,'checks':[]}
for name,script,extra in items:
 command=['node',str(script),base,port,str(output/name),*extra]
 print(json.dumps({'check':name,'status':'started'}),flush=True);start=time.monotonic()
 result=subprocess.run(command,cwd=root,text=True,capture_output=True)
 (output/(name+'.log')).write_text(result.stdout+result.stderr)
 row={'name':name,'exit_code':result.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'command':command,'log':str(output/(name+'.log'))}
 report['checks'].append(row);(output/'suite.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(row),flush=True)
 if result.returncode:
  print((result.stdout+result.stderr)[-4500:],flush=True);raise SystemExit(result.returncode)
changed=[str(p.relative_to(root)) for p in source_files if hashlib.sha256(p.read_bytes()).hexdigest()!=freeze[str(p.relative_to(root))]]
report['unchanged_sources']=not changed;report['changed_sources']=changed
(output/'suite.json').write_text(json.dumps(report,indent=2)+'\n')
assert not changed,changed
print(json.dumps({'checks':len(items),'all_passed':True,'unchanged_sources':True}),flush=True)
