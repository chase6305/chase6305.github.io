"""Check the two-response GRPO example and its final value share a PDF page."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();e=json.loads((a.directory/'exports.json').read_text());rows=[]
for item in e['pdfs']:
 pdf=Path(item['file']);parts=[''.join(s.split())for s in pdf.with_suffix('.txt').read_text().split('\f')];joined=''.join(parts);start='例如p=0.5、两条回答为';assert joined.count(start)==1,(pdf.name,joined.count(start));first=joined.index(start);last=joined.index('=0.5',first+len(start))+len('=0.5')-1;offset=0;locations=[]
 for index,part in enumerate(parts,1):
  if offset<=first<offset+len(part):locations.append(index)
  if offset<=last<offset+len(part):locations.append(index)
  offset+=len(part)
 assert len(locations)==2;rows.append({'pdf':pdf.name,'example_start_page':locations[0],'final_value_page':locations[1],'same_page':locations[0]==locations[1]})
failures=[r for r in rows if not r['same_page']];a.output.write_text(json.dumps({'scope':__doc__,'cases':len(rows),'failures':failures,'rows':rows},indent=2)+'\n');print(json.dumps({'cases':len(rows),'split_examples':len(failures)}))
if a.check and failures:raise SystemExit(1)
