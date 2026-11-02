"""Check selected shell commands and a short Python unit stay on one PDF page."""
import argparse,json,re
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();exports=json.loads((a.directory/'exports.json').read_text());rows=[]
 specs=[('flat','python-Bflat_spectrum.py',0,'>flat-bf16.json'),('buffers_gloo','python-Bprobe_official_vla_buffers.py',0,'--gloo'),('buffers_nccl','python-Bprobe_official_vla_buffers.py',1,'--gloo--nccl'),('padding_two','python-Bprobe_official_vla_padding.py',0,'--world-size2--outputpadding-two.json'),('padding_three','python-Bprobe_official_vla_padding.py',1,'--world-size3--outputpadding-three.json'),('candidate_python','#两个分支各替换一次',0,'g=torch.zeros_like(update_buffer_views[self.rank])')]
 for item in exports['pdfs']:
  pdf=Path(item['file']);pages=pdf.with_suffix('.txt').read_text().split('\f');parts=[''.join(page.split())for page in pages];joined=''.join(parts);bounds=[];offset=0
  for index,part in enumerate(parts,1):bounds.append((offset,offset+len(part),index));offset+=len(part)
  def page_at(pos):return next(index for lo,hi,index in bounds if lo<=pos<hi)
  for name,start,occurrence,end in specs:
   hits=[m.start()for m in re.finditer(re.escape(start),joined)];expected=1 if name in ('flat','candidate_python') else 2
   if len(hits)!=expected:raise ValueError((pdf.name,name,hits))
   first=hits[occurrence];last=joined.find(end,first)+len(end)-1
   if last<len(end)-1:raise ValueError((pdf.name,name,'end missing'))
   start_page,end_page=page_at(first),page_at(last);rows.append({'pdf':pdf.name,'command':name,'unit_type':'python' if name=='candidate_python' else 'shell','start_page':start_page,'end_page':end_page,'same_page':start_page==end_page})
 failures=[r for r in rows if not r['same_page']];record={'scope':__doc__,'pdfs':len(exports['pdfs']),'cases':len(rows),'shell_cases':sum(r['unit_type']=='shell' for r in rows),'python_cases':sum(r['unit_type']=='python' for r in rows),'failures':failures,'rows':rows};a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'pdfs':record['pdfs'],'cases':len(rows),'split_units':len(failures)}))
 if a.check and failures:raise SystemExit(1)
if __name__=='__main__':main()
