"""Check fixed new numerical rows on actual native PDF text lines."""
import argparse,json
from pathlib import Path

ROWS={
 'copy1':'1 4 1.000000 1','copy2':'2 8 1.142857 1',
 'copy8':'8 32 1.225806 1','copy32':'32 128 1.244094 1',
 'grpo001':'0.01 0.077255 0.083432 309.959421 12.944095',
 'grpo010':'0.10 0.569533 1.258836 45.535177 1.755825',
 'grpo050':'0.50 0.992188 60.901986 117.937799 1.007874',
 'ema1':'1 (0.5, 0, 0) 1','ema2':'2 (0.25, 0.5, 0) 2',
 'ema3':'3 (0.125, 0.25, 0.5) 3',
 'hidden':'隐藏矩阵 W 8×8 8','head':'标量输出头 1×8 1',
}
def compact(text):return ''.join(text.split()).replace('\u200b','')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args();rows=[]
 files=sorted(args.directory.glob('pion-*.txt'))
 if not files:raise SystemExit('No exported native PDF text found')
 for file in files:
  pages=[page.splitlines()for page in file.read_text(encoding='utf-8').split('\f')]
  for name,text in ROWS.items():
   matches=[{'page':n,'line':line}for n,page in enumerate(pages,1)for line in page if compact(line)==compact(text)]
   rows.append({'pdf':file.with_suffix('.pdf').name,'row':name,'expected':text,'matches':matches,'passed':len(matches)==1})
 report={'pdfs':len(files),'cases':len(rows),'failures':sum(not r['passed']for r in rows),'rows':rows,'scope':'Twelve fixed new rows: label/values must occur together on one actual PDF text line. This is not a complete table, typography or visual audit.'};args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:report[k]for k in ['pdfs','cases','failures']}))
 if args.check and report['failures']:raise SystemExit('A numerical row was lost, duplicated or split; inspect the actual PDF')
if __name__=='__main__':main()
