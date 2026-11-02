"""Check Figure3's actual PDF image page against its complete caption."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[3]
IMAGE=ROOT/'content/posts/ai/pion-muon-vla-rlvr/assets/spectral-polynomials-v2.png'
TOKENS=['由论文公式(3)、(7)、(8)直接计算','列出全部六种调度','这四个面板不表示训练成功率']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    data=IMAGE.read_bytes()
    if data[:8]!=b'\x89PNG\r\n\x1a\n' or data[12:16]!=b'IHDR':
        raise ValueError('Expected the source PNG header')
    width,height=struct.unpack('>II',data[16:24]);ratio=width/height
    rows=[]
    for pdf in sorted(args.directory.glob('pion-*.pdf')):
        listing=subprocess.check_output(['pdfimages','-list',str(pdf)],text=True,timeout=20)
        image_pages=set()
        for line in listing.splitlines():
            fields=line.split()
            if len(fields)>=8 and fields[0].isdigit() and fields[2]=='image' and fields[5]in['icc','rgb']:
                w,h=map(int,fields[3:5])
                if w>=320 and h>=240 and abs(w/h-ratio)<.005:
                    image_pages.add(int(fields[0]))
        pages=[''.join(p.split()).replace('\u200b','')for p in pdf.with_suffix('.txt').read_text(encoding='utf-8').split('\f')]
        caption_pages=[[n+1for n,text in enumerate(pages)if token in text]for token in TOKENS]
        shared=set(image_pages)
        for found in caption_pages:shared.intersection_update(found)
        rows.append({'pdf':pdf.name,'image_pages':sorted(image_pages),'caption_token_pages':caption_pages,
                     'together_pages':sorted(shared),'passed':bool(shared)})
    if not rows:raise ValueError('No native PDF exports found')
    record={'source_image_sha256':hashlib.sha256(data).hexdigest(),'source_pixels':[width,height],
            'cases':len(rows),'failures':sum(not r['passed']for r in rows),'rows':rows,
            'scope':'Published Figure3 aspect in colored PDF image objects and full caption bounds on actual text pages; not a general image-quality audit.'}
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cases':record['cases'],'failures':record['failures']}))
    if args.check and record['failures']:raise SystemExit('Figure/caption pagination failed')

if __name__=='__main__':main()
