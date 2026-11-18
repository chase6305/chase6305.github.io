#!/usr/bin/env python3
"""Render selected native PDF pages for visual review; requires Pillow/Poppler."""
import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
normal=json.loads((args.root/'final-native/results.json').read_text())['pdfs']
forced=json.loads((args.root/'final-native-forced/results.json').read_text())['pdfs']


def pick(rows,size,theme='light'):
    return next(r for r in rows if r['paper']=='A4' and r['scripts'] and r['size']==size and r['theme']==theme)


def find_page(row,needle):
    text=subprocess.check_output(['pdftotext','-layout',row['file'],'-'],text=True)
    return next(i for i,p in enumerate(text.split('\f'),1) if needle in ''.join(p.split()))


small=pick(normal,100);large=pick(normal,200);forced_large=pick(forced,200,'dark')
cases=[
    ('runtime-overview',small,find_page(small,'局部重试不经过一次新的模型决策')),
    ('paired-outcomes',small,find_page(small,'配对结果绘制')),
    ('large-result-card',large,large['pagination']['cards'][-1]['locations'][0]['page']),
    ('large-pooled-equation',large,find_page(large,'275+149+192+189+256')),
    ('large-advantages',large,find_page(large,'1.291')),
    ('forced-large-result-card',forced_large,forced_large['pagination']['cards'][-1]['locations'][0]['page']),
]
rows=[]
for label,row,page in cases:
    target=args.output/label
    subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-singlefile','-scale-to','1400','-png',row['file'],str(target)],check=True,timeout=60)
    image=target.with_suffix('.png')
    with Image.open(image) as img:
        rgb=img.convert('RGB');w,h=rgb.size
        edges=[rgb.getpixel(p) for p in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]]
        assert all(min(pixel)>=250 for pixel in edges),(label,edges)
    rows.append({'label':label,'pdf':row['file'],'page':page,'image':str(image),'paper_corners':edges})
(args.output/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False))
