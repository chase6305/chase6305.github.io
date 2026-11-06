#!/usr/bin/env python3
"""Separate PDF font-metric overshoot from painted text at paper boundaries.

Poppler word boxes on Chrome Type-3 Chinese headings can extend above a page
while the actual glyphs remain inset. Report those boxes, render their pages,
and inspect only the affected word columns at the corresponding paper edge.
This is an additional raster diagnostic, not a proof of all glyph semantics.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as E
from PIL import Image

NS={'x':'http://www.w3.org/1999/xhtml'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report={'pdfs':0,'pages':0,'horizontal_outside':[],'vertical_metric_pages':[],
            'paint_at_affected_edges':[], 'scope':'Font boxes and raster paper-edge checks, not PDF/UA certification or semantic glyph verification.'}
    with tempfile.TemporaryDirectory(prefix='chase-pdf-edge-') as temporary:
        for directory in args.directories:
            for file in sorted(directory.glob('*.pdf')):
                report['pdfs']+=1
                root=E.fromstring(subprocess.check_output(['pdftotext','-bbox',str(file),'-']))
                pages=root.findall('.//x:page',NS)
                for number,page in enumerate(pages,1):
                    report['pages']+=1
                    width,height=float(page.attrib['width']),float(page.attrib['height'])
                    affected=[]
                    for word in page.findall('.//x:word',NS):
                        a={key:float(value) for key,value in word.attrib.items()}
                        if a['xMin']<-.5 or a['xMax']>width+.5:
                            report['horizontal_outside'].append({'file':str(file),'page':number,'word':word.text,'bounds':a})
                        if a['yMin']<-.5 or a['yMax']>height+.5:affected.append({'word':word.text,**a})
                    if not affected:continue
                    prefix=Path(temporary)/'page'
                    subprocess.run(['pdftoppm','-f',str(number),'-l',str(number),'-r','144','-png','-singlefile',str(file),str(prefix)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    picture=Image.open(prefix.with_suffix('.png')).convert('RGB')
                    counts=[]
                    for word in affected:
                        left=max(0,int(word['xMin']/width*picture.width))
                        right=min(picture.width,int(word['xMax']/width*picture.width)+1)
                        edge='top' if word['yMin']<0 else 'bottom'
                        y0,y1=(0,2) if edge=='top' else (picture.height-2,picture.height)
                        pixels=picture.crop((left,y0,right,y1))
                        data=pixels.load()
                        ink=sum(min(data[x,y])<240 for y in range(pixels.height) for x in range(pixels.width))
                        counts.append({'word':word['word'],'edge':edge,'nonwhite_edge_pixels':ink})
                        if ink:report['paint_at_affected_edges'].append({'file':str(file),'page':number,**counts[-1]})
                    report['vertical_metric_pages'].append({'file':str(file),'page':number,'boxes':affected,'raster_edge_checks':counts})
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pdfs':report['pdfs'],'pages':report['pages'],
                      'horizontal_outside':len(report['horizontal_outside']),
                      'vertical_metric_pages':len(report['vertical_metric_pages']),
                      'affected_edges_with_ink':len(report['paint_at_affected_edges'])}))
    if report['horizontal_outside'] or report['paint_at_affected_edges']:raise SystemExit(1)


if __name__=='__main__':main()
