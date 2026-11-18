#!/usr/bin/env python3
"""Locate each printed label and its value by word geometry, not reading order."""
import json
import sys
import xml.etree.ElementTree as ET

cards=json.loads(sys.argv[1])
root=ET.fromstring(sys.stdin.buffer.read())
ns={'x':'http://www.w3.org/1999/xhtml'}
pages=[[dict(text=w.text or '',**{k:float(v) for k,v in w.attrib.items()})
        for w in p.findall('.//x:word',ns)] for p in root.findall('.//x:page',ns)]
compact=lambda text:''.join(text.split())


def matches(words,text):
    text=compact(text)
    def walk(group,remaining):
        if not remaining:
            yield {key:op(word[key] for word in group) for key,op in
                   [('xMin',min),('yMin',min),('xMax',max),('yMax',max)]}
            return
        if len(group)>24:return
        previous=group[-1] if group else None
        for word in words:
            part=compact(word['text'])
            if not part or not remaining.startswith(part) or word in group:continue
            if previous:
                center=lambda w:(w['yMin']+w['yMax'])/2
                height=max(word['yMax']-word['yMin'],previous['yMax']-previous['yMin'])
                dy=center(word)-center(previous)
                same=abs(dy)<height*.3 and -1<=word['xMin']-previous['xMax']<height*.8
                following=height*.3<=dy<height*1.8 and abs(word['xMin']-group[0]['xMin'])<5
                if not(same or following):continue
            yield from walk(group+[word],remaining[len(part):])
    yield from walk([],text)


results=[]
for card in cards:
    found=[]
    for number,words in enumerate(pages,1):
        for label in matches(words,card['label']):
            for value in matches(words,card['value']):
                if value['xMin']>label['xMin']+15 and label['yMin']-6<=value['yMin']<=label['yMax']+12:
                    found.append({'page':number,'label':label,'value':value})
    results.append({**card,'locations':found,'passed':bool(found)})
print(json.dumps({'pages':len(pages),'cards':results,'passed':all(r['passed'] for r in results)},ensure_ascii=False))
