#!/usr/bin/env python3
"""Deterministic SVG data-flow diagrams; check emitted arrows and source sync."""
from pathlib import Path
import html, sys, xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'content/posts/ai/diffusion-models/assets'
class Diagram:
    def __init__(self,w,h,title):
        self.nodes={};self.edges=[]
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title"><title id="title">{html.escape(title)}</title><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8 Z" fill="#334155"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" fill="#172033" font-size="22">']
    def label(self,x,y,text,size=22):
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{html.escape(text)}</text>')
    def box(self,key,x,y,w,h,label,fill='#eaf2ff'):
        self.nodes[key]=(x,y,w,h)
        self.parts.append(f'<g id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#64748b" stroke-width="1.5"/>')
        self.label(x+w/2,y+h/2+8,label);self.parts.append('</g>')
    def arrow(self,a,b,points,label=None,label_at=None):
        self.edges.append((a,b))
        dashed = ' stroke-dasharray="6 4"' if a in {'time','embedding','random_t','sample_t'} else ''
        self.parts.append(f'<polyline{dashed} data-from="{a}" data-to="{b}" points="{points}" fill="none" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>')
        if label:self.label(*label_at,label,18)
    def finish(self):return '\n'.join(self.parts+['</g></svg>'])+'\n'

def unet():
    d=Diagram(1000,620,'Time-conditioned U-Net feature path and optional cross-attention')
    for args in [('xt',40,230,100,65,'x_t'),('encoder',210,230,180,65,'Encoder'),('middle',210,425,180,65,'Middle features'),('attention',530,425,180,65,'Cross-attention','#fff3d6'),('decoder',530,230,180,65,'Decoder','#f6ebff'),('head',790,230,170,65,'Prediction head'),('condition',530,535,180,55,'Condition c','#fff3d6'),('time',40,50,100,55,'t','#eaf7e8'),('embedding',300,50,220,55,'Time embedding','#eaf7e8')]:d.box(*args)
    for args in [('xt','encoder','140,262 210,262'),('encoder','middle','300,295 300,425'),('middle','attention','390,457 530,457','Q: features',(460,439)),('condition','attention','620,535 620,490','K / V',(675,521)),('attention','decoder','620,425 620,295'),('encoder','decoder','390,262 530,262','Skip',(460,245)),('decoder','head','710,262 790,262'),('time','embedding','140,77 300,77'),('embedding','encoder','355,105 355,165 300,165 300,230'),('embedding','middle','410,105 410,185 175,185 175,457 210,457'),('embedding','decoder','465,105 465,145 620,145 620,230')]:d.arrow(*args)
    d.label(875,330,'epsilon_theta',20)
    d.label(270,560,'Dashed: time inputs',20)
    return d

def training():
    d=Diagram(1100,860,'DDPM noise-prediction training and reverse sampling')
    d.parts.append('<rect x="20" y="20" width="510" height="815" rx="18" fill="none" stroke="#94a3b8"/><rect x="560" y="20" width="520" height="815" rx="18" fill="none" stroke="#94a3b8"/>')
    d.label(275,65,'Training: update theta',26);d.label(820,65,'Sampling: fixed theta',26)
    for args in [('clean',40,110,100,55,'x_0'),('random_t',200,110,100,55,'Random t','#eaf7e8'),('noise',360,110,110,55,'Noise eps'),('forward',140,230,250,60,'Forward formula'),('train_net',140,365,250,65,'Denoiser (x_t, t)','#f6ebff'),('prediction',140,495,250,55,'Predicted eps'),('loss',140,630,250,55,'MSE (prediction, eps)','#fff3d6'),('sample_xt',690,110,240,55,'Current x_t'),('sample_t',590,240,100,55,'t','#eaf7e8'),('sample_net',750,240,270,55,'Denoiser (x_t, t)','#f6ebff'),('sample_eps',750,365,270,55,'Predicted eps'),('step',750,495,270,65,'Reverse update','#fff3d6'),('next',750,630,270,55,'x_(t-1)'),('sample_z',590,570,100,55,'z')]:d.box(*args)
    for args in [('clean','forward','90,165 90,260 140,260'),('random_t','forward','250,165 250,230'),('noise','forward','415,165 415,260 390,260'),('forward','train_net','265,290 265,365','x_t',(290,332)),('random_t','train_net','300,137 325,137 325,185 490,185 490,397 390,397','t',(460,345)),('train_net','prediction','265,430 265,495'),('prediction','loss','265,550 265,630'),('noise','loss','470,137 510,137 510,657 390,657'),('loss','train_net','140,657 70,657 70,397 140,397','Gradients',(80,580)),('sample_xt','sample_net','850,165 850,240'),('sample_t','sample_net','690,267 750,267'),('sample_net','sample_eps','885,295 885,365'),('sample_eps','step','885,420 885,495'),('sample_xt','step','930,137 1050,137 1050,527 1020,527','x_t',(1048,460)),('sample_t','step','640,295 640,527 750,527','t',(695,515)),('sample_z','step','640,570 640,550 750,550'),('step','next','885,560 885,630'),('next','sample_xt','1020,657 1065,657 1065,90 810,90 810,110')]:d.arrow(*args)
    d.label(275,745,'Same sampled t enters both paths',20)
    d.label(275,780,'Dashed: time inputs',20)
    d.label(820,710,'Repeat with t <- t-1',20)
    d.label(820,745,'Start x_T ~ N(0,I); t = T,...,1',20)
    d.label(820,780,'No gradients; stop after x_0',20)
    return d

def main():
    diagrams={'time-conditioned-unet-flow.svg':unet(),'training-vs-sampling-flow.svg':training()}
    for name,d in diagrams.items():
        text=d.finish();tree=ET.fromstring(text)
        emitted={(p.attrib['data-from'],p.attrib['data-to']) for p in tree.iter() if p.tag.endswith('polyline')}
        assert emitted==set(d.edges)
        assert all(a in d.nodes and b in d.nodes for a,b in emitted)
        if 'unet' in name:
            assert {('decoder','head'),('middle','attention'),('condition','attention'),('attention','decoder')}<=emitted
            assert ('condition','head') not in emitted
        else:assert {('random_t','forward'),('random_t','train_net'),('sample_t','sample_net'),('sample_t','step')}<=emitted
        p=ASSETS/name
        if '--check' in sys.argv:assert p.read_text()==text, str(p)
        else:p.write_text(text)
    print('PASS two SVG sources and explicit data-flow edges')
if __name__=='__main__':main()
