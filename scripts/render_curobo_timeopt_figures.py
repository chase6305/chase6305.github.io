#!/usr/bin/env python3
"""Construct exact scalar timing/rest-torque figures; no paper benchmark data."""
from __future__ import annotations
import argparse
import importlib.util
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'content/posts/robotics/curobo-v2-fast'
spec=importlib.util.spec_from_file_location('timeopt_sanity',BUNDLE/'timeopt_sanity.py')
example=importlib.util.module_from_spec(spec);sys.modules[spec.name]=example;spec.loader.exec_module(example)

def text(x,y,label,size=19,anchor='start',weight='normal'):
 return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(label))}</text>'
def doc(title,desc,body,width=1120,height=570):
 return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><style>text{{font-family:Arial,Helvetica,sans-serif;fill:#172033}}</style><rect width="{width}" height="{height}" fill="white"/>'+''.join(body)+'</svg>\n'
def axes(body,x,y,width,height,xmax,ymax,xlabel,ylabel,yticks,xticks):
 px=lambda v:x+v/xmax*width
 py=lambda v:y+height-v/ymax*height
 body.append(f'<path d="M{x},{y} V{y+height} H{x+width}" fill="none" stroke="#334155" stroke-width="2"/>')
 for t in yticks:
  body.append(f'<path d="M{x},{py(t)} H{x+width}" stroke="#cbd5e1" stroke-dasharray="3 5"/>');body.append(text(x-12,py(t)+6,f'{t:g}',17,'end'))
 for t in xticks:body.append(text(px(t),y+height+28,f'{t:g}',17,'middle'))
 body.extend([text(x,y-18,ylabel,19),text(x+width/2,y+height+61,xlabel,18,'middle')]);return px,py

def curve(body,px,py,points,color,dash=''):
 data=' '.join(('M'if i==0 else'L')+f'{px(x):.4f},{py(y):.4f}'for i,(x,y)in enumerate(points))
 body.append(f'<path d="{data}" stroke="{color}" stroke-width="3" fill="none" stroke-dasharray="{dash}"/>')

def timing():
 b=[text(30,42,'One fixed path, two positive-speed time laws',28,weight='bold'),text(30,78,'Constructed interior segment: no actuator bounds or start/stop constraints imposed.',19)]
 x,y=axes(b,85,145,420,255,1,.6,'Path progress s','Squared path speed x (1/s^2)',[0,.25,.5],[0,.5,1])
 samples=[i/200 for i in range(201)]
 curve(b,x,y,[(s,.25)for s in samples],'#9a3412','9 5');curve(b,x,y,[(s,example.squared_speed(s)[0])for s in samples],'#6d28d9')
 x,y=axes(b,650,145,420,255,1,2.2,'Path progress s','Accumulated time t (s)',[0,1,2],[0,.5,1])
 curve(b,x,y,[(s,2*s)for s in samples],'#9a3412','9 5');curve(b,x,y,[(s,math.asinh(2*(s-.5))+math.asinh(1))for s in samples],'#6d28d9')
 b.extend([text(60,508,'Dashed: x = 0.25; total time 2.00000 s',20),text(60,540,'Solid: x = 0.25 + (s - 0.5)^2; total time 1.76275 s',20)])
 return doc('Fixed-path timing arithmetic','Both time laws traverse the same cubic joint path. The variable positive speed is faster near the two ends of this interior segment. The 11.86 percent constructed duration change does not establish constrained feasibility.',b)

def payload():
 b=[text(30,42,'Slower motion cannot repair an infeasible required rest pose',26,weight='bold'),text(30,78,'Single pendulum at q = pi/2; qdot = qddot = 0; payload lever arm 0.5 m.',19)]
 x,y=axes(b,100,145,890,270,2,12,'Payload mass (kg)','Required gravity torque (N m)',[0,5,10],[0,.5,1,1.5,2])
 curve(b,x,y,[(i/100,example.inverse_dynamics(math.pi/2,0,0,i/100))for i in range(201)],'#6d28d9')
 curve(b,x,y,[(0,5),(2,5)],'#9a3412','9 5')
 limit=5/(9.81*.5)-.3
 b.append(f'<circle cx="{x(limit)}" cy="{y(5)}" r="6" fill="white" stroke="#334155" stroke-width="2"/>')
 b.extend([text(120,508,'Solid: gravity torque = (0.3 + payload) x 9.81 x 0.5',20),text(120,540,'Dashed: 5 N m actuator limit; 2 kg payload requires 11.2815 N m at rest.',19)])
 return doc('Payload and static gravity torque','At a prescribed horizontal rest pose, a two kilogram payload needs 11.2815 newton meters, above the constructed five newton meter limit. Retiming cannot change the required static gravity torque.',b)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--check',action='store_true');args=parser.parse_args();example.check()
 images={'fixed-path-timing.svg':timing(),'payload-static-limit.svg':payload()}
 if args.output:
  args.output.mkdir(parents=True,exist_ok=True)
  for name,data in images.items():(args.output/name).write_text(data)
 if args.check:
  import xml.etree.ElementTree as ET
  for name,data in images.items():
   ET.fromstring(data)
   assert (BUNDLE/'assets'/name).read_text()==data, f'Stale figure: {name}'
 print('Independent chain rule, inverse dynamics, duration and SVG checks passed.')
if __name__=='__main__':main()
