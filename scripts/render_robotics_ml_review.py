#!/usr/bin/env python3
"""Exact constructed sampling/control and zero-failure statistical figures."""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'content/posts/ai/machine-learning-for-robotics'
spec=importlib.util.spec_from_file_location('deployment_sanity',BUNDLE/'deployment_sanity.py')
example=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=example
spec.loader.exec_module(example)


def text(x,y,label,size=19,anchor='start',weight='normal'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(label))}</text>'


def path(points,color,width=3,dash=''):
    d=' '.join(('M'if i==0 else'L')+f'{x:.4f},{y:.4f}'for i,(x,y)in enumerate(points))
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'


def doc(title,description,body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="620" viewBox="0 0 1120 620" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc><style>text{{font-family:Arial,Helvetica,sans-serif;fill:#172033}}</style><rect width="1120" height="620" fill="white"/>'+''.join(body)+'</svg>\n'


def axes(body,xlabel,ylabel,xticks,yticks,px,py):
    body.append(path([(95,130),(95,430),(1030,430)],'#334155',2))
    for value,label in xticks:
        body.append(text(px(value),457,label,17,'middle'))
    for value,label in yticks:
        body.append(path([(95,py(value)),(1030,py(value))],'#cbd5e1',1,'4 5'))
        body.append(text(82,py(value)+6,label,17,'end'))
    body.extend([text(95,114,ylabel,19),text(560,495,xlabel,19,'middle')])


def sampled_cbf():
    body=[text(30,43,'Instantaneous CBF feasibility and sampled execution',28,weight='bold'),
          text(30,80,'Constructed integrator: xdot = u; x(0) = 0 m; boundary = 0.3 m; alpha = 10 /s.',19)]
    px=lambda t:95+935*t/.2
    py=lambda x:430-300*x/.65
    axes(body,'Time (s)','Position x (m)',[(t,f'{t:g}')for t in [0,.05,.1,.15,.2]],[(v,f'{v:g}')for v in [0,.15,.3,.45,.6]],px,py)
    points=[i/1000 for i in range(201)]
    body.append(path([(px(t),py(example.continuous_boundary(t)))for t in points],'#075985'))
    body.append(path([(px(t),py(3*t))for t in points],'#9a3412',3,'9 5'))
    body.append(path([(px(0),py(.3)),(px(.2),py(.3))],'#334155',2,'3 5'))
    for x,color in [(example.continuous_boundary(.2),'#075985'),(.6,'#9a3412')]:
        body.append(f'<circle cx="{px(.2)}" cy="{py(x)}" r="5" fill="white" stroke="{color}" stroke-width="2"/>')
    body.extend([text(60,540,'Solid: continuously recomputed u = 10(0.3 - x); x(0.2) = 0.259399 m.',19),
                 text(60,575,'Dashed: hold the initially feasible u = 3 m/s for 0.2 s; x(0.2) = 0.600000 m.',19),
                 text(60,605,'Dotted horizontal: x = 0.3 m boundary. No model error or delay is included.',18)])
    return doc('Continuous CBF versus one held control','The continuously updated scalar boundary feedback stays below 0.3 meters. Holding the initially feasible three meters per second command for 0.2 seconds reaches 0.6 meters. This is an analytic counterexample, not robot data.',body)


def zero_failure():
    body=[text(30,43,'Zero failures do not imply zero failure probability',28,weight='bold'),
          text(30,80,'One-sided 95% exact binomial upper bound; fixed policy and IID trials.',19)]
    px=lambda n:95+935*(math.log10(n)-1)/3
    py=lambda percent:430-300*percent/30
    axes(body,'Number of trials n (logarithmic scale)','Failure probability upper bound (%)',[(n,f'{n:g}')for n in [10,100,1000,10000]],[(p,f'{p:g}')for p in [0,5,10,15,20,25,30]],px,py)
    points=[]
    for i in range(301):
        n=10**(1+3*i/300)
        bound=-math.expm1(math.log(.05)/n)
        points.append((px(n),py(100*bound)))
    body.append(path(points,'#5b21b6'))
    for n in [20,100,1000]:
        p=100*example.zero_failure_upper(n)
        body.append(f'<circle cx="{px(n)}" cy="{py(p)}" r="5" fill="white" stroke="#5b21b6" stroke-width="2"/>')
    body.extend([text(60,540,'20 zero-failure trials: 13.910834%; 100 trials: 2.951305%.',20),
                 text(60,575,'1000 zero-failure trials: 0.299125%. Formula: 1 - 0.05^(1/n).',20),
                 text(60,605,'The bound concerns the sampled task distribution, not all possible operating conditions.',18)])
    return doc('Zero-failure binomial upper bound','The exact one-sided 95 percent failure probability upper bound decreases as the number of independent fixed-policy trials grows. At one thousand zero-failure trials it remains about 0.299 percent. The x axis is logarithmic.',body)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    example.check()
    import xml.etree.ElementTree as ET
    for name,data in {'sampled-cbf.svg':sampled_cbf(),'zero-failure-bound.svg':zero_failure()}.items():
        ET.fromstring(data)
        file=BUNDLE/'assets'/name
        if args.write:file.write_text(data)
        if args.check:assert file.read_text()==data,f'Stale figure: {name}'
    print('PASS: deterministic control and confidence-bound SVGs')


if __name__=='__main__':
    main()
