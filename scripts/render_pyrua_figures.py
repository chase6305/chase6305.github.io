#!/usr/bin/env python3
"""Render paper result counts with Matplotlib; --check needs only stdlib."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT/'content/posts/ai/pyrua-lean'
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('pyrua_arithmetic', POST/'interface_sanity.py')
numbers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numbers)


def check():
    assert sum(row[1] for row in numbers.MAIN) == 700
    assert sum(row[2] for row in numbers.MAIN) == 406+36
    assert sum(row[3] for row in numbers.MAIN) == 406+96
    assert 406+36+96+162 == 700
    assert [row[1] for row in numbers.MAIN] == [200,250,90,160]
    print('PASS: Figure counts match the paper groups and paired-outcome totals')


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'pyrua-v2-counts',
                         'axes.edgecolor':'#6b7280','text.color':'#17263b','axes.labelcolor':'#17263b',
                         'xtick.color':'#344359','ytick.color':'#344359'})
    fig = plt.figure(figsize=(10.4,8.2), facecolor='white')
    grid = fig.add_gridspec(2,1,height_ratios=(1.2,1),hspace=.62,left=.27,right=.97,top=.89,bottom=.08)
    ax = fig.add_subplot(grid[0])
    tool_color, code_color = '#c4ddf4', '#d9cdf1'
    for i, (_, n, tool, code, _, _) in enumerate(numbers.MAIN):
        for offset, count, color, hatch in ((-.18,tool,tool_color,'///'),(.18,code,code_color,None)):
            rate = 100*count/n
            ax.barh(i+offset,rate,height=.31,color=color,edgecolor='#344359',hatch=hatch,linewidth=.8)
            ax.text(rate+1.3,i+offset,f'{rate:.1f}%',ha='left',va='center',fontsize=11)
    ax.set_yticks(range(4),[f'{name}\n(n={n})' for name,n,*_ in numbers.MAIN])
    ax.invert_yaxis();ax.set_xlim(0,108);ax.set_xticks(range(0,101,20))
    ax.set_xlabel('Task success rate (%) — all instances')
    ax.set_axisbelow(True);ax.grid(axis='x',color='#e1e6ed',linewidth=.7)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=12)
    ax.legend(handles=[Patch(facecolor=tool_color,edgecolor='#344359',hatch='///',label='Tool calling'),
                       Patch(facecolor=code_color,edgecolor='#344359',label='Code interface')],
              loc='lower left',bbox_to_anchor=(-.01,1.06),ncol=2,frameon=False)
    ax.set_title('A. Success on the same 700 paired task instances',loc='left',fontsize=14,pad=48)
    matrix=fig.add_subplot(grid[1]);matrix.axis('off');matrix.set_xlim(-.25,2.25);matrix.set_ylim(-.15,2.4)
    cells=[(0,1,406,'Both solved','#e5edf8'),(1,1,36,'Tool only','#f6e8d8'),
           (0,0,96,'Code only','#eee7f8'),(1,0,162,'Neither solved','#f0f2f5')]
    for x,y,count,label,color in cells:
        matrix.add_patch(Rectangle((x+.025,y+.025),.95,.95,facecolor=color,edgecolor='#617086',linewidth=.9))
        matrix.text(x+.5,y+.62,str(count),ha='center',va='center',fontsize=26,weight='bold')
        matrix.text(x+.5,y+.27,label,ha='center',va='center',fontsize=12)
    for x,label in ((.5,'Code success'),(1.5,'Code failure')):
        matrix.text(x,2.13,label,ha='center',va='center',fontsize=12)
    for y,label in ((1.5,'Tool success'),(.5,'Tool failure')):
        matrix.text(-.07,y,label,ha='right',va='center',fontsize=12)
    matrix.set_title('B. Paired outcomes: 60 net additional successes',loc='left',fontsize=14,pad=12)
    fig.text(.27,.025,'Paper v2, Table 2 and Section 3.5. One run per agent per instance.',fontsize=10,color='#344359')
    output.mkdir(parents=True,exist_ok=True)
    target=output/'main-outcomes.svg'
    fig.savefig(target,metadata={'Date':None,'Creator':'Matplotlib; render_pyrua_figures.py'},facecolor='white')
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    check()
    if args.output:
        render(args.output)
