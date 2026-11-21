#!/usr/bin/env python3
"""Check PEAR figure data with stdlib; --output additionally needs Matplotlib."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pear_article_math',ROOT/'content/posts/ai/pear/pear_sanity.py')
article=importlib.util.module_from_spec(spec)
spec.loader.exec_module(article)


def check():
    assert sum(k for _,k,_ in article.REAL_COUNTS)==25
    assert article.REAL_COUNTS==(('base',0,30),('libero_only',7,30),('final',18,30))
    for g in (4,8,16):
        for percent in range(101):
            p=percent/100
            direct=sum(math.comb(g,k)*p**k*(1-p)**(g-k) for k in range(1,g))
            actual=article.sampling(percent,g)['mixed']
            assert math.isclose(actual,direct,abs_tol=3e-15,rel_tol=3e-14)
    for percent in (2,.4):
        result=article.sampling(percent)
        assert math.isclose(result['candidate_rollouts']*result['mixed'],1024,rel_tol=1e-14)
    print(json.dumps({'mixed_curve_cases':303,'reported_trial_counts':'0/30, 7/30, 18/30','constructed_candidate_counts':'checked'}))


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'svg.fonttype':'none','svg.hashsalt':'pear-20261121','axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':'#526070','text.color':'#172330',
        'axes.labelcolor':'#172330','xtick.color':'#263544','ytick.color':'#263544'})
    output.mkdir(parents=True,exist_ok=True)
    def save_figure(figure, name):
        target=output/name
        figure.savefig(target,metadata={'Date':None})
        # Matplotlib leaves spaces at the end of multiline path coordinates.
        # Newlines remain separators, so stripping them changes no SVG geometry.
        target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),gridspec_kw={'width_ratios':[1.2,1]})
    x=[i/10 for i in range(1001)]
    for group,color,style in [(4,'#27648e','-'),(8,'#68438e','--'),(16,'#9b5816','-.')]:
        axes[0].plot(x,[100*article.sampling(p,group)['mixed'] for p in x],
                     label=f'G = {group}',color=color,linestyle=style,linewidth=2)
    point=article.sampling(2)['mixed']*100
    axes[0].scatter([2],[point],s=35,color='#68438e',zorder=4)
    axes[0].annotate('p = 2%, G = 8\nmixed = 14.92%',xy=(2,point),xytext=(21,22),
                     arrowprops={'arrowstyle':'->','color':'#263544'},fontsize=10)
    axes[0].set(xlim=(0,100),ylim=(0,105),xlabel='Per-episode success probability p (%)',ylabel='Mixed-group probability (%)')
    axes[0].set_title('A group needs both outcomes',fontweight='bold',pad=14)
    axes[0].legend(loc='lower center',frameon=False,ncol=3)
    axes[0].grid(axis='y',alpha=.22)
    values=[article.sampling(p)['candidate_rollouts'] for p in (2,.4)]
    bars=axes[1].bar([0,1],values,width=.55,color=['#dcd4ec','#eed8be'],edgecolor='#344454',linewidth=1.2)
    bars[1].set_hatch('//')
    for bar,value in zip(bars,values):
        axes[1].text(bar.get_x()+bar.get_width()/2,value+800,f'{value:,.0f}',ha='center',fontweight='bold')
    axes[1].axhline(1024,color='#344454',linestyle=':',linewidth=1.5)
    axes[1].text(.25,2000,'1,024 retained',fontsize=10,ha='center')
    axes[1].set(xticks=[0,1],xticklabels=['p = 2%','p = 0.4%'],ylim=(0,max(values)*1.16),
                ylabel='Expected candidate rollouts')
    axes[1].set_title('128 mixed groups, G = 8',fontweight='bold',pad=14)
    axes[1].grid(axis='y',alpha=.22)
    fig.suptitle('Fixed-probability sampling and group filtering',fontsize=17,fontweight='bold',y=.98)
    fig.text(.5,.025,'Constructed IID trials; no policy updates or measured training throughput.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.075,right=.98,bottom=.17,top=.81,wspace=.36)
    save_figure(fig,'group-signal.svg')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(11.8,4.9))
    labels=['Base','LIBERO-only','Final']
    colors=['#dce3eb','#dcd4ec','#cce4d3']
    for y,(label,(_,successes,trials),color) in enumerate(zip(labels,article.REAL_COUNTS,colors)):
        percent=successes/trials*100
        lo,hi=(100*v for v in article.wilson(successes,trials))
        ax.barh(y,percent,height=.46,color=color,edgecolor='#344454')
        ax.errorbar(percent,y,xerr=[[percent-lo],[hi-percent]],fmt='o',color='#23323f',capsize=6,linewidth=1.7)
        ax.text(hi+2,y,f'{successes} / {trials}',va='center',fontsize=12,fontweight='bold')
    ax.set(yticks=range(3),yticklabels=labels,xlim=(-1,102),xticks=range(0,101,20),xlabel='Observed success (%)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Real-robot trials: counts and sample size',fontsize=17,fontweight='bold',y=.97)
    fig.text(.52,.865,'Two task families; 30 trials per model',ha='center',fontsize=11)
    fig.text(.52,.045,'Intervals: illustrative 95% Wilson under an IID assumption, not certified coverage for fixed scenes.',ha='center',fontsize=9.5)
    fig.subplots_adjust(left=.16,right=.96,bottom=.2,top=.79)
    save_figure(fig,'real-trials.svg')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.check and args.output is None:parser.error('Use --check or --output')
    check()
    if args.output is not None:render(args.output)
