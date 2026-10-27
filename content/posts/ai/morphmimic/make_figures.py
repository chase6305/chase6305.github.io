#!/usr/bin/env python3
"""Plot differences from the report's rounded Table I; no inferred error bars."""
import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from morphmimic_contracts import TABLE_I

ROOT = Path(__file__).resolve().parent


def render():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(10.2, 5.1), facecolor='white')
    ax.set_facecolor('white')
    ax.axvspan(-.02, .02, color='#eef5eb', zorder=0)
    for bound in [-.02, .02]:
        ax.axvline(bound, color='#54784b', linestyle='--', linewidth=1.2)
    ax.axvline(0, color='#334155', linewidth=1)
    labels=[]
    for index,(robot,interface,specialist,unified,_,_) in enumerate(TABLE_I):
        delta=round(unified-specialist,4)
        ax.barh(index,delta,height=.54,color='#91b9e5' if delta>=0 else '#edb58a',
                edgecolor='#31516d' if delta>=0 else '#915b31',linewidth=.8,zorder=3)
        ax.text(delta+.0006,index,f'{delta:+.4f}',
                va='center',ha='left',fontsize=10,color='#1e293b')
        labels.append(robot+' / '+interface)
    ax.set_yticks(range(6),labels)
    ax.invert_yaxis()
    ax.set_xlim(-.027,.027)
    ax.xaxis.set_major_locator(MultipleLocator(.01))
    ax.grid(axis='x',color='#e2e8f0',linewidth=.7,zorder=1)
    ax.set_xlabel('Mean normalized return: unified minus specialist')
    ax.set_title('MorphMimic: six robot-interface comparisons',loc='left',pad=16,fontweight='bold')
    fig.text(.01,.015,'Source: report Table I (rounded values). Dashed lines: preset ±0.02 tolerance; not confidence intervals.',
             fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.05,1,1))
    output=io.BytesIO()
    fig.savefig(output,format='png',dpi=180,metadata={'Software':'MorphMimic article comparison; Matplotlib'})
    plt.close(fig)
    return output.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    target=ROOT/'assets/return-differences.png'
    data=render()
    if args.check:
        if not target.is_file() or target.read_bytes()!=data:
            raise SystemExit('Figure differs from the current generator/environment')
        print('Return-difference figure matches the generator')
    else:
        target.write_bytes(data)
        print(target)


if __name__=='__main__':
    main()
