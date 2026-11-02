"""Reproduce the article's exact scalar-map figure with NumPy/Matplotlib."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pion_spectral import PROMOTION, SUPPRESSION, half_response, muon, pion, polynomial, report

ROOT=Path(__file__).resolve().parent


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#8793a1','axes.labelcolor':'#253448',
                         'text.color':'#253448','xtick.color':'#4a5b6f','ytick.color':'#4a5b6f',
                         'svg.fonttype':'none'})
    x=np.linspace(0,1,1501)
    fig=plt.figure(figsize=(14,12),layout='constrained')
    grid=fig.add_gridspec(4,2,height_ratios=[1,.22,1,.38])
    axes=np.array([[fig.add_subplot(grid[0,col]) for col in range(2)],
                   [fig.add_subplot(grid[2,col]) for col in range(2)]])
    legend_axes=np.array([[fig.add_subplot(grid[1,col]) for col in range(2)],
                          [fig.add_subplot(grid[3,col]) for col in range(2)]])
    for legend_ax in legend_axes.flat:
        legend_ax.set_axis_off()
    legends=[]
    def separate_legend(row,col):
        handles,labels=axes[row,col].get_legend_handles_labels()
        legends.append(legend_axes[row,col].legend(handles,labels,loc='upper left',fontsize=15))
    blue='#27669a';orange='#a95d22';green='#327b65';purple='#7b5f9c'
    for ax in axes.flat:
        ax.grid(alpha=.16);ax.set_axisbelow(True)
    ax=axes[0,0]
    ax.plot(x,x,color='#738194',ls=':',label='Identity')
    ax.plot(x,[muon(float(v)) for v in x],color=orange,ls='--',label='Muon: 5 NS steps')
    ax.plot(x,[pion(float(v),1) for v in x],color=blue,lw=2,label='Pion: 1 promotion + 4 suppression')
    ax.set(xlim=(0,1),ylim=(0,1.25),xlabel='Normalized input singular value',ylabel='Output coefficient',title='A. Five-step spectral responses')
    separate_legend(0,0)
    ax=axes[0,1]
    ax.plot(x,x,color='#738194',ls=':',label='Identity')
    ax.plot(x,[polynomial(v,PROMOTION) for v in x],color=green,label='Promotion: one step')
    ax.plot(x,[polynomial(v,SUPPRESSION) for v in x],color=purple,ls='--',label='Suppression: one step')
    ax.axvline(np.sqrt(2/3),color='#738194',ls='--',lw=1,label='Fixed point: sqrt(2/3) = 0.8165')
    ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Normalized input singular value',ylabel='Output coefficient',title='B. Promotion and suppression')
    separate_legend(0,1)
    ax=axes[1,0]
    for kp,color,style in [(0,purple,'-'),(1,blue,'--'),(2,green,'-.'),(3,orange,':'),(4,'#95536d',(0,(3,1,1,1))),(5,'#636b74',(0,(5,2,1,2)))]:
        ax.plot(x,[pion(float(v),kp) for v in x],color=color,ls=style,label=f'{kp} promotion + {5-kp} suppression')
    ax.axhline(.5,color='#738194',ls=':',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Normalized input singular value',ylabel='Output coefficient',title='C. Fixed budget, different schedules')
    separate_legend(1,0)
    ax=axes[1,1];data=report();floor=1e-12;positions=np.arange(1,5)
    for key,color,marker,label in [('example_normalized_singular_values','#636b74','o','Normalized input'),('example_muon',orange,'s','Muon: 5 steps'),('example_pion_kp1',blue,'D','Pion: 1 + 4')]:
        values=np.asarray(data[key]);ax.plot(positions,np.maximum(values,floor),color=color,marker=marker,label=label)
        masked=values<floor
        ax.scatter(positions[masked],np.full(masked.sum(),floor),color=color,marker='v',s=90,zorder=4)
    ax.set(xticks=positions,yscale='log',ylim=(4e-13,3),xlabel='Original singular direction\nDown triangles: below 1e-12',ylabel='Coefficient (log scale)',title='D. Constructed diagonal spectrum')
    separate_legend(1,1)
    fig.suptitle('Pion reshapes singular-value magnitudes\nnot temporal frequencies',fontsize=20)
    # Check and save the same fixed layout at the actual output resolution.
    fig.set_dpi(160)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    renderer=fig.canvas.get_renderer()
    for legend in legends:
        box=legend.get_window_extent(renderer)
        if any(box.overlaps(ax.get_tightbbox(renderer)) for ax in axes.flat):
            raise RuntimeError('A legend overlaps a data panel; review the figure layout')
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            raise RuntimeError('A legend extends outside the figure')
    path=ROOT/'assets/spectral-polynomials-v2.png';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=160,facecolor='white',metadata={'Software':'Matplotlib scalar-map reproduction'});plt.close(fig)


if __name__=='__main__':main()
