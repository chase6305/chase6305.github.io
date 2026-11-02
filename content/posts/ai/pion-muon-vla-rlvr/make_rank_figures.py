"""Plot the constructed flat-spectrum reference; requires Matplotlib."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/flat-spectrum-rms.png')
    args = parser.parse_args()
    data = json.loads((ROOT / 'assets/flat-spectrum-reference.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#8793a1', 'axes.labelcolor': '#253448',
                         'text.color': '#253448', 'xtick.color': '#4a5b6f',
                         'ytick.color': '#4a5b6f'})
    fig, ax = plt.subplots(figsize=(11.2, 7.2), layout='constrained')
    ranks = [row['rank'] for row in data['rows']]
    styles = [(0, '#7b5f9c', '-'), (1, '#27669a', '--'), (2, '#327b65', '-.')]
    for kp, color, style in styles:
        values = [row['pion'][str(kp)]['log10_coefficient_decimal'] for row in data['rows']]
        ax.plot(ranks, values, color=color, ls=style, marker='o', markersize=4,
                lw=2, label=f'Pion: {kp} promotion + {5-kp} suppression')
    ax.plot(ranks, [math.log10(abs(row['muon_float64_coefficient'])) for row in data['rows']],
            color='#a95d22', ls=(0, (5, 2, 1, 2)), lw=1.7, label='Muon: 5 polynomial steps (Float64)')
    ax.axhline(0, color='#636b74', ls=':', lw=1.5, label='Ideal polar factor: ratio 1')
    references = [(-133*math.log10(2), 'BF16: smallest positive format value', 2000, -72),
                  (math.log10(math.ulp(0.)), 'Float64 smallest positive value', 3, None)]
    for value, label, x, text_y in references:
        ax.axhline(value, color='#8793a1', lw=1, ls=(0, (3, 4)), zorder=0)
        label_style = {'fontsize': 11, 'color': '#4a5b6f',
                       'bbox': {'facecolor': 'white', 'alpha': .92, 'edgecolor': 'none', 'pad': 2}}
        if text_y is None:
            ax.text(x, value+6, label, **label_style)
        else:
            ax.annotate(label, xy=(x, value), xytext=(x, text_y),
                        arrowprops={'arrowstyle': '-', 'color': '#8793a1', 'lw': .8},
                        **label_style)
    ax.set_xscale('log', base=2)
    ticks = [1, 4, 16, 64, 256, 1024, 4096, 65536]
    ax.set_xticks(ticks, [f'{rank:,}' for rank in ticks])
    ax.set_yticks([0, -40, -100, -200, -300, -400, -500])
    ax.set(xlim=(1, 65536), ylim=(-550, 14),
           xlabel='Flat spectrum dimension r (full-rank identity)',
           ylabel=r'$\log_{10}[\mathrm{RMS}(\Delta W_{\mathrm{grad}})/(\eta\gamma)]$',
           title='Flat spectra: update amplitude depends on dimension')
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=11.5, framealpha=.96)
    fig.get_layout_engine().set(rect=(0, .065, 1, .935))
    fig.text(.015, .015,
             'Constructed M = I_r; epsilon = 0; weight decay excluded.\n'
             'Pion curves use 100-digit Decimal logarithms; format limits are references, not kernel guarantees.',
             fontsize=10.5, color='#4a5b6f')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor='white',
                metadata={'Software': 'Matplotlib flat-spectrum arithmetic'})
    plt.close(fig)


if __name__ == '__main__':
    main()
